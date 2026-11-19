"""Develop, freeze, and evaluate the Bank Marketing coursework models."""

from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
from pathlib import Path
from time import perf_counter
from functools import wraps

os.environ.setdefault(
    'MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bank-marketing-matplotlib')
)
import joblib
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import ParameterGrid, TimeSeriesSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from threadpoolctl import threadpool_limits

from bank_data import BOUNDARY, TOTAL, sha, source, write_json

ROOT = Path(__file__).resolve().parents[1]
SEED, THRESHOLD, THREADS = 42, 0.5, 3
FIELDS = (
    'age',
    'job',
    'marital',
    'education',
    'default',
    'housing',
    'loan',
    'previous',
    'poutcome',
)
ECON = ('emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed')
CONTACT = ('contact', 'month', 'day_of_week', 'campaign')
NAMES = ('logistic_regression', 'decision_tree', 'knn')
MODELS = ('dummy_prior', *NAMES, 'soft_voting')
RESULTS = Path('results/model_experiments')
MODEL_DIR = Path('models/model_experiments')
CLEANED = Path('data/processed/bank_marketing_cleaned.csv')
DEVELOPMENT_FILES = (
    'folds.csv',
    'preprocessing_folds.csv',
    'preprocessing_summary.csv',
    'baseline_folds.csv',
    'tuning_folds.csv',
    'tuned_and_ensemble_folds.csv',
    'development_summary.csv',
)
HOLDOUT_FILES = (
    'holdout_metrics.csv',
    'holdout_predictions.csv',
    'holdout_block_sensitivity.csv',
    'holdout_precision_recall.png',
)
OUTPUTS = (*[RESULTS / file for file in (*DEVELOPMENT_FILES, *HOLDOUT_FILES)], CLEANED)
PROFILES = {
    'historical_full': dict(conditional=True, interval=True, fill_unknown=False),
    'without_economic': dict(conditional=True, interval=True, fill_unknown=False),
    'precall_interval': dict(conditional=False, interval=True, fill_unknown=False),
    'precall_no_interval': dict(conditional=False, interval=False, fill_unknown=False),
    'precall_impute_unknown': dict(conditional=False, interval=True, fill_unknown=True),
}
BASE = {
    'logistic_regression': {'C': 1.0, 'class_weight': 'balanced'},
    'decision_tree': {'max_depth': None, 'min_samples_leaf': 1, 'class_weight': None},
    'knn': {'n_neighbors': 5, 'weights': 'uniform'},
}
GRIDS = {
    'logistic_regression': {'C': [0.01, 0.1, 1.0, 10.0], 'class_weight': [None, 'balanced']},
    'decision_tree': {
        'max_depth': [3, 6, 12, None],
        'min_samples_leaf': [20, 100],
        'class_weight': [None, 'balanced'],
    },
    'knn': {'n_neighbors': [5, 15, 45, 101], 'weights': ['uniform', 'distance']},
}


def single_run(operation):
    """Serialize operations on a project. The OS releases the lock on exit."""

    @wraps(operation)
    def run(root, *args, **kwargs):
        with (root / '.bank-experiment.lock').open('a+b') as lock:
            try:
                if os.name == 'nt':
                    import msvcrt

                    lock.seek(0)
                    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError(
                    'Another experiment is using this project folder. Retry when it finishes.'
                ) from exc
            return operation(root, *args, **kwargs)

    return run


def target(frame):
    y = frame.y.map({'no': 0, 'yes': 1})
    if y.isna().any() or y.nunique() != 2:
        raise ValueError('Expected both valid binary labels.')
    return y.astype(int)


def features(frame, profile):
    """Select the profile inputs and apply the fixed contact-history transformation."""
    profile_settings = PROFILES[profile]
    fields = list(FIELDS)
    if profile_settings['conditional']:
        fields += list(CONTACT)
    if profile == 'historical_full':
        fields += list(ECON)
    input_features = frame[fields].copy()
    if profile_settings['interval']:
        input_features['pdays'] = frame.pdays.where(frame.pdays.ne(999), np.nan)
        input_features['pdays_recorded'] = input_features.pdays.notna().astype('int64')
    if profile_settings['fill_unknown']:
        for column in input_features.select_dtypes(exclude='number'):
            input_features[column] = (
                input_features[column].astype(object).replace('unknown', np.nan)
            )
    assert not {'y', 'duration', 'source_row', 'partition'} & set(input_features)
    return input_features


def make_model(name, params, x):
    """Build a pipeline whose learned preprocessing is fitted with the estimator."""
    if name == 'dummy_prior':
        return DummyClassifier(strategy='prior')
    estimator = {
        'logistic_regression': lambda: LogisticRegression(
            max_iter=2000, random_state=SEED, **params
        ),
        'decision_tree': lambda: DecisionTreeClassifier(random_state=SEED, **params),
        'knn': lambda: KNeighborsClassifier(
            algorithm='brute', metric='euclidean', n_jobs=1, **params
        ),
    }[name]()
    numeric_columns = x.select_dtypes(include='number').columns.tolist()
    categorical_columns = x.select_dtypes(exclude='number').columns.tolist()
    numeric_steps = [('impute', SimpleImputer(strategy='median', keep_empty_features=True))]
    if name in ('logistic_regression', 'knn'):
        numeric_steps.append(('scale', StandardScaler()))
    preprocessor = ColumnTransformer(
        [
            ('numeric', Pipeline(numeric_steps), numeric_columns),
            (
                'categorical',
                Pipeline(
                    [
                        (
                            'impute',
                            SimpleImputer(strategy='most_frequent', keep_empty_features=True),
                        ),
                        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False)),
                    ]
                ),
                categorical_columns,
            ),
        ]
    )
    return Pipeline([('preprocess', preprocessor), ('model', estimator)])


def positive(model, x):
    return model.predict_proba(x)[:, list(model.classes_).index(1)]


def metrics(y, p):
    # Use the same threshold rule for all models; scores are not claimed calibrated.
    predicted_labels = (p > THRESHOLD).astype(int)
    true_negatives, false_positives, false_negatives, true_positives = confusion_matrix(
        y, predicted_labels, labels=[0, 1]
    ).ravel()
    return dict(
        average_precision=average_precision_score(y, p),
        roc_auc=roc_auc_score(y, p),
        precision=precision_score(y, predicted_labels, zero_division=0),
        recall=recall_score(y, predicted_labels, zero_division=0),
        f1=f1_score(y, predicted_labels, zero_division=0),
        balanced_accuracy=balanced_accuracy_score(y, predicted_labels),
        brier=brier_score_loss(y, p),
        tn=int(true_negatives),
        fp=int(false_positives),
        fn=int(false_negatives),
        tp=int(true_positives),
    )


def cv_run(factory, x, y, folds, train_score=False):
    """Fit separate pipelines on earlier records and score each following validation block."""
    fold_results = []
    predictions = []
    for number, (train_indices, validation_indices) in enumerate(folds, 1):
        assert (
            x.index[train_indices[-1]] < x.index[validation_indices[0]]
            and x.index[validation_indices[-1]] < BOUNDARY
        )
        model = factory()
        start = perf_counter()
        model.fit(x.iloc[train_indices], y.iloc[train_indices])
        fit_time = perf_counter() - start
        start = perf_counter()
        validation_scores = positive(model, x.iloc[validation_indices])
        score_time = perf_counter() - start
        fold_result = dict(
            fold=number,
            fit_seconds=fit_time,
            predict_seconds=score_time,
            **metrics(y.iloc[validation_indices], validation_scores),
        )
        if train_score:
            fold_result['train_average_precision'] = average_precision_score(
                y.iloc[train_indices], positive(model, x.iloc[train_indices])
            )
        fold_results.append(fold_result)
        predictions.append(validation_scores)
    return fold_results, predictions


def mean_ap(rows):
    return float(np.mean([r['average_precision'] for r in rows]))


def factory_for(name, params, x):
    return lambda: make_model(name, params, x)


@single_run
def develop(root):
    """Choose preprocessing and model settings using only the development records."""
    results_dir = root / RESULTS
    models_dir = root / MODEL_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    if any(
        (results_dir / name).exists()
        for name in ('holdout_started.json', 'holdout_metrics.csv', 'holdout_outputs.json')
    ):
        raise RuntimeError(
            'This evaluation is already opened. Preserve it; use a separate research workspace for further development.'
        )
    if (results_dir / 'frozen_protocol.json').exists():
        frozen_protocol(root, require_holdout=False)
        print(
            'Development is already frozen and verified. Run evaluate to score the holdout.',
            flush=True,
        )
        return
    raw_path, raw_digest = source(root)
    raw_development = pd.read_csv(raw_path, sep=';', nrows=BOUNDARY)
    assert raw_development.shape == (BOUNDARY, 21)
    development = raw_development.drop_duplicates(keep='first').copy()
    target_values = target(development)
    folds = list(TimeSeriesSplit(n_splits=3).split(development))
    assert len(development) == 32940 and int(target_values.sum()) == 2100
    fold_rows = [
        dict(
            fold=i,
            train_rows=len(train_indices),
            validation_rows=len(validation_indices),
            train_last_source_row=int(development.index[train_indices[-1]]),
            validation_first_source_row=int(development.index[validation_indices[0]]),
            validation_positive_rate=float(target_values.iloc[validation_indices].mean()),
        )
        for i, (train_indices, validation_indices) in enumerate(folds, 1)
    ]
    pd.DataFrame(fold_rows).to_csv(results_dir / 'folds.csv', index=False)
    preprocessing_rows = []
    for profile in PROFILES:
        feature_values = features(development, profile)
        fold_results, _ = cv_run(
            factory_for('logistic_regression', BASE['logistic_regression'], feature_values),
            feature_values,
            target_values,
            folds,
        )
        preprocessing_rows.extend(
            dict(profile=profile, eligible=not PROFILES[profile]['conditional'], **r)
            for r in fold_results
        )
    preprocessing_results = pd.DataFrame(preprocessing_rows)
    preprocessing_results.to_csv(results_dir / 'preprocessing_folds.csv', index=False)
    preprocessing_summary = (
        preprocessing_results.groupby(['profile', 'eligible'], sort=False)
        .average_precision.agg(['mean', 'std'])
        .reset_index()
    )
    preprocessing_summary.to_csv(results_dir / 'preprocessing_summary.csv', index=False)
    selected_profile = str(
        preprocessing_summary[preprocessing_summary.eligible]
        .sort_values('mean', ascending=False, kind='stable')
        .iloc[0]
        .profile
    )
    feature_values = features(development, selected_profile)
    print(
        f'Frozen eligible preprocessing profile for model comparisons: {selected_profile}',
        flush=True,
    )
    baseline_rows = []
    tuning_rows = []
    tuned_rows = []
    best_parameters = {}
    development_scores = {}
    tuned_predictions = {}
    for name in ('dummy_prior', *NAMES):
        fold_results, _ = cv_run(
            factory_for(name, BASE.get(name, {}), feature_values),
            feature_values,
            target_values,
            folds,
        )
        baseline_rows.extend(dict(model=name, **r) for r in fold_results)
        if name == 'dummy_prior':
            continue
        start = perf_counter()
        candidates = list(ParameterGrid(GRIDS[name]))
        if BASE[name] not in candidates:
            candidates.insert(0, BASE[name])
        best_average_precision = -1.0
        for i, parameters in enumerate(candidates, 1):
            candidate_folds = (
                fold_results
                if parameters == BASE[name]
                else cv_run(
                    factory_for(name, parameters, feature_values),
                    feature_values,
                    target_values,
                    folds,
                )[0]
            )
            average_precision = mean_ap(candidate_folds)
            tuning_rows.extend(
                dict(
                    model=name,
                    candidate=i,
                    params=json.dumps(parameters, sort_keys=True),
                    mean_average_precision=average_precision,
                    **r,
                )
                for r in candidate_folds
            )
            if average_precision > best_average_precision + 1e-12:
                best_average_precision = average_precision
                best_parameters[name] = parameters
        tuned_folds, tuned_predictions[name] = cv_run(
            factory_for(name, best_parameters[name], feature_values),
            feature_values,
            target_values,
            folds,
            train_score=True,
        )
        development_scores[name] = mean_ap(tuned_folds)
        tuned_rows.extend(dict(model=name, **r) for r in tuned_folds)
        print(
            f'{name}: {len(candidates)} candidates in {perf_counter()-start:.1f}s; AP {mean_ap(fold_results):.4f} -> {development_scores[name]:.4f}; {best_parameters[name]}',
            flush=True,
        )
    pd.DataFrame(baseline_rows).to_csv(results_dir / 'baseline_folds.csv', index=False)
    pd.DataFrame(tuning_rows).to_csv(results_dir / 'tuning_folds.csv', index=False)
    ranking = sorted(NAMES, key=lambda name: -development_scores[name])
    ensemble_members = ranking[:2]
    ensemble_factory = lambda: VotingClassifier(
        [
            (name, make_model(name, best_parameters[name], feature_values))
            for name in ensemble_members
        ],
        voting='soft',
        n_jobs=1,
    )
    fold_results, ensemble_predictions = cv_run(
        ensemble_factory, feature_values, target_values, folds, train_score=True
    )
    for ensemble_scores, first_model_scores, second_model_scores in zip(
        ensemble_predictions, *[tuned_predictions[n] for n in ensemble_members]
    ):
        np.testing.assert_allclose(
            ensemble_scores, (first_model_scores + second_model_scores) / 2, atol=1e-12, rtol=0
        )
    tuned_rows.extend(dict(model='soft_voting', **r) for r in fold_results)
    pd.DataFrame(tuned_rows).to_csv(results_dir / 'tuned_and_ensemble_folds.csv', index=False)
    summary = []
    for name in MODELS:
        baseline_results = [r for r in baseline_rows if r['model'] == name]
        selected_results = [r for r in tuned_rows if r['model'] == name] or baseline_results
        average_precision = mean_ap(selected_results)
        summary.append(
            dict(
                model=name,
                baseline_ap=mean_ap(baseline_results) if baseline_results else np.nan,
                development_ap=average_precision,
                fold_ap_sd=np.std([r['average_precision'] for r in selected_results], ddof=1),
                gain_vs_baseline=(
                    average_precision - mean_ap(baseline_results) if baseline_results else np.nan
                ),
                mean_train_ap=(
                    np.mean([r['train_average_precision'] for r in selected_results])
                    if 'train_average_precision' in selected_results[0]
                    else np.nan
                ),
                selected_params=json.dumps(best_parameters.get(name, {}), sort_keys=True),
            )
        )
    pd.DataFrame(summary).to_csv(results_dir / 'development_summary.csv', index=False)
    model_manifest = {}
    for name in MODELS:
        model = (
            ensemble_factory()
            if name == 'soft_voting'
            else make_model(name, best_parameters.get(name, {}), feature_values)
        )
        start = perf_counter()
        model.fit(feature_values, target_values)
        fit_time = perf_counter() - start
        model_path = models_dir / f'{name}.joblib'
        joblib.dump(model, model_path, compress=3)
        np.testing.assert_allclose(
            positive(model, feature_values.iloc[:32]),
            positive(joblib.load(model_path), feature_values.iloc[:32]),
        )
        model_manifest[name] = {
            'file': model_path.name,
            'sha256': sha(model_path),
            'bytes': model_path.stat().st_size,
            'development_refit_seconds': fit_time,
        }
    protocol = dict(
        raw_sha256=raw_digest,
        source_rows=TOTAL,
        holdout_boundary=BOUNDARY,
        development_rows=len(development),
        profile=selected_profile,
        features=list(feature_values.columns),
        candidates=best_parameters,
        ranking=ranking,
        ensemble_members=ensemble_members,
        excluded_model=ranking[-1],
        ensemble_weights=[1, 1],
        threshold=THRESHOLD,
        threshold_rule='positive when score strictly greater than threshold',
        selection_metric='unweighted mean fold average precision',
        development_files={name: sha(results_dir / name) for name in DEVELOPMENT_FILES},
        model_manifest=model_manifest,
        seed=SEED,
        threads=THREADS,
        python=platform.python_version(),
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        pandas=pd.__version__,
    )
    write_json(results_dir / 'frozen_protocol.json', protocol)
    print(
        f'Protocol frozen before reading holdout outcomes: {ensemble_members}; omitted {ranking[-1]}',
        flush=True,
    )


def frozen_protocol(root, *, require_holdout):
    result = root / RESULTS
    protocol = result / 'frozen_protocol.json'
    spec = json.loads(protocol.read_text(encoding='utf-8'))
    protocol_hash = sha(protocol)
    marker = result / 'holdout_started.json'
    if marker.exists():
        expected = {
            'protocol_sha256': protocol_hash,
            'first_test_row': BOUNDARY,
            'test_rows': TOTAL - BOUNDARY,
        }
        saved_marker = json.loads(marker.read_text(encoding='utf-8'))
        if saved_marker not in (expected, {**expected, 'output_receipt_required': True}):
            raise ValueError(
                'The frozen protocol or holdout marker changed after the test was opened.'
            )
    elif require_holdout or any(
        (result / name).exists() for name in ('holdout_metrics.csv', 'holdout_outputs.json')
    ):
        raise FileNotFoundError(
            'The holdout marker is missing. Restore it from the saved run before using these results.'
        )
    path, digest = source(root)
    if (digest, BOUNDARY, THRESHOLD, TOTAL) != (
        spec['raw_sha256'],
        spec['holdout_boundary'],
        spec['threshold'],
        spec['source_rows'],
    ):
        raise ValueError(
            'The frozen protocol does not match the source data or evaluation settings.'
        )
    if spec['sklearn'] != sklearn.__version__:
        raise RuntimeError(
            f"The frozen models require scikit-learn {spec['sklearn']}. Use the pinned environment."
        )
    if set(spec['model_manifest']) != set(MODELS):
        raise ValueError('The frozen protocol must describe all five models.')
    for name, detail in spec['model_manifest'].items():
        filename = f'{name}.joblib'
        if detail['file'] != filename or sha(root / MODEL_DIR / filename) != detail['sha256']:
            raise ValueError(f'Model file differs from the frozen protocol: {filename}.')
    hashes = spec.get('development_files')
    if hashes is not None and any(
        hashes.get(name) != sha(result / name) for name in DEVELOPMENT_FILES
    ):
        raise ValueError('Development evidence differs from the frozen protocol.')
    return path, spec, protocol_hash


def record_holdout_outputs(root, protocol_hash):
    files = {path.as_posix(): sha(root / path) for path in OUTPUTS}
    write_json(
        root / RESULTS / 'holdout_outputs.json', {'protocol_sha256': protocol_hash, 'files': files}
    )
    marker = root / RESULTS / 'holdout_started.json'
    state = json.loads(marker.read_text(encoding='utf-8'))
    if not state.get('output_receipt_required'):
        write_json(marker, {**state, 'output_receipt_required': True})


def read_verified_holdout(root):
    path, spec, protocol_hash = frozen_protocol(root, require_holdout=True)
    result = root / RESULTS
    receipt = result / 'holdout_outputs.json'
    if receipt.exists():
        expected = json.loads(receipt.read_text(encoding='utf-8'))
        if expected['protocol_sha256'] != protocol_hash or set(expected['files']) != {
            p.as_posix() for p in OUTPUTS
        }:
            raise ValueError(
                'The saved results receipt does not match the frozen protocol or output inventory.'
            )
        for output in OUTPUTS:
            if sha(root / output) != expected['files'][output.as_posix()]:
                raise ValueError(f'Saved experiment file changed: {output.as_posix()}.')
    elif json.loads((result / 'holdout_started.json').read_text(encoding='utf-8')).get(
        'output_receipt_required'
    ):
        raise FileNotFoundError(
            'The saved results receipt is missing. Run evaluate to recover the holdout outputs.'
        )
    predictions = pd.read_csv(result / 'holdout_predictions.csv', float_precision='round_trip')
    if not receipt.exists():
        # Older completed runs did not save an output receipt. Compare their scores with the frozen estimators.
        test = pd.read_csv(path, sep=';').iloc[BOUNDARY:]
        x = features(test, spec['profile'])
        y = target(test)
        np.testing.assert_array_equal(predictions.source_row, test.index)
        np.testing.assert_array_equal(predictions.y, y)
        table = pd.read_csv(result / 'holdout_metrics.csv').set_index('model')
        for name in MODELS:
            score = positive(joblib.load(root / MODEL_DIR / f'{name}.joblib'), x)
            np.testing.assert_allclose(predictions[name], score, atol=1e-12, rtol=0)
            for metric, value in metrics(y, score).items():
                np.testing.assert_allclose(table.loc[name, metric], value, atol=1e-12, rtol=0)
    return spec, predictions


@single_run
def verify(root):
    read_verified_holdout(root)
    print('Frozen model files and saved holdout results verified.', flush=True)


@single_run
def evaluate(root):
    result = root / RESULTS
    path, spec, protocol_hash = frozen_protocol(root, require_holdout=False)
    receipt = result / 'holdout_outputs.json'
    if receipt.exists():
        saved = json.loads(receipt.read_text(encoding='utf-8'))
        if saved['protocol_sha256'] != protocol_hash or set(saved['files']) != {
            p.as_posix() for p in OUTPUTS
        }:
            raise ValueError(
                'The saved results receipt does not match the frozen protocol or output inventory.'
            )
        for name in DEVELOPMENT_FILES:
            file = RESULTS / name
            if sha(root / file) != saved['files'].get(file.as_posix()):
                raise ValueError(f'Development evidence changed: {file.as_posix()}.')
        if all(
            (root / file).is_file() and sha(root / file) == saved['files'][file.as_posix()]
            for file in (*[RESULTS / name for name in HOLDOUT_FILES], CLEANED)
        ):
            print('Saved holdout evaluation verified and reused.', flush=True)
            return
    started = result / 'holdout_started.json'
    if not started.exists():
        write_json(
            started,
            {
                'protocol_sha256': protocol_hash,
                'first_test_row': BOUNDARY,
                'test_rows': TOTAL - BOUNDARY,
            },
        )
    with tempfile.TemporaryDirectory(prefix='.holdout-', dir=result) as directory:
        staging = Path(directory)
        score_holdout(root, path, spec, staging)
        (root / CLEANED.parent).mkdir(parents=True, exist_ok=True)
        for output in (*[RESULTS / name for name in HOLDOUT_FILES], CLEANED):
            (staging / output.name).replace(root / output)
        record_holdout_outputs(root, protocol_hash)


def score_holdout(root, path, spec, staging):
    """Score the reserved records with the frozen models and stage their outputs."""
    models_dir = root / MODEL_DIR
    raw_data = pd.read_csv(path, sep=';')
    assert raw_data.shape == (TOTAL, 21)
    development = raw_data.iloc[:BOUNDARY].drop_duplicates(keep='first')
    holdout = raw_data.iloc[BOUNDARY:]
    test_features = features(holdout, spec['profile'])
    target_values = target(holdout)
    assert list(test_features.columns) == spec['features'] and len(holdout) == 8238
    prediction_table = pd.DataFrame({'source_row': holdout.index, 'y': target_values.to_numpy()})
    metric_rows = []
    scores_by_model = {}
    for name, model_details in spec['model_manifest'].items():
        model = joblib.load(models_dir / model_details['file'])
        start = perf_counter()
        model_scores = positive(model, test_features)
        prediction_seconds = perf_counter() - start
        scores_by_model[name] = model_scores
        assert np.isfinite(model_scores).all() and ((model_scores >= 0) & (model_scores <= 1)).all()
        metric_rows.append(
            dict(
                model=name,
                **metrics(target_values, model_scores),
                test_rows=len(target_values),
                positives=int(target_values.sum()),
                predict_seconds=prediction_seconds,
                fit_seconds=model_details['development_refit_seconds'],
                model_bytes=model_details['bytes'],
            )
        )
        prediction_table[name] = model_scores
    first_member, second_member = spec['ensemble_members']
    np.testing.assert_allclose(
        scores_by_model['soft_voting'],
        (scores_by_model[first_member] + scores_by_model[second_member]) / 2,
        atol=1e-12,
        rtol=0,
    )
    assert abs(metric_rows[0]['average_precision'] - target_values.mean()) < 1e-12
    assert all(
        sum(r[k] for k in ('tn', 'fp', 'fn', 'tp')) == len(target_values) for r in metric_rows
    )
    pd.DataFrame(metric_rows).to_csv(staging / 'holdout_metrics.csv', index=False)
    prediction_table.to_csv(staging / 'holdout_predictions.csv', index=False)
    block_results = []
    for index, block_indices in enumerate(np.array_split(np.arange(len(target_values)), 3), 1):
        for name, model_scores in scores_by_model.items():
            block_results.append(
                dict(
                    block=index,
                    rows=len(block_indices),
                    positive_rate=float(target_values.iloc[block_indices].mean()),
                    model=name,
                    **metrics(target_values.iloc[block_indices], model_scores[block_indices]),
                )
            )
    pd.DataFrame(block_results).to_csv(staging / 'holdout_block_sensitivity.csv', index=False)
    partitions = []
    for partition, frame in (('development', development), ('holdout', holdout)):
        cleaned_partition = features(frame, spec['profile']).copy()
        cleaned_partition.insert(0, 'source_row', frame.index)
        cleaned_partition.insert(1, 'partition', partition)
        cleaned_partition['y'] = target(frame)
        partitions.append(cleaned_partition)
    cleaned = pd.concat(partitions, ignore_index=True)
    assert cleaned.source_row.is_unique and len(cleaned) == 41178
    cleaned.to_csv(staging / CLEANED.name, index=False)
    figure, axes = plt.subplots(figsize=(7.5, 4.7))
    for row in metric_rows[1:]:
        name = row['model']
        precision_values, recall_values, _ = precision_recall_curve(
            target_values, scores_by_model[name]
        )
        axes.plot(
            recall_values,
            precision_values,
            label=f"{name.replace('_',' ')} (AP {row['average_precision']:.3f})",
        )
    axes.axhline(
        target_values.mean(),
        color='0.35',
        linestyle='--',
        label=f'Positive rate {target_values.mean():.3f}',
    )
    axes.set(
        xlim=(0, 1),
        ylim=(0, 1),
        xlabel='Recall',
        ylabel='Precision',
        title='Reserved holdout precision and recall',
    )
    axes.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(staging / 'holdout_precision_recall.png', dpi=180)
    plt.close(figure)
    print(
        pd.DataFrame(metric_rows)[
            ['model', 'average_precision', 'roc_auc', 'f1', 'balanced_accuracy']
        ].to_string(index=False),
        flush=True,
    )


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('phase', choices=['develop', 'evaluate', 'verify'])
    cli.add_argument('--root', type=Path, default=ROOT)
    args = cli.parse_args()
    with threadpool_limits(limits=THREADS):
        {'develop': develop, 'evaluate': evaluate, 'verify': verify}[args.phase](
            args.root.resolve()
        )
