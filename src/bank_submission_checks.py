"""Reproduce the report's checks using the frozen experiment."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from warnings import catch_warnings, filterwarnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from bank_data import BOUNDARY, sha, source, write_json
from bank_experiments import (
    RESULTS,
    SEED,
    THREADS,
    cv_run,
    features,
    make_model,
    positive,
    read_verified_holdout,
    single_run,
    target,
)

CHECKS = Path('results/submission_checks')
CHECK_FILES = (
    'call_budget_sensitivity.csv',
    'training_duplicate_sensitivity.csv',
    'preprocessing_sensitivity.csv',
    'holdout_ensemble_uncertainty.csv',
    'row_audit.json',
)
CHECK_INPUTS = (
    'frozen_protocol.json',
    'folds.csv',
    'baseline_folds.csv',
    'tuned_and_ensemble_folds.csv',
    'holdout_predictions.csv',
)
CHECK_SCHEMA_VERSION = 2


def evaluate_budget(predictions, models):
    """Average subscriber yield over ties at each fixed call budget."""
    outcomes = predictions.y.to_numpy()
    prevalence = outcomes.mean()
    rows = []
    for budget in (0.10, 0.20):
        call_count = int(len(outcomes) * budget)
        for model in models:
            scores = predictions[model].to_numpy()
            boundary = np.partition(scores, len(outcomes) - call_count)[len(outcomes) - call_count]
            above_boundary, at_boundary = scores > boundary, scores == boundary
            remaining_calls = call_count - above_boundary.sum()
            expected_subscribers = float(
                outcomes[above_boundary].sum() + remaining_calls * outcomes[at_boundary].mean()
            )
            rows.append(
                {
                    'call_share': budget,
                    'calls': call_count,
                    'model': model,
                    'expected_subscribers': expected_subscribers,
                    'precision_at_budget': expected_subscribers / call_count,
                    'recall_at_budget': expected_subscribers / outcomes.sum(),
                    'lift': expected_subscribers / (call_count * prevalence),
                    'boundary_ties': int(at_boundary.sum()),
                    'slots_at_boundary': int(remaining_calls),
                }
            )
    return pd.DataFrame(rows)


def evaluate_preprocessing(predictors, labels, folds, candidates, original, baseline):
    """Vary one preprocessing choice at a time at the selected model settings."""
    rows = []
    for name, settings in candidates.items():
        def reference_model():
            return make_model(name, settings, predictors).set_params(
                preprocess__categorical__encode__drop='first'
            )

        # Reference coding deliberately maps an unseen level to the all-zero vector.
        with catch_warnings():
            filterwarnings('ignore', message='Found unknown categories in columns', category=UserWarning)
            reference, _ = cv_run(reference_model, predictors, labels, folds)

        unscaled_by_fold = {}
        if name != 'decision_tree':
            def unscaled_model():
                return make_model(name, settings, predictors).set_params(
                    preprocess__numeric__scale='passthrough'
                )

            unscaled, _ = cv_run(unscaled_model, predictors, labels, folds)
            unscaled_by_fold = {row['fold']: row['average_precision'] for row in unscaled}

        for row in reference:
            fold = row['fold']
            final_ap = original.loc[(name, fold), 'average_precision']
            baseline_ap = baseline.loc[(name, fold), 'average_precision']
            rows.append({
                'model': name,
                'fold': fold,
                'final_ap': final_ap,
                'without_scaling_ap': unscaled_by_fold.get(fold),
                'reference_encoding_ap': row['average_precision'],
                'untuned_ap': baseline_ap,
                'tuning_gain': final_ap - baseline_ap,
            })
    return pd.DataFrame(rows)


def ensemble_uncertainty(predictions):
    """Resample contiguous rows within each period and score both rankings on the same draws."""
    outcomes = predictions.y.to_numpy()
    logistic = predictions.logistic_regression.to_numpy()
    ensemble = predictions.soft_voting.to_numpy()
    observed_gain = (
        average_precision_score(outcomes, ensemble) - average_precision_score(outcomes, logistic)
    )
    periods = np.array_split(np.arange(len(outcomes)), 3)
    replicates = 2000
    rows = []
    for block_rows in (100, 200):
        rng = np.random.default_rng(SEED)
        offsets = np.arange(block_rows)
        differences = []
        for _ in range(replicates):
            sampled = []
            for period in periods:
                starts = rng.integers(
                    period[0], period[-1] - block_rows + 2,
                    size=(len(period) + block_rows - 1) // block_rows,
                )
                sampled.append((starts[:, None] + offsets).ravel()[:len(period)])
            chosen = np.concatenate(sampled)
            differences.append(
                average_precision_score(outcomes[chosen], ensemble[chosen])
                - average_precision_score(outcomes[chosen], logistic[chosen])
            )
        low, high = np.quantile(differences, [0.025, 0.975])
        rows.append({
            'block_rows': block_rows,
            'periods': len(periods),
            'replicates': replicates,
            'seed': SEED,
            'observed_ap_gain': observed_gain,
            'lower_95': low,
            'upper_95': high,
        })
    return pd.DataFrame(rows)


@single_run
def checks(root, *, reuse=False):
    result = root / RESULTS
    protocol, predictions = read_verified_holdout(root)
    output = root / CHECKS
    receipt = output / 'checks_outputs.json'
    inputs = {name: sha(result / name) for name in CHECK_INPUTS}
    if reuse and receipt.is_file() and all((output / name).is_file() for name in CHECK_FILES):
        try:
            saved = json.loads(receipt.read_text(encoding='utf-8'))
        except (json.JSONDecodeError, UnicodeDecodeError):
            saved = None
        expected = {
            'schema_version': CHECK_SCHEMA_VERSION,
            'inputs': inputs,
            'outputs': {name: sha(output / name) for name in CHECK_FILES},
        }
        if saved == expected:
            print('Saved descriptive checks verified.', flush=True)
            return

    path, _ = source(root)
    raw = pd.read_csv(path, sep=';')
    development = raw.iloc[:BOUNDARY]
    cleaned = development.drop_duplicates()
    holdout = raw.iloc[BOUNDARY:]
    matched = holdout.merge(cleaned, how='inner', on=list(raw.columns))
    profile = protocol['profile']
    predictors = features(cleaned, profile)
    labels = target(cleaned)
    profile_groups = (
        predictors.assign(y=labels)
        .groupby(list(predictors.columns))
        .y.agg(['size', 'nunique'])
    )
    mixed_profiles = profile_groups[profile_groups['nunique'] > 1]
    summary = {
        'holdout_exact_duplicates_within': int(holdout.duplicated().sum()),
        'holdout_exact_matches_in_development': len(matched),
        'unique_development_feature_profiles': len(profile_groups),
        'mixed_label_feature_profiles': len(mixed_profiles),
        'rows_in_mixed_label_profiles': int(mixed_profiles['size'].sum()),
    }

    fold_rows = pd.read_csv(result / 'folds.csv')
    duplicate_checks = []
    folds = []
    for fold in fold_rows.itertuples(index=False):
        split = fold.validation_first_source_row
        training = development.iloc[:split]
        validation = cleaned.loc[cleaned.index >= split].iloc[: fold.validation_rows]
        assert len(validation) == fold.validation_rows
        train_indices = np.arange(fold.train_rows)
        validation_indices = np.arange(fold.train_rows, fold.train_rows + fold.validation_rows)
        assert cleaned.index[train_indices[-1]] == fold.train_last_source_row
        np.testing.assert_array_equal(cleaned.index[validation_indices], validation.index)
        folds.append((train_indices, validation_indices))
        validation_features, validation_target = features(validation, profile), target(validation)
        for name, training_rows in [
            ('retained', training),
            ('removed', training.drop_duplicates()),
        ]:
            training_features, training_target = features(training_rows, profile), target(
                training_rows
            )
            model = make_model(
                'logistic_regression',
                protocol['candidates']['logistic_regression'],
                training_features,
            )
            model.fit(training_features, training_target)
            duplicate_checks.append(
                {
                    'fold': fold.fold,
                    'training_duplicates': name,
                    'training_rows': len(training_rows),
                    'average_precision': average_precision_score(
                        validation_target, positive(model, validation_features)
                    ),
                }
            )
    audit = pd.DataFrame(duplicate_checks)
    tuned = pd.read_csv(result / 'tuned_and_ensemble_folds.csv', float_precision='round_trip')
    baseline = pd.read_csv(result / 'baseline_folds.csv', float_precision='round_trip')
    np.testing.assert_allclose(
        audit.query("training_duplicates == 'removed'").average_precision,
        tuned.query("model == 'logistic_regression'").average_precision,
        atol=1e-12,
        rtol=0,
    )

    np.testing.assert_array_equal(predictions.source_row, holdout.index)
    np.testing.assert_array_equal(predictions.y, target(holdout))
    budget = evaluate_budget(predictions, tuple(protocol['model_manifest']))
    preprocessing = evaluate_preprocessing(
        predictors, labels, folds, protocol['candidates'],
        tuned.set_index(['model', 'fold']), baseline.set_index(['model', 'fold']),
    )
    uncertainty = ensemble_uncertainty(predictions)
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.checks-', dir=output) as directory:
        staged = Path(directory)
        budget.to_csv(staged / 'call_budget_sensitivity.csv', index=False)
        audit.to_csv(staged / 'training_duplicate_sensitivity.csv', index=False)
        preprocessing.to_csv(staged / 'preprocessing_sensitivity.csv', index=False)
        uncertainty.to_csv(staged / 'holdout_ensemble_uncertainty.csv', index=False)
        write_json(staged / 'row_audit.json', summary)
        for path in staged.iterdir():
            path.replace(output / path.name)
    write_json(
        receipt,
        {
            'schema_version': CHECK_SCHEMA_VERSION,
            'inputs': inputs,
            'outputs': {name: sha(output / name) for name in CHECK_FILES},
        },
    )
    print(pd.Series(summary).to_string())
    print(audit.to_string(index=False))
    print(budget.to_string(index=False))
    print('\nMean development AP by preprocessing option:')
    print(preprocessing.groupby('model', sort=False)[[
        'final_ap', 'without_scaling_ap', 'reference_encoding_ap', 'tuning_gain',
    ]].mean().to_string())
    print('\nHistorical holdout ensemble AP gain over logistic regression:')
    print(uncertainty.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    with threadpool_limits(limits=THREADS):
        checks(parser.parse_args().root)
