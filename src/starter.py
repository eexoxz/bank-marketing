"""Development-only EDA and baseline validation for Bank Marketing.

The final chronological holdout is never scored by this starter.
"""
from __future__ import annotations

import json
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bank-marketing-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, precision_score, recall_score, f1_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bank_data import BOUNDARY, TOTAL, source

ROOT = Path(__file__).resolve().parents[1]
SEED = 42


def load_raw(*, development_only: bool = False) -> pd.DataFrame:
    path, _ = source(ROOT)
    expected_rows = BOUNDARY if development_only else TOTAL
    frame = pd.read_csv(path, sep=';', nrows=BOUNDARY if development_only else None)
    if frame.shape != (expected_rows, 21):
        raise ValueError(f'Unexpected dataset shape: {frame.shape}; expected ({expected_rows}, 21).')
    if set(frame['y'].unique()) != {'yes', 'no'}:
        raise ValueError('Unexpected target labels.')
    return frame


def development_data(raw: pd.DataFrame, *, source_rows: int | None = None) -> tuple[pd.DataFrame, dict]:
    # The source is ordered chronologically. Freeze this boundary before EDA.
    source_rows = len(raw) if source_rows is None else source_rows
    boundary = int(source_rows * 0.8)
    development = raw.iloc[:boundary].copy()
    duplicates = int(development.duplicated().sum())
    # Preliminary policy: remove only exact raw-row duplicates, preserving order.
    # Same engineered features do not establish that two clients are identical.
    development = development.drop_duplicates(keep='first').copy()
    development['source_row'] = development.index
    manifest = {
        'raw_rows': source_rows,
        'raw_input_features': int(raw.shape[1] - 1),
        'development_raw_rows': boundary,
        'development_clean_rows': int(len(development)),
        'development_exact_duplicates_removed': duplicates,
        'test_start_source_row': boundary,
        'reserved_test_rows': source_rows - boundary,
        'split_policy': 'First 80% development; last 20% final holdout; source order preserved.',
        'holdout_evaluated': False,
    }
    return development, manifest


def build_features(development: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = development['y'].map({'no': 0, 'yes': 1}).astype(int)
    features = development.drop(columns=['y', 'duration', 'source_row']).copy()
    # Domain-defined transformation; no parameters are learned from the full data.
    features['pdays_recorded'] = (features['pdays'] != 999).astype(int)
    features['pdays'] = features['pdays'].replace(999, np.nan)
    return features, target


def preprocessing(features: pd.DataFrame) -> ColumnTransformer:
    numeric = features.select_dtypes(include='number').columns.tolist()
    categorical = features.select_dtypes(exclude='number').columns.tolist()
    numeric_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='median', keep_empty_features=True)),
        ('scale', StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ('impute', SimpleImputer(strategy='most_frequent')),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=True)),
    ])
    return ColumnTransformer([
        ('numeric', numeric_pipeline, numeric),
        ('categorical', categorical_pipeline, categorical),
    ])


def save_eda(development: pd.DataFrame, manifest: dict) -> dict:
    destination = ROOT / 'results'
    figures = destination / 'figures'
    metrics = destination / 'metrics'
    figures.mkdir(parents=True, exist_ok=True)
    metrics.mkdir(parents=True, exist_ok=True)
    view = development.drop(columns='source_row')
    numeric = view.select_dtypes(include='number')
    numeric.describe().T.to_csv(metrics / 'development_descriptive_statistics.csv')
    numeric.skew().rename('skewness').to_csv(metrics / 'development_skewness.csv')
    numeric.corr().to_csv(metrics / 'development_correlations.csv')
    pd.DataFrame({
        'null_count': view.isna().sum(),
        'unknown_count': view.eq('unknown').sum(),
        'dtype': view.dtypes.astype(str),
    }).to_csv(metrics / 'development_missingness.csv')
    q1, q3 = numeric.quantile(0.25), numeric.quantile(0.75)
    iqr = q3 - q1
    outlier_counts = ((numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr))).sum()
    outlier_counts.rename('iqr_flag_count').to_csv(metrics / 'development_outlier_flags.csv')
    # Flags are exploratory; unusual values are not automatically errors to remove.
    summary = dict(manifest)
    summary['development_positive_rate'] = float(view['y'].eq('yes').mean())
    summary['development_target_counts'] = {str(k): int(v) for k, v in view['y'].value_counts().items()}
    summary['development_unknown_counts'] = {str(k): int(v) for k, v in view.eq('unknown').sum().items() if v}
    summary['development_literal_nulls'] = int(view.isna().sum().sum())
    summary['pdays_unrecorded_interval_code'] = 999
    (metrics / 'data_audit.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(6, 4))
    view['y'].value_counts().reindex(['no', 'yes']).plot.bar(ax=ax, color=['#4c78a8', '#f58518'], rot=0)
    ax.set(title='Development data: term deposit subscription', xlabel='Subscribed', ylabel='Number of records')
    fig.tight_layout()
    fig.savefig(figures / 'development_target_balance.png', dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for axis, column in zip(axes, ['age', 'campaign', 'previous']):
        axis.hist(view[column], bins=35, color='#4c78a8', edgecolor='white')
        axis.set(title=column, ylabel='Number of records')
    fig.suptitle('Development data: feature distributions')
    fig.tight_layout()
    fig.savefig(figures / 'development_distributions.png', dpi=160)
    plt.close(fig)

    correlations = numeric.corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    image = ax.imshow(correlations, vmin=-1, vmax=1, cmap='coolwarm')
    ax.set_xticks(range(len(correlations)), correlations.columns, rotation=90)
    ax.set_yticks(range(len(correlations)), correlations.columns)
    ax.set_title('Development data: numeric correlations')
    fig.colorbar(image, ax=ax, label='Pearson correlation')
    fig.tight_layout()
    fig.savefig(figures / 'development_correlations.png', dpi=160)
    plt.close(fig)
    return summary


def validate_baselines(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    cv = TimeSeriesSplit(n_splits=3)
    estimators = {
        'dummy_prior': DummyClassifier(strategy='prior'),
        'logistic_regression': LogisticRegression(max_iter=2000, class_weight='balanced', random_state=SEED),
    }
    scoring = {
        'average_precision': 'average_precision',
        'roc_auc': 'roc_auc',
        'balanced_accuracy': 'balanced_accuracy',
        'precision': make_scorer(precision_score, zero_division=0),
        'recall': make_scorer(recall_score, zero_division=0),
        'f1': make_scorer(f1_score, zero_division=0),
        'accuracy': 'accuracy',
    }
    rows = []
    folds = []
    for number, (train, valid) in enumerate(cv.split(features), start=1):
        if target.iloc[train].nunique() != 2 or target.iloc[valid].nunique() != 2:
            raise ValueError(f'Fold {number} does not contain both classes.')
        folds.append({
            'fold': number, 'train_rows': len(train), 'validation_rows': len(valid),
            'train_last_source_row': int(features.index[train[-1]]),
            'validation_first_source_row': int(features.index[valid[0]]),
            'validation_positive_rate': float(target.iloc[valid].mean()),
        })
    for name, estimator in estimators.items():
        pipeline = Pipeline([('preprocess', preprocessing(features)), ('model', estimator)])
        scores = cross_validate(pipeline, features, target, cv=cv, scoring=scoring,
                                n_jobs=1, error_score='raise', return_train_score=False)
        for fold in range(cv.n_splits):
            row = {'model': name, 'fold': fold + 1}
            row.update({key: float(value[fold]) for key, value in scores.items()})
            rows.append(row)
    metrics = ROOT / 'results' / 'metrics'
    metrics.mkdir(parents=True, exist_ok=True)
    detailed = pd.DataFrame(rows)
    detailed.to_csv(metrics / 'baseline_fold_results.csv', index=False)
    pd.DataFrame(folds).to_csv(metrics / 'cross_validation_folds.csv', index=False)
    aggregate = detailed.drop(columns='fold').groupby('model').agg(['mean', 'std'])
    aggregate.columns = ['_'.join(column) for column in aggregate.columns]
    aggregate.to_csv(metrics / 'baseline_summary.csv')
    return aggregate


def run() -> dict:
    raw = load_raw(development_only=True)
    development, manifest = development_data(raw, source_rows=TOTAL)
    processed = ROOT / 'data' / 'processed'
    processed.mkdir(parents=True, exist_ok=True)
    features, target = build_features(development)
    # This is development data only; learned imputation/scaling stays in the pipeline.
    cleaned = features.copy()
    cleaned['y'] = target
    cleaned.index.name = 'source_row'
    cleaned.to_csv(processed / 'development_cleaned.csv')
    summary = save_eda(development, manifest)
    results = validate_baselines(features, target)
    runtime = {'python': platform.python_version(), 'pandas': pd.__version__,
               'numpy': np.__version__, 'scikit_learn': sklearn.__version__}
    (ROOT / 'results' / 'metrics' / 'runtime_versions.json').write_text(json.dumps(runtime, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2))
    print('\nDevelopment cross-validation only:')
    print(results[['test_average_precision_mean', 'test_roc_auc_mean', 'test_recall_mean', 'test_f1_mean']].round(4))
    print('\nFinal holdout has not been evaluated. Baselines are not tuned or final recommendations.')
    return summary


if __name__ == '__main__':
    run()
