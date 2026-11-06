"""Additional development diagnostics; no model fitting or final holdout evaluation."""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'bank-marketing-matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from bank_data import BOUNDARY, TOTAL, source

VOCABULARY = {
    'job': ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown'],
    'marital': ['divorced', 'married', 'single', 'unknown'],
    'education': ['basic.4y', 'basic.6y', 'basic.9y', 'high.school', 'illiterate', 'professional.course', 'university.degree', 'unknown'],
    'default': ['no', 'yes', 'unknown'], 'housing': ['no', 'yes', 'unknown'],
    'loan': ['no', 'yes', 'unknown'], 'contact': ['cellular', 'telephone'],
    'month': ['jan', 'feb', 'mar', 'apr', 'may', 'jun', 'jul', 'aug', 'sep', 'oct', 'nov', 'dec'],
    'day_of_week': ['mon', 'tue', 'wed', 'thu', 'fri'],
    'poutcome': ['failure', 'nonexistent', 'success'],
}


def run(root: Path, output: Path) -> dict:
    raw_path, file_hash = source(root)
    raw_development = pd.read_csv(raw_path, sep=';', nrows=BOUNDARY)
    if raw_development.shape != (BOUNDARY, 21):
        raise ValueError(f'Unexpected development data shape: {raw_development.shape}.')
    dev = raw_development.drop_duplicates().copy()
    numeric = dev.select_dtypes(include='number').columns.tolist()
    categorical = dev.select_dtypes(exclude='number').drop(columns='y').columns.tolist()
    summary = {
        'prepared_on': datetime.now(timezone(timedelta(hours=8))).date().isoformat(), 'raw_sha256': file_hash,
        'scope': 'First 32950 source rows only; 10 exact duplicates removed before fold diagnostics.',
        'holdout_evaluated': False, 'model_fitted': False,
        'raw_rows': TOTAL, 'raw_input_features': len(raw_development.columns) - 1,
        'original_numeric_input_features': len(numeric),
        'original_categorical_input_features': len(categorical),
        'development_rows': len(dev),
        'development_positive_count': int(dev.y.eq('yes').sum()),
        'development_positive_rate': float(dev.y.eq('yes').mean()),
        'unknowns': {column: {'count': int(dev[column].eq('unknown').sum()),
                              'percent': float(dev[column].eq('unknown').mean() * 100)}
                     for column in categorical if dev[column].eq('unknown').any()},
        'numeric_summary': {
            col: {'median': float(dev[col].median()), 'min': float(dev[col].min()),
                  'max': float(dev[col].max()), 'skewness': float(dev[col].skew())}
            for col in numeric},
        'pdays_unrecorded_interval_count': int(dev.pdays.eq(999).sum()),
        'pdays_recorded_contact_count': int(dev.pdays.ne(999).sum()),
        'pdays_recorded_contact_median': float(dev.loc[dev.pdays.ne(999), 'pdays'].median()),
        'unrecognised_categories': {col: sorted(set(dev[col].unique()) - set(allowed))
                                    for col, allowed in VOCABULARY.items()},
        'outer_whitespace_counts': {col: int(dev[col].ne(dev[col].str.strip()).sum())
                                    for col in categorical},
        'basic_domain_check_counts': {
            'age_nonpositive': int(dev.age.le(0).sum()),
            'campaign_below_one': int(dev.campaign.lt(1).sum()),
            'previous_negative': int(dev.previous.lt(0).sum()),
            'pdays_negative': int(dev.pdays.lt(0).sum()),
            'previous_zero_with_existing_outcome': int((dev.previous.eq(0) & dev.poutcome.ne('nonexistent')).sum()),
            'recorded_pdays_with_previous_zero': int((dev.pdays.ne(999) & dev.previous.eq(0)).sum()),
            'pdays_999_with_previous_positive_for_review': int((dev.pdays.eq(999) & dev.previous.gt(0)).sum()),
        },
    }
    # Correlations among actual-valued model inputs, excluding the pdays sentinel.
    corr_columns = [col for col in numeric if col not in {'duration', 'pdays'}]
    corr = dev[corr_columns].corr()
    summary['strong_numeric_correlations'] = [
        {'feature_1': a, 'feature_2': b, 'pearson_r': float(corr.loc[a, b])}
        for i, a in enumerate(corr_columns) for b in corr_columns[i + 1:]
        if abs(corr.loc[a, b]) >= 0.8]
    folds = []
    features = [col for col in numeric if col not in {'duration', 'pdays'}]
    range_percentages = []
    for number, (train_indices, valid_indices) in enumerate(TimeSeriesSplit(n_splits=3).split(dev), 1):
        train, valid = dev.iloc[train_indices], dev.iloc[valid_indices]
        assert train.index.max() < valid.index.min() < BOUNDARY
        outside = {col: float((valid[col].lt(train[col].min()) | valid[col].gt(train[col].max())).mean() * 100)
                   for col in features}
        folds.append({
            'fold': number, 'train_rows': len(train), 'validation_rows': len(valid),
            'train_last_source_row': int(train.index.max()),
            'validation_first_source_row': int(valid.index.min()),
            'validation_last_source_row': int(valid.index.max()),
            'train_positive_rate': float(train.y.eq('yes').mean()),
            'validation_positive_rate': float(valid.y.eq('yes').mean()),
            'numeric_constant_in_training': [col for col in numeric if col != 'duration' and train[col].nunique() == 1],
            'validation_values_outside_training_range_percent': outside,
            'categorical_values_not_seen_in_training': {col: sorted(set(valid[col].unique()) - set(train[col].unique()))
                                                       for col in categorical},
        })
        range_percentages.append(outside)
    summary['folds'] = folds
    output.mkdir(parents=True, exist_ok=True)
    (output / 'diagnostics.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(folds))
    ax.bar(x - .18, [f['train_positive_rate'] * 100 for f in folds], .36, label='Training')
    ax.bar(x + .18, [f['validation_positive_rate'] * 100 for f in folds], .36, label='Validation')
    ax.set(xticks=x, xticklabels=[f'Fold {f["fold"]}' for f in folds],
           ylabel='Subscribers (%)', title='Development subscription rates by chronological fold')
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / 'fold_prevalence.png', dpi=160)
    plt.close(fig)
    matrix = pd.DataFrame(range_percentages)[features]
    fig, ax = plt.subplots(figsize=(11, 3.5))
    image = ax.imshow(matrix, vmin=0, vmax=100, cmap='Blues', aspect='auto')
    ax.set_xticks(range(len(features)), features, rotation=30, ha='right')
    ax.set_yticks(range(3), ['Fold 1', 'Fold 2', 'Fold 3'])
    for row in range(3):
        for col in range(len(features)):
            value = matrix.iloc[row, col]
            ax.text(col, row, f'{value:.0f}', ha='center', va='center',
                    color='white' if value > 55 else 'black')
    ax.set_title('Validation values outside the corresponding training range (%)')
    fig.colorbar(image, ax=ax, label='Validation records (%)')
    fig.tight_layout()
    fig.savefig(output / 'fold_range_changes.png', dpi=160)
    plt.close(fig)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    run(args.root, args.output_dir or args.root / 'results/development_diagnostics')
