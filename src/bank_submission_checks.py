"""Reproduce descriptive checks for the report without changing the frozen model."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from threadpoolctl import threadpool_limits

from bank_data import BOUNDARY, sha, source, write_json
from bank_experiments import (
    RESULTS,
    THREADS,
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
    'row_audit.json',
)
CHECK_INPUTS = (
    'frozen_protocol.json',
    'folds.csv',
    'tuned_and_ensemble_folds.csv',
    'holdout_predictions.csv',
)
CHECK_SCHEMA_VERSION = 1


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
    predictors = features(cleaned, protocol['profile'])
    profile_groups = (
        predictors.assign(y=target(cleaned))
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
    profile = protocol['profile']
    for fold in fold_rows.itertuples(index=False):
        split = fold.validation_first_source_row
        training = development.iloc[:split]
        validation = cleaned.loc[cleaned.index >= split].iloc[: fold.validation_rows]
        assert len(validation) == fold.validation_rows
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
    expected = pd.read_csv(result / 'tuned_and_ensemble_folds.csv').query(
        "model == 'logistic_regression'"
    )
    np.testing.assert_allclose(
        audit.query("training_duplicates == 'removed'").average_precision,
        expected.average_precision,
        atol=1e-12,
        rtol=0,
    )

    np.testing.assert_array_equal(predictions.source_row, holdout.index)
    np.testing.assert_array_equal(predictions.y, target(holdout))
    budget = evaluate_budget(predictions, tuple(protocol['model_manifest']))
    output.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.checks-', dir=output) as directory:
        staged = Path(directory)
        budget.to_csv(staged / 'call_budget_sensitivity.csv', index=False)
        audit.to_csv(staged / 'training_duplicate_sensitivity.csv', index=False)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    with threadpool_limits(limits=THREADS):
        checks(parser.parse_args().root)
