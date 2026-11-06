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
from bank_experiments import RESULTS, THREADS, features, make_model, positive, read_verified_holdout, single_run, target


CHECKS = Path('results/submission_checks')
CHECK_FILES = ('call_budget_sensitivity.csv', 'training_duplicate_sensitivity.csv', 'row_audit.json')
CHECK_INPUTS = ('frozen_protocol.json', 'folds.csv', 'tuned_and_ensemble_folds.csv', 'holdout_predictions.csv')
CHECK_SCHEMA_VERSION = 1


def evaluate_budget(predictions, models):
    y = predictions.y.to_numpy()
    prevalence = y.mean()
    rows = []
    for budget in (0.10, 0.20):
        k = int(len(y) * budget)
        for model in models:
            score = predictions[model].to_numpy()
            boundary = np.partition(score, len(y) - k)[len(y) - k]
            above, tied = score > boundary, score == boundary
            slots = k - above.sum()
            subscribers = float(y[above].sum() + slots * y[tied].mean())
            rows.append({'call_share': budget, 'calls': k, 'model': model,
                         'expected_subscribers': subscribers, 'precision_at_budget': subscribers / k,
                         'recall_at_budget': subscribers / y.sum(), 'lift': subscribers / (k * prevalence),
                         'boundary_ties': int(tied.sum()), 'slots_at_boundary': int(slots)})
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
        expected = {'schema_version': CHECK_SCHEMA_VERSION, 'inputs': inputs,
                    'outputs': {name: sha(output / name) for name in CHECK_FILES}}
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
    group = predictors.assign(y=target(cleaned)).groupby(list(predictors.columns)).y.agg(['size', 'nunique'])
    mixed = group[group['nunique'] > 1]
    summary = {'holdout_exact_duplicates_within': int(holdout.duplicated().sum()),
               'holdout_exact_matches_in_development': len(matched),
               'unique_development_feature_profiles': len(group),
               'mixed_label_feature_profiles': len(mixed), 'rows_in_mixed_label_profiles': int(mixed['size'].sum())}

    fold_rows = pd.read_csv(result / 'folds.csv')
    duplicate_checks = []
    profile = protocol['profile']
    for fold in fold_rows.itertuples(index=False):
        split = fold.validation_first_source_row
        training = development.iloc[:split]
        validation = cleaned.loc[cleaned.index >= split].iloc[:fold.validation_rows]
        assert len(validation) == fold.validation_rows
        xval, yval = features(validation, profile), target(validation)
        for name, train in [('retained', training), ('removed', training.drop_duplicates())]:
            x, y = features(train, profile), target(train)
            model = make_model('logistic_regression', protocol['candidates']['logistic_regression'], x)
            model.fit(x, y)
            duplicate_checks.append({'fold': fold.fold, 'training_duplicates': name, 'training_rows': len(train),
                                     'average_precision': average_precision_score(yval, positive(model, xval))})
    audit = pd.DataFrame(duplicate_checks)
    expected = pd.read_csv(result / 'tuned_and_ensemble_folds.csv').query("model == 'logistic_regression'")
    np.testing.assert_allclose(audit.query("training_duplicates == 'removed'").average_precision,
                               expected.average_precision, atol=1e-12, rtol=0)

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
    write_json(receipt, {'schema_version': CHECK_SCHEMA_VERSION, 'inputs': inputs,
                         'outputs': {name: sha(output / name) for name in CHECK_FILES}})
    print(pd.Series(summary).to_string())
    print(audit.to_string(index=False))
    print(budget.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    with threadpool_limits(limits=THREADS):
        checks(parser.parse_args().root)
