# Bank Marketing

This project compares logistic regression, a decision tree, k-nearest neighbours and a soft-voting ensemble for ranking customers by their likelihood of subscribing to a term deposit before a marketing call.

The ensemble achieved an average precision of 0.540 on the reserved 8,238-row holdout, compared with 0.530 for logistic regression and 0.308 for the prior-only baseline. Logistic regression had higher recall at the fixed score cutoff. The two models were effectively tied at the smaller of the two illustrative call budgets.

## Run the notebooks

Install 64-bit Python 3.12. From the repository root in PowerShell:

```powershell
.\setup.ps1
.\run.ps1 -Mode notebook
```

Setup checks `python` on your PATH and falls back to the Windows Python launcher. To select a specific Python 3.12 interpreter, pass its path with `-PythonCommand`. In VS Code, select `.venv\Scripts\python.exe` as the notebook kernel.

1. Open `notebooks/01_data_audit_and_baseline.ipynb` for data provenance, exploratory analysis and the initial baseline.
2. Open `notebooks/02_model_experiments.ipynb` for preprocessing, tuning, the reserved holdout and the additional checks.

Both notebooks can run from the repository root or the `notebooks` folder.

## Verify the saved experiment

```powershell
.\run.ps1 -Mode verify
```

This checks the raw data, frozen protocol, fitted models and holdout outputs. Notebook 02 uses the saved evidence after verification. Use the pinned environment in `requirements.txt` when loading the fitted models.

For a full rerun, copy `src/`, `notebooks/`, `data/raw/`, both requirements files, and the two PowerShell scripts into a fresh folder. Set up Python there and run Notebook 02, or run:

```powershell
.\run.ps1 -Mode develop
.\run.ps1 -Mode evaluate
.\run.ps1 -Mode verify
.\run.ps1 -Mode checks
```

Development uses earlier records for feature, model and ensemble selection. The project freezes the selected settings before opening the reserved test. Once a working copy has opened its holdout, use a fresh folder to repeat development. Runtime measurements depend on the machine.

Run `\.\run.ps1 -Mode checks` on a verified experiment to reproduce the report's supplementary results. In `results/submission_checks/`, `preprocessing_sensitivity.csv` has average precision by fold for the final pipeline, numeric scaling removed where applicable, and one-hot encoding with one reference category omitted per field. Each alternative uses the same development folds and selected model settings. The file also records the untuned AP and tuning gain for each fold. Blank scaling results for the tree mean that its final pipeline already used the original numeric units.

`holdout_ensemble_uncertainty.csv` summarises the paired AP difference between the ensemble and logistic regression using the saved holdout predictions. There are 2,000 resamples for each block length (100 and 200 rows), drawn within the three ordered test periods with seed 42. The intervals describe sensitivity within this historical holdout.

## Data and results

- `data/raw/` contains the UCI `bank-additional-full.csv` variant, its data dictionary and source checksums. The original data has 41,188 rows and 20 predictors.
- `data/processed/bank_marketing_cleaned.csv` contains 41,178 rows after removing exact duplicates from development. The `source_row` and `partition` columns record provenance, and `y=1` represents a subscription.
- `results/metrics/` and `results/development_diagnostics/` contain the exploratory audit, including the development correlation matrix. `results/model_experiments/` contains the tuning results, frozen protocol, holdout scores and precision–recall plot. `results/submission_checks/` contains the duplicate, equal-call-budget, preprocessing and ensemble uncertainty checks.
- `models/model_experiments/` contains the fitted pipelines recorded in the frozen protocol.

The source dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Data set]. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5K306)
