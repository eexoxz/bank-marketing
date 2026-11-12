# Bank Marketing

This project compares logistic regression, a decision tree, k-nearest neighbours and a soft-voting ensemble for ranking customers by their likelihood of subscribing to a term deposit before a marketing call.

The ensemble achieved an average precision of 0.540 on the reserved 8,238-row holdout, compared with 0.530 for logistic regression and 0.308 for the prior-only baseline. Logistic regression had higher recall at the fixed score cutoff. The two models were effectively tied at the smaller of the two illustrative call budgets.

## Run the notebooks

Install 64-bit Python 3.12. From the repository root in PowerShell:

```powershell
.\setup.ps1
.\run.ps1 -Mode notebook
```

If Python 3.12 is available as `python` and you do not have the Windows Python launcher, use `.\setup.ps1 -PythonCommand python`. In VS Code, select `.venv\Scripts\python.exe` as the notebook kernel.

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

## Data and results

- `data/raw/` contains the UCI `bank-additional-full.csv` variant, its data dictionary and source checksums. The original data has 41,188 rows and 20 predictors.
- `data/processed/bank_marketing_cleaned.csv` contains 41,178 rows after removing exact duplicates from development. The `source_row` and `partition` columns record provenance, and `y=1` represents a subscription.
- `results/metrics/` and `results/development_diagnostics/` contain the exploratory audit. `results/model_experiments/` contains the tuning results, frozen protocol, holdout scores and precision–recall plot. `results/submission_checks/` contains the duplicate and equal-call-budget checks.
- `models/model_experiments/` contains the fitted pipelines recorded in the frozen protocol.

The source dataset is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Cite Moro, S., Rita, P., & Cortez, P. (2014). *Bank Marketing* [Data set]. UCI Machine Learning Repository. [Dataset DOI](https://doi.org/10.24432/C5K306)
