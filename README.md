# Bank Marketing Machine Learning Project

This project investigates whether a bank can prioritise customers likely to subscribe to a term deposit before a marketing call. The first milestone provides the original public dataset, a development data audit, exploratory plots, and chronological cross-validation of a dummy baseline and logistic regression. The completed experiments compare preprocessing profiles, three tuned classifiers and a soft-voting ensemble on a reserved holdout.

## Open and run

Open this folder in VS Code. For notebooks, select `.venv\Scripts\python.exe` as the kernel. Run `notebooks/01_data_audit_and_baseline.ipynb` for the original audit, then open `notebooks/02_model_experiments.ipynb` for the final models and results. Use Restart Kernel and Run All for each notebook.

From PowerShell in this folder:

```powershell
.\run.ps1
# Or open JupyterLab:
.\run.ps1 -Mode notebook
```

If PowerShell script execution is restricted, run the environment executable directly without changing execution policy:

```powershell
.\.venv\Scripts\python.exe src/starter.py
.\.venv\Scripts\python.exe -m jupyterlab notebooks
```

## Environment

An isolated Python 3.12 environment has been prepared locally using the available bundled runtime. The environment's base Python is managed by the desktop app; it is not a standalone system Python installation. To move the project or recreate it on another machine, install standard 64-bit Python 3.12 and run `setup.ps1`. Do not copy the `.venv` directory between machines. `requirements.txt` records the dependency versions used for verification; `requirements.in` lists the direct dependencies.

## Data provenance

Use the **bank-additional-full.csv** variant: 41,188 records, 20 original input features, and target `y`. The smaller bank-additional.csv has only 4,119 rows and does not meet the assignment's minimum size. The landing page describes multiple variants; use the selected file's documentation rather than mixing their schemas.

Source: https://archive.ics.uci.edu/dataset/222/bank+marketing

Citation: Moro, S., Rita, P., & Cortez, P. (2014). Bank Marketing [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5K306

Licence: Creative Commons Attribution 4.0. The original data and publisher's names file are retained in `data/raw/`. `source_manifest.json` records file hashes. To download them again:

```powershell
.\.venv\Scripts\python.exe src/download_data.py
```

## Initial audit and baseline decisions

- Binary classification: `yes` is the positive class. Supervised learning directly addresses the supplied labelled outcome; clustering can explore segments but does not directly answer this prediction question.
- Exclude `duration`: it is unknown until after the call and therefore unavailable for the intended prediction.
- Reserve the last 20% of source rows as an untouched final holdout. The publisher describes the full file as ordered by date. Use three expanding-window validation folds on the first 80%.
- The file has no complete per-record timestamps or customer IDs. Source order is a proxy for time, and repeated customers cannot be reliably grouped. Document these limitations and test split sensitivity later.
- Remove only exact full raw-row duplicates in development, preserving the first occurrence and source order. This is a preliminary assumption to review. Matching reduced feature vectors are not automatically duplicate clients.
- Keep `unknown` as an explicit categorical level for the first baseline. Compare this with alternative missing-value handling later.
- The starter baseline used a flag for whether a `pdays` interval was recorded. Some records coded `999` also report previous contacts. The final pipeline leaves out `pdays` and retains `previous` and `poutcome` as separate recorded history fields.
- Average precision is the initial primary ranking metric because subscribers are a minority. Also record ROC-AUC, balanced accuracy, precision, recall, F1, accuracy, and computation time. Threshold-dependent scores currently use the estimator's default threshold; threshold optimisation is future work.
- The dummy comparator is not one of the three required substantive models. Logistic regression is the first candidate, followed by a decision tree and k-nearest neighbours, subject to validation findings.
- PCA, SMOTE, a web app, and a database are not automatically required. Evaluate optional preprocessing only when the data or experiments justify it.

## Outputs

- `results/metrics/`: development audit, descriptive statistics, missingness, correlations, skewness, outlier flags, validation folds, baseline scores, and runtime versions.
- `results/figures/`: development class balance, numeric distributions, and correlations.
- `data/processed/development_cleaned.csv`: development-only domain transformations and target. Learned preprocessing is intentionally applied inside pipelines.

The cleaned development export and starter results document the first milestone. See the final experiment outputs below for the selected schema and holdout comparison.


## Historical notes from 30 October 2026

`references/source_register.md` records ten academic papers, six recent publications, and the depth of reading completed for each. The active coursework report is the Word file in `Downloads/6006CEM`.

`docs/feature_availability.md` covers prediction-time assumptions and the pdays contact-history discrepancy. `docs/progress_2026_10_30.md` records the next session's work.

To reproduce the added development checks without fitting models or evaluating the final holdout:

```powershell
.\.venv\Scripts\python.exe src/audit_development.py
```

Results are written to `results/development_diagnostics/`. This adds to the earlier baseline evidence and does not replace it.

## Final model experiments

Run `.\run.ps1 -Mode verify` to check the saved experiment files. Notebook 02 verifies recorded results before displaying them and generates results on a fresh copy. From a fresh project with the raw CSV and manifest but no final model outputs, run these stages in order from the project root.

```powershell
.\.venv\Scripts\python.exe src/bank_experiments.py develop
.\.venv\Scripts\python.exe src/bank_experiments.py evaluate
```

Keep the original results intact when reproducing the experiment. Use a fresh working copy for a full rerun. Once the reserved test has been opened in a working copy, the script blocks development in that copy. Evaluation can be repeated with the same frozen protocol and verifies the saved model hashes. Load only the project models you generated or verified.

- Development used 32,940 earlier observations and three expanding folds. It compared five preprocessing profiles, then tuned logistic regression, a decision tree and k-nearest neighbours. The final profile retains nine pre-call inputs, keeps encoded `unknown` values as categories and omits contact duration, the ambiguous `pdays` interval and fields of unverified availability at the targeting decision.
- The ensemble equally averages the two strongest individual models by mean development average precision, logistic regression and the decision tree. Holdout decisions use a score strictly above 0.5 for every model. The threshold was fixed before the test and the scores have not been calibrated.
- On 8,238 holdout rows, average precision was 0.540 for soft voting, 0.530 for logistic regression, 0.452 for the tree, 0.451 for k-nearest neighbours and 0.308 for the prior-only comparator. At the fixed cutoff, soft voting had precision 0.580 and recall 0.416. Logistic regression had precision 0.371 and recall 0.772. The three ordered test blocks showed large changes in subscription rate and performance.
- `results/model_experiments/` records the preprocessing and tuning results, frozen protocol, test scores, predictions, precision–recall plot and block sensitivity check.
- `data/processed/bank_marketing_cleaned.csv` contains 41,178 observations after excluding development duplicates. `source_row` is a zero-based record number and `partition` identifies the split. Both are audit fields. `y` is binary, with `1` for a subscription. Learned transformations live in the fitted model pipelines under `models/model_experiments/`.
- The current technical report and presentation are in `Downloads/6006CEM`.

## Additional checks used in the report

Notebook 02 verifies and reuses the saved descriptive checks for training duplicates and equal call budgets. It recalculates them if their inputs or outputs change. After the reserved test has been evaluated, run the checks separately from the project root with:

```powershell
.\.venv\Scripts\python.exe src/bank_submission_checks.py
```

Outputs go to `results/submission_checks/`. The tests use the frozen configuration and saved holdout scores. The coursework report and submission files are kept separately in the `6006CEM` folder in Downloads.

## Fresh Git checkout

The checkout contains the public UCI source data, the two notebooks, and the pinned setup. On Windows with 64-bit Python 3.12 installed, run `.setup.ps1` and then `.un.ps1 -Mode notebook`. Run both notebooks in order to regenerate the local outputs. The `verify` mode checks a run after Notebook 02 or the `evaluate` command has produced the holdout results. The Word report, presentation, and technical ZIP are stored separately in Downloads.
