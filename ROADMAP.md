# Assignment Work Plan

The brief states a deadline of 22 November 2026 and a report of 5,000 words plus or minus 20%. These are assignment facts, not a guarantee about later changes in Canvas. The filename says 6006CEM while the document's module code says 6001CEM; check the official module label before preparing the cover page.

| Stage | Brief coverage | Evidence to produce | Status |
|---|---|---|---|
| Foundation | Local environment and data requirements | Runnable notebook, dataset documentation, dependencies | Prepared |
| Problem formulation | A and B | Student's business objective, prediction timing, paradigm comparison and sources | Initial notes only |
| Data investigation | C | Descriptive statistics, data quality, imbalance, noise and inconsistencies | Initial audit prepared; interpretation pending |
| Preparation | D | Compare justified missingness, encoding, feature and sampling options | First pipeline prepared; comparative experiments pending |
| Three models | E | Logistic regression, decision tree and KNN; principles and limitations | First baseline prepared |
| Optimisation | F | Tune all three with chronological validation; record search spaces and gains | Pending |
| Ensemble | G | Select best two using development validation; voting or suitable ensemble | Pending |
| Evaluation | H | Freeze decisions, score three tuned models and ensemble on final holdout | Pending |
| Writing and slides | Deliverables | Report, source, notebook, cleaned data, README, slides | Pending |
| Viva | I | Explain every choice and your own contribution | Practice throughout |

## Suggested pacing

- 3-9 October: understand the dataset, define the prediction time, inspect initial audit, and establish baselines.
- 10-16 October: complete EDA interpretation and preprocessing comparisons.
- 17-23 October: develop and explain the three candidate models.
- 24-30 October: tune all three models and assess time and generalisation.
- 31 October-6 November: select two candidates and build the ensemble; check robustness.
- 7-13 November: freeze decisions, evaluate the untouched holdout, draft the report from verified results.
- 14-20 November: refine report and slides, verify notebook execution, check citations, rehearse viva.
- 21 November: package and review submission before the stated 22 November deadline.

## Evidence to collect as we work

Maintain a results table linking model, preprocessing, parameters, validation scheme, metric mean and spread, fit time, and seed. Keep experiment outputs separate from report prose. Assemble at least ten academic references; the brief requests at least six peer-reviewed journal or conference papers published within the last five years. The 2014 dataset paper is useful provenance but does not meet that recent-paper quota.

Review model assumptions, complexity, interpretability, robustness, overfitting, and deployment suitability. Classification metrics alone do not cover all of Part H. Do not claim the ensemble improves performance until experiments support that claim.
