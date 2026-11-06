# Decision Log

Started 3 October 2026. These are working notes for the student's review, not report text ready for submission.

| Decision | Reason | What still needs checking |
|---|---|---|
| UCI Bank Marketing additional full variant | Public real-world data above the required row and attribute counts | Student understands the banking objective and data's historical context |
| Classification | The provided outcome is a labelled yes/no subscription | Compare supervised and unsupervised approaches with academic sources |
| Exclude call duration | Unknown at the intended prediction time | Audit availability of all other features before contact; campaign count includes current contact |
| Chronological holdout and validation | Later records approximate future deployment more closely than shuffled folds | No exact timestamps or client IDs; add a split sensitivity experiment and document uncertainty |
| Exact development duplicates removed | Prevent exact repeated records dominating experiments | Rows may represent distinct clients with identical measurements; compare retaining duplicates |
| Explicit unknown categories initially | Preserve the information that a value was unrecorded | Compare alternative missingness policies without validation leakage |
| Contact-history indicator plus pdays sentinel handling | 999 means no prior contact, not 999 actual elapsed days | Compare alternative representations |
| Balanced logistic regression baseline | Interpretable baseline that accommodates class imbalance | Compare unweighted fitting; assess probabilities and threshold selection |
| Average precision as initial primary score | Measures ranking quality for the minority positive outcome | Compare with prevalence, business costs and recall at practical contact budgets |

Before interpreting model scores, explain aloud why an always-negative classifier can have high accuracy while finding no subscribers. Then explain why a model using duration can look successful while failing the intended business use.


## Review on 30 October 2026

The original baseline is preserved. Additional development-only checks reveal marked differences between fold input ranges and outcome prevalence. These observations motivate controlled feature-group experiments; they do not prove the cause of poor model performance.

The broad name `previously_contacted` needs review because it currently means only `pdays != 999`, while 2,101 development records have `pdays=999` and `previous>0`. See `feature_availability.md` before interpreting or changing this feature. No automatic row deletion or reinterpretation has been applied.

The partial report uses the user's requested connected, plain explanatory style. The final holdout remains reserved, and no new model was fitted during this session.

## Report format preference

The user requested DOCX and PDF report files. Continue edits in report/report_draft.docx and refresh report/report_draft.pdf for review. The outline and completion guide are combined in report/report_planning.docx. Markdown report copies were removed from the report folder after verified conversion.
