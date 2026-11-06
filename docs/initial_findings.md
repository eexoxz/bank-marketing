# First Experiment Findings

The original selected file has 41,188 observations and 20 input attributes, satisfying the brief's minimums. The source-order split reserves 8,238 observations for final evaluation. Initial analysis uses the earlier records only.

Development data has 32,950 raw records and 10 exact full-row duplicates. After provisional duplicate removal, 32,940 records remain, including 2,100 subscribers: a positive rate of approximately 6.38%. There are no literal null cells, but explicit unknown categories occur in six fields, including 8,016 records with unknown credit-default status. Treating these as fully observed data would misrepresent data quality.

The dummy prior baseline has mean validation average precision of approximately 0.0745 across three chronological folds. The first balanced logistic regression pipeline has approximately 0.0708. It therefore does not improve the primary ranking score in this initial experiment. Default-threshold recall varies substantially by fold. These are preliminary validation results, not final test scores.

Validation positive rates differ: approximately 5.43%, 5.85%, and 11.07%. This is direct evidence of outcome prevalence changing between validation blocks. Distribution shift in individual features and its contribution to the logistic model's weak results still need investigation; prevalence change alone does not prove the cause.

Next, inspect feature distributions by validation period, compare weighted and unweighted logistic regression, test transformations of sentinel-coded and economic features, and add decision tree and KNN candidates. Preserve the same development validation policy for comparable experiments. Any alternative split must be reported as a separate sensitivity analysis, not silently substituted because it produces better scores.

The final holdout has not been evaluated. No ensemble, tuned model, or recommended deployment solution has been produced at this milestone.
