# Feature Availability and Domain Checks

The intended prediction point is before a selected marketing call. This audit distinguishes fields in the historical file from inputs a live system could reliably obtain at that time. Source definitions come from `data/raw/bank-additional-names.txt` for the additional full variant. Availability conditions are project design judgements that need verification before deployment claims.

| Field or group | Role and availability condition | Current handling or next check |
|---|---|---|
| `age` | Customer age; requires a record valid at prediction time | Retained provisionally; historical/current record availability needs confirmation |
| `job` | Customer occupation | Retained; preserve unknown values and assess missingness |
| `marital` | Marital status | Retained; consider whether using this input is justified for the intended decision |
| `education` | Education category | Retained; explicit unknown category |
| `default` | Credit-default status | Retained provisionally; 24.34% unknown in development; assess the treatment and use of this field |
| `housing` | Housing-loan status | Retained; availability and unknown-value policy need review |
| `loan` | Personal-loan status | Retained; availability and unknown-value policy need review |
| `contact` | Channel recorded for the last contact | Conditional: use only if the intended channel is known before the selected call |
| `month` | Month of recorded contact | Conditional: derive from the planned call date; do not assume an eventual contact date was known earlier |
| `day_of_week` | Weekday of recorded contact | Same scheduling condition as month |
| `duration` | Completed call duration | Excluded because it is unavailable before the call |
| `campaign` | Current-campaign contact count including the recorded call | Conditional: the planned call number must be derivable from contact history; do not use a future cumulative count |
| `pdays` | Elapsed-day measure with special code 999 | Preserve source values for audit; investigate its relationship to previous contacts before interpreting the sentinel |
| `previous` | Count of contacts before the current campaign | Requires history already recorded at prediction time; no reliable customer IDs are provided to verify it independently |
| `poutcome` | Previous campaign outcome | Requires an outcome resolved before the current decision; review source consistency |
| `emp.var.rate` | Quarterly employment variation indicator | Check release timing, period alignment and which historical version would have been available |
| `cons.price.idx` | Monthly consumer-price indicator | Check release timing and alignment with prediction date |
| `cons.conf.idx` | Monthly consumer-confidence indicator | Check release timing and alignment with prediction date |
| `euribor3m` | Daily three-month interest rate | Check when the corresponding daily value becomes known relative to the call |
| `nr.employed` | Quarterly employment-level indicator | Check release timing and historical version availability |
| `y` | Subscription outcome | Target only; never a predictor |
| `source_row` | Position in the original file | Audit identifier only; never a predictor |

## Contact history issue discovered on 30 October

Development diagnostics find 2,101 records with `pdays=999` and `previous>0`. The selected names document describes 999 as no previous contact, while `previous` counts contacts before the campaign. These fields therefore require a more precise source interpretation. Do not delete these rows or overwrite either value simply because the broad descriptions appear inconsistent.

The existing baseline's derived field named `previously_contacted` is calculated solely from `pdays != 999`. That expression is verified, but the broad name may imply more than the source data establishes. In the next controlled preprocessing experiment, use an explicit name such as `pdays_recorded` and distinguish it from `has_previous_contacts = previous > 0`. Compare representations under the same development folds and document any change. The original baseline is retained as an experiment record.

## Interpreting the diagnostic checks

The observed categories match the publisher's vocabulary. No outer whitespace or selected basic numeric violations were found. These results validate particular checks, not the correctness of every label, the lack of noise, or the independence of customers. Values outside a training fold's numeric range describe a change of support and are not automatically data errors.

The file contains no full per-record date or reliable customer identifier. Chronological source order is used as an approximation; it does not permit exact reconstruction of release-time information, repeated-customer grouping, or a full temporal embargo. These limits need to remain visible in the report.
