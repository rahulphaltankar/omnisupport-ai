# Classical ML: escalation classification and resolution-time regression

Scripts: `scripts/03_ml_classification.py`, `scripts/04_ml_regression.py` · Evidence: `artifacts/ml_classification/`, `artifacts/ml_regression/` · Prediction time **T0 = ticket creation** (see `reports/problem_framing.md`, `reports/leakage_register.md`).

## Protocol (shared)

* 996,000 cleaned rows, **temporal split**: train < 2025-09-01, validation Sept–Oct 2025, **test Nov–Dec 2025 (touched once, after model/features/threshold were frozen)**. Group-aware 3-fold CV (`GroupKFold` by customer) on a train subsample for model comparison and tuning.
* All fitted transforms (median imputation, train-quantile winsorising, one-hot, scaling) live inside sklearn pipelines fitted on training folds only.
* T0 features only (24 numeric + 4 categorical). `assert_features_valid` fails the run if a post-outcome or triage field slips in. Age band, language and accessibility are **not** inputs (used only in `reports/subgroup_report.md`).
* Success criteria were written before the test run: classification PR-AUC ≥ 0.50, recall ≥ 0.60, ROC-AUC ≥ 0.70; regression MAE at least 30 % better than the median baseline.

## 1. Escalation classification (target: `escalated`, base rate ≈ 31 %)

| model (5-fold grouped CV, PR-AUC mean ± sd) | PR-AUC | ROC-AUC |
|---|---:|---:|
| dummy prior (baseline) | 0.310 ± 0.002 | 0.500 |
| logistic regression | **0.573 ± 0.004** | 0.733 |
| HistGradientBoosting (defaults) | 0.569 ± 0.003 | 0.728 |
| random forest | 0.565 ± 0.002 | 0.726 |
| HistGradientBoosting **tuned** (RandomizedSearchCV, 6 params) | 0.573 | – |

**Honest reading:** the three real models are within about one standard deviation of each other; the signal is mostly linear/low-order (previous tickets, order value, issue type). Tuning bought +0.004 PR-AUC, the same as the fold noise. We kept the tuned HGB because it is calibrated (validation ECE 0.004), supports the cost-based threshold cleanly and its interactions might matter if the data drift, **not** because it wins; logistic regression on the frozen test set is statistically indistinguishable (test PR-AUC 0.569 vs 0.572, bootstrap 95 % CI for HGB 0.563–0.579).

**Threshold.** Business cost: a missed escalation (FN) is 3× as costly as an unnecessary priority review (FP). Minimising expected cost on *validation* gives **threshold 0.24** (not 0.5, which would flag only 16 % and miss 67 % of escalations — `threshold_table_validation.csv`).

**Frozen test result (166,640 tickets, Nov–Dec 2025):** ROC-AUC 0.734, PR-AUC 0.572 (base rate 0.307), **recall 0.776, precision 0.429, F1 0.552**, Brier 0.179; confusion matrix TP 39,672 / FP 52,836 / FN 11,478 / TN 62,654; 55.5 % of tickets flagged. All three pre-registered criteria passed.

**Business interpretation.** The score is a *prioritisation aid*, not a decision: it catches ≈ 78 % of tickets that will escalate while flagging about 55 % of the queue, i.e. precision 43 % against a 31 % base rate (lift ≈ 1.4×). That is useful for ordering a queue and for staffing a "priority review" lane, but not strong enough to automate any customer-facing action.

**Ablations / leakage demonstration** (`ablation_*.csv`, `leakage_demonstration.csv`, `fig_leakage_demonstration.png`). Removing text-derived features, calendar, interactions or tenure changes validation PR-AUC by ≤ 0.0003; removing `issue_title` costs 0.010. Adding the sensitive attributes would add only +0.0003, so excluding them costs nothing. For contrast, adding fields that are **not** known at T0 inflates the score: triage labels +0.014, post-contact sentiment +0.028, **CSAT +0.184**, **resolution time +0.228**, the final disposition itself → PR-AUC 1.0. A paper-perfect model would therefore have been a leakage artefact.

**Error analysis (validation, threshold 0.24).** (1) *Missed escalations concentrate in first-contact customers* (`previous_ticket_count` = 0: 67 % of true escalations missed; =1: 46 %). (2) *Over-flagging of repeat contacts and lost-parcel/long-delay tickets* (≥ 5 previous tickets: ~100 % of non-escalations flagged; `Parcel appears lost` 98 %; delay 11–20 days 97 %): these groups are escalated so often that the model cannot discriminate within them. (3) *Product-information tickets* are missed 43 % of the time. Mitigations are in `reports/risk_register.md` (do not rely on the score for first-contact customers; always allow manual escalation).

## 2. Resolution time regression (target: hours to resolution; long right tail)

| model (5-fold grouped CV) | MAE (h) | RMSE (h) |
|---|---:|---:|
| median baseline | 6.75 | 14.74 |
| mean baseline | 6.78 | 14.72 |
| ridge | **3.57** | 12.45 |
| HistGradientBoosting | 3.59 | 12.45 |
| random forest | 3.64 | 12.48 |
| HGB tuned (absolute-error loss, winsorised target) | 3.57 (validation) | 13.68 |

Again the linear model ties the boosted one: the structure is additive. **Frozen test:** tuned HGB MAE **3.49 h**, RMSE 11.38 h, median AE 2.78 h, R² 0.31 (median baseline MAE 6.72 h → **48 % lower MAE**, 17 % lower RMSE). Ridge reaches MAE 3.53 / RMSE 11.35, so the choice of model is again not material.
MAE ≫ median-error gap and RMSE ≈ 3× MAE reflect the heavy tail of very long resolutions that no T0 feature explains (excluding the top 0.1 % of targets MAE falls to 3.31 h). Validation R² (0.24) is lower than test R² (0.31): the periods differ in how much variance sits in the tail, a reminder that R² is not stable here.

**Prediction interval.** Quantile HGBs give a 90 % interval; empirical coverage 0.898 on validation and **0.898 on test** (nominal 0.90), mean width 13.4 h — usable as "typically between X and Y hours".
**SLA risk.** For "> 36 h" tickets, a predicted-time cut-off of 34 h (max-F1 on validation) gives precision 0.91 and recall 0.61 on test.
**Error patterns:** the model under-predicts long tickets (mean error −4.4 h in the 36–60 h bucket, −2.3 h at 24–36 h; regression to the mean) and has its largest MAE for "Login not working" (4.0 h) and "Card charged incorrectly" (3.9 h).

## Limits

* Data are synthetic and the label structure is simple, so conclusions about *which family wins* will not transfer to real systems; the protocol (temporal split, group CV, leakage register, frozen test) does.
* One test evaluation per task; no multiple-testing correction needed, but the validation set was used for threshold and cut-off choice.
* The cost ratio (FN = 3 × FP) is an assumption (stated in `reports/problem_framing.md`); a different ratio moves the threshold along `threshold_table_validation.csv`.
