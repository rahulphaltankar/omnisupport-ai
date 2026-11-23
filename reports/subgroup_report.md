# Subgroup analysis (frozen test set)

Classification: escalation model at the validation-chosen threshold **0.24** on **166,640** held-out test tickets (overall recall 0.776, FPR 0.457, flag rate 0.555).

Age band, language and accessibility flag were **not** model inputs (leakage/sensitivity register); they are used only to audit outcomes. Groups with n < 1000 are flagged and not interpreted. Recall CIs are Wilson 95%.

## customer_age_band

| value   |     n |   base_rate |   mean_pred |   calib_gap |   flag_rate |   recall |   recall_lo |   recall_hi |   fpr |   precision |
|:--------|------:|------------:|------------:|------------:|------------:|---------:|------------:|------------:|------:|------------:|
| 18-24   | 16745 |       0.307 |       0.31  |       0.003 |       0.553 |    0.777 |       0.765 |       0.788 | 0.454 |       0.432 |
| 25-34   | 41764 |       0.311 |       0.312 |       0.001 |       0.557 |    0.778 |       0.771 |       0.785 | 0.458 |       0.434 |
| 35-44   | 38176 |       0.307 |       0.31  |       0.003 |       0.553 |    0.775 |       0.767 |       0.782 | 0.454 |       0.43  |
| 45-54   | 31664 |       0.305 |       0.309 |       0.004 |       0.556 |    0.774 |       0.766 |       0.782 | 0.461 |       0.425 |
| 55-64   | 23200 |       0.302 |       0.31  |       0.009 |       0.554 |    0.769 |       0.759 |       0.779 | 0.461 |       0.418 |
| 65+     | 15091 |       0.309 |       0.311 |       0.003 |       0.557 |    0.783 |       0.771 |       0.795 | 0.456 |       0.434 |

## preferred_language

| value    |      n |   base_rate |   mean_pred |   calib_gap |   flag_rate |   recall |   recall_lo |   recall_hi |   fpr |   precision |
|:---------|-------:|------------:|------------:|------------:|------------:|---------:|------------:|------------:|------:|------------:|
| English  | 148421 |       0.307 |       0.311 |       0.003 |       0.555 |    0.775 |       0.771 |       0.778 | 0.458 |       0.428 |
| Polish   |   4161 |       0.304 |       0.305 |       0.001 |       0.547 |    0.776 |       0.752 |       0.798 | 0.447 |       0.431 |
| Punjabi  |   4043 |       0.303 |       0.306 |       0.003 |       0.548 |    0.777 |       0.752 |       0.799 | 0.449 |       0.429 |
| Romanian |   3367 |       0.301 |       0.306 |       0.005 |       0.547 |    0.773 |       0.747 |       0.798 | 0.449 |       0.426 |
| Urdu     |   4181 |       0.308 |       0.317 |       0.009 |       0.567 |    0.806 |       0.783 |       0.827 | 0.461 |       0.438 |
| Welsh    |   2467 |       0.314 |       0.313 |      -0     |       0.559 |    0.782 |       0.751 |       0.809 | 0.458 |       0.438 |

## accessibility_support_flag

|   value |      n |   base_rate |   mean_pred |   calib_gap |   flag_rate |   recall |   recall_lo |   recall_hi |   fpr |   precision |
|--------:|-------:|------------:|------------:|------------:|------------:|---------:|------------:|------------:|------:|------------:|
|       0 | 160501 |       0.305 |        0.31 |       0.005 |       0.555 |    0.776 |       0.772 |       0.779 | 0.458 |       0.427 |
|       1 |   6139 |       0.35  |        0.31 |      -0.04  |       0.555 |    0.772 |       0.753 |       0.789 | 0.438 |       0.487 |

## customer_region

| value            |     n |   base_rate |   mean_pred |   calib_gap |   flag_rate |   recall |   recall_lo |   recall_hi |   fpr |   precision |
|:-----------------|------:|------------:|------------:|------------:|------------:|---------:|------------:|------------:|------:|------------:|
| East Midlands    | 13133 |       0.309 |       0.312 |       0.003 |       0.556 |    0.78  |       0.767 |       0.793 | 0.455 |       0.433 |
| London           | 29554 |       0.309 |       0.31  |       0.001 |       0.553 |    0.77  |       0.762 |       0.779 | 0.456 |       0.43  |
| North West       | 19638 |       0.305 |       0.31  |       0.006 |       0.555 |    0.768 |       0.757 |       0.778 | 0.462 |       0.421 |
| Northern Ireland |  6429 |       0.31  |       0.311 |       0     |       0.559 |    0.787 |       0.769 |       0.804 | 0.457 |       0.437 |
| Scotland         | 13183 |       0.309 |       0.312 |       0.003 |       0.565 |    0.777 |       0.764 |       0.79  | 0.47  |       0.425 |
| South East       | 24467 |       0.303 |       0.31  |       0.007 |       0.556 |    0.78  |       0.771 |       0.79  | 0.459 |       0.425 |
| South West       | 14869 |       0.301 |       0.31  |       0.009 |       0.552 |    0.776 |       0.763 |       0.788 | 0.455 |       0.424 |
| Unknown          |  2613 |       0.304 |       0.312 |       0.009 |       0.553 |    0.782 |       0.752 |       0.809 | 0.452 |       0.43  |
| Wales            | 10025 |       0.31  |       0.311 |       0.001 |       0.557 |    0.779 |       0.764 |       0.793 | 0.457 |       0.434 |
| West Midlands    | 16490 |       0.309 |       0.31  |       0.001 |       0.552 |    0.778 |       0.766 |       0.789 | 0.452 |       0.435 |
| Yorkshire        | 16239 |       0.309 |       0.31  |       0.001 |       0.551 |    0.773 |       0.761 |       0.785 | 0.452 |       0.433 |

## support_channel

| value      |     n |   base_rate |   mean_pred |   calib_gap |   flag_rate |   recall |   recall_lo |   recall_hi |   fpr |   precision |
|:-----------|------:|------------:|------------:|------------:|------------:|---------:|------------:|------------:|------:|------------:|
| email      | 49991 |       0.302 |       0.31  |       0.008 |       0.554 |    0.778 |       0.771 |       0.784 | 0.457 |       0.424 |
| mobile_app | 33092 |       0.311 |       0.311 |      -0     |       0.556 |    0.774 |       0.765 |       0.782 | 0.457 |       0.433 |
| phone      | 16765 |       0.308 |       0.311 |       0.002 |       0.554 |    0.775 |       0.763 |       0.786 | 0.456 |       0.431 |
| web_chat   | 66792 |       0.309 |       0.31  |       0.002 |       0.556 |    0.775 |       0.77  |       0.781 | 0.459 |       0.43  |

## product_category

| value       |     n |   base_rate |   mean_pred |   calib_gap |   flag_rate |   recall |   recall_lo |   recall_hi |   fpr |   precision |
|:------------|------:|------------:|------------:|------------:|------------:|---------:|------------:|------------:|------:|------------:|
| appliances  | 16324 |       0.376 |       0.377 |       0     |       0.685 |    0.865 |       0.856 |       0.874 | 0.576 |       0.475 |
| beauty      | 16740 |       0.289 |       0.295 |       0.006 |       0.527 |    0.75  |       0.737 |       0.762 | 0.437 |       0.411 |
| electronics | 31785 |       0.351 |       0.353 |       0.002 |       0.645 |    0.837 |       0.83  |       0.844 | 0.541 |       0.456 |
| fashion     | 30148 |       0.284 |       0.289 |       0.005 |       0.513 |    0.742 |       0.732 |       0.751 | 0.423 |       0.41  |
| grocery     | 16682 |       0.255 |       0.259 |       0.004 |       0.44  |    0.681 |       0.667 |       0.695 | 0.357 |       0.395 |
| home        | 25032 |       0.297 |       0.297 |       0     |       0.529 |    0.748 |       0.738 |       0.757 | 0.437 |       0.419 |
| sports      | 16583 |       0.294 |       0.299 |       0.005 |       0.537 |    0.755 |       0.743 |       0.767 | 0.446 |       0.414 |
| toys        | 13346 |       0.293 |       0.299 |       0.006 |       0.529 |    0.747 |       0.733 |       0.761 | 0.438 |       0.414 |

## Findings

- Largest absolute recall gap on a sensitive/support attribute: **0.030**; largest FPR gap: **0.020**.
- Groups (n >= 1000) whose recall CI excludes the overall recall: `preferred_language=Urdu` (+0.030), `product_category=appliances` (+0.090), `product_category=beauty` (-0.026), `product_category=electronics` (+0.061), `product_category=fashion` (-0.034), `product_category=grocery` (-0.094), `product_category=home` (-0.028), `product_category=sports` (-0.020), `product_category=toys` (-0.028)
- Flag-rate spread (relative to overall): `product_category=grocery` x0.79; `product_category=appliances` x1.23. Differences in flag rate largely track differences in base rate (see `base_rate` column), which is expected for a risk score; recall/FPR parity is the better fairness lens.

## Regression (resolution time) by sensitive attribute

| attribute                  | value    |      n |   mae |   bias |   mean_actual |   mae_ratio_vs_overall |
|:---------------------------|:---------|-------:|------:|-------:|--------------:|-----------------------:|
| customer_age_band          | 18-24    |  16745 | 3.592 | -0.841 |        24.084 |                  1.029 |
| customer_age_band          | 25-34    |  41764 | 3.444 | -0.712 |        23.976 |                  0.986 |
| customer_age_band          | 35-44    |  38176 | 3.482 | -0.726 |        23.868 |                  0.997 |
| customer_age_band          | 45-54    |  31664 | 3.56  | -0.815 |        23.943 |                  1.02  |
| customer_age_band          | 55-64    |  23200 | 3.471 | -0.68  |        23.867 |                  0.994 |
| customer_age_band          | 65+      |  15091 | 3.424 | -0.679 |        23.917 |                  0.981 |
| preferred_language         | English  | 148421 | 3.49  | -0.732 |        23.924 |                  0.999 |
| preferred_language         | Polish   |   4161 | 3.417 | -0.773 |        23.799 |                  0.979 |
| preferred_language         | Punjabi  |   4043 | 3.551 | -0.844 |        24.051 |                  1.017 |
| preferred_language         | Romanian |   3367 | 3.644 | -0.975 |        24.162 |                  1.044 |
| preferred_language         | Urdu     |   4181 | 3.372 | -0.596 |        24.18  |                  0.966 |
| preferred_language         | Welsh    |   2467 | 3.634 | -0.948 |        23.942 |                  1.041 |
| accessibility_support_flag | 0        | 160501 | 3.481 | -0.722 |        23.921 |                  0.997 |
| accessibility_support_flag | 1        |   6139 | 3.771 | -1.215 |        24.318 |                  1.08  |


## Interpretation and limits

- The data are synthetic; a gap here is a property of the generator and model, not evidence about real customers. The procedure is what transfers.
- Equal recall across groups was **not** enforced; the model is advisory and a human decides. If a significant gap were operationally material we would (1) check base-rate differences, (2) evaluate group-specific thresholds on validation only, (3) re-audit on a fresh period.
- Accessibility-flagged customers should never receive a *worse* service because of a score: use the score only to raise priority for review (a false positive costs a few minutes; a false negative costs a delayed escalation).
- Intersectional groups were not analysed (small n); this is a known limitation (see risk register).