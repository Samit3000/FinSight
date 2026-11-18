# P1.3 screening (folds F1, F2)

Generated 2026-11-18T18:13:02Z · data sha256 `74cba5b396ce…` · 300 step-block bootstrap resamples · locked test (≥651) not read.

## Disclosure

F1 and F2 are screening folds. F3 is the least-used fold, not an independent one: the recipient count was discovered by analysing the frozen champion's errors on steps 451-650 (which includes F3's test window), the champion configuration E0 was chosen using steps 551-743, and the 1:50 class weight was chosen on 451-550 and compared on 551-743. F2 is the most reused fold. The locked test (651-743) was not read in P1.3 (no labels, predictions or metrics) and gives no evidence here.

## Results are screening results

These folds were used to discover and choose the candidates. Treat them as optimistic.

## Configurations

| Label | Config | Features | Class weight |
|---|---|---|---|
| E0 | `hgb_baseline8_balanced` | 8: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER | balanced |
| E1 | `hgb_baseline8_rcpt_balanced` | 9: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER, recipient_prior_in_count | balanced |
| E2 | `hgb_baseline8_cw50_f32` | 8: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER | {0: 1.0, 1: 50.0} |
| E3 | `hgb_baseline8_rcpt_cw50` | 9: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER, recipient_prior_in_count | {0: 1.0, 1: 50.0} |
| E4 | `hgb_baseline7_rcpt_cw50` | 8: amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER, recipient_prior_in_count | {0: 1.0, 1: 50.0} |

All use the champion's float32 amount path and hyperparameters (max_iter 50, lr 0.1, 31 leaves, seed 42).

## Fold F1 (test steps 351–450, 1,070,802 rows, 1,156 fraud, prevalence 0.108%, 4.17 simulated days)

| | AP | AP 95% CI | ROC-AUC | Budget | Alerts | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0** | 0.3889 | 0.316–0.465 | 0.9478 | 50/day | 208 | 196.0 | 12.0 | 960.0 | 0.942 | 0.170 | 0.287 |
| | | | | 100/day | 417 | 325.7 | 91.3 | 830.3 | 0.781 | 0.282 | 0.414 |
| | | | | 200/day | 833 | 470.8 | 362.2 | 685.2 | 0.565 | 0.407 | 0.473 |
| **E1** | 0.4952 | 0.418–0.575 | 0.9733 | 50/day | 208 | 202.0 | 6.0 | 954.0 | 0.971 | 0.175 | 0.296 |
| | | | | 100/day | 417 | 356.2 | 60.8 | 799.8 | 0.854 | 0.308 | 0.453 |
| | | | | 200/day | 833 | 538.0 | 295.0 | 618.0 | 0.646 | 0.465 | 0.541 |
| **E2** | 0.4189 | 0.348–0.490 | 0.9522 | 50/day | 208 | 198.7 | 9.3 | 957.3 | 0.955 | 0.172 | 0.291 |
| | | | | 100/day | 417 | 353.5 | 63.5 | 802.5 | 0.848 | 0.306 | 0.450 |
| | | | | 200/day | 833 | 491.3 | 341.7 | 664.7 | 0.590 | 0.425 | 0.494 |
| **E3** | 0.5092 | 0.432–0.587 | 0.9743 | 50/day | 208 | 196.0 | 12.0 | 960.0 | 0.942 | 0.170 | 0.287 |
| | | | | 100/day | 417 | 379.0 | 38.0 | 777.0 | 0.909 | 0.328 | 0.482 |
| | | | | 200/day | 833 | 538.1 | 294.9 | 617.9 | 0.646 | 0.466 | 0.541 |
| **E4** | 0.5092 | 0.432–0.587 | 0.9743 | 50/day | 208 | 196.0 | 12.0 | 960.0 | 0.942 | 0.170 | 0.287 |
| | | | | 100/day | 417 | 379.0 | 38.0 | 777.0 | 0.909 | 0.328 | 0.482 |
| | | | | 200/day | 833 | 538.1 | 294.9 | 617.9 | 0.646 | 0.466 | 0.541 |

Paired AP differences on F1 (same rows, step-block bootstrap):

| Comparison | Isolates | ΔAP | 95% CI | Share of resamples not better |
|---|---|---:|---|---:|
| E1 vs E0 | recipient count (balanced weights) | +0.1063 | +0.0801 to +0.1381 | 0.00 |
| E2 vs E0 | class weight 1:50 (no recipient count) | +0.0300 | +0.0070 to +0.0507 | 0.00 |
| E3 vs E1 | class weight 1:50 (with recipient count) | +0.0140 | -0.0115 to +0.0369 | 0.20 |
| E3 vs E2 | recipient count (class weight 1:50) | +0.0903 | +0.0608 to +0.1163 | 0.00 |
| E4 vs E3 | removing log1p_amount | +0.0000 | +0.0000 to +0.0000 | 1.00 |
| E3 vs E0 | recipient count + class weight 1:50 combined | +0.1203 | +0.0750 to +0.1653 | 0.00 |
| E4 vs E0 | recipient count + class weight 1:50 - log1p_amount combined | +0.1203 | +0.0750 to +0.1653 | 0.00 |

Errors at 100 alerts/day by transaction hour on F1 (FN / FP, expected values; fraud per segment in header):

| | 00-05 (326 fraud, 5,235 rows) | 06-08 (122 fraud, 36,916 rows) | 09-17 (458 fraud, 660,482 rows) | 18-23 (250 fraud, 368,169 rows) |
|---|---:|---:|---:|---:|
| E0 | 88 / 91 | 107 / 0 | 413 / 0 | 222 / 0 |
| E1 | 80 / 55 | 105 / 0 | 402 / 5 | 213 / 1 |
| E2 | 102 / 50 | 102 / 2 | 388 / 10 | 210 / 2 |
| E3 | 88 / 3 | 105 / 0 | 390 / 10 | 194 / 25 |
| E4 | 88 / 3 | 105 / 0 | 390 / 10 | 194 / 25 |

Errors at 100 alerts/day by recipient prior count on F1 (FN / FP, expected values; fraud per segment in header):

| | 0 (731 fraud, 459,104 rows) | 1-5 (243 fraud, 282,694 rows) | 6-20 (144 fraud, 261,606 rows) | 21+ (38 fraud, 67,398 rows) |
|---|---:|---:|---:|---:|
| E0 | 551 / 4 | 147 / 26 | 107 / 42 | 25 / 19 |
| E1 | 534 / 5 | 141 / 20 | 100 / 31 | 24 / 5 |
| E2 | 551 / 7 | 134 / 12 | 95 / 31 | 22 / 13 |
| E3 | 515 / 21 | 138 / 2 | 99 / 14 | 25 / 1 |
| E4 | 515 / 21 | 138 / 2 | 99 / 14 | 25 / 1 |

## Fold F2 (test steps 451–550, 151,302 rows, 1,008 fraud, prevalence 0.666%, 4.17 simulated days)

| | AP | AP 95% CI | ROC-AUC | Budget | Alerts | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0** | 0.3115 | 0.249–0.379 | 0.9509 | 50/day | 208 | 140.9 | 67.1 | 867.1 | 0.677 | 0.140 | 0.232 |
| | | | | 100/day | 417 | 211.4 | 205.6 | 796.6 | 0.507 | 0.210 | 0.297 |
| | | | | 200/day | 833 | 303.5 | 529.5 | 704.5 | 0.364 | 0.301 | 0.330 |
| **E1** | 0.4717 | 0.390–0.554 | 0.9748 | 50/day | 208 | 192.3 | 15.7 | 815.7 | 0.924 | 0.191 | 0.316 |
| | | | | 100/day | 417 | 246.8 | 170.2 | 761.2 | 0.592 | 0.245 | 0.346 |
| | | | | 200/day | 833 | 392.2 | 440.8 | 615.8 | 0.471 | 0.389 | 0.426 |
| **E2** | 0.3663 | 0.307–0.432 | 0.9514 | 50/day | 208 | 168.8 | 39.2 | 839.2 | 0.811 | 0.168 | 0.278 |
| | | | | 100/day | 417 | 248.4 | 168.6 | 759.6 | 0.596 | 0.246 | 0.349 |
| | | | | 200/day | 833 | 365.0 | 468.0 | 643.0 | 0.438 | 0.362 | 0.397 |
| **E3** | 0.5162 | 0.446–0.592 | 0.9769 | 50/day | 208 | 192.4 | 15.6 | 815.6 | 0.925 | 0.191 | 0.316 |
| | | | | 100/day | 417 | 313.0 | 104.0 | 695.0 | 0.751 | 0.310 | 0.439 |
| | | | | 200/day | 833 | 448.0 | 385.0 | 560.0 | 0.538 | 0.444 | 0.487 |
| **E4** | 0.5162 | 0.446–0.592 | 0.9769 | 50/day | 208 | 192.4 | 15.6 | 815.6 | 0.925 | 0.191 | 0.316 |
| | | | | 100/day | 417 | 313.0 | 104.0 | 695.0 | 0.751 | 0.310 | 0.439 |
| | | | | 200/day | 833 | 448.0 | 385.0 | 560.0 | 0.538 | 0.444 | 0.487 |

Paired AP differences on F2 (same rows, step-block bootstrap):

| Comparison | Isolates | ΔAP | 95% CI | Share of resamples not better |
|---|---|---:|---|---:|
| E1 vs E0 | recipient count (balanced weights) | +0.1603 | +0.1307 to +0.1918 | 0.00 |
| E2 vs E0 | class weight 1:50 (no recipient count) | +0.0548 | +0.0394 to +0.0692 | 0.00 |
| E3 vs E1 | class weight 1:50 (with recipient count) | +0.0444 | +0.0155 to +0.0747 | 0.01 |
| E3 vs E2 | recipient count (class weight 1:50) | +0.1499 | +0.1167 to +0.1838 | 0.00 |
| E4 vs E3 | removing log1p_amount | +0.0000 | +0.0000 to +0.0000 | 1.00 |
| E3 vs E0 | recipient count + class weight 1:50 combined | +0.2047 | +0.1686 to +0.2414 | 0.00 |
| E4 vs E0 | recipient count + class weight 1:50 - log1p_amount combined | +0.2047 | +0.1686 to +0.2414 | 0.00 |

Errors at 100 alerts/day by transaction hour on F2 (FN / FP, expected values; fraud per segment in header):

| | 00-05 (254 fraud, 3,579 rows) | 06-08 (138 fraud, 1,036 rows) | 09-17 (340 fraud, 96,527 rows) | 18-23 (276 fraud, 50,160 rows) |
|---|---:|---:|---:|---:|
| E0 | 100 / 206 | 126 / 0 | 311 / 0 | 260 / 0 |
| E1 | 77 / 170 | 129 / 0 | 300 / 0 | 255 / 0 |
| E2 | 100 / 164 | 118 / 5 | 293 / 0 | 249 / 0 |
| E3 | 61 / 93 | 118 / 1 | 289 / 5 | 227 / 5 |
| E4 | 61 / 93 | 118 / 1 | 289 / 5 | 227 / 5 |

Errors at 100 alerts/day by recipient prior count on F2 (FN / FP, expected values; fraud per segment in header):

| | 0 (633 fraud, 67,669 rows) | 1-5 (211 fraud, 38,653 rows) | 6-20 (133 fraud, 35,810 rows) | 21+ (31 fraud, 9,170 rows) |
|---|---:|---:|---:|---:|
| E0 | 502 / 14 | 170 / 64 | 104 / 97 | 21 / 30 |
| E1 | 486 / 22 | 159 / 63 | 94 / 76 | 23 / 9 |
| E2 | 497 / 13 | 147 / 54 | 97 / 77 | 18 / 25 |
| E3 | 447 / 26 | 134 / 70 | 93 / 8 | 21 / 0 |
| E4 | 447 / 26 | 134 / 70 | 93 / 8 | 21 / 0 |

## Shortlist

Rule (declared before any F3 evaluation): A candidate (E1, E3, E4) is shortlisted for F3 confirmation only if, in BOTH F1 and F2: (a) the paired 95% CI of AP(candidate) - AP(E0) lies entirely above 0, and (b) tie-aware recall at 100 and at 200 alerts/day is not lower than E0's.

| Candidate | Fold | ΔAP CI low vs E0 | CI above 0 | Recall@100 not lower | Recall@200 not lower |
|---|---|---:|---|---|---|
| E1 | F1 | +0.0801 | True | True | True |
| E1 | F2 | +0.1307 | True | True | True |
| E3 | F1 | +0.0750 | True | True | True |
| E3 | F2 | +0.1686 | True | True | True |
| E4 | F1 | +0.0750 | True | True | True |
| E4 | F2 | +0.1686 | True | True | True |

**Shortlisted for F3:** E1, E3, E4.

## Notes

- Precision, recall and F1 are reported at fixed review capacities (alerts per simulated day), the protocol's operating points. No per-fold decision threshold is reported: no window exists that is disjoint from both a fold's training and test data without changing the model, and threshold policy is P1.4.
- Confidence intervals are given for AP and paired AP differences (test-sample variation only, not training variation). Budget metrics have no interval in the current protocol.
- Budget counts are expected values under random tie-breaking at the cut-off score, so they can be fractional.
- Scores are risk scores, not calibrated probabilities.
