# P1.3 confirmation (fold F3)

Generated 2026-11-18T18:19:22Z · data sha256 `74cba5b396ce…` · 300 step-block bootstrap resamples · locked test (≥651) not read.

## Disclosure

F1 and F2 are screening folds. F3 is the least-used fold, not an independent one: the recipient count was discovered by analysing the frozen champion's errors on steps 451-650 (which includes F3's test window), the champion configuration E0 was chosen using steps 551-743, and the 1:50 class weight was chosen on 451-550 and compared on 551-743. F2 is the most reused fold. The locked test (651-743) was not read in P1.3 (no labels, predictions or metrics) and gives no evidence here.

## Results are confirmation results

Shortlist carried over unchanged from the screening report: ['E1', 'E3', 'E4'].

F3 is not fully independent (see disclosure).

## Configurations

| Label | Config | Features | Class weight |
|---|---|---|---|
| E0 | `hgb_baseline8_balanced` | 8: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER | balanced |
| E1 | `hgb_baseline8_rcpt_balanced` | 9: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER, recipient_prior_in_count | balanced |
| E2 | `hgb_baseline8_cw50_f32` | 8: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER | {0: 1.0, 1: 50.0} |
| E3 | `hgb_baseline8_rcpt_cw50` | 9: amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER, recipient_prior_in_count | {0: 1.0, 1: 50.0} |
| E4 | `hgb_baseline7_rcpt_cw50` | 8: amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER, recipient_prior_in_count | {0: 1.0, 1: 50.0} |

All use the champion's float32 amount path and hyperparameters (max_iter 50, lr 0.1, 31 leaves, seed 42).

## Fold F3 (test steps 551–650, 115,770 rows, 1,126 fraud, prevalence 0.973%, 4.17 simulated days)

| | AP | AP 95% CI | ROC-AUC | Budget | Alerts | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E0** | 0.4222 | 0.329–0.523 | 0.9520 | 50/day | 208 | 176.2 | 31.8 | 949.8 | 0.847 | 0.157 | 0.264 |
| | | | | 100/day | 417 | 280.2 | 136.8 | 845.8 | 0.672 | 0.249 | 0.363 |
| | | | | 200/day | 833 | 469.0 | 364.0 | 657.0 | 0.563 | 0.416 | 0.479 |
| **E1** | 0.6249 | 0.534–0.716 | 0.9771 | 50/day | 208 | 204.2 | 3.8 | 921.8 | 0.982 | 0.181 | 0.306 |
| | | | | 100/day | 417 | 342.0 | 75.0 | 784.0 | 0.820 | 0.304 | 0.443 |
| | | | | 200/day | 833 | 584.0 | 249.0 | 542.0 | 0.701 | 0.519 | 0.596 |
| **E2** | 0.4621 | 0.378–0.554 | 0.9526 | 50/day | 208 | 195.0 | 13.0 | 931.0 | 0.938 | 0.173 | 0.292 |
| | | | | 100/day | 417 | 315.3 | 101.7 | 810.7 | 0.756 | 0.280 | 0.409 |
| | | | | 200/day | 833 | 469.6 | 363.4 | 656.4 | 0.564 | 0.417 | 0.479 |
| **E3** | 0.6530 | 0.573–0.729 | 0.9768 | 50/day | 208 | 206.0 | 2.0 | 920.0 | 0.990 | 0.183 | 0.309 |
| | | | | 100/day | 417 | 375.0 | 42.0 | 751.0 | 0.899 | 0.333 | 0.486 |
| | | | | 200/day | 833 | 583.2 | 249.8 | 542.8 | 0.700 | 0.518 | 0.595 |
| **E4** | 0.6530 | 0.573–0.729 | 0.9768 | 50/day | 208 | 206.0 | 2.0 | 920.0 | 0.990 | 0.183 | 0.309 |
| | | | | 100/day | 417 | 375.0 | 42.0 | 751.0 | 0.899 | 0.333 | 0.486 |
| | | | | 200/day | 833 | 583.2 | 249.8 | 542.8 | 0.700 | 0.518 | 0.595 |

Paired AP differences on F3 (same rows, step-block bootstrap):

| Comparison | Isolates | ΔAP | 95% CI | Share of resamples not better |
|---|---|---:|---|---:|
| E1 vs E0 | recipient count (balanced weights) | +0.2027 | +0.1684 to +0.2356 | 0.00 |
| E2 vs E0 | class weight 1:50 (no recipient count) | +0.0400 | +0.0181 to +0.0552 | 0.00 |
| E3 vs E1 | class weight 1:50 (with recipient count) | +0.0281 | +0.0080 to +0.0461 | 0.00 |
| E3 vs E2 | recipient count (class weight 1:50) | +0.1908 | +0.1648 to +0.2185 | 0.00 |
| E4 vs E3 | removing log1p_amount | +0.0000 | +0.0000 to +0.0000 | 1.00 |
| E3 vs E0 | recipient count + class weight 1:50 combined | +0.2308 | +0.1879 to +0.2613 | 0.00 |
| E4 vs E0 | recipient count + class weight 1:50 - log1p_amount combined | +0.2308 | +0.1879 to +0.2613 | 0.00 |

Errors at 100 alerts/day by transaction hour on F3 (FN / FP, expected values; fraud per segment in header):

| | 00-05 (310 fraud, 3,110 rows) | 06-08 (126 fraud, 3,506 rows) | 09-17 (408 fraud, 69,137 rows) | 18-23 (282 fraud, 40,017 rows) |
|---|---:|---:|---:|---:|
| E0 | 102 / 137 | 100 / 0 | 383 / 0 | 261 / 0 |
| E1 | 71 / 75 | 105 / 0 | 362 / 0 | 246 / 0 |
| E2 | 119 / 100 | 104 / 2 | 347 / 0 | 241 / 0 |
| E3 | 86 / 39 | 108 / 0 | 349 / 1 | 208 / 2 |
| E4 | 86 / 39 | 108 / 0 | 349 / 1 | 208 / 2 |

Errors at 100 alerts/day by recipient prior count on F3 (FN / FP, expected values; fraud per segment in header):

| | 0 (714 fraud, 51,571 rows) | 1-5 (239 fraud, 30,236 rows) | 6-20 (136 fraud, 27,099 rows) | 21+ (37 fraud, 6,864 rows) |
|---|---:|---:|---:|---:|
| E0 | 538 / 13 | 182 / 43 | 99 / 60 | 27 / 21 |
| E1 | 505 / 19 | 156 / 45 | 98 / 11 | 25 / 0 |
| E2 | 542 / 15 | 157 / 35 | 91 / 39 | 21 / 12 |
| E3 | 474 / 20 | 156 / 21 | 98 / 1 | 23 / 0 |
| E4 | 474 / 20 | 156 / 21 | 98 / 1 | 23 / 0 |

## Confirmation decision

Rule: A shortlisted candidate is confirmed on F3 only if the paired 95% CI of AP(candidate) - AP(E0) lies entirely above 0 and tie-aware recall at 100 and 200 alerts/day is not lower than E0's on F3. Confirmation is not promotion: promotion and the single locked-test evaluation need separate approval.

| Candidate | ΔAP vs E0 | 95% CI | Recall@100 vs E0 | Recall@200 vs E0 | Confirmed |
|---|---:|---|---|---|---|
| E1 | +0.2027 | +0.1684 to +0.2356 | 0.304 vs 0.249 | 0.519 vs 0.416 | **True** |
| E3 | +0.2308 | +0.1879 to +0.2613 | 0.333 vs 0.249 | 0.518 vs 0.416 | **True** |
| E4 | +0.2308 | +0.1879 to +0.2613 | 0.333 vs 0.249 | 0.518 vs 0.416 | **True** |

## Notes

- Precision, recall and F1 are reported at fixed review capacities (alerts per simulated day), the protocol's operating points. No per-fold decision threshold is reported: no window exists that is disjoint from both a fold's training and test data without changing the model, and threshold policy is P1.4.
- Confidence intervals are given for AP and paired AP differences (test-sample variation only, not training variation). Budget metrics have no interval in the current protocol.
- Budget counts are expected values under random tie-breaking at the cut-off score, so they can be fractional.
- Scores are risk scores, not calibrated probabilities.
