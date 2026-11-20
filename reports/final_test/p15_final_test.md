# P1.5 sealed final test (steps 651-743)

Run 2026-11-20T06:27:30Z · code `1ec3a74b5a` · pre-registered at `355db664d2` · data sha256 `74cba5b396ce…`

## Decision: PROMOTE E4 (pending separate promotion approval)

Rule (pre-registered): Promote E4 (hgb_baseline7_rcpt_cw50) over the champion configuration E0 (hgb_baseline8_balanced), both trained on steps 1-550, only if on the sealed steps 651-743: (a) the 95% paired step-block bootstrap interval of AP(E4) - AP(E0) lies entirely above 0, and (b) tie-aware recall at 100 and at 200 alerts per simulated day of E4 is not lower than E0's. Otherwise keep the champion. The frozen champion artifact, the rule engine and the alert policies are reported for context and cannot change the decision. No setting is changed after the sealed labels are read.

| Check | Result |
|---|---|
| dAP_ci_above_0 | True |
| recall_100_not_lower | True |
| recall_200_not_lower | True |

AP(E4) - AP(E0) = +0.2004, 95% CI +0.1441 to +0.2308.

## Disclosure

The sealed steps are not pristine: before P0 they were inside the old holdout (steps 551-743), which was used to choose the champion's 8-feature set and to compare the 1:50 class weight (EXP-07). They were never used to discover or screen the recipient feature (P1.1-P1.3 read steps <= 650 only). Development folds F1-F3 were reused many times; this run is the only use of steps 651-743 in P1.

## Systems on the sealed steps

| System | AP | AP 95% CI | ROC-AUC | Budget | Alerts | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| **E4** | 0.6966 | 0.638–0.765 | 0.9728 | 50/day | 194 | 194.0 | 0.0 | 828.0 | 1.000 | 0.190 | 0.319 |
| | | | | 100/day | 388 | 376.0 | 12.0 | 646.0 | 0.969 | 0.368 | 0.533 |
| | | | | 200/day | 775 | 559.0 | 216.0 | 463.0 | 0.721 | 0.547 | 0.622 |
| **E0** | 0.4962 | 0.424–0.601 | 0.9451 | 50/day | 194 | 181.8 | 12.2 | 840.2 | 0.937 | 0.178 | 0.299 |
| | | | | 100/day | 388 | 321.8 | 66.2 | 700.2 | 0.829 | 0.315 | 0.456 |
| | | | | 200/day | 775 | 451.0 | 324.0 | 571.0 | 0.582 | 0.441 | 0.502 |
| **champion_artifact** | 0.4909 | 0.420–0.600 | 0.9447 | 50/day | 194 | 180.9 | 13.1 | 841.1 | 0.932 | 0.177 | 0.297 |
| | | | | 100/day | 388 | 313.0 | 75.0 | 709.0 | 0.807 | 0.306 | 0.444 |
| | | | | 200/day | 775 | 448.0 | 327.0 | 574.0 | 0.578 | 0.438 | 0.499 |

Frozen champion at its threshold 0.966: 173.2 alerts/day, precision 0.6423, recall 0.4217.
Rule engine HIGH tier: 66.1 alerts/day, precision 0.781, recall 0.196.

## Alert policies (context)

| Policy | Alerts/day | Whole-day min / median / max | Precision | Recall | R 00-05 | R 06-23 | R <100k |
|---|---:|---|---:|---:|---:|---:|---:|
| E4 B @50 | 50.1 | 53 / 54 / 56 | 1.000 | 0.190 | 0.4315 | 0.1124 | 0.0 |
| E4 Q12 @50 | 49.3 | 50 / 51 / 52 | 1.000 | 0.187 | 0.3669 | 0.1292 | 0.0094 |
| E4 B @100 | 90.3 | 86 / 94 / 103 | 0.989 | 0.339 | 0.754 | 0.2054 | 0.0939 |
| E4 Q12 @100 | 98.6 | 100 / 101 / 102 | 0.970 | 0.363 | 0.7114 | 0.2508 | 0.1201 |
| E4 B @200 | 183.0 | 182 / 212 / 242 | 0.753 | 0.522 | 0.9718 | 0.3786 | 0.3099 |
| E4 Q12 @200 | 197.7 | 200 / 202 / 205 | 0.790 | 0.592 | 0.9597 | 0.4746 | 0.4366 |
| E0 B @50 | 65.0 | 58 / 74 / 89 | 0.901 | 0.222 | 0.6452 | 0.0866 | 0.0798 |
| E0 Q12 @50 | 49.8 | 50 / 51 / 52 | 0.962 | 0.182 | 0.3901 | 0.115 | 0.0078 |
| E0 B @100 | 89.8 | 89 / 106 / 124 | 0.842 | 0.287 | 0.879 | 0.0969 | 0.1643 |
| E0 Q12 @100 | 97.8 | 100 / 100 / 100 | 0.908 | 0.337 | 0.7502 | 0.2045 | 0.1475 |
| E0 B @200 | 137.0 | 150 / 169 / 188 | 0.770 | 0.400 | 0.9677 | 0.2183 | 0.2347 |
| E0 Q12 @200 | 186.6 | 200 / 202 / 204 | 0.732 | 0.518 | 0.9395 | 0.3826 | 0.3149 |

## Notes

- Every model was trained on steps 1-550; thresholds and the queue were calibrated on steps 551-650 (scores only).
- Budget counts are expected values under random tie-breaking at the cut-off score.
- Intervals reflect test-sample variation only, not training variation.
- Promotion is a separate step (P1.6) and needs its own approval.
