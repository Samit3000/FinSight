# P1.4 alert policy — selection (F1, F2)

Config `hgb_baseline7_rcpt_cw50` (E4) · generated 2026-11-18T19:25:08Z · data sha256 `74cba5b396ce…` · locked test (≥651) not read.

## Disclosure

Fold models here are trained on steps <= T-100 (T = the fold's training end), so they differ from the P1.3 fold models and their numbers are not directly comparable. Thresholds are set only on the calibration window (T-99..T), which lies before the test window. F1 and F2 choose the policy; F3 was already used for P1.3 confirmation, so this is its second use and it is not independent. The locked test (651-743) is not read.

Policies: **A** top-K over the whole test period (retrospective reference) · **B** one fixed threshold set on the calibration window · **C_s** fixed thresholds per hour band with share s of the budget reserved for 06-23.

## Fold F1

Model trained on steps 1–250 (3,248,292 rows, 2,841 fraud); thresholds set on 251–350 (1,695,623 rows); test 351–450 (1,070,802 rows, 1,156 fraud, 4.17 simulated days).

| Budget | Policy | Alerts/day | Full-day alerts min / median / max | TP | FP | FN | Precision | Recall | F1 | R 00-05 | R 06-08 | R 09-17 | R 18-23 | R 06-23 | R <100k | FP share 00-05 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | A | 49.9 | 42 / 46 / 64 | 201.0 | 7.0 | 955.0 | 0.966 | 0.174 | 0.295 | 0.297 | 0.115 | 0.118 | 0.144 | 0.125 | 0.000 | 0.000 |
| 50 | B | 71.8 | 65 / 73 / 90 | 290.0 | 9.0 | 866.0 | 0.970 | 0.251 | 0.399 | 0.568 | 0.123 | 0.118 | 0.144 | 0.127 | 0.060 | 0.222 |
| 50 | C_0.3 | 70.1 | 69 / 76 / 85 | 284.0 | 8.0 | 872.0 | 0.973 | 0.246 | 0.392 | 0.681 | 0.066 | 0.063 | 0.100 | 0.075 | 0.080 | 0.375 |
| 50 | C_0.5 | 72.2 | 67 / 73 / 90 | 290.0 | 11.0 | 866.0 | 0.964 | 0.251 | 0.398 | 0.568 | 0.123 | 0.118 | 0.144 | 0.127 | 0.060 | 0.182 |
| 100 | A | 100.1 | 92 / 97 / 123 | 371.0 | 46.0 | 785.0 | 0.890 | 0.321 | 0.472 | 0.706 | 0.147 | 0.138 | 0.240 | 0.170 | 0.100 | 0.109 |
| 100 | B | 114.5 | 106 / 108 / 144 | 412.0 | 65.0 | 744.0 | 0.864 | 0.356 | 0.505 | 0.770 | 0.172 | 0.159 | 0.268 | 0.194 | 0.104 | 0.108 |
| 100 | C_0.3 | 126.7 | 117 / 144 / 147 | 421.0 | 107.0 | 735.0 | 0.797 | 0.364 | 0.500 | 0.908 | 0.131 | 0.120 | 0.216 | 0.151 | 0.179 | 0.757 |
| 100 | C_0.5 | 115.9 | 106 / 115 / 139 | 421.0 | 62.0 | 735.0 | 0.872 | 0.364 | 0.514 | 0.822 | 0.147 | 0.153 | 0.260 | 0.184 | 0.119 | 0.210 |
| 200 | A | 199.9 | 145 / 216 / 228 | 480.2 | 352.8 | 675.8 | 0.576 | 0.415 | 0.483 | 0.868 | 0.221 | 0.197 | 0.320 | 0.237 | 0.169 | 0.119 |
| 200 | B | 228.7 | 148 / 233 / 278 | 488.0 | 465.0 | 668.0 | 0.512 | 0.422 | 0.463 | 0.868 | 0.221 | 0.214 | 0.320 | 0.247 | 0.169 | 0.090 |
| 200 | C_0.3 | 198.7 | 194 / 214 / 260 | 491.0 | 337.0 | 665.0 | 0.593 | 0.425 | 0.495 | 0.982 | 0.213 | 0.162 | 0.284 | 0.206 | 0.229 | 0.801 |
| 200 | C_0.5 | 264.2 | 193 / 278 / 322 | 516.0 | 585.0 | 640.0 | 0.469 | 0.446 | 0.457 | 0.954 | 0.221 | 0.214 | 0.320 | 0.247 | 0.209 | 0.277 |

Calibration (thresholds and B's share of calibration alerts in 06-23):

| Budget | B threshold | B share 06-23 | C_0.3 thresholds (00-05 / 06-23) | C_0.5 thresholds (00-05 / 06-23) |
|---:|---:|---:|---|---|
| 50 | 0.9680 | 0.48 | 0.9589 / 0.9872 | 0.9680 / 0.9660 |
| 100 | 0.9368 | 0.56 | 0.8545 / 0.9608 | 0.9135 / 0.9473 |
| 200 | 0.8919 | 0.88 | 0.5910 / 0.9202 | 0.7788 / 0.8919 |

## Fold F2

Model trained on steps 1–350 (4,943,915 rows, 3,901 fraud); thresholds set on 351–450 (1,070,802 rows); test 451–550 (151,302 rows, 1,008 fraud, 4.17 simulated days).

| Budget | Policy | Alerts/day | Full-day alerts min / median / max | TP | FP | FN | Precision | Recall | F1 | R 00-05 | R 06-08 | R 09-17 | R 18-23 | R 06-23 | R <100k | FP share 00-05 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | A | 49.9 | 49 / 53 / 58 | 195.0 | 13.0 | 813.0 | 0.938 | 0.194 | 0.321 | 0.512 | 0.043 | 0.112 | 0.076 | 0.086 | 0.057 | 0.923 |
| 50 | B | 29.0 | 26 / 30 / 37 | 119.0 | 2.0 | 889.0 | 0.984 | 0.118 | 0.211 | 0.295 | 0.015 | 0.071 | 0.065 | 0.058 | 0.000 | 0.500 |
| 50 | C_0.3 | 29.5 | 28 / 30 / 35 | 120.0 | 3.0 | 888.0 | 0.976 | 0.119 | 0.212 | 0.335 | 0.015 | 0.044 | 0.065 | 0.046 | 0.000 | 1.000 |
| 50 | C_0.5 | 29.0 | 30 / 30 / 35 | 119.0 | 2.0 | 889.0 | 0.984 | 0.118 | 0.211 | 0.216 | 0.043 | 0.109 | 0.076 | 0.085 | 0.000 | 0.500 |
| 100 | A | 100.1 | 91 / 97 / 105 | 308.0 | 109.0 | 700.0 | 0.739 | 0.306 | 0.432 | 0.764 | 0.123 | 0.144 | 0.174 | 0.151 | 0.113 | 0.973 |
| 100 | B | 64.1 | 62 / 68 / 69 | 250.0 | 17.0 | 758.0 | 0.936 | 0.248 | 0.392 | 0.634 | 0.072 | 0.132 | 0.123 | 0.118 | 0.083 | 0.882 |
| 100 | C_0.3 | 88.8 | 79 / 81 / 97 | 265.0 | 105.0 | 743.0 | 0.716 | 0.263 | 0.385 | 0.760 | 0.051 | 0.123 | 0.083 | 0.096 | 0.113 | 0.991 |
| 100 | C_0.5 | 63.6 | 64 / 68 / 71 | 250.0 | 15.0 | 758.0 | 0.943 | 0.248 | 0.393 | 0.531 | 0.130 | 0.144 | 0.174 | 0.152 | 0.057 | 0.800 |
| 200 | A | 199.9 | 163 / 197 / 231 | 449.0 | 384.0 | 559.0 | 0.539 | 0.445 | 0.488 | 0.933 | 0.391 | 0.212 | 0.312 | 0.281 | 0.222 | 0.758 |
| 200 | B | 194.2 | 158 / 192 / 225 | 435.0 | 374.0 | 573.0 | 0.538 | 0.431 | 0.479 | 0.925 | 0.341 | 0.209 | 0.297 | 0.265 | 0.211 | 0.759 |
| 200 | C_0.3 | 196.1 | 152 / 163 / 201 | 374.0 | 443.0 | 634.0 | 0.458 | 0.371 | 0.410 | 0.965 | 0.159 | 0.150 | 0.203 | 0.171 | 0.222 | 0.987 |
| 200 | C_0.5 | 189.1 | 148 / 190 / 225 | 432.0 | 356.0 | 576.0 | 0.548 | 0.429 | 0.481 | 0.913 | 0.341 | 0.209 | 0.297 | 0.265 | 0.211 | 0.747 |

Calibration (thresholds and B's share of calibration alerts in 06-23):

| Budget | B threshold | B share 06-23 | C_0.3 thresholds (00-05 / 06-23) | C_0.5 thresholds (00-05 / 06-23) |
|---:|---:|---:|---|---|
| 50 | 0.9789 | 0.38 | 0.9764 / 0.9806 | 0.9812 / 0.9704 |
| 100 | 0.9585 | 0.42 | 0.9387 / 0.9665 | 0.9694 / 0.9358 |
| 200 | 0.8281 | 0.49 | 0.6451 / 0.9204 | 0.8565 / 0.8217 |

## Selection

Rule (declared before F3): At 100 alerts/day, policy C_s is adopted over B only if, in BOTH F1 and F2: (a) total recall of C_s >= B, (b) recall on hours 06-23 of C_s > B, and (c) realised alerts of C_s <= 1.10 x B's (same review load). If several shares pass, the one with the higher mean total recall over F1 and F2 is chosen; if none passes, B is chosen.

| Policy | Fold | Recall vs B | Recall not lower | 06-23 recall higher | Alerts within tolerance |
|---|---|---|---|---|---|
| C_0.3 | F1 | 0.364 vs 0.356 | True | False | False |
| C_0.3 | F2 | 0.263 vs 0.248 | True | False | False |
| C_0.5 | F1 | 0.364 vs 0.356 | True | False | True |
| C_0.5 | F2 | 0.248 vs 0.248 | True | True | True |

**Chosen policy:** B.

## Notes

- B and C counts are deterministic (every row at or above its threshold is alerted). A's counts are expected values under random tie-breaking, as in `evaluate.py`.
- Daily alert counts use whole simulated days only (24 steps inside the test window); partial days at the window edges are listed in the JSON report.
- No confidence intervals: this stage compares operating policies, not models.
- Scores are risk scores, not calibrated probabilities.
