# P1.4 alert policy — confirmation (F3)

Config `hgb_baseline7_rcpt_cw50` (E4) · generated 2026-11-18T19:27:22Z · data sha256 `74cba5b396ce…` · locked test (≥651) not read.

## Disclosure

Fold models here are trained on steps <= T-100 (T = the fold's training end), so they differ from the P1.3 fold models and their numbers are not directly comparable. Thresholds are set only on the calibration window (T-99..T), which lies before the test window. F1 and F2 choose the policy; F3 was already used for P1.3 confirmation, so this is its second use and it is not independent. The locked test (651-743) is not read.

Policies: **A** top-K over the whole test period (retrospective reference) · **B** one fixed threshold set on the calibration window · **C_s** fixed thresholds per hour band with share s of the budget reserved for 06-23.

Chosen policy carried over unchanged from the selection report: **B**. The other policies are shown for reference only: they were not chosen on F1/F2, and results on F3 cannot change the choice.

## Fold F3

Model trained on steps 1–450 (6,014,717 rows, 5,057 fraud); thresholds set on 451–550 (151,302 rows); test 551–650 (115,770 rows, 1,126 fraud, 4.17 simulated days).

| Budget | Policy | Alerts/day | Full-day alerts min / median / max | TP | FP | FN | Precision | Recall | F1 | R 00-05 | R 06-08 | R 09-17 | R 18-23 | R 06-23 | R <100k | FP share 00-05 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | A | 49.9 | 44 / 50 / 55 | 204.2 | 3.8 | 921.8 | 0.982 | 0.181 | 0.306 | 0.413 | 0.087 | 0.091 | 0.099 | 0.093 | 0.022 | 0.739 |
| 50 | B | 59.5 | 55 / 61 / 62 | 238.0 | 10.0 | 888.0 | 0.960 | 0.211 | 0.346 | 0.510 | 0.087 | 0.101 | 0.099 | 0.098 | 0.054 | 0.900 |
| 50 | C_0.3 | 52.3 | 48 / 53 / 55 | 208.0 | 10.0 | 918.0 | 0.954 | 0.185 | 0.309 | 0.510 | 0.040 | 0.056 | 0.078 | 0.061 | 0.054 | 0.900 |
| 50 | C_0.5 | 59.0 | 54 / 58 / 66 | 242.0 | 4.0 | 884.0 | 0.984 | 0.215 | 0.353 | 0.416 | 0.127 | 0.118 | 0.174 | 0.139 | 0.025 | 0.750 |
| 100 | A | 100.1 | 81 / 90 / 138 | 358.0 | 59.0 | 768.0 | 0.859 | 0.318 | 0.464 | 0.710 | 0.135 | 0.132 | 0.238 | 0.169 | 0.100 | 0.949 |
| 100 | B | 97.9 | 81 / 90 / 131 | 353.0 | 55.0 | 773.0 | 0.865 | 0.314 | 0.460 | 0.694 | 0.135 | 0.132 | 0.238 | 0.169 | 0.096 | 0.946 |
| 100 | C_0.3 | 98.4 | 81 / 88 / 136 | 351.0 | 59.0 | 775.0 | 0.856 | 0.312 | 0.457 | 0.707 | 0.135 | 0.128 | 0.223 | 0.162 | 0.100 | 0.949 |
| 100 | C_0.5 | 101.0 | 87 / 100 / 116 | 392.0 | 29.0 | 734.0 | 0.931 | 0.348 | 0.507 | 0.661 | 0.278 | 0.162 | 0.305 | 0.229 | 0.092 | 0.724 |
| 200 | A | 199.9 | 157 / 168 / 279 | 586.0 | 247.0 | 540.0 | 0.704 | 0.520 | 0.598 | 0.929 | 0.627 | 0.213 | 0.468 | 0.365 | 0.322 | 0.725 |
| 200 | B | 161.3 | 123 / 140 / 237 | 501.0 | 171.0 | 625.0 | 0.746 | 0.445 | 0.557 | 0.884 | 0.429 | 0.174 | 0.362 | 0.278 | 0.226 | 0.895 |
| 200 | C_0.3 | 158.6 | 114 / 138 / 239 | 480.0 | 181.0 | 646.0 | 0.726 | 0.426 | 0.537 | 0.906 | 0.309 | 0.164 | 0.330 | 0.244 | 0.230 | 0.950 |
| 200 | C_0.5 | 155.3 | 125 / 131 / 230 | 503.0 | 144.0 | 623.0 | 0.777 | 0.447 | 0.567 | 0.803 | 0.532 | 0.181 | 0.401 | 0.311 | 0.201 | 0.792 |

Calibration (thresholds and B's share of calibration alerts in 06-23):

| Budget | B threshold | B share 06-23 | C_0.3 thresholds (00-05 / 06-23) | C_0.5 thresholds (00-05 / 06-23) |
|---:|---:|---:|---|---|
| 50 | 0.9738 | 0.34 | 0.9738 / 0.9834 | 0.9750 / 0.9637 |
| 100 | 0.9457 | 0.31 | 0.9449 / 0.9510 | 0.9530 / 0.9010 |
| 200 | 0.8406 | 0.36 | 0.7823 / 0.8890 | 0.9260 / 0.7932 |

## Notes

- B and C counts are deterministic (every row at or above its threshold is alerted). A's counts are expected values under random tie-breaking, as in `evaluate.py`.
- Daily alert counts use whole simulated days only (24 steps inside the test window); partial days at the window edges are listed in the JSON report.
- No confidence intervals: this stage compares operating policies, not models.
- Scores are risk scores, not calibrated probabilities.
