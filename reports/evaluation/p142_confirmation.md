# P1.4.2 review queue — confirmation (F3)

Config `hgb_baseline7_rcpt_cw50` (E4) · generated 2026-11-18T20:23:58Z · data sha256 `74cba5b396ce…` · locked test (≥651) not read.

## Disclosure

No fold is independent. F1 and F2 were used for feature discovery (P1.1), candidate screening (P1.3), policy selection (P1.4) and the P1.4.1 investigation; F3 was used for P1.3 and P1.4 confirmation and in P1.4.1, so this is its fourth use. The queue design itself was proposed after seeing P1.4.1 results on F1-F3. Models are the P1.4 fold models (E4 trained on steps <= T-100); floor and escalation thresholds come from the calibration window (T-99..T). The locked test (651-743) is not read. PaySim has no settlement timing, so the cost of a review delay cannot be measured: a fraud reviewed 20 hours late counts as detected here.

Policies: **B** fixed threshold (P1.4 reference, uncapped) · **D1** each day's top-K (needs the whole day; upper bound) · **D2** B's threshold + first-come daily cap (P1.4.1) · **Qd** priority queue, K/24 reviews per hour, deadline d hours, floor = calibration threshold for 4×K/day · **+esc** adds automatic holds above the calibration threshold for 10 holds/day (no analyst capacity used).

Choice carried over unchanged from the selection report: queue **Q12**, reference policy **B**. Other policies are shown for reference only; F3 cannot change the choice.

## Fold F3

Model trained on 1–450; thresholds from 451–550; test 551–650 (115,770 rows, 1,126 fraud).

| K | Policy | Reviews/day | Whole-day min / median / max | CV | Precision | Recall | F1 | R 00-05 | R 06-23 | R 09-17 | R <100k | FP share 00-05 |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 50 | B | 59.5 | 55 / 61 / 62 | 0.046 | 0.960 | 0.211 | 0.346 | 0.510 | 0.098 | 0.101 | 0.054 | 0.900 |
| 50 | D1 | 49.9 | 50 / 50 / 50 | 0.000 | 0.976 | 0.180 | 0.304 | 0.400 | 0.097 | 0.093 | 0.006 | 0.800 |
| 50 | D2 | 49.5 | 50 / 50 / 50 | 0.000 | 0.952 | 0.174 | 0.294 | 0.500 | 0.050 | 0.061 | 0.054 | 0.900 |
| 50 | Q6 | 49.7 | 49 / 50 / 50 | 0.009 | 0.957 | 0.176 | 0.297 | 0.242 | 0.151 | 0.147 | 0.000 | 0.111 |
| 50 | Q6+esc | 48.7 | 46 / 50 / 50 | 0.034 | 0.946 | 0.171 | 0.289 | 0.261 | 0.136 | 0.128 | 0.004 | 0.091 |
| 50 | Q12 | 50.2 | 49 / 50 / 51 | 0.017 | 0.976 | 0.181 | 0.306 | 0.323 | 0.128 | 0.122 | 0.000 | 0.200 |
| 50 | Q12+esc | 49.9 | 50 / 50 / 50 | 0.000 | 0.976 | 0.180 | 0.304 | 0.358 | 0.113 | 0.101 | 0.017 | 0.400 |
| 50 | Q24 | 49.0 | 49 / 50 / 51 | 0.017 | 0.976 | 0.177 | 0.299 | 0.394 | 0.094 | 0.093 | 0.004 | 0.800 |
| 50 | Q24+esc | 49.0 | 49 / 50 / 51 | 0.017 | 0.960 | 0.174 | 0.294 | 0.464 | 0.064 | 0.069 | 0.048 | 1.000 |
| 100 | B | 97.9 | 81 / 90 / 131 | 0.205 | 0.865 | 0.314 | 0.460 | 0.694 | 0.169 | 0.132 | 0.096 | 0.946 |
| 100 | D1 | 99.8 | 100 / 100 / 100 | 0.000 | 0.909 | 0.336 | 0.490 | 0.732 | 0.185 | 0.137 | 0.121 | 0.895 |
| 100 | D2 | 89.8 | 81 / 90 / 100 | 0.091 | 0.860 | 0.286 | 0.429 | 0.684 | 0.135 | 0.105 | 0.096 | 0.981 |
| 100 | Q6 | 100.1 | 96 / 101 / 104 | 0.029 | 0.883 | 0.327 | 0.477 | 0.488 | 0.266 | 0.241 | 0.095 | 0.161 |
| 100 | Q6+esc | 100.1 | 97 / 101 / 103 | 0.023 | 0.866 | 0.321 | 0.468 | 0.497 | 0.254 | 0.228 | 0.126 | 0.196 |
| 100 | Q12 | 100.1 | 96 / 101 / 104 | 0.029 | 0.911 | 0.338 | 0.492 | 0.629 | 0.227 | 0.181 | 0.071 | 0.568 |
| 100 | Q12+esc | 100.1 | 97 / 101 / 103 | 0.023 | 0.895 | 0.332 | 0.484 | 0.639 | 0.215 | 0.160 | 0.096 | 0.642 |
| 100 | Q24 | 98.6 | 92 / 100 / 108 | 0.059 | 0.894 | 0.326 | 0.478 | 0.724 | 0.175 | 0.135 | 0.121 | 0.885 |
| 100 | Q24+esc | 98.4 | 92 / 100 / 108 | 0.058 | 0.861 | 0.314 | 0.460 | 0.723 | 0.158 | 0.115 | 0.130 | 0.930 |
| 200 | B | 161.3 | 123 / 140 / 237 | 0.282 | 0.746 | 0.445 | 0.557 | 0.884 | 0.278 | 0.174 | 0.226 | 0.895 |
| 200 | D1 | 200.2 | 200 / 200 / 200 | 0.000 | 0.722 | 0.535 | 0.614 | 0.913 | 0.391 | 0.292 | 0.435 | 0.634 |
| 200 | D2 | 151.4 | 123 / 140 / 200 | 0.195 | 0.740 | 0.414 | 0.531 | 0.870 | 0.241 | 0.147 | 0.221 | 0.933 |
| 200 | Q6 | 199.9 | 197 / 198 / 203 | 0.012 | 0.711 | 0.526 | 0.604 | 0.793 | 0.424 | 0.348 | 0.397 | 0.237 |
| 200 | Q6+esc | 199.9 | 197 / 200 / 201 | 0.007 | 0.683 | 0.505 | 0.581 | 0.784 | 0.400 | 0.324 | 0.424 | 0.246 |
| 200 | Q12 | 199.9 | 196 / 198 / 204 | 0.016 | 0.723 | 0.535 | 0.615 | 0.843 | 0.418 | 0.328 | 0.402 | 0.484 |
| 200 | Q12+esc | 199.9 | 197 / 199 / 202 | 0.010 | 0.699 | 0.517 | 0.594 | 0.837 | 0.395 | 0.313 | 0.427 | 0.465 |
| 200 | Q24 | 195.1 | 168 / 184 / 231 | 0.126 | 0.725 | 0.523 | 0.608 | 0.900 | 0.380 | 0.255 | 0.364 | 0.670 |
| 200 | Q24+esc | 195.4 | 166 / 182 / 235 | 0.141 | 0.694 | 0.502 | 0.583 | 0.881 | 0.358 | 0.238 | 0.381 | 0.606 |

Queue diagnostics (fraud counts are expected values):

| K | Policy | Fraud expired | Fraud pending at end | Fraud below floor | Median / p90 review delay of fraud (h) | Fraud reviewed within the hour | Holds/day | False holds/day | Fraud held | Recall incl. holds |
|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 50 | Q6 | 269.0 | 34.0 | 625 | 0 / 4 | 0.57 | 0.0 | 0.00 | 0 | – |
| 50 | Q6+esc | 244.0 | 31.0 | 625 | 0 / 4 | 0.56 | 8.4 | 0.24 | 34 | 0.201 |
| 50 | Q12 | 260.0 | 37.0 | 625 | 1 / 9 | 0.45 | 0.0 | 0.00 | 0 | – |
| 50 | Q12+esc | 231.0 | 33.0 | 625 | 1 / 9 | 0.43 | 8.4 | 0.24 | 34 | 0.210 |
| 50 | Q24 | 234.0 | 68.0 | 625 | 2 / 14 | 0.42 | 0.0 | 0.00 | 0 | – |
| 50 | Q24+esc | 213.2 | 58.0 | 625 | 2 / 17 | 0.35 | 8.4 | 0.24 | 34 | 0.204 |
| 100 | Q6 | 286.6 | 39.0 | 432 | 0 / 4 | 0.61 | 0.0 | 0.00 | 0 | – |
| 100 | Q6+esc | 263.0 | 36.0 | 432 | 0 / 4 | 0.60 | 8.4 | 0.24 | 34 | 0.351 |
| 100 | Q12 | 263.0 | 51.0 | 432 | 0 / 8 | 0.53 | 0.0 | 0.00 | 0 | – |
| 100 | Q12+esc | 237.6 | 49.0 | 432 | 1 / 8 | 0.50 | 8.4 | 0.24 | 34 | 0.362 |
| 100 | Q24 | 245.6 | 81.0 | 432 | 1 / 12 | 0.50 | 0.0 | 0.00 | 0 | – |
| 100 | Q24+esc | 232.0 | 75.0 | 432 | 1 / 12 | 0.47 | 8.4 | 0.24 | 34 | 0.344 |
| 200 | Q6 | 297.1 | 24.0 | 213 | 0 / 3 | 0.69 | 0.0 | 0.00 | 0 | – |
| 200 | Q6+esc | 288.1 | 22.0 | 213 | 0 / 3 | 0.70 | 8.4 | 0.24 | 34 | 0.535 |
| 200 | Q12 | 265.8 | 45.0 | 213 | 0 / 6 | 0.64 | 0.0 | 0.00 | 0 | – |
| 200 | Q12+esc | 257.0 | 40.0 | 213 | 0 / 5 | 0.64 | 8.4 | 0.24 | 34 | 0.547 |
| 200 | Q24 | 252.0 | 72.0 | 213 | 0 / 11 | 0.61 | 0.0 | 0.00 | 0 | – |
| 200 | Q24+esc | 248.0 | 66.0 | 213 | 0 / 10 | 0.61 | 8.4 | 0.24 | 34 | 0.532 |

## Notes

- Review counts and recall are expected values where capacity is split among items with identical score and arrival hour.
- CV is the standard deviation of whole-day review counts divided by their mean (whole days inside the test window only).
- Escalation results are shown for the business decision only; they are not part of the selection rule.
- No confidence intervals: this compares operating policies, not models.
