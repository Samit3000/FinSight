# P2 — Rule-engine backstop study

**Question:** do rules catch fraud that E4 misses, better than simply giving E4 the same extra review capacity?

**Answer:** No. No configuration met the pre-declared rule; giving E4 the same extra review slots catches more fraud. Rules are kept for reason codes only. F3 confirmation: nothing was selected, so F3 (used once) reports the baseline rules only; they also lose to E4 with the same extra slots.

## Protocol

Selection rule (fixed before F3): Among configurations that, in BOTH F1 and F2: (a) add at most 30 extra alerts beyond E4's 100/day on EVERY whole simulated day (strict daily cap, not an average), (b) catch more frauds that E4 missed than E4 itself catches when given the same total number of extra review slots (its next-ranked transactions), and (c) catch at least 1.0 extra fraud per day on average; choose the one with the most extra frauds over F1+F2. Ties: higher extra precision, then fewer active rules. If none qualifies, recommend no backstop alerts (rules give reason codes only).

Confirmation rule: The selected configuration is confirmed only if conditions (a) and (b) also hold on F3.

F1 and F2 select; F3 is used once here but has been used many times before (P1.1 discovery, P1.3 and P1.4 confirmations, P1.4.1, P1.4.2, and the champion's threshold calibration on steps 551-650), so it is not independent evidence. The retired sealed test (steps 651-743) is not read.

E4 alerts come from fold copies of the E4 configuration at 100 alerts per simulated day; the champion is untouched.

## Key configurations

| Configuration | Fold | Extra alerts/day (mean) | Whole-day min / median / max | Extra fraud caught | E4 with same extra slots | Combined recall | E4 + slots recall | Passes |
|---|---|---:|---|---:|---:|---:|---:|---|
| hv200k_nonew_cut70 | F1 | 55.9 | 52 / 62 / 87 | 48.0 | 104.0 | 0.369 | 0.418 | no |
| hv200k_nonew_cut70 | F2 | 60.5 | 29 / 33 / 74 | 25.0 | 80.4 | 0.335 | 0.390 | no |
| hv200k_nonew_cut70 | F3 (confirmation) | 27.4 | 12 / 18 / 65 | 25.0 | 69.8 | 0.355 | 0.395 | no |
| hv500k_nonew_cut70 | F1 | 28.1 | 17 / 20 / 62 | 27.0 | 64.0 | 0.351 | 0.383 | no |
| hv500k_nonew_cut70 | F2 | 15.4 | 7 / 11 / 21 | 8.0 | 27.0 | 0.319 | 0.337 | no |
| hv2000k_nonew_cut70 | F1 | 19.7 | 6 / 10 / 54 | 22.0 | 47.4 | 0.347 | 0.369 | no |
| hv2000k_nonew_cut70 | F2 | 4.1 | 2 / 3 / 5 | 8.0 | 4.0 | 0.319 | 0.315 | yes |
| hv500k_new1000k@25_cut70 | F1 | 735.1 | 60 / 1001 / 1319 | 142.0 | 343.3 | 0.451 | 0.625 | no |
| hv500k_new1000k@25_cut70 | F2 | 144.5 | 84 / 127 / 221 | 119.0 | 166.0 | 0.429 | 0.475 | no |
| hv200k_new2000k@25_cut70 | F1 | 308.4 | 70 / 464 / 504 | 114.0 | 251.0 | 0.426 | 0.545 | no |
| hv200k_new2000k@25_cut70 | F2 | 115.0 | 70 / 87 / 155 | 67.0 | 145.0 | 0.377 | 0.454 | no |

Across all 72 configurations: daily cap met on every whole day by 3 (F1) and 6 (F2); beat E4 with the same extra slots in 0 (F1) and 1 (F2). Qualifying in both folds: 0.

## All configurations (F1 | F2: mean extra alerts/day, max day, extra fraud vs E4 with same slots)

| Configuration | F1 | F2 |
|---|---|---|
| hv200k_nonew_cut70 | 55.9/day, max 87, 48 vs 104 | 60.5/day, max 74, 25 vs 80 |
| hv200k_nonew_cut80 | 39.1/day, max 58, 26 vs 82 | 57.8/day, max 73, 17 vs 79 |
| hv200k_new250k@25_cut70 | 4829.8/day, max 8359, 318 vs 588 | 736.3/day, max 1012, 270 vs 515 |
| hv200k_new250k@25_cut80 | 4821.1/day, max 8359, 318 vs 588 | 735.8/day, max 1012, 270 vs 515 |
| hv200k_new250k@45_cut70 | 4829.8/day, max 8359, 318 vs 588 | 736.3/day, max 1012, 270 vs 515 |
| hv200k_new250k@45_cut80 | 4821.1/day, max 8359, 318 vs 588 | 735.8/day, max 1012, 270 vs 515 |
| hv200k_new500k@25_cut70 | 1601.0/day, max 2802, 225 vs 461 | 303.4/day, max 373, 213 vs 318 |
| hv200k_new500k@25_cut80 | 1592.4/day, max 2802, 225 vs 459 | 302.9/day, max 373, 213 vs 317 |
| hv200k_new500k@45_cut70 | 1601.0/day, max 2802, 225 vs 461 | 303.4/day, max 373, 213 vs 318 |
| hv200k_new500k@45_cut80 | 1592.4/day, max 2802, 225 vs 459 | 302.9/day, max 373, 213 vs 317 |
| hv200k_new1000k@25_cut70 | 763.0/day, max 1361, 163 vs 350 | 189.6/day, max 247, 136 vs 238 |
| hv200k_new1000k@25_cut80 | 754.3/day, max 1361, 163 vs 345 | 189.1/day, max 247, 136 vs 237 |
| hv200k_new1000k@45_cut70 | 763.0/day, max 1361, 163 vs 350 | 189.6/day, max 247, 136 vs 238 |
| hv200k_new1000k@45_cut80 | 754.3/day, max 1361, 163 vs 345 | 189.1/day, max 247, 136 vs 237 |
| hv200k_new2000k@25_cut70 | 308.4/day, max 504, 114 vs 251 | 115.0/day, max 155, 67 vs 145 |
| hv200k_new2000k@25_cut80 | 299.8/day, max 468, 114 vs 249 | 114.5/day, max 155, 67 vs 145 |
| hv200k_new2000k@45_cut70 | 308.4/day, max 504, 114 vs 251 | 115.0/day, max 155, 67 vs 145 |
| hv200k_new2000k@45_cut80 | 299.8/day, max 468, 114 vs 249 | 114.5/day, max 155, 67 vs 145 |
| hv500k_nonew_cut70 | 28.1/day, max 62, 27 vs 64 | 15.4/day, max 21, 8 vs 27 |
| hv500k_nonew_cut80 | 11.3/day, max 16, 5 vs 31 | 12.7/day, max 20, 0 vs 19 |
| hv500k_new250k@25_cut70 | 1575.4/day, max 2763, 210 vs 458 | 259.4/day, max 347, 199 vs 285 |
| hv500k_new250k@25_cut80 | 1564.6/day, max 2760, 204 vs 456 | 257.8/day, max 347, 196 vs 285 |
| hv500k_new250k@45_cut70 | 4804.1/day, max 8320, 303 vs 588 | 692.4/day, max 986, 256 vs 509 |
| hv500k_new250k@45_cut80 | 1566.7/day, max 2763, 210 vs 457 | 259.0/day, max 347, 199 vs 285 |
| hv500k_new500k@25_cut70 | 1573.2/day, max 2760, 204 vs 458 | 258.2/day, max 347, 196 vs 285 |
| hv500k_new500k@25_cut80 | 1564.6/day, max 2760, 204 vs 456 | 257.8/day, max 347, 196 vs 285 |
| hv500k_new500k@45_cut70 | 1573.2/day, max 2760, 204 vs 458 | 258.2/day, max 347, 196 vs 285 |
| hv500k_new500k@45_cut80 | 1564.6/day, max 2760, 204 vs 456 | 257.8/day, max 347, 196 vs 285 |
| hv500k_new1000k@25_cut70 | 735.1/day, max 1319, 142 vs 343 | 144.5/day, max 221, 119 vs 166 |
| hv500k_new1000k@25_cut80 | 726.5/day, max 1319, 142 vs 341 | 144.0/day, max 221, 119 vs 165 |
| hv500k_new1000k@45_cut70 | 735.1/day, max 1319, 142 vs 343 | 144.5/day, max 221, 119 vs 166 |
| hv500k_new1000k@45_cut80 | 726.5/day, max 1319, 142 vs 341 | 144.0/day, max 221, 119 vs 165 |
| hv500k_new2000k@25_cut70 | 280.6/day, max 479, 93 vs 237 | 69.8/day, max 129, 50 vs 84 |
| hv500k_new2000k@25_cut80 | 271.9/day, max 443, 93 vs 234 | 69.4/day, max 129, 50 vs 84 |
| hv500k_new2000k@45_cut70 | 280.6/day, max 479, 93 vs 237 | 69.8/day, max 129, 50 vs 84 |
| hv500k_new2000k@45_cut80 | 271.9/day, max 443, 93 vs 234 | 69.4/day, max 129, 50 vs 84 |
| hv1000k_nonew_cut70 | 22.6/day, max 60, 22 vs 51 | 7.0/day, max 6, 8 vs 9 |
| hv1000k_nonew_cut80 | 5.8/day, max 11, 0 vs 14 | 4.3/day, max 4, 0 vs 4 |
| hv1000k_new250k@25_cut70 | 732.0/day, max 1312, 144 vs 342 | 137.3/day, max 220, 122 vs 164 |
| hv1000k_new250k@25_cut80 | 721.0/day, max 1309, 137 vs 340 | 135.6/day, max 220, 119 vs 163 |
| hv1000k_new250k@45_cut70 | 4798.8/day, max 8310, 299 vs 588 | 684.0/day, max 985, 256 vs 509 |
| hv1000k_new250k@45_cut80 | 723.4/day, max 1312, 144 vs 340 | 136.8/day, max 220, 122 vs 164 |
| hv1000k_new500k@25_cut70 | 729.8/day, max 1309, 138 vs 341 | 136.1/day, max 220, 119 vs 163 |
| hv1000k_new500k@25_cut80 | 721.0/day, max 1309, 137 vs 340 | 135.6/day, max 220, 119 vs 163 |
| hv1000k_new500k@45_cut70 | 1567.9/day, max 2750, 200 vs 457 | 249.8/day, max 346, 196 vs 278 |
| hv1000k_new500k@45_cut80 | 721.2/day, max 1309, 138 vs 340 | 135.6/day, max 220, 119 vs 163 |
| hv1000k_new1000k@25_cut70 | 729.6/day, max 1309, 137 vs 341 | 136.1/day, max 220, 119 vs 163 |
| hv1000k_new1000k@25_cut80 | 721.0/day, max 1309, 137 vs 340 | 135.6/day, max 220, 119 vs 163 |
| hv1000k_new1000k@45_cut70 | 729.6/day, max 1309, 137 vs 341 | 136.1/day, max 220, 119 vs 163 |
| hv1000k_new1000k@45_cut80 | 721.0/day, max 1309, 137 vs 340 | 135.6/day, max 220, 119 vs 163 |
| hv1000k_new2000k@25_cut70 | 275.0/day, max 477, 88 vs 235 | 61.4/day, max 128, 50 vs 81 |
| hv1000k_new2000k@25_cut80 | 266.4/day, max 441, 88 vs 231 | 61.0/day, max 128, 50 vs 81 |
| hv1000k_new2000k@45_cut70 | 275.0/day, max 477, 88 vs 235 | 61.4/day, max 128, 50 vs 81 |
| hv1000k_new2000k@45_cut80 | 266.4/day, max 441, 88 vs 231 | 61.0/day, max 128, 50 vs 81 |
| hv2000k_nonew_cut70 | 19.7/day, max 54, 22 vs 47 | 4.1/day, max 5, 8 vs 4 |
| hv2000k_nonew_cut80 | 2.9/day, max 5, 0 vs 7 | 1.4/day, max 1, 0 vs 1 |
| hv2000k_new250k@25_cut70 | 274.6/day, max 473, 95 vs 234 | 59.8/day, max 127, 53 vs 80 |
| hv2000k_new250k@25_cut80 | 263.5/day, max 435, 88 vs 230 | 58.1/day, max 127, 50 vs 79 |
| hv2000k_new250k@45_cut70 | 4795.9/day, max 8306, 299 vs 588 | 681.1/day, max 984, 256 vs 509 |
| hv2000k_new250k@45_cut80 | 265.9/day, max 437, 95 vs 231 | 59.3/day, max 127, 53 vs 80 |
| hv2000k_new500k@25_cut70 | 272.4/day, max 471, 89 vs 234 | 58.6/day, max 127, 50 vs 80 |
| hv2000k_new500k@25_cut80 | 263.5/day, max 435, 88 vs 230 | 58.1/day, max 127, 50 vs 79 |
| hv2000k_new500k@45_cut70 | 1565.0/day, max 2746, 200 vs 456 | 247.0/day, max 345, 196 vs 278 |
| hv2000k_new500k@45_cut80 | 263.8/day, max 435, 89 vs 230 | 58.1/day, max 127, 50 vs 79 |
| hv2000k_new1000k@25_cut70 | 272.2/day, max 471, 88 vs 234 | 58.6/day, max 127, 50 vs 80 |
| hv2000k_new1000k@25_cut80 | 263.5/day, max 435, 88 vs 230 | 58.1/day, max 127, 50 vs 79 |
| hv2000k_new1000k@45_cut70 | 726.7/day, max 1305, 137 vs 341 | 133.2/day, max 219, 119 vs 163 |
| hv2000k_new1000k@45_cut80 | 263.5/day, max 435, 88 vs 230 | 58.1/day, max 127, 50 vs 79 |
| hv2000k_new2000k@25_cut70 | 272.2/day, max 471, 88 vs 234 | 58.6/day, max 127, 50 vs 80 |
| hv2000k_new2000k@25_cut80 | 263.5/day, max 435, 88 vs 230 | 58.1/day, max 127, 50 vs 79 |
| hv2000k_new2000k@45_cut70 | 272.2/day, max 471, 88 vs 234 | 58.6/day, max 127, 50 vs 80 |
| hv2000k_new2000k@45_cut80 | 263.5/day, max 435, 88 vs 230 | 58.1/day, max 127, 50 vs 79 |

## Notes

- Counts are expected values where ties at E4's cut-off score are shared.
- Rules use no balance fields, IDs or flags; the first-time-recipient condition uses the leakage-safe recipient count.
- No confidence intervals: this compares review policies on fixed folds.
