# P1.6 — Champion promotion record

**Promoted:** 2026-11-20 · **Decision source:** pre-registered P1.5 sealed test (`p15_final_test.md`)

| | Active champion (E4) | Previous champion |
|---|---|---|
| Artifact | `models/hgb_e4_recipient.joblib` | `models/hist_gradient_boosting_no_day.joblib` |
| SHA-256 | `296132a5f5433b5cc7141b9698bb7e95729011a7673a6da36ce779665a5f3821` | `56a40585aabf5b548faf3a6f62d685721e1c62ea7880b1ee0847f209fb99c534` |
| Config | `hgb_baseline7_rcpt_cw50` (rebuild: `hgb_e4_final`) | `hgb_baseline8_balanced` |
| Training / calibration steps | 1–550 / 551–650 | 1–450 / 451–550 |
| Operating threshold | 0.9463 (≈ 100 alerts/day on 551–650) | 0.966 (validation F1-optimal) |
| Sealed test AP, steps 651–743 | 0.6966 (95% CI 0.638–0.765) | 0.4909 (95% CI 0.420–0.600) |

## What was done

1. **Copied, not overwritten.** The exact model scored in P1.5 (`models/candidates/p15_e4_final.joblib`) was copied to
   `models/hgb_e4_recipient.joblib`; its SHA-256 matches the value P1.5 recorded. No existing model file was changed or removed.
2. **Registry.** `models/champion.json` names the active and previous champion, their hashes, configs, windows,
   thresholds at 50/100/200 alerts per day and the P1.5 evidence. `finsight.champion` loads only hash-matching artifacts.
3. **Reproducible.** `py src/finsight/train.py --config hgb_e4_final --verify-against models/hgb_e4_recipient.joblib`
   retrains the champion and reproduces it exactly: maximum score difference 0.0 on all 6,281,789 development rows, same
   threshold, same 3,140 alerts (`models/candidates/hgb_e4_final.manifest.json`).
4. **Benchmark view regenerated.** Development partitions keep the previous champion, legacy logistic regression, rule
   engine and dummy rows (values unchanged). A `sealed_final_test` partition is copied from the recorded P1.5 result; the
   sealed labels were not read again.
5. **Explanations regenerated** for the active champion on 2,000 transactions from steps 551–650 (seed 42); stale text
   about earlier models was removed. Steps 551–650 are the champion's calibration window, so these explain behaviour and
   are not a performance estimate.

## Not done (needs separate approval)

- Applying the regenerated benchmark view to the live PostgreSQL database.
- Retraining on steps 1–650 (that would be a different, untested model).
- Making the 12-hour review queue the official alert policy.

## Caveats

The sealed steps 651–743 have now been used once and must not be used again for selection. They sat inside the pre-P0
holdout, so the result is strong but not pristine. Development folds F1–F3 were reused many times.
