# P3 — Scoring store: disposable-database run

**Date:** 2026-11-20 · **Target:** a throwaway PostgreSQL 18 cluster (`localhost:55432`, database `finsight_p3_test`),
created for this test. **The live database (`localhost:5432/finsight_db`) was not written to**; a read-only check after
the run found none of the P3 objects there, `fct_transactions` at 6,362,620 rows and the benchmark view at 14 rows.

This is an engineering check of the scoring store. It contains **no performance claims**: fraud labels were never read.

## What was run

1. Full copy of the transactions loaded into the disposable database (6,362,620 rows, IDs 1–6,362,620).
2. `sql/schema/02_scoring_schema.sql` applied **twice** (second run skipped every object: idempotent), then
   `sql/views/05_vw_alert_queue.sql`.
3. `score_transactions.py --mode bulk` for steps 1–700 (one run), then `--mode hourly` for steps 701–743 (43 runs, one
   per hour).

## Threshold and alert counts

The operating threshold 0.9462684948346494 is the 417th-highest champion score on calibration steps 551–650:
k = round(100 alerts/day × 100 hours / 24) = 417, with one row at the threshold.

| Period | Steps | Rows | Alerts | Alerts per simulated day | Note |
|---|---|---:|---:|---:|---|
| training | 1–550 | 6,166,019 | 2,723 | 118.8 | The model learned from these hours |
| calibration | 551–650 | 115,770 | 417 | 100.1 | Equal to k by construction |
| sealed_test_reported | 651–743 | 80,831 | 350 | 90.3 | Same count as recorded in P1.5 |
| **Total** | 1–743 | **6,362,620** | **3,490** | | |

(The earlier "3,140 alerts" figure is steps 1–650: 2,723 + 417.)

## Verification (independent script, read-only)

| Check | Result |
|---|---|
| Every transaction scored exactly once; IDs equal row numbers 1…6,362,620 | Yes; 0 transactions without a score |
| Mapping to `fct_transactions` (step, type, amount, sender, recipient) | Verified by the scorer for all 6,362,620 rows before writing |
| Stored scores vs `champion.score_frame` on all rows | Max difference 0.0 |
| Hourly (701–743) vs bulk recomputation | 11,421 rows; max score difference 0.0; recipient counts identical |
| Reason codes | 3,490 = alerts; none missing, none on non-alerts; every record has exactly 3 contributors; max additivity error 7.8e-14 |
| Run log | 1 bulk run (6,351,199 rows, 3,337 alerts) + 43 hourly runs (11,421 rows, 153 alerts) |
| `vw_alert_queue` | 3,490 rows; no fraud-label column; every row carries the model disclaimer and rule note |
| Database checks | Period/step consistency and `is_alert = (score >= threshold)` enforced by CHECK constraints (tested) |

## Example rows from `vw_alert_queue` (calibration hours)

| transaction_id | Step | Type | Amount | Score | Top model contributors | Rule conditions met |
|---:|---:|---|---:|---:|---|---|
| 6,166,271 | 551 | TRANSFER | $3,774,778 | 0.9678 | first-time recipient +4.31; amount +2.09; not CASH_IN +1.20 | RULE_001, RULE_002 (MEDIUM) |
| 6,166,272 | 551 | CASH_OUT | $3,774,778 | 0.9885 | amount +3.85; type CASH_OUT +3.82; not PAYMENT +1.36 | RULE_001, RULE_002 (MEDIUM) |
| 6,168,486 | 553 | TRANSFER | $1,688,761 | 0.9748 | first-time recipient +3.10; hour 00:00 +2.09; amount +1.73 | RULE_001, RULE_002 (MEDIUM) |

Model contributions are model attribution (log-odds), not evidence of fraud; rule conditions are shown for context and
do not raise alerts.

## Runtime

Disposable load of the transactions: 707 s. Bulk scoring and writing of steps 1–700 plus 43 hourly runs: about 12 minutes
in total, dominated by the full mapping check and the database writes.
