# Archived reports — superseded, historical only

These reports were moved here on 2026-11-18 during the P0 evaluation-integrity work.
They are kept, with history, for traceability. **Do not cite figures from them.**
Most were hand-written or partly hand-edited, and several contradict the code or each other.

Current, generated sources of truth:

| Topic | Current source | Generator |
|---|---|---|
| Walk-forward model evaluation | `reports/evaluation/walk_forward_<config>.{json,md}` | `src/finsight/evaluate.py` |
| Benchmark rows / SQL view | `reports/evaluation/benchmark_rows.json`, `sql/views/04_vw_risk_detection_benchmark.sql` | `src/finsight/benchmark_view.py` |
| Model provenance | `models/candidates/<config>.manifest.json` | `src/finsight/train.py` |
| SHAP explanations | `reports/explainability/` | `src/finsight/explainability.py` |

## Contents

| Folder | Origin |
|---|---|
| `ml_improvement/` | ML improvement sprint (`ml_experiments.execute_full_ml_sprint`), feature/imbalance experiments EXP-01–EXP-14 |
| `remediation/` | Remediation and verification reports, frozen-model evaluation, metric reconciliation |
| `model_evaluation/` | Milestone 4 baseline report (10-feature models that use `transaction_day`) |

## Known errors in these reports

1. **Alert budgets about 4× too low.** `ml_improvement/threshold_and_alert_budget_analysis.md` says "~50 reviews/day" at t = 0.966.
   The holdout (steps 551–743) covers 193 hourly steps = 8.04 simulated days, so 1,590 alerts ≈ **198 alerts/day**.
   Likewise, t = 0.990 (962 alerts) ≈ 120/day, not "~30". `remediation/calibration_and_thresholds.md` describes a
   "4-day validation window"; the validation window (steps 451–550) is 4.17 days.
2. **Stale rule-engine MEDIUM+HIGH figures.** Reports and the old hand-typed benchmark view state 48,930 alerts,
   3.51% precision, 79.93% recall on the holdout. The current rule engine gives 50,357 alerts (49,799 MEDIUM + 558 HIGH),
   3.12% precision, 73.18% recall. The HIGH-tier figures (558 / 72.04% / 18.72%) are correct.
3. **Contradictory EXP-07 narrative.** `ml_improvement/feature_engineering_log.md` says EXP-07/08 reached AP 0.289–0.301
   and underperformed. `feature_experiments.json` (and a fresh retrain) show validation AP 0.3663 / 0.3413, an improvement.
4. **Holdout used for selection.** The EXP-07 "MODEL_IMPROVED" gate in `champion_selection_rationale.md` was decided by
   comparing holdout AP. The model-family benchmark also ranked models on the holdout. Holdout figures here are therefore
   not unbiased test results. The gate is now validation-only, and promotion requires walk-forward evaluation.
5. **"Probabilities" are risk scores.** Class-weighted training inflates scores (mean score ≈ 0.18 vs fraud prevalence ≈ 0.011);
   they are not calibrated probabilities.
6. **Missed-fraud segment shares** quoted elsewhere (41.2% below 100k, >65% between 09:00–18:00) do not reproduce for the frozen
   champion (27.0% and 58.9% on the old holdout).
7. **Training/inference numeric path.** The frozen champion was trained on float32-loaded amounts (via `models.py`), while these
   reports scored it with float64 features. Headline metrics agree to 4 decimal places, but `train.py` now reproduces the
   original training path exactly.
