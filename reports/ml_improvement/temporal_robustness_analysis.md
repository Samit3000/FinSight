# FinSight — Phase 4: Temporal Robustness & Chronological Drift Report

**Audit Date:** 2026-10-19 11:34:15 UTC  
**Evaluated Champion:** `HistGradientBoosting No Day`  
**Dataset:** Full 31-Day Simulation Log (6,362,620 transactions)  

---

## 1. Executive Summary & Forensic Findings

1. **Stationary Attack Generation vs Non-Stationary Legitimate Volume:**
   - PaySim generates an unvarying background fraud frequency of approximately **250 to 300 fraudulent account liquidations every 24 hours**.
   - Legitimate transaction volume drops precipitously from **~500,000 transactions/day** during Days 1–2 down to **~20,000 transactions/day** during Days 23–31.
   - Consequently, the mathematical fraud rate rises from **0.05%** up to **1.3%** strictly because the denominator vanishes.
2. **`transaction_day` Ablation Justification:**
   - If included, tree models split heavily on late calendar days (`day > 23`) to exploit this artificial prior inflation.
   - The frozen 8-feature champion strictly omits `transaction_day`, forcing predictions to rely on stationary behavioral attributes (`amount`, `log1p_amount`, `transaction_hour`, and rail types).

---

## 2. Chronological Segment Evaluation Matrix

| Segment Name | Step Range | Calendar Days | Population (N) | Fraud Count | Fraud Rate (%) | Model PR-AUC (AP) | Model ROC-AUC | F1 at Locked Threshold (t=0.966) |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| **Train Early (Days 1-6)** | — | — | 1,509,597 | 1,632 | 0.1081% | **0.1957** | **0.9537** | **0.1874** |
| **Train Mid (Days 7-12)** | — | — | 2,448,719 | 1,647 | 0.0673% | **0.3095** | **0.9563** | **0.3105** |
| **Train Late (Days 13-18)** | — | — | 2,056,401 | 1,778 | 0.0865% | **0.3932** | **0.9596** | **0.4240** |
| **Validation (Days 19-23)** | — | — | 151,302 | 1,008 | 0.6662% | **0.3115** | **0.9509** | **0.3773** |
| **Early Holdout (Days 24-27)** | — | — | 115,770 | 1,126 | 0.9726% | **0.4133** | **0.9517** | **0.4763** |
| **Late Holdout (Days 28-31)** | — | — | 80,831 | 1,022 | 1.2644% | **0.4909** | **0.9447** | **0.5092** |

---

## 3. Subperiod Generalization on Holdout

Evaluating performance across early holdout (Days 24–27) vs late holdout (Days 28–31):
- **Early Holdout PR-AUC (Days 24–27):** **`0.4133`**
- **Late Holdout PR-AUC (Days 28–31):** **`0.4909`**
- **Temporal Drift Assessment:** PR-AUC remains robust and increases in late holdout as fraud accounts continue their liquidation patterns while background consumer traffic stabilizes at low steady-state volume.

---

## 4. Visual Artifacts
- **Daily Volume vs Fraud Rate Dynamics:** Saved to [`reports/ml_improvement/figures/temporal_daily_dynamics.png`](figures/temporal_daily_dynamics.png)
- **Segment PR Curves:** Saved to [`reports/ml_improvement/figures/temporal_segment_pr_curves.png`](figures/temporal_segment_pr_curves.png)
