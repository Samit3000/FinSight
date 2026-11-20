# FinSight — Temporal Distribution Shift & Shortcut Learning Analysis (Phase 2)

**Analysis Date:** 2026-10-19  
**Auditor:** Senior ML Engineer & Fraud Analytics Specialist  
**Dataset:** PaySim Synthetic Financial Logs (`PS_20174392719_1491204439457_log.csv` / `data/processed/transactions.csv.gz`)  
**Partitioning:** Train (`step <= 450`, Days 1–19), Validation (`451 <= step <= 550`, Days 19–23), Holdout (`step >= 551`, Days 23–31)  

---

## 1. Executive Summary

This study conducted an empirical forensic investigation into temporal distribution shift and shortcut learning in the FinSight fraud detection models.

### Key Discoveries:
1. **The "Prevalence Explosion" Mechanism in PaySim:**
   The reported rise in fraud prevalence from **0.0841%** in training to **0.6662%** in validation and **1.0926%** in holdout is **not** caused by an escalation in fraud attacks. Fraudulent transactions occur at a virtually constant frequency throughout the 31-day simulation ($\approx 250\text{--}300$ frauds per day). Instead, the denominator (legitimate consumer transactions) collapses by over **95%** after Day 17 (falling from $\approx 450,000$ tx/day down to $\approx 10,000\text{--}50,000$ tx/day).
2. **`transaction_day` is an Overfitted Simulator Shortcut:**
   Because legitimate volume drops while fraud remains constant, tree models trained with `transaction_day` learn high-importance splits on late calendar days (Days 19–31) to artificially boost the fraud prior. This accounted for `transaction_day` receiving 24.28% of total TreeSHAP attribution (Rank 2 globally).
3. **Controlled Feature Ablation Evidence:**
   In controlled experiments across identical partitions and seeds:
   - **Removing `transaction_day` (`model_8_no_day`) dramatically improves model performance:**
     - **Validation PR-AUC (Average Precision):** Improves from **0.2519** to **0.3115** (**+23.6% relative gain**).
     - **Holdout PR-AUC:** Improves from **0.3836** to **0.4479** (**+16.8% relative gain**).
     - **Holdout F1 Score:** Improves from **0.4767** to **0.4912**.
     - **Subperiod Generalization:** PR-AUC in early holdout (Days 23–27) rises from $0.3565 \to 0.4133$; in late holdout (Days 28–31) rises from $0.4188 \to 0.4909$.
   - **`transaction_hour` is a Genuine Behavioral Feature:**
     Removing both `transaction_day` and `transaction_hour` (`model_7_no_temporal`) causes Validation PR-AUC to decline to **0.2665** and Holdout PR-AUC to drop to **0.3222**. Off-hours activity (diurnal cycle: 1 AM–5 AM) represents a stationary behavioral signal that generalizes across all 31 days.
4. **Final Architecture Decision:**
   We recommend **permanently ablating `transaction_day`** from the primary portfolio model artifact. The resulting 8-feature model (`amount`, `log1p_amount`, `transaction_hour`, and 5 one-hot rail types) is both statistically superior and real-world defensible.

---

## 2. Temporal & Distribution Shift Diagnostics

### A. Daily Volume, Fraud Counts, and Incidence Rates

| Day Range | Average Daily Volume | Average Daily Fraud | Mean Daily Fraud Rate | Dominant Dynamics |
| :--- | :--- | :--- | :--- | :--- |
| **Days 1–2** | 514,746 tx/day | 290 frauds/day | **0.0575%** | Initial simulation high consumer volume. |
| **Days 3–5** | 13,033 tx/day | 275 frauds/day | **10.82%** (Day 3 peak: 28.97%) | Simulator artifact: sharp dip in legitimate transactions. |
| **Days 6–17** | 413,842 tx/day | 266 frauds/day | **0.0652%** | Steady-state weekday consumer payments. |
| **Days 18–22** | 26,011 tx/day | 258 frauds/day | **1.2646%** | Legitimate consumer volume permanently falls. |
| **Days 23–31** | 22,968 tx/day | 251 frauds/day | **1.2825%** | Low background volume maintains elevated fraud rate. |

**Key Takeaway:** PaySim's fraud generator runs a relentless, unthrottled background bot producing $\sim 260$ fraudulent account liquidations every 24 hours. The apparent temporal fraud surge is an artifact of varying background consumer volume, not attack intensity.

---

### B. Fraud Prevalence by Transaction Type Across Partitions

| Transaction Rail | Training (`step <= 450`) Fraud Rate | Validation (`451-550`) Fraud Rate | Holdout (`step > 550`) Fraud Rate | Structural Notes |
| :--- | :--- | :--- | :--- | :--- |
| **`CASH_OUT`** | 0.1168% ($2,538 / 2.17\text{M}$) | 0.9080% ($504 / 55.5\text{k}$) | 1.4886% ($1,071 / 71.9\text{k}$) | 50.1% of all frauds in dataset. |
| **`TRANSFER`** | 0.4907% ($2,519 / 513.3\text{k}$) | 3.8647% ($504 / 13.0\text{k}$) | 6.4777% ($1,077 / 16.6\text{k}$) | 49.9% of all frauds in dataset. |
| **`PAYMENT`** | **0.0000%** ($0 / 2.05\text{M}$) | **0.0000%** ($0 / 51.5\text{k}$) | **0.0000%** ($0 / 69.2\text{k}$) | 0 frauds across all 31 days. |
| **`CASH_IN`** | **0.0000%** ($0 / 1.33\text{M}$) | **0.0000%** ($0 / 33.1\text{k}$) | **0.0000%** ($0 / 37.9\text{k}$) | 0 frauds across all 31 days. |
| **`DEBIT`** | **0.0000%** ($0 / 39.5\text{k}$) | **0.0000%** ($0 / 965$) | **0.0000%** ($0 / 1,006$) | 0 frauds across all 31 days. |

---

### C. Amount Distribution Quantiles by Label & Partition

| Partition | Class | 25th Pct ($) | Median ($) | 75th Pct ($) | 90th Pct ($) | 99th Pct ($) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Training** | Legitimate | $13,293 | $74,809 | $207,678 | $363,222 | $1,617,144 |
| **Training** | Fraud | $124,008 | **$438,719** | $1,509,471 | $4,586,163 | $10,000,000 |
| **Validation**| Legitimate | $13,674 | $77,882 | $215,920 | $378,397 | $1,714,668 |
| **Validation**| Fraud | $126,908 | **$445,862** | $1,475,699 | $4,710,214 | $10,000,000 |
| **Holdout**   | Legitimate | $13,858 | $79,622 | $218,806 | $384,196 | $1,749,429 |
| **Holdout**   | Fraud | $132,189 | **$444,792** | $1,507,786 | $4,792,028 | $10,000,000 |

**Stability Analysis:** Amount distributions for both legitimate transactions and fraud cases remain highly stationary across all three chronological partitions. Fraud amounts consistently exhibit a median of $\approx \$440,000$ and a 99th percentile capped at the PaySim limit of $\$10,000,000$.

---

## 3. Controlled Feature Ablation Experiments

All six models were trained on the identical training partition (`step <= 450`, $N = 6,014,717$, `class_weight='balanced'`, `max_iter=50`, seed `42`). Optimal decision thresholds were selected on validation data (`451 <= step <= 550`, $N = 151,302$) using a fine candidate grid (step size $0.001$). The final holdout (`step >= 551`, $N = 196,601$) was evaluated strictly using the validation-selected thresholds.

| Experiment Name | Feature Count | Features Included | Val PR-AUC (AP) | Val ROC-AUC | Optimal Val Cutoff | Val F1 Score | Holdout PR-AUC | Holdout ROC-AUC | Holdout Precision | Holdout Recall | Holdout F1 Score | Holdout Alerts |
| :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`model_9_baseline`** | 9 | `amount`, `log1p`, `hour`, `day`, 5 types | 0.2519 | 0.9229 | 0.9830 | 0.3826 | 0.3836 | 0.9185 | 0.5456 | 0.4232 | **0.4767** | 1,666 |
| **`model_8_no_day`** | **8** | **`amount`, `log1p`, `hour`, 5 types** | **0.3115** | **0.9312** | **0.9660** | **0.3773** | **0.4479** | **0.9358** | **0.5369** | **0.4525** | **0.4912** | **1,810** |
| **`model_7_no_temporal`** | 7 | `amount`, `log1p`, 5 types | 0.2665 | 0.8988 | 0.9710 | 0.3210 | 0.3222 | 0.8974 | 0.4452 | 0.2379 | **0.3103** | 1,148 |
| **`model_7_no_amount`** | 7 | `hour`, `day`, 5 types | 0.1117 | 0.8652 | 0.9910 | 0.2865 | 0.2197 | 0.8643 | 0.3444 | 0.2793 | **0.3084** | 1,742 |
| **`model_4_no_type`** | 4 | `amount`, `log1p`, `hour`, `day` | 0.1680 | 0.8614 | 0.9860 | 0.2317 | 0.2686 | 0.8617 | 0.3542 | 0.2831 | **0.3147** | 1,716 |
| **`model_7_business_core`**| 7 | `amount`, `log1p`, 5 types | 0.2665 | 0.8988 | 0.9710 | 0.3210 | 0.3222 | 0.8974 | 0.4452 | 0.2379 | **0.3103** | 1,148 |

---

## 4. Subperiod Temporal Stability on Holdout

To test whether models hold up as time progresses, the holdout partition ($N = 196,601$) was split into Early Holdout (Days 23–27, `step <= 650`, $N = 164,037$) and Late Holdout (Days 28–31, `step > 650`, $N = 32,564$):

| Model Variant | Early Holdout PR-AUC (Days 23–27) | Late Holdout PR-AUC (Days 28–31) | $\Delta \text{PR-AUC}$ (Late - Early) | Temporal Stability Assessment |
| :--- | :---: | :---: | :---: | :--- |
| **`model_9_baseline`** | 0.3565 | 0.4188 | +0.0623 | Sensitive to calendar day splits. |
| **`model_8_no_day`** | **0.4133** | **0.4909** | **+0.0776** | **Superior precision-recall across both early and late periods.** |
| **`model_7_no_temporal`** | 0.3076 | 0.3390 | +0.0314 | Underperforms due to omission of diurnal hour. |
| **`model_7_no_amount`** | 0.2000 | 0.2468 | +0.0468 | Severe degradation across all periods. |
| **`model_4_no_type`** | 0.2339 | 0.3299 | +0.0960 | High false positive rate without rail restrictions. |

---

## 5. Architectural Recommendation & Governance Decision

1. **Ablation of `transaction_day`:**
   `transaction_day` should be permanently removed from the primary production model. The empirical evidence is decisive: removing `transaction_day` boosts Validation PR-AUC by **+23.6%** (from $0.2519 \to 0.3115$) and Holdout PR-AUC by **+16.8%** (from $0.3836 \to 0.4479$). It eliminates an ungeneralizable synthetic simulator artifact while significantly improving discrimination.
2. **Retention of `transaction_hour`:**
   `transaction_hour` should be preserved. Dropping it causes Holdout PR-AUC to drop from $0.4479 \to 0.3222$ (-28.1%). Diurnal variations in transaction intensity represent legitimate, stationary behavioral signals.
3. **Primary Portfolio Model Specification:**
   The recommended 8-feature specification is:
   `['amount', 'log1p_amount', 'transaction_hour', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']`.
