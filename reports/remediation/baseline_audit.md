# FinSight — Remediation Audit Report: Phase 0 (Baseline Reproduction)

**Audit Execution Date:** 2026-10-19  
**Auditor:** Senior ML Engineer & Quality Auditor  
**Repository State:** `master` (working tree dirty: uncommitted changes in `README.md`, `requirements.txt`, and new milestone modules)  
**Python Runtime:** 3.14.3 (tags/v3.14.3:323c59a, Feb 3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]  
**Key Package Versions:**  
- `scikit-learn`: 1.8.0  
- `pandas`: 3.0.1  
- `numpy`: 2.4.2  
- `shap`: 0.54.1  
- `pytest`: 9.1.1  

---

## 1. Executive Summary

A comprehensive forensic audit and empirical baseline reproduction of the FinSight ML pipeline was executed. The purpose was to independently verify the saved estimator artifacts, feature definitions, chronological partition boundaries, metric calculations, and threshold dynamics before undertaking deeper model remediation.

### Key Audit Findings:
1. **Saved Artifact Verification:** All four serialized models in `models/` were successfully loaded and inspected. The primary candidate `models/hist_gradient_boosting_ablated.joblib` contains a `HistGradientBoostingClassifier` trained on exactly 9 non-leaky features, with locked operational threshold `0.9900`.
2. **Holdout Baseline Reproduction:** Evaluating `models/hist_gradient_boosting_ablated.joblib` on the unseen final holdout partition (`step > 550`, $N = 196,601$, $2,148$ simulated frauds) at threshold $0.9900$ reproduces the exact reported metrics:
   - **Average Precision (AP):** $0.3836$
   - **ROC-AUC:** $0.9185$
   - **Precision:** $0.6901$
   - **Recall:** $0.3431$
   - **F1 Score:** $0.4583$
   - **Confusion Matrix:** $\text{TP} = 737$, $\text{FP} = 331$, $\text{FN} = 1,411$, $\text{TN} = 194,122$ (Sum: $196,601$, Positives: $1,068$).
3. **High-Risk Rule Engine Benchmark Reproduction:**
   - **Precision:** $0.7204$
   - **Recall:** $0.1872$
   - **F1 Score:** $0.2971$ (reported as $0.2972$ due to 4th-decimal rounding)
   - **Flagged Positives:** $558$
4. **Root Cause of Threshold Search Discrepancy Identified:**
   The original threshold optimizer in `find_optimal_threshold` used `np.linspace(0.01, 0.99, 100)`, yielding a coarse step size of $\approx 0.0099$. The grid jumped directly from $t = 0.9801$ (where validation F1 was $0.2254$) to $t = 0.9900$ (where validation F1 was $0.3434$). High-resolution grid evaluation on validation data reveals that the true validation F1 peak lies at $t = 0.9830$ (Validation F1: $0.3826$, $\text{Precision} = 0.3462$, $\text{Recall} = 0.4276$, Alerts: $1,245$), confirming that coarse candidate discretization caused the omission of superior operating points.

---

## 2. Environment & Dependency Inventory

| Component | Reported Version / Specification | Verified Active Version | Status |
| :--- | :--- | :--- | :--- |
| **Python** | 3.11+ | 3.14.3 (64-bit AMD64) | Compliant |
| **scikit-learn** | 1.4+ | 1.8.0 | Compliant |
| **pandas** | 2.0+ | 3.0.1 | Compliant |
| **numpy** | 1.26+ | 2.4.2 | Compliant |
| **shap** | 0.45+ | 0.54.1 | Compliant |
| **pytest** | 8.0+ | 9.1.1 | Compliant |
| **Source Data** | `PS_20174392719_1491204439457_log.csv` / `data/processed/transactions.csv.gz` | $6,362,620$ rows, $8,213$ frauds | Verified |

---

## 3. Dataset Partitioning & Prevalence Verification

Chronological boundaries were verified against `data/processed/transactions.csv.gz`:

| Partition Name | Step Boundary | Simulated Days | Total Rows | Fraud Count | Fraud Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Training** | `step <= 450` | Days 1–19 | 6,014,717 | 5,057 | **0.0841%** (1 in 1,189) |
| **Validation** | `451 <= step <= 550` | Days 19–23 | 151,302 | 1,008 | **0.6662%** (1 in 150) |
| **Final Holdout** | `step >= 551` | Days 23–31 | 196,601 | 2,148 | **1.0926%** (1 in 92) |
| **Full Dataset** | `1 <= step <= 743` | Days 1–31 | 6,362,620 | 8,213 | **0.1291%** (1 in 775) |

**Observations on Temporal Prevalence Drift:**
- Fraud prevalence increases by **7.9×** from training to validation, and by **13.0×** from training to holdout.
- This extreme non-stationarity in PaySim is driven by simulated fraud campaigns clustered in the second half of the month.
- Models evaluating on holdout face a radically different prior odds ratio than during training.

---

## 4. Model Artifact Inspection

All four serialized `.joblib` bundles in `models/` were inspected for metadata integrity, estimator class, feature names, and locked decision thresholds:

```text
1. dummy_baseline.joblib
   - Estimator: DummyClassifier(strategy='prior')
   - Features (10): ['amount', 'log1p_amount', 'transaction_hour', 'transaction_day',
                     'is_merchant_dest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
                     'type_PAYMENT', 'type_TRANSFER']
   - Locked Threshold: 0.5000

2. logistic_regression.joblib
   - Estimator: Pipeline(StandardScaler -> LogisticRegression(class_weight='balanced'))
   - Features (10): ['amount', 'log1p_amount', 'transaction_hour', 'transaction_day',
                     'is_merchant_dest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
                     'type_PAYMENT', 'type_TRANSFER']
   - Locked Threshold: 0.9009 (Validation optimal)

3. hist_gradient_boosting_canonical.joblib
   - Estimator: HistGradientBoostingClassifier(class_weight='balanced', max_iter=50)
   - Features (10): ['amount', 'log1p_amount', 'transaction_hour', 'transaction_day',
                     'is_merchant_dest', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT',
                     'type_PAYMENT', 'type_TRANSFER']
   - Locked Threshold: 0.9900 (Validation optimal)

4. hist_gradient_boosting_ablated.joblib
   - Estimator: HistGradientBoostingClassifier(class_weight='balanced', max_iter=50)
   - Features (9): ['amount', 'log1p_amount', 'transaction_hour', 'transaction_day',
                    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT',
                    'type_TRANSFER']
   - Locked Threshold: 0.9900 (Validation optimal)
   - Ablated Feature: `is_merchant_dest` excluded due to collinearity with `type_PAYMENT`
```

---

## 5. Metric Reconciliation & Holdout Ground Truth Verification

### A. Holdout Performance: HistGradientBoosting Ablated

Evaluated on `step >= 551` ($N = 196,601$, $y = 1 \implies 2,148$):

| Metric | Machine-Generated Evaluation | Previous Reported Baseline | Reconciled Discrepancy |
| :--- | :--- | :--- | :--- |
| **Average Precision (AP)** | 0.3836 | 0.3836 | Exact match |
| **ROC-AUC** | 0.9185 | 0.9185 | Exact match |
| **Decision Threshold** | 0.9900 | 0.9900 | Exact match |
| **Precision** | 0.6901 | 0.6901 | Exact match |
| **Recall** | 0.3431 | 0.3431 | Exact match |
| **F1 Score** | 0.4583 | 0.4583 | Exact match |
| **True Positives (TP)** | 737 | 737 | Exact match |
| **False Positives (FP)** | 331 | 331 | Exact match |
| **False Negatives (FN)** | 1,411 | 1,411 | Exact match |
| **True Negatives (TN)** | 194,122 | 194,122 | Exact match |
| **Predicted Positives (TP + FP)** | 1,068 | 1,068 | Exact match |
| **Sum of Confusion Matrix** | 196,601 | 196,601 | Verified ($= \text{Total Holdout}$) |

### B. Rule Engine Benchmark: HIGH Risk Tier (Score $\ge 70$)

Evaluated on the same holdout slice ($N = 196,601$):
- **Precision:** $0.7204$ (reported $0.7204$)
- **Recall:** $0.1872$ (reported $0.1872$)
- **F1 Score:** $0.2971$ (reported $0.2972$ due to rounding)
- **Flagged Alerts:** $558$

---

## 6. Threshold Optimization Sensitivity Audit

The candidate threshold grid was investigated on validation data (`451 <= step <= 550`, $N = 151,302$, $y = 1 \implies 1,008$):

```text
Original Grid: np.linspace(0.01, 0.99, 100) -> step size ~ 0.009899
  - Candidate 98: t = 0.9801 -> P = 0.1487, R = 0.4653, F1 = 0.2254, Alerts = 3,153
  - Candidate 99: t = 0.9900 -> P = 0.3498, R = 0.3373, F1 = 0.3434, Alerts = 972
  -> Selected: t = 0.9900 (F1 = 0.3434)

Fine Grid Audit: np.linspace(0.01, 0.999, 990) -> step size = 0.001
  - t = 0.9800 -> P = 0.1487, R = 0.4653, F1 = 0.2254, Alerts = 3,153
  - t = 0.9820 -> P = 0.2063, R = 0.4504, F1 = 0.2830, Alerts = 2,201
  - t = 0.9830 -> P = 0.3462, R = 0.4276, F1 = 0.3826, Alerts = 1,245  <-- TRUE PEAK
  - t = 0.9840 -> P = 0.3457, R = 0.4266, F1 = 0.3819, Alerts = 1,244
  - t = 0.9850 -> P = 0.3414, R = 0.4187, F1 = 0.3761, Alerts = 1,236
  - t = 0.9860 -> P = 0.3387, R = 0.4137, F1 = 0.3725, Alerts = 1,231
  - t = 0.9870 -> P = 0.3391, R = 0.4087, F1 = 0.3707, Alerts = 1,215
  - t = 0.9880 -> P = 0.3325, R = 0.3948, F1 = 0.3610, Alerts = 1,197
  - t = 0.9890 -> P = 0.3303, R = 0.3909, F1 = 0.3580, Alerts = 1,193
  - t = 0.9900 -> P = 0.3498, R = 0.3373, F1 = 0.3434, Alerts = 972
```

**Conclusion on Threshold Limitation:**
The threshold discrepancy where $t=0.985$ was reported to achieve higher F1 than $0.990$ is fully explained by candidate discretization. The original 100-step grid skipped the entire sub-interval $[0.9810, 0.9899]$. In Phase 1, we will replace this with a fine-grid or quantile-based search on validation data.

---

## 7. Current Test Suite Status

Executed `py -m pytest tests/ -v`:
- **Total Tests Collected:** 74
- **Passed:** 74
- **Failed:** 0
- **Duration:** 22.09s
- **Coverage:** Feature extraction, leakage prevention, PostgreSQL loader, SQL analytics, model baselines, rule engine determinism, TreeSHAP mathematical additivity, local explanation schemas, and artifact consistency.

---

## 8. Baseline Audit Sign-Off

The repository state and reported metrics are fully verified and reproduced. Phase 0 is complete.
Proceeding to **Phase 1: Evaluation Correctness and Threshold Selection**.
