# FinSight — Frozen Model Evaluation Summary

**Evaluated Model:** `HistGradientBoosting No Day`  
**Model Artifact:** `models/hist_gradient_boosting_no_day.joblib`  
**Locked Decision Threshold:** `0.9660`  
**Feature Count:** `8`  
**Features:** `amount, log1p_amount, transaction_hour, type_CASH_IN, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER`  

---

## 1. Validation Partition Performance (`450 < step <= 550`)
- **Row Count:** 151,302 transactions (1,008 frauds, 0.6662% prevalence)
- **Average Precision (PR-AUC):** `0.3115`
- **ROC-AUC:** `0.9509`
- **Precision:** `0.3351`
- **Recall:** `0.4315`
- **F1 Score:** `0.3773`
- **Alert Count:** `1,298` (0.86% alert rate)

---

## 2. Frozen Holdout Partition Performance (`step > 550`)
- **Row Count:** 196,601 transactions (2,148 frauds, 1.0926% prevalence)
- **Average Precision (PR-AUC):** `0.4479`
- **ROC-AUC:** `0.9487`
- **Precision:** `0.5774`
- **Recall:** `0.4274`
- **F1 Score:** `0.4912`
- **Alert Count:** `1,590` (0.81% alert rate)
- **Confusion Matrix:**
  - True Positives (TP): `918`
  - False Positives (FP): `672`
  - False Negatives (FN): `1,230`
  - True Negatives (TN): `193,781`
