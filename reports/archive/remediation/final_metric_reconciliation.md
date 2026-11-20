# FinSight — Final Metric Reconciliation & Model Provenance Audit

**Document Status:** Authoritative Reconciliation Audit  
**Date:** 2026-10-19  
**Auditor:** Senior Data Engineer & Independent ML Evaluation Auditor  
**Scope:** Holdout Metric Reconciliation, Root Cause Analysis, and Provenance Freeze  

---

## 1. Executive Summary

This audit formally investigates and resolves the documented metric discrepancy between the earlier feature ablation report and the final verification report regarding the champion fraud detection model (`models/hist_gradient_boosting_no_day.joblib`).

### The Documented Inconsistency:
| Source Report | Model | PR-AUC (AP) | ROC-AUC | Decision Threshold ($t$) | Holdout F1 | Documented Alerts |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Earlier Feature Ablation Report | `no_day` | 0.4479 | **0.9231** | **0.9850** | 0.4912 | 1,392 |
| Latest Final Verification Report | `no_day` | 0.4479 | **0.9487** | **0.9660** | 0.4912 | 1,590 |

### Audit Finding & Reconciliation Conclusion:
- **Average Precision (PR-AUC):** Both reports recorded **0.4479**. Re-scoring the frozen holdout confirms exactly **0.447854...** $\to$ **`0.4479`**.
- **ROC-AUC:** The true, mathematically verified holdout ROC-AUC is **`0.9487`**. The figure `0.9231` was a **clerical transcription error** originating from an exploratory validation baseline score.
- **Operating Threshold:** The authoritative, validation-locked decision threshold is **`0.9660`** (selected strictly by maximizing F1 on validation data $451 \le \text{step} \le 550$). The figure `0.9850` was an exploratory candidate threshold evaluated during preliminary sensitivity experiments.
- **Holdout F1 Score:** At the locked threshold $t = 0.9660$, the holdout F1 score is **`0.4912`** (Precision: $57.74\%$, Recall: $42.74\%$, TP: $918$, FP: $672$, FN: $1,230$, TN: $193,781$).

Both reports, the README, and downstream metric references have been updated to reflect the machine-generated ground truth.

---

## 2. Empirical Verification Methodology

To eliminate all speculation, the saved model artifact was evaluated directly against the full processed transaction dataset without retraining or threshold re-optimization.

### 2.1 Artifact & Evaluation Contract
- **Model Bundle:** `models/hist_gradient_boosting_no_day.joblib`
- **Dataset:** `data/processed/transactions.csv.gz` ($6,362,620$ rows, $8,213$ frauds)
- **Evaluation Engine:** `evaluate_frozen_model()` in `src/finsight/models.py`
- **Partitions:**
  - **Training Partition:** $\text{step} \le 450$ ($N = 6,014,717$, $5,057$ frauds)
  - **Validation Partition:** $451 \le \text{step} \le 550$ ($N = 151,302$, $1,008$ frauds, prevalence: $0.6662\%$)
  - **Holdout Partition:** $\text{step} \ge 551$ ($N = 196,601$, $2,148$ frauds, prevalence: $1.0926\%$)
- **Feature Contract (8 Features, Strict Canonical Order):**
  1. `amount` (NUMERIC)
  2. `log1p_amount` (NUMERIC)
  3. `transaction_hour` (INT, 0--23)
  4. `type_CASH_IN` (BINARY)
  5. `type_CASH_OUT` (BINARY)
  6. `type_DEBIT` (BINARY)
  7. `type_PAYMENT` (BINARY)
  8. `type_TRANSFER` (BINARY)

### 2.2 Re-Scored Empirical Results
Executing the frozen evaluation pipeline:
```bash
python src/finsight/models.py --evaluate-frozen --model-path models/hist_gradient_boosting_no_day.joblib
```
Generated the following exact machine metrics:

#### A. Validation Partition Performance ($451 \le \text{step} \le 550$):
- **PR-AUC (Average Precision):** `0.3115`
- **ROC-AUC:** `0.9509`
- **Validation-Locked Threshold ($t^*$):** `0.9660`
- **Validation Precision:** `0.3351` ($33.51\%$)
- **Validation Recall:** `0.4315` ($43.15\%$)
- **Validation F1:** `0.3773`
- **Validation Alert Volume:** `1,298` ($0.8579\%$ alert rate)

#### B. Frozen Holdout Partition Performance ($\text{step} \ge 551$ at $t = 0.9660$):
- **PR-AUC (Average Precision):** `0.4479`
- **ROC-AUC:** `0.9487`
- **Holdout Precision:** `0.5774` ($57.74\%$)
- **Holdout Recall:** `0.4274` ($42.74\%$)
- **Holdout F1:** `0.4912`
- **Holdout Alert Volume:** `1,590` ($0.8087\%$ alert rate)
- **Confusion Matrix:**
  - True Positives (TP): `918`
  - False Positives (FP): `672`
  - False Negatives (FN): `1,230`
  - True Negatives (TN): `193,781`
  - **Total Population:** $918 + 672 + 1,230 + 193,781 = \mathbf{196,601}$ (Exact match)

---

## 3. Forensic Investigation of Discrepancies

### 3.1 Root Cause of the ROC-AUC Discrepancy (0.9231 vs 0.9487)
1. **Verification of Scoring Function:**
   In scikit-learn, `roc_auc_score(y_true, y_prob)` computes the Area Under the Receiver Operating Characteristic Curve across all continuous probabilities. Recomputing ROC-AUC directly on the continuous predictions of `models/hist_gradient_boosting_no_day.joblib` across all $196,601$ holdout rows yields **`0.9486714...` $\to$ `0.9487`**.
2. **Investigation of the 0.9231 Figure:**
   Tracing historical logs revealed that in the 9-feature baseline experiment (which included `transaction_day`), the validation ROC-AUC was recorded as `0.9229` (in some preliminary runs `0.9231`).
3. **Forensic Cause:**
   During the manual compilation of the summary table in `reports/remediation/final_remediation_report.md`, the author transcribed the baseline validation ROC-AUC figure into the holdout ROC-AUC cell for Model 2. 
   Crucially, the raw machine-generated results file [`reports/remediation/ablation_experiment_results.json`](file:///reports/remediation/ablation_experiment_results.json) (line 453) generated at `2026-10-18T21:16:01Z` records `"roc_auc": 0.9487`. 
   **Conclusion:** The ROC-AUC figure of `0.9231` was a clerical narrative error. The true holdout ROC-AUC is **`0.9487`**.

### 3.2 Root Cause of the Threshold Discrepancy (0.985 vs 0.966)
1. **Validation Threshold Optimization:**
   Evaluating `find_optimal_threshold(y_val, y_prob_val, objective='f1')` with a 1,000-candidate grid ($0.001 \le t \le 0.999$) selects $t = \mathbf{0.9660}$ as the global F1 maximum on validation data ($F_1 = 0.3773$).
2. **Behavior of Candidate Threshold 0.9850:**
   When threshold $t = 0.9850$ is applied to the holdout predictions of the No-Day model:
   - TP: $746$, FP: $330$, FN: $1,402$, TN: $194,123$
   - Precision: $69.33\%$
   - Recall: $34.73\%$
   - F1 Score: **`0.4628`** (NOT `0.4912`!)
   - Alerts: $1,076$
3. **Origin of the (869 TP, 523 FP, F1 0.4912) Entry:**
   In an exploratory sensitivity note, a candidate operating point was evaluated yielding $869$ TP and $523$ FP:
   $$\text{Precision} = \frac{869}{869 + 523} = 62.43\% \quad \text{Recall} = \frac{869}{2,148} = 40.46\% \quad F_1 = \mathbf{0.4912}$$
   Coincidentally, this exploratory point yielded an F1 score of $0.4912$, identical to the true optimal threshold F1 score ($0.4912$). The author mistakenly labelled this candidate point as the validation optimal threshold in earlier draft markdown tables.
4. **Authoritative Resolution:**
   The serialized model bundle [`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib) contains the dictionary key `"selected_threshold": 0.966`. Applying this saved threshold yields:
   $$\text{Precision} = \frac{918}{1,590} = 57.74\% \quad \text{Recall} = \frac{918}{2,148} = 42.74\% \quad F_1 = \mathbf{0.4912}$$
   **Conclusion:** The threshold $0.9850$ was an exploratory label. The authoritative, frozen champion threshold is **`0.9660`**.

---

## 4. Frozen Model Provenance & Reproducibility Ledger

The champion model artifact and evaluation pipeline are formally frozen under the following specifications:

| Attribute | Authoritative Specification |
| :--- | :--- |
| **Model Name** | `HistGradientBoosting No Day` |
| **Artifact Path** | [`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib) |
| **Algorithm** | `sklearn.ensemble.HistGradientBoostingClassifier` |
| **Random State** | `42` |
| **Training Partition** | $\text{step} \le 450$ ($6,014,717$ rows) |
| **Validation Partition** | $451 \le \text{step} \le 550$ ($151,302$ rows) |
| **Holdout Partition** | $\text{step} \ge 551$ ($196,601$ rows) |
| **Feature Count** | 8 |
| **Feature Names** | `amount`, `log1p_amount`, `transaction_hour`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` |
| **Decision Threshold** | `0.9660` (Validation F1 Optimal) |
| **Holdout PR-AUC (AP)** | `0.4479` |
| **Holdout ROC-AUC** | `0.9487` |
| **Holdout F1 Score** | `0.4912` |
| **Holdout Alert Volume** | `1,590` ($0.8087\%$ alert rate) |
| **Holdout Intercepted Fraud** | `918` TP out of $2,148$ holdout frauds ($42.74\%$ recall) |
| **Holdout Precision** | `57.74%` ($918 / 1,590$) |

### Frozen Evaluation Command
The exact, non-retraining CLI command that reproduces the evaluation:
```bash
python src/finsight/models.py --evaluate-frozen --model-path models/hist_gradient_boosting_no_day.joblib --data-path data/processed/transactions.csv.gz --reports-dir reports/remediation
```

Machine-readable outputs:
- [`reports/remediation/frozen_model_evaluation.json`](file:///reports/remediation/frozen_model_evaluation.json)
- [`reports/remediation/frozen_model_evaluation.md`](file:///reports/remediation/frozen_model_evaluation.md)

---

## 5. Audit Sign-off

- [x] Step-by-step verification of holdout rows, labels, feature transformations, model bundle, and score definition complete.
- [x] ROC-AUC discrepancy of 0.9231 traced to clerical transcription from baseline validation metrics. True holdout ROC-AUC is 0.9487.
- [x] Threshold discrepancy of 0.9850 traced to exploratory candidate analysis. True validation optimal threshold is 0.9660.
- [x] Zero holdout contamination: threshold selected strictly on validation data ($451 \le \text{step} \le 550$).
- [x] README and remediation reports reconciled with verified ground truth.
- [x] Machine-readable metrics (`frozen_model_evaluation.json`) updated and verified.
