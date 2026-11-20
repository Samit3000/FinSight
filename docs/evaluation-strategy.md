# FinSight — Milestone 4 Evaluation Strategy & ML Governance Blueprint

> **Status (2026-11-20):** Sections 1, 3 and 4 still apply. **Section 2 (the 1–550 / 551–743 split) is superseded**: steps 551–743 were used for model selection, so the protocol was replaced in P0. The current protocol is below.

## Current protocol (P0–P1)

| Period | Steps | Use |
|---|---|---|
| Development | 1–650 | Training, feature and model selection, thresholds |
| Walk-forward folds | F1 ≤350→351–450 · F2 ≤450→451–550 · F3 ≤550→551–650 | Model comparison |
| Sealed final test | 651–743 | One pre-registered evaluation (P1.5, done). **Retired** |

- **Primary metrics:** average precision (with step-block bootstrap 95% intervals and paired differences), and recall/precision at a fixed review capacity (50, 100, 200 alerts per simulated day, ties counted at their expected value). Precision and F1 at a threshold are comparable only within one period, because fraud prevalence rises about 13× across the simulation.
- **Selection disclosure:** every training config records which periods influenced it; reports state reuse.
- **Enforcement:** `evaluate.py` raises `LockedTestAccessError` on steps ≥ 651 outside the pre-registered `final_test.py` run.
- **Champion:** E4 (`models/champion.json`); see [`current-state.md`](current-state.md).

---

## Original Milestone 4 blueprint (sections 2 superseded)

**Document Version:** 1.0.0  
**Status:** Approved Specification  
**Scope:** Evaluation Methodology for Supervised Fraud Detection Baseline  

---

## 1. Class Imbalance Context & High-Consequence Metrics

In the PaySim digital payments ecosystem (and real-world transaction banking), fraud is an extreme-rarity event:
- **Total Validated Transactions:** `6,362,620`
- **Simulated Fraud Cases:** `8,213`
- **Class Imbalance Ratio:** `1:773.7` (Base rate: **0.1291%**)

### Mandatory Metric Standard (Accuracy Ban)
Evaluating fraud detection models on **Accuracy** alone is **strictly prohibited**. A trivial zero-rule baseline that classifies 100% of transactions as legitimate achieves **99.8709% Accuracy**, while missing every single fraud event and allowing $12.06 Billion in financial exposure.

For all subsequent model benchmarks in Milestone 4, models must be evaluated against the following primary metrics:

1. **Precision (Positive Predictive Value)**:
   $$\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$$
   *Operational Meaning:* Percentage of flagged transactions that are actually fraudulent. Directly dictates investigator workload and false-positive customer friction.
2. **Recall (Sensitivity / Detection Rate)**:
   $$\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$$
   *Operational Meaning:* Percentage of actual fraud captured by the system. Directly dictates prevented financial exposure.
3. **F1-Score (Harmonic Mean)**:
   $$\text{F1} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$
4. **Average Precision (PR-AUC / Precision-Recall Area Under Curve)**:
   - Primary metric for ranking models across threshold ranges. ROC-AUC is misleading in extreme imbalance because a large number of true negatives artificially inflates the false positive rate denominator ($\text{FPR} = \frac{\text{FP}}{\text{FP} + \text{TN}}$). PR-AUC focuses exclusively on the positive minority class.
5. **Full Confusion Matrix**:
   - Explicit accounting of True Positives (TP), False Positives (FP), True Negatives (TN), and False Negatives (FN).

---

## 2. Chronological Train-Test Split (Zero Lookahead Bias)

Random $k$-fold cross-validation or random shuffling (`train_test_split(shuffle=True)`) is **forbidden** for financial time-series transaction data. Random sampling allows transactions from later days to leak into the training partition, creating artificial lookahead bias where future patterns inform past decisions.

### Chronological Partitioning Boundary:
The dataset spans 744 simulation steps (31 days). The canonical split strategy:
- **Training Set**: Steps `1` through `550` (~Days 1 to 23). Approximately 74% of chronological data.
- **Validation / Tuning Window**: Step range `450` to `550` for threshold tuning and hyperparameter selection.
- **Holdout Test Set**: Steps `551` through `743` (~Days 24 to 31). Approximately 26% of chronological data.

```text
[======================= Train Period (Steps 1–550) =======================] [=== Test Period (Steps 551–743) ===]
                                                   [== Validation (450-550) =]
```

---

## 3. Preprocessing Governance & Leakage Prevention

1. **Strict Fit-on-Train Protocol**:
   - Any preprocessing scaler (e.g. `StandardScaler`, `MinMaxScaler`), categorical frequency encoders, or decision threshold optimizers must be fit **exclusively on the training partition**.
   - Transformations are applied to the holdout test set using parameters learned on train.
2. **Prohibited Feature Policy**:
   - As enforced in [`src/finsight/features.py`](file:///c:/Users/samit/OneDrive/Desktop/FinSight/src/finsight/features.py), candidate predictor feature matrices must never contain:
     - `is_fraud` (evaluation target).
     - `is_flagged_fraud` (pre-existing heuristic flag).
     - `name_orig`, `name_dest` (sparse identifiers).
     - `oldbalance_org`, `newbalance_orig`, `oldbalance_dest`, `newbalance_dest` (simulator balance artifacts and liquidation leakage).
     - `transaction_id`, `created_at` (database metadata).

---

## 4. PaySim Simulator Limitations & Caveats

1. **Synthetic Behavioral Uniformity**:
   - In PaySim, fraudulent agents follow fixed procedural routines (e.g., immediate liquidation of account balance via TRANSFER followed by CASH_OUT). Real-world fraud exhibits evolving, adversarial evasion techniques.
2. **Overnight Denominator Anomaly**:
   - Legitimate volume drops by over 99% between hours 1 AM and 5 AM, whereas synthetic fraudulent agents trigger at steady hourly intervals. This creates an artificially high overnight fraud rate (~22%) due to small volume denominators rather than an explosion of absolute attack volume.
3. **Absence of Chargeback Delays**:
   - In real payment operations, fraud labels take 30 to 90 days to confirm via cardholder dispute disputes. PaySim provides instantaneous ground-truth labels upon transaction occurrence.
