# FinSight — Milestone 4: Supervised ML Baselines Report

**Generated UTC:** 2026-10-18T15:06:06Z  
**Random Seed:** 42  
**Scikit-Learn Baselines:** Dummy (Majority), Logistic Regression (Balanced), HistGradientBoosting  

---

## 1. Chronological Partitioning Design

Temporal ordering strictly prevents future leakage into past model training.

| Partition | Step Range | Calendar Days | Row Count | Fraud Count | Fraud Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Training** | `step <= 450` | Days 1–19 | 6,014,717 | 5,057 | 0.0841% |
| **Validation** | `451 <= step <= 550` | Days 19–23 | 151,302 | 1,008 | 0.6662% |
| **Final Holdout** | `step >= 551` | Days 23–31 | 196,601 | 2,148 | 1.0926% |
| **Total** | `1 <= step <= 743` | 31 Days | 6,362,620 | 8,213 | 0.1291% |

> **Leakage Governance Rule:**
> - `StandardScaler` and all estimators fit **strictly** on the Model Training partition.
> - Validation partition was used **exclusively** for hyperparameter assessment, ablation, and decision threshold optimization.
> - Final Holdout partition was evaluated **once** with locked models and locked thresholds.

---

## 2. Validation Partition Results & Threshold Selection

Decision thresholds were selected on the validation partition to optimize the F1 score under extreme class imbalance.

| Model | PR-AUC (AP) | ROC-AUC | Selected Threshold | Threshold Selection Method | Precision | Recall | F1 Score | Predicted Positives |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.0000 | 0.5000 | 0.5000 | Majority class reference | 0.0000 | 0.0000 | 0.0000 | 0 |
| **Logistic Regression (Balanced)** | 0.1104 | 0.9156 | 0.9009 | Grid search maximizing Val F1 | 0.0994 | 0.2768 | 0.1463 | 2,807 |
| **HistGradientBoosting (Canonical)** | 0.2519 | 0.9229 | 0.9900 | Grid search maximizing Val F1 | 0.3498 | 0.3373 | 0.3434 | 972 |
| **HistGradientBoosting (Ablated)** | 0.2519 | 0.9229 | 0.9900 | Grid search maximizing Val F1 | 0.3498 | 0.3373 | 0.3434 | 972 |

---

## 3. Feature Ablation Study: `is_merchant_dest`

Ablation performed on candidate tree-based model (**HistGradientBoostingClassifier**) on the validation partition:

- **Canonical Features (including `is_merchant_dest`):** PR-AUC = **0.2519**, F1 = **0.3434**
- **Ablated Features (excluding `is_merchant_dest`):** PR-AUC = **0.2519**, F1 = **0.3434**
- **Delta PR-AUC:** **+0.0000**

### Domain & Simulation Risk Rationale
In PaySim, `is_merchant_dest` is strictly 1 only on PAYMENT transactions, and in PaySim 100% of frauds occur exclusively on TRANSFER and CASH_OUT rails. Consequently, `is_merchant_dest` is collinear with `type_PAYMENT` and provides zero marginal predictive power (+0.0000 delta PR-AUC). In real-world payment networks, merchant accounts experience significant fraud (e.g., merchant bust-out, fake storefronts, card-not-present fraud). Treating merchant destination as an immunity signal is an artifactual PaySim shortcut and should be avoided in real-world deployment.

---

## 4. Final Holdout Evaluation (Locked Configuration)

Evaluated once on final unseen future transactions (`step > 550`, 196,601 transactions, 2,148 simulated frauds).

| Model | PR-AUC (AP) | ROC-AUC | Locked Threshold | Precision | Recall | F1 Score | TP | FP | FN | TN |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Baseline** | 0.0000 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 | 0 | 2,148 | 194,453 |
| **Logistic Regression** | 0.1805 | 0.9152 | 0.9009 | 0.2430 | 0.2914 | 0.2650 | 626 | 1,950 | 1,522 | 192,503 |
| **HistGradientBoosting (Canonical)** | **0.3836** | **0.9185** | **0.9900** | **0.6901** | **0.3431** | **0.4583** | **737** | **331** | **1,411** | **194,122** |
| **HistGradientBoosting (Ablated)** | **0.3836** | **0.9185** | **0.9900** | **0.6901** | **0.3431** | **0.4583** | **737** | **331** | **1,411** | **194,122** |

---

## 5. Comparison with Milestone 3 Explainable Rule Engine

| System | Primary Role | Operational Threshold | Precision | Recall | F1 Score | Flagged Volume |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Rule Engine (Milestone 3)** | Deterministic operational cues | `HIGH` risk tier (score >= 70) | **0.7204** | 0.1872 | 0.2972 | 558 reviews |
| **Logistic Regression (ML)** | Simple linear baseline | $t = 0.9009$ | 0.2430 | 0.2914 | 0.2650 | 2,576 alerts |
| **HistGradientBoosting (ML)** | Non-linear tree baseline | $t = 0.9900$ | 0.6901 | **0.3431** | **0.4583** | 1,068 alerts |

### Key Trade-offs:
1. **Rule Engine:** High precision (72.0%) on very high-certainty structural patterns, but misses 81.3% of simulated frauds due to fixed heuristic bounds.
2. **HistGradientBoosting:** Nearly doubles recall (from 18.7% to 34.3%) while maintaining high precision (69.0%), flagging 1,068 suspicious transactions.
3. **Threshold Tunability:** ML models provide continuous probabilistic risk ranking, enabling compliance teams to tune alert volume dynamically to team capacity.

---

## 6. Synthetic Simulation Limitations

1. **Synthetic Generation Artifacts:**
   - Results are strictly reflective of PaySim synthetic simulator mechanics.
   - PaySim only injects fraud into `TRANSFER` and `CASH_OUT`, and generates 100% zero fraud on `PAYMENT`, `DEBIT`, and `CASH_IN`.
   - Real-world payment systems experience merchant fraud, authorized push payment (APP) scams, friendly fraud, and identity takeovers across all transaction rails.
2. **No Production Readiness Claim:**
   - These models serve as defensible, reproducible supervised baselines for benchmark comparison, not production-ready detection systems.
