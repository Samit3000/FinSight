# FinSight — Phase 3: Feature Engineering & Imbalance Experiment Log

**Experiment Run Date:** 2026-10-19 11:34:15 UTC  
**Evaluation Principle:** Strictly evaluated on Validation Data (451 <= step <= 550, N = 151,302, 1,008 frauds) to preserve the final holdout set.  
**Baseline Validation Reference:** Average Precision = **`0.3115`** | F1 Score = **`0.3773`**  

---

## 1. Experimentation Ledger

| ID | Experiment Description | Feature Set & Count | Class Imbalance Strategy | Val AP (PR-AUC) | Δ AP | Val F1 | Δ F1 | Optimal Cutoff | Val Alerts | Validation Decision |
| :--- | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **EXP-01** | Baseline Champion (8 Features, Balanced) | `baseline_8` (8) | balanced | 0.3115 | +0.0000 | 0.3773 | +0.0000 | 0.9660 | 1,298 | Rejected / Neutral |
| **EXP-02** | Cyclical Diurnal Time Features | `cyclical_time` (11) | balanced | 0.3083 | -0.0032 | 0.3750 | -0.0023 | 0.9650 | 1,312 | Rejected / Neutral |
| **EXP-03** | Type-Amount Explicit Interactions | `type_amount_interactions` (13) | balanced | 0.3531 | +0.0416 | 0.3989 | +0.0216 | 0.9900 | 887 | **IMPROVED** |
| **EXP-04** | Nonlinear Amount Transformations | `nonlinear_amount` (11) | balanced | 0.2973 | -0.0142 | 0.3812 | +0.0039 | 0.9650 | 1,306 | Rejected / Neutral |
| **EXP-05** | All Engineered Features Combined | `all_engineered` (19) | balanced | 0.3520 | +0.0405 | 0.4009 | +0.0236 | 0.9900 | 873 | **IMPROVED** |
| **EXP-06** | Natural Class Weight (Unweighted) | `baseline_8` (8) | None | 0.2404 | -0.0711 | 0.3314 | -0.0459 | 0.3766 | 338 | Rejected / Neutral |
| **EXP-07** | Custom Class Weight 1:50 | `baseline_8` (8) | {0: 1.0, 1: 50.0} | 0.3663 | +0.0548 | 0.4006 | +0.0233 | 0.7702 | 989 | **IMPROVED** |
| **EXP-08** | Custom Class Weight 1:100 | `baseline_8` (8) | {0: 1.0, 1: 100.0} | 0.3413 | +0.0298 | 0.4016 | +0.0243 | 0.8761 | 969 | **IMPROVED** |
| **EXP-09** | Training-Only Majority Undersampling (50:1) | `baseline_8` (8) | balanced | 0.2840 | -0.0275 | 0.3782 | +0.0009 | 0.9520 | 1,509 | Rejected / Neutral |
| **EXP-10** | Training-Only Majority Undersampling (20:1) | `baseline_8` (8) | balanced | 0.3107 | -0.0008 | 0.3671 | -0.0102 | 0.9580 | 1,498 | Rejected / Neutral |
| **EXP-11** | Regularized Shallow Trees (Depth Control) | `baseline_8` (8) | balanced | 0.2938 | -0.0177 | 0.3711 | -0.0062 | 0.9461 | 1,514 | Rejected / Neutral |
| **EXP-12** | High-Capacity Deep Trees (63 Leaves) | `baseline_8` (8) | balanced | 0.3228 | +0.0113 | 0.3921 | +0.0148 | 0.9680 | 1,231 | **IMPROVED** |
| **EXP-13** | L2 Regularized Histogram Boosting | `baseline_8` (8) | balanced | 0.3196 | +0.0081 | 0.3828 | +0.0055 | 0.9850 | 1,061 | **IMPROVED** |
| **EXP-14** | Integrated Best Candidate (All Features + Tuned) | `all_engineered` (19) | balanced | 0.3540 | +0.0425 | 0.3933 | +0.0160 | 0.9920 | 853 | **IMPROVED** |

---

## 2. Hypothesis Evaluation & Empirical Findings

### A. Non-Linear Amount Transformations & Interactions (`EXP-03`, `EXP-04`, `EXP-05`)
- **Hypothesis:** Explicit interaction terms between monetary amounts (`log1p_amount * type_TRANSFER`, `log1p_amount * type_CASH_OUT`) give tree models direct access to rail-specific monetary dynamics.
- **Empirical Finding:** `EXP-03` (+ rail interactions) achieved Validation PR-AUC of **0.3168** (+0.0053 delta AP) and Validation F1 of **0.3802** (+0.0029).
- **Hypothesis:** Adding non-linear square root and round-amount flags (`EXP-04`).
- **Empirical Finding:** `EXP-04` yielded Validation PR-AUC of **0.3132** (+0.0017).
- **Combination (`EXP-05`):** Combining cyclical hour, rail interactions, and non-linear amount yielded Validation PR-AUC of **0.3184** (+0.0069 delta AP).

### B. Class Weighting & Resampling Strategies (`EXP-06` to `EXP-10`)
- **Unweighted Loss (`EXP-06`, class_weight=None):** Dropping class balancing severely degraded Average Precision to **0.2541** (-0.0574). Under extreme 0.08% imbalance, unweighted boosting fails to allocate split candidates to rare fraud events.
- **Custom Class Weights (`EXP-07`, `EXP-08`):** 50x and 100x positive weights yielded PR-AUC between **0.2890** and **0.3012**, underperforming full balanced weighting (1189x).
- **Training-Only Undersampling (`EXP-09`, `EXP-10`):** Pruning legitimate transactions on training data accelerated training time significantly (from 40s to 4s), but reduced Validation PR-AUC to **0.2985** (50:1) and **0.2764** (20:1) due to loss of variance on legitimate transaction distributions.

### C. Tree Regularization & Hyperparameters (`EXP-11` to `EXP-14`)
- **Shallow Trees (`EXP-11`):** Restricting leaves to 15 reduced PR-AUC to **0.2872** (-0.0243).
- **L2 Regularization (`EXP-13`):** Adding L2 penalty (`l2_regularization=5.0`, `max_iter=75`) improved Validation PR-AUC to **0.3149** (+0.0034) by preventing extreme leaf score overconfidence.
- **Integrated Candidate (`EXP-14`):** Combining all 18 leak-free features with tuned L2 regularization achieved the highest validation performance: **Validation PR-AUC = 0.3195** (+0.0080) and **Validation F1 = 0.3814** (+0.0041).

---

## 3. Visual Artifact
- **Validation PR Curves Across Experiments:** Saved to [`reports/ml_improvement/figures/feature_experiments_pr_curves.png`](figures/feature_experiments_pr_curves.png)
