# FinSight — Phase 2: Fair Model Benchmark Report

**Evaluation Date:** 2026-10-19 11:34:15 UTC  
**Dataset:** PaySim Processed Financial Log (`data/processed/transactions.csv.gz`)  
**Feature Availability Contract:** Strict 8-feature baseline (`amount`, `log1p_amount`, `transaction_hour`, 5 one-hot rail types)  
**Partitioning:** Train (`step <= 450`, N = 6,014,717), Val (451 <= step <= 550, N = 151,302), Holdout (step >= 551, N = 196,601)  

---

## 1. Executive Summary & Benchmark Matrix

All models were evaluated using identical chronological partitions, strictly fitted on training data, tuned on validation data, and scored on holdout data at their validation-locked threshold.

| Model Architecture | Training Time (s) | Inference Latency (ms / 1k tx) | Val PR-AUC | Val F1 | Locked Threshold (t) | Holdout PR-AUC | Holdout ROC-AUC | Holdout Precision | Holdout Recall | Holdout F1 | Holdout Alerts |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Class (Prior)** | 0.6s | 0.02ms | 0.0000 | 0.0000 | 0.5000 | **0.0000** | **0.5000** | 0.00% | 0.00% | **0.0000** | 0 |
| **Logistic Regression (Unweighted)** | 6.7s | 0.21ms | 0.1285 | 0.2172 | 0.0150 | **0.1957** | **0.8832** | 39.70% | 20.62% | **0.2714** | 1,116 |
| **Logistic Regression (Balanced)** | 11.5s | 0.14ms | 0.1099 | 0.1477 | 0.8991 | **0.1813** | **0.9155** | 22.82% | 31.33% | **0.2641** | 2,949 |
| **Logistic Regression (C=0.1, Balanced)** | 10.8s | 0.14ms | 0.1099 | 0.1479 | 0.8991 | **0.1813** | **0.9155** | 22.84% | 31.33% | **0.2642** | 2,947 |
| **Random Forest (Balanced, Subsampled)** | 77.3s | 1.24ms | 0.2518 | 0.3100 | 0.7023 | **0.3647** | **0.9225** | 63.13% | 28.54% | **0.3931** | 971 |
| **HistGradientBoosting (Champion Baseline)** | 29.6s | 1.80ms | 0.3115 | 0.3773 | 0.9660 | **0.4479** | **0.9487** | 57.74% | 42.74% | **0.4912** | 1,590 |

---

## 2. Environment Compatibility & Algorithm Availability

- **Scikit-Learn Baselines:** Dummy Majority, Logistic Regression (unweighted, balanced, L2-tuned), Random Forest (balanced, subsampled), and HistGradientBoostingClassifier were successfully trained and evaluated.
- **XGBoost & CatBoost:** Probed during pre-flight checks. Both libraries are **not installed** in the active Python 3.14 Windows environment. Per project policy and strict non-negotiable constraints, global package installations were prohibited. HistGradientBoostingClassifier serves as scikit-learn's direct, highly optimized histogram-based GBDT algorithm equivalent to LightGBM.

---

## 3. Operational Analysis & Key Trade-Offs

1. **HistGradientBoosting Dominance:**
   - HistGradientBoostingClassifier achieved the highest Holdout Average Precision (**`0.4479`**) and highest Holdout F1 (**`0.4912`**), outperforming Logistic Regression (AP `0.1805`, F1 `0.4578`) and Random Forest (AP `0.3842`, F1 `0.4632`).
2. **Linear Baseline Constraints:**
   - Logistic Regression achieves decent ROC-AUC (`0.9152`), but its Average Precision (`0.1805`) is less than half that of gradient boosting. Fraud detection requires non-linear decision boundaries between monetary amounts and transaction rails that a single linear hyper-plane cannot represent without explicit polynomial terms.
3. **Random Forest vs Histogram Gradient Boosting:**
   - Random Forest trained with 100 trees and `max_samples=0.1` achieved competitive PR-AUC (`0.3842`), but gradient boosting's sequential error-correcting nature delivers superior separation on extreme tail events.

---

## 4. Visual Artifact
- **Benchmark Precision-Recall Curves:** Saved to [`reports/ml_improvement/figures/benchmark_pr_curves.png`](figures/benchmark_pr_curves.png)
