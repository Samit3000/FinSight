# FinSight — Phase 1: Baseline Reproduction & Forensic Audit Report

**Audit Date:** 2026-10-19 11:34:15 UTC  
**Auditor:** Senior Applied ML Engineer (Fraud Systems & Imbalance Evaluation)  
**Evaluated Artifact:** `C:\Users\samit\OneDrive\Desktop\FinSight\models\hist_gradient_boosting_no_day.joblib`  
**Model Identifier:** `HistGradientBoosting No Day`  
**Evaluation Dataset:** `data/processed/transactions.csv.gz` (6,362,620 transactions, 8,213 frauds)  

---

## 1. Executive Summary & Verification Findings

This report documents the empirical audit and exact numerical reproduction of FinSight's frozen champion model (`models/hist_gradient_boosting_no_day.joblib`). 

### Authoritative Verification Status: **VERIFIED EXACT MATCH**
All reported baseline metrics were successfully reproduced from the serialized model artifact and raw transaction data without retraining or parameter manipulation.

| Metric | Reported Baseline | Re-Scored Empirical Holdout | Verification Status | Exact Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Average Precision (PR-AUC)** | `0.4479` | **`0.4479`** | **EXACT MATCH** | 0.0000 |
| **ROC-AUC** | `0.9487` | **`0.9487`** | **EXACT MATCH** | 0.0000 |
| **Precision** | `57.74%` | **`57.74%`** | **EXACT MATCH** | 0.00% |
| **Recall** | `42.74%` | **`42.74%`** | **EXACT MATCH** | 0.00% |
| **F1 Score** | `0.4912` | **`0.4912`** | **EXACT MATCH** | 0.0000 |
| **Decision Threshold ($t$)** | `0.9660` | **`0.9660`** | **EXACT MATCH** | 0.0000 |
| **Holdout Alerts Flagged** | `1,590` | **`1,590`** | **EXACT MATCH** | 0 |
| **True Positives (TP)** | `918` | **`918`** | **EXACT MATCH** | 0 |
| **False Positives (FP)** | `672` | **`672`** | **EXACT MATCH** | 0 |
| **False Negatives (FN)** | `1,230` | **`1,230`** | **EXACT MATCH** | 0 |
| **True Negatives (TN)** | `193,781` | **`193,781`** | **EXACT MATCH** | 0 |

---

## 2. Partition Population & Prevalence Verification

The dataset is partitioned chronologically across simulation steps to strictly guarantee that past models never learn from future data:

| Partition | Step Boundaries | Calendar Days | Total Transactions | Fraud Count | Fraud Prevalence |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Model Training** | `step <= 450` | Days 1–19 | 6,014,717 | 5,057 | **0.0841%** (1 in 1,189) |
| **Validation Tuning** | `451 <= step <= 550` | Days 19–23 | 151,302 | 1,008 | **0.6662%** (1 in 150) |
| **Final Holdout** | `step > 550` | Days 23–31 | 196,601 | 2,148 | **1.0926%** (1 in 92) |
| **Full Dataset** | `1 <= step <= 743` | 31 Days | 6,362,620 | 8,213 | 0.1291% (1 in 775) |

### Prevalence Shift Mechanism:
The 13-fold increase in fraud prevalence between training (0.0841%) and holdout (1.0926%) is driven by a **legitimate transaction volume drop**, not an escalation in fraudulent attacks. While fraudulent account compromise attacks occur at a stationary cadence ($pprox 260$ liquidations per day), consumer transaction volume plunges by over 95% after Day 17.

---

## 3. Threshold Provenance & Validation Tuning Audit

The decision threshold t = 0.9660 was evaluated by re-running the unconstrained F1 grid search on validation data (451 <= step <= 550):
- **Candidate Grid:** 1,000 threshold points between 0.001 and 0.999.
- **Validation Optimal Cutoff Derived:** **`0.9660`** (Exact match with locked artifact).
- **Validation Metrics at Locked Threshold ($t = 0.9660$):**
  - Average Precision: **`0.3115`**
  - ROC-AUC: **`0.9509`**
  - Precision: **`33.51%`**
  - Recall: **`43.15%`**
  - F1 Score: **`0.3773`**
  - Alert Volume: **`1,298`** (0.86% alert rate)

---

## 4. Operational Precision-at-Recall & Alert-Budget Profiles

### Holdout Precision at Predefined Recall Targets:
| Recall Target | Maximum Achievable Precision | Operational Feasibility |
| :---: | :---: | :--- |
| **20%** | **83.30%** | High-precision automated auto-freeze tier |
| **30%** | **69.96%** | Strict compliance automated queue |
| **40%** | **64.78%** | Primary operational triage zone |
| **50%** | **36.43%** | High-volume fraud escalation review |
| **60%** | **23.27%** | Diluted review zone (increased analyst fatigue) |
| **70%** | **15.44%** | Impractical operational threshold (severe false positive surge) |
| **80%** | **9.41%** | Heavy false alarm flooding |

### Holdout Recall at Capacity-Constrained Alert Budgets:
| Alert Budget (Max Reviews) | Alert Rate (%) | Intercepted Frauds (TP) | Captured Recall (%) | Resulting Precision (%) |
| :---: | :---: | :---: | :---: | :---: |
| **500 alerts** | 0.25% | 422 | 19.65% | 84.40% |
| **1,000 alerts** | 0.51% | 691 | 32.17% | 69.10% |
| **1,500 alerts** | 0.76% | 908 | 42.27% | 60.53% |
| **2,000 alerts** | 1.02% | 995 | 46.32% | 49.75% |

---

## 5. Visual Artifacts
- **Precision-Recall Curve:** Saved to [`reports/ml_improvement/figures/baseline_precision_recall_curve.png`](figures/baseline_precision_recall_curve.png)
- **Threshold Sensitivity Analysis:** Saved to [`reports/ml_improvement/figures/baseline_threshold_sensitivity.png`](figures/baseline_threshold_sensitivity.png)
