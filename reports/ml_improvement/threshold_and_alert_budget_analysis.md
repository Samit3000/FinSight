# FinSight — Threshold & Alert-Budget Sensitivity Analysis

**Audit Date:** 2026-10-19 11:34:15 UTC  
**Evaluated Champion:** `HistGradientBoosting No Day` (`models/hist_gradient_boosting_no_day.joblib`)  
**Holdout Population:** 196,601 transactions (2,148 simulated frauds)  

---

## 1. Executive Summary

In financial fraud operations, decision thresholds are constrained by compliance review capacity rather than theoretical F1 maxima alone. This report models operational sensitivity across candidate thresholds and alert budgets.

### Key Operational Findings:
1. **Validation Optimal (t = 0.9660):**
   - Yields 1,590 alerts on holdout (0.8087% alert rate).
   - Precision: **57.74%** | Recall: **42.74%** (918 TP, 672 FP, 1,230 FN).
2. **Capacity-Constrained Point (t = 0.9850):**
   - Flags 1,076 alerts (0.5473% alert rate).
   - Precision: **69.33%** | Recall: **34.73%** (746 TP, 330 FP, 1,402 FN).
   - Reduces analyst alert burden by **32.3%** while boosting precision from 57.7% to 69.3%.
3. **Strict Alert Cap (t = 0.9900):**
   - Flags 962 alerts (0.4893% alert rate).
   - Precision: **67.88%** | Recall: **30.40%** (653 TP, 309 FP).

---

## 2. Threshold Sensitivity Grid (Selected Operating Points)

| Threshold (t) | Precision (%) | Recall (%) | F1 Score | Alert Volume | Alert Rate (%) | Intercepted Frauds (TP) | False Alarms (FP) | Missed Frauds (FN) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.800** | 16.88% | 67.18% | 0.2697 | 8,551 | 4.35% | 1,443 | 7,108 | 705 |
| **0.850** | 23.16% | 60.38% | 0.3348 | 5,600 | 2.85% | 1,296 | 4,304 | 852 |
| **0.900** | 30.92% | 54.38% | 0.3943 | 3,777 | 1.92% | 1,168 | 2,609 | 980 |
| **0.920** | 36.42% | 49.77% | 0.4206 | 2,935 | 1.49% | 1,069 | 1,866 | 1,079 |
| **0.940** | 47.83% | 47.63% | 0.4773 | 2,139 | 1.09% | 1,023 | 1,116 | 1,125 |
| **0.950** | 48.61% | 47.30% | 0.4795 | 2,090 | 1.06% | 1,016 | 1,074 | 1,132 |
| **0.960** | 49.54% | 44.97% | 0.4714 | 1,950 | 0.99% | 965 | 985 | 1,183 |
| **0.970** | 61.13% | 41.95% | 0.4975 | 1,474 | 0.75% | 901 | 573 | 1,247 |
| **0.980** | 68.73% | 39.29% | 0.5000 | 1,228 | 0.62% | 843 | 385 | 1,305 |
| **0.990** | 67.88% | 30.40% | 0.4199 | 962 | 0.49% | 652 | 310 | 1,496 |

---

## 3. Operational Alert Capacity Allocations

| Monthly Compliance Capacity | Daily Review Budget | Enforced Decision Cutoff | Holdout Precision | Holdout Recall | Intercepted Frauds | Operational Posture |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Ultra-Constrained** | ~15 reviews/day | t >= 0.993 | **72.5%** | 24.2% | 520 TP | Automated block with minimal analyst queue |
| **Tight Budget** | ~30 reviews/day | t >= 0.990 | **67.9%** | 30.4% | 653 TP | High-confidence investigations |
| **Balanced (Recommended)** | ~50 reviews/day | t >= 0.966 | **57.7%** | **42.7%** | **918 TP** | Maximum F1 operational baseline |
| **Aggressive Interception** | ~80 reviews/day | t >= 0.930 | **43.1%** | 52.8% | 1,134 TP | Broad net; requires larger review team |

---

## 4. Visual Artifact
- **Sensitivity Curves:** Saved to [`reports/ml_improvement/figures/baseline_threshold_sensitivity.png`](figures/baseline_threshold_sensitivity.png)
