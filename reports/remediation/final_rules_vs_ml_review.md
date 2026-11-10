# FinSight — Final Rule Engine vs. Machine Learning Benchmark Review

**Document Status:** Final Frozen  
**Date:** 2026-10-19  
**Stage:** Phase F — Validate Rule-Engine and ML Comparisons  
**Evaluation Population:** Unseen Final Holdout (`step >= 551`, $N = 196,601$, $2,148$ simulated frauds)  

---

## 1. Executive Summary

This report establishes the empirical validation of the FinSight heuristic risk engine ([`src/finsight/risk_rules.py`](file:///src/finsight/risk_rules.py)) and provides an objective, head-to-head benchmark against the supervised machine learning estimators on an identical, frozen holdout population.

### Key Audit Conclusions:
1. **Empirical Grounding of the Overnight Rule (RULE_003):** 
   - A full dataset census of all 6,362,620 transactions confirms that transaction volume drops from a peak of $647,814$ transactions at Hour 18 (6 PM) to a low of $1,241$ transactions at Hour 3 (3 AM)—an empirical decline of **$99.81\%$**.
   - Concurrently, fraud rate surges from $0.0528\%$ at 6 PM to $22.30\%$ at 4 AM—a **$422\times$ concentration factor**.
   - *Simulated-Time Caveat:* While empirically grounded in the PaySim dataset, `step` represents discrete simulation ticks, not a real-world clock timestamp or geo-located timezone.
2. **Complementary Architectural Roles:**
   - **Rule Engine `HIGH` Tier (Score $\ge 70$):** Operates as a surgical, high-precision filter ($72.04\%$ precision, $18.72\%$ recall, $558$ alerts). Highly effective for instant auto-flagging of unambiguous high-value drainage events.
   - **HistGradientBoosting (`hist_gradient_boosting_no_day.joblib` at $t = 0.9660$):** Operates as a broad, high-recall detector ($57.74\%$ precision, $42.74\%$ recall, $1,590$ alerts, capturing $918$ frauds vs. the rule engine's $402$).
   - **Conclusion:** Neither system is "categorically superior." They represent different operating points along the precision/recall Pareto frontier.

---

## 2. Heuristic Rule Definitions & Empirical Accuracy

| Rule ID | Rule Name | Target Rail & Condition | Heuristic Risk Points | Individual Rule Precision (Holdout) | Individual Rule Recall (Holdout) | Individual Alerts (Holdout) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `RULE_001` | `HIGH_VALUE_TRANSACTION` | `amount > 200,000` | +25 pts | 8.87% | 76.54% | 18,531 |
| `RULE_002` | `DRAINAGE_PAYMENT_RAIL` | `type in ('TRANSFER', 'CASH_OUT')` | +30 pts | 4.39% | **100.00%** | 48,930 |
| `RULE_003` | `OVERNIGHT_DRAINAGE` | `hour in (1,2,3,4,5)` AND `type in ('TRANSFER', 'CASH_OUT')` | +25 pts | 59.57% | 15.46% | 554 |
| `RULE_004` | `LARGE_ROUND_AMOUNT` | `amount >= 100,000` AND `amount % 10,000 == 0` | +20 pts | **80.92%** | 12.38% | 325 |

### 2.1 Empirical Verification of Diurnal Volume Collapse (All 6.36M Rows)
The rule engine's documentation states: *"Overnight transaction volume drops by over 99%."* To verify this claim with scientific rigor:
- **Peak Volume Window (Hour 18 / 6 PM):** $647,814$ transactions, $342$ frauds ($0.0528\%$ fraud rate).
- **Trough Volume Window (Hour 3 / 3 AM):** $1,241$ transactions, $249$ frauds ($20.06\%$ fraud rate).
- **Lowest Peak Volume (Hour 4 / 4 AM):** $1,341$ transactions, $299$ frauds ($22.30\%$ fraud rate).
- **Mathematical Drop:** $\frac{647,814 - 1,241}{647,814} = 99.8084\% \approx \mathbf{99.81\%}$.
- **Verification Verdict:** **FULLY VERIFIED.** The claim of a $>99\%$ drop is empirically accurate for PaySim simulation ticks.

---

## 3. Head-to-Head Benchmark on Identical Holdout Data

Evaluated across the final holdout partition (`step >= 551`, $N = 196,601$, $2,148$ simulated frauds):

| Detection System | Operating Decision Rule | Total Alerts | Alert Rate (%) | Precision (%) | Recall (%) | F1 Score | True Positives (TP) | False Positives (FP) | False Negatives (FN) | True Negatives (TN) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy Majority** | Always predict 0 ($t=0.5$) | 0 | 0.00% | 0.00% | 0.00% | 0.0000 | 0 | 0 | 2,148 | 194,453 |
| **Heuristic Rule Engine** | `HIGH` Tier (Score $\ge 70$) | **558** | **0.28%** | **72.04%** | 18.72% | 0.2971 | 402 | 156 | 1,746 | 194,297 |
| **Heuristic Rule Engine** | `MED`+`HIGH` Tier (Score $\ge 30$) | 48,930 | 24.89% | 3.51% | **79.93%** | 0.0673 | 1,717 | 47,213 | 431 | 147,240 |
| **Logistic Regression** | $t = 0.9009$ (Val F1) | 2,576 | 1.31% | 24.30% | 29.14% | 0.2650 | 626 | 1,950 | 1,522 | 192,503 |
| **HistGradientBoosting (Baseline)** | $t = 0.9900$ (Constrained) | 1,068 | 0.54% | 69.01% | 34.31% | 0.4583 | 737 | 331 | 1,411 | 194,122 |
| **HistGradientBoosting (No Day)** | $t = 0.9660$ (Val F1) | 1,590 | 0.81% | 57.74% | 42.74% | **0.4912** | **918** | 672 | 1,230 | 193,781 |
| **HistGradientBoosting (No Day)** | $t = 0.9850$ (High Precision) | 1,392 | 0.71% | 62.43% | 40.46% | 0.4912 | 869 | 523 | 1,279 | 193,930 |

---

## 4. Key Comparative Deductions

1. **Precision vs. Recall Trade-Off:**
   - The Rule Engine `HIGH` tier achieves the highest precision in the entire platform (**$72.04\%$**), but captures only $402$ out of $2,148$ frauds ($18.72\%$ recall).
   - The No-Day HGB model at $t = 0.9660$ captures **$918$ frauds** ($42.74\%$ recall, more than double the rule engine) while maintaining an operationally solid precision of **$57.74\%$**.
2. **Operational Alert Burden:**
   - Relaxing the Rule Engine to `MED`+`HIGH` (Score $\ge 30$) increases recall to $79.93\%$, but floods investigation queues with **$48,930$ alerts** ($96.5\%$ false-positive rate), which would paralyze a real compliance operations team.
   - The machine learning model provides superior non-linear triage, flagging only $1,590$ transactions to achieve $918$ true detections.
3. **Semantic Distinction:**
   - The Rule Engine score is an integer sum of deterministic rule triggers (0–100).
   - The Machine Learning score is a continuous monotonic risk index ($0.0\text{--}1.0$).
   - A transaction scoring $\ge 70$ points in the rule engine is NOT a confirmed fraud case, and a score $< 30$ (`AUTO_APPROVE`) is NOT an absolute guarantee of safety.
