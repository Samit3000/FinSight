# FinSight — Final SHAP Verification & Interpretability Governance

**Document Status:** Final Frozen  
**Date:** 2026-10-19  
**Stage:** Phase E — Align SHAP with the Final Model  
**Authoritative Model:** [`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib)  
**Explainer Backend:** TreeSHAP (`shap.TreeExplainer`)  

---

## 1. Executive Summary

This report establishes the complete verification, mathematical additivity proof, and governance guidelines for the Explainable AI (XAI) layer in FinSight. 

All explainability artifacts in [`reports/explainability/`](file:///reports/explainability) have been regenerated directly from the authoritative frozen champion model ([`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib)) across an unseen holdout sample ($N = 2,000$, `step >= 551`, seed 42).

### Key Verification Findings:
1. **Mathematical Additivity Proven:** TreeSHAP exact efficiency / additivity was validated on the raw decision margin (log-odds $\text{logit}(p) = \ln\frac{p}{1-p}$), with a maximum reconstruction error of **$1.07 \times 10^{-14}$** (well within machine precision $\epsilon \approx 10^{-12}$).
2. **Elimination of Temporal Shortcut:** Unlike the historical baseline where `transaction_day` absorbed $24.28\%$ of model attributions, the frozen 8-feature model contains zero calendar-day attribution, forcing the ensemble to rely entirely on transaction rails, diurnal hour, and monetary magnitude.
3. **Grouped Feature Representation:**
   - **Payment Rails (5 features):** **$88.88\%$** of total model importance.
   - **Monetary Magnitude (`amount` + `log1p_amount`):** **$6.18\%$** of total model importance.
   - **Diurnal Time (`transaction_hour`):** **$4.95\%$** of total model importance.
4. **Complete Multi-Artifact Synchronization:** Bitwise agreement verified between [`global_shap_importance.csv`](file:///reports/explainability/global_shap_importance.csv), [`explainability_metadata.json`](file:///reports/explainability/explainability_metadata.json), [`local_explanations.json`](file:///reports/explainability/local_explanations.json), and [`explainability_report.md`](file:///reports/explainability/explainability_report.md).

---

## 2. Experimental Setup & Sampling Provenance

- **Model Artifact:** [`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib)
- **Estimator Class:** `HistGradientBoostingClassifier` (50 trees, max depth unconstrained, max leaf nodes 31, balanced class weights).
- **Target Feature List & Order (8 features):**
  `['amount', 'log1p_amount', 'transaction_hour', 'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']`
- **Evaluation Partition:** Final unseen holdout (`step >= 551`).
- **Sampling Strategy:** Deterministic stratified sample of $N = 2,000$ transactions using `random_state=42`.
- **Sample Class Breakdown:** $1,983$ legitimate transactions ($99.15\%$), $17$ fraudulent transactions ($0.85\%$).
- **Base Expected Value ($\phi_0$):** $-3.416074$ log-odds (corresponds to an uncalibrated prior score of $0.031797 \approx 3.18\%$).
- **Decision Threshold:** $t = 0.9660$ ($+3.348$ log-odds margin).

---

## 3. Global Feature Importance Table

Computed as the mean absolute SHAP value across all 2,000 holdout transactions:

$$\text{Importance}(j) = \frac{1}{N} \sum_{i=1}^N |\phi_j(x^{(i)})|$$

| Rank | Feature Name | Mean Absolute SHAP (Log-Odds) | Importance Share (%) | Primary Impact Direction | Operational Interpretation |
| :---: | :--- | :---: | :---: | :--- | :--- |
| **1** | `type_PAYMENT` | 1.2372 | 33.67% | Strongly Negative (-1.24) | Dominant commercial rail with 0% fraud; strongly suppresses risk score toward zero. |
| **2** | `type_CASH_IN` | 1.0733 | 29.21% | Strongly Negative (-1.07) | Deposit rail with 0% fraud; pulls risk scores safely below decision threshold. |
| **3** | `type_CASH_OUT` | 0.6536 | 17.78% | Strongly Positive (+0.65) | Primary liquidation attack rail; elevates transaction risk toward alert status. |
| **4** | `type_TRANSFER` | 0.2891 | 7.87% | Strongly Positive (+0.29) | Primary account draining rail; elevates transaction risk toward alert status. |
| **5** | `transaction_hour` | 0.1819 | 4.95% | Bi-directional | Overnight hours (1 AM–5 AM) elevate risk; business daytime hours suppress risk. |
| **6** | `amount` | 0.1643 | 4.47% | Strongly Positive | High values ($>\$200\text{k}$) act as continuous escalators pushing cash-outs past $t=0.966$. |
| **7** | `log1p_amount` | 0.0627 | 1.71% | Bi-directional | Refines binning for small and medium-value transactions. |
| **8** | `type_DEBIT` | 0.0129 | 0.35% | Neutral / Negative | Low-volume debit rail with 0% fraud; minor safe-rail suppression. |
| **Total** | *All 8 Features* | **3.6750** | **100.00%** | — | **Exact mathematical sum of attributions.** |

---

## 4. Grouped Semantic Attribution Analysis

Because one-hot encoded variables represent mutually exclusive levels of a single underlying categorical variable (`type`), reporting individual one-hot shares can obscure business meaning. The grouped attribution perspective synthesizes individual features into cohesive risk drivers:

```
+---------------------------------------------------------------------------------------+
|                               GROUPED FEATURE IMPORTANCE                              |
+--------------------------+-----------------------+------------------------------------+
| Semantic Feature Group   | Member Features       | Grouped Importance Share           |
+--------------------------+-----------------------+------------------------------------+
| Payment Rail Types       | type_PAYMENT, CASH_IN | 88.88%                             |
|                          | CASH_OUT, TRANSFER,   | (Dominant structural prior)        |
|                          | DEBIT                 |                                    |
+--------------------------+-----------------------+------------------------------------+
| Monetary Magnitude       | amount, log1p_amount  | 6.18%                              |
|                          |                       | (Critical non-linear escalator)    |
+--------------------------+-----------------------+------------------------------------+
| Diurnal Time of Day      | transaction_hour      | 4.95%                              |
|                          |                       | (Diurnal concentration signal)     |
+--------------------------+-----------------------+------------------------------------+
```

---

## 5. Local Explanation Phrasing and Governance Standards

To ensure regulatory compliance and ethical AI practices, analysts and dashboard consumers must adhere to the following standards:

1. **Non-Causal Explanation Phrasing:**
   - **Correct:** *"The model assigned an elevated risk score of 0.9842 because the transaction occurred on the CASH_OUT rail (+1.82 log-odds) for an unusually large amount of $450,000 (+1.14 log-odds) during overnight hour 3 (+0.42 log-odds)."*
   - **Prohibited:** *"The customer committed fraud because the amount was $450,000 and they transacted at 3 AM."*
2. **One-Hot Absence Interpretation:**
   - In one-hot encoding, features like `type_PAYMENT = 0` or `type_CASH_IN = 0` appear with positive attributions on fraud transactions. Analysts must understand that `type_PAYMENT = 0` indicates the **absence of a safe commercial rail**, NOT an independent illicit event.
3. **Correlated Attributions:**
   - `amount` and `log1p_amount` share attribution for monetary scale. A modest SHAP share for raw amount does not imply the amount is unimportant; it reflects distributed credit between the linear and logarithmic bin representations.
4. **No Real-World Transfer Guarantee:**
   - SHAP attributions reflect the decision boundaries learned on PaySim synthetic logs. They do not prove that real-world fraudsters will exhibit identical signatures.
