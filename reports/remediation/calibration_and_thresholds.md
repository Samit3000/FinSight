# FinSight — Calibration, Alert Interpretation & Threshold Optimization (Phase 5)

**Analysis Execution Date:** 2026-10-19  
**Auditor:** Senior ML Engineer & Fraud Analytics Specialist  
**Evaluated Data:** Validation partition (`451 <= step <= 550`, $N = 151,302$, $y=1 \implies 1,008$) & Holdout partition (`step >= 551`, $N = 196,601$, $y=1 \implies 2,148$)  

---

## 1. Executive Summary

This study evaluated the calibration properties of the FinSight fraud detection models, diagnosed why model outputs cannot be interpreted as empirical probabilities, resolved candidate threshold discretization, and defined operational alert semantics for downstream consumers.

### Key Findings:
1. **Model Scores Are Radically Uncalibrated by Design:**
   - On validation data, the model exhibits a **Brier score of 0.3376** and a **Log Loss of 1.0971**.
   - When the model outputs a score of $\approx 0.950$, the actual empirical fraction of fraud in that bucket is only **$2.07\%$**.
   - **Root Cause:** The models are fitted with `class_weight='balanced'` on the training partition ($0.0841\%$ fraud prevalence). This artificially inflates the gradient updates for positive instances by a factor of $\approx 1,189:1$, shifting the model's raw decision margins upward by $\ln(1,189) \approx +7.08$ log-odds. The output of `predict_proba` is therefore an **uncalibrated ranking score**, not a posterior probability.
2. **Infeasibility of Post-Hoc Probability Calibration:**
   Because fraud prevalence experiences severe non-stationarity across chronological periods ($0.084\%$ in train $\to$ $0.666\%$ in val $\to$ $1.093\%$ in holdout), any probability calibration (e.g. Platt scaling or isotonic regression) fitted on earlier data will be invalidated by the $8\times\text{--}13\times$ change in prior class odds. Attempting to report "calibrated probabilities" would provide a dangerously false sense of precision. We firmly recommend **retaining uncalibrated ranking-score terminology**.
3. **Threshold Selection Audit & Resolution:**
   - The discrepancy where validation candidate $t = 0.985$ was noted to achieve higher F1 than $0.990$ was caused by a coarse 100-step grid in `find_optimal_threshold`, which skipped the entire interval $[0.9801, 0.9900]$.
   - Fine-grid evaluation on validation data identifies the true unconstrained F1 peak at **$t = 0.9830$** (Validation F1: **0.3826**, Precision: **0.3462**, Recall: **0.4276**, Alerts: **1,245**).
   - Under an **alert-volume constraint** ($\le 1,000$ alerts across the 4-day validation window, i.e. $\le 250$ alerts/day), the optimal threshold is **$t = 0.9900$** (Validation F1: **0.3434**, Precision: **0.3498**, Recall: **0.3373**, Alerts: **972**).
4. **Alert Semantics Defined:**
   Strict operational terminology is established: scores are risk indices, thresholds are decision gates, and flagged transactions are review candidates, never "confirmed fraud."

---

## 2. Validation Calibration Diagnostics

Evaluated on validation partition ($N = 151,302$, $1,008$ fraud cases):

### A. Summary Calibration Loss Metrics
- **Brier Score Loss:** **0.337567** (a well-calibrated baseline at 0.67% base rate should achieve Brier score $\approx 0.0066$).
- **Binary Cross-Entropy (Log Loss):** **1.097124**.

### B. Reliability Curve (Observed vs. Predicted by Decile)

| Decile Range | Mean Predicted Score | Observed Fraction True Fraud | Calibration Residual ($\text{Obs} - \text{Pred}$) | Interpretation |
| :---: | :---: | :---: | :---: | :--- |
| **0.00 – 0.10** | 0.0036 | 0.0000 | -0.0036 | Non-fraud rails accurately suppressed. |
| **0.10 – 0.20** | 0.1429 | 0.0000 | -0.1429 | Zero fraud observed in this range. |
| **0.30 – 0.40** | 0.3817 | 0.0000 | -0.3817 | Zero fraud observed in this range. |
| **0.40 – 0.50** | 0.4166 | 0.3333 | -0.0833 | Small sparse cluster. |
| **0.50 – 0.60** | 0.5716 | 0.0150 | **-0.5566** | Severe over-prediction due to class balancing. |
| **0.60 – 0.70** | 0.6611 | 0.0148 | **-0.6464** | Severe over-prediction due to class balancing. |
| **0.70 – 0.80** | 0.7464 | 0.0071 | **-0.7393** | Severe over-prediction due to class balancing. |
| **0.80 – 0.90** | 0.8569 | 0.0023 | **-0.8546** | Severe over-prediction due to class balancing. |
| **0.90 – 1.00** | 0.9502 | 0.0207 | **-0.9295** | Severe over-prediction due to class balancing. |

**The Mathematical Mechanism:**
When `class_weight='balanced'` is enabled, scikit-learn assigns weight $w_1 = \frac{N}{2 \cdot N_1} \approx 594.7$ and $w_0 = \frac{N}{2 \cdot N_0} \approx 0.5004$. The loss function optimizes as if the class distribution were 50/50. Consequently, any transaction on a high-risk rail with moderate amount receives a score above $0.80$, even though the real-world base rate is under $1\%$.

---

## 3. Threshold Optimization Sensitivity on Validation Data

Candidate threshold behavior was evaluated across three distinct operational objectives:

```text
=========================================================================================================
OBJECTIVE                 CRITERIA / CONSTRAINT         SELECTED t   PRECISION   RECALL   F1     ALERTS
=========================================================================================================
1. Unconstrained Max F1   Maximize F1 on Validation     0.9830       0.3462      0.4276   0.3826 1,245
2. Precision-Floor        Max Recall @ Precision >= 30% 0.9830       0.3462      0.4276   0.3826 1,245
3. Alert-Constrained      Max F1 @ Alerts <= 1,000      0.9900       0.3498      0.3373   0.3434   972
4. Severe Capacity Cap    Max F1 @ Alerts <= 800        0.9940       0.3400      0.2540   0.2907   753
=========================================================================================================
```

### Detailed Validation vs. Holdout Sensitivity Matrix
*(Note: Holdout metrics are displayed strictly for audit comparison; holdout labels were not used during selection).*

| Threshold ($t$) | Validation Precision | Validation Recall | Validation F1 | Validation Alerts | Holdout Precision | Holdout Recall | Holdout F1 | Holdout Alerts |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **0.9800** | 0.1487 | 0.4653 | 0.2254 | 3,153 | 0.2371 | 0.4595 | 0.3128 | 4,163 |
| **0.9820** | 0.2063 | 0.4504 | 0.2830 | 2,201 | 0.3204 | 0.4418 | 0.3715 | 2,962 |
| **0.9830** | **0.3462** | **0.4276** | **0.3826** | **1,245** | **0.5456** | **0.4232** | **0.4767** | **1,666** |
| **0.9840** | 0.3457 | 0.4266 | 0.3819 | 1,244 | 0.6078 | 0.4213 | 0.4977 | 1,489 |
| **0.9850** | 0.3414 | 0.4187 | 0.3761 | 1,236 | 0.6241 | 0.4143 | 0.4980 | 1,426 |
| **0.9880** | 0.3325 | 0.3948 | 0.3610 | 1,197 | 0.6421 | 0.3934 | 0.4879 | 1,316 |
| **0.9900** | **0.3498** | **0.3373** | **0.3434** | **972** | **0.6901** | **0.3431** | **0.4583** | **1,068** |
| **0.9920** | 0.3305 | 0.3095 | 0.3197 | 944 | 0.6952 | 0.3133 | 0.4320 | 968 |

### Primary Threshold Choice Rationale:
- **Baseline Model Retains $t = 0.9900$:**
  In the baseline 9-feature model artifact, $t = 0.9900$ represents the **alert-capacity-constrained optimum** ($\le 1,000$ alerts in validation window). It delivers a high-precision operating posture ($\text{Precision} = 0.6901$ on holdout, generating $1,068$ total alerts).
- **Corrected Threshold Search in Code:**
  The optimizer function `find_optimal_threshold` is updated to evaluate fine grids ($1,000$ steps) and support user-declared objectives (`f1`, `alert_constrained`, `recall_at_min_precision`), with deterministic tie-breaking.

---

## 4. Operational Alert Semantics & Governance Dictionary

To prevent misleading interpretations by business users, risk analysts, or auditors, the following terminology standards are enforced across FinSight:

1. **Uncalibrated Model Score ($\in [0, 1]$):**
   - **Definition:** The sigmoid transformation of the tree ensemble's raw decision margin log-odds under balanced training loss.
   - **Prohibited Phrasing:** *"The transaction has a 99% probability of fraud."*
   - **Required Phrasing:** *"The transaction received a model risk score of 0.9900 (decision margin +4.598), exceeding the review alert threshold."*
2. **Decision Threshold ($t$):**
   - **Definition:** An operational cutoff selected on validation data to trade off false alarms against missed fraud. It is a decision boundary, not a statement of confidence.
3. **Alert Tier Semantics:**
   - **`HIGH_RISK_SCORE` (or `FLAG_SUSPICIOUS`):** Transaction score meets or exceeds the alert cutoff. Classified as a **Review Candidate** for tier-2 manual investigation.
   - **`REVIEW_CANDIDATE` (or `MANUAL_REVIEW`):** Intermediate risk queue.
   - **`LOW_RISK_SCORE` (or `AUTO_APPROVE`):** Operational default action permitting straight-through processing. It is an operational queuing designation, **not a guarantee that the transaction is legitimate**.
4. **Rule Score vs. Model Score:**
   - **Rule Score ($0\text{--}100$):** Deterministic sum of triggered heuristic risk points based on business rules.
   - **Model Score ($0.0\text{--}1.0$):** Continuous statistical score produced by the machine learning estimator.
   - These two signals are maintained as distinct, complementary indicators in audit logs.
