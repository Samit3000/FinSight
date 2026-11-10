# FinSight — Final Probability Calibration Review & Score Semantics

**Document Status:** Final Frozen  
**Date:** 2026-10-19  
**Stage:** Phase D — Verify Calibration and Score Semantics  
**Authoritative Model:** [`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib)  

---

## 1. Executive Summary

This review establishes the mathematical and operational semantics of all risk outputs produced by the FinSight platform. A rigorous audit of probability calibration across the validation (`451 <= step <= 550`) and holdout (`step >= 551`) partitions confirms that:
1. **Model Outputs Are Uncalibrated Ranking Indices:** Due to training under `class_weight='balanced'`, the model's loss function applies an artificial margin offset of $\ln(1,189) \approx +7.08$ log-odds. Consequently, a predicted score of $0.95$ corresponds to an empirical validation fraud prevalence of only $2.07\%$.
2. **Static Probability Calibration Is Infeasible Under Prevalence Drift:** Fraud prevalence surges $13\times$ across the chronological partitions (from $0.0841\%$ in training to $0.6662\%$ in validation and $1.0926\%$ in holdout). Any parametric calibration mapping (e.g., Platt scaling or isotonic regression) fit on validation data would systematically miscalibrate on holdout due to non-stationary baseline priors.
3. **Formal Architectural Decision: Retain Uncalibrated Scores with Explicit Labels:** Rather than forcing an unstable probability calibration that creates false precision for analysts, the platform permanently retains uncalibrated ranking scores and enforces strict semantic separation between ranking scores, decision thresholds, and heuristic rule points.

---

## 2. Empirical Calibration Diagnostics

### 2.1 Validation Partition Metrics (`451 <= step <= 550`)
- **Total Validation Records:** $151,302$ transactions ($1,008$ frauds; true prevalence: $0.6662\%$).
- **Raw Score Brier Score:** $0.3376$ (Substantially worse than the uninformative zero-baseline Brier score of $0.0066$ due to balanced class weighting inflating raw scores).
- **Log Loss:** $1.0971$ (reflects probability miscalibration on raw margins).
- **Ranking Metrics (Unaffected by Monotonic Margin Shifts):**
  - **Average Precision (PR-AUC):** $0.3115$ (strong discriminative ranking).
  - **ROC-AUC:** $0.9509$ (excellent separation of positives from negatives).

### 2.2 Score Decile Reliability Table (Validation Partition)

| Model Output Score Bin | Total Transactions | Observed Frauds | Empirical Fraud Rate (%) | Ratio of Stated Score to Reality |
| :---: | :---: | :---: | :---: | :---: |
| $[0.00, 0.10)$ | 134,812 | 114 | 0.0846% | — |
| $[0.10, 0.30)$ | 8,940 | 182 | 2.0358% | ~10x over-stated |
| $[0.30, 0.50)$ | 2,815 | 86 | 3.0551% | ~13x over-stated |
| $[0.50, 0.70)$ | 1,842 | 92 | 4.9946% | ~12x over-stated |
| $[0.70, 0.90)$ | 1,321 | 99 | 7.4943% | ~11x over-stated |
| $[0.90, 0.95)$ | 520 | 51 | 9.8077% | ~9x over-stated |
| $[0.95, 0.98)$ | 642 | 116 | 18.0685% | ~5x over-stated |
| $[0.98, 1.00]$ | 410 | 268 | 65.3659% | ~1.5x over-stated |

**Core Diagnostic Takeaway:** Monotonicity is preserved perfectly—higher model scores consistently reflect higher true fraud density. However, the score scale is shifted upward by ~10x due to balanced training weights. A score of $0.90$ is NOT a 90% probability of fraud; it is a high-ranking percentile indicating the top ~1% most suspicious transaction profile.

---

## 3. Four Distinct Platform Risk Concepts

To prevent misleading communication in dashboards, SQL queries, and executive reports, the platform establishes four non-interchangeable risk concepts:

```
+---------------------------------------------------------------------------------------+
|                               FOUR DISTINCT RISK CONCEPTS                             |
+--------------------------+-----------------------+------------------------------------+
| Risk Concept             | Scale / Data Type     | Mathematical & Operational Meaning |
+--------------------------+-----------------------+------------------------------------+
| 1. Model Ranking Score   | Continuous Float      | Monotonic risk index derived from  |
|                          | [0.0000, 1.0000]      | gradient-boosted trees. Used for   |
|                          |                       | relative sorting, NOT probability. |
+--------------------------+-----------------------+------------------------------------+
| 2. Decision Threshold    | Discrete Cutoff       | Operational triage rule (t=0.9660) |
|    Flag                  | Binary (0 or 1)       | separating high-risk alerts from   |
|                          |                       | standard review candidates.        |
+--------------------------+-----------------------+------------------------------------+
| 3. Rule-Engine Score     | Integer Points        | Deterministic heuristic score      |
|                          | [0, 100]              | summing triggered expert rules     |
|                          |                       | (Rule 001–004).                    |
+--------------------------+-----------------------+------------------------------------+
| 4. Calibrated Posterior  | Probability           | Theoretical true Bayesian rate.    |
|    Probability           | [0.0000, 1.0000]      | UNCALIBRATED in FinSight due to    |
|                          |                       | non-stationary prevalence shift.   |
+--------------------------+-----------------------+------------------------------------+
```

### 3.1 Strict Vocabulary Guidelines:
1. **NEVER** display model output scores as "Fraud Probability: 96.6%". Always label them as `Model Risk Score: 0.9660` or `Risk Percentile Index`.
2. **NEVER** treat an alert threshold flag as "Confirmed Fraud". Flags represent `Review Candidates` or `Investigative Alerts`.
3. **NEVER** present `AUTO_APPROVE` as a guarantee that a transaction is harmless. It denotes an un-flagged, sub-threshold transaction within operational tolerance.
4. **NEVER** add rule points (0–100) to model scores (0.0–1.0). They represent fundamentally different measurement systems.

---

## 4. Operational Alert Tiering

For business intelligence dashboards and operational workflows, the continuous model score is mapped into three standardized triage tiers based on the frozen validation threshold ($t = 0.9660$):

| Tier Name | Score Range | Operational Definition | Recommended Workflow |
| :--- | :---: | :--- | :--- |
| `LOW_RISK_SCORE` | $[0.0000, 0.8500)$ | Standard commercial traffic exhibiting strong safe-rail signals (`PAYMENT`, `CASH_IN`). | Automated straight-through processing (STP). |
| `REVIEW_CANDIDATE` | $[0.8500, 0.9660)$ | Elevated risk profile (e.g. moderate transfer amounts, off-peak hours) below primary alert capacity. | Secondary audit sampling; passive monitoring. |
| `HIGH_RISK_SCORE` | $[0.9660, 1.0000]$ | Critical risk priority exceeding the frozen decision threshold. Top ~0.8% of transaction volume. | Mandatory investigator review; manual queue assignment. |
