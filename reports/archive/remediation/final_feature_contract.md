# FinSight — Final Feature Contract & Leakage Audit

**Document Status:** Final Frozen  
**Date:** 2026-10-19  
**Stage:** Phase C — Freeze the Feature Set and Recheck Shortcut Risk  
**Authoritative Model:** [`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib) (8 features)  

---

## 1. Executive Summary

This feature contract serves as the single source of truth for all machine learning feature extraction, scoring-time inference, and analytical marts across the FinSight platform. Every candidate feature from the PaySim dataset was audited against three mandatory criteria:
1. **Prediction-Time Availability:** Can the feature be known before transaction authorization?
2. **Target & Simulator Leakage:** Does the feature incorporate post-transaction settlement outcomes or simulator-specific accounting identities?
3. **Generalizability vs. Shortcut Risk:** Does the feature reflect genuine behavioral patterns rather than artificial data generation shortcuts?

Based on empirical controlled ablations across all 6.36M transactions:
- **`transaction_day` is permanently excluded:** It acts as an overfitted calendar-day shortcut in PaySim due to an artificial 95% volume drop post-day 17. Dropping it boosts Holdout PR-AUC by $+16.8\%$ ($0.3836 \to 0.4479$).
- **`is_merchant_dest` is permanently excluded:** It is perfectly collinear with `type_PAYMENT` (where fraud is 0%) and provides zero marginal predictive power ($\Delta\text{PR-AUC} = +0.0000$).
- **Balance fields are permanently excluded:** They exhibit post-transaction lookahead and hardcoded liquidation identities ($98.05\%$ zero-balance liquidation in fraud vs. $0.00\%$ in legitimate transfers).
- **`transaction_hour` is frozen:** Captures diurnal human sleep/wake cycles ($99.81\%$ volume drop overnight, $422\times$ fraud concentration).
- **`amount` and `log1p_amount` are frozen:** Indispensable magnitude escalators; removing them crashes PR-AUC by $42.7\%$.

---

## 2. Definitive Feature Contract Table

| Column Name | In/Out | Exact Definition & Formula | Source Field | Available at Scoring? | Leakage & Shortcut Assessment | Operational & Statistical Rationale |
| :--- | :---: | :--- | :--- | :---: | :--- | :--- |
| `amount` | **IN (1)** | Raw transaction value in currency units ($[0.00, 92.4\text{M}]$). | `amount` | **YES** | **Zero leakage.** Pre-transaction parameter declared at authorization request. | Primary continuous scale indicator. In PaySim, fraudulent transfers average $\$1.47\text{M}$ vs $\$178\text{k}$ legitimate. Removing amount drops PR-AUC from $0.3836 \to 0.2197$ (-42.7%). |
| `log1p_amount` | **IN (2)** | Natural logarithm of amount plus one: $\ln(1 + \text{amount})$. | `amount` | **YES** | **Zero leakage.** Deterministic continuous transform of pre-transaction amount. | Compresses extreme right-skewed tail (kurtosis $1,781 \to 0.46$). Allows tree quantile binner (`max_bins=255`) to partition low-value and sub-threshold payments with high granularity without clipping multimillion-dollar heists. |
| `transaction_hour` | **IN (3)** | Diurnal simulation hour (0–23): `(step - 1) % 24`. | `step` | **YES** | **Zero leakage.** Represents current time of day at transaction arrival. | Captures diurnal human behavior. Overnight hours (1 AM–5 AM) experience a $99.81\%$ collapse in legitimate traffic, concentrating fraud prevalence to $22.30\%$ (vs $0.05\%$ at 6 PM). Dropping hour drops Holdout PR-AUC by $28.1\%$. |
| `type_CASH_IN` | **IN (4)** | Binary indicator: `1` if type is `CASH_IN`, else `0`. | `type` | **YES** | **Zero leakage.** Declared payment rail at transaction initiation. | Major non-fraud suppression rail (1.40M transactions, 0 frauds). Generates large negative TreeSHAP attribution (-1.07), safely pulling deposit scores away from alert thresholds. |
| `type_CASH_OUT` | **IN (5)** | Binary indicator: `1` if type is `CASH_OUT`, else `0`. | `type` | **YES** | **Zero leakage.** Declared payment rail at transaction initiation. | Major fraud attack rail (2.24M transactions, 4,116 frauds). Terminal stage of PaySim account draining. Critical positive prior. |
| `type_DEBIT` | **IN (6)** | Binary indicator: `1` if type is `DEBIT`, else `0`. | `type` | **YES** | **Zero leakage.** Declared payment rail at transaction initiation. | Low-volume non-fraud rail (41.4k transactions, 0 frauds). Neutral background signal ($0.35\%$ SHAP share). |
| `type_PAYMENT` | **IN (7)** | Binary indicator: `1` if type is `PAYMENT`, else `0`. | `type` | **YES** | **Zero leakage.** Declared payment rail at transaction initiation. | Dominant consumer purchase rail (2.15M transactions, 0 frauds). Strongest single risk suppressor in the platform ($33.67\%$ TreeSHAP share). |
| `type_TRANSFER` | **IN (8)** | Binary indicator: `1` if type is `TRANSFER`, else `0`. | `type` | **YES** | **Zero leakage.** Declared payment rail at transaction initiation. | Primary fraud attack rail (533k transactions, 4,097 frauds). Initial stage of account takeover. Critical positive prior. |
| `transaction_day` | **EXCLUDED** | Simulation day (1–31): `((step - 1) // 24) + 1`. | `step` | YES | **HIGH SHORTCUT RISK.** Captures simulator volume collapse post-day 17. | PaySim legitimate volume drops 95% after day 17 while fraud remains static. The model used day as an overfitted calendar shortcut. Removing it improved Holdout PR-AUC from $0.3836 \to 0.4479$ (+16.8%). |
| `is_merchant_dest` | **EXCLUDED** | Binary: `1` if `nameDest.startswith('M')`, else `0`. | `nameDest` | YES | **COLLINEAR SHORTCUT.** Perfect duplicate of `type_PAYMENT`. | PaySim only assigns merchant IDs to PAYMENT transactions. Adding it yields $\Delta\text{PR-AUC} = +0.0000$. Excluded to prevent redundant split splits and ungrounded real-world assumptions. |
| `oldbalanceOrg` | **EXCLUDED** | Sender account balance prior to transaction. | Raw Log | CONDITIONAL | **TARGET & SIMULATOR LEAKAGE.** | In PaySim, $55.85\%$ of fraud cases exhibit exact full-balance liquidation (`oldbalanceOrg == amount`) vs $0.00\%$ of legitimate transfers. Unrealistic synthetic identity. |
| `newbalanceOrig` | **EXCLUDED** | Sender account balance after transaction settlement. | Raw Log | **NO** | **SETTLEMENT LOOKAHEAD LEAKAGE.** | Value is calculated *after* transaction execution. In PaySim, $98.05\%$ of frauds end with `newbalanceOrig == 0.0`. Pure post-authorization lookahead leakage. |
| `oldbalanceDest` | **EXCLUDED** | Recipient account balance prior to transaction. | Raw Log | **NO** | **CROSS-BANK INFORMATION LEAKAGE.** | Originating bank cannot see recipient's ledger balance at authorization time. In PaySim, destination balances are 0.0 in 46% of records (simulator defect). |
| `newbalanceDest` | **EXCLUDED** | Recipient account balance after transaction settlement. | Raw Log | **NO** | **POST-SETTLEMENT LOOKAHEAD LEAKAGE.** | Settlement outcome unavailable at authorization time. |
| `isFlaggedFraud` | **EXCLUDED** | Heuristic business rule flag (`amount > 200,000`). | Raw Log | NO | **LABEL LEAKAGE.** | Pre-existing operational rule output (16 records in full dataset). Must never be used as a predictor. |
| `isFraud` | **TARGET** | Ground-truth fraud label (`0` or `1`). | Raw Log | NO | **TARGET COLUMN.** | Prediction objective. Formally prohibited from entering feature matrix. |
| `nameOrig`, `nameDest` | **EXCLUDED** | Account entity identifiers (`C...`, `M...`). | Raw Log | YES | **SPARSE IDENTIFIER RISK.** | Over 99.9% of origin accounts appear exactly once. Raw IDs cause memorization without generalization. Preserved in reference tables for investigative join queries. |

---

## 3. Coexistence of `amount` and `log1p_amount`

Although `amount` and `log1p_amount` exhibit a Pearson correlation of $r = 0.3873$ (linear) and Spearman rank correlation of $r = 1.0000$ (monotonic), both features are intentionally preserved in the frozen contract:
1. **Decision Tree Split Mechanics:** While a single decision tree on one feature is invariant to monotonic transformations, gradient-boosted ensembles with histogram binning (`max_bins=255`) construct bins differently across linear vs log space.
2. **Resolution Across Scale Regimes:**
   - Linear `amount` places nearly all 255 bins in the extreme high-value range ($\$1\text{M}\text{--}\$92\text{M}$), providing fine split resolution for large-scale liquidations.
   - Logarithmic `log1p_amount` places bins evenly across orders of magnitude ($\$10, \$100, \$1\text{k}, \$10\text{k}, \$100\text{k}$), providing fine split resolution for everyday consumer payments.
3. **Empirical Ablation Verification:** Removing both amount features caused the largest performance collapse of any ablation study (Holdout PR-AUC plummeted from $0.3836 \to 0.2197$, a $42.7\%$ drop). In TreeSHAP, they jointly contribute $6.18\%$ of model attributions ($4.47\%$ raw amount, $1.71\%$ log amount).

---

## 4. Enforcement and Validation Invariants

Feature generation is enforced by automated contract tests in [`tests/test_features.py`](file:///tests/test_features.py) and [`src/finsight/features.py`](file:///src/finsight/features.py):
1. **Strict Prohibited Column Assertion:**
   ```python
   PROHIBITED_FEATURE_COLUMNS = {
       "isFraud", "is_fraud", "isFlaggedFraud", "is_flagged_fraud",
       "oldbalanceOrg", "oldbalanceorg", "newbalanceOrig", "newbalanceorig",
       "oldbalanceDest", "oldbalancedest", "newbalanceDest", "newbalancedest",
       "nameOrig", "name_orig", "nameDest", "name_dest",
       "transaction_id", "created_at"
   }
   ```
2. **Column Order Invariant:** Slicing is executed explicitly by column name matching the model's saved feature order (`models/hist_gradient_boosting_no_day.joblib`), ensuring immunity to input DataFrame column reordering.
3. **Null and Infinite Value Rejection:** `log1p_amount` asserts `amount >= 0.0` and rejects negative or NaN values at pipeline boundary.
