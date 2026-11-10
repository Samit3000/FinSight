# FinSight — Feature Quality, Leakage & Plausibility Audit (Phase 3)

**Audit Execution Date:** 2026-10-19  
**Auditor:** Senior ML Engineer & Data Quality Auditor  
**Scope:** Predictor evaluation, leakage prevention boundaries, balance field assessment, and representation semantics  

---

## 1. Comprehensive Predictor Review

Every candidate and engineered predictor in FinSight was audited against scoring-time availability, behavioral plausibility, and leakage risk:

| Feature Name | Source / Derivation | Available at Pre-Auth Scoring? | Predictive Rationale | Leakage & Shortcut Risk | Governance Decision |
| :--- | :--- | :---: | :--- | :--- | :--- |
| **`amount`** | Raw transaction dollar value | Yes | Fraud liquidations target large capital sums (median: $\$441\text{k}$ vs $\$75\text{k}$ for legitimate). | Low risk. Core financial magnitude signal. Verified non-negative. | **Retained** |
| **`log1p_amount`** | $\ln(1 + \text{amount})$ | Yes | Compresses extreme right-tailed financial skew ($\max = \$92.4\text{M}$); stabilizes gradient boosting tree split points. | Pearson $r = 0.3873$ with raw amount; non-linear scaling improves numeric tree binning. | **Retained** |
| **`transaction_hour`** | $(\text{step} - 1) \bmod 24$ | Yes | Human retail payments drop overnight; automated liquidation scripts run 24/7. | Low risk. Stationary diurnal cycle (0–23). | **Retained** |
| **`transaction_day`** | $((\text{step} - 1) // 24) + 1$ | Yes | Tracked day of simulation (1–31). | **Severe simulator shortcut risk.** Fraud is constant ($\sim 260$/day) but background volume drops by $>95\%$ in days 18–31. Tree models memorized calendar days. | **Ablated** |
| **`type_TRANSFER`** | One-hot binary (0/1) | Yes | Primary exfiltration rail (49.9% of simulated fraud cases). | Categorical encoding of primary payment channel. | **Retained** |
| **`type_CASH_OUT`** | One-hot binary (0/1) | Yes | Outbound cash withdrawal rail (50.1% of simulated fraud cases). | Categorical encoding of cash conversion channel. | **Retained** |
| **`type_PAYMENT`** | One-hot binary (0/1) | Yes | POS merchant payments (0% fraud in PaySim). | Strong negative non-fraud suppression signal. | **Retained** |
| **`type_CASH_IN`** | One-hot binary (0/1) | Yes | Inbound cash deposits (0% fraud in PaySim). | Strong negative non-fraud suppression signal. | **Retained** |
| **`type_DEBIT`** | One-hot binary (0/1) | Yes | Direct bank debit rail (0% fraud in PaySim). | Baseline bank transfer rail. | **Retained** |
| **`is_merchant_dest`** | $1 \text{ if } \text{nameDest}[0] == \text{'M'} \text{ else } 0$ | Yes | Indicates merchant destination. | **Collinear with `type_PAYMENT` in PaySim.** $100\%$ collinearity with zero marginal PR-AUC ($\Delta\text{PR-AUC} = 0.0000$). | **Ablated** |

---

## 2. Prohibited & Leakage Columns Audit

The following columns are strictly prohibited from entering feature matrix $X$:

```text
PROHIBITED_FEATURE_COLUMNS = (
    "isFraud", "is_fraud",
    "isFlaggedFraud", "is_flagged_fraud",
    "nameOrig", "name_orig",
    "nameDest", "name_dest",
    "oldbalanceOrg", "oldbalance_org",
    "newbalanceOrig", "newbalance_orig",
    "oldbalanceDest", "oldbalance_dest",
    "newbalanceDest", "newbalance_dest",
    "transaction_id", "created_at",
)
```

### Forensic Risk Rationale:
1. **`is_fraud` (Target Label):** Ground truth. Direct inclusion is $100\%$ target leakage.
2. **`is_flagged_fraud` (Pre-existing Simulator Rule):** A crude rule (`TRANSFER` $> 200,000$) that catches only 16 frauds and misses 8,197 frauds. Inclusion creates circular evaluation leakage.
3. **`name_orig` & `name_dest` (Raw Account IDs):** $99.85\%$ of origin accounts appear only once. High-cardinality encoding memorizes training account strings without generalizable behavioral signal.
4. **`transaction_id` & `created_at` (Database Metadata):** Surrogate auto-incrementing database integers and ingestion timestamps contain zero economic signal.

---

## 3. Balance Field Forensic Assessment: Why Balance Features Must Remain Prohibited

A dedicated empirical investigation evaluated whether balance-derived features (e.g. `oldbalanceOrg`, `newbalanceOrig`, balance differences) should be admitted. The data reveals severe synthetic artifacts:

### A. Origin Liquidation Artifact
- In PaySim, fraudulent bot agents execute a script that liquidates **100% of available origin funds**:
  - **`newbalanceOrig == 0.00`** occurs in **98.05%** of all fraud transactions, compared to 56.68% for legitimate transactions.
  - **Exact liquidation ($\text{oldbalanceOrg} == \text{amount}$)** occurs in **55.85%** of all fraud transactions, compared to **0.00%** (zero occurrences out of $6,354,407$ transactions) for legitimate transactions!
- A model given `oldbalanceOrg` and `newbalanceOrig` achieves near-perfect score simply by checking `oldbalanceOrg == amount`. However, in real-world retail banking, fraudsters rarely liquidate accounts to the exact penny, and legitimate overdraft/zero-balance accounts are frequent.

### B. Timing & Prediction-Time Availability
- At transaction pre-authorization time (scoring time), the post-transaction balance (`newbalanceOrig`, `newbalanceDest`) has **not yet occurred**. Using `newbalanceOrig` relies on future post-settlement state.

### C. Zero-Balance Origin Anomalies in PaySim
- **33.09%** of legitimate transactions in PaySim begin with `oldbalanceOrg == 0.00` (even for `CASH_OUT` and `TRANSFER`, which in real life would fail NSF checks or require overdraft lines).
- In contrast, only **0.50%** of simulated fraud transactions start with zero balance.

**Governance Conclusion:** Including balance features causes models to exploit simulator-specific liquidation identities rather than learning robust fraud signals. Balance features **must remain strictly prohibited**.

---

## 4. Amount Feature Importance Investigation

The exploratory SHAP report assigned a relatively modest global mean $| \text{SHAP} |$ to `amount` ($2.31\%$) and `log1p_amount` ($1.13\%$). Our audit reconciled why:

1. **Heavy Skew in Representative Holdout Samples:**
   - In a random holdout sample ($N = 2,000$), $>98\%$ of rows are legitimate transactions.
   - Legitimate transaction amounts have a median of $\$74,685$ and 75th percentile of $\$208,365$.
   - For transactions below $\$200\text{k}$, `amount` attributions are modest (mean $| \text{SHAP} | \approx 0.13$).
2. **Tail Escalation Role:**
   - Fraudulent transactions have a median amount of $\$441,423$ and 90th percentile of $\$4,521,724$ ($12.4\times$ higher than legitimate).
   - On high-value cash-outs ($> \$1\text{M}$), `amount` contributes up to $+1.309$ in log-odds, acting as the decisive catalyst pushing the score past the $0.98\text{--}0.99$ decision threshold.
3. **Controlled Ablation Impact:**
   - In our Phase 2 controlled ablation (`model_7_no_amount`), completely dropping `amount` and `log1p_amount` caused:
     - **Validation PR-AUC to crash from $0.2519 \to 0.1117$ (-55.7%)**.
     - **Holdout PR-AUC to collapse from $0.3836 \to 0.2197$ (-42.7%)**.
     - **Holdout F1 to plunge from $0.4767 \to 0.3084$ (-35.3%)**.
   - This proves that while global mean SHAP on a population sample is modest, amount is **paramount for model discrimination**.

---

## 5. One-Hot Transaction Type Encoding Semantics

The five one-hot features (`type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`) represent mutually exclusive levels of a single categorical variable:

1. **Grouped Importance:**
   - Combined, the 5 transaction rail features account for **69.59%** of total SHAP attribution in the baseline model.
2. **Encoding Semantics vs. Real-World Events:**
   - In local explanations, an attribution for `type_PAYMENT = 0` must **never** be interpreted as a distinct criminal event. It simply means the transaction was executed on a rail other than PAYMENT.
   - Because `PAYMENT` transactions in PaySim have 0% simulated fraud, the model learns a large negative attribution for `type_PAYMENT = 1`. Consequently, when `type_PAYMENT = 0`, the transaction is not suppressed by this negative adjustment, shifting the relative log-odds toward the baseline.
3. **Human-Readable Presentation:**
   - Local explanation audit records now explicitly decouple observed properties (`observed_attributes: { "transaction_type": "TRANSFER", "amount_usd": 6270000.0 }`) from one-hot indicator mechanics.
