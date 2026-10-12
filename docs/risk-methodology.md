# FinSight — Risk Methodology & Scoring Engine Specification

**Document Version:** 1.0.0  
**Domain:** Financial Fraud Risk & Transaction Monitoring  
**Target Engine:** Explainable Rule-Based Evaluation  

---

## 1. Executive Philosophy: Explainable Risk vs. Black-Box AI

In high-consequence financial services, every risk decision carries legal, regulatory, and customer trust implications. Regulatory frameworks (such as FCRA, GDPR Article 22, and AML/BSA guidelines) require financial institutions to provide **adverse action notices** and clear explanations when transactions are blocked or accounts restricted.

While black-box machine learning models (e.g., deep neural networks or dense gradient-boosted ensembles) can detect complex non-linear interactions, they suffer from two major operational deficiencies in early-stage payment operations:
1. **Lack of Direct Auditability**: They cannot deterministically explain to an operations analyst *why* a transaction was flagged beyond vague feature-attribution approximations (like SHAP).
2. **Cold Start & Label Lag**: Supervised ML models require labeled historical chargeback data, which typically lags transaction authorization by 60 to 90 days.

**FinSight implements an Explainable, Configurable Rule-Based Scoring Engine**. This methodology guarantees:
- **100% Deterministic Reproducibility**: Given the same inputs, the engine will always produce the identical score and reason trace.
- **Human-Readable Audit Trails**: Every score component is explicitly mapped to a business rule, triggered parameter, and plain-English explanation.
- **Dynamic Configuration**: Risk policies, point weights, and thresholds are externalized in `config/risk_rules.yaml` without modifying backend Python code.

---

## 2. Composite Scoring Framework

Transactions start with a baseline risk score of `0` and accumulate weighted risk points as specific anomalous behaviors trigger individual rules.

### 2.1. Score Range & Cutoffs

$$\text{Risk Score} = \min\left(100, \sum_{i \in \text{Triggered Rules}} \text{Risk Points}_i \right)$$

| Score Range | Risk Tier | Recommended Operational Action | Customer Experience Impact |
| :---: | :---: | :---: | :--- |
| **0 – 29** | **LOW** | `AUTO_APPROVE` | Frictionless instant settlement. |
| **30 – 69** | **MEDIUM** | `MANUAL_REVIEW` | Routed to analyst review queue or step-up authentication (3DS / OTP). |
| **70 – 100** | **HIGH** | `FLAG_SUSPICIOUS` | Immediate alert raised; transaction isolated for investigation. |

---

## 3. Core Rule Specifications

### 3.1. `RULE_001_HIGH_VALUE` (Excessive Single Transaction Amount)
- **Category**: Value Anomaly
- **Severity**: High (35 Points)
- **Mathematical Logic**:
  $$\text{Triggered} \iff \text{amount} > \text{threshold\_amount} \quad (\text{Default: } \$5,000.00)$$
- **FinTech Rationale**: Digital payment networks observe log-normal transaction distributions where over 95% of payments fall below $250. Outlier amounts represent outsized chargeback exposure and require operational verification.
- **Reason Template**: `"Transaction amount ${amount} exceeds the single transaction monitoring ceiling of ${threshold_amount}."`

---

### 3.2. `RULE_002_HIGH_VELOCITY` (Short-Window Velocity Spike)
- **Category**: Velocity / Account Takeover
- **Severity**: High (40 Points)
- **Mathematical Logic**:
  $$\text{Triggered} \iff \text{Count}(\text{TXN}_{\text{customer}} \text{ within } [t - \Delta t, t]) > N_{\text{threshold}}$$
  *(Defaults: $\Delta t = 10 \text{ minutes}$, $N = 3 \text{ transactions}$)*
- **FinTech Rationale**: Rapid-fire payment bursts are standard signatures of automated bot scripts, script kiddie card testing, or sudden account takeover (ATO) prior to cardholder cancellation.
- **Reason Template**: `"Account initiated {tx_count} transactions within {window_minutes} minutes, breaching the velocity threshold of {max_transaction_count}."`

---

### 3.3. `RULE_003_RAPID_GEO_HOP` (Impossible Travel Velocity)
- **Category**: Geographic Discrepancy
- **Severity**: Critical (50 Points)
- **Mathematical Logic**:
  $$\text{Triggered} \iff (\text{country}_t \neq \text{country}_{t-1}) \land (\text{timestamp}_t - \text{timestamp}_{t-1} < \Delta t_{\text{geo}})$$
  *(Default: $\Delta t_{\text{geo}} = 2 \text{ hours}$)*
- **FinTech Rationale**: A payment card physically processed in London followed 45 minutes later by a payment in Singapore represents physical impossibility ("impossible transit"). This is a primary indicator of cloned card data or compromised digital credentials.
- **Reason Template**: `"Transactions detected in different countries ({prev_country} -> {current_country}) within {elapsed_hours} hours."`

---

### 3.4. `RULE_004_HIGH_RISK_MCC` (Monitored Merchant Categories)
- **Category**: Merchant Risk Exposure
- **Severity**: Medium (25 Points)
- **Mathematical Logic**:
  $$\text{Triggered} \iff \text{mcc} \in \{\text{High-Risk Registry}\}$$
  *(Default Codes: `7995` Gambling, `6051` Crypto/Foreign Currency, `4829` Wire Transfers, `5993` Tobacco)*
- **FinTech Rationale**: Merchant Category Codes (MCCs) designate business types. Certain industries carry statistically elevated chargeback rates, money laundering risks, or legal restrictions.
- **Reason Template**: `"Merchant Category Code {mcc} matches high-risk monitoring registry ({mcc_description})."`

---

### 3.5. `RULE_005_CONSECUTIVE_DECLINES` (Repeated Declines Preceding Approval)
- **Category**: Behavioral / Card Testing
- **Severity**: High (30 Points)
- **Mathematical Logic**:
  $$\text{Triggered} \iff (\text{status}_t = \text{'SUCCESS'}) \land (\text{Count}(\text{DECLINED}_{\text{customer}} \text{ within } [t - 30\text{m}, t]) \ge 2)$$
- **FinTech Rationale**: Fraudsters often test stolen credentials by trying multiple incorrect CVVs or expiration dates until achieving a valid authorization. When a success follows multiple rapid declines, the probability of fraudulent compromise is exceptionally high.
- **Reason Template**: `"Successful transaction preceded by {failure_count} declined attempts within {window_minutes} minutes."`

---

## 4. Reason Code & Audit Structure

Every evaluation outputs structured metadata into PostgreSQL table `fct_risk_evaluations`:

```json
{
  "transaction_id": "TXN_8492019",
  "risk_score": 75,
  "risk_tier": "HIGH",
  "action_recommended": "FLAG_SUSPICIOUS",
  "triggered_rule_count": 2,
  "triggered_rules": ["RULE_001_HIGH_VALUE", "RULE_002_HIGH_VELOCITY"],
  "reason_codes": [
    {
      "rule_id": "RULE_001_HIGH_VALUE",
      "severity": "HIGH",
      "points": 35,
      "message": "Transaction amount $7,200.00 exceeds the single transaction monitoring ceiling of $5,000.00."
    },
    {
      "rule_id": "RULE_002_HIGH_VELOCITY",
      "severity": "HIGH",
      "points": 40,
      "message": "Account initiated 4 transactions within 10 minutes, breaching the velocity threshold of 3."
    }
  ]
}
```

---

## 5. Governance & Policy Lifecycle

1. **Version Control**: Rule parameters and thresholds live in `config/risk_rules.yaml`. Any change to weights or thresholds is tracked as a Git commit with documented approval.
2. **Backtesting & Simulation**: Before modifying rule thresholds in production, proposed YAML configurations can be run against historical benchmark datasets in `data/sample/` to measure false positive impact and flag volume shift.
3. **Auditability**: Regulatory inspectors can query any historical transaction and review the exact parameter values and rules that were active at the time of evaluation.
