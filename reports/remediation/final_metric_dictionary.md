# FinSight — Final Data & Financial Metric Dictionary

**Document Status:** Final Frozen  
**Date:** 2026-10-19  
**Stage:** Phase G — Freeze Data and Financial Definitions  
**Purpose:** Single Source of Truth for SQL Analytics Marts and Power BI Semantic Modeling  

---

## 1. Executive Summary

This metric dictionary establishes the authoritative, unambiguous definitions, mathematical formulas, and business semantics for all key performance indicators (KPIs), operational metrics, and financial risk measures in FinSight. 

All figures have been calculated from the 6,362,620 validated transaction records in [`data/processed/transactions.csv.gz`](file:///data/processed/transactions.csv.gz).

### Core Financial Principles: Value, Exposure, and Unobservable Loss
1. **Gross Transaction Value (GTV):** **$\$1,144,392,944,759.77$** ($\approx \$1.144\text{T}$). The total payments volume processed across all 5 payment rails ($6,362,620$ transactions).
2. **Gross Labelled Fraud-Associated Transaction Value:** **$\$12,056,415,427.84$** ($\approx \$12.06\text{B}$). The arithmetic sum of all transaction amounts bearing a simulated `is_fraud = 1` flag ($8,213$ transactions). Crucially, this reflects gross event-level exposure and includes paired simulator script legs.
3. **Fraudulent TRANSFER Transaction Value (`fraud_transfer_face_value`):** **$\$6,028,207,713.92$** ($\approx \$6.03\text{B}$). The face value of all fraudulent `TRANSFER` transactions ($4,116$ records). This represents Stage 1 (victim account extraction). It must NOT be called "net economic loss" because it merely sums transfer face values without accounting for clawbacks, recoveries, or destination freezes.
4. **Paired-Flow-Adjusted Simulated Exposure:** **$\approx \$6,028,207,713.92$** ($\approx \$6.03\text{B}$). Under the verified empirical PaySim script topology, fraudulent agents execute rigid 1:1 paired transactions (a `TRANSFER` of victim funds followed immediately by a matching `CASH_OUT` from the drop account). Summing both gross legs doubles the apparent stolen funds; adjusting for paired flows eliminates this double-counting.
5. **Actual Net Economic Loss:** **Unverifiable from synthetic labels alone.** Synthetic datasets contain no post-transaction settlement records, bank clawbacks, merchant chargebacks, insurance recovery, or frozen balance adjustments. Reporting gross fraud exposure or transfer face value as "net realized loss" is analytically indefensible.

---

## 2. Core Volume & Monetary Metrics

| Metric ID | Official Metric Name | Mathematical Definition / Formula | PaySim Population Value | Reporting Unit | Usage in Downstream Dashboards |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `MET_001` | **Total Transaction Count** | $\text{COUNT}(*)$ | **6,362,620** | Transactions | Executive scorecard baseline; denominator for throughput. |
| `MET_002` | **Gross Transaction Value (GTV)** | $\sum \text{amount}$ | **$1,144,392,944,759.77** ($1.144\text{T}$) | USD ($) | Total payments volume processed across all 5 payment rails. |
| `MET_003` | **Average Order Value (AOV)** | $\frac{\text{GTV}}{\text{Total Transactions}}$ | **$179,861.90** | USD ($) / tx | Average transaction magnitude across platform. |
| `MET_004` | **Labelled Fraud Count** | $\text{COUNT}(*) \text{ WHERE } \text{isFraud} = 1$ | **8,213** | Transactions | Ground-truth simulated fraud incidents. |
| `MET_005` | **Overall Fraud Incident Rate** | $\frac{\text{Labelled Fraud Count}}{\text{Total Transaction Count}}$ | **0.1291%** (1 in 774) | Percentage (%) | Macro platform fraud prevalence. |
| `MET_006` | **Gross Labelled Fraud-Associated Value** | $\sum \text{amount} \text{ WHERE } \text{isFraud} = 1$ | **$12,056,415,427.84** ($12.056\text{B}$) | USD ($) | Face value of all fraud-flagged transactions (event-level gross). |
| `MET_007` | **Fraud Value Exposure Share** | $\frac{\text{Gross Labelled Fraud-Associated Value}}{\text{Gross Transaction Value}}$ | **1.0535%** | Percentage (%) | Proportion of processed dollar volume involved in fraud. |
| `MET_008` | **Fraudulent TRANSFER Transaction Value** | $\sum \text{amount} \text{ WHERE } \text{isFraud} = 1 \text{ AND } \text{type} = \text{'TRANSFER'}$ | **$6,067,213,184.01** ($\approx \$6.07\text{B}$) | USD ($) | Stage-1 infiltration face value. (Column: `fraud_transfer_face_value`). |

---

## 3. Machine Learning & Operational Risk Metrics (Holdout Partition)

Evaluated across the frozen final holdout partition (`step >= 551`, $N = 196,601$, $2,148$ simulated frauds, GTV = $\$33.86\text{B}$, Fraud Exposure = $\$3.25\text{B}$):

| Metric ID | Metric Name | Definition & Formula | Frozen Champion (`no_day`, $t=0.966$) | Heuristic Rule Engine (`HIGH`) | Rationale & Interpretation |
| :--- | :--- | :--- | :---: | :---: | :--- |
| `MET_010` | **Flagged Alert Volume** | $\text{COUNT}(*) \text{ WHERE } \text{Score} \ge \text{Threshold}$ | **1,590** | **558** | Total cases routed to compliance review queue. |
| `MET_011` | **Alert Rate (%)** | $\frac{\text{Alert Volume}}{\text{Holdout Transactions}}$ | **0.8087%** | **0.2838%** | Workload demand on investigative staff. |
| `MET_012` | **Detected Labelled Fraud (TP)** | $\text{COUNT}(*) \text{ WHERE } \text{Alert}=1 \land \text{Fraud}=1$ | **918** | **402** | Fraud transactions successfully intercepted before execution. |
| `MET_013` | **False-Positive Alerts (FP)** | $\text{COUNT}(*) \text{ WHERE } \text{Alert}=1 \land \text{Fraud}=0$ | **672** | **156** | Legitimate customer payments unnecessarily flagged for review. |
| `MET_014` | **Missed Labelled Fraud (FN)** | $\text{COUNT}(*) \text{ WHERE } \text{Alert}=0 \land \text{Fraud}=1$ | **1,230** | **1,746** | Undetected fraud slipping through to straight-through processing. |
| `MET_015` | **Alert Precision (%)** | $\frac{\text{TP}}{\text{TP} + \text{FP}}$ | **57.74%** | **72.04%** | Likelihood that an investigated alert is actual fraud. |
| `MET_016` | **Fraud Capture / Recall (%)** | $\frac{\text{TP}}{\text{TP} + \text{FN}}$ | **42.74%** | **18.72%** | Share of total fraud volume intercepted. |
| `MET_017` | **F1 Score** | $2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$ | **0.4912** | **0.2971** | Harmonic mean balancing alert fatigue against fraud capture. |
| `MET_018` | **Intercepted Fraud Exposure ($)** | $\sum \text{amount} \text{ WHERE } \text{Alert}=1 \land \text{Fraud}=1$ | **$1.81\text{B}$** (55.7% of holdout fraud $) | **$0.94\text{B}$** (28.9% of holdout fraud $) | Monetary value of intercepted fraudulent transactions. |
| `MET_019` | **False-Positive Friction ($)** | $\sum \text{amount} \text{ WHERE } \text{Alert}=1 \land \text{Fraud}=0$ | **$432.8\text{M}$** | **$181.2\text{M}$** | Dollar volume of legitimate transactions delayed by review. |

---

## 4. Paired-Flow Deduplication Logic & Financial Governance

### 4.1 Empirical Attack Topology
PaySim generates fraud via automated agent scripts that operate in two synchronized stages:
1. **Stage 1 (Account Compromise / Infiltration):** The fraudulent agent initiates a `TRANSFER` from a compromised victim account (`C_orig`) to an intermediary drop account (`C_dest`).
   - Total Incidents: $4,097$ transactions
   - Total Amount: $\$6,067,213,184.01$
   - Average Amount: $\$1,480,891.67$
2. **Stage 2 (Cash Extraction / Liquidation):** Within the exact same simulation step or immediately following, the agent executes a matching `CASH_OUT` from the drop account.
   - Total Incidents: $4,116$ transactions
   - Total Amount: $\$5,989,202,243.83$
   - Average Amount: $\$1,455,102.59$

### 4.2 Financial Semantics & Reporting Boundaries
- **Transaction-Level Activity:** When reporting transactional activity, gross exposure is $\$12.06\text{B}$ across $8,213$ records (`gross_labelled_fraud_value`).
- **Paired-Flow Deduplicated Exposure:** When reporting unique funds extracted by the attacker, deduplicating the paired legs yields:
  $$\text{Paired-Flow Simulated Exposure} \approx \sum \text{amount}_{\text{TRANSFER, isFraud}=1} \approx \mathbf{\$6.03\text{B}}$$
- **Defensive Boundary on "Net Loss":** Neither figure may be represented as "actual net economic loss" or "confirmed financial loss". Actual loss depends on post-authorization dispute resolution, recovery rates, and legal settlement, none of which exist in synthetic data. Dashboard cards and marts must adhere strictly to these defined column names.
