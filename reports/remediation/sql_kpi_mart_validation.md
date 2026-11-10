# FinSight — PostgreSQL KPI Marts Reconciliation & Validation Report

**Document Status:** Production Validated  
**Date:** 2026-10-19  
**Auditor:** Senior Data Engineer & ML Evaluation Auditor  
**Scope:** PostgreSQL Analytical Views Deployment, Fact Reconciliation, and Semantic Modeling Handoff  

---

## 1. Executive Summary

All four required PostgreSQL analytical views have been implemented, deployed, and mathematically reconciled against the live production PostgreSQL instance (`localhost:5432/finsight_db`).

Every output across all four views reconciles with zero variance against the authoritative ground truth of `fct_transactions` ($6,362,620$ rows, $\$1,144,392,944,759.77$ GTV, $8,213$ simulated frauds).

### Core Reconciliation Summary Table:
| Metric | Source Fact Table (`fct_transactions`) | `vw_daily_transaction_kpis` | `vw_hourly_risk_dynamics` | `vw_payment_rail_performance` | Variance / Discrepancy |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Total Transactions** | **6,362,620** | 6,362,620 | 6,362,620 | 6,362,620 | **0 (Exact)** |
| **Gross Transaction Value (GTV)** | **$1,144,392,944,759.77** | $1,144,392,944,759.77 | $1,144,392,944,759.77 | $1,144,392,944,759.77 | **$0.00 (Exact)** |
| **Labelled Fraud Count** | **8,213** | 8,213 | 8,213 | 8,213 | **0 (Exact)** |
| **Gross Labelled Fraud Value** | **$12,056,415,427.84** | $12,056,415,427.84 | $12,056,415,427.84 | $12,056,415,427.84 | **$0.00 (Exact)** |
| **Fraudulent TRANSFER Value** | **$6,067,213,184.01** | $6,067,213,184.01 | $6,067,213,184.01 | — | **$0.00 (Exact)** |
| **Summed Volume Share** | — | — | — | 100.00% | **0.00% (Exact)** |
| **Summed Value Share** | — | — | — | 100.00% | **0.00% (Exact)** |

---

## 2. PostgreSQL Schema & View Definitions

All view DDL files are maintained under [`sql/views/`](file:///sql/views/) and are completely idempotent (`CREATE OR REPLACE VIEW`).

### 2.1 View 1: `vw_daily_transaction_kpis`
- **File:** [`sql/views/01_vw_daily_transaction_kpis.sql`](file:///sql/views/01_vw_daily_transaction_kpis.sql)
- **Grain:** One record per simulation day (`transaction_day`, 1 to 31).
- **Row Count:** Exactly 31 rows.
- **Fields Provided:**
  - `simulation_day`: Day index (1--31).
  - `transaction_count`: Volume of transactions on that day.
  - `gross_transaction_value`: Total payments volume in USD.
  - `avg_transaction_amount`: Mean transaction size in USD.
  - `labelled_fraud_transaction_count`: Ground-truth simulated fraud count.
  - `gross_labelled_fraud_value`: Total face value of simulated fraud transactions.
  - `fraud_transaction_rate_pct`: Fraud incidence percentage (`labelled_fraud_count / transaction_count * 100`).
  - `fraud_transfer_face_value`: Stage-1 extraction volume (TRANSFER fraud face value).
  - `fraud_value_exposure_share_pct`: Dollar share of daily volume involved in fraud.

### 2.2 View 2: `vw_hourly_risk_dynamics`
- **File:** [`sql/views/02_vw_hourly_risk_dynamics.sql`](file:///sql/views/02_vw_hourly_risk_dynamics.sql)
- **Grain:** One record per simulation step (`step`, 1 to 743).
- **Row Count:** Exactly 743 rows.
- **Temporal Semantics Note:** `step` represents 1-indexed simulation hours. Hours are synthetic simulation counters and do not map to real calendar timestamps or real-world timezones.
- **Fields Provided:**
  - `simulated_hour_step`: Simulation hour counter (1--743).
  - `simulation_day`: Day index (1--31).
  - `hour_of_day`: Diurnal hour index (0--23).
  - `transaction_count`: Hourly transaction volume.
  - `gross_transaction_value`: Hourly dollar volume.
  - `avg_transaction_amount`: Hourly average transaction magnitude.
  - `labelled_fraud_count`: Hourly simulated fraud incidents.
  - `gross_labelled_fraud_value`: Hourly gross fraud-associated face value.
  - `labelled_fraud_prevalence_pct`: Hourly fraud prevalence (`fraud_count / transaction_count * 100`).
  - `fraud_transfer_face_value`: Hourly stage-1 transfer fraud value.

### 2.3 View 3: `vw_payment_rail_performance`
- **File:** [`sql/views/03_vw_payment_rail_performance.sql`](file:///sql/views/03_vw_payment_rail_performance.sql)
- **Grain:** One record per payment rail / transaction type (5 rows).
- **Row Count:** Exactly 5 rows (`CASH_IN`, `CASH_OUT`, `DEBIT`, `PAYMENT`, `TRANSFER`).
- **Fields Provided:**
  - `payment_rail`: Transaction type code.
  - `transaction_count`: Volume of payments on this rail.
  - `gross_transaction_value`: Total dollar volume on this rail.
  - `avg_transaction_amount`: Average transaction magnitude on this rail.
  - `labelled_fraud_count`: Fraud count (confined to `TRANSFER` and `CASH_OUT`).
  - `gross_labelled_fraud_value`: Face value of fraud on this rail.
  - `labelled_fraud_rate_pct`: Fraud incident rate on this rail.
  - `fraud_value_exposure_share_pct`: Percentage of rail dollar volume in fraud.
  - `rail_volume_share_pct`: Proportion of total platform transaction count.
  - `rail_value_share_pct`: Proportion of total platform dollar volume.

#### Rail Performance Empirical Breakdown:
| Payment Rail | Transaction Count | Gross Value (USD) | Volume Share | Value Share | Fraud Count | Fraud Value (USD) | Fraud Rate (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `TRANSFER` | 532,909 | $485,291,987,263.17 | 8.38% | 42.41% | 4,097 | $6,067,213,184.01 | 0.7688% |
| `CASH_OUT` | 2,237,500 | $394,412,995,224.49 | 35.17% | 34.46% | 4,116 | $5,989,202,243.83 | 0.1840% |
| `CASH_IN` | 1,399,284 | $236,367,391,912.46 | 21.99% | 20.65% | 0 | $0.00 | 0.0000% |
| `PAYMENT` | 2,151,495 | $28,093,371,138.37 | 33.81% | 2.45% | 0 | $0.00 | 0.0000% |
| `DEBIT` | 41,432 | $227,199,221.28 | 0.65% | 0.02% | 0 | $0.00 | 0.0000% |
| **Total** | **6,362,620** | **$1,144,392,944,759.77** | **100.00%** | **100.00%** | **8,213** | **$12,056,415,427.84** | **0.1291%** |

### 2.4 View 4: `vw_risk_detection_benchmark`
- **File:** [`sql/views/04_vw_risk_detection_benchmark.sql`](file:///sql/views/04_vw_risk_detection_benchmark.sql)
- **Grain:** One record per evaluation partition, system/model identifier, and operating point.
- **Row Count:** Exactly 8 benchmark rows.
- **Integrity Guarantee:** Confusion matrices ($TP + FP + FN + TN$) strictly sum to the exact evaluation population ($196,601$ for holdout, $151,302$ for validation). Average Precision and ROC-AUC are populated only for continuous probability models.

#### Benchmark Verification Table:
| Partition | System Identifier | Operating Point | TP | FP | FN | TN | Total Pop | Alerts | Alert Rate | Precision | Recall | F1 Score | PR-AUC (AP) | ROC-AUC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `holdout` | **HistGradientBoosting (Champion No Day)** | $t = 0.9660$ (Val F1 Optimal) | **918** | **672** | **1,230** | **193,781** | **196,601** | **1,590** | **0.8087%** | **57.74%** | **42.74%** | **0.4912** | **0.4479** | **0.9487** |
| `holdout` | **HistGradientBoosting (Baseline 9-feat)** | $t = 0.9830$ (Val F1 Optimal) | 845 | 551 | 1,303 | 193,902 | 196,601 | 1,396 | 0.7101% | 60.53% | 39.34% | 0.4769 | 0.3836 | 0.9185 |
| `holdout` | **HistGradientBoosting (Baseline 9-feat)** | $t = 0.9900$ (Val Capacity Constrained) | 737 | 331 | 1,411 | 194,122 | 196,601 | 1,068 | 0.5432% | 69.01% | 34.31% | 0.4583 | 0.3836 | 0.9185 |
| `holdout` | **Heuristic Rule Engine (HIGH Tier)** | Score $\ge 70$ (HIGH Tier) | 402 | 156 | 1,746 | 194,297 | 196,601 | 558 | 0.2838% | 72.04% | 18.72% | 0.2971 | — | — |
| `holdout` | **Logistic Regression (Balanced)** | $t = 0.9009$ (Val F1 Optimal) | 626 | 1,950 | 1,522 | 192,503 | 196,601 | 2,576 | 1.3103% | 24.30% | 29.14% | 0.2650 | 0.1805 | 0.9152 |
| `holdout` | **Heuristic Rule Engine (MED+HIGH Tier)** | Score $\ge 30$ (MED+HIGH Tier) | 1,717 | 47,213 | 431 | 147,240 | 196,601 | 48,930 | 24.8879% | 3.51% | 79.93% | 0.0672 | — | — |
| `holdout` | **Dummy Majority Baseline** | $t = 0.5000$ (Always Negative) | 0 | 0 | 2,148 | 194,453 | 196,601 | 0 | 0.0000% | 0.00% | 0.00% | 0.0000 | 0.0109 | 0.5000 |
| `validation` | **HistGradientBoosting (Champion No Day)** | $t = 0.9660$ (Val F1 Optimal) | **435** | **863** | **573** | **149,431** | **151,302** | **1,298** | **0.8579%** | **33.51%** | **43.15%** | **0.3773** | **0.3115** | **0.9509** |

---

## 3. Financial Semantics & Metric Reconciliation

### 3.1 Strict Financial Nomenclature
To prevent deceptive executive dashboard visuals, the following standards are verified:
1. **Gross Labelled Fraud-Associated Value (`$12.06B`):** The sum of all fraud-flagged records (`is_fraud = 1`). This is an event-level exposure metric.
2. **Fraudulent TRANSFER Value (`$6.07B`):** Stage-1 extraction volume (`$6,067,213,184.01` across $4,097$ transactions). Formally replaces `net_simulated_economic_leakage`.
3. **Paired-Flow Deduplicated Exposure (`$6.07B`):** Accounts for the 1:1 paired nature of TRANSFER $\to$ CASH_OUT attacks to prevent double-counting.
4. **Net Economic Loss:** Unobservable in synthetic data; no visual or column may claim to represent "realized loss".

---

## 4. Power BI Consumption Specifications

Power BI report authors must connect directly to these four analytical views:

```
[ FinSight Power BI Semantic Layer ]
├── vw_daily_transaction_kpis       -> Daily trends, macro volume cards, fraud rate over time.
├── vw_hourly_risk_dynamics         -> 24-hour diurnal cycle decomposition, simulated hour trends.
├── vw_payment_rail_performance     -> Payment mix breakdown, rail risk concentration (TREEMAP / BAR).
└── vw_risk_detection_benchmark     -> Model vs Rule performance comparison, ROC/PR curves, confusion matrices.
```

### Table & Column Mapping:
1. **Executive Overview Page:**
   - Consume `vw_daily_transaction_kpis` for total GTV (`SUM(gross_transaction_value)`), fraud volume (`SUM(gross_labelled_fraud_value)`), and fraud rate line chart (`simulation_day` vs `fraud_transaction_rate_pct`).
   - Consume `vw_payment_rail_performance` for rail throughput share.
2. **Diurnal Risk Page:**
   - Consume `vw_hourly_risk_dynamics` for diurnal risk concentration (`hour_of_day` vs `AVG(labelled_fraud_prevalence_pct)`).
3. **Model Governance & Compliance Review Page:**
   - Consume `vw_risk_detection_benchmark` for F1, Precision, Recall, and confusion matrix visual matrix cards.
