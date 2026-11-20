# FinSight — Engineering Status, Findings, & System Limitations

## Current findings (end of P1, 2026-11-20)

1. **Earlier results were inflated.** The pre-P0 holdout (steps 551–743) was used to choose features and the model, so figures such as AP 0.4479 are optimistic. They are kept in `reports/archive/` with known errors.
2. **One genuine new signal.** Fraud goes to first-time recipients far more often (about 63% vs 15% of legitimate transfers). The leakage-safe recipient count lifted average precision by 0.11–0.20 on every fold.
3. **Champion E4 confirmed once on sealed data.** AP 0.697 vs 0.491 for the previous champion on steps 651–743; 36.8% of fraud caught at 100 alerts/day at 97% precision.
4. **Most fraud is still missed**, especially in daytime (sealed test: 15.5% recall at 09:00–17:00, 30.1% at 18:00–23:00), because PaySim offers no other legitimate signal that separates it from ordinary large transfers.
5. **Balance fields are a simulator artifact** (near-perfect AP) and are excluded.
6. **Alert workload:** a fixed threshold drifts with traffic volume; a first-come daily cap loses fraud when it runs out; a score-ordered queue with a 12-hour deadline keeps the load steady.
7. **Rule engine** (4 hand-set rules) is weaker than the model and untuned.

## Current limitations

- Synthetic data; hourly time resolution; no settlement timing, so the cost of review delay is unknown.
- Development folds were reused many times; the sealed test is now retired, so further model selection has no untouched data.
- Scores are not calibrated probabilities.
- Batch, single-node processing; not a real-time authorization system.

---

## Stage 1 record (historical)

> The sections below were written at Stage 1, before any data was processed. Status items, `config/*.yaml` and the 'next step' are obsolete.

**Document Version:** 1.0.0  
**Current Milestone:** Stage 1 — Engineering Initialization & System Setup  
**Last Updated:** Current Initialization Stage  

---

## 1. Project Implementation Status

As of Stage 1, the FinSight repository has been initialized with engineering standards, architectural blueprints, schema contracts, and configuration definitions.

### Current Implementation State
- [x] **Repository & Version Control**: Git repository initialized with strict `.gitignore` rules preventing secret leaks, virtualenv bloat, or untracked data dumps.
- [x] **Project Architecture**: Layered architecture defined (Ingestion -> Validation & Quarantine -> Risk Scoring -> PostgreSQL -> Power BI).
- [x] **Configuration Layer**: Externalized pipeline parameters (`config/pipeline.yaml`) and explainable risk rules (`config/risk_rules.yaml`) established.
- [x] **Documentation Suite**: PRD, Architecture, Data Dictionary, Metric Dictionary, and Risk Methodology fully drafted.
- [ ] **Data Pipeline (Stage 2 & 3)**: *Not yet implemented*. No data ingestion scripts, ETL transforms, or Pydantic models have been executed.
- [ ] **PostgreSQL Database Loader (Stage 4)**: *Not yet executed*. DDL scripts and SQLAlchemy loaders are planned for Stage 4.
- [ ] **Empirical Analytics & Findings**: *None yet*. In accordance with data integrity principles, no analytical findings, fraud rates, or business performance conclusions have been fabricated prior to running real or synthetic data.

---

## 2. Platform Architecture Limitations

### 2.1. Batch-Oriented Architecture vs. Real-Time Streaming
- **Limitation**: FinSight is designed as a batch-oriented transaction intelligence platform operating on periodic data extracts (e.g., hourly or daily CSV files).
- **Impact**: It cannot be deployed as an inline payment authorization service where millisecond-level card authorization decisions are made.
- **Future Evolution**: Streaming platforms (Apache Kafka / Apache Flink) could be introduced in a post-MVP release if real-time low-latency fraud gating is required.

### 2.2. Single-Node In-Memory Scalability
- **Limitation**: Tabular data transformations in the MVP rely on Python and pandas. Data processing takes place in single-node system memory.
- **Impact**: Batch files larger than available RAM (typically > 4GB) will require chunked iterator processing or migration to an out-of-core engine like Polars, DuckDB, or PySpark.
- **Mitigation in MVP**: `config/pipeline.yaml` specifies a chunked processing model (`batch_chunk_size: 10000`) to maintain a predictable memory footprint.

### 2.3. Rule-Based Scoring vs. Machine Learning
- **Limitation**: The initial risk engine relies on deterministic, expert-crafted business rules with additive weights rather than statistical machine learning.
- **Impact**: While 100% explainable and compliant with regulatory audit demands, the rule engine may experience false positives on edge cases and cannot detect novel, complex fraud rings without manual rule updates.
- **Future Evolution**: Stage 6 or 7 can introduce a supervised ML model (e.g., LightGBM / XGBoost) operating in parallel to generate an advisory risk probability alongside the deterministic rules.

### 2.4. Local Database Dependency
- **Limitation**: Analytical queries rely on a PostgreSQL 15+ instance.
- **Impact**: The environment requires either a running local PostgreSQL server or a containerized instance. The setup has been kept decoupled so SQLite or DuckDB could serve as fallback testing fixtures if PostgreSQL is unavailable in local environments.

---

## 3. Data Integrity & Synthetic Data Caveats

- **Absence of Real PII**: FinSight will operate exclusively on synthetic, programmatically generated transaction logs designed to reflect realistic FinTech distributions (log-normal transaction amounts, power-law merchant distributions, weekend seasonality, and realistic fraud anomalies).
- **Absence of Chargeback Feedback**: In live payment processing, ground-truth fraud labels arrive 30–90 days post-transaction via card network chargebacks. Synthetic test datasets simulate these outcomes directly to enable metric validation.

---

## 4. Immediate Next Step (Stage 2)

The immediate next engineering milestone is **Stage 2: Synthetic Data Generation & Data Contracts**:
1. Implement a deterministic, reproducible synthetic transaction data generator in `src/finsight/` that produces clean transactions alongside realistic edge cases (negative values, duplicate IDs, missing customer keys, rapid geo-hops, and velocity spikes).
2. Generate an initial sanitized benchmark sample dataset (`data/sample/sample_transactions.csv`) to validate schemas.
3. Author Pydantic data validation schemas matching the Data Dictionary.
