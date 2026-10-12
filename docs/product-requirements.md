# FinSight — Product Requirements Document (PRD)

**Document Version:** 1.0.0  
**Project:** FinSight — Transaction Intelligence and Risk Analytics Platform  
**Target Milestone:** MVP (Portfolio & Technical Interview Ready)  
**Author:** FinSight Engineering Team  

---

## 1. Executive Summary & Business Context

In contemporary financial technology and digital payment systems, payment processors handle a vast stream of multi-channel transactions. Operational failure to detect anomalies or reconcile errors can lead to chargebacks, regulatory sanctions, financial loss, and degraded customer trust.

**FinSight** is an end-to-end analytical data platform that ingests, validates, enriches, and analyzes historical transaction records from a fictional digital payments network. The platform bridges the gap between raw data engineering pipelines, regulatory compliance, and business intelligence. It provides risk analysts, operations teams, and executive stakeholders with auditable risk metrics, payment performance KPIs, and data quality guarantees.

---

## 2. Product Scope Boundaries

### In Scope (MVP)
- **Batch Transaction Ingestion**: Processing tabular transaction extracts (CSV format) with batch metadata.
- **Defensive Data Quality (DQ) Enforcement**: Explicit schema contracts, range validation, categorical checks, and duplicate detection.
- **Quarantine Workflow**: Automated segregation of non-compliant records into quarantine tables without pipeline failure.
- **PostgreSQL Analytical Storage**: Structured schema modeling separating staging, fact, and dimension layers.
- **Explainable Rule-Based Risk Engine**: Deterministic, weighted risk scoring based on externalized configuration (`config/risk_rules.yaml`).
- **Standard FinTech Business Metrics**: SQL-based calculation of Gross Payment Volume (GPV), Success Rate, Average Order Value (AOV), and payment rail distribution.
- **Power BI Reporting**: Executive KPI dashboard, payment rail operational view, and granular risk investigation drill-down.
- **Automated Testing Suite**: High test coverage using `pytest` for ingestion, validation, risk evaluation, and SQL schema integrity.

### Out of Scope (Explicitly Excluded from MVP)
- Real-time streaming ingestion via Kafka, Flink, or message brokers.
- Automated payment gateway decisioning (e.g., inline blocking/authorization of active credit cards).
- Black-box Machine Learning models (e.g., unexplainable neural networks or opaque ensembles).
- Cloud-hosted managed services requiring recurring billing (e.g., AWS Redshift, Snowflake, Databricks).
- Customer-facing web applications or user authentication portals.

---

## 3. User Personas

| Persona | Role | Core Objectives & Pain Points |
| :--- | :--- | :--- |
| **P-1: FinTech Risk Analyst** | Fraud & Risk Operations | Needs transparent, explainable reason codes for why a transaction was flagged. Requires intuitive drill-down to review high-risk accounts and consecutive decline patterns. |
| **P-2: Payment Operations Lead** | Payment Infrastructure | Monitors authorization rates, payment rail uptime, processor decline trends, and currency breakdowns to ensure SLA compliance. |
| **P-3: Analytics Engineer** | Data Platform Engineering | Requires reliable ingestion pipelines with rigorous schema contracts, automated quarantine for bad records, and clean dimensional models. |
| **P-4: Product / Business Analyst** | Growth & Monetization | Analyzes merchant volume concentration, customer transaction velocity, AOV trends, and payment method market share to drive product strategy. |

---

## 4. Functional Requirements & Acceptance Criteria

### FR-1: Transaction Ingestion & Contract Validation
- **Requirement**: The system shall ingest raw transaction batches and enforce strict Pydantic/tabular data contracts before persisting any records.
- **Traceable Acceptance Criteria**:
  - `AC-1.1`: Pipeline successfully parses UTF-8 encoded CSV files containing transaction batches.
  - `AC-1.2`: Mandatory fields (`transaction_id`, `customer_id`, `merchant_id`, `amount`, `currency`, `timestamp`, `payment_method`, `status`) must be present and validated for data types.
  - `AC-1.3`: Amount must be numeric, strictly positive (`> 0.00`), and within acceptable boundary ceilings.
  - `AC-1.4`: Datetime timestamps must conform to standard ISO formats and must not represent future timestamps relative to batch extraction time.

### FR-2: Data Quality Monitoring & Quarantine Mechanism
- **Requirement**: The system shall isolate corrupted or schema-violating rows into a dedicated quarantine store while generating an auditable Data Quality Summary Report.
- **Traceable Acceptance Criteria**:
  - `AC-2.1`: Transactions with null primary keys, negative amounts, or invalid categorical values are routed to `dq_quarantine_records` with an exact failure reason code.
  - `AC-2.2`: Valid rows proceed downstream to clean fact tables without pipeline termination.
  - `AC-2.3`: Pipeline logs an audit record in `dq_audit_log` detailing total rows processed, clean row count, quarantine count, duplicate count, and execution timestamp.
  - `AC-2.4`: If `strict_mode: true` is enabled in configuration, critical schema violations halt the pipeline before loading.

### FR-3: PostgreSQL Storage & Relational Modeling
- **Requirement**: Cleaned transactions, risk evaluations, and dimensions must be stored in a normalized, indexed PostgreSQL database schema.
- **Traceable Acceptance Criteria**:
  - `AC-3.1`: DDL scripts establish `stg_transactions`, `fct_transactions`, `fct_risk_evaluations`, `dq_quarantine_records`, and `dq_audit_log`.
  - `AC-3.2`: Primary key uniqueness constraints are strictly enforced on `fct_transactions(transaction_id)`.
  - `AC-3.3`: Foreign keys or logical referential consistency links `fct_risk_evaluations` to `fct_transactions`.
  - `AC-3.4`: Indexes are created on high-cardinality query columns (`customer_id`, `timestamp`, `merchant_id`, `risk_score`, `status`).

### FR-4: SQL Analytical Metrics & Reporting Views
- **Requirement**: Analytical views and scripts must compute standardized payment performance and operational KPIs directly within PostgreSQL.
- **Traceable Acceptance Criteria**:
  - `AC-4.1`: Compute Gross Payment Volume (GPV), Total Transaction Count, and Success Rate partitioned by day, week, and payment method.
  - `AC-4.2`: Calculate Average Order Value (AOV) and standard deviation across merchant categories.
  - `AC-4.3`: Aggregate failure reasons and calculate processor decline ratios.
  - `AC-4.4`: Implement SQL window functions to analyze customer transaction velocity and inter-transaction intervals.

### FR-5: Configurable & Explainable Risk Scoring Engine
- **Requirement**: The system shall evaluate transactions against externalized, deterministic risk rules defined in `config/risk_rules.yaml` and assign a composite score (0–100) with clear reason codes.
- **Traceable Acceptance Criteria**:
  - `AC-5.1`: High-value transaction threshold triggers `RULE_001_HIGH_VALUE` and assigns configured points.
  - `AC-5.2`: Short-window transaction bursts (velocity) trigger `RULE_002_HIGH_VELOCITY` using configurable time windows.
  - `AC-5.3`: Geographic hops across country borders within impossible transit windows trigger `RULE_003_RAPID_GEO_HOP`.
  - `AC-5.4`: Transactions at high-risk Merchant Category Codes (e.g., gambling, unregulated crypto) trigger `RULE_004_HIGH_RISK_MCC`.
  - `AC-5.5`: Consecutive declines followed by an approval trigger `RULE_005_CONSECUTIVE_DECLINES`.
  - `AC-5.6`: Each evaluated transaction outputs: `risk_score` (INT 0-100), `risk_tier` (LOW, MEDIUM, HIGH), `action` (AUTO_APPROVE, MANUAL_REVIEW, FLAG_SUSPICIOUS), and a concatenated list of explainable `reason_codes`.

### FR-6: Power BI Reporting & Investigation Dashboard
- **Requirement**: A Power BI reporting suite must connect to PostgreSQL analytical views and present multi-tiered insights.
- **Traceable Acceptance Criteria**:
  - `AC-6.1`: **Executive Overview Page**: Displays headline KPIs (GPV, Success Rate, Total Transactions, Flagged Volume %) with dynamic date slicers.
  - `AC-6.2`: **Payment Performance Page**: Analyzes volume and failure rates across payment methods, currencies, and merchant tiers.
  - `AC-6.3`: **Risk & Fraud Operations Page**: Visualizes risk score distribution, top triggering rules, and an interactive drill-down matrix for transaction-level analyst review.

### FR-7: Reproducibility, Logging, and Automated Testing
- **Requirement**: The entire platform must be reproducible, fully testable, and provide clear operational logging.
- **Traceable Acceptance Criteria**:
  - `AC-7.1`: Pipeline emits structured logs (`INFO`, `WARNING`, `ERROR`) tracking execution phases and timing.
  - `AC-7.2`: Automated test suite in `pytest` covers unit validation, risk scoring rules, database loading idempotency, and SQL view syntax.
  - `AC-7.3`: Project configuration relies strictly on externalized YAML and environment variables, with no hardcoded credentials or paths.

---

## 5. Non-Functional Requirements (NFR)

- **Performance**: Batch ingestion and risk scoring must process a standard 50,000-record benchmark batch in `< 60 seconds` on a standard development machine.
- **Data Integrity & Idempotency**: Re-running the ingestion pipeline on the same source batch must not generate duplicate records in fact tables.
- **Security & Privacy**: No plain-text passwords or secret keys in source control. Database credentials must be loaded via `.env`.
- **Maintainability & Code Quality**: Modular Python code organized into distinct single-responsibility modules, with type hints and PEP-8 compliance.

---

## 6. Success Metrics & Interview Evaluation Criteria

1. **Explainability**: Can an analyst explain why transaction `TX10928` was scored 85? (Yes: via audit trail and reason codes).
2. **Resilience**: Does the pipeline crash when encountering corrupted data rows? (No: quarantine isolation protects downstream tables).
3. **Data Modeling Rigor**: Are analytical tables properly modeled, indexed, and documented in a formal data dictionary? (Yes).
4. **Reproducibility**: Can a technical recruiter or hiring manager clone the repository, run the setup steps, and reproduce the data pipeline and tests without ad-hoc fixes? (Yes).
