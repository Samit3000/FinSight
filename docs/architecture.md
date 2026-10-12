# FinSight — System Architecture & Technical Design

**Document Version:** 1.0.0  
**Status:** Architecture Specification (Stage 1)  
**System:** FinSight Transaction Intelligence Platform  

---

## 1. System Overview

FinSight is designed as a modular, decoupled analytical data platform. The architecture follows modern data engineering practices, emphasizing contract-driven ingestion, deterministic risk evaluation, relational dimensional modeling, and interactive business intelligence.

The platform processes batch transaction data, isolates corrupted records into an auditable quarantine store, enriches clean transactions with explainable risk features, loads them into an indexed PostgreSQL analytical warehouse, and provides analytical SQL views consumed by Power BI.

---

## 2. End-to-End Data Pipeline Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                             1. Ingestion Layer                              │
│                                                                             │
│  [ Raw Transaction Files (CSV) ] ──► [ Batch Chunk Reader & Type Normalizer]│
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         2. Data Quality & Contract Layer                    │
│                                                                             │
│               [ Pydantic Data Contract & Constraint Validator ]             │
│                                      │                                      │
│                ┌─────────────────────┴─────────────────────┐                │
│                ▼                                           ▼                │
│       [ Validation Failures ]                     [ Valid Records ]         │
│                │                                           │                │
│                ▼                                           ▼                │
│   [ Quarantine Store & Audit Log ]           3. Explainable Risk Engine     │
│   (dq_quarantine_records / dq_audit_log)     (config/risk_rules.yaml)       │
│                                                            │                │
│                                              - Velocity Spikes              │
│                                              - Value Outliers               │
│                                              - Geo Hop Discrepancies        │
│                                              - High-Risk MCC                │
│                                              - Behavioral Declines          │
└────────────────────────────────────────────────────────────┬────────────────┘
                                                             │
                                                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         4. PostgreSQL Analytical Storage                    │
│                                                                             │
│   ┌─────────────────────┐   ┌──────────────────────┐  ┌──────────────────┐  │
│   │  stg_transactions   ├──►│   fct_transactions   │◄─┤  dim_customers   │  │
│   │  (raw landing zone) │   └──────────┬───────────┘  └──────────────────┘  │
│   └─────────────────────┘              │                                    │
│                             ┌──────────▼───────────┐  ┌──────────────────┐  │
│                             │ fct_risk_evaluations │  │  dim_merchants   │  │
│                             └──────────────────────┘  └──────────────────┘  │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         5. Analytical & Metric Marts                        │
│                                                                             │
│   [ SQL Analytical Views: vw_daily_kpis, vw_risk_summary, vw_payment_rails] │
└────────────────────────────────────────┬────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         6. Presentation Layer (Power BI)                    │
│                                                                             │
│   [ Executive Overview ]    [ Payment Operations ]    [ Risk Drill-down ]   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Component Deep Dive

### 3.1. Ingestion Layer (`src/finsight/ingestion`)
- **Responsibilities**:
  - Ingest raw transaction files from local or staged filesystem paths.
  - Parse date formats, handle encoding variations (UTF-8), and strip leading/trailing whitespace.
  - Process records in memory-safe chunks (configurable chunk size in `config/pipeline.yaml`).
- **Design Pattern**: Reader/Adapter pattern isolating storage format details from processing logic.

### 3.2. Data Quality & Validation Engine (`src/finsight/validation`)
- **Responsibilities**:
  - Enforce schema contracts using typed models (Pydantic / vector validators).
  - Check non-null constraints on mandatory primary/foreign keys (`transaction_id`, `customer_id`, `merchant_id`).
  - Validate numeric boundaries (`amount > 0.00`, amount ceiling checks).
  - Verify categorical validity against allowed domain sets (`currency`, `payment_method`, `status`).
  - Identify duplicate primary keys within incoming batches and against existing database records.
- **Quarantine Workflow**:
  - Clean rows flow directly to the risk scoring engine.
  - Invalid rows are flagged with specific failure codes (e.g., `ERR_NULL_CUSTOMER_ID`, `ERR_NEGATIVE_AMOUNT`) and routed to `dq_quarantine_records`.
  - Batch summary metrics are recorded into `dq_audit_log`.

### 3.3. Explainable Risk Scoring Engine (`src/finsight/risk`)
- **Responsibilities**:
  - Read declarative, version-controlled risk rules from `config/risk_rules.yaml`.
  - Perform windowed aggregations (e.g., customer transaction counts within 10-minute rolling windows).
  - Evaluate country transitions to detect impossible travel velocities.
  - Match merchant codes against monitored risk classifications.
  - Calculate a cumulative composite risk score (0 to 100), classify into discrete tiers (`LOW`, `MEDIUM`, `HIGH`), and construct human-readable audit reason codes.
- **Why Declarative YAML?**: Decouples business policy from execution code, enabling risk analysts and compliance teams to audit and adjust thresholds without code deployments.

### 3.4. Relational Storage Layer (PostgreSQL)
- **Database Architecture**:
  - **Staging Layer (`stg_transactions`)**: Ephemeral, unconstrained table matching raw input structure for auditing and rollback.
  - **Fact Layer (`fct_transactions`)**: Clean, validated transaction records with enforced constraints, primary keys, and typed columns.
  - **Risk Fact Layer (`fct_risk_evaluations`)**: Enriched risk scores, action recommendations, and JSON/array reason codes linked 1:1 with transactions.
  - **Dimension Tables (`dim_customers`, `dim_merchants`)**: Slowly changing attributes providing analytical context.
  - **Data Quality Tables (`dq_quarantine_records`, `dq_audit_log`)**: Segregated audit records detailing validation errors.

### 3.5. Analytical Mart & Metric Layer (`sql/views`)
- **Responsibilities**:
  - Transform normalized transaction and risk facts into business-ready analytical views.
  - Pre-aggregate daily, weekly, and merchant-level performance indicators.
  - Implement window functions for cohort retention, running cumulative volumes, and velocity metrics.
  - Serve as the contract between the relational warehouse and downstream BI tools.

### 3.6. Presentation Layer (`powerbi/`)
- **Responsibilities**:
  - Provide an interactive, low-latency dashboard connected to PostgreSQL via direct query or import mode.
  - Deliver three core reporting views:
    1. **Executive Performance**: Top-line revenue, GPV, transaction count, authorization rate.
    2. **Payment Operations**: Payment rail health, processor error breakdown, currency volumes.
    3. **Risk Investigation Matrix**: Filterable transaction drill-down, high-risk customer queues, and rule-firing frequency.

---

## 4. Key Architectural Trade-Offs & Decisions

| Decision Area | Chosen Approach | Alternative Considered | Trade-Off & Justification |
| :--- | :--- | :--- | :--- |
| **Data Processing Engine** | **Python (pandas / vectorized logic)** | Apache Spark / PySpark | **Why chosen**: FinSight MVP handles batch sizes up to hundreds of thousands of transactions. Python + pandas provides near-instant local setup, zero cluster overhead, and complete reproducibility. Spark introduces high operational complexity (JVM, master/worker coordination, cluster tuning) that is overkill for MVP data scales. |
| **Database Engine** | **PostgreSQL** | Cloud Data Warehouse (Snowflake / BigQuery / Redshift) | **Why chosen**: PostgreSQL is open-source, runs natively on developer hardware, supports ACID transactions, and offers enterprise-grade analytical SQL window functions. Eliminates cloud billing requirements while maintaining professional dimensional modeling. |
| **Risk Modeling Approach** | **Configurable Rule Engine (YAML)** | Machine Learning Classifier (XGBoost / Random Forest) | **Why chosen**: In regulated FinTech environments, risk decisions require auditable, explainable rationales (e.g., adverse action notices). Rule engines provide transparent, deterministic decision traces. ML models can be added in later stages as a complementary signal, but cannot replace the baseline rule layer. |
| **Storage Modeling** | **Relational Star / Snowflake Schema** | Wide Flat Table / Document Store (MongoDB) | **Why chosen**: Fact/dimension modeling prevents data duplication, optimizes query performance, and adheres to analytical engineering best practices taught across enterprise data teams. |
| **Configuration Management** | **External YAML + `.env`** | Hardcoded Constants / DB-backed Config | **Why chosen**: Follows 12-Factor App methodology. Separates infrastructure secrets (`.env`) from analytical business logic (`config/*.yaml`), ensuring zero credentials in git history. |

---

## 5. Security, Privacy, and Environment Isolation

1. **Credential Isolation**: All database URLs, user accounts, and passwords reside exclusively in `.env`, strictly ignored by `.gitignore`.
2. **Synthetically Generated Data**: Only sanitized, synthetic, or fully anonymized transaction logs are used. No Personally Identifiable Information (PII) or real credit card PANs (Primary Account Numbers) are ever ingested or stored.
3. **Audit Trail**: Every pipeline run logs execution metadata, record counts, and quarantine rates, providing a transparent chain of custody for all analytical figures.
