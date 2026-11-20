# FinSight — Transaction Intelligence & Risk Analytics Platform

[![Python Version](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13%20%7C%203.14-blue.svg)](https://www.python.org/)
[![Database](https://img.shields.io/badge/PostgreSQL-15%2B-blue.svg)](https://www.postgresql.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Stage-1%3A%20Engineering%20Initialization-amber.svg)](#project-status)

FinSight is a portfolio-grade transaction intelligence and risk analytics platform designed for a fictional digital payments ecosystem. Built with a focus on data engineering rigor, transparent risk modeling, and business-focused analytics, FinSight delivers end-to-end visibility into transaction performance, data quality health, and explainable fraud risk indicators.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Target Personas & Use Cases](#target-personas--use-cases)
- [Core Architecture](#core-architecture)
- [Technology Stack](#technology-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Configuration](#environment-configuration)
- [Explainable AI with SHAP (Milestone 5)](#explainable-ai-with-shap-milestone-5)
- [Master ML Remediation & Reliability Phase](#master-ml-remediation--reliability-phase)
- [Documentation Index](#documentation-index)
- [Project Status & Roadmap](#project-status--roadmap)
- [Engineering Standards & Interview Readiness](#engineering-standards--interview-readiness)

---

## Project Overview

Modern digital payment platforms process millions of transactions across diverse rails (cards, digital wallets, bank transfers, instant payments). Analytics teams in FinTech face three interlocking challenges:
1. **Data Quality & Ingestion Integrity**: Raw transaction logs frequently suffer from schema drift, timestamp anomalies, missing customer keys, and duplicate entries.
2. **Explainable Risk Intelligence**: Traditional black-box fraud scores fail regulatory compliance demands for clear, auditable explanations behind transaction flags.
3. **Operational & Executive Visibility**: Risk analysts, compliance officers, and business leaders require unified views connecting transaction performance (volume, success rate, AOV) directly to risk exposure.

FinSight addresses these challenges through a modular pipeline that ingests, validates, profiles, enriches with explainable risk scores, stores in a normalized analytical database, and exposes data for business intelligence in Power BI.

> **Note on Scope**: FinSight is a historical-data analytics and batch intelligence prototype. It is not an inline, low-latency payment processing engine or an automated card authorization gateway.

---

## Target Personas & Use Cases

| Persona | Primary Goal | FinSight Feature |
| :--- | :--- | :--- |
| **FinTech Risk Analyst** | Investigate flagged transactions and evaluate suspicious patterns | Rule-based scoring engine with reason codes, risk drill-downs, and audit logging. |
| **Payment Operations Lead** | Monitor payment rail health, success rates, and decline patterns | SQL metrics and Power BI reports tracking throughput, payment rail failures, and latency. |
| **Analytics Engineer** | Build resilient data pipelines and clean dimensional models | Automated validation contracts, quarantine handling, and modular PostgreSQL schemas. |
| **Product / Business Analyst** | Assess customer transaction volume, AOV, and revenue drivers | Unified metric dictionary and executive dashboards highlighting growth and merchant concentration. |

---

## Core Architecture

FinSight follows a layered, decoupled data engineering architecture:

```text
[ Raw Transaction Logs (CSV) ]
             │
             ▼
[ Ingestion & Contract Validation (Pydantic / pandas) ]
             ├── Invalid Rows ──► [ DQ Quarantine & Audit Log ]
             └── Valid Rows
                     │
                     ▼
[ Explainable Risk Engine (Configurable YAML Rules) ]
             │ (Score: 0-100, Tier: Low/Med/High, Reason Codes)
             ▼
[ Relational Storage Layer (PostgreSQL) ]
      ├── Staging (stg_transactions)
      ├── Fact Transactions (fct_transactions)
      ├── Fact Risk Evaluations (fct_risk_evaluations)
      └── Analytical Mart Views (vw_daily_kpis, vw_risk_summary)
             │
             ▼
[ Semantic Presentation Layer (Power BI Dashboard) ]
```

---

## Technology Stack

| Layer | Technology | Rationale & Selection Criteria |
| :--- | :--- | :--- |
| **Language** | Python 3.11+ | Industry standard for data engineering, robust type hinting, rich data ecosystem. |
| **Data Processing** | pandas, NumPy | High-performance in-memory tabular manipulation and vectorization. |
| **Validation** | Pydantic | Strict, schema-driven data contracts and typed boundary validation. |
| **Database** | PostgreSQL 15+ | Enterprise-grade relational OLAP/OLTP engine with rich SQL window functions. |
| **Database ORM/Driver** | SQLAlchemy, psycopg2 | Standard, battle-tested Python-to-Postgres connection and execution layer. |
| **Configuration** | PyYAML, python-dotenv | Decoupled configuration files and secure 12-Factor environment isolation. |
| **Testing** | pytest, pytest-cov | Automated unit, schema, and regression testing with coverage reporting. |
| **Business Intelligence** | Microsoft Power BI | Tier-1 interactive dashboarding, DAX modeling, and executive visualization. |

---

## Project Structure

```text
FinSight/
├── README.md                          # Project documentation and setup guide
├── .gitignore                         # Git exclusion rules
├── .env.example                       # Environment configuration template
├── requirements.txt                   # Pinned project dependencies
├── data/
│   ├── raw/                           # Raw incoming transaction batches (git-ignored)
│   ├── processed/                     # Validated and enriched datasets (git-ignored)
│   └── sample/                        # Sanitized benchmark datasets for testing
├── docs/
│   ├── product-requirements.md        # Detailed PRD with traceable acceptance criteria
│   ├── architecture.md                # System design, data flow, and trade-off analysis
│   ├── data-dictionary.md             # Schema definitions, field types, and constraints
│   ├── metric-dictionary.md           # Business and risk KPI formulas and SQL specifications
│   ├── risk-methodology.md            # Explainable scoring framework and rule specifications
│   └── findings-and-limitations.md    # Initial engineering status and platform constraints
├── sql/
│   ├── schema/                        # Non-destructive, idempotent DDL (CREATE ... IF NOT EXISTS)
│   ├── admin/                         # Manual-only destructive scripts (reset_schema.sql)
│   ├── views/                         # Materialized and analytical SQL views
│   └── analytics/                     # Ad-hoc investigation and KPI queries
├── src/
│   └── finsight/                      # Core package source code
│       ├── train.py                   # Reproducible training: config -> bundle + manifest
│       ├── evaluate.py                # Locked evaluation protocol (walk-forward, locked final test)
│       └── benchmark_view.py          # Generates sql/views/04 from artifacts and data
├── models/                            # Authoritative frozen artifacts (champion: hist_gradient_boosting_no_day.joblib)
│   └── candidates/                    # Committed train.py manifests; retrained .joblib bundles stay local (git-ignored)
├── tests/                             # Automated test suite (unit, integration, DQ)
│       └── __init__.py
├── reports/
│   ├── data_quality/                  # Generated data validation audit reports
│   ├── evaluation/                    # Generated walk-forward and benchmark results
│   ├── explainability/                # Generated SHAP outputs
│   └── archive/                       # Superseded historical reports (see archive/README.md)
└── powerbi/                           # Power BI semantic models and template files
```

---

## Getting Started

### Prerequisites

- **Python**: 3.11 or newer (Python 3.14 compatible)
- **Git**: Installed and configured
- **PostgreSQL**: Local instance or Docker container (optional for Stage 1 initialization)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd FinSight
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (PowerShell)
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS / Linux
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

### Environment Configuration

Copy the example environment file and configure local credentials:
```bash
cp .env.example .env
```
Update `.env` with your PostgreSQL connection parameters and local filesystem preferences.

---

## Transaction Pipeline (Milestone 1)

FinSight includes a memory-safe, chunked transaction ingestion and validation pipeline designed for the PaySim dataset (6.36 million records).

### Exact Execution Commands

```bash
# Run the pipeline on the full dataset (streaming chunk size: 250,000)
py src/finsight/pipeline.py

# Alternatively, run on a sample for quick verification (e.g. 50,000 rows)
py src/finsight/pipeline.py --max-rows 50000

# Execute automated pipeline and audit tests
py -m pytest tests/ -v
```

### Pipeline Input & Output Paths
- **Input Dataset**: `PS_20174392719_1491204439457_log.csv` (auto-resolved in `data/raw/` or project root).
- **Clean Output Dataset**: `data/processed/transactions.csv.gz` (gzip-compressed CSV, ~177 MB).
- **Data Quality Run Report**: `reports/data_quality/pipeline_run.json`.

### Validations Performed
- **Required Columns & Nullability**: Asserts presence of all 11 PaySim columns; quarantines rows with missing/null values (`MISSING_REQUIRED_FIELDS`).
- **Transaction Rails**: Restricts types to the 5 valid PaySim operations (`CASH_OUT`, `TRANSFER`, `PAYMENT`, `CASH_IN`, `DEBIT`); flags unknown types (`INVALID_TRANSACTION_TYPE`).
- **Numeric Ranges**: Rejects non-numeric or negative values for `step` (must be integer $\ge 1$), `amount` ($\ge 0.0$), and balances (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`).
- **Binary Target Constraints**: Enforces that ground-truth `isFraud` and operational `isFlaggedFraud` are strictly binary (`0` or `1`).
- **Entity Identifiers**: Validates that origin and destination identifiers are non-empty strings (`EMPTY_IDENTIFIER`).

### Feature Derivation & ML Boundary
- **`transaction_hour`**: Derived from PaySim's hourly simulation step: `(step - 1) % 24` mapping steps into 0-indexed hour of day (0–23). Documented simulation assumption; no calendar timestamps are fabricated.
- **`transaction_day`**: Derived as `((step - 1) // 24) + 1` representing the 1-indexed simulation calendar day (1–31).
- **ML Target & Exclusions**:
  - `isFraud` is preserved strictly as the ground-truth prediction target, never as a feature.
  - `isFlaggedFraud`, balance fields (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`), and raw sparse IDs are formally excluded from the initial ML candidate features to prevent synthetic simulator leakage and data artifacts. All original columns are preserved in `transactions.csv.gz` for SQL auditing and analytical KPI queries.

### Rejection Reporting & Accounting
- Every invalid row is captured with a specific primary rejection code (`rejection_reasons`).
- The pipeline enforces strict row accounting: $\text{Total Rows Read} = \text{Accepted Rows} + \text{Rejected Rows}$.
- Run metrics, elapsed duration, output file size, and fraud counts are saved in `reports/data_quality/pipeline_run.json`.
- Atomic writes guarantee that failed runs do not overwrite existing outputs with partial data.

---

## PostgreSQL & SQL Analytics (Milestone 2)

FinSight stores validated transaction records in PostgreSQL 15+ for high-performance relational analytics, dimensional aggregation, and KPI reporting.

### 1. PostgreSQL Prerequisites & Setup

#### Option A: Docker (Recommended for local development)
```bash
# Start a clean PostgreSQL 15+ container
docker run --name finsight-postgres -e POSTGRES_DB=finsight_db -e POSTGRES_USER=finsight_user -e POSTGRES_PASSWORD=finsight_secure_pass -p 5432:5432 -d postgres:16-alpine
```

#### Option B: Native Windows / Linux PostgreSQL
Ensure PostgreSQL 15 or newer is installed and running on port `5432`. Create the database and user:
```sql
CREATE DATABASE finsight_db;
CREATE USER finsight_user WITH ENCRYPTED PASSWORD 'finsight_secure_pass';
GRANT ALL PRIVILEGES ON DATABASE finsight_db TO finsight_user;
```

### 2. Environment Configuration

Set database connection parameters in `.env` (copy from `.env.example`):
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=finsight_db
DB_USER=finsight_user
DB_PASSWORD=finsight_secure_pass
DB_SCHEMA=public
```

### 3. Schema Setup & Data Modeling

Apply the core analytics schema. The script is **non-destructive and idempotent**: it only uses
`CREATE ... IF NOT EXISTS`, so re-running it against a loaded database changes nothing.
```bash
# Using psql
psql -h localhost -p 5432 -U finsight_user -d finsight_db -f sql/schema/01_schema_definition.sql

# Or using the loader CLI's automated setup flag
py src/finsight/load_to_postgres.py --create-schema
```

> **Destructive reset (manual only):** `sql/admin/reset_schema.sql` drops `fct_transactions`, `dq_audit_log`
> and the four analytical views. Nothing in the codebase runs it. Use it only when you intend to rebuild the
> database from scratch.

**Schema Design Highlights (`fct_transactions`):**
- **Surrogate Primary Key:** `transaction_id BIGSERIAL PRIMARY KEY` avoids flaky composite keys and supports non-unique customer/destination account names.
- **Constraints:** Positive amounts (`amount >= 0.00`), valid PaySim rails (`type IN ('CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT')`), and binary flags (`is_fraud IN (0, 1)`).
- **Targeted Indexes:** Composite indexes (`idx_fct_type_fraud`, `idx_fct_hour_type_fraud`) tailored specifically for volume and fraud rate analytics.

### 4. Bulk Loader Execution (`load_to_postgres.py`)

The loader streams compressed records in configurable chunks into PostgreSQL using high-throughput `COPY`.

```bash
# Test connection to database
py src/finsight/load_to_postgres.py --check-connection

# Dry-run validation (verifies file, types, and chunking without database writes)
py src/finsight/load_to_postgres.py --dry-run

# Load a sample for verification (e.g. 50,000 rows)
py src/finsight/load_to_postgres.py --max-rows 50000 --create-schema --truncate

# Load full processed dataset (6.36 million rows)
py src/finsight/load_to_postgres.py --chunk-size 100000 --truncate
```

**Rerun Safeguards:**
The loader refuses to silently append or double existing records. If the target table already contains data, execution halts with a `TableExistsError` unless `--truncate` is explicitly passed. A `--truncate` reload is atomic: `TRUNCATE` and `COPY` run in one transaction, so a failed reload rolls back to the previous rows.

### 5. SQL Analytics Pack (`sql/analytics/`)

Execute the analytics queries via `psql`:

```bash
# 01. Overall volume, gross value, distinct accounts, and simulated fraud totals
psql -h localhost -U finsight_user -d finsight_db -f sql/analytics/01_executive_summary.sql

# 02. Volume, gross value, and percentage shares by payment rail
psql -h localhost -U finsight_user -d finsight_db -f sql/analytics/02_transaction_types.sql

# 03. Simulated fraud counts and incidence rates by transaction rail
psql -h localhost -U finsight_user -d finsight_db -f sql/analytics/03_fraud_by_type.sql

# 04. Gross financial exposure and loss share associated with simulated fraud
psql -h localhost -U finsight_user -d finsight_db -f sql/analytics/04_fraud_loss_share.sql

# 05. Diurnal transaction volume and monetary value by hour of day (0-23)
psql -h localhost -U finsight_user -d finsight_db -f sql/analytics/05_hourly_volume.sql

# 06. Hourly simulated fraud rates across high-risk rails (CASH_OUT, TRANSFER)
psql -h localhost -U finsight_user -d finsight_db -f sql/analytics/06_hourly_fraud_patterns.sql

# 07. Ranked risk segments by simulated fraud rate with sample-size reliability guardrails
psql -h localhost -U finsight_user -d finsight_db -f sql/analytics/07_ranked_risk_segments.sql
```

### 6. Running Automated Tests

```bash
py -m pytest tests/ -v
```

> `tests/test_sql_views.py::test_live_postgres_view_reconciliation_if_available` runs
> `CREATE OR REPLACE VIEW` against the database configured in `.env` when it is reachable.

---

## Feature Engineering & Explainable Risk Engine (Milestone 3)

FinSight Milestone 3 implements a leak-free, reproducible feature-engineering pipeline ([`src/finsight/features.py`](file:///src/finsight/features.py)) and a deterministic, explainable rule engine baseline ([`src/finsight/risk_rules.py`](file:///src/finsight/risk_rules.py)).

### 1. Feature Engineering & Strict Governance Boundary

The feature matrix $X$ is deliberately conservative, mathematically stable, and isolated from the evaluation target $y$:

| Feature | Type | Source / Formula | Analytical & Modeling Rationale |
| :--- | :--- | :--- | :--- |
| `amount` | Float | Raw transaction amount | Core scale signal for transaction value. Verified non-negative. |
| `log1p_amount` | Float | $\ln(1 + \text{amount})$ | Stabilizes extreme right-skewed heavy tails in financial amounts. Maps $0.0 \to 0.0$. |
| `transaction_hour` | Int | $(\text{step} - 1) \bmod 24$ | Diurnal hour of day (0–23). Captures overnight inactivity vs peak business hours. |
| `transaction_day` | Int | $((\text{step} - 1) // 24) + 1$ | 1-indexed simulation calendar day (1–31). Tracks macro weekly volume cycles. |
| `is_merchant_dest` | Int | $1 \text{ if } \text{name\_dest starts with 'M'} \text{ else } 0$ | Indicator for merchant payees (zero fraud in PaySim, zero merchant destination balance). |
| `type_CASH_IN` | Int | One-hot binary (0 or 1) | Inbound account cash deposits (0% simulated fraud). |
| `type_CASH_OUT` | Int | One-hot binary (0 or 1) | Outbound physical cash withdrawals (50.1% of all fraud cases). |
| `type_DEBIT` | Int | One-hot binary (0 or 1) | Direct bank debit transactions (0% simulated fraud). |
| `type_PAYMENT` | Int | One-hot binary (0 or 1) | Customer-to-merchant POS payments (0% simulated fraud). |
| `type_TRANSFER` | Int | One-hot binary (0 or 1) | P2P fund transfers (49.9% of all fraud cases; primary exfiltration rail). |

#### Why Prohibited Columns Are Excluded from Model Features:
1. **`is_fraud` (Target Variable)**: Ground-truth label. Preserved strictly as prediction target $y$. Inclusion in $X$ would constitute 100% target leakage.
2. **`is_flagged_fraud` (Pre-existing Rule Flag)**: A simplistic simulator heuristic flag (>200,000 on TRANSFER) that catches only 16 frauds and misses 8,197 frauds (99.8% false negative rate). Excluded to prevent circular rule-leakage.
3. **`name_orig` & `name_dest` (Raw Identifiers)**: 99.85% of origin accounts appear exactly once. High-cardinality categorical encoding would cause severe overfitting and memorization of training-set IDs without generalizable signal.
4. **`oldbalance_org`, `newbalance_orig`, `oldbalance_dest`, `newbalance_dest` (Balance Fields)**:
   - *Synthetic Artifacts:* 33% of origin accounts start with zero balance; 100% of merchant destinations have zero balance.
   - *Severe Liquidation Leakage:* 98.05% of simulated fraud transactions liquidate the origin account to exactly $0.00$. Including these fields would cause the model to learn simulator balance artifacts rather than real financial patterns.
5. **`transaction_id` & `created_at` (Database Metadata)**: Surrogate keys and database timestamps carry zero predictive real-world meaning.

Enforced programmatically: [`validate_no_leakage_columns`](file:///src/finsight/features.py) raises [`FeatureEngineeringError`](file:///src/finsight/features.py) if any prohibited column enters the candidate feature matrix.

---

### 2. Explainable Rule Engine (`risk_rules.py`)

The rule engine computes deterministic risk scores (0–100) and actionable reason codes. Rules are investigation signals, not confirmed fraud decisions:

$$\text{Risk Score} = \min\left(100, \sum_{i \in \text{Triggered Rules}} \text{Risk Points}_i \right)$$

| Rule ID | Rule Name | Points | Trigger Logic | Rationale & Empirical Limitations |
| :--- | :--- | :---: | :--- | :--- |
| `RULE_001_HIGH_VALUE` | Excessive Single Amount | 35 | `amount >= $200,000.00` | Fraudulent cash-outs and transfers average ~$1.47M vs ~$173k for legitimate. *Limitation:* Large legitimate B2B and payroll transfers also exceed this. |
| `RULE_002_DRAINAGE_RAIL` | Drainage Rail Exposure | 25 | `type IN ('TRANSFER', 'CASH_OUT')` | In PaySim, 100% of simulated fraud occurs on these two rails. *Limitation:* >99% of transfers/cash-outs are legitimate consumer activity. |
| `RULE_003_OVERNIGHT_DRAINAGE` | Off-Hours Activity | 25 | `hour IN (1..5) AND type IN ('TRANSFER', 'CASH_OUT') AND amount >= $10,000` | Legitimate consumer activity drops >99% overnight; automated scripts run continuously. *Limitation:* Legitimate night-shift/overseas transactions. |
| `RULE_004_LARGE_ROUND_AMOUNT` | Elevated Round Amount | 15 | `amount >= $50,000 AND amount % 5,000 == 0 AND type IN ('TRANSFER', 'CASH_OUT')` | Fraudulent extractions frequently target round limit ceilings. *Limitation:* Inter-bank round savings deposits are common. |

#### Operational Risk Tiers & Actions:
- **`LOW` (Score 0 – 29)**: Standard baseline risk; recommended action `AUTO_APPROVE`.
- **`MEDIUM` (Score 30 – 69)**: Elevated risk signal; recommended action `MANUAL_REVIEW` (step-up authentication or analyst review queue).
- **`HIGH` (Score 70 – 100)**: Concentrated anomaly indicators; recommended action `FLAG_SUSPICIOUS` (priority alert).

---

### 3. Planned Evaluation Strategy (Milestone 4 Blueprint)

Detailed specification: [`docs/evaluation-strategy.md`](file:///docs/evaluation-strategy.md).

- **Extreme Class Imbalance Protocol**: Fraud prevalence is 0.1291% (1:773.7 ratio). Relying on accuracy alone is strictly prohibited (all-zero baseline achieves 99.87% accuracy).
- **Core Evaluation Metrics**: Precision, Recall, F1-Score, Average Precision (PR-AUC), and complete Confusion Matrix.
- **Chronological Train/Test Partition**: Dataset spans simulation hours 1–743 (31 days). Random splitting is forbidden to avoid lookahead bias.
  - *Train Set:* Steps 1–550 (~first 23 days).
  - *Holdout Test Set:* Steps 551–743 (~last 8 days).
- **Strict Preprocessing Isolation**: All scalers, frequency encoders, and decision thresholds must be fitted on train data only, never leaking holdout test distributions.

---

### 4. Running Features, Rules, Models & Automated Tests

```bash
# Execute full test suite (180 tests collected; one live-database test applies views when PostgreSQL is reachable)
py -m pytest tests/ -v

# Run Python interactive verification of feature extraction
py -c "import pandas as pd; from finsight.features import extract_features; df = pd.DataFrame([{'amount': 1500.0, 'type': 'TRANSFER', 'step': 14, 'nameDest': 'C123'}]); print(extract_features(df))"

# Run Python interactive verification of explainable rule scoring
py -c "from finsight.risk_rules import RiskEngine; engine = RiskEngine(); res = engine.evaluate_transaction({'amount': 350000.0, 'type': 'TRANSFER', 'transaction_hour': 3}); print(res.to_dict())"

# Frozen champion at its locked threshold on validation (451–550) and the development holdout (551–650).
# The locked final test (step >= 651) is excluded. Writes reports/frozen_evaluation/.
py src/finsight/models.py --evaluate-frozen --model-path models/hist_gradient_boosting_no_day.joblib

# TreeSHAP explanations for the frozen champion on a development-holdout sample (551–650)
py src/finsight/explainability.py --data-path data/processed/transactions.csv.gz --model-path models/hist_gradient_boosting_no_day.joblib --sample-size 2000 --random-seed 42
```

---

## Explainable AI with SHAP (Milestone 5)

FinSight implements an axiomatic, tested explainability layer using **TreeSHAP** (Tree-based Shapley Additive exPlanations) to explain predictions from the frozen champion gradient-boosting model ([`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib)). SHAP attributions explain the model's score; they do not establish that a transaction is fraudulent.

### 1. What SHAP is & Why FinSight Uses It

SHAP connects cooperative game theory with machine learning by calculating Shapley values—the unique attribution values satisfying four fundamental axioms: **Efficiency / Additivity**, **Symmetry**, **Dummy / Null Player**, and **Additivity Across Ensembles**. 

Unlike heuristic importance metrics (such as Gini impurity or split counts) which suffer from consistency defects, TreeSHAP computes **exact, additive contributions** without Monte Carlo sampling approximations for tree ensembles. In FinSight:
- **Efficiency / Additivity holds exactly**: The model's raw decision margin (log-odds $\text{logit}(p) = \log\frac{p}{1-p}$) is identical to the base value $\phi_0$ plus the sum of feature attributions $\sum_{j=1}^M \phi_j(x)$ (empirical max reconstruction error $\le 1.34 \times 10^{-14}$).
- **Operational Auditing**: Every transaction receives a transparent breakdown of which financial and temporal signals pushed its risk score above or below the locked alert threshold ($t = 0.9660$).

### 2. How to Run the Explainability Workflow

```bash
# Execute full explainability pipeline on a development-holdout sample (steps 551–650, seed 42)
py src/finsight/explainability.py --data-path data/processed/transactions.csv.gz --model-path models/hist_gradient_boosting_no_day.joblib --sample-size 2000 --random-seed 42

# Run explainability test suite
py -m pytest tests/test_explainability.py -v

# Run full project test suite (180 tests collected; one live-database test applies views when PostgreSQL is reachable)
py -m pytest tests/ -v
```

### 3. Generated Report Paths

All explainability outputs are strictly isolated under `reports/explainability/`:
- **Global Feature Importance CSV**: [`reports/explainability/global_shap_importance.csv`](file:///reports/explainability/global_shap_importance.csv)
- **Global SHAP Beeswarm Summary Plot**: [`reports/explainability/global_shap_summary.png`](file:///reports/explainability/global_shap_summary.png)
- **Local Transaction Explanations JSON**: [`reports/explainability/local_explanations.json`](file:///reports/explainability/local_explanations.json)
- **Human-Readable Markdown Report**: [`reports/explainability/explainability_report.md`](file:///reports/explainability/explainability_report.md)
- **Audit Run Metadata**: [`reports/explainability/explainability_metadata.json`](file:///reports/explainability/explainability_metadata.json)

### 4. How to Interpret Global & Local Explanations

#### Global Explanations (Population Perspective)
Global importance is quantified by the **mean absolute SHAP value** across a development-holdout sample ($N = 2,000$ transactions from steps 551–650; this window was inspected during earlier model selection, and the locked final test is excluded):
1. **`type_PAYMENT` (Rank 1, 27.4% share)** & **`type_CASH_IN` (Rank 3, 23.5% share)**: These non-fraud rails generate large negative SHAP values, strongly pulling down transaction risk away from the baseline prior.
2. **`transaction_day` (Rank 2, 24.3% share)**: Modulates risk across simulated monthly cycles.
3. **`type_CASH_OUT` (Rank 4, 14.0% share)** & **`type_TRANSFER` (Rank 5, 4.6% share)**: The primary positive structural rails where fraud occurs.
4. **`amount` & `log1p_amount` (Ranks 7 & 8, ~3.4% combined share)**: Act as continuous magnitude escalators that push high-value cash-outs past the alert threshold.

#### Local Explanations (Per-Transaction Audit)
Local explanations decompose an individual score into additive feature contributions relative to the prior base value ($\phi_0 = -3.5265 \implies 2.86\%$ uncalibrated model score):
- **Observed Attributes vs. Feature Attributions**: Local audit payloads cleanly isolate human-readable transaction properties (observed rail `TRANSFER`, amount `$6.27M`, simulated day `25`, hour `14`) from mathematical one-hot feature contributions.
- **One-Hot Encoding Semantics**: Under one-hot encoding, indicators like `type_PAYMENT = 0` or `type_CASH_IN = 0` represent the absence of safe payment/deposit rail suppression rather than separate real-world criminal acts.
- **High-Risk Alert (e.g. True Positive CASH_OUT, Amount $6.27M, Score 99.49%)**: Pushed higher into alert status by non-suppression flags `type_CASH_IN = 0` (+1.979) and `type_PAYMENT = 0` (+1.850), amplified by large transaction magnitude `amount = $6.27M` (+1.309).
- **Normal Transaction (e.g. True Negative CASH_IN, Amount $247k, Score 0.33%)**: Suppressed safely below threshold by active non-fraud rail `type_CASH_IN = 1` (-2.663) and non-cashout flag `type_CASH_OUT = 0` (-0.737).

### 5. Governance: Feature Attribution vs. Real-World Causation

> [!WARNING]
> **Attribution is Not Causation**:
> SHAP values quantify the **statistical dependence learned by the decision tree ensemble on historical training observations**. They answer: *"By how many log-odds points did this feature shift the model's score compared to the average background transaction?"*
> 
> SHAP values do **NOT**:
> 1. Prove real-world criminal intent or legal guilt.
> 2. Demonstrate physical cause-and-effect (e.g. lowering the amount or changing the timestamp in real life does not guarantee an underlying transaction is harmless).
> 3. Account for latent confounders, unobserved customer account history, or out-of-distribution adversarial evasion.

### 6. Limitations of Explanations on PaySim Synthetic Data

1. **Simulator Rule Artifacts**: PaySim simulates fraudulent behavior exclusively through automated agent scripts that transfer money and immediately cash out (`TRANSFER` $\to$ `CASH_OUT`). Models trained on PaySim heavily reward rail exclusion (`type_PAYMENT = 0`), which reflects model encoding mechanics rather than generalizable banking dynamics.
2. **Ablation of Collinear Merchant Flag**: The ablated model artifact excludes `is_merchant_dest` because PaySim strictly flags merchant destinations on `PAYMENT` transactions (where fraud is 0%). Leaving it in would create redundant, collinear attributions without incremental predictive power ($\Delta\text{PR-AUC} = +0.0000$).
3. **Operational Scope Notice**: FinSight is an educational, portfolio-grade analytics prototype. It is **not production-ready**, has not undergone algorithmic fairness audits for protected classes, and should not be used as an automated credit or adverse-action decision system.

---

## Evaluation Protocol & Reproducible Training (P0)

All model evaluation follows `src/finsight/evaluate.py`:

- **Development data:** steps 1–650. Training, tuning, thresholds and feature decisions use only these steps.
- **Locked final test:** steps 651–743 (80,831 transactions, 1,022 labelled fraud). It is scored once per promoted
  model through an explicitly acknowledged call. Enforcement: `evaluate.py`'s evaluation helpers and `train.py` raise
  `LockedTestAccessError` on these rows; frozen evaluation (`models.py`) and SHAP sampling (`explainability.py`) stop at
  step 650; the legacy sprint steps in `ml_experiments.py` refuse to run without `allow_locked_test_access=True`.
  It was previously visible inside the old holdout (steps ≥ 551), so it is not pristine.
- **Walk-forward folds:** train ≤350 → test 351–450; ≤450 → 451–550; ≤550 → 551–650.
- **Comparisons at fixed review capacity** (alerts per simulated day), because fraud prevalence rises ~13× across the
  simulation. Precision and F1 are only comparable within a fold. Rows tied at the cut-off score count at their
  expected value under random tie-breaking. Uncertainty uses a step-block bootstrap (test-sample variation only).
- **Selection disclosure:** each training config records which periods influenced its selection. The champion was
  chosen using steps 451–743, so folds F2 and F3 are optimistic for it; F1 is the least affected.
- **Scores are risk scores**, not calibrated probabilities.

```bash
# Reproduce the frozen champion bit-for-bit and verify it against models/hist_gradient_boosting_no_day.joblib.
# Writes a local, git-ignored bundle to models/candidates/ (never overwrites). The committed
# models/candidates/hgb_baseline8_balanced.manifest.json records the config, data hash, library versions and the
# successful verification. Its artifact_sha256 refers to that run's bundle, which embeds a save timestamp, so a
# retrain is verified by --verify-against (identical scores and threshold), not by file hash.
py src/finsight/train.py --config hgb_baseline8_balanced --verify-against models/hist_gradient_boosting_no_day.joblib

# Walk-forward evaluation of a training config (writes reports/evaluation/walk_forward_<config>.{json,md})
py src/finsight/evaluate.py --config hgb_baseline8_balanced

# Regenerate the benchmark view SQL and its source rows from artifacts and data
py src/finsight/benchmark_view.py
```

Walk-forward baseline for the champion configuration (`reports/evaluation/walk_forward_hgb_baseline8_balanced.md`):

| Fold | Test steps | Fraud prevalence | AP | AP 95% CI | Recall at 100 alerts/day |
|---|---|---:|---:|---|---:|
| F1 | 351–450 | 0.108% | 0.3889 | 0.316–0.465 | 28.2% |
| F2 | 451–550 | 0.666% | 0.3115 | 0.249–0.379 | 21.0% |
| F3 | 551–650 | 0.973% | 0.4222 | 0.329–0.523 | 24.9% |

F2 and F3 are optimistic for this configuration (see the selection disclosure above).

### Recipient history feature (P1.2)

`features.compute_recipient_prior_in_count` counts the transactions a recipient (`nameDest`) received in **strictly
earlier steps**. P1.1 error analysis found it to be the only non-artifact signal that separates the champion's missed
fraud from similar legitimate transfers (63% of fraud goes to a recipient with no prior incoming transactions,
against about 15% of legitimate TRANSFER/CASH_OUT).

Leakage boundaries, each covered by `tests/test_recipient_history.py`:

- Only `nameDest` and `step` are read. Labels, amounts and balances never are.
- Every row of the current step is excluded, not only the row itself, because row order within an hourly step is
  not a timestamp.
- A row's value depends only on earlier steps, so later rows never change it. Unseen recipients get 0.
- History is computed on the whole chronological frame (`train.load_development_data` / `add_history_features`)
  before any training, validation or fold slicing. `train.build_features` refuses a frame without it rather than
  recomputing on a window.

P1.3 comparison configs (all use the champion's float32 amount path and hyperparameters; none promoted):

| Config | Change from champion |
|---|---|
| `hgb_baseline8_cw50_f32` | class weight 1:50 only (comparability baseline) |
| `hgb_baseline8_rcpt_balanced` | + `recipient_prior_in_count` |
| `hgb_baseline8_rcpt_cw50` | + `recipient_prior_in_count`, class weight 1:50 |
| `hgb_baseline7_rcpt_cw50` | + `recipient_prior_in_count`, class weight 1:50, without `log1p_amount` |

No fold is independent evidence for this feature. It was discovered by analysing the champion's errors on
steps 451–650 (which includes F3) and screened on F1 and F2. In P1.3, F1 and F2 are screening folds and F3 only
confirms shortlisted candidates; F3 is the least-used fold, not an independent one. The locked test (651–743)
stays sealed during P1.3 and is not used as evidence; it is reserved for a single approved final evaluation and is
itself not pristine, because it sat inside the pre-P0 holdout.

### Candidate comparison (P1.3)

`src/finsight/compare_candidates.py` runs the five comparison configs through identical folds and metrics in two
recorded stages. Screening on F1 and F2 applies a shortlist rule fixed in code before F3 is touched; confirmation
then reads that shortlist and evaluates it on F3. Neither stage overwrites an existing report, writes a model file or
reads steps 651–743.

```bash
py src/finsight/compare_candidates.py --stage screening      # reports/evaluation/p13_screening.{json,md}
py src/finsight/compare_candidates.py --stage confirmation   # reports/evaluation/p13_confirmation.{json,md}
```

Average precision by fold (F1 and F2 are screening results; F3 is a confirmation result and not independent):

| Config | F1 | F2 | F3 | Recall at 100 alerts/day (F1 / F2 / F3) |
|---|---:|---:|---:|---|
| E0 champion config | 0.3889 | 0.3115 | 0.4222 | 28.2% / 21.0% / 24.9% |
| E1 + recipient count | 0.4952 | 0.4717 | 0.6249 | 30.8% / 24.5% / 30.4% |
| E2 class weight 1:50 | 0.4189 | 0.3663 | 0.4621 | 30.6% / 24.6% / 28.0% |
| E3 + recipient count, class weight 1:50 | 0.5092 | 0.5162 | 0.6530 | 32.8% / 31.0% / 33.3% |
| E4 = E3 without `log1p_amount` | 0.5092 | 0.5162 | 0.6530 | identical to E3 |

No model has been promoted. The locked test has not been read.

### Alert policy (P1.4)

`src/finsight/alert_policy.py` tests whether the P1.3 ranking result works as a prospective policy for E4. In each fold,
the model is trained on all but the last 100 training steps. Thresholds are fixed on those 100 steps and applied
unchanged to the test window. The policy is chosen on F1 and F2 by a rule declared in code, and F3 reports it once
(its second use, after P1.3).

```bash
py src/finsight/alert_policy.py --stage selection      # reports/evaluation/p14_selection.{json,md}
py src/finsight/alert_policy.py --stage confirmation   # reports/evaluation/p14_confirmation.{json,md}
```

Chosen policy: **B, one fixed threshold** (target 100 alerts/day). Neither hour-band split passed the rule in both F1
and F2.

| Fold | Realised alerts/day | Whole-day range | Precision | Recall |
|---|---:|---|---:|---:|
| F1 | 114.5 | 106–144 | 0.864 | 35.6% |
| F2 | 64.1 | 62–69 | 0.936 | 24.8% |
| F3 (confirmation) | 97.9 | 81–131 | 0.865 | 31.4% |

A fixed threshold does not hold the review load steady: it ran 15% over target in F1 and 36% under in F2, because
transaction volume and the score distribution shift between windows. Results are development-only; no model has been
promoted and the locked test has not been read.

### Priority review queue (P1.4.2)

`src/finsight/review_queue.py` simulates a real-time review queue for E4:
- Transactions scoring at or above a floor enter a queue.
- Analysts review K/24 items per hour, always the highest score first (oldest first among equal scores).
- An item that waits longer than its deadline (6, 12 or 24 hours) expires unreviewed.
- An optional escalation sends the very highest scores to an automatic hold.

The P1.4.1 investigation (`reports/evaluation/p141_rolling_cap.md`) motivated this design. As in P1.3 and P1.4, a
rule fixed in code chooses on F1 and F2, then F3 reports the choice once (its fourth use).

```bash
py src/finsight/review_queue.py --stage selection      # reports/evaluation/p142_selection.{json,md}
py src/finsight/review_queue.py --stage confirmation   # reports/evaluation/p142_confirmation.{json,md}
```

Recall at 100 reviews/day (whole-day load CV in brackets):

| Policy | F1 | F2 | F3 (confirmation) |
|---|---|---|---|
| B fixed threshold | 35.6% (0.15, 114/day) | 24.8% (0.05, 64/day) | 31.4% (0.21) |
| D2 threshold + first-come cap | 32.0% (0) | 24.8% (0.05, 64/day) | 28.6% (0.09) |
| Q12 queue, 12-hour deadline (chosen queue) | 31.2% (0.02) | 34.2% (0) | 33.8% (0.03) |

The rule chose Q12. It did **not** replace B as the reference policy, because on F1 its recall was 0.8 points below
D2. Results are development-only; nothing has been promoted.

### Sealed final test (P1.5)

`src/finsight/final_test.py` scored the sealed steps 651–743 exactly once:
- **Pre-registered first.** The code, configs, data hash and decision rule were committed in
  `reports/final_test/p15_preregistration.json` before any sealed label was read.
- **Same setup as development.** Every model was trained on steps 1–550, with thresholds and the queue
  calibrated on 551–650.
- **Reproduction check passed.** Before the sealed labels were read, both models reproduced their P1.3
  F3 results (AP 0.6530 and 0.4222).

| System (steps 651–743) | AP (95% CI) | Recall at 100 alerts/day | Precision at 100/day |
|---|---|---:|---:|
| E4 `hgb_baseline7_rcpt_cw50` | 0.697 (0.638–0.765) | 36.8% | 0.969 |
| E0 champion configuration, retrained the same way | 0.496 (0.424–0.601) | 31.5% | 0.829 |
| Frozen champion artifact | 0.491 (0.420–0.600) | 30.6% | 0.807 |

AP(E4) − AP(E0) = +0.200 (95% CI +0.144 to +0.231). **The pre-registered rule says promote E4.** Promotion
itself (P1.6) needs separate approval. The sealed steps are not pristine: before P0 they sat inside the old
holdout used to choose the champion's features. Full report: `reports/final_test/p15_final_test.md`.

---

## Master ML Remediation & Reliability Phase (historical)

> **Superseded.** The reports from this phase are in `reports/archive/` and contain known errors
> (alert budgets about 4× too low, stale rule-engine figures, holdout-based model selection).
> See `reports/archive/README.md`. The summary below is kept for history only.

Prior to dashboard development, FinSight completed a rigorous, 9-phase technical and methodological audit ensuring the correctness, interpretability, and defensibility of all analytical components:

1. **Baseline Reproduction**: Exactly reproduced holdout metrics for the ablated gradient-boosting model ($AP = 0.3836$, $ROC = 0.9185$, $F1 = 0.4583$ at $t = 0.9900$) and heuristic rule engine ($Precision = 72.04\%$, $Recall = 18.72\%$).
2. **Threshold Selection Grid Discretization**: Resolved a grid discretization bug in `find_optimal_threshold()` by increasing candidate resolution from 100 to 1,000 steps ($0.001\text{--}0.999$), supporting explicit operational objectives (`f1`, `alert_constrained`, `recall_at_min_precision`) and deterministic tie-breaking. Uncovered that $t = 0.9900$ represents a capacity-constrained operational triage point ($\le 1,000$ validation alerts), while unconstrained validation F1 peaks at $t = 0.9830$.
3. **Temporal Distribution Shift & Feature Ablations**: Diagnosed that the apparent surge in fraud prevalence ($0.084\% \to 0.666\% \to 1.093\%$) is driven by a $>95\%$ collapse in legitimate volume after Day 17. Controlled ablations demonstrated that removing `transaction_day` eliminates an overfitted shortcut, boosting Holdout PR-AUC by $+16.8\%$ ($0.3836 \to 0.4479$).
4. **Feature Quality & Leakage Protection**: Empirically proved that post-transaction balance fields (`oldbalanceOrg`, `newbalanceOrig`) represent settlement lookahead and simulator artifacts ($98.05\%$ liquidation in fraud vs $0.00\%$ in legitimate), reaffirming their strict exclusion from ML training.
5. **SHAP Axiomatic Additivity & Semantics**: Verified TreeSHAP exact additivity on margin log-odds ($\max\text{error} \le 1.34 \times 10^{-14}$) and established strict non-causal explanation semantics.
6. **Probability Calibration Assessment**: Demonstrated that balanced loss weighting offsets the decision margin by $+7.08$ log-odds, outputting uncalibrated risk ranking scores rather than true probabilities.
7. **Rule Engine & Diurnal Behavior**: Empirically verified a $99.81\%$ drop in overnight volume (from 6 PM to 3 AM) and a $422\times$ overnight fraud concentration.
8. **Financial Exposure Grounding**: Discovered paired simulator extractions that double-count fraud values, distinguishing gross simulated exposure ($\$12.06\text{B}$) from net economic leakage ($\approx \$6.03\text{B}$).
9. **Dashboard-Readiness Gate**: All 16 exit criteria satisfied; officially certified as `READY_FOR_DASHBOARDING`.

---

## Documentation & Remediation Index

### Core Documentation
- [Product Requirements Document](file:///docs/product-requirements.md)
- [System Architecture & Design Decisions](file:///docs/architecture.md)
- [Data Dictionary & Schema Specification](file:///docs/data-dictionary.md)
- [Metric Dictionary & KPI Definitions](file:///docs/metric-dictionary.md)
- [Risk Methodology & Scoring Engine](file:///docs/risk-methodology.md)
- [Evaluation Strategy & ML Governance](file:///docs/evaluation-strategy.md)
- [Walk-forward Evaluation (generated)](file:///reports/evaluation/walk_forward_hgb_baseline8_balanced.md)
- [Archived Reports Index & Known Errors](file:///reports/archive/README.md)
- [Model Explainability Report (TreeSHAP)](file:///reports/explainability/explainability_report.md)
- [Engineering Status & Known Limitations](file:///docs/findings-and-limitations.md)

### Archived Remediation Reports (historical, superseded)
All former `reports/remediation/`, `reports/ml_improvement/` and `reports/model_evaluation/` files now live under
`reports/archive/`. Read `reports/archive/README.md` before citing any figure from them.

---

## Project Status & Roadmap

| Stage | Focus Area | Status | Deliverables |
| :---: | :--- | :---: | :--- |
| **1** | **Engineering Initialization** | **Complete** | Repo structure, documentation suite, configuration contracts, git hygiene. |
| **2** | **PostgreSQL Data Warehouse Modeling & Import** | **Complete** | DDL schema, 6.36M row streaming COPY bulk loader, rerun safeguard, 7 SQL analytics queries. |
| **3** | **Feature Engineering & Explainable Rule Engine** | **Complete** | Leak-free feature extraction (`features.py`), explainable rule baseline (`risk_rules.py`), evaluation blueprint (`evaluation-strategy.md`). |
| **4** | **Supervised Machine Learning Baselines** | **Complete** | Chronological 3-way partition (`models.py`), Dummy, Logistic Regression, HistGradientBoosting, feature ablation, locked threshold evaluation. |
| **5** | **Explainable AI with SHAP** | **Complete** | TreeSHAP module (`explainability.py`), exact additivity verification, global ranking CSV/PNG, local TP/TN/FP/FN JSON & Markdown reports. |
| **Audit** | **Master ML Remediation & Verification** | Superseded | Model freeze (`no_day`); reports archived with known errors (see `reports/archive/README.md`). |
| **P0** | **Evaluation Integrity & Data Safety** | **Complete** | Non-destructive schema, atomic reload, locked evaluation protocol, bit-for-bit reproducible champion training, generated benchmark view, 123 tests. |
| **P1.2** | **Recipient History Feature** | **Complete** | Leakage-safe recipient prior count (`features.py`), history-aware training configs for P1.3 candidates, 12 leakage tests. |
| **P1.3** | **Candidate Comparison** | **Complete** | Five configs on identical folds; F1/F2 screening, pre-declared shortlist, F3 confirmation; nothing promoted. |
| **P1.4** | **Alert Policy** | **Complete** | Prospective thresholds for E4; fixed threshold (B) chosen on F1/F2, F3 reported once; volume drift documented. |
| **P1.4.1** | **Rolling Cap Investigation** | **Complete** | First-come caps lose fraud when the budget runs out; ranking within the day does not. Report only. |
| **P1.4.2** | **Priority Review Queue** | **Complete** | Queue with deadlines simulated; Q12 chosen; B stays the reference (F1 shortfall vs D2). |
| **P1.5** | **Sealed Final Test** | **Complete** | Pre-registered one-time run on steps 651–743: E4 AP 0.697 vs 0.496; rule says promote; promotion pending approval. |
| **6** | **SQL KPI Marts & Analytics Views** | Planned | Analytical views, window functions, operational KPI benchmarks. |
| **7** | **Power BI Reporting & Presentation** | Planned | Star-schema DAX model, executive summary, risk drill-down report. |

---

## Frozen Model Baselines Performance Summary (historical)

> These figures are on the **old holdout** (`step >= 551`, 196,601 transactions, 2,148 simulated frauds), which was
> used for model selection and is therefore not an unbiased test set. They are kept for history. Use the walk-forward
> results above and the generated `vw_risk_detection_benchmark` for current comparisons.

| Model | PR-AUC (AP) | ROC-AUC | Operating Threshold | Precision | Recall | F1 Score | Flagged Alerts |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Dummy Majority Baseline** | 0.0109 | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| **Rule Engine (Milestone 3)** | — | — | `HIGH` Tier (Score >= 70) | **0.7204** | 0.1872 | 0.2971 | 558 |
| **Logistic Regression (Balanced)** | 0.1805 | 0.9152 | 0.9009 (Val F1 optimal) | 0.2430 | 0.2914 | 0.2650 | 2,576 |
| **HistGradientBoosting (Baseline)** | 0.3836 | 0.9185 | 0.9900 (Val Capacity Constrained) | 0.6901 | 0.3431 | 0.4583 | 1,068 |
| **HistGradientBoosting (Baseline)** | 0.3836 | 0.9185 | 0.9830 (Val F1 Optimal) | 0.6052 | 0.3934 | 0.4767 | 1,396 |
| **HistGradientBoosting (No Day)** | **0.4479** | **0.9487** | 0.9660 (Val F1 Optimal) | 0.5774 | **0.4274** | **0.4912** | 1,590 |

---

## Engineering Standards & Interview Readiness

FinSight was architected as an interview-ready demonstration of software and data engineering craft:
- **Separation of Concerns**: Environment configuration (`.env`), business logic (`src/`), analytical queries (`sql/`), and testing (`tests/`) are decoupled.
- **Defensive Data Engineering**: No transaction data enters downstream analytics without validation against strict data quality contracts.
- **Explainability First**: Every risk score maps directly to deterministic rules, thresholds, and human-readable audit notes.
- **Reproducibility**: Parameterized pipelines, pinned dependencies, deterministic seeds, and portable SQL schemas guarantee identical execution across environments.


