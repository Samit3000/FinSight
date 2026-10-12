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
├── config/
│   ├── pipeline.yaml                  # Ingestion & data quality pipeline parameters
│   └── risk_rules.yaml                # Explainable risk scoring rules & thresholds
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
│   ├── schema/                        # DDL tables and schema migration scripts
│   ├── views/                         # Materialized and analytical SQL views
│   └── analytics/                     # Ad-hoc investigation and KPI queries
├── src/
│   └── finsight/                      # Core package source code
│       └── __init__.py
├── tests/                             # Automated test suite (unit, integration, DQ)
│       └── __init__.py
├── reports/
│   └── data_quality/                  # Generated data validation audit reports
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

## Documentation Index

- [Product Requirements Document](file:///docs/product-requirements.md)
- [System Architecture & Design Decisions](file:///docs/architecture.md)
- [Data Dictionary & Schema Specification](file:///docs/data-dictionary.md)
- [Metric Dictionary & KPI Definitions](file:///docs/metric-dictionary.md)
- [Risk Methodology & Scoring Engine](file:///docs/risk-methodology.md)
- [Engineering Status & Known Limitations](file:///docs/findings-and-limitations.md)

---

## Project Status & Roadmap

| Stage | Focus Area | Status | Deliverables |
| :---: | :--- | :---: | :--- |
| **1** | **Engineering Initialization** | **Current** | Repo structure, documentation suite, configuration contracts, git hygiene. |
| **2** | **Synthetic Data Generation & Data Contracts** | Planned | Realistic FinTech dataset generator, Pydantic schemas, raw sample data. |
| **3** | **Data Ingestion & Validation Pipeline** | Planned | Automated validator, quarantine logic, DQ reporting engine, unit tests. |
| **4** | **PostgreSQL Modeling & Loading** | Planned | DDL scripts, staging/fact tables, loading logic, idempotency tests. |
| **5** | **Configurable Risk Scoring Engine** | Planned | Vectorized rule evaluation, reason code generation, risk tiering. |
| **6** | **SQL KPI Marts & Analytics** | Planned | Analytical views, window functions, operational KPI benchmarks. |
| **7** | **Power BI Reporting & Presentation** | Planned | Star-schema DAX model, executive summary, risk drill-down report. |

---

## Engineering Standards & Interview Readiness

FinSight was architected as an interview-ready demonstration of software and data engineering craft:
- **Separation of Concerns**: Configuration (`yaml`), business logic (`src/`), analytical queries (`sql/`), and testing (`tests/`) are strictly decoupled.
- **Defensive Data Engineering**: No transaction data enters downstream analytics without validation against strict data quality contracts.
- **Explainability First**: Every risk score maps directly to deterministic rules, thresholds, and human-readable audit notes.
- **Reproducibility**: Parameterized pipelines, pinned dependencies, deterministic seeds, and portable SQL schemas guarantee identical execution across environments.
