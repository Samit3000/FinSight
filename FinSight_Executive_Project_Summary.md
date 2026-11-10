# FinSight — Executive Project Summary

> **Project status as of 19 October 2026** · Based on the project brief and implementation reports shared in this conversation

## Executive overview

**FinSight — Transaction Intelligence & Risk Analytics** is an end-to-end data analytics and machine-learning project designed around a fictional digital payments company. Its purpose is to transform raw transaction records into structured business intelligence, interpretable risk indicators, and machine-learning-based fraud predictions.

The project combines data engineering, relational database design, analytical SQL, supervised machine learning, explainability, and a planned Power BI reporting layer. Rather than treating fraud detection as an isolated classification task, FinSight aims to connect transaction-level data to operational decisions: which payment patterns warrant investigation, how risk varies across transaction types and time, and how a review team might prioritize alerts.

The current implementation uses Python and pandas for data processing, PostgreSQL for persistent storage and analytics, scikit-learn for machine learning, a configurable rule-based risk engine, and SHAP-based explainability. The Power BI layer is planned but has not yet been completed.

**Current assessment:** the data pipeline and SQL analytics are substantially developed, and a first ML experimentation cycle has identified a promising candidate model. The project is not yet a fully finished product, and its final model performance and operational claims need further validation.

| Project indicator | Current reported status |
|---|---|
| Source dataset | 6.36 million transaction records |
| Fraud-labelled records | 8,213 (approximately 0.1291%) |
| SQL analytics | Four views reported as deployed and reconciled |
| ML experiments | 14 configurations reported in the latest sprint |

*These figures are drawn from the project's reports. They have not all been independently reproduced in this conversation.*

## 1. The problem FinSight is trying to solve

Digital payments businesses process large volumes of transactions and need to answer two related sets of questions.

### Business and operational intelligence

- How many transactions are being processed, and what is their total value?
- How does transaction activity vary by payment type and simulated time?
- Which transaction segments contain disproportionate amounts of risk?
- How can analysts investigate individual records and understand why they were flagged?

### Fraud-risk prioritization

- Can historical transaction attributes help identify records labelled as fraudulent?
- How should a system balance missed fraud against false-positive alerts?
- Can a rule-based system provide transparent reasons for flagging transactions?
- Can a machine-learning model rank transactions by risk and help prioritize investigation?

FinSight brings these functions together in one workflow. The intended output is not an automated declaration that a transaction is fraudulent. It is an analytical system that provides evidence and prioritization signals for human review.

The project is a **historical-data prototype**, not a live payment-processing or production fraud-prevention system.

## 2. Dataset and data understanding

### Dataset selected

FinSight uses **PaySim — Synthetic Financial Datasets for Fraud Detection**, a simulated mobile-money transaction dataset.

The processed dataset reported in the project logs contains 6,362,620 records and 8,213 positive fraud labels, giving an overall labelled-fraud prevalence of approximately 0.1291%. The dataset's synthetic nature is central to interpreting every result: performance on PaySim does not establish performance against real financial crime.

### Dataset features

The original PaySim schema contains the following core fields. The descriptions below explain their role in the dataset; they do not imply that all fields are used as model inputs.

| Field | Meaning | Role in FinSight |
|---|---|---|
| `step` | Simulated time step, representing an hour in the simulation | Temporal analysis and split construction |
| `type` | Transaction type, such as `PAYMENT`, `TRANSFER` or `CASH_OUT` | Business analysis and model input |
| `amount` | Transaction amount | Core numerical input and transaction-value analysis |
| `nameOrig` | Originating account identifier | SQL and account-level analysis; excluded from the initial model inputs |
| `oldbalanceOrg` | Origin account balance before transaction | Available in raw data; excluded from the initial model inputs |
| `newbalanceOrig` | Origin account balance after transaction | Available in raw data; excluded from the initial model inputs |
| `nameDest` | Destination account identifier | SQL and account-level analysis; excluded from the initial model inputs |
| `oldbalanceDest` | Destination balance before transaction | Available in raw data; excluded from the initial model inputs |
| `newbalanceDest` | Destination balance after transaction | Available in raw data; excluded from the initial model inputs |
| `isFraud` | Dataset's fraud label | Target variable |
| `isFlaggedFraud` | Existing flag generated by the dataset's own rules | Excluded from model predictors to avoid relying on an existing detection signal |

The processed schema also includes derived fields such as `transaction_hour` and `transaction_day`. These support temporal analysis and feature engineering.

### Dataset-specific challenges

**Extreme class imbalance.** Only about 0.1291% of all records are fraud-labelled. A model that predicts every transaction as legitimate would achieve high accuracy while detecting no fraud. Accuracy alone is therefore unsuitable for evaluating this project.

**Limited fraud coverage.** The reports state that all fraud labels occur on `TRANSFER` and `CASH_OUT` transactions. This restriction matters because the model may learn strong patterns associated with these transaction types without necessarily learning more general fraud behaviour.

**Potential leakage and simulator artifacts.** Balance fields may encode post-transaction effects of the simulator's fraud-generation process. The initial model excludes them rather than risk inflated performance from artifacts that would not generalize. `isFlaggedFraud` is also excluded because it is an existing detection flag, not an independent predictive signal.

**Limited account history.** The reported analysis indicates that approximately 99.85% of origin accounts occur only once, and all 8,213 fraud-origin accounts transact once in the dataset. If verified, this sharply limits what account-history and behavioural features can be learned from the current data.

**Temporal distribution shift.** Fraud prevalence differs substantially between the chronological training, validation and holdout periods. That complicates model selection and may reveal a limitation of the dataset's simulated time structure.

The data is large, but size alone does not guarantee that it contains enough realistic behavioural information to support a highly generalizable fraud model.

## 3. System architecture

FinSight is structured as a batch-processing analytics pipeline rather than a real-time streaming system.

```text
┌─────────────────────────────────────────────────────────┐
│  1. SOURCE DATA                                          │
│  PaySim CSV · 6.36 million simulated transactions        │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  2. PYTHON INGESTION AND PROCESSING                       │
│  pandas · schema checks · validation · transformations  │
│  · data-quality report                                   │
└───────────────────────────┬─────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  3. POSTGRESQL STORAGE                                    │
│  fct_transactions · constraints · indexes · SQL analytics│
└───────────────────────────┬─────────────────────────────┘
                            ↓
              ┌─────────────┴─────────────┐
              ↓                           ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ 4A. RULE ENGINE           │  │ 4B. ML PIPELINE           │
│ Transparent heuristic    │  │ Classification ·          │
│ risk scores and flags    │  │ evaluation · SHAP         │
└─────────────┬────────────┘  └─────────────┬────────────┘
              └─────────────┬───────────────┘
                            ↓
┌─────────────────────────────────────────────────────────┐
│  5. BUSINESS INTELLIGENCE LAYER                           │
│  SQL views → planned Power BI dashboard → analyst insights│
└─────────────────────────────────────────────────────────┘
```

*Conceptual architecture reconstructed from the supplied implementation reports. The diagram does not assert that every component is already connected end to end.*

### Component responsibilities

**Python and pandas.** The data pipeline processes the raw dataset into a compressed processed file, checks data quality and creates derived fields. This provides a consistent input for database loading and machine learning.

**PostgreSQL.** The main fact table, `fct_transactions`, stores the transaction data. The reported schema includes a surrogate transaction key, integrity constraints and indexes intended to support common analytical queries.

**SQL analytics.** SQL is used to calculate transaction KPIs, temporal patterns, payment-rail breakdowns and risk-detection comparisons. This is the layer that translates raw records into business-facing measures.

**Rule-based risk engine.** Explicit heuristic rules flag transactions based on conditions such as high amounts, particular payment rails, simulated transaction hours and round amounts. The score is a prioritization heuristic, not a probability of fraud.

**Machine-learning pipeline.** The project builds and evaluates supervised binary classifiers. The main algorithm is scikit-learn's `HistGradientBoostingClassifier`, with alternative classifiers used as benchmarks. Model predictions and rule-based risk scores are separate signals.

**Explainability.** The project includes SHAP-based explainability tooling to help interpret model predictions. Its explanations need to be interpreted in the context of the features available to the model.

**Power BI.** This is the planned presentation layer for executive KPIs, payment performance, risk intelligence and transaction exploration. The SQL layer is reportedly ready to support it, but the dashboard itself remains unfinished.

The technology choices are appropriate for a portfolio project because they demonstrate a connected workflow across data processing, databases, ML and business reporting without requiring unnecessary streaming infrastructure.

## 4. Milestones achieved

The milestones below are reconstructed from the project brief and implementation reports. A reported milestone means the logs describe it as completed; it does not mean every artifact has been independently audited here.

### Milestone 1 — Data processing and quality pipeline

**Status: Reported complete**

**What was implemented**

- Python-based data processing using pandas.
- A processed, compressed transaction dataset.
- Data-quality reporting and derived temporal fields.
- A consistent dataset for downstream SQL and ML workflows.

**Why it matters**

This is the foundation of the project. Without a repeatable processing step, database results and model metrics could depend on inconsistent manual preparation.

**Remaining validation**

Confirm that row counts, target counts, data types, duplicate handling, missing-value handling and derived fields reconcile exactly with the source dataset. Dataset provenance and licensing should also be documented in the final repository.

### Milestone 2 — PostgreSQL data model and ingestion

**Status: Reported complete**

**What was implemented**

- PostgreSQL database `finsight_db`.
- A central `fct_transactions` table.
- A bulk loader in `src/finsight/load_to_postgres.py`.
- Chunked loading through PostgreSQL `COPY`, using batches of approximately 100,000 rows.
- Integrity constraints, indexes and safeguards against accidental repeated loads.
- A reported full import of 6,362,620 records.

The loader report describes transaction-safe loading, environment-based database configuration and safeguards against unintentionally reloading populated tables.

**Why it matters**

This moves FinSight beyond a notebook-based exercise. It establishes a persistent data layer, enables reusable SQL analysis and demonstrates practical ingestion engineering.

**Remaining validation**

The current schema should be reviewed for reproducibility and safe migrations. The schema SQL reportedly contains a `DROP TABLE IF EXISTS ... CASCADE` statement; even if the loader only invokes schema creation under specific conditions, destructive DDL should be separated from normal setup and documented clearly.

### Milestone 3 — SQL analytics and business metrics

**Status: Reported complete**

The project has a reported seven-query analytical SQL pack and four deployed analytical views:

- `vw_daily_transaction_kpis`
- `vw_hourly_risk_dynamics`
- `vw_payment_rail_performance`
- `vw_risk_detection_benchmark`

These views cover transaction KPIs, temporal activity, payment-rail performance and comparisons between risk-detection methods.

**Why it matters**

This milestone establishes the business-analytics side of FinSight. It allows the project to answer questions about volume, transaction value and risk patterns rather than only outputting a model score.

**Remaining validation**

The final report should define each KPI and its denominator, specify the simulated time basis, and reconcile the SQL outputs with the metrics displayed in Power BI. Flagged transaction value must not be presented as verified financial loss.

### Milestone 4 — Rule-based risk scoring

**Status: Implemented; benchmarking reported**

The rule engine includes four reported heuristics:

| Rule | Reported condition | Intended purpose |
|---|---|---|
| High value | Amount at least $200,000 | Prioritize unusually large transactions |
| Drainage rail | `TRANSFER` or `CASH_OUT` | Flag activity on rails where fraud labels occur in this dataset |
| Overnight drainage | Simulated hour 1–5, eligible rail, amount at least $10,000 | Flag transactions meeting a time-and-amount heuristic |
| Large round amount | Amount at least $50,000 and divisible by $5,000 on an eligible rail | Flag selected high-value round transactions |

The reported scoring system assigns heuristic points to matching rules, with separate medium- and high-risk groupings.

The sprint report gives the following historical benchmark results:

- High-risk-only rules: precision 72.04%, recall 18.72%, F1 0.2971, 558 alerts.
- Medium- and high-risk rules combined: precision 3.51%, recall 79.93%, 48,930 alerts.

**Why it matters**

Rules are transparent and straightforward to explain to an analyst. They also provide a useful baseline against which to assess the ML model.

**Remaining validation**

The rules should be compared with ML at matched alert volumes or a shared review budget. Otherwise, a high-recall ruleset that produces tens of thousands of alerts cannot be fairly compared with a model producing around a thousand alerts. The conditions also require careful interpretation because PaySim is synthetic and fraud labels are concentrated in two transaction types.

### Milestone 5 — Initial supervised ML pipeline

**Status: Reported complete**

The project introduced supervised binary classification, benchmark models, a frozen champion model, temporal evaluation and explainability tooling.

The reported model roster includes:

- `HistGradientBoostingClassifier` as the principal model.
- Logistic Regression as a simpler benchmark.
- Random Forest as an ensemble benchmark.
- Dummy Classifier as a naive reference point.

The frozen champion artifact is reported as `models/hist_gradient_boosting_no_day.joblib`.

Its eight input features are:

1. `amount`
2. `log1p_amount`
3. `transaction_hour`
4. `type_CASH_IN`
5. `type_CASH_OUT`
6. `type_DEBIT`
7. `type_PAYMENT`
8. `type_TRANSFER`

Account identifiers, balance fields and `isFlaggedFraud` are excluded from this initial feature set. `isFraud` remains the target, not an input.

**Why it matters**

The project has moved from merely calculating risk rules to learning a classification function from labelled data. It also has a foundation for comparing predictive performance and explaining the trade-offs between false positives and false negatives.

### Milestone 6 — ML improvement and experimentation sprint

**Status: Candidate identified; validation caveats remain**

The latest sprint added `src/finsight/ml_experiments.py` and `tests/test_ml_experiments.py`, and reports 14 experimental configurations. The experiments cover feature transformations, interactions, class weighting, undersampling and model complexity.

The most promising reported candidate is **EXP-07**, a HistGradientBoosting configuration with custom class weights `{0: 1.0, 1: 50.0}`. It retains the same eight baseline features.

The report gives this comparison:

| Metric | Frozen baseline | EXP-07 |
|---|---:|---:|
| Validation average precision | 0.3115 | 0.3663 |
| Validation F1 | 0.3773 | 0.4006 |
| Holdout average precision | 0.4479 | 0.4848 |
| Holdout ROC-AUC | 0.9487 | 0.9505 |
| Holdout precision | 57.74% | 71.26% |
| Holdout recall | 42.74% | 38.55% |
| Holdout F1 | 0.4912 | 0.5003 |
| Holdout alerts | 1,590 | 1,162 |

These are reported metrics, not independently reproduced results in this conversation.

**What changed technically**

Class weighting changes how the training algorithm penalizes errors involving the minority class. The validation-selected threshold of 0.7702 changes which scores become alerts. The threshold is not proof that model scores are calibrated probabilities.

**Why it matters**

EXP-07 is a promising candidate for a higher-precision, lower-alert-volume operating point. But it does not yet prove that FinSight has solved fraud detection or achieved a large breakthrough in underlying discrimination. The improvement in holdout F1 is modest, recall declines, and the candidate was evaluated on a holdout that was also used to evaluate other candidates.

The project brief and sprint reports together support the conclusion that the ML experimentation cycle has advanced, while independent final validation is still needed.

### Milestone 7 — Explainability and documentation

**Status: Tooling and reports exist; final audit pending**

The project includes SHAP explainability tooling and a reported collection of ML reports, experiment JSON files and visualizations covering benchmark performance, feature experiments and temporal robustness.

This is important for a portfolio project because it helps explain model behaviour rather than treating performance metrics as a black box.

The latest sprint report claims 95 repository tests passed. The available pasted log, however, does not show the complete pytest output needed to verify that claim independently. It also describes a manual recovery of `feature_experiments.json` after the file became malformed.

**Remaining work:** verify the actual test output, validate the recovered JSON against the original experiment results, and ensure all report tables agree with their underlying machine-readable artifacts.

### Milestone 8 — Power BI and end-to-end delivery

**Status: In progress**

The SQL views are reportedly deployed and ready for the dashboarding stage. The final Power BI report is not yet completed.

The planned dashboard should include:

- Executive Overview — volume, transaction value and trends.
- Payment Performance — breakdowns by transaction type and simulated time.
- Risk Intelligence — rule flags, model metrics and alert trade-offs.
- Transaction Explorer — transaction-level detail and reasons for flags.
- Data Quality — pipeline checks and validation outcomes, if supported by the available records.

This milestone is necessary to turn the existing technical components into a coherent, demonstrable analytics product.

## 5. Current ML evaluation design

The model uses a chronological split rather than a random split. The reported boundaries are:

| Split | Time steps | Records | Fraud labels | Fraud prevalence |
|---|---|---:|---:|---:|
| Training | `step ≤ 450` | 6,014,717 | 5,057 | 0.0841% |
| Validation | `451–550` | 151,302 | 1,008 | 0.6662% |
| Holdout | `step ≥ 551` | 196,601 | 2,148 | 1.0926% |
| Full dataset | All | 6,362,620 | 8,213 | 0.1291% |

The purpose is to simulate a more realistic scenario: train on earlier transactions, select the model and threshold on a later period, and evaluate on a still-later period.

The split has one important complication: fraud prevalence increases substantially across the periods. The sprint report attributes this to a sharp decline in simulated transaction volume after approximately Day 17 while fraud counts remain relatively high. That explanation should be verified against daily transaction and fraud counts before being treated as established.

The holdout was also used to compare multiple experimental candidates. It therefore cannot serve as a completely untouched final evaluation for the current model-selection cycle.

## 6. Major caveats and potential issues

These are the main issues I would resolve before describing FinSight as portfolio-ready.

### 1. Synthetic data limits real-world validity

PaySim simulates transactions and fraud behaviour. High scores on this dataset do not prove that the model would perform similarly on real payment data.

**Mitigation:** explicitly disclose the dataset's limitations. After auditing the current pipeline, consider a richer independent benchmark such as IEEE-CIS Fraud Detection, subject to its licensing, schema and evaluation constraints. Treat it as a separate benchmark rather than assuming its scores will be directly comparable.

### 2. Temporal distribution shift

Fraud prevalence differs significantly across the reported training, validation and holdout splits. Model performance may depend heavily on the simulated period.

**Mitigation:** plot daily volume and fraud prevalence, inspect performance across time segments, and use rolling or expanding-window evaluation where feasible. Establish whether the shift is expected dataset behaviour or a data-processing issue.

### 3. Holdout contamination through model selection

Multiple candidates were evaluated on the holdout, and those results informed the comparison. That weakens its status as an independent estimate of final performance.

**Mitigation:** freeze the next candidate using training and validation evidence, record its configuration, and evaluate it once on a fresh time period or independent dataset. Retain the existing results as exploratory holdout results.

### 4. Limited predictive features

The current model uses only eight features, primarily amount, simulated hour and transaction type. It has no explicit features describing sequences of activity or deviations from account behaviour.

**Mitigation:** investigate sender and recipient transaction history, transaction frequency over defined windows, amount deviations from historical behaviour, and repeated interactions between accounts—provided the data supports them and features are computed without using future information.

### 5. Operational metrics require reconciliation

The sprint report's alert-budget table gives approximately 38 candidate reviews per day, but 1,162 alerts over an eight-day holdout would equal approximately 145 alerts per day. The baseline figures have a similar mismatch.

**Mitigation:** recalculate the number of alerts using the actual evaluation period and clearly define the unit of review capacity. Compare models at the same alert budget or review rate.

### 6. Experiment-report integrity and reproducibility

The experiment JSON reportedly required manual recovery, and the pasted logs do not provide complete visible test output. Some report claims therefore need stronger evidence.

**Mitigation:** validate JSON schemas, regenerate reports from raw experiment outputs, record configurations and random seeds, preserve artifact hashes, and save the full test output. Ensure the report, metrics and model artifacts all agree.

### 7. Fraud-associated value is not verified financial loss

The reported transaction values are derived from synthetic transaction records and fraud labels. They do not establish actual net loss, recoverability or realized financial impact.

**Mitigation:** label these metrics precisely as fraud-labelled transaction value or simulated fraud-associated value. Do not present them as confirmed losses without supporting outcome data.

### 8. The business-facing product is unfinished

The database, SQL views and ML workflow are reported as implemented, but the Power BI dashboard and final end-to-end presentation remain incomplete.

**Mitigation:** build the dashboard only after the KPI definitions and model metrics are reconciled. Include filters, drill-through, risk explanations, data-quality indicators and clear warnings that flags are not proof of fraud.

## 7. Recommended roadmap from the current state

I would organize the remaining work into four gated milestones rather than immediately running another large batch of model experiments.

### 01 — Integrity and reproducibility audit

*First priority · Do not add new experiments yet*

Verify data counts, split boundaries, feature definitions, model artifacts, experiment JSONs, test results and alert-budget arithmetic. Document the holdout evaluations already performed.

### 02 — Model diagnosis and feature engineering

*Second priority · Improve information, not just tuning*

Analyze false negatives and false positives, identify the segments where the model fails, and test leakage-safe behavioural features. Define an explicit evaluation objective before choosing the next candidate.

### 03 — Independent evaluation and model decision

*Third priority · Establish defensible performance*

Freeze a candidate using validation results, evaluate it on fresh data, compare against the rules at matched alert volumes, and document where the model is and is not useful.

### 04 — Power BI and portfolio packaging

*Fourth priority · Deliver the actual product*

Complete the dashboard, add reproducible setup instructions and architecture documentation, publish evidence-backed findings, and prepare a concise demo explaining the business problem, technical decisions and limitations.

## 8. Final assessment

FinSight has made substantial progress in building the *infrastructure of an analytics product*: the data-processing pipeline, PostgreSQL ingestion, SQL analytics, rule engine, supervised ML workflow and experiment-reporting layer are reported as implemented.

The ML improvement sprint has also produced a promising candidate, but its reported performance should not yet be treated as an unbiased final result. The limited feature set, synthetic dataset, temporal distribution shift and incomplete operational validation remain important constraints.

The project's next stage should focus on making its results more trustworthy and its predictive features more informative, then completing the Power BI experience. More model complexity is not automatically the answer.

**The target end state:** a reproducible analytics product that can explain what happened in the transaction data, identify which records warrant investigation, justify how its model performs, and present its findings clearly to a business stakeholder—without overstating what synthetic data can prove.
