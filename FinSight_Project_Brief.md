# FinSight — Transaction Intelligence & Risk Analytics

## 1. Project Overview
**FinSight** is a portfolio-grade analytics platform for a fictional digital payments company. It combines transaction performance analytics, data quality monitoring, explainable risk rules, and machine-learning-based fraud classification.

**Goal:** Build a reproducible end-to-end project demonstrating Python, SQL, PostgreSQL, machine learning, data visualization, and business interpretation. This is a historical-data prototype—not a live payment processor or production fraud-prevention system.

## 2. Problem Statement
Payments teams need to understand transaction success and failure, identify operational patterns, and prioritise unusual transactions for investigation. FinSight will turn transaction records into reliable KPIs, explainable risk flags, ML predictions, and an interactive Power BI report.

## 3. Recommended Dataset
**Primary recommendation:** [PaySim — Synthetic Financial Datasets for Fraud Detection](https://www.kaggle.com/datasets/ealaxi/paysim1)

PaySim contains simulated mobile-money transactions, transaction attributes, balance-related fields, and an `isFraud` target. Its simulated nature must be disclosed in the report. Start with a manageable sample for development, then scale up if needed. Check the dataset's licence and actual schema before submission.

Use small synthetic test fixtures as well to test malformed records, duplicate IDs, unusual amounts, and repeated failures.

**Leakage warning:** Do not use `isFraud` as an input feature. Do not use `isFlaggedFraud` as a predictor for `isFraud`, because it is a pre-existing rule-based flag and could make evaluation misleading. Inspect all features for target leakage before training.

## 4. Core Technology Stack
- **Python + pandas:** ingestion, validation, transformation, and model training.
- **PostgreSQL:** persistent storage, constraints, and analytical queries.
- **SQL:** KPI definitions, trends, cohorts, and segment analysis.
- **scikit-learn:** preprocessing, classification models, and evaluation.
- **Power BI:** interactive performance and risk dashboards.
- **pytest:** automated tests.
- **Git/GitHub:** version control and portfolio presentation.

Keep the MVP simple. A separate web app, streaming system, cloud infrastructure, and orchestration platform are not required initially.

## 5. System Workflow
1. Ingest the dataset and preserve source/run metadata.
2. Validate schema, types, missing values, duplicates, and transaction identifiers.
3. Transform accepted data and load it into PostgreSQL using safe, repeatable logic.
4. Calculate business KPIs with documented SQL definitions.
5. Apply transparent, configurable rule-based risk checks.
6. Train and compare ML classifiers using a leakage-safe evaluation process.
7. Present performance, risk flags, model metrics, and transaction details in Power BI.
8. Document findings, limitations, and reproducible setup instructions.

## 6. Recommended ML Algorithms
Use **supervised binary classification** to predict the dataset's labelled `isFraud` outcome.

### A. Logistic Regression — Baseline
Estimates the probability of fraud from transaction features. It provides a relatively simple benchmark and helps explain the effects of features. Consider scaling where appropriate and use class-weighting or other imbalance handling only within the training process.

### B. Decision Tree Classifier — Interpretable Model
Learns non-linear feature-based decision rules. It is useful for demonstrating tree splits, depth control, overfitting, and interpretability. Tune or constrain the tree rather than allowing unrestricted growth.

### C. Random Forest Classifier — Ensemble Comparison
Combines many decision trees to improve generalisation compared with a single tree in some settings. It provides a useful ensemble comparison and can capture non-linear relationships.

**Recommended comparison:** Train all three using the same leakage-safe train/test strategy, compare results, and select based on evidence—not assumptions. Do not claim Random Forest is best before testing.

**Optional extension:** Isolation Forest for anomaly detection, only if the course requirements justify an unsupervised method. It is not required for the MVP.

## 7. ML Methodology and Evaluation
- Explore class distribution and missingness before modelling.
- Split data before fitting transformations; use a stratified split for classification where appropriate. If timestamps or transaction sequences create temporal dependence, consider a time-based split instead.
- Fit preprocessing only on training data.
- Avoid target leakage and document feature exclusions.
- Fraud datasets are often highly imbalanced, so accuracy alone is inadequate.
- Report **precision, recall, F1-score, PR-AUC / average precision, and a confusion matrix**.
- Discuss false positives and false negatives. Choose a classification threshold based on the intended review workflow and measured trade-offs.
- Report model limitations honestly. Performance on simulated data does not establish real-world fraud-detection performance.

## 8. Explainable Rule-Based Risk Engine
Keep rule-based flags separate from ML predictions. Candidate rules—only when supported by available fields—include:
- High transaction amount against a configurable threshold.
- Unusually frequent activity within a time window.
- Repeated failed attempts for an available entity.
- Duplicate transaction/source identifiers as a data-integrity alert.
- Deviation from historical behaviour where enough history exists.

Every flag should include a rule ID/version, triggering evidence, threshold or condition, and evaluation time. A flagged or model-predicted transaction is **not confirmed fraud** unless supported by trustworthy outcome labels or review.

## 9. Core Business Metrics
- Total distinct transactions.
- Transaction value, with currency and refund/reversal rules documented.
- Successful transactions and success rate.
- Failed transactions and failure rate, with a clear denominator.
- Average transaction value.
- Distinct flagged transaction count and flagged transaction rate.
- Flagged transaction value, explicitly not described as financial loss.
- Data rejection rate and duplicate counts.

Define each KPI once and ensure SQL and Power BI agree. Do not report revenue, chargeback losses, fraud rate, or detection accuracy unless the required fields and labels support them.

## 10. Power BI Pages
1. **Executive Overview:** volume, value, success rate, trends, and flagged-transaction overview.
2. **Payment Performance:** success/failure trends and breakdowns by available payment method, merchant, and time.
3. **Risk Intelligence:** distinct flagged transactions, rule counts, risk trends, and ML evaluation summaries where suitable.
4. **Transaction Explorer:** filtered transaction details, rule explanations, and predictions.
5. **Data Quality (optional):** rejected rows, duplicate counts, validation issues, and pipeline-run history.

Use clear denominators, date filters, data freshness, drill-through, and explicit labels separating flags from confirmed fraud.

## 11. Data Engineering and Quality Requirements
- Separate staging/raw records from curated analytical tables.
- Use stable keys, constraints, appropriate decimal types for money, and documented table grain.
- Make reprocessing idempotent so reruns do not duplicate canonical transactions or risk flags.
- Log pipeline runs and validation outcomes.
- Keep credentials in environment variables; never commit secrets.
- Test ingestion, validation, transformations, idempotency, rules, and KPI calculations.
- Document dataset provenance, schema, metrics, rule logic, and setup steps.

## 12. MVP Deliverables
- Dataset source, licence/provenance note, and data dictionary.
- Python ETL pipeline and data-quality report.
- PostgreSQL schema and SQL analytical queries.
- Three trained and evaluated ML classifiers.
- Explainable risk rules with tests.
- Power BI report.
- Automated tests and reproducible setup instructions.
- Architecture diagram, README, findings, and limitations.
- At least three evidence-backed analytical findings after analysis; do not invent findings in advance.

## 13. Important Limitations
- PaySim is simulated data, not proof of real-world fraud patterns.
- Rule flags and ML predictions are investigation signals, not proof of fraud.
- Results depend on feature quality, label reliability, class imbalance, leakage controls, and evaluation design.
- Confirm the professor's project rubric and permitted algorithms before finalising the academic submission.
- Do not fabricate model results or claim an algorithm is superior before testing.

## 14. Suggested Build Order
1. Confirm dataset and academic requirements.
2. Inspect the actual dataset schema and licence.
3. Set up the repository and PostgreSQL schema.
4. Implement and test ingestion/validation.
5. Build SQL metrics and rule-based risk flags.
6. Train and evaluate Logistic Regression, Decision Tree, and Random Forest.
7. Build Power BI pages and reconcile metrics.
8. Write findings, limitations, setup instructions, and a demo walkthrough.
