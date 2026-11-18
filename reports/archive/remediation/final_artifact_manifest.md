# FinSight — Final Model & Artifact Manifest

**Document Status:** Final Frozen  
**Date:** 2026-10-19  
**Stage:** Phase A — Repository and Artifact Reconciliation  
**Target Environment:** Python 3.14.3 | scikit-learn 1.8.0 | pandas 3.0.1 | shap 0.54.1 | pytest 9.1.1  

---

## 1. Executive Summary

This manifest provides the definitive, audit-grade inventory of all serialized machine learning models, feature matrices, metric reports, explainability outputs, and test suites in the FinSight repository.

### Key Manifest Findings:
1. **Authoritative Frozen Champion Model:** [`models/hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib)
   - **Feature Count:** 8 features (strictly omitting `is_merchant_dest` and `transaction_day`).
   - **Operational Rationale:** Eliminates the overfitted calendar-day shortcut created by PaySim's post-day-17 95% volume collapse. Achieves higher Holdout Average Precision ($0.4479$ vs $0.3836$) and higher Holdout ROC-AUC ($0.9487$ vs $0.9185$) than the 9-feature model.
   - **Operating Threshold:** $t = 0.9660$ (derived from unconstrained validation F1 optimization on `451 <= step <= 550`).
2. **Historical Baseline Ablated Reference:** [`models/hist_gradient_boosting_ablated.joblib`](file:///models/hist_gradient_boosting_ablated.joblib)
   - **Feature Count:** 9 features (omits `is_merchant_dest`, but *retains* `transaction_day`).
   - **Critical Audit Distinction:** Despite the name `ablated`, this model was ablated only with respect to `is_merchant_dest`. It retains `transaction_day`, which explains its vulnerability to temporal shortcut learning.
   - **Operating Threshold:** $t = 0.9900$ (capacity-constrained operational triage threshold).
3. **Canonical Feature Matrix:** [`models/hist_gradient_boosting_canonical.joblib`](file:///models/hist_gradient_boosting_canonical.joblib) (10 features, retained for historical ablation comparison).
4. **Linear & Heuristic Benchmarks:** [`models/logistic_regression.joblib`](file:///models/logistic_regression.joblib), [`models/dummy_baseline.joblib`](file:///models/dummy_baseline.joblib), and the rule-based engine in [`src/finsight/risk_rules.py`](file:///src/finsight/risk_rules.py).

---

## 2. Serialized Model Inventory (`models/`)

| Model Filename | Architecture | Feature Count | Exact Feature Ordering | Selected Threshold | Operating Objective | Partition Boundaries | Holdout AP (PR-AUC) | Holdout ROC-AUC | Status in Platform |
| :--- | :--- | :---: | :--- | :---: | :--- | :--- | :---: | :---: | :--- |
| [`hist_gradient_boosting_no_day.joblib`](file:///models/hist_gradient_boosting_no_day.joblib) | `HistGradientBoostingClassifier` | 8 | `amount`, `log1p_amount`, `transaction_hour`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` | **0.9660** | Unconstrained Validation F1 | Train $\le 450$<br>Val $451\text{--}550$<br>Holdout $\ge 551$ | **0.4479** | **0.9487** | **Authoritative Frozen Champion** |
| [`hist_gradient_boosting_ablated.joblib`](file:///models/hist_gradient_boosting_ablated.joblib) | `HistGradientBoostingClassifier` | 9 | `amount`, `log1p_amount`, `transaction_hour`, `transaction_day`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` | **0.9900** | Alert-Constrained ($\le 1\text{k}$ alerts) | Train $\le 450$<br>Val $451\text{--}550$<br>Holdout $\ge 551$ | 0.3836 | 0.9185 | Historical Reference Baseline |
| [`hist_gradient_boosting_canonical.joblib`](file:///models/hist_gradient_boosting_canonical.joblib) | `HistGradientBoostingClassifier` | 10 | `amount`, `log1p_amount`, `transaction_hour`, `transaction_day`, `is_merchant_dest`, `type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER` | 0.9900 | Validation F1 / Alert-Constrained | Train $\le 450$<br>Val $451\text{--}550$<br>Holdout $\ge 551$ | 0.3836 | 0.9185 | Collinear Merchant Audit |
| [`logistic_regression.joblib`](file:///models/logistic_regression.joblib) | `Pipeline(StandardScaler, LogisticRegression)` | 10 | Canonical 10 features | 0.9009 | Validation F1 | Train $\le 450$<br>Val $451\text{--}550$<br>Holdout $\ge 551$ | 0.1805 | 0.9152 | Linear Baseline |
| [`dummy_baseline.joblib`](file:///models/dummy_baseline.joblib) | `DummyClassifier(strategy='most_frequent')` | 10 | Canonical 10 features | 0.5000 | Majority Class | Train $\le 450$<br>Val $451\text{--}550$<br>Holdout $\ge 551$ | 0.0109 | 0.5000 | Trivial Baseline |

---

## 3. Authoritative Champion Model Specification

### 3.1 Estimator Hyperparameters & Provenance
```python
HistGradientBoostingClassifier(
    loss="log_loss",
    learning_rate=0.1,
    max_iter=50,
    max_leaf_nodes=31,
    min_samples_leaf=20,
    l2_regularization=0.0,
    max_bins=255,
    categorical_features="from_dtype",
    monotonic_cst=None,
    interaction_cst=None,
    warm_start=False,
    early_stopping="auto",
    scoring="loss",
    validation_fraction=0.1,
    n_iter_no_change=10,
    tol=1e-07,
    verbose=0,
    random_state=42,
    class_weight="balanced"
)
```

### 3.2 Preprocessing & Feature Pipeline
- **Numerical Scaling:** Not required. Histogram-based gradient boosting uses quantile-based binning (`max_bins=255`), making tree split decisions invariant to monotonic numerical scaling.
- **Logarithmic Amount Transformation:** `log1p_amount = np.log1p(amount)`. Retained alongside raw amount; provides smoother gradient binning across low-value payments without altering extreme outliers.
- **Temporal Transformation:** `transaction_hour = (step - 1) % 24`. 0-indexed hour of day (0–23). Captures diurnal human sleep/wake cycles.
- **Categorical Encoding:** One-hot encoding across the 5 native PaySim operation types (`type_CASH_IN`, `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`) with `int8` representation.
- **Prohibited Exclusions:** Strictly excludes raw IDs (`nameOrig`, `nameDest`), settlement balances (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`), operational rules (`isFlaggedFraud`), and target (`isFraud`).

### 3.3 Operating Threshold Provenance
- **Validation Partition:** `451 <= step <= 550` ($N = 151,302$, $1,008$ frauds).
- **Candidate Search Grid:** 1,000 steps ($0.001\text{--}0.999$).
- **Objective:** Maximum F1 on validation predictions.
- **Selected Threshold:** **$t = 0.9660$**.
  - Validation Precision: $33.51\%$
  - Validation Recall: $43.15\%$
  - Validation F1: $0.3773$
  - Validation Alerts: $1,590$
- **Holdout Evaluation ($t = 0.9660$):**
  - Holdout Precision: $57.74\%$
  - Holdout Recall: $42.74\%$
  - Holdout F1: $0.4912$
  - True Positives: $918$ | False Positives: $672$ | False Negatives: $1,230$ | True Negatives: $193,781$
  - Total Alerts: $1,590$ (Alert rate: $0.8087\%$)

---

## 4. Explainability & Metric Reports Association

| Artifact Category | File Path | Subject / Target Model | Key Content |
| :--- | :--- | :--- | :--- |
| **Global SHAP Table** | [`reports/explainability/global_shap_importance.csv`](file:///reports/explainability/global_shap_importance.csv) | `hist_gradient_boosting_no_day.joblib` | Ranked mean absolute SHAP values across 8 features ($N=2,000$ holdout sample, seed 42). |
| **SHAP Summary Beeswarm** | [`reports/explainability/global_shap_summary.png`](file:///reports/explainability/global_shap_summary.png) | `hist_gradient_boosting_no_day.joblib` | High-res beeswarm plot illustrating feature directionality and magnitude impact. |
| **Local Audit Explanations** | [`reports/explainability/local_explanations.json`](file:///reports/explainability/local_explanations.json) | `hist_gradient_boosting_no_day.joblib` | Machine-readable waterfall breakdowns for TP, TN, FP, FN audit cases. |
| **SHAP Metadata** | [`reports/explainability/explainability_metadata.json`](file:///reports/explainability/explainability_metadata.json) | `hist_gradient_boosting_no_day.joblib` | Provenance metadata: sample size (2,000), fraud count (17), max reconstruction error ($1.07 \times 10^{-14}$). |
| **Explainability Report** | [`reports/explainability/explainability_report.md`](file:///reports/explainability/explainability_report.md) | `hist_gradient_boosting_no_day.joblib` | Full Markdown explainability report with non-causal analyst guidelines. |
| **Ablation Results JSON** | [`reports/remediation/ablation_experiment_results.json`](file:///reports/remediation/ablation_experiment_results.json) | All 6 ablation variants | Validation and holdout metrics across 6 controlled experiments. |
| **Baseline Metric Report** | [`reports/model_evaluation/metrics.json`](file:///reports/model_evaluation/metrics.json) | Milestone 4 baseline models | Historical metrics for Dummy, Logistic Regression, and Canonical/Ablated HGB. |

---

## 5. Automated Test Coverage Matrix

The authoritative model and related data pipelines are protected by 80 automated tests:

| Test Module | Test Functions Covering Component | Tested Invariants |
| :--- | :--- | :--- |
| [`tests/test_models.py`](file:///tests/test_models.py) | `test_hist_gradient_boosting_training_and_eval`<br>`test_find_optimal_threshold_*`<br>`test_save_and_load_model_artifact_roundtrip` | Estimator probability bounds $[0, 1]$, fine grid resolution (1,000 steps), multi-objective constraints, tie-breaking, bundle serialization. |
| [`tests/test_features.py`](file:///tests/test_features.py) | `test_extract_features_strict_leakage_exclusion`<br>`test_validate_no_leakage_columns_fails_on_leakage` | Absolute exclusion of prohibited columns, deterministic one-hot ordering, feature parity. |
| [`tests/test_explainability.py`](file:///tests/test_explainability.py) | `test_compute_shap_values_dimensions_finiteness_and_additivity`<br>`test_explainability_artifacts_consistency` | TreeSHAP exact efficiency / additivity ($\text{error} < 10^{-12}$), metadata consistency, report synchronization. |
| [`tests/test_risk_rules.py`](file:///tests/test_risk_rules.py) | `test_rule_score_determinism`<br>`test_evaluate_against_ground_truth` | Deterministic point scoring (0–100), tier boundaries, confusion matrix computation. |
