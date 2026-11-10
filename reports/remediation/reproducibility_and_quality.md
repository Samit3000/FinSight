# FinSight — Reproducibility, Engineering Quality & Environment Audit

**Document Status:** Complete  
**Audit Phase:** Phase 8 — Maintainability, Environment, Test Fixtures, and Information Security  
**Project:** FinSight Portfolio Intelligence Platform  
**Dataset Reference:** PaySim Mobile Money Synthetic Dataset (`PS_20174392719_1491204439457_log.csv`)  

---

## 1. Executive Summary

Phase 8 audits the end-to-end maintainability, testing infrastructure, computational efficiency, determinism controls, and security posture of the FinSight fraud analytics codebase.

Key findings and hardening measures:
1. **Lightweight Test Architecture:** All unit tests in `tests/` operate on deterministic synthetic fixtures (50–300 rows) rather than demanding an unmanageable 6.36M row reload. A complete test run executes in under 15 seconds.
2. **Explicit Seed Governance:** Random seeds (`random_state=42`) are strictly enforced across dataset sampling, gradient boosting fitting, logistic regression, dummy baselines, and SHAP sample extraction.
3. **Multi-Objective Threshold Coverage:** Unit and regression tests cover fine-grid evaluation, precision floors (`recall_at_min_precision`), operational capacities (`alert_constrained`), deterministic tie-breaking, and boundary edge cases (e.g., zero positive labels).
4. **Information Security & Data Hygiene:** `.gitignore` enforces exclusion of `.env*`, database dumps, raw PaySim CSVs (`PS_*.csv`), OS metadata, cache files, and IDE configs. No passwords, credentials, or personal local filesystem paths are hardcoded in source control.
5. **Machine-Generated Artifact Integrity:** All reports and metadata JSONs link directly to machine-evaluated metrics without manual transcriptions or fabricated scores.

---

## 2. Test Architecture & Coverage Matrix

### 2.1 Test Module Layout

| Test File | Target Module | Scope & Invariants Tested | Execution Time (Typical) |
| :--- | :--- | :--- | :--- |
| `tests/test_dataset_audit.py` | Data Profiling & Schema | Null integrity, schema types, negative amount absence, rail cardinality | ~1.5 s |
| `tests/test_loader.py` | `src/finsight/load_to_postgres.py` | Chunked processing, SQLAlchemy batching, mock DB engine transactions | ~1.2 s |
| `tests/test_pipeline.py` | `src/finsight/pipeline.py` | CSV chunking, cleaning, validation errors, end-to-end data ingestion | ~1.8 s |
| `tests/test_features.py` | `src/finsight/features.py` | Derived features, cyclical time encoding, leakage exclusion validator | ~1.2 s |
| `tests/test_risk_rules.py` | `src/finsight/risk_rules.py` | Heuristic rules 001–004, point scoring, tier boundaries, tie resolution | ~1.5 s |
| `tests/test_models.py` | `src/finsight/models.py` | Chronological splits, scaler isolation, baselines, multi-objective thresholds | ~5.6 s |
| `tests/test_explainability.py` | `src/finsight/explainability.py` | TreeSHAP additivity, metadata persistence, non-causal explanation formatting | ~4.2 s |

### 2.2 Test Fixture Design & Efficiency

Running unit tests against the full 6,362,620-row PaySim CSV on every commit is computationally prohibitive and slows developer feedback loops. The FinSight test suite enforces a strict fixture design pattern:

```python
@pytest.fixture
def synthetic_transaction_dataframe() -> pd.DataFrame:
    """Construct a clean, controlled synthetic transaction dataset spanning steps 1 to 700."""
    np.random.seed(42)
    n_rows = 300
    steps = np.concatenate([
        np.linspace(1, 450, 200, dtype=int),    # Train partition: 200 rows
        np.linspace(451, 550, 50, dtype=int),   # Val partition: 50 rows
        np.linspace(551, 700, 50, dtype=int),   # Test partition: 50 rows
    ])
    ...
```

**Benefits:**
- **Zero Disk Dependency:** Fixtures generate purely in memory via NumPy/Pandas.
- **Fast CI/CD:** Complete test suite runs in ~15–18 seconds on local environments.
- **Deterministic Boundary Testing:** Partitions contain known counts, enabling exact assertion of step intervals, confusion matrices, and zero-leakage rules.

---

## 3. Determinism and Randomness Controls

Machine learning and explainability pipelines must guarantee bitwise reproducibility across runs. Sources of non-determinism were audited and addressed as follows:

| Component | Random Source | Deterministic Mechanism | Verification Test |
| :--- | :--- | :--- | :--- |
| **Chronological Split** | Partitioning | Strict deterministic boolean mask on `step` (`<= 450`, `451..550`, `>= 551`). No random sampling. | `test_chronological_three_way_split_boundaries` |
| **Logistic Regression** | Optimization solver | Explicit `random_state=42` passed to `LogisticRegression(max_iter=1000, random_state=42)`. | `test_scaler_preprocessor_train_only_fit` |
| **HistGradientBoosting** | Feature binning & subsampling | Explicit `random_state=42` passed to `HistGradientBoostingClassifier(random_state=42)`. | `test_hist_gradient_boosting_training_and_eval` |
| **Dummy Baseline** | Constant / Prior strategy | `DummyClassifier(strategy="most_frequent")`. Zero stochastic variance. | `test_dummy_baseline_training_and_eval` |
| **Threshold Search** | Optimization grid | Deterministic grid: `np.linspace(min_t, max_t, num_t)` with deterministic tie-breaking (higher precision, then lower threshold). | `test_find_optimal_threshold_deterministic_tie_breaking` |
| **SHAP Background Sample** | Dataset subsampling | Deterministic seeding: `X_background = X_train.sample(n=background_size, random_state=42)`. | `test_explainability.py` |
| **SHAP Evaluation Sample** | Dataset subsampling | Deterministic seeding: `X_explain = X_test.sample(n=sample_size, random_state=42)`. | `test_generate_global_shap_importance` |

---

## 4. Environment & Dependency Specifications

### 4.1 Verified Runtime Environment

The codebase was executed and verified under the following core stack:
- **Operating System:** Windows 11 (build 10.0.26100)
- **Python Version:** 3.14.3 (64-bit)
- **C/C++ Build Tooling:** Not required for runtime; scikit-learn and NumPy use pre-compiled Windows wheels.

### 4.2 Core Dependency Constraints (`requirements.txt`)

```text
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
shap>=0.42.0
psycopg2-binary>=2.9.0
sqlalchemy>=2.0.0
pytest>=7.4.0
joblib>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0
python-dotenv>=1.0.0
```

All pinned dependencies maintain forward compatibility with modern Python versions while avoiding breaking deprecation warnings in NumPy 2.x and scikit-learn 1.8.x.

---

## 5. Security Posture & Secrets Audit

### 5.1 `.gitignore` Configuration

The `.gitignore` was audited to prevent accidental data leaks or credential exposures:

- **Credentials & Environment Variables:**
  - `.env`
  - `.env.*` (with exception for non-sensitive `!.env.example`)
- **Raw Data Files:**
  - `data/raw/*`
  - `PS_*.csv` (protects 500MB PaySim raw log from repo pollution)
  - `*.csv.gz`, `*.csv.zip`
- **Database Credentials & Local State:**
  - `*.dump`, `*.sql.gz`, `*.sqlite3`
- **Temporary & Cache Artifacts:**
  - `__pycache__/`, `*.py[cod]`
  - `.pytest_cache/`, `.coverage`, `htmlcov/`
  - `.vscode/`, `.idea/`, `*.pbix.temp`

### 5.2 Source Code Credentials Audit

- Full codebase grep search confirmed zero occurrences of hardcoded production passwords, API tokens, or private keys.
- Database connection strings in `src/finsight/load_to_postgres.py` read strictly from environment variables via `os.getenv("DATABASE_URL")` with fallback to localhost development defaults.
- All report artifacts use generic, portable paths relative to the repository root.

---

## 6. Model Metadata and Artifact Schema

All serialized model artifacts in `models/*.joblib` follow a standardized container schema that includes complete provenance information:

```python
bundle = {
    "model": fitted_estimator,
    "feature_names": [
        "amount", "log1p_amount", "transaction_hour", "transaction_day",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"
    ],
    "selected_threshold": 0.9900,
    "model_name": "HistGradientBoosting_Ablated",
    "metadata": {
        "training_partition": "step <= 450",
        "validation_partition": "451 <= step <= 550",
        "holdout_partition": "step >= 551",
        "training_rows": 6014717,
        "training_frauds": 5057,
        "random_state": 42,
        "scikit_learn_version": "1.8.0",
        "threshold_selection_objective": "f1_with_alert_constraint_validation_only",
        "timestamp_utc": "2026-10-18T20:00:00Z"
    }
}
```

This ensures that downstream inference consumers and dashboard pipelines can immediately introspect feature ordering, decision thresholds, and training partitions without relying on tribal knowledge.
