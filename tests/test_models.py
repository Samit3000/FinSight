"""
FinSight — Unit Tests for Supervised Machine Learning Baselines Module.

Milestone 4: Validates chronological partitioning integrity, zero-leakage boundaries,
preprocessing isolation, model training pipelines, metric calculations, threshold
optimization, and model artifact persistence using lightweight synthetic fixtures.
Avoids retraining on the full 6.36M dataset to ensure fast, reproducible test execution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline

from finsight.features import CANONICAL_FEATURE_COLUMNS, PROHIBITED_FEATURE_COLUMNS
from finsight.models import (
    DEFAULT_TRAIN_STEP,
    DEFAULT_VAL_STEP,
    ModelPipelineError,
    chronological_three_way_split,
    evaluate_frozen_model,
    evaluate_predictions,
    find_optimal_threshold,
    load_model_artifact,
    prepare_partitioned_datasets,
    run_milestone4_training_and_evaluation,
    save_model_artifact,
    train_dummy_baseline,
    train_hist_gradient_boosting,
    train_logistic_regression,
)


@pytest.fixture
def synthetic_transaction_dataframe() -> pd.DataFrame:
    """Construct a clean, controlled synthetic transaction dataset spanning steps 100 to 700."""
    np.random.seed(42)
    n_rows = 300
    steps = np.concatenate([
        np.linspace(1, 450, 200, dtype=int),    # Train partition: 200 rows
        np.linspace(451, 550, 50, dtype=int),   # Val partition: 50 rows
        np.linspace(551, 700, 50, dtype=int),   # Test partition: 50 rows
    ])

    types = np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], size=n_rows)
    amounts = np.random.exponential(scale=50000.0, size=n_rows).round(2)
    # Ensure some frauds in transfer and cash_out
    is_fraud = np.zeros(n_rows, dtype=int)
    for i in range(n_rows):
        if types[i] in ["TRANSFER", "CASH_OUT"] and amounts[i] > 80000.0:
            is_fraud[i] = 1 if np.random.rand() > 0.4 else 0

    dest_ids = [
        f"M{i:06d}" if types[i] == "PAYMENT" else f"C{i:06d}" for i in range(n_rows)
    ]

    return pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": [f"C{i:06d}" for i in range(n_rows)],
        "oldbalanceOrg": np.random.uniform(0, 100000, n_rows),
        "newbalanceOrig": np.random.uniform(0, 100000, n_rows),
        "nameDest": dest_ids,
        "oldbalanceDest": np.random.uniform(0, 100000, n_rows),
        "newbalanceDest": np.random.uniform(0, 100000, n_rows),
        "isFraud": is_fraud,
        "isFlaggedFraud": 0,
    })


def test_chronological_three_way_split_boundaries(synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify exact partition step boundaries and non-empty subsets."""
    train_df, val_df, test_df = chronological_three_way_split(
        synthetic_transaction_dataframe, train_step=450, val_step=550
    )

    assert len(train_df) == 200
    assert len(val_df) == 50
    assert len(test_df) == 50

    assert train_df["step"].max() <= 450
    assert val_df["step"].min() > 450
    assert val_df["step"].max() <= 550
    assert test_df["step"].min() > 550


def test_chronological_three_way_split_invalid_boundaries(synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify exception if train_step >= val_step."""
    with pytest.raises(ModelPipelineError, match="Invalid split boundaries"):
        chronological_three_way_split(synthetic_transaction_dataframe, train_step=550, val_step=450)


def test_prepare_partitioned_datasets_leakage_exclusion(synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify that no prohibited columns enter any of the partition feature matrices."""
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = prepare_partitioned_datasets(
        synthetic_transaction_dataframe,
        target_col="isFraud",
        train_step=450,
        val_step=550,
        include_merchant_dest=True,
    )

    # Check row counts
    assert len(X_train) == 200
    assert len(X_val) == 50
    assert len(X_test) == 50

    # Ensure prohibited columns are strictly excluded
    for prohibited in PROHIBITED_FEATURE_COLUMNS:
        assert prohibited not in X_train.columns
        assert prohibited not in X_val.columns
        assert prohibited not in X_test.columns

    # Verify canonical column presence
    assert set(X_train.columns) == set(CANONICAL_FEATURE_COLUMNS)
    assert metadata["train"].row_count == 200
    assert metadata["validation"].row_count == 50
    assert metadata["holdout"].row_count == 50


def test_prepare_partitioned_datasets_ablation_flag(synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify that is_merchant_dest is cleanly omitted when include_merchant_dest=False."""
    X_train, _, X_val, _, X_test, _, _ = prepare_partitioned_datasets(
        synthetic_transaction_dataframe,
        target_col="isFraud",
        train_step=450,
        val_step=550,
        include_merchant_dest=False,
    )
    assert "is_merchant_dest" not in X_train.columns
    assert "is_merchant_dest" not in X_val.columns
    assert "is_merchant_dest" not in X_test.columns
    assert len(X_train.columns) == len(CANONICAL_FEATURE_COLUMNS) - 1


def test_scaler_preprocessor_train_only_fit(synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify that the StandardScaler in Logistic Regression pipeline is fit strictly on train."""
    X_train, y_train, X_val, _, _, _, _ = prepare_partitioned_datasets(
        synthetic_transaction_dataframe, target_col="isFraud"
    )

    pipeline = train_logistic_regression(X_train, y_train, max_iter=50)
    assert isinstance(pipeline, Pipeline)
    scaler = pipeline.named_steps["scaler"]

    # Scaler mean must match X_train mean, NOT X_val mean
    np.testing.assert_allclose(scaler.mean_, X_train.mean().values, rtol=1e-5)


def test_dummy_baseline_training_and_eval(synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify Dummy Classifier fits and generates valid probability outputs."""
    X_train, y_train, X_val, y_val, _, _, _ = prepare_partitioned_datasets(
        synthetic_transaction_dataframe, target_col="isFraud"
    )

    dummy = train_dummy_baseline(X_train, y_train)
    assert isinstance(dummy, DummyClassifier)

    val_prob = dummy.predict_proba(X_val)[:, 1]
    assert len(val_prob) == len(X_val)

    metrics = evaluate_predictions(y_val, val_prob, threshold=0.5, model_name="Dummy")
    assert metrics["metrics"]["precision"] >= 0.0
    assert metrics["metrics"]["recall"] >= 0.0
    assert "confusion_matrix" in metrics


def test_hist_gradient_boosting_training_and_eval(synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify HistGradientBoostingClassifier trains and predicts continuous probabilities."""
    X_train, y_train, X_val, y_val, _, _, _ = prepare_partitioned_datasets(
        synthetic_transaction_dataframe, target_col="isFraud"
    )

    clf = train_hist_gradient_boosting(X_train, y_train, max_iter=20)
    assert isinstance(clf, HistGradientBoostingClassifier)

    probs = clf.predict_proba(X_val)[:, 1]
    assert len(probs) == len(X_val)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    metrics = evaluate_predictions(y_val, probs, threshold=0.5, model_name="HGB")
    assert "average_precision_pr_auc" in metrics["metrics"]
    assert 0.0 <= metrics["metrics"]["average_precision_pr_auc"] <= 1.0


def test_find_optimal_threshold_selects_valid_cutoff() -> None:
    """Verify that find_optimal_threshold identifies a reasonable cutoff."""
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    # Clear probabilistic separation
    y_prob = np.array([0.1, 0.2, 0.25, 0.3, 0.7, 0.8, 0.85, 0.9])

    best_t, best_metrics = find_optimal_threshold(y_true, y_prob, metric="f1", num_thresholds=20)
    assert 0.3 <= best_t <= 0.7
    assert best_metrics["metrics"]["f1_score"] == 1.0
    assert best_metrics["metrics"]["precision"] == 1.0
    assert best_metrics["metrics"]["recall"] == 1.0


def test_find_optimal_threshold_alert_constrained() -> None:
    """Verify alert-constrained objective respects maximum alert capacity."""
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.8, 0.85, 0.95])

    # Unconstrained F1 would choose t around 0.5 (alerts=5, TP=5, FP=0)
    # But if max_alerts is 3, threshold must increase to limit alerts <= 3 (at t > 0.7, alerts = 3)
    best_t, metrics = find_optimal_threshold(
        y_true, y_prob, objective="alert_constrained", max_alerts=3, num_thresholds=100
    )
    assert metrics["predicted_positive_count"] <= 3
    assert best_t > 0.7


def test_find_optimal_threshold_min_precision_constraint() -> None:
    """Verify recall_at_min_precision guarantees minimum precision floor."""
    y_true = np.array([0, 0, 1, 1, 1])
    y_prob = np.array([0.2, 0.6, 0.5, 0.8, 0.9])

    # If min_precision is 1.0, threshold must be > 0.6 to avoid FP at 0.6
    best_t, metrics = find_optimal_threshold(
        y_true, y_prob, objective="recall_at_min_precision", min_precision=1.0, num_thresholds=100
    )
    assert metrics["metrics"]["precision"] == 1.0
    assert best_t > 0.6


def test_find_optimal_threshold_deterministic_tie_breaking() -> None:
    """Verify deterministic tie-breaking favors higher precision, then lower threshold."""
    y_true = np.array([0, 0, 1, 1])
    # Scores giving identical F1 but different precisions
    y_prob = np.array([0.2, 0.2, 0.8, 0.8])

    best_t, metrics = find_optimal_threshold(y_true, y_prob, metric="f1", num_thresholds=50)
    assert 0.2 < best_t <= 0.8
    assert metrics["metrics"]["precision"] == 1.0


def test_find_optimal_threshold_fine_grid_resolution() -> None:
    """Verify high-resolution grid (1,000 steps) evaluates fine-grained candidates."""
    y_true = np.array([0] * 90 + [1] * 10)
    y_prob = np.linspace(0.01, 0.985, 100)

    best_t, metrics = find_optimal_threshold(y_true, y_prob, num_thresholds=1000)
    assert metrics["metrics"]["f1_score"] >= 0.0
    assert round(best_t, 4) == best_t


def test_find_optimal_threshold_empty_and_mismatch_errors() -> None:
    """Verify ModelPipelineError is raised on empty inputs or length mismatch."""
    with pytest.raises(ModelPipelineError, match="cannot be empty"):
        find_optimal_threshold(np.array([]), np.array([]))

    with pytest.raises(ModelPipelineError, match="Length mismatch"):
        find_optimal_threshold(np.array([1, 0]), np.array([0.5]))


def test_find_optimal_threshold_all_negative_labels() -> None:
    """Verify threshold search handles edge case with zero positive labels without dividing by zero."""
    y_true = np.zeros(20, dtype=int)
    y_prob = np.linspace(0.1, 0.9, 20)

    best_t, metrics = find_optimal_threshold(y_true, y_prob, num_thresholds=50)
    assert 0.0 <= best_t <= 1.0
    assert metrics["metrics"]["recall"] == 0.0
    assert metrics["metrics"]["f1_score"] == 0.0


def test_evaluate_predictions_metrics_accuracy() -> None:
    """Verify arithmetic correctness of precision, recall, F1, and confusion matrix."""
    y_true = np.array([1, 1, 0, 0])
    y_prob = np.array([0.9, 0.4, 0.2, 0.8])  # At t=0.5: TP=1, FP=1, FN=1, TN=1

    result = evaluate_predictions(y_true, y_prob, threshold=0.5)
    cm = result["confusion_matrix"]
    assert cm["true_positives"] == 1
    assert cm["false_positives"] == 1
    assert cm["false_negatives"] == 1
    assert cm["true_negatives"] == 1

    assert result["metrics"]["precision"] == 0.5
    assert result["metrics"]["recall"] == 0.5
    assert result["metrics"]["f1_score"] == 0.5
    assert result["predicted_positive_count"] == 2


def test_save_and_load_model_artifact_roundtrip(tmp_path: Path, synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify that model artifacts can be saved to disk and loaded back with full fidelity."""
    X_train, y_train, _, _, _, _, _ = prepare_partitioned_datasets(
        synthetic_transaction_dataframe, target_col="isFraud"
    )
    clf = train_dummy_baseline(X_train, y_train)

    dest_file = tmp_path / "test_dummy.joblib"
    saved_path = save_model_artifact(
        model=clf,
        feature_names=list(X_train.columns),
        selected_threshold=0.55,
        model_name="TestDummy",
        filepath=dest_file,
        metadata={"test_run": True},
    )

    assert saved_path.exists()
    bundle = load_model_artifact(saved_path)
    assert bundle["model_name"] == "TestDummy"
    assert bundle["selected_threshold"] == 0.55
    assert bundle["feature_names"] == list(X_train.columns)
    assert bundle["metadata"]["test_run"] is True

    # Test prediction through loaded model
    loaded_model = bundle["model"]
    preds = loaded_model.predict(X_train)
    assert len(preds) == len(X_train)


def test_end_to_end_smoke_pipeline_synthetic(tmp_path: Path, synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify that run_milestone4_training_and_evaluation executes end-to-end on synthetic data."""
    csv_path = tmp_path / "synthetic_transactions.csv"
    synthetic_transaction_dataframe.to_csv(csv_path, index=False)

    models_dir = tmp_path / "models"
    reports_dir = tmp_path / "reports"

    results = run_milestone4_training_and_evaluation(
        data_path=csv_path,
        models_dir=models_dir,
        reports_dir=reports_dir,
        train_step=450,
        val_step=550,
        random_state=42,
    )

    assert "validation_evaluations" in results
    assert "holdout_evaluations" in results
    assert "ablation_comparison" in results

    # Check created files
    assert (models_dir / "dummy_baseline.joblib").exists()
    assert (models_dir / "logistic_regression.joblib").exists()
    assert (models_dir / "hist_gradient_boosting_canonical.joblib").exists()
    assert (models_dir / "hist_gradient_boosting_ablated.joblib").exists()
    assert (reports_dir / "metrics.json").exists()
    assert (reports_dir / "model_comparison.md").exists()


def test_evaluate_frozen_model_synthetic(tmp_path: Path, synthetic_transaction_dataframe: pd.DataFrame) -> None:
    """Verify that evaluate_frozen_model executes on synthetic data and serialized artifact without errors."""
    csv_path = tmp_path / "synthetic_transactions.csv"
    synthetic_transaction_dataframe.to_csv(csv_path, index=False)

    # Prepare model
    X_train, y_train, _, _, _, _, _ = prepare_partitioned_datasets(
        synthetic_transaction_dataframe, target_col="isFraud", include_merchant_dest=False
    )
    clf = train_hist_gradient_boosting(X_train, y_train, max_iter=10)

    model_path = tmp_path / "frozen_test_model.joblib"
    save_model_artifact(
        model=clf,
        feature_names=list(X_train.columns),
        selected_threshold=0.8,
        model_name="TestFrozenModel",
        filepath=model_path,
    )

    reports_dir = tmp_path / "reports"
    results = evaluate_frozen_model(
        model_path=model_path,
        data_path=csv_path,
        reports_dir=reports_dir,
        train_step=450,
        val_step=550,
    )

    assert "validation_evaluation" in results
    assert "holdout_evaluation" in results
    assert results["locked_decision_threshold"] == 0.8
    assert results["feature_count"] == len(X_train.columns)

    # Check generated report files
    assert (reports_dir / "frozen_model_evaluation.json").exists()
    assert (reports_dir / "frozen_model_evaluation.md").exists()
