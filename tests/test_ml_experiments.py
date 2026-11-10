"""
Unit tests for FinSight ML Performance Improvement & Experimentation Engine.

Validates:
1. Feature engineering extensions (cyclical time, interactions, non-linear amount).
2. Zero-leakage boundaries across extended feature configurations.
3. Precision-at-recall and recall-at-alerts metric computations.
4. Model factory training across all supported algorithm families.
5. Baseline audit execution on controlled synthetic data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline

from finsight.features import FeatureEngineeringError
from finsight.ml_experiments import (
    BASELINE_8_FEATURES,
    audit_baseline_model,
    compute_precision_at_recall_levels,
    compute_recall_at_alert_limits,
    extract_extended_features,
    train_model_variant,
)


@pytest.fixture
def synthetic_experiment_dataframe() -> pd.DataFrame:
    """Construct lightweight synthetic transaction data for testing."""
    np.random.seed(42)
    n_rows = 200
    steps = np.random.randint(1, 744, size=n_rows)
    types = np.random.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], size=n_rows)
    amounts = np.random.exponential(scale=50000.0, size=n_rows).round(2)
    is_fraud = np.zeros(n_rows, dtype=int)
    for i in range(n_rows):
        if types[i] in ["TRANSFER", "CASH_OUT"] and amounts[i] > 60000.0:
            is_fraud[i] = 1 if np.random.rand() > 0.5 else 0

    return pd.DataFrame({
        "step": steps,
        "type": types,
        "amount": amounts,
        "nameOrig": [f"C{i:05d}" for i in range(n_rows)],
        "nameDest": [f"M{i:05d}" if types[i] == "PAYMENT" else f"C{i:05d}" for i in range(n_rows)],
        "isFraud": is_fraud,
    })


def test_extract_extended_features_baseline_8(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify baseline_8 feature configuration extracts exactly the 8 expected features."""
    X = extract_extended_features(synthetic_experiment_dataframe, feature_set="baseline_8")
    assert list(X.columns) == list(BASELINE_8_FEATURES)
    assert len(X) == len(synthetic_experiment_dataframe)
    assert not X.isnull().any().any()


def test_extract_extended_features_cyclical_time(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify cyclical_time feature configuration."""
    X = extract_extended_features(synthetic_experiment_dataframe, feature_set="cyclical_time")
    assert "hour_sin" in X.columns
    assert "hour_cos" in X.columns
    assert "is_night_hour" in X.columns
    assert X["hour_sin"].min() >= -1.0
    assert X["hour_sin"].max() <= 1.0
    assert set(X["is_night_hour"].unique()).issubset({0, 1})


def test_extract_extended_features_type_amount_interactions(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify type_amount_interactions feature configuration."""
    X = extract_extended_features(synthetic_experiment_dataframe, feature_set="type_amount_interactions")
    assert "is_high_risk_rail" in X.columns
    assert "log1p_x_transfer" in X.columns
    assert "log1p_x_cashout" in X.columns
    assert "amount_x_transfer" in X.columns
    assert "amount_x_cashout" in X.columns


def test_extract_extended_features_nonlinear_amount(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify nonlinear_amount feature configuration."""
    X = extract_extended_features(synthetic_experiment_dataframe, feature_set="nonlinear_amount")
    assert "sqrt_amount" in X.columns
    assert "log1p_sq" in X.columns
    assert "is_round_1000" in X.columns


def test_extract_extended_features_all_engineered(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify all_engineered feature configuration includes all extensions."""
    X = extract_extended_features(synthetic_experiment_dataframe, feature_set="all_engineered")
    expected_cols = [
        "amount", "log1p_amount", "transaction_hour",
        "type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER",
        "hour_sin", "hour_cos", "is_night_hour",
        "is_high_risk_rail", "log1p_x_transfer", "log1p_x_cashout", "amount_x_transfer", "amount_x_cashout",
        "sqrt_amount", "log1p_sq", "is_round_1000",
    ]
    for col in expected_cols:
        assert col in X.columns


def test_extract_extended_features_zero_leakage_assertion(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify that prohibited columns are strictly barred from the extended feature matrix."""
    for fset in ["baseline_8", "cyclical_time", "type_amount_interactions", "nonlinear_amount", "all_engineered"]:
        X = extract_extended_features(synthetic_experiment_dataframe, feature_set=fset)
        for prohibited in ["is_fraud", "isFraud", "isFlaggedFraud", "nameOrig", "oldbalanceOrg"]:
            assert prohibited not in X.columns


def test_extract_extended_features_empty_dataframe_raises() -> None:
    """Verify exception on empty DataFrame."""
    with pytest.raises(FeatureEngineeringError):
        extract_extended_features(pd.DataFrame())


def test_compute_precision_at_recall_levels() -> None:
    """Verify precision at recall calculation on controlled synthetic arrays."""
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9])
    res = compute_precision_at_recall_levels(y_true, y_prob, recall_levels=(0.25, 0.50, 0.75, 1.0))
    assert "p_at_recall_25" in res
    assert "p_at_recall_50" in res
    assert 0.0 <= res["p_at_recall_50"] <= 1.0


def test_compute_recall_at_alert_limits() -> None:
    """Verify recall at alert budget limits."""
    y_true = np.array([0, 0, 1, 0, 1, 1, 0, 1, 0, 0])
    y_prob = np.array([0.1, 0.2, 0.7, 0.3, 0.8, 0.6, 0.4, 0.9, 0.05, 0.15])
    res = compute_recall_at_alert_limits(y_true, y_prob, alert_limits=(2, 4))
    assert res["budget_2_alerts"]["actual_alerts"] == 2
    assert res["budget_2_alerts"]["true_positives"] == 2  # top 2 scores are 0.9 (idx 7, y=1) and 0.8 (idx 4, y=1)
    assert res["budget_2_alerts"]["precision"] == 1.0
    assert res["budget_2_alerts"]["recall"] == 0.5


def test_train_model_variant_factory(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify model factory correctly initializes and fits supported architectures."""
    X = extract_extended_features(synthetic_experiment_dataframe, "baseline_8")
    y = synthetic_experiment_dataframe["isFraud"]

    # Dummy
    m_dummy, t_dummy = train_model_variant("dummy", X, y)
    assert isinstance(m_dummy, DummyClassifier)
    assert t_dummy >= 0.0

    # Logistic Regression
    m_lr, t_lr = train_model_variant("logistic_regression", X, y, class_weight="balanced")
    assert isinstance(m_lr, Pipeline)
    assert t_lr >= 0.0

    # Random Forest
    m_rf, t_rf = train_model_variant("random_forest", X, y, n_estimators=5, max_depth=3, max_samples=1.0)
    assert isinstance(m_rf, RandomForestClassifier)
    assert t_rf >= 0.0

    # HistGradientBoosting
    m_hgb, t_hgb = train_model_variant("hist_gradient_boosting", X, y, max_iter=5)
    assert isinstance(m_hgb, HistGradientBoostingClassifier)
    assert t_hgb >= 0.0


def test_train_model_variant_unsupported_family_raises(synthetic_experiment_dataframe: pd.DataFrame) -> None:
    """Verify ValueError on unknown model family."""
    X = extract_extended_features(synthetic_experiment_dataframe, "baseline_8")
    y = synthetic_experiment_dataframe["isFraud"]
    with pytest.raises(ValueError, match="Unsupported model family"):
        train_model_variant("invalid_family", X, y)
