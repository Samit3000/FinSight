"""
FinSight — Unit Tests for Explainable AI (XAI) Module.

Milestone 5: Validates TreeSHAP explainer loading, feature alignment, leakage isolation,
Shapley additivity / mathematical reconstruction, global importance ranking, and
local per-transaction explanation contracts using lightweight synthetic fixtures.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingClassifier

from finsight.explainability import (
    ExplainabilityError,
    ExplainabilityMetadata,
    build_explanation_dataset,
    compute_global_feature_importance,
    compute_shap_values,
    generate_local_explanation,
    init_tree_explainer,
    load_model_for_explainability,
)
from finsight.features import CANONICAL_FEATURE_COLUMNS, PROHIBITED_FEATURE_COLUMNS
from finsight.models import save_model_artifact


@pytest.fixture
def synthetic_tree_model_and_data(tmp_path: Path):
    """Create a small, deterministic fitted HistGradientBoostingClassifier and dataset."""
    np.random.seed(42)
    n_rows = 150
    feature_names = [
        "amount",
        "log1p_amount",
        "transaction_hour",
        "transaction_day",
        "type_CASH_IN",
        "type_CASH_OUT",
        "type_DEBIT",
        "type_PAYMENT",
        "type_TRANSFER",
    ]

    amounts = np.random.exponential(scale=20000.0, size=n_rows).round(2)
    log1p_amounts = np.log1p(amounts)
    hours = np.random.randint(0, 24, size=n_rows)
    days = np.random.randint(1, 31, size=n_rows)

    type_transfer = (np.random.rand(n_rows) > 0.7).astype(int)
    type_cash_out = (np.random.rand(n_rows) > 0.5).astype(int)
    type_cash_in = np.zeros(n_rows, dtype=int)
    type_debit = np.zeros(n_rows, dtype=int)
    type_payment = (1 - (type_transfer | type_cash_out)).clip(min=0)

    X = pd.DataFrame({
        "amount": amounts,
        "log1p_amount": log1p_amounts,
        "transaction_hour": hours,
        "transaction_day": days,
        "type_CASH_IN": type_cash_in,
        "type_CASH_OUT": type_cash_out,
        "type_DEBIT": type_debit,
        "type_PAYMENT": type_payment,
        "type_TRANSFER": type_transfer,
    })[feature_names]

    # Target correlated with amount and transfer/cash_out
    y = ((X["amount"] > 30000) & (X["type_TRANSFER"] == 1)).astype(int)

    model = HistGradientBoostingClassifier(max_iter=15, random_state=42)
    model.fit(X, y)

    artifact_path = tmp_path / "test_hgb_model.joblib"
    save_model_artifact(
        model=model,
        feature_names=feature_names,
        selected_threshold=0.85,
        model_name="TestHGB",
        filepath=artifact_path,
    )

    # Raw transaction records for pipeline testing
    raw_df = pd.DataFrame({
        "step": hours + 1,
        "type": np.where(type_transfer == 1, "TRANSFER", np.where(type_cash_out == 1, "CASH_OUT", "PAYMENT")),
        "amount": amounts,
        "nameOrig": [f"C{i:06d}" for i in range(n_rows)],
        "nameDest": [f"C{i+1000:06d}" for i in range(n_rows)],
        "oldbalanceOrg": np.random.uniform(0, 50000, n_rows),
        "newbalanceOrig": np.random.uniform(0, 50000, n_rows),
        "oldbalanceDest": np.random.uniform(0, 50000, n_rows),
        "newbalanceDest": np.random.uniform(0, 50000, n_rows),
        "isFraud": y,
        "isFlaggedFraud": 0,
    })

    return {
        "model": model,
        "feature_names": feature_names,
        "artifact_path": artifact_path,
        "X": X,
        "y": y,
        "raw_df": raw_df,
    }


def test_load_model_for_explainability_valid(synthetic_tree_model_and_data):
    """Verify loading valid serialized model artifact for explainability."""
    path = synthetic_tree_model_and_data["artifact_path"]
    model, features, threshold, bundle = load_model_for_explainability(path)

    assert isinstance(model, HistGradientBoostingClassifier)
    assert features == synthetic_tree_model_and_data["feature_names"]
    assert threshold == 0.85
    assert bundle["model_name"] == "TestHGB"


def test_load_model_for_explainability_missing_file():
    """Verify that attempting to load a non-existent artifact raises ExplainabilityError."""
    with pytest.raises(ExplainabilityError, match="Model artifact file not found"):
        load_model_for_explainability("models/non_existent_model.joblib")


def test_build_explanation_dataset_alignment_and_leakage_exclusion(synthetic_tree_model_and_data):
    """Verify feature matrix is correctly aligned and strictly free of prohibited columns."""
    raw_df = synthetic_tree_model_and_data["raw_df"]
    expected_features = synthetic_tree_model_and_data["feature_names"]

    X, ref_df = build_explanation_dataset(raw_df, expected_features)

    # Feature alignment
    assert list(X.columns) == expected_features
    assert len(X) == len(raw_df)

    # Strict leakage exclusion from X
    for col in PROHIBITED_FEATURE_COLUMNS:
        assert col not in X.columns

    # Reference dataframe contains traceability metadata
    assert "nameOrig" in ref_df.columns
    assert "nameDest" in ref_df.columns
    assert "isFraud" in ref_df.columns
    assert len(ref_df) == len(raw_df)


def test_build_explanation_dataset_empty_raises(synthetic_tree_model_and_data):
    """Verify that an empty DataFrame raises ExplainabilityError."""
    empty_df = pd.DataFrame()
    with pytest.raises(ExplainabilityError, match="cannot be None or empty"):
        build_explanation_dataset(empty_df, synthetic_tree_model_and_data["feature_names"])


def test_init_tree_explainer(synthetic_tree_model_and_data):
    """Verify TreeExplainer successfully initializes on tree models."""
    model = synthetic_tree_model_and_data["model"]
    explainer = init_tree_explainer(model)
    assert explainer is not None
    assert hasattr(explainer, "expected_value")


def test_compute_shap_values_dimensions_finiteness_and_additivity(synthetic_tree_model_and_data):
    """Verify SHAP attributions dimensions, finiteness, and exact mathematical additivity."""
    model = synthetic_tree_model_and_data["model"]
    X = synthetic_tree_model_and_data["X"]

    explainer = init_tree_explainer(model)
    shap_vals = compute_shap_values(explainer, X, model=model, tolerance=1e-4)

    # Shape contract
    assert shap_vals.shape == X.shape
    # Finiteness contract
    assert np.all(np.isfinite(shap_vals))

    # Additivity contract
    base_val = float(np.atleast_1d(explainer.expected_value)[0])
    raw_scores = model.decision_function(X)
    reconstructed = shap_vals.sum(axis=1) + base_val
    np.testing.assert_allclose(raw_scores, reconstructed, atol=1e-5)


def test_compute_shap_values_empty_raises(synthetic_tree_model_and_data):
    """Verify that passing empty X to compute_shap_values raises ExplainabilityError."""
    model = synthetic_tree_model_and_data["model"]
    explainer = init_tree_explainer(model)
    empty_X = pd.DataFrame(columns=synthetic_tree_model_and_data["feature_names"])

    with pytest.raises(ExplainabilityError, match="Cannot compute SHAP values for an empty dataset"):
        compute_shap_values(explainer, empty_X)


def test_compute_global_feature_importance(synthetic_tree_model_and_data):
    """Verify calculation and ranking of global feature importance."""
    model = synthetic_tree_model_and_data["model"]
    X = synthetic_tree_model_and_data["X"]
    features = synthetic_tree_model_and_data["feature_names"]

    explainer = init_tree_explainer(model)
    shap_vals = compute_shap_values(explainer, X)

    importance_df = compute_global_feature_importance(shap_vals, features)

    assert len(importance_df) == len(features)
    assert list(importance_df.columns) == ["rank", "feature", "mean_abs_shap", "importance_share_pct"]
    assert list(importance_df["rank"]) == list(range(1, len(features) + 1))
    assert (importance_df["mean_abs_shap"] >= 0).all()
    # Decreasing order
    assert importance_df["mean_abs_shap"].is_monotonic_decreasing
    # Sum of shares should be approx 100%
    assert 99.0 <= importance_df["importance_share_pct"].sum() <= 101.0


def test_generate_local_explanation_structure_and_narrative(synthetic_tree_model_and_data):
    """Verify local per-transaction explanation contracts and non-causal language."""
    model = synthetic_tree_model_and_data["model"]
    X = synthetic_tree_model_and_data["X"]
    features = synthetic_tree_model_and_data["feature_names"]
    raw_df = synthetic_tree_model_and_data["raw_df"]

    explainer = init_tree_explainer(model)
    shap_vals = compute_shap_values(explainer, X)
    base_val = float(np.atleast_1d(explainer.expected_value)[0])

    # Explain row 0
    tx_ref = raw_df.iloc[0]
    x_row = X.iloc[0]
    shap_row = shap_vals[0]

    explanation = generate_local_explanation(
        transaction_ref=tx_ref,
        feature_values=x_row,
        shap_row=shap_row,
        feature_names=features,
        base_value=base_val,
        threshold=0.85,
        top_k=2,
    )

    # Contract checks
    assert "transaction_ref" in explanation
    assert "observed_attributes" in explanation
    assert "model_assessment" in explanation
    assert "top_positive_drivers" in explanation
    assert "top_negative_drivers" in explanation
    assert "human_readable_explanation" in explanation

    # Observed transaction attributes contract
    obs = explanation["observed_attributes"]
    assert "transaction_type" in obs
    assert "amount_usd" in obs
    assert obs["amount_usd"] == round(float(tx_ref["amount"]), 2)
    assert "simulated_day" in obs
    assert "simulated_hour" in obs

    # Model assessment contract
    assess = explanation["model_assessment"]
    assert "raw_log_odds_score" in assess
    assert 0.0 <= assess["predicted_probability"] <= 1.0
    assert assess["score_type"] == "uncalibrated_model_score_sigmoid"
    assert assess["locked_decision_threshold"] == 0.85
    assert isinstance(assess["predicted_positive_alert"], bool)

    # Driver semantic descriptions contract
    all_drivers = explanation["top_positive_drivers"] + explanation["top_negative_drivers"]
    assert len(all_drivers) > 0
    for driver in all_drivers:
        assert "feature" in driver
        assert "feature_value" in driver
        assert "shap_attribution" in driver
        assert "semantic_description" in driver

    # Narrative checks: must use non-causal and uncalibrated score language
    narrative = explanation["human_readable_explanation"]
    assert "because" not in narrative.lower() or "score was" in narrative.lower()
    assert "fraud risk score" in narrative.lower()
    assert "encoding mechanics" in narrative.lower()


def test_explainability_artifacts_consistency():
    """Verify consistency across generated explainability report, CSV, and metadata."""
    report_dir = Path("reports/explainability")
    csv_path = report_dir / "global_shap_importance.csv"
    meta_path = report_dir / "explainability_metadata.json"
    report_path = report_dir / "explainability_report.md"

    if not (csv_path.exists() and meta_path.exists() and report_path.exists()):
        pytest.skip("Explainability artifacts not yet generated in reports/explainability/")

    import json
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    df_csv = pd.read_csv(csv_path)
    with open(report_path, "r", encoding="utf-8") as f:
        report_text = f.read()

    # 1. Feature counts and columns
    assert len(df_csv) == len(meta["feature_names"])
    assert list(df_csv["feature"]) == meta["feature_names"] or set(df_csv["feature"]) == set(meta["feature_names"])

    # 2. Top feature consistency
    top_feature_csv = df_csv.iloc[0]["feature"]
    assert top_feature_csv in meta["feature_names"]
    assert top_feature_csv in report_text

    # 3. Sample size and fraud count consistency
    sample_size = meta["sample_size"]
    fraud_count = meta["sample_fraud_count"]
    assert f"{sample_size:,}" in report_text or str(sample_size) in report_text
    assert f"{fraud_count:,}" in report_text or str(fraud_count) in report_text

    # 4. Locked decision threshold consistency
    threshold = meta["decision_threshold"]
    assert f"{threshold:.4f}" in report_text or f"{threshold:.2f}" in report_text

    # 5. Explainer base value consistency
    base_val = meta["base_value_log_odds"]
    assert f"{base_val:.4f}" in report_text

