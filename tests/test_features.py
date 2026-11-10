"""
FinSight — Milestone 3 Feature Engineering Automated Tests.

Tests feature extraction contracts, log1p transformation safety,
temporal derivations, one-hot encoding determinism, leakage prevention,
and chronological dataset partitioning.
"""

import numpy as np
import pandas as pd
import pytest

from finsight.features import (
    CANONICAL_FEATURE_COLUMNS,
    ONE_HOT_TYPE_COLUMNS,
    PROHIBITED_FEATURE_COLUMNS,
    FeatureEngineeringError,
    chronological_split,
    compute_log1p_amount,
    derive_temporal_features,
    encode_transaction_types,
    extract_features,
    prepare_model_dataset,
    validate_no_leakage_columns,
)


@pytest.fixture
def sample_transaction_df() -> pd.DataFrame:
    """Fixture with valid transactions covering multiple rails and steps."""
    return pd.DataFrame({
        "step": [1, 12, 24, 25, 743],
        "type": ["PAYMENT", "CASH_OUT", "TRANSFER", "CASH_IN", "DEBIT"],
        "amount": [0.0, 1500.50, 250000.0, 50.0, 999.99],
        "nameOrig": ["C1001", "C1002", "C1003", "C1004", "C1005"],
        "oldbalanceOrg": [1000.0, 2000.0, 300000.0, 0.0, 5000.0],
        "newbalanceOrig": [1000.0, 499.50, 50000.0, 50.0, 4000.01],
        "nameDest": ["M2001", "C3002", "C3003", "C3004", "C3005"],
        "oldbalanceDest": [0.0, 500.0, 10000.0, 200.0, 100.0],
        "newbalanceDest": [0.0, 2000.50, 260000.0, 150.0, 1099.99],
        "isFraud": [0, 1, 1, 0, 0],
        "isFlaggedFraud": [0, 0, 1, 0, 0],
    })


def test_compute_log1p_amount_valid():
    """Verify log1p calculation for zero and positive amounts."""
    amounts = pd.Series([0.0, 9.0, 99.0, 1000.0])
    result = compute_log1p_amount(amounts)
    expected = np.log1p([0.0, 9.0, 99.0, 1000.0])

    np.testing.assert_allclose(result.values, expected, rtol=1e-5)
    assert result.iloc[0] == 0.0  # log1p(0) == 0


def test_compute_log1p_amount_negative_raises():
    """Verify negative amounts raise FeatureEngineeringError."""
    amounts = pd.Series([100.0, -5.0, 200.0])
    with pytest.raises(FeatureEngineeringError, match="negative transaction amounts"):
        compute_log1p_amount(amounts)


def test_compute_log1p_amount_null_raises():
    """Verify null or invalid amounts raise FeatureEngineeringError."""
    amounts = pd.Series([100.0, None, 200.0])
    with pytest.raises(FeatureEngineeringError, match="null or non-numeric"):
        compute_log1p_amount(amounts)


def test_derive_temporal_features_boundaries():
    """Verify step to transaction_hour and transaction_day mapping."""
    steps = pd.Series([1, 24, 25, 48, 744])
    hours, days = derive_temporal_features(steps)

    assert hours.tolist() == [0, 23, 0, 23, 23]
    assert days.tolist() == [1, 1, 2, 2, 31]


def test_derive_temporal_features_invalid_step():
    """Verify non-positive steps raise error."""
    steps = pd.Series([0, -1, 5])
    with pytest.raises(FeatureEngineeringError, match="Step values must be >= 1"):
        derive_temporal_features(steps)


def test_encode_transaction_types_all_five_present():
    """Verify one-hot encoder creates all 5 canonical columns even on single-rail subset."""
    types = pd.Series(["PAYMENT", "PAYMENT"])
    encoded = encode_transaction_types(types)

    assert list(encoded.columns) == list(ONE_HOT_TYPE_COLUMNS)
    assert len(encoded) == 2
    assert encoded["type_PAYMENT"].tolist() == [1, 1]
    assert encoded["type_CASH_OUT"].tolist() == [0, 0]
    assert encoded["type_TRANSFER"].tolist() == [0, 0]
    assert encoded["type_CASH_IN"].tolist() == [0, 0]
    assert encoded["type_DEBIT"].tolist() == [0, 0]


def test_encode_transaction_types_invalid_rail():
    """Verify unsupported transaction rails raise FeatureEngineeringError."""
    types = pd.Series(["WIRE_TRANSFER", "PAYMENT"])
    with pytest.raises(FeatureEngineeringError, match="unsupported transaction rails"):
        encode_transaction_types(types)


def test_extract_features_preserves_row_count_and_order(sample_transaction_df):
    """Verify extract_features retains identical row count, order, and indices."""
    features = extract_features(sample_transaction_df)

    assert len(features) == len(sample_transaction_df)
    assert (features.index == sample_transaction_df.index).all()
    assert list(features.columns) == list(CANONICAL_FEATURE_COLUMNS)

    # Merchant prefix check: first row nameDest is 'M2001', rest are 'C...'
    assert features["is_merchant_dest"].tolist() == [1, 0, 0, 0, 0]


def test_extract_features_strict_leakage_exclusion(sample_transaction_df):
    """Verify none of the prohibited columns exist in the extracted feature matrix."""
    features = extract_features(sample_transaction_df)

    for col in PROHIBITED_FEATURE_COLUMNS:
        assert col not in features.columns, f"Prohibited column {col} found in features!"


def test_validate_no_leakage_columns_fails_on_leakage():
    """Verify validator raises FeatureEngineeringError if prohibited columns exist."""
    leaked_cols = ["amount", "transaction_hour", "is_fraud", "name_orig"]
    with pytest.raises(FeatureEngineeringError, match="Prohibited leakage columns detected"):
        validate_no_leakage_columns(leaked_cols)


def test_prepare_model_dataset_separation(sample_transaction_df):
    """Verify X and y are properly partitioned and y is strictly binary."""
    X, y = prepare_model_dataset(sample_transaction_df, target_col="isFraud")

    assert len(X) == len(sample_transaction_df)
    assert len(y) == len(sample_transaction_df)
    assert "is_fraud" not in X.columns
    assert "isFraud" not in X.columns
    assert set(y.unique()).issubset({0, 1})
    assert y.tolist() == [0, 1, 1, 0, 0]


def test_chronological_split_logic(sample_transaction_df):
    """Verify chronological split partitions data based on step cutoff."""
    train_df, test_df = chronological_split(sample_transaction_df, split_step=24)

    assert len(train_df) == 3  # steps 1, 12, 24
    assert len(test_df) == 2   # steps 25, 743
    assert train_df["step"].max() <= 24
    assert test_df["step"].min() > 24


def test_extract_features_empty_dataframe():
    """Verify empty DataFrame returns empty feature DataFrame with correct schema."""
    empty_df = pd.DataFrame(columns=["step", "type", "amount", "nameDest"])
    features = extract_features(empty_df)

    assert len(features) == 0
    assert list(features.columns) == list(CANONICAL_FEATURE_COLUMNS)
