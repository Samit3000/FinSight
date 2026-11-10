"""
FinSight — Feature Engineering Module.

Milestone 3: Reproducible, leak-free feature engineering for financial transaction
risk modeling. Transforms validated transaction records into a strictly bounded,
fully numeric feature matrix suitable for machine learning and statistical profiling.

Leakage & Governance Boundaries:
1. Target Isolation: `is_fraud` is strictly preserved as the ground-truth prediction
   target and NEVER used to calculate or influence candidate features.
2. Rule Flag Exclusion: `is_flagged_fraud` is excluded from predictive features
   because it is an existing static heuristic rule (>200,000 on TRANSFER) that would
   introduce severe artificial circularity.
3. ID Exclusion: High-cardinality raw identifiers (`name_orig`, `name_dest`) are
   excluded from direct model features due to 99.85% single-use sparsity.
4. Balance Exclusion: All four balance fields (`oldbalance_org`, `newbalance_orig`,
   `oldbalance_dest`, `newbalance_dest`) are excluded from initial model features
   due to synthetic simulator artifacts (33% zero origin balances, 100% zero merchant
   balances, and 98% complete liquidation leakage in fraudulent records).
5. Database Metadata Exclusion: `transaction_id` and `created_at` are surrogate database
   keys and administrative audit timestamps, never predictive signals.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("finsight.features")

# Canonical transaction types supported in PaySim
ALLOWED_TRANSACTION_TYPES: Tuple[str, ...] = (
    "CASH_IN",
    "CASH_OUT",
    "DEBIT",
    "PAYMENT",
    "TRANSFER",
)

# Ordered one-hot column names
ONE_HOT_TYPE_COLUMNS: Tuple[str, ...] = tuple(
    f"type_{t}" for t in sorted(ALLOWED_TRANSACTION_TYPES)
)

# Prohibited feature columns across both snake_case (database) and camelCase (PaySim CSV)
PROHIBITED_FEATURE_COLUMNS: frozenset[str] = frozenset([
    # Ground-truth evaluation target
    "is_fraud",
    "isFraud",
    # Pre-existing operational heuristic rule flag
    "is_flagged_fraud",
    "isFlaggedFraud",
    # High-cardinality identifiers
    "name_orig",
    "nameOrig",
    "name_dest",
    "nameDest",
    # Balance fields (synthetic simulator leakage and artifacts)
    "oldbalance_org",
    "oldbalanceOrg",
    "newbalance_orig",
    "newbalanceOrig",
    "oldbalance_dest",
    "oldbalanceDest",
    "newbalance_dest",
    "newbalanceDest",
    # Database surrogate keys and metadata
    "transaction_id",
    "created_at",
])

# Canonical expected feature column list
CANONICAL_FEATURE_COLUMNS: Tuple[str, ...] = (
    "amount",
    "log1p_amount",
    "transaction_hour",
    "transaction_day",
    "is_merchant_dest",
    "type_CASH_IN",
    "type_CASH_OUT",
    "type_DEBIT",
    "type_PAYMENT",
    "type_TRANSFER",
)


class FeatureEngineeringError(Exception):
    """Raised when feature derivation or leakage validation fails."""


def validate_no_leakage_columns(columns: Sequence[str] | pd.Index) -> None:
    """Validate that a feature matrix contains no prohibited target or leakage columns.

    Args:
        columns: Column names of the candidate feature matrix.

    Raises:
        FeatureEngineeringError: If any prohibited column is present.
    """
    col_set = set(columns)
    prohibited_found = sorted(col_set.intersection(PROHIBITED_FEATURE_COLUMNS))
    if prohibited_found:
        raise FeatureEngineeringError(
            f"Prohibited leakage columns detected in feature matrix: {prohibited_found}.\n"
            "By governance policy, targets, heuristic flags, balance fields, raw IDs, "
            "and database metadata are strictly barred from predictor features."
        )


def compute_log1p_amount(amount_series: pd.Series) -> pd.Series:
    """Compute natural log(1 + amount) for transaction amounts.

    Ensures mathematical safety: log1p maps 0.0 -> 0.0 and compresses
    heavy right-skewed financial monetary amounts into a more normal distribution.

    Args:
        amount_series: Numeric Series of transaction amounts.

    Returns:
        pd.Series: Computed log1p values.

    Raises:
        FeatureEngineeringError: If any amount is negative or non-numeric.
    """
    if len(amount_series) == 0:
        return pd.Series(dtype=float, index=amount_series.index)

    # Coerce to numeric
    numeric_amounts = pd.to_numeric(amount_series, errors="coerce")
    if numeric_amounts.isnull().any():
        null_count = int(numeric_amounts.isnull().sum())
        raise FeatureEngineeringError(
            f"Cannot compute log1p on amounts containing {null_count} null or non-numeric values."
        )

    if (numeric_amounts < 0.0).any():
        neg_count = int((numeric_amounts < 0.0).sum())
        min_neg = float(numeric_amounts.min())
        raise FeatureEngineeringError(
            f"Cannot compute log1p on negative transaction amounts ({neg_count} negative rows, min: {min_neg})."
        )

    return np.log1p(numeric_amounts)


def derive_temporal_features(step_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Derive transaction_hour (0-23) and transaction_day (1-31) from simulation step.

    Step represents 1-indexed simulation hour:
    - hour = (step - 1) % 24
    - day  = ((step - 1) // 24) + 1

    Args:
        step_series: Numeric Series of simulation steps (1 to 744).

    Returns:
        Tuple of (hour_series: pd.Series, day_series: pd.Series).

    Raises:
        FeatureEngineeringError: If steps are invalid or non-positive.
    """
    if len(step_series) == 0:
        return (
            pd.Series(dtype=int, index=step_series.index),
            pd.Series(dtype=int, index=step_series.index),
        )

    numeric_steps = pd.to_numeric(step_series, errors="coerce")
    if numeric_steps.isnull().any():
        raise FeatureEngineeringError("Step series contains null or non-numeric values.")

    if (numeric_steps < 1).any():
        raise FeatureEngineeringError("Step values must be >= 1 for 1-indexed simulation.")

    step_int = numeric_steps.astype(int)
    hour = (step_int - 1) % 24
    day = ((step_int - 1) // 24) + 1
    return hour, day


def encode_transaction_types(type_series: pd.Series) -> pd.DataFrame:
    """One-hot encode transaction type into deterministic fixed binary columns.

    Guarantees that all 5 PaySim rails (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER)
    are present as binary columns in sorted alphabetical order, even if a slice
    contains only a subset of the transaction rails.

    Args:
        type_series: Series containing transaction type strings.

    Returns:
        pd.DataFrame: Deterministic one-hot encoded matrix with columns `type_<RAIL>`.

    Raises:
        FeatureEngineeringError: If unsupported transaction types are present.
    """
    if len(type_series) == 0:
        return pd.DataFrame(
            0,
            index=type_series.index,
            columns=list(ONE_HOT_TYPE_COLUMNS),
            dtype=int,
        )

    str_types = type_series.astype(str).str.strip().str.upper()
    unsupported = set(str_types.unique()) - set(ALLOWED_TRANSACTION_TYPES)
    if unsupported:
        raise FeatureEngineeringError(
            f"Encountered unsupported transaction rails: {sorted(unsupported)}. "
            f"Allowed rails: {sorted(ALLOWED_TRANSACTION_TYPES)}"
        )

    # Initialize dataframe with zeros for all canonical columns
    encoded_df = pd.DataFrame(
        0,
        index=type_series.index,
        columns=list(ONE_HOT_TYPE_COLUMNS),
        dtype=int,
    )

    for rail in ALLOWED_TRANSACTION_TYPES:
        col_name = f"type_{rail}"
        encoded_df[col_name] = (str_types == rail).astype(int)

    return encoded_df


def extract_features(
    df: pd.DataFrame,
    include_merchant_flag: bool = True,
) -> pd.DataFrame:
    """Transform raw or processed transaction DataFrame into a model-ready feature matrix.

    Extracts conservative, defensible predictor features:
    - `amount`: Raw non-negative monetary transaction amount.
    - `log1p_amount`: Natural log(1 + amount) for heavy-tailed scale stabilization.
    - `transaction_hour`: Hour of day (0-23) derived from simulation step.
    - `transaction_day`: Calendar day of simulation (1-31).
    - `is_merchant_dest`: Binary indicator (1 if destination ID starts with 'M', else 0).
    - `type_<RAIL>`: One-hot encoded transaction rails (5 binary columns).

    Strict Governance Guarantee:
    - Never modifies input DataFrame.
    - Never silently drops rows or alters row ordering.
    - Validates that NO prohibited leakage columns enter the returned feature matrix.

    Args:
        df: Input DataFrame containing transaction records.
        include_merchant_flag: Whether to include `is_merchant_dest` feature.

    Returns:
        pd.DataFrame: Purely numeric feature matrix with identical index to `df`.

    Raises:
        FeatureEngineeringError: If required inputs are missing or invalid.
    """
    if df is None:
        raise FeatureEngineeringError("Input DataFrame cannot be None.")

    n_rows = len(df)
    if n_rows == 0:
        cols = list(CANONICAL_FEATURE_COLUMNS)
        if not include_merchant_flag:
            cols.remove("is_merchant_dest")
        return pd.DataFrame(columns=cols, dtype=float)

    # Resolve amount column (support both standard and DB naming)
    amount_col = "amount" if "amount" in df.columns else None
    if amount_col is None:
        raise FeatureEngineeringError("Input DataFrame missing required 'amount' column.")

    amount_series = pd.to_numeric(df[amount_col], errors="coerce")
    if amount_series.isnull().any():
        raise FeatureEngineeringError("Transaction 'amount' column contains null values.")
    if (amount_series < 0.0).any():
        raise FeatureEngineeringError("Transaction 'amount' contains negative values.")

    log1p_amount = compute_log1p_amount(amount_series)

    # Resolve or derive temporal features
    if "transaction_hour" in df.columns and "transaction_day" in df.columns:
        hour_series = pd.to_numeric(df["transaction_hour"], errors="coerce").astype(int)
        day_series = pd.to_numeric(df["transaction_day"], errors="coerce").astype(int)
    elif "step" in df.columns:
        hour_series, day_series = derive_temporal_features(df["step"])
    else:
        raise FeatureEngineeringError(
            "Input DataFrame must contain either 'step' or both 'transaction_hour' and 'transaction_day'."
        )

    # Resolve and encode transaction types
    if "type" not in df.columns:
        raise FeatureEngineeringError("Input DataFrame missing required 'type' column.")
    encoded_types = encode_transaction_types(df["type"])

    # Construct features DataFrame
    features = pd.DataFrame(index=df.index)
    features["amount"] = amount_series.astype(float)
    features["log1p_amount"] = log1p_amount.astype(float)
    features["transaction_hour"] = hour_series.astype(int)
    features["transaction_day"] = day_series.astype(int)

    if include_merchant_flag:
        # Determine destination entity prefix (M = Merchant, C = Customer)
        dest_col = None
        for candidate in ["name_dest", "nameDest"]:
            if candidate in df.columns:
                dest_col = candidate
                break

        if dest_col is not None:
            dest_str = df[dest_col].astype(str).str.strip().str.upper()
            features["is_merchant_dest"] = dest_str.str.startswith("M").astype(int)
        else:
            # If destination column is not present in input, default to 0 with warning
            logger.warning("Destination identifier column not found; setting 'is_merchant_dest' to 0.")
            features["is_merchant_dest"] = 0

    # Append one-hot type columns
    for col in ONE_HOT_TYPE_COLUMNS:
        features[col] = encoded_types[col]

    # Validate output against prohibited leakage columns
    validate_no_leakage_columns(features.columns)

    # Verify identical row length and index alignment
    if len(features) != n_rows:
        raise FeatureEngineeringError(
            f"Row count mismatch after feature extraction: expected {n_rows}, got {len(features)}"
        )

    return features


def prepare_model_dataset(
    df: pd.DataFrame,
    target_col: str = "is_fraud",
    include_merchant_flag: bool = True,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Extract model feature matrix X and target label vector y from a dataset.

    Guarantees strict separation: X never contains the target or leakage columns,
    and row indices remain perfectly aligned.

    Args:
        df: Input DataFrame containing transactions and target label.
        target_col: Name of target column (checks 'is_fraud' and 'isFraud').
        include_merchant_flag: Whether to include merchant indicator feature.

    Returns:
        Tuple of (X: pd.DataFrame, y: pd.Series).

    Raises:
        FeatureEngineeringError: If target column is missing or invalid.
    """
    resolved_target = None
    for cand in [target_col, "is_fraud", "isFraud"]:
        if cand in df.columns:
            resolved_target = cand
            break

    if resolved_target is None:
        raise FeatureEngineeringError(
            f"Target column '{target_col}' not found in DataFrame. Available columns: {list(df.columns)}"
        )

    y_raw = pd.to_numeric(df[resolved_target], errors="coerce")
    if y_raw.isnull().any():
        raise FeatureEngineeringError(f"Target column '{resolved_target}' contains null values.")

    unique_vals = set(y_raw.unique())
    if not unique_vals.issubset({0, 1}):
        raise FeatureEngineeringError(
            f"Target variable must be binary (0 or 1), found unique values: {sorted(unique_vals)}"
        )

    y = y_raw.astype(int)
    y.name = "is_fraud"

    # Extract features from df without target
    X = extract_features(df, include_merchant_flag=include_merchant_flag)

    return X, y


def chronological_split(
    df: pd.DataFrame,
    split_step: int = 550,
    step_col: str = "step",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partition a transaction dataset into train and test sets chronologically.

    Temporal splitting prevents lookahead bias and data leakage in time-series
    payment logs. Models must be trained on past transactions and tested on future
    transactions.

    Args:
        df: Input DataFrame containing transactions.
        split_step: Simulation step cutoff (inclusive for train, exclusive for test).
            Default 550 corresponds to ~day 23 out of 31 days.
        step_col: Name of step column (checks 'step').

    Returns:
        Tuple of (train_df: pd.DataFrame, test_df: pd.DataFrame).

    Raises:
        FeatureEngineeringError: If step column is missing.
    """
    if step_col not in df.columns:
        raise FeatureEngineeringError(f"Step column '{step_col}' not found in DataFrame.")

    steps = pd.to_numeric(df[step_col], errors="coerce")
    train_mask = steps <= split_step
    test_mask = steps > split_step

    train_df = df[train_mask].copy()
    test_df = df[test_mask].copy()

    logger.info(
        "Chronological split at step %d: %d train rows (%.1f%%), %d test rows (%.1f%%)",
        split_step,
        len(train_df),
        (len(train_df) / len(df) * 100) if len(df) > 0 else 0,
        len(test_df),
        (len(test_df) / len(df) * 100) if len(df) > 0 else 0,
    )

    return train_df, test_df
