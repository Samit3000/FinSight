"""P1.2 — leakage tests for the recipient prior incoming-count feature."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight.evaluate import FINAL_TEST_MIN_STEP  # noqa: E402
from finsight.features import (  # noqa: E402
    RECIPIENT_PRIOR_IN_COUNT,
    FeatureEngineeringError,
    compute_recipient_prior_in_count,
)
from finsight.ml_experiments import BASELINE_8_FEATURES  # noqa: E402
from finsight.train import (  # noqa: E402
    MODEL_CONFIGS,
    add_history_features,
    build_features,
    load_development_data,
    required_columns,
    train_bundle,
)

HISTORY_CONFIGS = [n for n, c in MODEL_CONFIGS.items() if c.get("history_features")]


def _brute_force(df: pd.DataFrame) -> np.ndarray:
    dest, step = df["nameDest"].to_numpy(), df["step"].to_numpy()
    return np.array([int(((dest == d) & (step < s)).sum()) for d, s in zip(dest, step)], dtype=np.int64)


def _random_frame(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "step": rng.integers(1, 40, n),
        "nameDest": rng.choice([f"C{i}" for i in range(60)], n),
        "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n),
        "amount": rng.exponential(1e5, n).round(2),
        "isFraud": rng.integers(0, 2, n),
    })


def test_matches_brute_force_reference():
    df = _random_frame()
    got = compute_recipient_prior_in_count(df)
    assert got.name == RECIPIENT_PRIOR_IN_COUNT and got.dtype == np.int64
    np.testing.assert_array_equal(got.to_numpy(), _brute_force(df))


def test_whole_current_step_is_excluded_and_unseen_recipient_is_zero():
    df = pd.DataFrame({
        "step": [1, 1, 1, 2, 2, 5, 5],
        "nameDest": ["A", "A", "B", "A", "C", "A", "B"],
    })
    got = compute_recipient_prior_in_count(df).tolist()
    # Same-step siblings never count each other; only earlier steps do.
    assert got == [0, 0, 0, 2, 0, 3, 1]


def test_future_rows_never_change_earlier_values():
    df = _random_frame(seed=1)
    full = compute_recipient_prior_in_count(df)
    for cut in (5, 17, 30):
        prefix = df[df["step"] <= cut]
        pd.testing.assert_series_equal(compute_recipient_prior_in_count(prefix), full[prefix.index])


def test_row_order_does_not_matter_and_index_is_preserved():
    df = _random_frame(seed=2)
    df.index = df.index * 7 + 3  # non-default index
    full = compute_recipient_prior_in_count(df)
    shuffled = df.sample(frac=1.0, random_state=4)
    got = compute_recipient_prior_in_count(shuffled)
    assert list(got.index) == list(shuffled.index)
    pd.testing.assert_series_equal(got.sort_index(), full.sort_index())


def test_labels_and_amounts_are_never_read():
    df = _random_frame(seed=3)
    base = compute_recipient_prior_in_count(df)
    flipped = df.assign(isFraud=1 - df["isFraud"], amount=df["amount"][::-1].to_numpy())
    pd.testing.assert_series_equal(compute_recipient_prior_in_count(flipped), base)
    pd.testing.assert_series_equal(compute_recipient_prior_in_count(df[["step", "nameDest"]]), base)


def test_snake_case_column_and_missing_columns():
    df = _random_frame(seed=5)
    snake = df.rename(columns={"nameDest": "name_dest"})
    pd.testing.assert_series_equal(compute_recipient_prior_in_count(snake), compute_recipient_prior_in_count(df))
    with pytest.raises(FeatureEngineeringError):
        compute_recipient_prior_in_count(df.drop(columns=["nameDest"]))
    with pytest.raises(FeatureEngineeringError):
        compute_recipient_prior_in_count(df.assign(nameDest=None))
    assert len(compute_recipient_prior_in_count(df.iloc[:0])) == 0


# ------------------------------------------------------------------------------
# Training integration
# ------------------------------------------------------------------------------
def _frame_for_build(seed: int = 6) -> pd.DataFrame:
    df = _random_frame(seed=seed)
    return df.assign(transaction_hour=(df["step"] - 1) % 24)


def test_build_features_refuses_window_without_precomputed_history():
    df = _frame_for_build()
    for name in HISTORY_CONFIGS:
        with pytest.raises(ValueError, match="add_history_features"):
            build_features(df, MODEL_CONFIGS[name])


def test_history_computed_on_full_frame_before_slicing():
    df = add_history_features(_frame_for_build(), MODEL_CONFIGS["hgb_baseline8_rcpt_balanced"])
    window = df[df["step"] > 20]
    X = build_features(window, MODEL_CONFIGS["hgb_baseline8_rcpt_balanced"])
    np.testing.assert_array_equal(X[RECIPIENT_PRIOR_IN_COUNT].to_numpy(), _brute_force(df)[(df["step"] > 20).to_numpy()])


def test_candidate_feature_lists():
    df = add_history_features(_frame_for_build(), MODEL_CONFIGS["hgb_baseline8_rcpt_balanced"])
    e0 = build_features(df, MODEL_CONFIGS["hgb_baseline8_balanced"])
    e1 = build_features(df, MODEL_CONFIGS["hgb_baseline8_rcpt_balanced"])
    e4 = build_features(df, MODEL_CONFIGS["hgb_baseline7_rcpt_cw50"])
    assert list(e0.columns) == list(BASELINE_8_FEATURES)
    assert list(e1.columns) == list(BASELINE_8_FEATURES) + [RECIPIENT_PRIOR_IN_COUNT]
    pd.testing.assert_frame_equal(e1[list(BASELINE_8_FEATURES)], e0)  # E1 = E0 + exactly one column
    assert list(e4.columns) == [c for c in BASELINE_8_FEATURES if c != "log1p_amount"] + [RECIPIENT_PRIOR_IN_COUNT]
    assert "nameDest" not in e1.columns and "isFraud" not in e1.columns


def test_champion_config_is_unchanged_by_history_support():
    cfg = MODEL_CONFIGS["hgb_baseline8_balanced"]
    assert not cfg.get("history_features") and not cfg.get("drop_features")
    assert required_columns(cfg) == ["step", "type", "amount", "transaction_hour", "isFraud"]
    df = _frame_for_build()
    assert add_history_features(df, cfg) is df


def _history_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(8)
    steps = np.repeat(np.arange(1, 744, 2), 6)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n)
    dest = rng.choice([f"C{i}" for i in range(150)], n)
    df = pd.DataFrame({
        "step": steps, "type": types, "amount": rng.exponential(5e4, n).round(2),
        "nameOrig": [f"O{i}" for i in range(n)], "nameDest": dest,
        "transaction_hour": (steps - 1) % 24,
    })
    df["isFraud"] = (np.isin(types, ["TRANSFER", "CASH_OUT"]) & (df["amount"] > 9e4)).astype(int)
    path = tmp_path / "tx.csv"
    df.to_csv(path, index=False)
    return path


def test_load_development_data_drops_locked_rows(tmp_path: Path):
    path = _history_csv(tmp_path)
    df = load_development_data(path, MODEL_CONFIGS["hgb_baseline8_rcpt_balanced"])
    assert df["step"].max() < FINAL_TEST_MIN_STEP
    assert RECIPIENT_PRIOR_IN_COUNT in df.columns
    np.testing.assert_array_equal(df[RECIPIENT_PRIOR_IN_COUNT].to_numpy(), _brute_force(df))


def test_train_bundle_records_history_feature(tmp_path: Path):
    path = _history_csv(tmp_path)
    artifact = train_bundle("hgb_baseline8_rcpt_balanced", data_path=path, out_dir=tmp_path / "c")
    bundle = joblib.load(artifact)
    assert bundle["feature_names"][-1] == RECIPIENT_PRIOR_IN_COUNT
    manifest = json.loads(artifact.with_suffix(".manifest.json").read_text(encoding="utf-8"))
    assert manifest["config"]["history_features"] == [RECIPIENT_PRIOR_IN_COUNT]
    assert manifest["validation_step_range"][1] < FINAL_TEST_MIN_STEP
