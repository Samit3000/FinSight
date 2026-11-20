"""FinSight — Tests for the locked evaluation protocol (P0-2) and reproducible training (P0-3)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finsight.evaluate import (
    FINAL_TEST_MIN_STEP,
    WALK_FORWARD_FOLDS,
    Fold,
    LockedTestAccessError,
    assert_development_steps,
    bootstrap_ap_ci,
    evaluate_scores,
    metrics_at_alert_budget,
    run_walk_forward,
    score_locked_final_test,
)


@pytest.fixture
def synthetic_steps_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    steps = np.repeat(np.arange(1, 744), 20)
    amount = rng.lognormal(10, 1, len(steps))
    is_fraud = (rng.random(len(steps)) < 0.02 + 0.2 * (amount > np.quantile(amount, 0.95))).astype(int)
    return pd.DataFrame({"step": steps, "amount": amount, "isFraud": is_fraud})


def test_folds_are_chronological_and_end_before_locked_test():
    for fold in WALK_FORWARD_FOLDS:
        assert fold.train_max_step < fold.test_min_step <= fold.test_max_step < FINAL_TEST_MIN_STEP
    assert FINAL_TEST_MIN_STEP == 651


def test_fold_reaching_locked_test_is_rejected():
    with pytest.raises(LockedTestAccessError):
        Fold("bad", train_max_step=550, test_min_step=551, test_max_step=700)
    with pytest.raises(ValueError):
        Fold("overlap", train_max_step=450, test_min_step=450, test_max_step=500)


def test_development_guard_rejects_locked_rows():
    assert_development_steps([1, 300, 650])
    with pytest.raises(LockedTestAccessError):
        assert_development_steps([1, 300, 651])


def test_evaluate_scores_refuses_locked_period():
    y = np.array([0, 1, 0, 1])
    with pytest.raises(LockedTestAccessError):
        evaluate_scores(y, np.array([0.1, 0.9, 0.2, 0.8]), np.array([640, 650, 651, 700]), n_boot=10)


def test_walk_forward_never_sees_locked_rows_and_respects_time(synthetic_steps_df):
    seen = []

    def fit_fn(train_df, y):
        seen.append(("train", int(train_df["step"].min()), int(train_df["step"].max())))
        return None

    def score_fn(model, test_df):
        seen.append(("test", int(test_df["step"].min()), int(test_df["step"].max())))
        return test_df["amount"].to_numpy()

    result = run_walk_forward(synthetic_steps_df, fit_fn, score_fn, n_boot=20)
    assert set(result["folds"]) == {"F1", "F2", "F3"}
    assert max(hi for _, _, hi in seen) < FINAL_TEST_MIN_STEP
    for i in range(0, len(seen), 2):
        assert seen[i][2] < seen[i + 1][1], "training window must end before its test window"


def test_alert_budget_is_exact_per_simulated_day():
    steps = np.repeat(np.arange(1, 49), 10)  # 48 distinct steps = 2 simulated days
    y = np.zeros(len(steps), dtype=int)
    y[:5] = 1
    score = np.linspace(1, 0, len(steps))
    m = metrics_at_alert_budget(y, score, steps, alerts_per_day=10)
    assert m["alerts"] == 20
    assert m["true_positives"] == 5 and m["recall"] == 1.0
    assert m["precision"] == 0.25


def test_alert_budget_ties_use_expected_value_and_ignore_row_order():
    steps = np.arange(1, 25)  # 24 steps = 1 simulated day
    score = np.array([1.0, 0.5, 0.5, 0.5, 0.5] + [0.0] * 19)
    y = np.array([1, 1, 1, 0, 0] + [0] * 19)
    m = metrics_at_alert_budget(y, score, steps, alerts_per_day=3)
    # 1 row above the cut-off (fraud) + 2 of 4 tied rows (2 fraud among them) -> E[TP] = 1 + 2 * 2/4
    assert m["alerts"] == 3 and m["rows_tied_at_cutoff"] == 4 and m["tied_rows_reviewed"] == 2
    assert m["true_positives"] == 2.0 and m["false_positives"] == 1.0 and m["false_negatives"] == 1.0
    perm = np.random.default_rng(0).permutation(len(y))
    m2 = metrics_at_alert_budget(y[perm], score[perm], steps[perm], alerts_per_day=3)
    assert {k: m2[k] for k in ("true_positives", "precision", "recall")} == {k: m[k] for k in ("true_positives", "precision", "recall")}


def test_bootstrap_ci_is_deterministic(synthetic_steps_df):
    d = synthetic_steps_df[synthetic_steps_df["step"] <= 200]
    args = (d["isFraud"].to_numpy(), d["amount"].to_numpy(), d["step"].to_numpy())
    a = bootstrap_ap_ci(*args, n_boot=30, seed=1)
    b = bootstrap_ap_ci(*args, n_boot=30, seed=1)
    assert a == b and a[0] <= a[1]


def test_locked_final_test_requires_explicit_acknowledgement(synthetic_steps_df):
    with pytest.raises(LockedTestAccessError):
        score_locked_final_test(synthetic_steps_df, lambda d: d["amount"].to_numpy())
    out = score_locked_final_test(
        synthetic_steps_df, lambda d: d["amount"].to_numpy(),
        i_understand_this_is_the_single_final_evaluation=True,
    )
    assert out["rows"] == int((synthetic_steps_df["step"] >= FINAL_TEST_MIN_STEP).sum())


# ------------------------------------------------------------------------------
# Training (P0-3)
# ------------------------------------------------------------------------------
from finsight.train import MODEL_CONFIGS, build_features, train_bundle  # noqa: E402
from finsight.features import extract_features  # noqa: E402
from finsight.ml_experiments import BASELINE_8_FEATURES  # noqa: E402


def test_train_bundle_refuses_to_overwrite(tmp_path: Path):
    out_dir = tmp_path / "candidates"
    out_dir.mkdir()
    (out_dir / "hgb_baseline8_balanced.joblib").write_bytes(b"existing")
    with pytest.raises(FileExistsError):
        train_bundle("hgb_baseline8_balanced", data_path=tmp_path / "unused.csv", out_dir=out_dir)
    assert (out_dir / "hgb_baseline8_balanced.joblib").read_bytes() == b"existing"


def test_champion_config_reproduces_legacy_float32_feature_path():
    df = pd.DataFrame({
        "step": [1, 25, 300],
        "type": ["TRANSFER", "CASH_OUT", "PAYMENT"],
        "amount": [181.0, 123456.789, 9839.64],
        "transaction_hour": [0, 0, 11],
        "isFraud": [1, 0, 0],
    })
    cfg = MODEL_CONFIGS["hgb_baseline8_balanced"]
    got = build_features(df, cfg)
    legacy_input = df[["step", "type", "amount"]].astype({"amount": "float32"})
    expected = extract_features(legacy_input, include_merchant_flag=False)[list(BASELINE_8_FEATURES)]
    pd.testing.assert_frame_equal(got, expected)
    assert list(got.columns) == list(BASELINE_8_FEATURES)


def test_all_training_configs_stay_in_development_period():
    for name, cfg in MODEL_CONFIGS.items():
        pol = cfg["threshold_policy"]
        assert cfg["train_max_step"] < pol["val_min_step"] <= pol["val_max_step"] < FINAL_TEST_MIN_STEP, name


# ------------------------------------------------------------------------------
# Locked-test fence on legacy paths (B2)
# ------------------------------------------------------------------------------
def _synthetic_csv(tmp_path: Path) -> tuple[Path, pd.DataFrame]:
    rng = np.random.default_rng(3)
    steps = np.repeat(np.arange(1, 744, 3), 4)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"], size=n)
    amount = rng.exponential(50000.0, n).round(2)
    is_fraud = (np.isin(types, ["TRANSFER", "CASH_OUT"]) & (amount > 80000)).astype(int)
    df = pd.DataFrame({
        "step": steps, "type": types, "amount": amount,
        "nameOrig": [f"C{i}" for i in range(n)], "nameDest": [f"C9{i}" for i in range(n)],
        "isFraud": is_fraud,
    })
    path = tmp_path / "tx.csv"
    df.to_csv(path, index=False)
    return path, df


def test_explainability_sampling_never_returns_locked_rows(tmp_path: Path):
    from finsight.explainability import ExplainabilityError, sample_holdout_for_explanation

    path, df = _synthetic_csv(tmp_path)
    sampled = sample_holdout_for_explanation(path, sample_size=10_000)
    assert sampled["step"].min() >= 551 and sampled["step"].max() < FINAL_TEST_MIN_STEP
    assert len(sampled) == int(((df.step >= 551) & (df.step < FINAL_TEST_MIN_STEP)).sum())
    with pytest.raises(ExplainabilityError):
        sample_holdout_for_explanation(path, max_step=FINAL_TEST_MIN_STEP)


def test_frozen_evaluation_and_legacy_training_exclude_locked_rows(tmp_path: Path):
    from finsight.models import (
        evaluate_frozen_model, run_milestone4_training_and_evaluation, save_model_artifact,
        train_hist_gradient_boosting,
    )
    from finsight.ml_experiments import extract_extended_features

    path, df = _synthetic_csv(tmp_path)
    results = run_milestone4_training_and_evaluation(path, tmp_path / "m", tmp_path / "r")
    assert results["split_boundaries"]["holdout"]["max_step"] < FINAL_TEST_MIN_STEP

    train = df[df.step <= 450]
    X = extract_extended_features(train, "baseline_8")
    model_path = tmp_path / "frozen.joblib"
    save_model_artifact(train_hist_gradient_boosting(X, train.isFraud, max_iter=5), list(X.columns), 0.5, "t", model_path)
    out = evaluate_frozen_model(model_path, path, tmp_path / "fr")
    expected = int(((df.step > 550) & (df.step < FINAL_TEST_MIN_STEP)).sum())
    assert out["holdout_evaluation"]["total_samples"] == expected


@pytest.mark.parametrize("name", [
    "audit_baseline_model", "run_model_benchmarks", "run_feature_and_imbalance_experiments",
    "run_temporal_robustness_analysis", "execute_full_ml_sprint",
])
def test_legacy_sprint_steps_refuse_without_explicit_opt_in(name):
    import finsight.ml_experiments as mx

    with pytest.raises(LockedTestAccessError):
        getattr(mx, name)(df=pd.DataFrame()) if name != "execute_full_ml_sprint" else mx.execute_full_ml_sprint()


def test_every_training_config_discloses_selection_history():
    for name, cfg in MODEL_CONFIGS.items():
        assert cfg.get("selection_history"), f"{name} must state which periods influenced its selection"


def test_walk_forward_report_renders_selection_disclosure():
    from finsight.evaluate import render_walk_forward_markdown

    results = {
        "fold_definitions": [{"name": "F1", "train_max_step": 350, "test_min_step": 351, "test_max_step": 450}],
        "folds": {}, "summary": {"average_precision_by_fold": {}, "average_precision_min": 0.0,
                                 "average_precision_mean": 0.0},
    }
    md = render_walk_forward_markdown("t", results, {"selection_history": "Chosen on steps 451-550."})
    assert "## Selection disclosure" in md and "Chosen on steps 451-550." in md
