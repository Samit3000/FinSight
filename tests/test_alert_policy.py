"""P1.4 — alert policy: window separation, thresholds, tie policy, selection rule, stage guards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import alert_policy as ap  # noqa: E402
from finsight.evaluate import FINAL_TEST_MIN_STEP, WALK_FORWARD_FOLDS, Fold, LockedTestAccessError  # noqa: E402
from finsight.train import MODEL_CONFIGS, add_history_features  # noqa: E402


def test_windows_are_ordered_and_before_locked_test():
    for fold in WALK_FORWARD_FOLDS:
        w = ap.policy_windows(fold)
        assert w["train"][1] < w["calibration"][0] and w["calibration"][1] == fold.train_max_step
        assert w["calibration"][1] < w["test"][0] and w["test"][1] < FINAL_TEST_MIN_STEP
        assert w["calibration"][1] - w["calibration"][0] + 1 == ap.CALIBRATION_STEPS
    with pytest.raises(LockedTestAccessError):
        Fold("X", train_max_step=600, test_min_step=601, test_max_step=700)
    with pytest.raises(ValueError):
        ap.policy_windows(Fold("Y", train_max_step=50, test_min_step=51, test_max_step=60))


def test_threshold_for_budget():
    s = np.array([0.9, 0.8, 0.8, 0.8, 0.5, 0.1])
    assert ap.threshold_for_budget(s, 1) == 0.9
    t = ap.threshold_for_budget(s, 2)  # tie: all 0.8 rows alerted
    assert t == 0.8 and (s >= t).sum() == 4 and (s > t).sum() < 2
    assert ap.threshold_for_budget(s, 0) == float("inf")
    assert ap.threshold_for_budget(s, 99) == 0.1


def test_calibration_uses_band_budgets():
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(1, 49), 30)
    hours = (steps - 1) % 24
    score = rng.random(len(steps))
    cal = ap.calibrate(score, steps, hours, budget=20)
    k = cal["calibration_alerts_target"]
    assert k == 40
    assert (score >= cal["B"]["threshold"]).sum() == k
    c = cal["C_0.5"]
    w = ap.apply_policy("C_0.5", cal, score, hours)
    assert w[ap.is_night(hours)].sum() == k - 20 and w[~ap.is_night(hours)].sum() == 20
    assert c["day_share"] == 0.5


def test_policy_metrics_daily_counts_and_segments():
    steps = np.repeat(np.arange(1, 73), 2)  # 3 full days
    hours = (steps - 1) % 24
    y = (np.arange(len(steps)) % 7 == 0).astype(int)
    w = (np.arange(len(steps)) % 3 == 0).astype(float)
    m = ap.policy_metrics(w, y, steps, hours, np.full(len(steps), 5e4))
    assert m["daily"]["full_days"] == 3 and sum(m["daily"]["per_day"].values()) == pytest.approx(w.sum())
    assert m["true_positives"] + m["false_negatives"] == y.sum()
    assert m["segment_recall"]["amount<100k"]["recall"] == m["recall"]


def _fold_result(recall_b, recall_c, r_day_b, r_day_c, alerts_b, alerts_c):
    def pol(r, rd, a):
        return {"recall": r, "alerts": a, "segment_recall": {"06-23": {"recall": rd}}}
    P = {"B": pol(recall_b, r_day_b, alerts_b)}
    for s in ap.DAY_SHARES:
        P[f"C_{s:g}"] = pol(recall_c, r_day_c, alerts_c)
    return {"budgets": {str(ap.PRIMARY_BUDGET): {"policies": P}}}


def test_selection_rule():
    good = {f: _fold_result(0.30, 0.32, 0.10, 0.15, 400, 420) for f in ("F1", "F2")}
    assert ap.apply_selection_rule(good)["chosen_policy"].startswith("C_")
    too_many_alerts = {f: _fold_result(0.30, 0.32, 0.10, 0.15, 400, 500) for f in ("F1", "F2")}
    assert ap.apply_selection_rule(too_many_alerts)["chosen_policy"] == "B"
    one_fold_fails = {"F1": _fold_result(0.30, 0.32, 0.10, 0.15, 400, 420),
                      "F2": _fold_result(0.30, 0.29, 0.10, 0.15, 400, 420)}
    assert ap.apply_selection_rule(one_fold_fails)["chosen_policy"] == "B"
    assert ap.SELECTION_FOLDS == ("F1", "F2")


def _synthetic() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    steps = np.repeat(np.arange(1, 651), 30)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n)
    amount = rng.exponential(5e4, n).round(2)
    df = pd.DataFrame({"step": steps, "type": types, "amount": amount,
                       "nameDest": rng.choice([f"C{i}" for i in range(2000)], n), "transaction_hour": (steps - 1) % 24})
    df["isFraud"] = (np.isin(types, ["TRANSFER", "CASH_OUT"]) & (amount > 1.2e5) & (rng.random(n) < 0.5)).astype(int)
    return add_history_features(df, MODEL_CONFIGS[ap.CONFIG_NAME])


def test_run_fold_end_to_end_and_tie_policy_matches_evaluate():
    r = ap.run_fold(_synthetic(), "F1", MODEL_CONFIGS[ap.CONFIG_NAME])
    assert r["windows"]["calibration"] == (251, 350) and r["windows"]["test"] == (351, 450)
    for b, blk in r["budgets"].items():
        assert blk["policies"]["A"]["recall"] == pytest.approx(blk["policies"]["A"]["check_vs_evaluate"], abs=1e-4)
        assert set(blk["policies"]) == {"A", "B"} | {f"C_{s:g}" for s in ap.DAY_SHARES}
    out = {"folds": {"F1": r, "F2": r}, "meta": {"generated_utc": "t", "data_sha256": "0" * 64}}
    out["selection"] = ap.apply_selection_rule(out["folds"])
    md = ap.render_markdown("selection", out)
    assert "## Disclosure" in md and "Chosen policy" in md


def test_confirmation_requires_recorded_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "confirmation", "--reports-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="selection first"):
        ap.main()
