"""P1.3 — candidate comparison runner: tie policy, shortlist rule, stage guards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import compare_candidates as cc  # noqa: E402
from finsight.evaluate import FINAL_TEST_MIN_STEP, LockedTestAccessError, metrics_at_alert_budget  # noqa: E402
from finsight.train import MODEL_CONFIGS, add_history_features  # noqa: E402


def test_selection_weights_match_budget_metric_tie_policy():
    rng = np.random.default_rng(0)
    score = rng.choice([0.1, 0.5, 0.9, 0.95], 2400)  # heavy ties
    y = (rng.random(2400) < score * 0.2).astype(int)
    steps = np.repeat(np.arange(1, 49), 50)
    for budget in (5, 20, 100):
        m = metrics_at_alert_budget(y, score, steps, budget)
        w = cc.selection_weights(score, m["alerts"])
        assert w.sum() == pytest.approx(m["alerts"])
        assert (w * y).sum() == pytest.approx(m["true_positives"], abs=0.01)
        assert (w * (1 - y)).sum() == pytest.approx(m["false_positives"], abs=0.01)


def test_comparison_set_pairs_differ_by_one_change_and_share_preprocessing():
    for lab, name in cc.CANDIDATE_SET.items():
        cfg = MODEL_CONFIGS[name]
        assert cfg["amount_dtype"] == "float32" and cfg["params"] == MODEL_CONFIGS["hgb_baseline8_balanced"]["params"], lab

    def changes(a, b):
        ca, cb = MODEL_CONFIGS[cc.CANDIDATE_SET[a]], MODEL_CONFIGS[cc.CANDIDATE_SET[b]]
        keys = ("class_weight", "history_features", "drop_features")
        return [k for k in keys if (ca.get(k) or None) != (cb.get(k) or None)]

    for a, b, _ in cc.COMPARISONS:
        if "combined" not in _:
            assert len(changes(a, b)) == 1, (a, b, changes(a, b))
    assert cc.CANDIDATE_SET["E0"] == "hgb_baseline8_balanced"  # champion config, untouched


def _blocks(rec_c, rec_0):
    blk = lambda r: {"at_alert_budget": {"100_per_day": {"recall": r}, "200_per_day": {"recall": r}}}
    return {"E0": {f: blk(rec_0) for f in ("F1", "F2")}, **{c: {f: blk(rec_c) for f in ("F1", "F2")} for c in cc.CANDIDATES}}


def test_shortlist_rule_requires_both_folds():
    diffs = {f"{c}-E0": {"F1": {"ci95": [0.01, 0.05]}, "F2": {"ci95": [0.02, 0.06]}} for c in cc.CANDIDATES}
    diffs["E4-E0"]["F2"]["ci95"] = [-0.001, 0.04]          # CI touches 0 in one fold
    dec = cc.apply_shortlist_rule(_blocks(0.30, 0.28), diffs)
    assert dec["E1"]["shortlisted"] and dec["E3"]["shortlisted"] and not dec["E4"]["shortlisted"]
    dec = cc.apply_shortlist_rule(_blocks(0.27, 0.28), diffs)  # recall lower than E0
    assert not any(d["shortlisted"] for d in dec.values())


def test_shortlist_rule_never_looks_at_f3():
    diffs = {f"{c}-E0": {"F1": {"ci95": [0.01, 0.05]}, "F2": {"ci95": [0.02, 0.06]}} for c in cc.CANDIDATES}
    assert cc.SCREENING_FOLDS == ("F1", "F2") and "F3" not in cc.SCREENING_FOLDS
    cc.apply_shortlist_rule(_blocks(0.3, 0.2), diffs)  # works with no F3 entries at all


def _synthetic(n_per_step: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    steps = np.repeat(np.arange(1, 651), n_per_step)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n)
    dest = rng.choice([f"C{i}" for i in range(3000)], n)
    amount = rng.exponential(5e4, n).round(2)
    df = pd.DataFrame({"step": steps, "type": types, "amount": amount, "nameDest": dest,
                       "transaction_hour": (steps - 1) % 24})
    df["isFraud"] = (np.isin(types, ["TRANSFER", "CASH_OUT"]) & (amount > 1.2e5) & (rng.random(n) < 0.5)).astype(int)
    return add_history_features(df, MODEL_CONFIGS["hgb_baseline8_rcpt_balanced"])


def test_run_stage_end_to_end_on_synthetic_data():
    out = cc.run_stage(_synthetic(), list(cc.CANDIDATE_SET), ["F1"], n_boot=5, n_jobs=1)
    r = out["results"]["E1"]["F1"]
    assert r["steps"] == [351, 450]
    assert set(r["at_alert_budget"]) == {"50_per_day", "100_per_day", "200_per_day"}
    bd = r["breakdown_at_100_per_day"]
    assert sum(s["fraud"] for s in bd["by_hour_band"]) == r["labelled_fraud"]
    assert sum(s["fraud"] for s in bd["by_recipient_prior_count"]) == r["labelled_fraud"]
    tp = r["at_alert_budget"]["100_per_day"]["true_positives"]
    assert sum(s["true_positives"] for s in bd["by_hour_band"]) == pytest.approx(tp, abs=0.05)
    assert out["results"]["E4"]["F1"]["features"][-1] == "recipient_prior_in_count"
    assert "log1p_amount" not in out["results"]["E4"]["F1"]["features"]
    assert set(out["paired_ap_differences"]) == {f"{a}-{b}" for a, b, _ in cc.COMPARISONS}
    out["meta"] = {"generated_utc": "t", "data_sha256": "0" * 64, "n_boot": 5}
    out["shortlist_decision"] = {c: {"checks": [], "shortlisted": False} for c in cc.CANDIDATES}
    out["shortlist"] = []
    md = cc.render_markdown("screening", out)
    assert "## Disclosure" in md and "screening results" in md


def test_run_stage_refuses_locked_rows():
    df = _synthetic(5)
    locked = df[df.step == 650].assign(step=FINAL_TEST_MIN_STEP)
    with pytest.raises(LockedTestAccessError):
        cc.assert_development_steps(pd.concat([df, locked])["step"], "x")


def test_confirmation_requires_recorded_screening(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "confirmation", "--reports-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="screening first"):
        cc.main()
