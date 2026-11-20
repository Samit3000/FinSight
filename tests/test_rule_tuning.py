"""P2 — rule backstop study: grid, rule masks, thresholds, alert-budget accounting, selection rule, guards."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import rule_tuning as rt  # noqa: E402
from finsight.features import compute_recipient_prior_in_count  # noqa: E402
from finsight.risk_rules import RiskEngine  # noqa: E402


def _frame(n_steps=96, per_step=40, seed=0):
    rng = np.random.default_rng(seed)
    steps = np.repeat(np.arange(1, n_steps + 1), per_step)
    n = len(steps)
    amounts = np.where(rng.random(n) < 0.2, rng.choice([50_000.0, 200_000.0, 500_000.0, 1_000_000.0, 2_000_000.0], n),
                       rng.exponential(3e5, n).round(2))
    df = pd.DataFrame({"step": steps, "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n),
                       "amount": amounts, "transaction_hour": (steps - 1) % 24,
                       "nameDest": rng.choice([f"C{i}" for i in range(300)], n)})
    df["recipient_prior_in_count"] = compute_recipient_prior_in_count(df)
    return df


def test_grid_has_72_configs_and_includes_the_baseline():
    g = rt.grid()
    assert len(g) == 72
    assert g[rt.BASELINE] == {"hv": 200_000, "new": None, "cut": 70}


def test_baseline_mask_equals_rule_engine_default_high_tier():
    df = _frame()
    high, n_rules = rt.rule_high(df, 200_000, None, 70)
    engine = RiskEngine().evaluate_dataframe(df)
    assert n_rules == 4
    np.testing.assert_array_equal(high, engine["risk_tier"].to_numpy() == "HIGH")


def test_amount_and_new_recipient_thresholds_are_inclusive():
    df = pd.DataFrame({"step": [1] * 5, "type": ["TRANSFER"] * 5, "transaction_hour": [12] * 5,
                       "amount": [499_999.99, 500_000.0, 500_000.0, 600_001.0, 2_000_000.0],
                       "recipient_prior_in_count": [0, 0, 3, 0, 3]})
    high, _ = rt.rule_high(df, 2_000_000, (500_000, 45), 70)
    # rail 25 + new 45 = 70 only when amount >= 500k AND first-time recipient (500k is also round: +15 = 85);
    # 2M meets the high-value threshold exactly: 35 + 25 + 15 (round) = 75
    assert high.tolist() == [False, True, False, True, True]
    high80, _ = rt.rule_high(df, 2_000_000, (500_000, 45), 80)
    assert high80.tolist() == [False, True, False, False, False]   # 85 passes 80; 70 and 75 do not


def test_new_recipient_rule_reads_only_prior_history():
    df = pd.DataFrame({"step": [1, 1, 2], "type": ["TRANSFER"] * 3, "transaction_hour": [0, 0, 1],
                       "amount": [3e6, 3e6, 3e6], "nameDest": ["A", "A", "A"]})
    df["recipient_prior_in_count"] = compute_recipient_prior_in_count(df)
    high, _ = rt.rule_high(df, 1e9, (2_000_000, 45), 70)
    # both step-1 payments are first-time (same-step siblings do not count); step 2 is not
    assert high.tolist() == [True, True, False]


def test_extra_alert_accounting_and_same_capacity_comparison():
    steps = np.repeat(np.arange(1, 49), 10)            # 2 whole days, 480 rows
    score = np.linspace(1, 0, len(steps))              # E4 ranks row order
    y = np.zeros(len(steps), int)
    y[[0, 5, 30, 300]] = 1
    rule = np.zeros(len(steps), bool)
    rule[[0, 300, 301]] = True                          # row 0 already in E4's top-K
    r = rt.extra_alert_accounting(y, steps, score, rule, base_per_day=10)   # K = 20
    assert r["extra_alerts_per_day_mean"] == 1.0        # rows 300, 301 over 2 days
    assert r["extra_fraud"] == 1.0                      # row 300
    assert r["e4_same_capacity_fraud"] == 0.0           # E4's next 2 rows (20, 21) are not fraud
    assert r["beats_e4_same_capacity"] is True
    assert r["extra_alerts_full_day_max"] == 2.0 and r["full_days"] == 2


def test_daily_cap_is_strict_per_day_not_average():
    r = {"extra_alerts_full_day_max": 31.0, "beats_e4_same_capacity": True, "extra_fraud_per_day": 2.0}
    assert rt.checks(r)["cap_every_full_day"] is False
    r["extra_alerts_full_day_max"] = 30.0
    assert all(rt.checks(r).values())


def _res(f1, f2, n_rules=4):
    mk = lambda x: {"checks": {"cap_every_full_day": x[0], "beats_e4_same_capacity": x[1], "min_extra_fraud": True},
                    "extra_fraud": x[2], "extra_alerts_per_day_mean": 10.0, "days": 4.0}
    return {"folds": {"F1": mk(f1), "F2": mk(f2)}, "n_rules": n_rules}


def test_selection_rule_needs_both_folds_and_picks_most_extra_fraud():
    results = {"a": _res((True, True, 5), (True, True, 5)), "b": _res((True, True, 9), (True, False, 9)),
               "c": _res((True, True, 7), (True, True, 7), n_rules=5)}
    qual, chosen = rt.choose(results)
    assert qual == ["a", "c"] and chosen == "c"
    assert rt.choose({"x": _res((False, True, 9), (True, True, 9))}) == ([], None)


def test_recorded_study_answers_the_question():
    sel = json.loads(Path("reports/evaluation/p2_rule_backstop_selection.json").read_text(encoding="utf-8"))
    conf = json.loads(Path("reports/evaluation/p2_rule_backstop_confirmation.json").read_text(encoding="utf-8"))
    assert sel["folds_used"] == ["F1", "F2"] and sel["max_step_loaded"] <= 650 and conf["max_step_loaded"] <= 650
    assert sel["chosen"] is None and sel["qualifying"] == [] and conf["confirmed"] is False
    assert len(sel["results"]) == 72
    base = conf["results"][rt.BASELINE]
    assert base["extra_fraud"] < base["e4_same_capacity_fraud"]


def test_stages_refuse_to_overwrite_and_need_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "confirmation", "--reports-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="selection first"):
        rt.main()
    (tmp_path / f"{rt.REPORT_STEM}_selection.json").write_text("{}")
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "selection", "--reports-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        rt.main()
