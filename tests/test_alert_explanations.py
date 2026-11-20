"""P2.3 — reason codes: separation of model and rule sections, top-3 SHAP, non-causal wording, no labels, guards."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import alert_explanations as ae  # noqa: E402
from finsight.champion import load_champion, operating_threshold, score_frame  # noqa: E402
from finsight.risk_rules import RiskEngine  # noqa: E402


def _frame():
    rng = np.random.default_rng(7)
    steps = np.repeat(np.arange(1, 49), 60)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n)
    amount = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.3),
                      rng.uniform(5e5, 5e6, n), rng.exponential(5e4, n)).round(2)
    return pd.DataFrame({"step": steps, "type": types, "amount": amount, "transaction_hour": (steps - 1) % 24,
                         "nameDest": rng.choice([f"C{i}" for i in range(400)], n)})


@pytest.fixture(scope="module")
def records():
    return ae.explain_alerts(_frame(), 25, 48)


def test_every_alert_is_a_champion_alert_at_the_operating_threshold(records):
    champ = load_champion()
    df = _frame()
    s = score_frame(champ, df)
    expected = int(((df.step >= 25) & (df.step <= 48)).to_numpy().__and__(s >= operating_threshold(champ)).sum())
    assert len(records) == expected and expected > 0
    assert all(r["model_explanation"]["score"] >= r["model_explanation"]["threshold"] for r in records)


def test_model_and_rule_sections_are_separate_and_complete(records):
    engine = RiskEngine()
    for r in records:
        m, rc = r["model_explanation"], r["rule_conditions"]
        assert set(m) >= {"score", "threshold", "top_contributors", "disclaimer", "summary"}
        assert "rule" not in " ".join(m).lower()
        assert m["disclaimer"] == ae.MODEL_DISCLAIMER and rc["note"] == ae.RULE_NOTE
        tx = {"amount": r["amount"], "type": r["type"], "transaction_hour": r["hour"], "step": r["step"]}
        expected = engine.evaluate_transaction(tx).triggered_rules or ["none"]
        assert rc["rule_ids"] == expected
        assert len(rc["rules_met"]) == (0 if expected == ["none"] else len(expected))


def test_top_three_contributors_by_absolute_value_and_additivity(records):
    for r in records:
        m = r["model_explanation"]
        c = [x["contribution_log_odds"] for x in m["top_contributors"]]
        assert len(c) == 3 and [abs(v) for v in c] == sorted((abs(v) for v in c), reverse=True)
        assert m["additivity_error"] < 1e-6
        for x in m["top_contributors"]:
            assert x["direction"] == ("raised the score" if x["contribution_log_odds"] > 0 else "lowered the score")


def test_top_contributors_helper():
    out = ae.top_contributors(np.array([0.1, -2.0, 0.5, 2.0]), [1.0, 5.0, 0.0, 1.0], ["a", "b", "c", "d"], 3)
    assert [x["feature"] for x in out] == ["b", "d", "c"]   # |-2.0| ties |2.0|: feature order breaks the tie


def test_text_is_never_causal(records):
    for r in records:
        ae.assert_non_causal(r["model_explanation"]["summary"])
    with pytest.raises(ValueError):
        ae.assert_non_causal("This transaction is fraud because the amount is high.")
    assert "not evidence" in ae.MODEL_DISCLAIMER and "not evidence" in ae.RULE_NOTE


def test_labels_are_never_read(records):
    df = _frame()
    with_labels = df.assign(isFraud=np.random.default_rng(1).integers(0, 2, len(df)))
    again = ae.explain_alerts(with_labels, 25, 48)
    assert [r["step"] for r in again] == [r["step"] for r in records]
    assert [r["model_explanation"]["score"] for r in again] == [r["model_explanation"]["score"] for r in records]


def test_guards():
    with pytest.raises(ValueError, match="sealed test"):
        ae.explain_alerts(_frame(), 640, 651)
    with pytest.raises(Exception, match="start at step 1"):
        ae.explain_alerts(_frame()[lambda d: d.step > 5], 25, 48)
