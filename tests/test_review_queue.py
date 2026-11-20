"""P1.4.2 — priority review queue: capacity, deadlines, priority, no lookahead, row-order invariance."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import review_queue as rq  # noqa: E402
from finsight.evaluate import STEPS_PER_DAY  # noqa: E402
from finsight.train import MODEL_CONFIGS, add_history_features  # noqa: E402


def _stream(seed=0, days=3, per_step=40):
    rng = np.random.default_rng(seed)
    steps = np.repeat(np.arange(1, days * STEPS_PER_DAY + 1), per_step)
    score = np.round(rng.random(len(steps)), 2)  # coarse scores -> many ties
    return score, steps


def test_hourly_capacities_sum_to_budget():
    for k in (1, 7, 50, 100, 200, 1000):
        c = rq.hourly_capacities(k)
        assert len(c) == STEPS_PER_DAY and c.sum() == k and c.min() >= 0 and c.max() - c.min() <= 1


def test_mass_is_conserved_and_capacity_never_exceeded():
    score, steps = _stream()
    sim = rq.simulate_queue(score, steps, per_day=100, deadline=6, floor=0.5)
    queued = ~sim["below_floor"] & ~sim["held"]
    total = sim["reviewed"] + sim["expired"] + sim["pending"]
    np.testing.assert_allclose(total[queued], 1.0, atol=1e-9)
    assert np.all(total[~queued] == 0)
    caps = rq.hourly_capacities(100)
    for s, used in sim["reviews_per_step"].items():
        assert used <= caps[(s - 1) % 24] + 1e-9
    assert sum(sim["reviews_per_step"].values()) == pytest.approx(sim["reviewed"].sum())


def test_deadline_and_no_lookahead():
    score, steps = _stream(seed=1)
    for d in (1, 6, 24):
        sim = rq.simulate_queue(score, steps, per_day=50, deadline=d, floor=0.3)
        rv = sim["reviewed"] > 0
        per_item = sim["delay_hours"][rv] / sim["reviewed"][rv]
        assert per_item.min() >= 0          # never reviewed before it arrives
        assert per_item.max() <= d - 1 + 1e-9  # never after its deadline


def test_highest_scores_reviewed_first():
    steps = np.array([1, 1, 1, 1])
    score = np.array([0.9, 0.2, 0.8, 0.95])
    caps = rq.hourly_capacities(48)  # 2 reviews per hour
    assert caps[0] == 2
    sim = rq.simulate_queue(score, steps, per_day=48, deadline=1, floor=0.0)
    np.testing.assert_allclose(sim["reviewed"], [1, 0, 0, 1])
    np.testing.assert_allclose(sim["pending"], [0, 1, 1, 0])  # window ends before they can expire
    later = rq.simulate_queue(np.r_[score, 0.1], np.r_[steps, 2], per_day=48, deadline=1, floor=0.0)
    np.testing.assert_allclose(later["expired"][:4], [0, 1, 1, 0])


def test_older_item_wins_tie_and_identical_items_share_capacity():
    steps = np.array([1, 2, 2])
    score = np.array([0.5, 0.5, 0.5])
    # 1 review per hour (24/day); step 1 reviews the only item, step 2 splits between two identical items
    sim = rq.simulate_queue(score, steps, per_day=24, deadline=24, floor=0.0)
    np.testing.assert_allclose(sim["reviewed"], [1.0, 0.5, 0.5])
    np.testing.assert_allclose(sim["pending"], [0.0, 0.5, 0.5])


def test_row_order_does_not_matter():
    score, steps = _stream(seed=2)
    base = rq.simulate_queue(score, steps, per_day=100, deadline=12, floor=0.4, escalate_at=0.98)
    perm = np.random.default_rng(3).permutation(len(score))
    shuf = rq.simulate_queue(score[perm], steps[perm], per_day=100, deadline=12, floor=0.4, escalate_at=0.98)
    for k in ("reviewed", "delay_hours", "expired", "pending"):
        np.testing.assert_allclose(shuf[k], base[k][perm], atol=1e-9)
    assert (shuf["held"] == base["held"][perm]).all()


def test_escalated_items_skip_the_queue():
    score, steps = _stream(seed=4)
    sim = rq.simulate_queue(score, steps, per_day=100, deadline=6, floor=0.5, escalate_at=0.95)
    assert sim["held"].sum() == (score >= 0.95).sum()
    assert sim["reviewed"][sim["held"]].sum() == 0


def test_warmup_rows_are_not_reported_and_must_precede():
    score, steps = _stream(seed=5)
    w_score, w_steps = _stream(seed=6, days=1)
    sim = rq.simulate_queue(score, steps + 24, per_day=100, deadline=24, floor=0.5,
                            warmup_score=w_score, warmup_steps=w_steps)
    assert len(sim["reviewed"]) == len(score)
    with pytest.raises(ValueError):
        rq.simulate_queue(score, steps, 100, 24, 0.5, warmup_score=w_score, warmup_steps=w_steps + 10)


def test_reference_policies():
    score, steps = _stream(seed=7)
    w1 = rq.daily_top_k(score, steps, 50)
    assert w1.sum() == pytest.approx(150)
    w2 = rq.first_come_cap(score >= 0.5, steps, 50)
    day = (steps - 1) // 24
    assert all(w2[day == d].sum() <= 50 + 1e-9 for d in np.unique(day))


def _fold_stub(recalls, d2, cv):
    pol = {f"Q{d}": {"recall": r, "load_cv": cv} for d, r in zip(rq.DEADLINES, recalls)}
    pol["D2"] = {"recall": d2}
    pol["B"] = {"recall": 0.3}
    return {"budgets": {str(rq.PRIMARY_BUDGET): {"policies": pol}}}


def test_selection_rule_prefers_shortest_deadline_within_tolerance():
    folds = {"F1": _fold_stub([0.315, 0.32, 0.322], 0.30, 0.0), "F2": _fold_stub([0.295, 0.305, 0.31], 0.25, 0.0)}
    sel = rq.apply_selection_rule(folds)
    assert sel["chosen_queue"] == "Q12" and sel["reference_policy"] == "Q12"   # Q6 is 0.01+ below in F2
    folds["F2"] = _fold_stub([0.295, 0.305, 0.31], 0.32, 0.0)                    # queue below D2 in F2
    assert rq.apply_selection_rule(folds)["reference_policy"] == "B"


def test_run_fold_end_to_end():
    rng = np.random.default_rng(9)
    steps = np.repeat(np.arange(1, 651), 25)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n)
    amount = rng.exponential(5e4, n).round(2)
    df = pd.DataFrame({"step": steps, "type": types, "amount": amount,
                       "nameDest": rng.choice([f"C{i}" for i in range(2000)], n), "transaction_hour": (steps - 1) % 24})
    df["isFraud"] = (np.isin(types, ["TRANSFER", "CASH_OUT"]) & (amount > 1.2e5) & (rng.random(n) < 0.5)).astype(int)
    df = add_history_features(df, MODEL_CONFIGS[rq.CONFIG_NAME])
    r = rq.run_fold(df, "F1", MODEL_CONFIGS[rq.CONFIG_NAME])
    P = r["budgets"]["100"]["policies"]
    assert {"B", "D1", "D2", "Q6", "Q12", "Q24", "Q6+esc"} <= set(P)
    assert P["Q24"]["alerts_per_day"] <= 100 + 1e-6
    out = {"folds": {"F1": r, "F2": r}, "meta": {"generated_utc": "t", "data_sha256": "0" * 64}}
    out["selection"] = rq.apply_selection_rule(out["folds"])
    md = rq.render_markdown("selection", out)
    assert "## Disclosure" in md and "Reference policy after P1.4.2" in md


def test_confirmation_requires_recorded_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["x", "--stage", "confirmation", "--reports-dir", str(tmp_path)])
    with pytest.raises(SystemExit, match="selection first"):
        rq.main()
