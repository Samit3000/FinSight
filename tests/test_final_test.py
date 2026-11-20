"""P1.5 — sealed final test: guards, decision rule, label separation (synthetic data only)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import final_test as ft  # noqa: E402
from finsight.evaluate import FINAL_TEST_MIN_STEP  # noqa: E402
from finsight.features import RECIPIENT_PRIOR_IN_COUNT  # noqa: E402
from finsight.train import MODEL_CONFIGS  # noqa: E402


# ---------------------------------------------------------------- decision rule
def test_decide_requires_every_check():
    good = ft.decide([0.01, 0.2], {100: 0.30, 200: 0.45}, {100: 0.25, 200: 0.40})
    assert good["promote_e4"] and good["decision"].startswith("PROMOTE")
    assert not ft.decide([0.0, 0.2], {100: 0.30, 200: 0.45}, {100: 0.25, 200: 0.40})["promote_e4"]   # CI touches 0
    assert not ft.decide([0.01, 0.2], {100: 0.24, 200: 0.45}, {100: 0.25, 200: 0.40})["promote_e4"]  # recall@100 lower
    assert not ft.decide([0.01, 0.2], {100: 0.30, 200: 0.39}, {100: 0.25, 200: 0.40})["promote_e4"]  # recall@200 lower


def test_decision_rule_is_part_of_the_frozen_settings():
    s = ft.frozen_settings()
    assert s["decision_rule"] == ft.DECISION_RULE
    assert set(s["config_sha256"]) == {ft.CANDIDATE, ft.BASELINE}
    assert s["train_max_step"] < s["calibration_steps"][0] <= s["calibration_steps"][1] < s["sealed_steps"][0]
    assert s["sealed_steps"][0] == FINAL_TEST_MIN_STEP


def test_config_change_changes_the_hash():
    cfg = copy.deepcopy(MODEL_CONFIGS[ft.CANDIDATE])
    h = ft.config_sha256(cfg)
    cfg["params"]["max_iter"] += 1
    assert ft.config_sha256(cfg) != h


# ---------------------------------------------------------------- run guards
def _ok_state():
    return {"head": "b", "clean": True, "prereg_tracked": True,
            "prereg_commit_is_ancestor": True, "source_unchanged_since_prereg": True}


def _manifest():
    return {"settings": ft.frozen_settings(), "data_sha256": "d", "champion_artifact_sha256": "c", "code_commit": "a"}


def test_preconditions_pass_when_everything_matches():
    ft.check_run_preconditions(_manifest(), _ok_state(), ft.frozen_settings(), "d", "c", result_exists=False)


@pytest.mark.parametrize("change, message", [
    (dict(result_exists=True), "already exists"),
    (dict(state={"clean": False}), "uncommitted"),
    (dict(state={"prereg_tracked": False}), "not committed"),
    (dict(state={"prereg_commit_is_ancestor": False}), "ancestor"),
    (dict(state={"source_unchanged_since_prereg": False}), "source code changed"),
    (dict(data_sha="other"), "data file"),
    (dict(champion_sha="other"), "champion artifact"),
    (dict(settings_change=True), "frozen settings"),
])
def test_preconditions_refuse(change, message):
    state = {**_ok_state(), **change.get("state", {})}
    current = ft.frozen_settings()
    if change.get("settings_change"):
        current = {**current, "train_max_step": 600}
    with pytest.raises(ft.FinalTestGuardError, match=message):
        ft.check_run_preconditions(_manifest(), state, current, change.get("data_sha", "d"),
                                   change.get("champion_sha", "c"), change.get("result_exists", False))


def test_reproduction_check():
    ft.check_reproduction({"E4": 0.65301, "E0": 0.42219}, {"E4": 0.653, "E0": 0.4222})
    with pytest.raises(ft.FinalTestGuardError, match="do not reproduce"):
        ft.check_reproduction({"E4": 0.6529, "E0": 0.4222}, {"E4": 0.653, "E0": 0.4222})


def test_expected_reproduction_reads_p13_f3_for_the_same_configs():
    exp = ft.expected_reproduction()
    assert set(exp) == {"E4", "E0"} and 0 < exp["E0"] < exp["E4"] < 1


def test_run_refuses_without_preregistration(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "PREREG_FILE", tmp_path / "none.json")
    with pytest.raises(ft.FinalTestGuardError, match="Pre-register first"):
        ft.run(tmp_path / "unused.csv")


def test_preregister_refuses_dirty_tree_and_second_attempt(tmp_path, monkeypatch):
    monkeypatch.setattr(ft, "PREREG_FILE", tmp_path / "prereg.json")
    monkeypatch.setattr(ft, "repo_state", lambda *a: {**_ok_state(), "clean": False})
    with pytest.raises(ft.FinalTestGuardError, match="Commit the code"):
        ft.preregister(tmp_path / "unused.csv")
    (tmp_path / "prereg.json").write_text("{}")
    with pytest.raises(ft.FinalTestGuardError, match="already exists"):
        ft.preregister(tmp_path / "unused.csv")


# ---------------------------------------------------------------- data and scoring
def _synthetic_csv(tmp_path: Path) -> Path:
    rng = np.random.default_rng(11)
    steps = np.repeat(np.arange(1, 744), 30)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n)
    amount = rng.exponential(5e4, n).round(2)
    df = pd.DataFrame({"step": steps, "type": types, "amount": amount, "nameOrig": [f"O{i}" for i in range(n)],
                       "nameDest": rng.choice([f"C{i}" for i in range(2500)], n), "transaction_hour": (steps - 1) % 24})
    df["isFraud"] = (np.isin(types, ["TRANSFER", "CASH_OUT"]) & (amount > 1.2e5) & (rng.random(n) < 0.5)).astype(int)
    path = tmp_path / "tx.csv"
    df.to_csv(path, index=False)
    return path


def test_load_frames_separates_sealed_labels_and_uses_full_history(tmp_path):
    path = _synthetic_csv(tmp_path)
    dev, sealed, labels = ft.load_frames(path)
    assert dev["step"].max() < FINAL_TEST_MIN_STEP and sealed["step"].min() >= FINAL_TEST_MIN_STEP
    assert "isFraud" not in sealed.columns and len(labels) == len(sealed)
    raw = pd.read_csv(path)
    first = sealed.iloc[0]
    expected = int(((raw.nameDest == first["nameDest"]) & (raw.step < first["step"])).sum())
    assert int(first[RECIPIENT_PRIOR_IN_COUNT]) == expected      # history reaches back into development steps


def test_train_and_score_refuses_labelled_sealed_rows(tmp_path):
    dev, sealed, labels = ft.load_frames(_synthetic_csv(tmp_path))
    with pytest.raises(ft.FinalTestGuardError):
        ft.train_and_score(dev, sealed.assign(isFraud=labels.to_numpy()), MODEL_CONFIGS[ft.CANDIDATE])


def test_sealed_evaluation_end_to_end(tmp_path):
    from finsight.risk_rules import RiskEngine

    dev, sealed, labels = ft.load_frames(_synthetic_csv(tmp_path))
    runs = {"E4": ft.train_and_score(dev, sealed, MODEL_CONFIGS[ft.CANDIDATE]),
            "E0": ft.train_and_score(dev, sealed, MODEL_CONFIGS[ft.BASELINE])}
    assert "log1p_amount" not in runs["E4"]["features"] and runs["E4"]["features"][-1] == RECIPIENT_PRIOR_IN_COUNT
    assert runs["E0"]["train_rows"] == int((dev.step <= ft.TRAIN_MAX_STEP).sum())
    assert 0 <= runs["E4"]["calibration_ap"] <= 1
    rule_high = RiskEngine().evaluate_dataframe(sealed)["risk_score"].to_numpy() >= ft.RULE_HIGH_SCORE
    systems = {"E4": runs["E4"]["scores"], "E0": runs["E0"]["scores"], "champion_artifact": runs["E0"]["scores"]}
    out = ft._evaluate_sealed(sealed, labels, systems, runs, 0.9, rule_high)
    assert out["systems"]["E4"]["rows"] == len(sealed)
    assert out["systems"]["E4"]["labelled_fraud"] == int(labels.sum())
    assert "at_threshold" in out["systems"]["champion_artifact"]
    assert {"E4 B @100", "E4 Q12 @100", "E0 B @200"} <= set(out["policies"])
    assert out["decision"]["decision"] in ("KEEP CHAMPION", "PROMOTE E4 (pending separate promotion approval)")
    out["meta"] = {"generated_utc": "t", "head": "0" * 40, "prereg_commit": "0" * 40, "data_sha256": "0" * 64,
                   "champion_threshold": 0.9}
    md = ft.render_markdown(out)
    assert "## Decision" in md and "## Disclosure" in md
