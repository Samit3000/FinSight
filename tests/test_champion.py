"""P1.6 — champion registry, loader, rebuild recipe and benchmark rows."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import champion as ch  # noqa: E402
from finsight.features import RECIPIENT_PRIOR_IN_COUNT, compute_recipient_prior_in_count  # noqa: E402
from finsight.train import MODEL_CONFIGS, build_features  # noqa: E402

P15 = Path("reports/final_test/p15_final_test.json")


def test_registry_artifacts_match_their_hashes():
    active = ch.load_champion()
    previous = ch.load_champion(which="previous")
    assert active["entry"]["artifact"] == "models/hgb_e4_recipient.joblib"
    assert previous["entry"]["artifact"] == "models/hist_gradient_boosting_no_day.joblib"
    assert active["feature_names"][-1] == RECIPIENT_PRIOR_IN_COUNT and "log1p_amount" not in active["feature_names"]
    assert len(previous["feature_names"]) == 8


def test_active_champion_is_the_model_the_sealed_test_evaluated():
    p15 = json.loads(P15.read_text(encoding="utf-8"))
    reg = ch.read_registry()
    assert reg["active"]["artifact_sha256"] == p15["meta"]["e4_bundle"]["sha256"]
    assert p15["decision"]["promote_e4"] is True
    assert reg["active"]["evidence"]["sealed_test_average_precision"] == p15["systems"]["E4"]["average_precision"]
    thr = p15["meta"]["training"]["E4"]["thresholds"]
    assert reg["active"]["thresholds_by_alerts_per_day"] == {k: v for k, v in thr.items()}


def test_operating_threshold_matches_bundle():
    c = ch.load_champion()
    assert ch.operating_threshold(c) == pytest.approx(float(c["bundle"]["selected_threshold"]))
    assert ch.operating_threshold(c, 200) < ch.operating_threshold(c) < ch.operating_threshold(c, 50)


def test_tampered_artifact_is_refused(tmp_path):
    reg = ch.read_registry()
    (tmp_path / "models").mkdir()
    fake = tmp_path / "models" / "hgb_e4_recipient.joblib"
    fake.write_bytes(b"not the model")
    (tmp_path / "models" / "champion.json").write_text(json.dumps(reg), encoding="utf-8")
    with pytest.raises(ch.ChampionRegistryError, match="does not match"):
        ch.load_champion(tmp_path / "models" / "champion.json")


def _frame(start_step=1):
    rng = np.random.default_rng(4)
    steps = np.repeat(np.arange(start_step, start_step + 60), 20)
    n = len(steps)
    return pd.DataFrame({"step": steps, "type": rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT"], n),
                         "amount": rng.exponential(1e5, n).round(2), "transaction_hour": (steps - 1) % 24,
                         "nameDest": rng.choice([f"C{i}" for i in range(80)], n)})


def test_score_frame_computes_history_itself():
    c = ch.load_champion()
    df = _frame()
    got = ch.score_frame(c, df)
    manual = df.assign(**{RECIPIENT_PRIOR_IN_COUNT: compute_recipient_prior_in_count(df)})
    expected = c["model"].predict_proba(build_features(manual, c["config"])[c["feature_names"]])[:, 1]
    np.testing.assert_array_equal(got, expected)
    assert got.shape == (len(df),) and np.all((got >= 0) & (got <= 1))


def test_score_frame_refuses_incomplete_history_or_missing_recipient():
    c = ch.load_champion()
    with pytest.raises(ch.ChampionRegistryError, match="start at step 1"):
        ch.score_frame(c, _frame(start_step=100))
    assert len(ch.score_frame(c, _frame(start_step=100), allow_truncated_history=True)) == 1200
    with pytest.raises(ch.ChampionRegistryError, match="recipient column"):
        ch.score_frame(c, _frame().drop(columns=["nameDest"]))


def test_rebuild_config_matches_e4_and_was_verified():
    e4, final = MODEL_CONFIGS["hgb_baseline7_rcpt_cw50"], MODEL_CONFIGS["hgb_e4_final"]
    for k in ("feature_set", "amount_dtype", "drop_features", "history_features", "class_weight", "params"):
        assert final[k] == e4[k], k
    assert final["train_max_step"] == 550
    assert final["threshold_policy"] == {"name": "alert_budget", "alerts_per_day": 100, "val_min_step": 551, "val_max_step": 650}
    manifest = json.loads(Path("models/candidates/hgb_e4_final.manifest.json").read_text(encoding="utf-8"))
    v = manifest["verification"]
    assert v["reproduced"] is True and v["max_abs_score_difference"] == 0.0
    assert v["reference"].replace("\\", "/") == "models/hgb_e4_recipient.joblib"


def test_benchmark_sealed_rows_come_from_the_recorded_p15_result():
    from finsight.benchmark_view import sealed_rows

    p15 = json.loads(P15.read_text(encoding="utf-8"))
    rows = sealed_rows(p15)
    committed = json.loads(Path("reports/evaluation/benchmark_rows.json").read_text(encoding="utf-8"))
    sealed_committed = [r for r in committed["rows"] if r["evaluation_partition"] == "sealed_final_test"]
    assert rows == sealed_committed
    for r in rows:
        assert r["true_positives"] + r["false_positives"] + r["false_negatives"] + r["true_negatives"] == p15["systems"]["E4"]["rows"]
        assert r["true_positives"] + r["false_negatives"] == p15["systems"]["E4"]["labelled_fraud"]
    assert committed["active_champion_artifact_sha256"] == ch.read_registry()["active"]["artifact_sha256"]
