"""Milestones 6/7 — label-free KPI views (06-10), the read-only data-bundle builder and the static front end."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import dashboard as db  # noqa: E402
from finsight.champion import load_champion, read_registry  # noqa: E402
from tests.test_score_transactions import (  # noqa: E402,F401  (fixtures and helpers reused)
    SCRATCH, _frame, _load_transactions, _meta, needs_scratch, scratch_db,
)

KPI_VIEWS = ["06_vw_alert_workload_daily.sql", "07_vw_alert_mix.sql", "08_vw_alert_reason_summary.sql",
             "09_vw_alert_workload_hourly.sql", "10_vw_score_distribution.sql"]
FRONT_END = Path("dashboard")


def _code(path: Path) -> str:
    return "\n".join(l.split("--", 1)[0] for l in path.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("name", KPI_VIEWS)
def test_kpi_views_are_label_free_and_replaceable(name):
    code = _code(Path("sql/views") / name).lower()
    assert "create or replace view" in code
    for word in ("is_fraud", "is_flagged_fraud", "drop ", "truncate", "delete ", "update ", "insert ", "alter "):
        assert word not in code, word
    assert "evaluation_period" in code and "model_artifact_sha256" in code


# ------------------------------------------------------------------ fake connection for unit tests
class _Cur:
    def __init__(self, conn):
        self.conn, self.description, self._rows = conn, [("x",)], []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.calls.append(sql)
        if sql.startswith("SHOW transaction_read_only"):
            self.description, self._rows = [("transaction_read_only",)], [(self.conn.read_only,)]
        else:
            self.description, self._rows = self.conn.result_for(sql)

    def fetchone(self):
        return self._rows[0]

    def fetchall(self):
        return self._rows


class _Conn:
    def __init__(self, read_only="on", leak=False):
        self.read_only, self.leak, self.calls, self.rolled_back = read_only, leak, [], False

    def set_session(self, **kw):
        self.calls.append(("set_session", kw))

    def cursor(self):
        return _Cur(self)

    def rollback(self):
        self.rolled_back = True

    def result_for(self, sql):
        if self.leak and "vw_daily_transaction_kpis" in sql:
            return [("day",), ("is_fraud",)], [(1, 0)]
        if "FROM vw_alert_queue" in sql:
            cols = ["id", "step", "day", "hour", "type", "amount", "recipient", "period", "recipient_prior", "score",
                    "top_contributors", "model_summary", "rule_ids", "rule_conditions", "rule_score", "rule_tier"]
            row = (7, 3, 1, 2, "TRANSFER", 1e6, "C1", "training", 0, 0.97123456789,
                   [{"feature": "amount", "value": 1e6, "contribution_log_odds": 4.2, "description": "amount $1,000,000.00",
                     "rank": 1}], "<script>x</script>", ["none"],
                   [{"rule_id": "RULE_001_HIGH_VALUE", "points": 35, "message": "m", "extra": 1}], 35, "LOW")
            return [(c,) for c in cols], [row]
        return [("x",)], []


def test_collect_refuses_a_writable_session():
    conn = _Conn(read_only="off")
    with pytest.raises(RuntimeError, match="read-only"):
        db.collect(conn, read_registry())
    assert conn.calls[0] == ("set_session", {"readonly": True})
    assert not any("vw_" in c for c in conn.calls if isinstance(c, str))   # nothing read before the check


def test_collect_rejects_label_columns():
    with pytest.raises(ValueError, match="label"):
        db.collect(_Conn(leak=True), read_registry())


def test_collect_compacts_alerts_and_ends_read_only():
    conn = _Conn()
    data = db.collect(conn, read_registry(), {"ingestion": {"status": "ok"}})
    assert conn.rolled_back
    assert not any(re.search(r"\b(insert|update|delete|create|drop|alter|truncate)\b", c, re.I)
                   for c in conn.calls if isinstance(c, str))
    (a,) = data["alerts"]
    assert "model_summary" not in a and a["rule_ids"] == [] and a["score"] == 0.971235
    assert a["top_contributors"] == [{"feature": "amount", "value": 1e6, "contribution": 4.2, "description": "amount $1,000,000.00"}]
    assert a["rule_conditions"] == [{"rule_id": "RULE_001_HIGH_VALUE", "points": 35, "message": "m"}]
    assert data["notes"] == {"model": db.MODEL_NOTE, "rule": db.RULE_NOTE}
    assert data["registry"]["active"]["artifact_sha256"] == read_registry()["active"]["artifact_sha256"]
    assert data["ingestion"] == {"status": "ok"}


def test_write_bundle_is_a_valid_script(tmp_path):
    path = db.write_bundle({"alerts": [], "note": "</script> & ünïcode"}, tmp_path)
    text = path.read_text(encoding="utf-8")
    m = re.fullmatch(r"// [^\n]*\nwindow\.FINSIGHT_DATA = (.*);\n", text, flags=re.S)
    assert m and json.loads(m.group(1))["note"] == "</script> & ünïcode"


def test_load_extras_reads_local_reports():
    extras = db.load_extras(Path("."))
    if Path("reports/data_quality/pipeline_run.json").exists():
        assert set(extras["ingestion"]) >= {"status", "rows", "duplicates"}


def test_front_end_is_self_contained_and_never_injects_html():
    pages = {"index.html": ("app.css", "present.css", "data.js", "present.js", "details.html"),   # presentation
             "details.html": ("app.css", "data.js", "app.js", "index.html")}                     # full report
    for page, refs in pages.items():
        html = (FRONT_END / page).read_text(encoding="utf-8")
        for name in refs:
            assert name in html, (page, name)
    for f in ("index.html", "details.html", "app.css", "present.css", "app.js", "present.js"):
        text = (FRONT_END / f).read_text(encoding="utf-8")
        assert not re.search(r"https?://(?!www\.w3\.org/2000/svg)", text), f            # no external resources
    for f in ("app.js", "present.js"):
        js = (FRONT_END / f).read_text(encoding="utf-8")
        assert not re.search(r"\.innerHTML\s*=|insertAdjacentHTML|document\.write|\beval\(", js), f   # text only
    css = (FRONT_END / "app.css").read_text(encoding="utf-8")
    assert "prefers-color-scheme: dark" in css and '[data-theme="dark"]' in css


def test_presentation_reads_numbers_from_the_bundle():
    js = (FRONT_END / "present.js").read_text(encoding="utf-8")
    for key in ("at_budget", "average_precision", "paired_ap_difference_E4_minus_E0", "breakdown_100"):
        assert key in js                                                   # headline figures come from data, not typed in
    assert "not proof" in js                                               # SHAP caveat on the explanation screen


def test_front_end_carries_the_disclaimers():
    js = (FRONT_END / "app.js").read_text(encoding="utf-8")
    assert "D.notes.model" in js and "D.notes.rule" in js
    assert "pre-registered" in js and "not recomputed" in js           # test result shown as the one-time record


@needs_scratch
def test_kpi_views_reconcile_and_bundle_builds(scratch_db, tmp_path):
    from finsight import score_transactions as sc

    champ = load_champion()
    with scratch_db.cursor() as cur:
        for name in ["01_vw_daily_transaction_kpis.sql", "03_vw_payment_rail_performance.sql",
                     "04_vw_risk_detection_benchmark.sql"] + KPI_VIEWS:
            cur.execute((Path("sql/views") / name).read_text(encoding="utf-8"))
    scratch_db.commit()
    df = _frame(n_steps=48, per_step=30)
    _load_transactions(scratch_db, df)
    rows, table = sc.bulk_score(df, champ, 1, 48)
    recs = sc.reason_records(rows, table, champ)
    sc.write_run(scratch_db, table, recs, {**_meta(), "model_artifact_sha256": champ["entry"]["artifact_sha256"]})
    n_alerts = int(table["is_alert"].sum())
    cur = scratch_db.cursor()
    cur.execute("SELECT sum(alerts), sum(transactions_scored) FROM vw_alert_workload_daily")
    assert cur.fetchone() == (n_alerts, len(table))
    cur.execute("SELECT sum(alerts), sum(transactions_scored), count(*) FROM vw_alert_workload_hourly")
    assert cur.fetchone() == (n_alerts, len(table), 48)
    cur.execute("SELECT sum(transactions) FROM vw_score_distribution")
    assert cur.fetchone()[0] == len(table)
    cur.execute("SELECT sum(alerts) FROM vw_alert_mix")
    assert cur.fetchone()[0] == n_alerts
    cur.execute("SELECT sum(alerts) FROM vw_alert_reason_summary WHERE reason_source = 'top_model_contributor'")
    assert cur.fetchone()[0] == n_alerts
    scratch_db.commit()
    data = db.collect(scratch_db, read_registry())
    assert len(data["alerts"]) == n_alerts and sum(r["alerts"] for r in data["workload_hourly"]) == n_alerts
    assert sum(r["transactions"] for r in data["score_distribution"]) == len(table)
    assert len(data["runs"]) == 1
    path = db.write_bundle(data, tmp_path)
    assert path.stat().st_size > 0


# ------------------------------------------------------------------ study bundle (recorded results only)
def test_confusion_metrics_match_hand_computation():
    m = db.confusion_metrics(346, 4, 676, 79805)          # E4 at the deployed threshold, test hours
    assert m["accuracy"] == round((346 + 79805) / 80831, 4) and m["precision"] == round(346 / 350, 4)
    assert m["recall"] == round(346 / 1022, 4) and m["specificity"] == round(79805 / 79809, 4)
    assert m["f1"] == round(2 * 346 / (2 * 346 + 4 + 676), 4)
    num = 346 * 79805 - 4 * 676
    den = ((350) * (1022) * (79809) * (80481)) ** 0.5
    assert m["mcc"] == round(num / den, 4)
    lo, hi = m["precision_ci95"]
    assert lo < m["precision"] < hi <= 1
    never = db.confusion_metrics(0, 0, 1022, 79809)
    assert never["precision"] is None and never["recall"] == 0 and never["accuracy"] == round(79809 / 80831, 4) and never["mcc"] == 0.0


def test_wilson_interval_known_value():
    assert db.wilson(0, 0) is None
    assert db.wilson(50, 100) == [0.4038, 0.5962]


def test_load_study_reads_recorded_results_only():
    reg = read_registry()
    s = db.load_study(Path("."), reg)
    sealed = s["sealed"]["systems"]
    assert sealed["E4"]["average_precision"] == reg["active"]["evidence"]["sealed_test_average_precision"]
    assert sealed["E0"]["average_precision"] == reg["active"]["evidence"]["baseline_E0_average_precision"]
    assert s["sealed"]["paired_ap_difference_E4_minus_E0"]["ci95"] == reg["active"]["evidence"]["paired_ap_difference_ci95"]
    op = sealed["E4"]["at_threshold"]
    assert (op["tp"], op["fp"], op["fn"], op["tn"]) == (346, 4, 676, 79805)
    assert sealed["dummy"]["at_threshold"]["alerts"] == 0
    assert set(s["development"]["results"]) == {"E0", "E1", "E2", "E3", "E4"}
    assert s["model_card"]["artifact_sha256"] == reg["active"]["artifact_sha256"]
    assert s["model_card"]["sklearn_params"]["class_weight"] in ({0: 1.0, 1: 50.0}, {"0": 1.0, "1": 50.0})
    assert len(s["model_card"]["features"]) == 8 and "log1p_amount" not in s["model_card"]["features"]


def test_model_card_refuses_a_mismatched_artifact():
    reg = read_registry()
    bad = {"active": {**reg["active"], "artifact_sha256": "0" * 64}}
    with pytest.raises(ValueError, match="SHA-256"):
        db.model_card(Path("."), bad)
