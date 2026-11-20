"""Milestone 6/7 — KPI views (06-08) and the HTML/CSS dashboard."""

from __future__ import annotations

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

KPI_VIEWS = ["06_vw_alert_workload_daily.sql", "07_vw_alert_mix.sql", "08_vw_alert_reason_summary.sql"]


def _code(path: Path) -> str:
    return "\n".join(l.split("--", 1)[0] for l in path.read_text(encoding="utf-8").splitlines())


@pytest.mark.parametrize("name", KPI_VIEWS)
def test_kpi_views_are_label_free_and_replaceable(name):
    code = _code(Path("sql/views") / name).lower()
    assert "create or replace view" in code
    for word in ("is_fraud", "is_flagged_fraud", "drop ", "truncate", "delete ", "update ", "insert ", "alter "):
        assert word not in code, word
    assert "evaluation_period" in code


def _snapshot(**over):
    d = {
        "dataset": {"transactions": 1000, "value": 1e6, "labelled_fraud": 10, "days": 2},
        "rails": [{"payment_rail": "TRANSFER", "transaction_count": 400, "rail_volume_share_pct": 40.0,
                   "labelled_fraud_count": 10, "labelled_fraud_rate_pct": 2.5}],
        "benchmark": [{"evaluation_partition": "sealed_final_test", "system_name": "HistGradientBoosting E4 (active champion, recipient count)",
                       "operating_point": "t", "alerts_per_day": 90.3, "precision": 0.989, "recall": 0.339, "average_precision": 0.6966,
                       "true_positives": 1, "false_positives": 0, "false_negatives": 1},
                      {"evaluation_partition": "sealed_final_test", "system_name": "HistGradientBoosting (previous champion, 8 features)",
                       "operating_point": "t", "alerts_per_day": 173.2, "precision": 0.64, "recall": 0.42, "average_precision": 0.4909,
                       "true_positives": 1, "false_positives": 1, "false_negatives": 1}],
        "workload": [{"simulated_day": 1, "evaluation_period": "training", "hours_covered": 24, "is_full_day": True,
                      "transactions_scored": 500, "alerts": 120, "first_time_recipient_alerts": 40}],
        "mix": [{"evaluation_period": "training", "payment_type": "TRANSFER", "hour_band": "00-05", "amount_band": ">=1M", "alerts": 120}],
        "reasons": [{"evaluation_period": "training", "reason_source": "top_model_contributor", "reason": "amount", "alerts": 120},
                    {"evaluation_period": "training", "reason_source": "rule_condition", "reason": "RULE_002_DRAINAGE_RAIL", "alerts": 120}],
        "queue": [{"transaction_id": 7, "step": 3, "simulated_day": 1, "transaction_hour": 2, "type": "TRANSFER", "amount": 1e6,
                   "evaluation_period": "training", "risk_score": 0.97, "model_summary": "<script>alert(1)</script> summary",
                   "rule_ids": ["RULE_001_HIGH_VALUE"], "rule_tier": "HIGH"}],
        "runs": {"runs": 1, "rows_scored": 500, "alerts": 120, "last_run": "2026-11-20"},
        "champion": {k: read_registry()["active"][k] for k in ("artifact", "artifact_sha256", "model_name", "operating_point", "evidence")},
        "generated_utc": "now", "source": {"database": "test"},
    }
    d.update(over)
    return d


def test_render_is_self_contained_escaped_and_carries_disclaimers():
    page = db.render(_snapshot())
    assert page.startswith("<!doctype html>") and "</html>" in page
    assert "<script>alert(1)</script>" not in page and "&lt;script&gt;" in page      # text is escaped
    assert not re.search(r"<script|https?://|<link ", page, flags=re.I)                 # no scripts or external resources
    assert db.MODEL_NOTE in page and db.RULE_NOTE in page
    assert "prefers-color-scheme:dark" in page and 'data-theme="dark"' in page          # both themes
    assert "0.697" in page and "0.491" in page and "class=hl" in page                   # champion row highlighted


def test_render_handles_an_empty_scoring_store():
    page = db.render(_snapshot(workload=[], mix=[], reasons=[], queue=[], runs={"runs": 0, "rows_scored": 0, "alerts": 0, "last_run": None}))
    assert page.count("No scores stored yet") >= 2 and "No alerts stored yet" in page


def test_collect_uses_a_read_only_session():
    calls = []

    class Cur:
        description = [("x",)]
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, sql, params=None):
            calls.append(sql)
            if "SHOW transaction_read_only" in sql:
                self.description = [("transaction_read_only",)]
        def fetchone(self): return ("off",)
        def fetchall(self): return []

    class Conn:
        def set_session(self, **kw): calls.append(("set_session", kw))
        def cursor(self): return Cur()
        def rollback(self): pass

    with pytest.raises(AssertionError):            # refuses to continue unless the session is read-only
        db.collect(Conn(), read_registry()["active"])
    assert calls[0] == ("set_session", {"readonly": True})


@needs_scratch
def test_kpi_views_reconcile_and_dashboard_builds(scratch_db):
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
    cur.execute("SELECT sum(alerts) FROM vw_alert_mix")
    assert cur.fetchone()[0] == n_alerts
    cur.execute("SELECT sum(alerts) FROM vw_alert_reason_summary WHERE reason_source = 'top_model_contributor'")
    assert cur.fetchone()[0] == n_alerts
    scratch_db.commit()
    snap = db.collect(scratch_db, champ["entry"])
    assert sum(r["alerts"] for r in snap["workload"]) == n_alerts and len(snap["queue"]) == min(25, n_alerts)
    assert "FinSight dashboard" in db.render({**snap, "generated_utc": "t", "source": {"database": "scratch"}})
