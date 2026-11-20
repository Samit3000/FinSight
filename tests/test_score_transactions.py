"""P3 — scoring store. Unit tests always run; integration tests need a DISPOSABLE PostgreSQL.

Set FINSIGHT_SCRATCH_PG="host:port:user" to run the integration tests. Each one creates its own throwaway
database on that server, applies the real SQL files, and drops it afterwards. They never use DB_NAME.
"""

from __future__ import annotations

import os
import re
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from finsight import score_transactions as sc  # noqa: E402
from finsight.alert_explanations import MODEL_DISCLAIMER, RULE_NOTE  # noqa: E402
from finsight.champion import load_champion, operating_threshold, score_frame  # noqa: E402

SCHEMA = Path("sql/schema/02_scoring_schema.sql")
VIEW = Path("sql/views/05_vw_alert_queue.sql")
ROLLBACK = Path("sql/admin/drop_scoring_objects.sql")


# ------------------------------------------------------------------------------ SQL files
def _code(path: Path) -> str:
    return "\n".join(l.split("--", 1)[0] for l in path.read_text(encoding="utf-8").splitlines())


def test_scoring_schema_is_non_destructive_and_idempotent():
    code = _code(SCHEMA).upper()
    for word in ("DROP ", "TRUNCATE", "ALTER ", "DELETE ", "UPDATE ", "INSERT "):
        assert word not in code, word
    creates = re.findall(r"CREATE (?:TABLE|INDEX)[^;]*", code)
    assert creates and all("IF NOT EXISTS" in c for c in creates)
    assert "FCT_TRANSACTIONS (TRANSACTION_ID)" in code   # references, never modifies, the fact table


def test_view_has_no_labels_and_rollback_touches_only_p3_objects():
    view = _code(VIEW).lower()
    assert "is_fraud" not in view and "is_flagged_fraud" not in view
    assert "create or replace view vw_alert_queue" in view
    rb = _code(ROLLBACK).lower()
    dropped = set(re.findall(r"drop (?:table|view) if exists (\w+)", rb))
    assert dropped == {"vw_alert_queue", "fct_alert_reasons", "fct_model_scores", "scoring_run_log"}
    assert "fct_transactions" not in rb and "dq_audit_log" not in rb


def test_period_labels():
    assert [sc.period_label(s) for s in (1, 550, 551, 650, 651, 743)] == [
        "training", "training", "calibration", "calibration", "sealed_test_reported", "sealed_test_reported"]


# ------------------------------------------------------------------------------ data and scoring
def _frame(n_steps=60, per_step=50, seed=3):
    rng = np.random.default_rng(seed)
    steps = np.repeat(np.arange(1, n_steps + 1), per_step)
    n = len(steps)
    types = rng.choice(["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN"], n)
    amount = np.where(np.isin(types, ["TRANSFER", "CASH_OUT"]) & (rng.random(n) < 0.3),
                      rng.uniform(5e5, 5e6, n), rng.exponential(5e4, n)).round(2)
    df = pd.DataFrame({"step": steps, "type": types, "amount": amount, "transaction_hour": (steps - 1) % 24,
                       "nameOrig": [f"O{i}" for i in range(n)], "nameDest": rng.choice([f"C{i}" for i in range(300)], n)})
    df.index = pd.RangeIndex(1, n + 1, name="transaction_id")
    return df


@pytest.fixture(scope="module")
def champ():
    return load_champion()


def test_load_unlabelled_never_returns_labels(tmp_path):
    df = _frame().reset_index(drop=True).assign(isFraud=1, isFlaggedFraud=0)
    path = tmp_path / "tx.csv"
    df.to_csv(path, index=False)
    out = sc.load_unlabelled(path, max_step=30)
    assert not set(sc.LABEL_COLUMNS) & set(out.columns)
    assert out.index[0] == 1 and out["step"].max() == 30
    assert (out.index.to_numpy() == np.arange(1, len(out) + 1)).all()


def test_bulk_scores_equal_the_champion(champ):
    df = _frame()
    rows, table = sc.bulk_score(df, champ, 1, 60)
    np.testing.assert_array_equal(table["risk_score"].to_numpy(), score_frame(champ, df))
    assert (table["is_alert"] == (table["risk_score"] >= operating_threshold(champ))).all()
    assert (table["transaction_id"].to_numpy() == df.index.to_numpy()).all()


def test_hourly_equals_bulk(champ):
    df = _frame()
    _, bulk = sc.bulk_score(df, champ, 21, 60)
    hourly = pd.concat([t for _, _, t in sc.hourly_score(df, champ, 21, 60)], ignore_index=True)
    pd.testing.assert_frame_equal(hourly.reset_index(drop=True), bulk.reset_index(drop=True))


def test_scorer_refuses_labels_and_truncated_history(champ):
    with pytest.raises(sc.ScoringError, match="label"):
        sc.bulk_score(_frame().assign(isFraud=0), champ, 1, 10)
    with pytest.raises(sc.ScoringError, match="label"):
        next(iter(sc.hourly_score(_frame().assign(is_fraud=0), champ, 1, 10)))
    with pytest.raises(sc.ScoringError, match="step 1"):
        sc.bulk_score(_frame()[lambda d: d.step > 3], champ, 10, 20)


def test_reason_records_one_per_alert(champ):
    df = _frame()
    rows, table = sc.bulk_score(df, champ, 1, 60)
    recs = sc.reason_records(rows, table, champ)
    alerts = table.loc[table["is_alert"], "transaction_id"].tolist()
    assert alerts and [r["transaction_id"] for r in recs] == alerts
    for r in recs:
        m = r["model_explanation"]
        assert len(m["top_contributors"]) == 3 and m["additivity_error"] < 1e-6
        assert m["disclaimer"] == MODEL_DISCLAIMER and r["rule_conditions"]["note"] == RULE_NOTE


# ------------------------------------------------------------------------------ write path (fake connection)
class _FakeCursor:
    def __init__(self, conn):
        self.conn = conn

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.conn.log.append(sql.split()[0])
        self._last = sql

    def fetchone(self):
        return (self.conn.existing,)

    def copy_expert(self, sql, buf):
        if self.conn.fail_on_copy:
            raise RuntimeError("disk full")
        self.conn.log.append("COPY")


class _FakeConn:
    def __init__(self, existing=0, fail_on_copy=False):
        self.existing, self.fail_on_copy, self.log = existing, fail_on_copy, []
        self.commits = self.rollbacks = 0

    def cursor(self):
        return _FakeCursor(self)

    def commit(self):
        self.commits += 1

    def rollback(self):
        self.rollbacks += 1


def _meta():
    return {"model_artifact": "m", "model_artifact_sha256": "0" * 64, "config_name": "c", "mode": "bulk",
            "data_sha256": "1" * 64, "code_commit": "abc", "started_at": pd.Timestamp.now('UTC').to_pydatetime()}


def _table(alerts=0):
    t = pd.DataFrame({"transaction_id": [1, 2], "step": [1, 1], "evaluation_period": ["training"] * 2,
                      "recipient_prior_in_count": [0, 0], "risk_score": [0.1, 0.2], "operating_threshold": 0.9,
                      "is_alert": [False, False]})
    if alerts:
        t.loc[0, ["risk_score", "is_alert"]] = [0.95, True]
    return t


def test_write_refuses_rescoring_without_replace(monkeypatch):
    monkeypatch.setattr(sc, "_insert_reasons", lambda *a: None)
    conn = _FakeConn(existing=5)
    with pytest.raises(sc.AlreadyScoredError):
        sc.write_run(conn, _table(), [], _meta())
    assert conn.commits == 0 and conn.rollbacks == 1 and "COPY" not in conn.log


def test_write_failure_rolls_back(monkeypatch):
    monkeypatch.setattr(sc, "_insert_reasons", lambda *a: None)
    conn = _FakeConn(fail_on_copy=True)
    with pytest.raises(RuntimeError, match="disk full"):
        sc.write_run(conn, _table(), [], _meta())
    assert conn.commits == 0 and conn.rollbacks == 1


def test_write_success_commits_once_and_needs_a_reason_per_alert(monkeypatch):
    monkeypatch.setattr(sc, "_insert_reasons", lambda *a: None)
    conn = _FakeConn()
    sc.write_run(conn, _table(), [], _meta())
    assert conn.commits == 1 and conn.rollbacks == 0 and conn.log.count("COPY") == 1
    with pytest.raises(sc.ScoringError, match="reason record"):
        sc.write_run(_FakeConn(), _table(alerts=1), [], _meta())


def test_mapping_mismatch_raises():
    rows = _frame(n_steps=1, per_step=3)

    class Cur(_FakeCursor):
        def fetchall(self):
            return [(1, 1, rows.iloc[0]["type"], rows.iloc[0]["amount"], "O0", rows.iloc[0]["nameDest"]),
                    (2, 1, rows.iloc[1]["type"], rows.iloc[1]["amount"], "WRONG", rows.iloc[1]["nameDest"]),
                    (3, 1, rows.iloc[2]["type"], rows.iloc[2]["amount"], "O2", rows.iloc[2]["nameDest"])]

    class Conn(_FakeConn):
        def cursor(self):
            return Cur(self)

    with pytest.raises(sc.ScoringError, match="mapping mismatch"):
        sc.verify_mapping(Conn(), rows)


def test_cli_refuses_wrong_database(monkeypatch):
    monkeypatch.setenv("DB_NAME", "finsight_db")
    monkeypatch.setattr(sys, "argv", ["x", "--mode", "bulk", "--confirm-database", "something_else"])
    with pytest.raises(SystemExit, match="does not match"):
        sc.main()


# ------------------------------------------------------------------------------ integration (disposable DB only)
SCRATCH = os.environ.get("FINSIGHT_SCRATCH_PG")
needs_scratch = pytest.mark.skipif(not SCRATCH, reason="set FINSIGHT_SCRATCH_PG=host:port:user for a DISPOSABLE PostgreSQL")


@pytest.fixture()
def scratch_db():
    import psycopg2

    host, port, user = SCRATCH.split(":")
    name = f"finsight_pytest_{uuid.uuid4().hex[:8]}"
    admin = psycopg2.connect(host=host, port=int(port), user=user, dbname="postgres")
    admin.autocommit = True
    admin.cursor().execute(f"CREATE DATABASE {name}")
    conn = psycopg2.connect(host=host, port=int(port), user=user, dbname=name)
    try:
        with conn.cursor() as cur:
            cur.execute(Path("sql/schema/01_schema_definition.sql").read_text(encoding="utf-8"))
            cur.execute(SCHEMA.read_text(encoding="utf-8"))
            cur.execute(SCHEMA.read_text(encoding="utf-8"))          # idempotent: second run is a no-op
            cur.execute(VIEW.read_text(encoding="utf-8"))
        conn.commit()
        yield conn
    finally:
        conn.close()
        admin.cursor().execute(f"DROP DATABASE {name}")
        admin.close()


def _load_transactions(conn, df):
    with conn.cursor() as cur:
        for tid, r in df.iterrows():
            cur.execute("INSERT INTO fct_transactions (transaction_id, step, type, amount, name_orig, oldbalance_org, "
                        "newbalance_orig, name_dest, oldbalance_dest, newbalance_dest, is_fraud, is_flagged_fraud, "
                        "transaction_hour, transaction_day) VALUES (%s,%s,%s,%s,%s,0,0,%s,0,0,0,0,%s,%s)",
                        (int(tid), int(r.step), r.type, float(r.amount), r.nameOrig, r.nameDest, int(r.transaction_hour),
                         int((r.step - 1) // 24 + 1)))
    conn.commit()


@needs_scratch
def test_db_roundtrip_idempotency_rollback_and_view(scratch_db, champ, monkeypatch):
    df = _frame(n_steps=30, per_step=30)
    _load_transactions(scratch_db, df)
    rows, table = sc.bulk_score(df, champ, 1, 30)
    assert sc.verify_mapping(scratch_db, rows) == len(rows)
    recs = sc.reason_records(rows, table, champ)
    meta = {**_meta(), "model_artifact_sha256": champ["entry"]["artifact_sha256"]}
    sc.write_run(scratch_db, table, recs, meta)
    cur = scratch_db.cursor()
    cur.execute("SELECT count(*), sum(is_alert::int) FROM fct_model_scores")
    assert cur.fetchone() == (len(table), int(table["is_alert"].sum()))
    cur.execute("SELECT count(*) FROM fct_alert_reasons")
    assert cur.fetchone()[0] == len(recs)
    cur.execute("SELECT count(*) FROM vw_alert_queue")
    assert cur.fetchone()[0] == len(recs)
    cur.execute("SELECT max(abs(s.risk_score - x.v)) FROM fct_model_scores s JOIN unnest(%s::bigint[], %s::float8[]) AS x(id, v) "
                "ON x.id = s.transaction_id", (table["transaction_id"].tolist(), table["risk_score"].tolist()))
    assert cur.fetchone()[0] < 1e-12
    with pytest.raises(sc.AlreadyScoredError):                    # idempotent
        sc.write_run(scratch_db, table, recs, meta)
    def boom(*a):
        raise RuntimeError("fail after scores were copied")
    monkeypatch.setattr(sc, "_insert_reasons", boom)
    with pytest.raises(RuntimeError):                             # --replace that fails mid-run
        sc.write_run(scratch_db, table, recs, meta, replace=True)
    cur.execute("SELECT count(*), (SELECT count(*) FROM fct_alert_reasons), (SELECT count(*) FROM scoring_run_log) "
                "FROM fct_model_scores")
    assert cur.fetchone() == (len(table), len(recs), 1)          # earlier run intact, nothing partial


@needs_scratch
def test_db_constraints_reject_inconsistent_rows(scratch_db):
    import psycopg2

    _load_transactions(scratch_db, _frame(n_steps=1, per_step=2))
    cur = scratch_db.cursor()
    for period, score, alert in (("calibration", 0.5, False), ("training", 0.95, False)):   # wrong period; wrong flag
        with pytest.raises(psycopg2.errors.CheckViolation):
            cur.execute("INSERT INTO fct_model_scores VALUES (1, %s, 'r', 1, %s, 0, %s, 0.9, %s)", ("0" * 64, period, score, alert))
        scratch_db.rollback()
