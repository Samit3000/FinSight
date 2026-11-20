"""FinSight — Database safety tests (P0-1).

Guards two properties:
1. Ordinary schema setup can never destroy data (no DROP/TRUNCATE in the setup DDL).
2. A `--truncate` reload is atomic: TRUNCATE and COPY share one transaction, so a
   failed reload rolls back to the previous rows instead of leaving an empty table.

The loader tests use an in-memory fake connection; they never touch a real database.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, List

import pandas as pd
import pytest

import finsight.load_to_postgres as loader
from finsight.load_to_postgres import EXPECTED_CSV_COLUMNS, DatabaseConfig, load_dataset

SCHEMA_DDL = Path("sql/schema/01_schema_definition.sql")
RESET_DDL = Path("sql/admin/reset_schema.sql")


def _sql_statements_without_comments(text: str) -> str:
    return "\n".join(line.split("--", 1)[0] for line in text.splitlines())


def test_schema_ddl_contains_no_destructive_statements():
    body = _sql_statements_without_comments(SCHEMA_DDL.read_text(encoding="utf-8")).upper()
    assert not re.search(r"\bDROP\b", body), "Setup DDL must never DROP objects"
    assert not re.search(r"\bTRUNCATE\b", body), "Setup DDL must never TRUNCATE"
    assert not re.search(r"\bDELETE\b", body), "Setup DDL must never DELETE"


def test_schema_ddl_is_idempotent():
    body = _sql_statements_without_comments(SCHEMA_DDL.read_text(encoding="utf-8"))
    creates = re.findall(r"CREATE\s+(?:TABLE|INDEX)\s+(?!IF NOT EXISTS)\w+", body, flags=re.I)
    assert creates == [], f"Non-idempotent CREATE statements found: {creates}"


def test_reset_script_is_isolated_from_code():
    assert RESET_DDL.exists()
    for py in Path("src").rglob("*.py"):
        assert "reset_schema" not in py.read_text(encoding="utf-8"), f"{py} references the reset script"


# ------------------------------------------------------------------------------
# Fake psycopg2 connection recording every database interaction in order.
# ------------------------------------------------------------------------------
class _FakeCursor:
    def __init__(self, conn: "_FakeConn") -> None:
        self.conn = conn
        self._result: Any = None

    def __enter__(self) -> "_FakeCursor":
        return self

    def __exit__(self, *exc: Any) -> None:
        return None

    def execute(self, sql: str, params: Any = None) -> None:
        text = " ".join(sql.split()).upper()
        if text.startswith("TRUNCATE"):
            self.conn.events.append("TRUNCATE")
        elif "INFORMATION_SCHEMA.TABLES" in text:
            self._result = (True,)
        elif text.startswith("SELECT COUNT(*), SUM(AMOUNT)"):
            self._result = (self.conn.copied_rows, 1.0)
        elif text.startswith("SELECT COUNT(*)"):
            self._result = (1000,)

    def fetchone(self) -> Any:
        return self._result

    def copy_expert(self, sql: str, buffer: Any) -> None:
        self.conn.copy_calls += 1
        if self.conn.fail_on_copy == self.conn.copy_calls:
            self.conn.events.append("COPY_FAILED")
            raise RuntimeError("simulated COPY failure")
        self.conn.copied_rows += sum(1 for _ in buffer)
        self.conn.events.append("COPY")


class _FakeConn:
    def __init__(self, fail_on_copy: int | None = None) -> None:
        self.events: List[str] = []
        self.copy_calls = 0
        self.copied_rows = 0
        self.fail_on_copy = fail_on_copy

    def cursor(self) -> _FakeCursor:
        return _FakeCursor(self)

    def commit(self) -> None:
        self.events.append("COMMIT")

    def rollback(self) -> None:
        self.events.append("ROLLBACK")

    def close(self) -> None:
        self.events.append("CLOSE")


@pytest.fixture
def processed_csv(tmp_path: Path) -> Path:
    n = 50
    df = pd.DataFrame({
        "step": [1] * n,
        "type": ["TRANSFER"] * n,
        "amount": [100.0] * n,
        "nameOrig": [f"C{i}" for i in range(n)],
        "oldbalanceOrg": [100.0] * n,
        "newbalanceOrig": [0.0] * n,
        "nameDest": [f"C9{i}" for i in range(n)],
        "oldbalanceDest": [0.0] * n,
        "newbalanceDest": [100.0] * n,
        "isFraud": [0] * n,
        "isFlaggedFraud": [0] * n,
        "transaction_hour": [0] * n,
        "transaction_day": [1] * n,
    })[EXPECTED_CSV_COLUMNS]
    path = tmp_path / "processed.csv.gz"
    df.to_csv(path, index=False)
    return path


def _patch_db(monkeypatch: pytest.MonkeyPatch, conn: _FakeConn) -> None:
    monkeypatch.setattr(loader, "PSYCOPG2_AVAILABLE", True)
    monkeypatch.setattr(loader, "check_postgres_connection", lambda cfg: (True, "fake"))

    class _FakePsycopg2:
        @staticmethod
        def connect(**kwargs: Any) -> _FakeConn:
            return conn

    monkeypatch.setattr(loader, "psycopg2", _FakePsycopg2)


def test_truncate_reload_commits_once_after_all_copies(monkeypatch, processed_csv):
    conn = _FakeConn()
    _patch_db(monkeypatch, conn)
    result = load_dataset(csv_path=processed_csv, config=DatabaseConfig(), truncate=True, chunk_size=20)
    assert result["rows_loaded"] == 50
    first_commit = conn.events.index("COMMIT")
    assert conn.events[:first_commit] == ["TRUNCATE", "COPY", "COPY", "COPY"], conn.events
    assert conn.events.count("COMMIT") == 1


def test_failed_truncate_reload_rolls_back_without_committing_truncate(monkeypatch, processed_csv):
    conn = _FakeConn(fail_on_copy=2)
    _patch_db(monkeypatch, conn)
    with pytest.raises(RuntimeError, match="simulated COPY failure"):
        load_dataset(csv_path=processed_csv, config=DatabaseConfig(), truncate=True, chunk_size=20)
    assert "COMMIT" not in conn.events, f"TRUNCATE must not be committed before COPY succeeds: {conn.events}"
    assert conn.events[-2:] == ["ROLLBACK", "CLOSE"]
