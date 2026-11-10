"""
FinSight — Milestone 2 Loader & SQL Schema Automated Tests.

Tests configuration parsing, schema validation, COPY buffer serialization,
dry-run processing, and verifies that all 7 SQL analytics queries execute
cleanly against an in-memory sample database without requiring a live PostgreSQL instance.
"""

import gzip
import io
import os
import sqlite3
from pathlib import Path
import pytest
import pandas as pd

from finsight.load_to_postgres import (
    CSV_TO_DB_COLUMNS,
    EXPECTED_CSV_COLUMNS,
    DatabaseConfig,
    format_chunk_for_copy,
    load_dataset,
    validate_input_schema,
)


def test_database_config_env(monkeypatch):
    """Verify DatabaseConfig parses environment variables correctly."""
    monkeypatch.setenv("DB_HOST", "db.internal.finsight")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("DB_NAME", "analytics_db")
    monkeypatch.setenv("DB_USER", "analyst_usr")
    monkeypatch.setenv("DB_PASSWORD", "secret_pass_123")
    monkeypatch.setenv("DB_SCHEMA", "analytics")

    cfg = DatabaseConfig.from_env()
    assert cfg.host == "db.internal.finsight"
    assert cfg.port == 5433
    assert cfg.name == "analytics_db"
    assert cfg.user == "analyst_usr"
    assert cfg.password == "secret_pass_123"
    assert cfg.schema == "analytics"

    sanitized = cfg.sanitized_connection_string()
    assert "secret_pass_123" not in sanitized
    assert "****" in sanitized
    assert "analyst_usr" in sanitized


def test_validate_input_schema_valid():
    """Verify schema validator accepts the real processed dataset header."""
    csv_path = Path("data/processed/transactions.csv.gz")
    assert csv_path.exists(), "Processed dataset must exist"
    cols = validate_input_schema(csv_path)
    for expected in EXPECTED_CSV_COLUMNS:
        assert expected in cols


def test_validate_input_schema_missing_column(tmp_path):
    """Verify schema validator raises ValueError on missing columns."""
    bad_csv = tmp_path / "bad.csv"
    df = pd.DataFrame({"step": [1], "type": ["PAYMENT"]})
    df.to_csv(bad_csv, index=False)

    with pytest.raises(ValueError, match="missing required columns"):
        validate_input_schema(bad_csv)


def test_validate_input_schema_missing_file(tmp_path):
    """Verify schema validator raises FileNotFoundError for missing path."""
    with pytest.raises(FileNotFoundError):
        validate_input_schema(tmp_path / "non_existent.csv.gz")


def test_format_chunk_for_copy():
    """Verify DataFrame chunk formats into valid CSV stream without header."""
    data = {
        "step": [1, 2],
        "type": ["PAYMENT", "CASH_OUT"],
        "amount": [100.50, 2500.00],
        "nameOrig": ["C1", "C2"],
        "oldbalanceOrg": [500.0, 0.0],
        "newbalanceOrig": [399.5, 0.0],
        "nameDest": ["M1", "C3"],
        "oldbalanceDest": [0.0, 100.0],
        "newbalanceDest": [0.0, 2600.0],
        "isFraud": [0, 1],
        "isFlaggedFraud": [0, 0],
        "transaction_hour": [0, 1],
        "transaction_day": [1, 1],
    }
    df = pd.DataFrame(data)
    buffer = format_chunk_for_copy(df)
    lines = buffer.getvalue().strip().split("\n")

    assert len(lines) == 2
    # Verify exactly 13 fields per row
    assert len(lines[0].split(",")) == 13
    assert lines[0].startswith("1,PAYMENT,100.5,C1")


def test_loader_dry_run():
    """Verify dry_run executes without connecting to database."""
    csv_path = Path("data/processed/transactions.csv.gz")
    result = load_dataset(
        csv_path=csv_path,
        dry_run=True,
        max_rows=500,
        chunk_size=200,
    )
    assert result["status"] == "DRY_RUN_SUCCESS"
    assert result["rows_inspected"] == 500
    assert result["chunks_inspected"] == 3
    assert result["dry_run"] is True


def test_sql_schema_ddl_syntax_and_contents():
    """Verify DDL file defines fct_transactions with primary key, constraints, and comments."""
    ddl_path = Path("sql/schema/01_schema_definition.sql")
    assert ddl_path.exists()
    sql_text = ddl_path.read_text(encoding="utf-8")

    assert "CREATE TABLE fct_transactions" in sql_text
    assert "transaction_id BIGSERIAL PRIMARY KEY" in sql_text
    assert "transaction_hour INT NOT NULL CHECK" in sql_text
    assert "transaction_day INT NOT NULL CHECK" in sql_text
    assert "is_fraud INT NOT NULL CHECK" in sql_text
    assert "is_flagged_fraud INT NOT NULL CHECK" in sql_text
    assert "COMMENT ON COLUMN fct_transactions.name_orig" in sql_text
    assert "COMMENT ON COLUMN fct_transactions.is_fraud" in sql_text
    assert "idx_fct_type_fraud" in sql_text


def test_sql_analytics_queries_execution():
    """Verify all 7 analytics queries execute without error and return expected structure."""
    analytics_dir = Path("sql/analytics")
    assert analytics_dir.exists()
    query_files = sorted(analytics_dir.glob("*.sql"))
    assert len(query_files) == 7

    # Load small sample into in-memory SQLite to verify SQL syntax and logic
    sample_df = pd.read_csv("data/processed/transactions.csv.gz", nrows=2000)
    db_df = sample_df.rename(columns=CSV_TO_DB_COLUMNS)

    con = sqlite3.connect(":memory:")
    db_df.to_sql("fct_transactions", con, index=False)

    for q_path in query_files:
        query_sql = q_path.read_text(encoding="utf-8")
        cur = con.execute(query_sql)
        rows = cur.fetchall()
        assert len(rows) > 0, f"Query {q_path.name} returned 0 rows"
        assert cur.description is not None, f"Query {q_path.name} returned no description"

    con.close()


def test_table_exists_error_class():
    """Verify TableExistsError exception behavior."""
    from finsight.load_to_postgres import TableExistsError
    with pytest.raises(TableExistsError, match="Target table already contains"):
        raise TableExistsError("Target table already contains 1,000 records")


def test_live_postgres_sample_count_if_available():
    """If live postgres is configured and reachable, verify fct_transactions has valid loaded row count."""
    from finsight.load_to_postgres import DatabaseConfig, check_postgres_connection, PSYCOPG2_AVAILABLE
    if not PSYCOPG2_AVAILABLE:
        pytest.skip("psycopg2 not installed")

    cfg = DatabaseConfig.from_env()
    connected, _ = check_postgres_connection(cfg)
    if not connected:
        pytest.skip("Live PostgreSQL not reachable")

    import psycopg2
    conn = psycopg2.connect(
        host=cfg.host, port=cfg.port, dbname=cfg.name, user=cfg.user, password=cfg.password
    )
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM fct_transactions;")
        count = cur.fetchone()[0]
        assert count in (1000, 6362620), f"Expected 1,000 sample rows or 6,362,620 full rows, found {count}"
    conn.close()

