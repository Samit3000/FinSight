"""FinSight — Tests for Analytical SQL Views and Financial Metric Reconciliation.

Validates view file existence, SQL syntax, financial metric nomenclature,
and (when live PostgreSQL is reachable) full database reconciliation against
fct_transactions ground truth.
"""

from pathlib import Path
import pytest

from finsight.load_to_postgres import DatabaseConfig, check_postgres_connection, PSYCOPG2_AVAILABLE


EXPECTED_VIEW_FILES = [
    "01_vw_daily_transaction_kpis.sql",
    "02_vw_hourly_risk_dynamics.sql",
    "03_vw_payment_rail_performance.sql",
    "04_vw_risk_detection_benchmark.sql",
]


def test_view_definition_files_exist():
    """Verify all 4 required analytical view DDL files exist in sql/views/."""
    views_dir = Path("sql/views")
    assert views_dir.exists() and views_dir.is_dir(), "sql/views directory missing"

    for vf in EXPECTED_VIEW_FILES:
        path = views_dir / vf
        assert path.exists(), f"View file missing: {path}"
        content = path.read_text(encoding="utf-8")
        assert "CREATE OR REPLACE VIEW" in content, f"View {vf} missing idempotent CREATE OR REPLACE VIEW statement"


def test_financial_metric_nomenclature_contracts():
    """Verify views use defensible financial KPI names and do not use misleading terminology."""
    views_dir = Path("sql/views")

    # Check daily view
    daily_sql = (views_dir / "01_vw_daily_transaction_kpis.sql").read_text(encoding="utf-8")
    assert "fraud_transfer_face_value" in daily_sql, "Daily view missing fraud_transfer_face_value"
    assert "net_simulated_economic_leakage" not in daily_sql, "Misleading metric name net_simulated_economic_leakage found in daily view"
    assert "gross_labelled_fraud_value" in daily_sql, "Daily view missing gross_labelled_fraud_value"

    # Check hourly view
    hourly_sql = (views_dir / "02_vw_hourly_risk_dynamics.sql").read_text(encoding="utf-8")
    assert "fraud_transfer_face_value" in hourly_sql, "Hourly view missing fraud_transfer_face_value"
    assert "net_simulated_economic_leakage" not in hourly_sql, "Misleading metric name in hourly view"

    # Check benchmark view
    bench_sql = (views_dir / "04_vw_risk_detection_benchmark.sql").read_text(encoding="utf-8")
    assert "total_evaluation_population" in bench_sql, "Benchmark view missing total_evaluation_population"
    assert "alert_rate_pct" in bench_sql, "Benchmark view missing alert_rate_pct"


def test_live_postgres_view_reconciliation_if_available():
    """Execute reconciliation checks on live PostgreSQL if reachable."""
    if not PSYCOPG2_AVAILABLE:
        pytest.skip("psycopg2 not installed")

    cfg = DatabaseConfig.from_env()
    connected, _ = check_postgres_connection(cfg)
    if not connected:
        pytest.skip("Live PostgreSQL instance not reachable")

    import psycopg2
    conn = psycopg2.connect(
        host=cfg.host, port=cfg.port, dbname=cfg.name, user=cfg.user, password=cfg.password
    )
    conn.autocommit = True
    cur = conn.cursor()

    # Apply views
    views_dir = Path("sql/views")
    for vf in EXPECTED_VIEW_FILES:
        sql = (views_dir / vf).read_text(encoding="utf-8")
        cur.execute(sql)

    # 1. Authoritative source counts
    cur.execute("""
        SELECT 
            COUNT(*), 
            ROUND(SUM(amount), 2), 
            COUNT(*) FILTER (WHERE is_fraud = 1),
            ROUND(SUM(amount) FILTER (WHERE is_fraud = 1), 2),
            ROUND(SUM(amount) FILTER (WHERE is_fraud = 1 AND type = 'TRANSFER'), 2)
        FROM fct_transactions;
    """)
    src_cnt, src_gtv, src_fraud_cnt, src_fraud_val, src_trans_val = cur.fetchone()

    # Skip full reconciliation if running on a dev subset
    if src_cnt != 6362620:
        conn.close()
        pytest.skip(f"Live database contains {src_cnt} rows, expected full 6,362,620 rows for strict reconciliation")

    assert float(src_gtv) == 1144392944759.77
    assert src_fraud_cnt == 8213
    assert float(src_fraud_val) == 12056415427.84
    assert float(src_trans_val) == 6067213184.01

    # 2. Daily view reconciliation
    cur.execute("""
        SELECT 
            COUNT(*),
            SUM(transaction_count),
            ROUND(SUM(gross_transaction_value), 2),
            SUM(labelled_fraud_transaction_count),
            ROUND(SUM(gross_labelled_fraud_value), 2),
            ROUND(SUM(fraud_transfer_face_value), 2)
        FROM vw_daily_transaction_kpis;
    """)
    d_rows, d_cnt, d_gtv, d_fraud_cnt, d_fraud_val, d_trans_val = cur.fetchone()
    assert d_rows == 31, f"Expected 31 simulation days, got {d_rows}"
    assert d_cnt == src_cnt, "Daily transaction counts do not sum to source count"
    assert float(d_gtv) == float(src_gtv), "Daily GTV does not sum to source GTV"
    assert d_fraud_cnt == src_fraud_cnt, "Daily fraud count does not sum to source fraud count"
    assert float(d_fraud_val) == float(src_fraud_val), "Daily fraud value does not sum to source fraud value"
    assert float(d_trans_val) == float(src_trans_val), "Daily transfer fraud value does not sum to source"

    # 3. Hourly view reconciliation
    cur.execute("""
        SELECT 
            COUNT(*),
            SUM(transaction_count),
            ROUND(SUM(gross_transaction_value), 2),
            SUM(labelled_fraud_count),
            ROUND(SUM(gross_labelled_fraud_value), 2),
            ROUND(SUM(fraud_transfer_face_value), 2)
        FROM vw_hourly_risk_dynamics;
    """)
    h_rows, h_cnt, h_gtv, h_fraud_cnt, h_fraud_val, h_trans_val = cur.fetchone()
    assert h_rows == 743, f"Expected 743 simulation hours, got {h_rows}"
    assert h_cnt == src_cnt, "Hourly transaction counts do not sum to source count"
    assert float(h_gtv) == float(src_gtv), "Hourly GTV does not sum to source GTV"
    assert h_fraud_cnt == src_fraud_cnt, "Hourly fraud count does not sum to source fraud count"
    assert float(h_fraud_val) == float(src_fraud_val), "Hourly fraud value does not sum to source fraud value"
    assert float(h_trans_val) == float(src_trans_val), "Hourly transfer fraud value does not sum to source"

    # 4. Payment rail reconciliation
    cur.execute("""
        SELECT 
            COUNT(*),
            SUM(transaction_count),
            ROUND(SUM(gross_transaction_value), 2),
            SUM(labelled_fraud_count),
            ROUND(SUM(gross_labelled_fraud_value), 2),
            ROUND(SUM(rail_volume_share_pct), 2),
            ROUND(SUM(rail_value_share_pct), 2)
        FROM vw_payment_rail_performance;
    """)
    r_rows, r_cnt, r_gtv, r_fraud_cnt, r_fraud_val, r_vol_share, r_val_share = cur.fetchone()
    assert r_rows == 5, f"Expected 5 rails, got {r_rows}"
    assert r_cnt == src_cnt, "Payment rail counts do not sum to source count"
    assert float(r_gtv) == float(src_gtv), "Payment rail GTV does not sum to source GTV"
    assert r_fraud_cnt == src_fraud_cnt, "Payment rail fraud count does not sum to source fraud count"
    assert float(r_fraud_val) == float(src_fraud_val), "Payment rail fraud value does not sum to source"
    assert float(r_vol_share) == 100.00, f"Volume share sum is {r_vol_share}, expected 100.00"
    assert float(r_val_share) == 100.00, f"Value share sum is {r_val_share}, expected 100.00"

    # 5. Risk detection benchmark reconciliation
    cur.execute("""
        SELECT 
            evaluation_partition,
            system_name,
            true_positives,
            false_positives,
            false_negatives,
            true_negatives,
            total_evaluation_population,
            alert_count,
            precision,
            recall,
            f1_score
        FROM vw_risk_detection_benchmark;
    """)
    rows = cur.fetchall()
    assert len(rows) == 8, f"Expected 8 benchmark entries, got {len(rows)}"

    for r in rows:
        part, name, tp, fp, fn, tn, pop, alerts, prec, rec, f1 = r
        expected_pop = 196601 if part == "holdout" else 151302
        assert pop == expected_pop, f"Population mismatch for {name}: {pop} != {expected_pop}"
        assert tp + fp == alerts, f"Alert count mismatch for {name}: {tp+fp} != {alerts}"
        assert tp + fn == (2148 if part == "holdout" else 1008), f"Ground truth fraud count mismatch for {name}"
        assert tp + fp + fn + tn == pop, f"Confusion matrix does not sum to population for {name}"

    conn.close()
