"""
FinSight — Milestone 2 Dataset Audit & Schema Contract Tests.

Verifies the integrity of the PaySim dataset audit report, schema assertions,
class distributions, identifier topologies, and leakage controls.
"""

import json
from pathlib import Path
import pytest


@pytest.fixture(scope="module")
def audit_report():
    report_path = Path("reports/data_quality/dataset_profile.json")
    if not report_path.exists():
        # Fallback to sample if full run is missing in test environment
        report_path = Path("reports/data_quality/dataset_profile_sample.json")
    assert report_path.exists(), f"Profile report missing at {report_path}"
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def test_audit_metadata_integrity(audit_report):
    meta = audit_report["metadata"]
    assert meta["total_columns"] == 11
    assert meta["total_rows"] > 0
    assert "file_name" in meta
    assert meta["file_size_mb"] > 0


def test_schema_expected_columns(audit_report):
    expected_cols = [
        "step",
        "type",
        "amount",
        "nameOrig",
        "oldbalanceOrg",
        "newbalanceOrig",
        "nameDest",
        "oldbalanceDest",
        "newbalanceDest",
        "isFraud",
        "isFlaggedFraud",
    ]
    actual_cols = audit_report["schema_audit"]["columns"]
    assert actual_cols == expected_cols, f"Mismatch in expected columns: {actual_cols}"


def test_zero_missing_values(audit_report):
    missingness = audit_report["schema_audit"]["missingness"]
    for col, stats in missingness.items():
        assert stats["null_count"] == 0, f"Column {col} has unexpected nulls: {stats['null_count']}"
        assert stats["null_pct"] == 0.0


def test_duplicate_signatures_zero(audit_report):
    dup = audit_report["schema_audit"]["duplicate_signatures"]
    assert dup["duplicate_count"] == 0
    assert dup["duplicate_pct"] == 0.0


def test_target_class_imbalance(audit_report):
    target = audit_report["target_distribution"]
    assert target["target_column"] == "isFraud"
    counts = target["counts"]
    assert int(counts["0"]) > 0
    assert int(counts["1"]) > 0
    assert target["fraud_rate_pct"] < 1.0  # PaySim fraud rate is under 1%


def test_fraud_transaction_types(audit_report):
    fraud_types = audit_report["target_distribution"]["fraud_by_transaction_type"]
    allowed_fraud_types = {"CASH_OUT", "TRANSFER"}
    assert set(fraud_types.keys()).issubset(allowed_fraud_types), (
        f"Fraud occurred outside CASH_OUT and TRANSFER: {fraud_types}"
    )

    zero_fraud = audit_report["transaction_types"]["zero_fraud_types"]
    assert "PAYMENT" in zero_fraud
    assert "CASH_IN" in zero_fraud
    assert "DEBIT" in zero_fraud


def test_flagged_fraud_leakage_audit(audit_report):
    flagged = audit_report["flagged_fraud_audit"]
    ct = flagged["crosstab_with_target"]
    assert ct["isFraud_0_isFlagged_1"] == 0  # No false positives in simulator flag
    # isFlaggedFraud misses the vast majority of actual frauds
    if audit_report["metadata"]["total_rows"] > 100000:
        assert ct["isFraud_1_isFlagged_0"] > ct["isFraud_1_isFlagged_1"]


def test_merchant_destinations_zero_balance(audit_report):
    bal = audit_report["balance_integrity_audit"]
    if "merchant_balance_audit" in bal:
        mba = bal["merchant_balance_audit"]
        assert mba["merchant_zero_balance_pct"] == 100.0


def test_identifier_recurrence_topology(audit_report):
    if "identifier_recurrence" in audit_report:
        recur = audit_report["identifier_recurrence"]
        orig = recur["origin_accounts"]
        dest = recur["destination_accounts"]
        # In PaySim, origins are overwhelmingly single-use (>99%)
        assert orig["single_use_pct"] > 99.0
        # Destinations recur more frequently than origins
        assert dest["recurring_pct"] > orig["recurring_pct"]
