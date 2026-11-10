"""
FinSight — Milestone 1 Transaction Pipeline Automated Tests.

Tests chunked streaming, schema validation contracts, rejection tracking,
feature derivation, atomic output replacement, and fatal error handling
using self-contained temporary CSV fixtures (no external dataset dependency).
"""

import gzip
import json
from pathlib import Path
import pytest
import pandas as pd

from finsight.pipeline import (
    REQUIRED_COLUMNS,
    PipelineValidationError,
    run_pipeline,
    validate_chunk,
)


@pytest.fixture
def valid_row_dict():
    return {
        "step": "1",
        "type": "PAYMENT",
        "amount": "9839.64",
        "nameOrig": "C1231006815",
        "oldbalanceOrg": "170136.0",
        "newbalanceOrig": "160296.36",
        "nameDest": "M1979787155",
        "oldbalanceDest": "0.0",
        "newbalanceDest": "0.0",
        "isFraud": "0",
        "isFlaggedFraud": "0",
    }


def write_test_csv(path: Path, rows: list[dict]):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def test_valid_transaction(tmp_path, valid_row_dict):
    """Verify a valid transaction is accepted and features are properly derived."""
    input_csv = tmp_path / "valid.csv"
    output_gz = tmp_path / "out.csv.gz"
    report_json = tmp_path / "report.json"

    write_test_csv(input_csv, [valid_row_dict])

    report = run_pipeline(
        input_path=input_csv,
        output_path=output_gz,
        report_path=report_json,
        chunk_size=10,
    )

    assert report["row_accounting"]["total_rows_read"] == 1
    assert report["row_accounting"]["accepted_rows"] == 1
    assert report["row_accounting"]["rejected_rows"] == 0
    assert report["status"] == "SUCCESS"

    # Verify compressed output
    df_out = pd.read_csv(output_gz)
    assert len(df_out) == 1
    assert "transaction_hour" in df_out.columns
    assert "transaction_day" in df_out.columns
    assert df_out["transaction_hour"].iloc[0] == 0  # (1 - 1) % 24 == 0
    assert df_out["transaction_day"].iloc[0] == 1   # ((1 - 1) // 24) + 1 == 1
    assert df_out["isFraud"].iloc[0] == 0
    assert df_out["type"].iloc[0] == "PAYMENT"


def test_missing_required_fields(tmp_path, valid_row_dict):
    """Verify row with null / missing required values is quarantined."""
    input_csv = tmp_path / "missing.csv"
    output_gz = tmp_path / "out.csv.gz"
    report_json = tmp_path / "report.json"

    row_missing = valid_row_dict.copy()
    row_missing["amount"] = ""  # empty / null amount

    write_test_csv(input_csv, [row_missing])

    report = run_pipeline(
        input_path=input_csv,
        output_path=output_gz,
        report_path=report_json,
    )

    assert report["row_accounting"]["total_rows_read"] == 1
    assert report["row_accounting"]["accepted_rows"] == 0
    assert report["row_accounting"]["rejected_rows"] == 1
    assert "MISSING_REQUIRED_FIELDS" in report["rejection_reasons"]
    assert report["rejection_reasons"]["MISSING_REQUIRED_FIELDS"] == 1


def test_invalid_target_values(tmp_path, valid_row_dict):
    """Verify non-binary fraud target labels are rejected."""
    input_csv = tmp_path / "invalid_target.csv"
    output_gz = tmp_path / "out.csv.gz"
    report_json = tmp_path / "report.json"

    row_bad_fraud = valid_row_dict.copy()
    row_bad_fraud["isFraud"] = "2"

    write_test_csv(input_csv, [row_bad_fraud])

    report = run_pipeline(
        input_path=input_csv,
        output_path=output_gz,
        report_path=report_json,
    )

    assert report["row_accounting"]["accepted_rows"] == 0
    assert report["row_accounting"]["rejected_rows"] == 1
    assert "INVALID_FRAUD_TARGET" in report["rejection_reasons"]


def test_unsupported_transaction_types(tmp_path, valid_row_dict):
    """Verify transaction types outside the 5 PaySim rails are rejected."""
    input_csv = tmp_path / "invalid_type.csv"
    output_gz = tmp_path / "out.csv.gz"
    report_json = tmp_path / "report.json"

    row_bad_type = valid_row_dict.copy()
    row_bad_type["type"] = "CRYPTO_SWAP"

    write_test_csv(input_csv, [row_bad_type])

    report = run_pipeline(
        input_path=input_csv,
        output_path=output_gz,
        report_path=report_json,
    )

    assert report["row_accounting"]["accepted_rows"] == 0
    assert report["row_accounting"]["rejected_rows"] == 1
    assert "INVALID_TRANSACTION_TYPE" in report["rejection_reasons"]


def test_invalid_amount_and_balance(tmp_path, valid_row_dict):
    """Verify negative amounts and negative balances are rejected."""
    input_csv = tmp_path / "negative_vals.csv"
    output_gz = tmp_path / "out.csv.gz"
    report_json = tmp_path / "report.json"

    row_neg_amt = valid_row_dict.copy()
    row_neg_amt["amount"] = "-100.00"

    row_neg_bal = valid_row_dict.copy()
    row_neg_bal["oldbalanceOrg"] = "-50.00"

    write_test_csv(input_csv, [row_neg_amt, row_neg_bal])

    report = run_pipeline(
        input_path=input_csv,
        output_path=output_gz,
        report_path=report_json,
    )

    assert report["row_accounting"]["total_rows_read"] == 2
    assert report["row_accounting"]["accepted_rows"] == 0
    assert report["row_accounting"]["rejected_rows"] == 2
    assert "INVALID_AMOUNT" in report["rejection_reasons"]
    assert "INVALID_BALANCE" in report["rejection_reasons"]


def test_feature_derivation(tmp_path, valid_row_dict):
    """Verify transaction_hour and transaction_day calculations across steps."""
    input_csv = tmp_path / "steps.csv"
    output_gz = tmp_path / "out.csv.gz"
    report_json = tmp_path / "report.json"

    rows = []
    test_steps = [1, 12, 24, 25, 48, 744]
    expected_hours = [0, 11, 23, 0, 23, 23]
    expected_days = [1, 1, 1, 2, 2, 31]

    for s in test_steps:
        r = valid_row_dict.copy()
        r["step"] = str(s)
        rows.append(r)

    write_test_csv(input_csv, rows)

    run_pipeline(
        input_path=input_csv,
        output_path=output_gz,
        report_path=report_json,
    )

    df_out = pd.read_csv(output_gz)
    assert len(df_out) == len(test_steps)
    assert df_out["transaction_hour"].tolist() == expected_hours
    assert df_out["transaction_day"].tolist() == expected_days


def test_accurate_accounting_counts(tmp_path, valid_row_dict):
    """Verify consistent row accounting: read == accepted + rejected with exact reasons."""
    input_csv = tmp_path / "mixed.csv"
    output_gz = tmp_path / "out.csv.gz"
    report_json = tmp_path / "report.json"

    # 4 valid rows (2 non-fraud, 2 fraud)
    valid_1 = valid_row_dict.copy()
    valid_2 = valid_row_dict.copy()
    valid_3 = valid_row_dict.copy()
    valid_3["isFraud"] = "1"
    valid_3["type"] = "CASH_OUT"
    valid_4 = valid_row_dict.copy()
    valid_4["isFraud"] = "1"
    valid_4["type"] = "TRANSFER"

    # 3 invalid rows
    inv_type = valid_row_dict.copy()
    inv_type["type"] = "WIRE_ROUTER"

    inv_fraud = valid_row_dict.copy()
    inv_fraud["isFraud"] = "99"

    inv_missing = valid_row_dict.copy()
    inv_missing["nameOrig"] = ""  # null / missing

    inv_empty_id = valid_row_dict.copy()
    inv_empty_id["nameDest"] = "   "  # whitespace empty identifier

    all_rows = [valid_1, valid_2, valid_3, valid_4, inv_type, inv_fraud, inv_missing, inv_empty_id]
    write_test_csv(input_csv, all_rows)

    report = run_pipeline(
        input_path=input_csv,
        output_path=output_gz,
        report_path=report_json,
        chunk_size=3,  # Multi-chunk test
    )

    acc = report["row_accounting"]
    assert acc["total_rows_read"] == 8
    assert acc["accepted_rows"] == 4
    assert acc["rejected_rows"] == 4
    assert acc["total_rows_read"] == acc["accepted_rows"] + acc["rejected_rows"]

    reasons = report["rejection_reasons"]
    assert reasons.get("INVALID_TRANSACTION_TYPE") == 1
    assert reasons.get("INVALID_FRAUD_TARGET") == 1
    assert reasons.get("MISSING_REQUIRED_FIELDS") == 1
    assert reasons.get("EMPTY_IDENTIFIER") == 1

    assert report["target_distribution"]["counts"]["0"] == 2
    assert report["target_distribution"]["counts"]["1"] == 2


def test_fatal_handling_missing_file(tmp_path):
    """Verify non-existent input file raises FileNotFoundError."""
    missing_file = tmp_path / "does_not_exist.csv"
    with pytest.raises(FileNotFoundError):
        run_pipeline(input_path=missing_file)


def test_fatal_handling_invalid_schema(tmp_path, valid_row_dict):
    """Verify missing required columns causes PipelineValidationError."""
    bad_schema_csv = tmp_path / "bad_schema.csv"
    bad_row = valid_row_dict.copy()
    del bad_row["isFraud"]  # Missing required target column
    write_test_csv(bad_schema_csv, [bad_row])

    with pytest.raises(PipelineValidationError):
        run_pipeline(input_path=bad_schema_csv)


def test_failure_preserves_existing_output(tmp_path, valid_row_dict):
    """Verify pipeline failure does NOT corrupt or replace existing final output with partial data."""
    output_gz = tmp_path / "final_output.csv.gz"
    report_json = tmp_path / "report.json"

    # Pre-create a known good output file
    original_sentinel = b"SENTINEL_ORIGINAL_TRANSACTION_DATA"
    with gzip.open(output_gz, "wb") as f:
        f.write(original_sentinel)

    # Input that will fail on schema validation
    bad_csv = tmp_path / "failing.csv"
    bad_row = {"step": "1", "amount": "100.0"}  # Missing all other columns
    write_test_csv(bad_csv, [bad_row])

    with pytest.raises(PipelineValidationError):
        run_pipeline(
            input_path=bad_csv,
            output_path=output_gz,
            report_path=report_json,
        )

    # Verify original file was completely preserved
    assert output_gz.exists()
    with gzip.open(output_gz, "rb") as f:
        content = f.read()
    assert content == original_sentinel

    # Verify no temporary files remain
    tmp_files = list(tmp_path.glob("*.tmp*"))
    assert len(tmp_files) == 0
