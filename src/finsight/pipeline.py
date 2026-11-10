"""
FinSight — Transaction Ingestion & Feature Engineering Pipeline.

Milestone 1: Stream-processes raw PaySim transaction logs in memory-safe chunks,
validates transaction records against business schemas, derives temporal features,
and produces a validated, compressed output dataset and data quality run report.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure package root (src) is on sys.path when executed directly
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

try:
    from finsight.profile_dataset import resolve_dataset_path
except ImportError:
    from profile_dataset import resolve_dataset_path  # type: ignore

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("finsight.pipeline")

# Canonical PaySim columns expected in raw extract
REQUIRED_COLUMNS: List[str] = [
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

# Supported PaySim transaction rails
ALLOWED_TRANSACTION_TYPES: frozenset[str] = frozenset(
    ["CASH_OUT", "TRANSFER", "PAYMENT", "CASH_IN", "DEBIT"]
)

# Canonical feature metadata documenting ML boundary vs audit fields
FEATURE_METADATA: Dict[str, Any] = {
    "target_variable": "isFraud",
    "target_interpretation": "Ground-truth binary label (1=fraud, 0=legitimate). Never used as predictor.",
    "temporal_derivation": {
        "transaction_hour": "(step - 1) % 24 (0-23 hour of day; step 1 represents simulation hour 0)",
        "transaction_day": "((step - 1) // 24) + 1 (1-indexed simulation calendar day)",
    },
    "initial_ml_feature_exclusions": [
        "isFraud",  # Target variable
        "isFlaggedFraud",  # Operational rule artifact (>200k Transfer); misses 99.8% of actual fraud
        "oldbalanceOrg",  # Synthetic accounting anomaly / leakage risk (33% zero-balance)
        "newbalanceOrig",  # Synthetic accounting anomaly / leakage risk (98% liquidated to zero in fraud)
        "oldbalanceDest",  # Synthetic accounting anomaly (100% zero for merchant destinations)
        "newbalanceDest",  # Synthetic accounting anomaly (100% zero for merchant destinations)
        "nameOrig",  # High-cardinality entity ID (99.85% single-use sparsity)
        "nameDest",  # High-cardinality entity ID
    ],
    "candidate_ml_features": [
        "step",
        "type",
        "amount",
        "transaction_hour",
        "transaction_day",
    ],
    "preserved_audit_and_analytics_columns": [
        "nameOrig",
        "nameDest",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
        "isFlaggedFraud",
    ],
    "deduplication_policy": (
        "No records are silently discarded using assumed composite keys. "
        "Duplicate signatures are audited and tracked in data quality logs."
    ),
}


class PipelineValidationError(Exception):
    """Raised when critical schema preconditions or file dependencies fail."""


def validate_chunk(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int]]:
    """Validate a chunk of transactions against schema and business contracts.

    Returns:
        Tuple containing:
        - accepted_df: DataFrame of validated rows with derived features.
        - rejected_df: DataFrame of invalid rows with rejection reasons.
        - reason_counts: Dict mapping rejection reason strings to row counts.
    """
    missing_cols = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing_cols:
        raise PipelineValidationError(
            f"Input chunk missing required columns: {sorted(missing_cols)}"
        )

    n_rows = len(df)
    if n_rows == 0:
        return pd.DataFrame(), pd.DataFrame(), {}

    # 1. Null / Missing field validation
    null_mask = df[REQUIRED_COLUMNS].isnull().any(axis=1)

    # 2. Step validation (numeric, >= 1, integer)
    step_num = pd.to_numeric(df["step"], errors="coerce")
    invalid_step_mask = step_num.isnull() | (step_num < 1) | (step_num % 1 != 0)

    # 3. Transaction type validation
    type_str = df["type"].astype(str)
    invalid_type_mask = ~type_str.isin(ALLOWED_TRANSACTION_TYPES)

    # 4. Amount validation (numeric, non-negative)
    amount_num = pd.to_numeric(df["amount"], errors="coerce")
    invalid_amount_mask = amount_num.isnull() | (amount_num < 0.0)

    # 5. Target variable validation (isFraud must be exactly 0 or 1)
    fraud_num = pd.to_numeric(df["isFraud"], errors="coerce")
    invalid_fraud_mask = fraud_num.isnull() | (~fraud_num.isin([0, 1]))

    # 6. Flagged fraud validation (isFlaggedFraud must be exactly 0 or 1)
    flagged_num = pd.to_numeric(df["isFlaggedFraud"], errors="coerce")
    invalid_flagged_mask = flagged_num.isnull() | (~flagged_num.isin([0, 1]))

    # 7. Balance validation (numeric, non-negative)
    old_orig = pd.to_numeric(df["oldbalanceOrg"], errors="coerce")
    new_orig = pd.to_numeric(df["newbalanceOrig"], errors="coerce")
    old_dest = pd.to_numeric(df["oldbalanceDest"], errors="coerce")
    new_dest = pd.to_numeric(df["newbalanceDest"], errors="coerce")
    invalid_bal_mask = (
        old_orig.isnull()
        | (old_orig < 0.0)
        | new_orig.isnull()
        | (new_orig < 0.0)
        | old_dest.isnull()
        | (old_dest < 0.0)
        | new_dest.isnull()
        | (new_dest < 0.0)
    )

    # 8. Identifier string validation (non-empty strings)
    name_orig_str = df["nameOrig"].astype(str).str.strip()
    name_dest_str = df["nameDest"].astype(str).str.strip()
    empty_id_mask = (name_orig_str == "") | (name_dest_str == "")

    # Assign primary rejection reasons in deterministic priority order
    rejection_reason = pd.Series("", index=df.index, dtype=object)
    rejection_reason[null_mask] = "MISSING_REQUIRED_FIELDS"
    rejection_reason[(rejection_reason == "") & empty_id_mask] = "EMPTY_IDENTIFIER"
    rejection_reason[(rejection_reason == "") & invalid_type_mask] = "INVALID_TRANSACTION_TYPE"
    rejection_reason[(rejection_reason == "") & invalid_step_mask] = "INVALID_STEP"
    rejection_reason[(rejection_reason == "") & invalid_amount_mask] = "INVALID_AMOUNT"
    rejection_reason[(rejection_reason == "") & invalid_fraud_mask] = "INVALID_FRAUD_TARGET"
    rejection_reason[(rejection_reason == "") & invalid_flagged_mask] = "INVALID_FLAGGED_FRAUD"
    rejection_reason[(rejection_reason == "") & invalid_bal_mask] = "INVALID_BALANCE"

    is_valid = rejection_reason == ""
    accepted_df = df[is_valid].copy()
    rejected_df = df[~is_valid].copy()

    reason_counts: Dict[str, int] = {}
    if not rejected_df.empty:
        rejected_df["rejection_reason"] = rejection_reason[~is_valid]
        reason_counts = rejected_df["rejection_reason"].value_counts().to_dict()

    if not accepted_df.empty:
        # Cast clean types
        accepted_df["step"] = step_num[is_valid].astype(int)
        accepted_df["type"] = type_str[is_valid].astype(str)
        accepted_df["amount"] = amount_num[is_valid].astype(float)
        accepted_df["nameOrig"] = name_orig_str[is_valid]
        accepted_df["oldbalanceOrg"] = old_orig[is_valid].astype(float)
        accepted_df["newbalanceOrig"] = new_orig[is_valid].astype(float)
        accepted_df["nameDest"] = name_dest_str[is_valid]
        accepted_df["oldbalanceDest"] = old_dest[is_valid].astype(float)
        accepted_df["newbalanceDest"] = new_dest[is_valid].astype(float)
        accepted_df["isFraud"] = fraud_num[is_valid].astype(int)
        accepted_df["isFlaggedFraud"] = flagged_num[is_valid].astype(int)

        # Feature derivations:
        # Step maps 1 step = 1 simulated hour (1..744).
        # (step - 1) % 24 derives the 0-indexed hour of day (0 to 23).
        accepted_df["transaction_hour"] = (accepted_df["step"] - 1) % 24
        # 1-indexed simulation calendar day (1 to 31).
        accepted_df["transaction_day"] = ((accepted_df["step"] - 1) // 24) + 1

    return accepted_df, rejected_df, reason_counts


def run_pipeline(
    input_path: Optional[str | Path] = None,
    output_path: str | Path = "data/processed/transactions.csv.gz",
    report_path: str | Path = "reports/data_quality/pipeline_run.json",
    chunk_size: int = 250000,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the end-to-end chunked transaction ingestion pipeline.

    Args:
        input_path: Path to raw PaySim CSV. If None, resolves via search candidates.
        output_path: Target path for the compressed CSV output (.csv.gz).
        report_path: Target path for the JSON execution report.
        chunk_size: Number of records to process per chunk.
        max_rows: Optional upper limit on rows to process (for testing/samples).

    Returns:
        Dict containing run metadata, accounting statistics, and performance metrics.
    """
    start_time = time.perf_counter()
    resolved_input = resolve_dataset_path(str(input_path) if input_path else None)
    out_file = Path(output_path)
    rep_file = Path(report_path)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    rep_file.parent.mkdir(parents=True, exist_ok=True)

    # Safe atomic replacement staging path
    temp_out_file = out_file.parent / f"{out_file.name}.tmp.{os.getpid()}"

    logger.info("Initializing FinSight pipeline run...")
    logger.info("Input source:   %s", resolved_input)
    logger.info("Output target:  %s", out_file)
    logger.info("Report target:  %s", rep_file)
    logger.info("Chunk size:     %d rows", chunk_size)
    if max_rows:
        logger.info("Max rows limit: %d rows", max_rows)

    total_rows_read = 0
    total_accepted = 0
    total_rejected = 0
    chunks_processed = 0
    rejection_reasons: Dict[str, int] = {}
    fraud_distribution: Dict[str, int] = {"0": 0, "1": 0}
    duplicate_signatures = 0
    is_first_write = True

    try:
        # Open gzip stream for streaming writes
        with gzip.open(temp_out_file, mode="wt", encoding="utf-8", newline="") as gz_out:
            reader = pd.read_csv(
                resolved_input,
                chunksize=chunk_size,
                nrows=max_rows,
                dtype=str,  # Read raw as strings to prevent premature unhandled casting errors
            )

            for chunk_df in reader:
                chunks_processed += 1
                chunk_len = len(chunk_df)
                total_rows_read += chunk_len

                # Enforce schema contracts on chunk before any column indexing
                missing_cols = set(REQUIRED_COLUMNS) - set(chunk_df.columns)
                if missing_cols:
                    raise PipelineValidationError(
                        f"Input schema missing required columns: {sorted(missing_cols)}"
                    )

                # Audit exact duplicate transaction signatures within chunk
                chunk_dups = chunk_df.duplicated(
                    subset=["step", "type", "amount", "nameOrig", "nameDest"]
                ).sum()
                duplicate_signatures += int(chunk_dups)

                # Validate and derive features
                accepted_df, rejected_df, reasons = validate_chunk(chunk_df)
                total_accepted += len(accepted_df)
                total_rejected += len(rejected_df)

                for r, cnt in reasons.items():
                    rejection_reasons[r] = rejection_reasons.get(r, 0) + cnt

                if not accepted_df.empty:
                    # Update fraud counts
                    f_counts = accepted_df["isFraud"].value_counts().to_dict()
                    fraud_distribution["0"] += int(f_counts.get(0, 0))
                    fraud_distribution["1"] += int(f_counts.get(1, 0))

                    # Stream write to gzip CSV
                    accepted_df.to_csv(
                        gz_out,
                        index=False,
                        header=is_first_write,
                    )
                    is_first_write = False

                elapsed = time.perf_counter() - start_time
                logger.info(
                    "Chunk %d processed: read %d rows (cumulative: %d read, %d accepted, %d rejected) [%.2fs elapsed]",
                    chunks_processed,
                    chunk_len,
                    total_rows_read,
                    total_accepted,
                    total_rejected,
                    elapsed,
                )

        # Atomic replacement: Replace output file only after entire pipeline finishes successfully
        os.replace(temp_out_file, out_file)
        logger.info("Successfully replaced final output file at: %s", out_file)

    except Exception as exc:
        logger.exception("Pipeline failed during execution: %s", exc)
        if temp_out_file.exists():
            try:
                temp_out_file.unlink()
                logger.info("Cleaned up orphaned temporary file: %s", temp_out_file)
            except Exception:
                pass
        raise

    elapsed_seconds = round(time.perf_counter() - start_time, 2)
    output_size_mb = round(out_file.stat().st_size / (1024 * 1024), 2)

    # Consistency assertion: Total rows read must equal accepted + rejected
    if total_rows_read != (total_accepted + total_rejected):
        raise AssertionError(
            f"Row accounting mismatch: read {total_rows_read} != accepted {total_accepted} + rejected {total_rejected}"
        )

    run_report: Dict[str, Any] = {
        "pipeline_name": "finsight_transaction_pipeline",
        "version": "1.0.0",
        "status": "SUCCESS",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed_seconds,
        "source_file": str(resolved_input.resolve()),
        "output_file": str(out_file.resolve()),
        "output_file_size_mb": output_size_mb,
        "chunks_processed": chunks_processed,
        "chunk_size": chunk_size,
        "row_accounting": {
            "total_rows_read": total_rows_read,
            "accepted_rows": total_accepted,
            "rejected_rows": total_rejected,
            "rejection_rate_pct": round(
                (total_rejected / total_rows_read * 100) if total_rows_read > 0 else 0.0, 4
            ),
        },
        "rejection_reasons": rejection_reasons,
        "duplicate_signatures_detected": duplicate_signatures,
        "target_distribution": {
            "target_column": "isFraud",
            "counts": fraud_distribution,
            "fraud_rate_pct": round(
                (fraud_distribution["1"] / total_accepted * 100) if total_accepted > 0 else 0.0, 4
            ),
        },
        "feature_metadata": FEATURE_METADATA,
    }

    # Write pipeline run report
    with open(rep_file, "w", encoding="utf-8") as rf:
        json.dump(run_report, rf, indent=2)

    logger.info("Pipeline run report saved to: %s", rep_file)
    logger.info("Pipeline finished: %d rows read -> %d accepted, %d rejected in %.2fs",
                total_rows_read, total_accepted, total_rejected, elapsed_seconds)

    return run_report


def main() -> None:
    """CLI entry point for FinSight transaction pipeline."""
    parser = argparse.ArgumentParser(
        description="FinSight Transaction Ingestion & Feature Engineering Pipeline"
    )
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default=None,
        help="Path to raw PaySim CSV file. If omitted, auto-resolves data/raw/ or project root.",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="data/processed/transactions.csv.gz",
        help="Target compressed CSV path (default: data/processed/transactions.csv.gz).",
    )
    parser.add_argument(
        "--report-path",
        "-r",
        type=str,
        default="reports/data_quality/pipeline_run.json",
        help="Target JSON run report path (default: reports/data_quality/pipeline_run.json).",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=250000,
        help="Chunk size for streaming processing (default: 250,000 rows).",
    )
    parser.add_argument(
        "--max-rows",
        "-n",
        type=int,
        default=None,
        help="Optional row limit to process (for testing/quick inspection).",
    )

    args = parser.parse_args()

    try:
        run_pipeline(
            input_path=args.input_path,
            output_path=args.output_path,
            report_path=args.report_path,
            chunk_size=args.chunk_size,
            max_rows=args.max_rows,
        )
    except FileNotFoundError as err:
        logger.error("Dataset not found: %s", err)
        sys.exit(1)
    except PipelineValidationError as err:
        logger.error("Schema contract validation failed: %s", err)
        sys.exit(2)
    except Exception as exc:
        logger.exception("Pipeline fatal error: %s", exc)
        sys.exit(3)


if __name__ == "__main__":
    main()
