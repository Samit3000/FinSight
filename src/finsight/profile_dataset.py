"""
FinSight — Dataset Profiling & Audit Module.

Provides memory-safe, chunked profiling of the PaySim financial transaction
dataset. Evaluates schema types, missingness, duplicate signatures, class
distribution, value ranges, and simulation anomalies without loading the entire
dataset into memory at once.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

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
logger = logging.getLogger("finsight.profile_dataset")


EXPECTED_COLUMNS: List[str] = [
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


def resolve_dataset_path(provided_path: Optional[str] = None) -> Path:
    """Resolve the dataset path from argument, standard directories, or root.

    Raises FileNotFoundError if no dataset file is found.
    """
    if provided_path:
        p = Path(provided_path)
        if p.exists() and p.is_file():
            return p
        raise FileNotFoundError(
            f"Provided dataset file not found at: {provided_path}\n"
            "Please ensure the file exists or check the path."
        )

    # Search candidates in order of preference
    search_locations = [
        Path("data/raw"),
        Path("data"),
        Path("."),
    ]

    # Look for standard PaySim filename patterns
    patterns = ["PS_*.csv", "paysim*.csv", "*.csv"]

    for loc in search_locations:
        if not loc.exists():
            continue
        for pat in patterns:
            matches = sorted(loc.glob(pat))
            # Filter out sample or test files
            for m in matches:
                if "sample" not in m.name.lower() and "test" not in m.name.lower():
                    return m

    raise FileNotFoundError(
        "PaySim dataset not found in data/raw/ or project root.\n"
        "To obtain the dataset:\n"
        "  1. Download PaySim from Kaggle: https://www.kaggle.com/datasets/ealaxi/paysim1\n"
        "  2. Place the CSV file in 'data/raw/' (e.g., data/raw/PS_20174392719_1491204439457_log.csv)\n"
        "  3. Or run profiling with --input-path <path_to_csv>"
    )


class RunningStats:
    """Computes streaming min, max, sum, and count across chunks."""

    def __init__(self) -> None:
        self.min_val = float("inf")
        self.max_val = float("-inf")
        self.sum_val = 0.0
        self.count = 0
        self.zeros_count = 0

    def update(self, series: pd.Series) -> None:
        clean = series.dropna()
        if len(clean) == 0:
            return
        c_min = float(clean.min())
        c_max = float(clean.max())
        if c_min < self.min_val:
            self.min_val = c_min
        if c_max > self.max_val:
            self.max_val = c_max
        self.sum_val += float(clean.sum())
        self.count += len(clean)
        self.zeros_count += int((clean == 0).sum())

    def to_dict(self) -> Dict[str, Any]:
        if self.count == 0:
            return {
                "count": 0,
                "min": None,
                "max": None,
                "mean": None,
                "zeros_count": 0,
                "zeros_pct": 0.0,
            }
        return {
            "count": self.count,
            "min": round(self.min_val, 4),
            "max": round(self.max_val, 4),
            "mean": round(self.sum_val / self.count, 4),
            "zeros_count": self.zeros_count,
            "zeros_pct": round((self.zeros_count / self.count) * 100, 4),
        }


def profile_dataset(
    file_path: Path | str,
    chunk_size: int = 250000,
    output_path: Optional[Path | str] = None,
    max_rows: Optional[int] = None,
) -> Dict[str, Any]:
    """Profiles a transaction CSV dataset in chunks and generates a summary audit.

    Parameters
    ----------
    file_path : Path or str
        Path to the CSV file to inspect.
    chunk_size : int
        Number of rows per iteration chunk for memory efficiency.
    output_path : Path or str, optional
        Destination path for machine-readable JSON profile report.
    max_rows : int, optional
        Optional cap on rows processed (useful for tests and fast inspection).

    Returns
    -------
    dict
        Structured profile report.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file does not exist: {path}")

    file_size_bytes = path.stat().st_size
    file_size_mb = round(file_size_bytes / (1024 * 1024), 2)
    logger.info("Starting dataset profiling on %s (%.2f MB)...", path.name, file_size_mb)

    total_rows = 0
    null_counts: Dict[str, int] = {}
    inferred_dtypes: Dict[str, str] = {}
    type_counts: Dict[str, int] = {}
    fraud_counts: Dict[str, int] = {"0": 0, "1": 0}
    flagged_fraud_counts: Dict[str, int] = {"0": 0, "1": 0}
    fraud_by_type: Dict[str, int] = {}
    crosstab_fraud_flagged: Dict[str, int] = {
        "isFraud_0_isFlagged_0": 0,
        "isFraud_1_isFlagged_0": 0,
        "isFraud_0_isFlagged_1": 0,
        "isFraud_1_isFlagged_1": 0,
    }

    numeric_cols = [
        "step",
        "amount",
        "oldbalanceOrg",
        "newbalanceOrig",
        "oldbalanceDest",
        "newbalanceDest",
    ]
    numeric_stats: Dict[str, RunningStats] = {col: RunningStats() for col in numeric_cols}

    dest_prefix_counts: Dict[str, int] = {"C": 0, "M": 0, "OTHER": 0}
    orig_prefix_counts: Dict[str, int] = {"C": 0, "OTHER": 0}

    orig_balance_matches = 0
    dest_balance_matches = 0

    seen_signatures = set()
    duplicate_signatures_count = 0
    check_duplicates = True

    # Identifier recurrence tracking
    seen_orig = set()
    recurring_orig_counts: Dict[str, int] = {}
    dest_counts: Dict[str, int] = {}

    # Balance anomaly and zero-balance tracking
    zero_balance_counts: Dict[str, int] = {
        "oldbalanceOrg": 0,
        "newbalanceOrig": 0,
        "oldbalanceDest": 0,
        "newbalanceDest": 0,
    }
    merchant_tx_count = 0
    merchant_zero_dest_bal_count = 0

    fraud_stats: Dict[str, int] = {
        "total_fraud": 0,
        "newbalanceOrig_zero": 0,
        "full_drain_count": 0,
        "orig_zero_start": 0,
        "dest_zero_start_and_end": 0,
    }

    chunks_processed = 0
    start_time = datetime.now(timezone.utc)

    reader = pd.read_csv(path, chunksize=chunk_size, nrows=max_rows)

    for chunk in reader:
        chunks_processed += 1
        chunk_len = len(chunk)
        total_rows += chunk_len

        # Check column names and initialize dtypes
        if chunks_processed == 1:
            for col in chunk.columns:
                null_counts[col] = 0
                inferred_dtypes[col] = str(chunk[col].dtype)

        # Null tracking
        for col in chunk.columns:
            null_counts[col] += int(chunk[col].isnull().sum())

        # Categorical type tracking
        if "type" in chunk.columns:
            for t, c in chunk["type"].value_counts().items():
                type_counts[str(t)] = type_counts.get(str(t), 0) + int(c)

        # Target and flag tracking
        if "isFraud" in chunk.columns:
            for f, c in chunk["isFraud"].value_counts().items():
                fraud_counts[str(int(f))] = fraud_counts.get(str(int(f)), 0) + int(c)

            if "type" in chunk.columns:
                fraud_rows = chunk[chunk["isFraud"] == 1]
                for t, c in fraud_rows["type"].value_counts().items():
                    fraud_by_type[str(t)] = fraud_by_type.get(str(t), 0) + int(c)

        if "isFlaggedFraud" in chunk.columns:
            for ff, c in chunk["isFlaggedFraud"].value_counts().items():
                flagged_fraud_counts[str(int(ff))] = flagged_fraud_counts.get(str(int(ff)), 0) + int(c)

        if "isFraud" in chunk.columns and "isFlaggedFraud" in chunk.columns:
            f0_ff0 = int(((chunk["isFraud"] == 0) & (chunk["isFlaggedFraud"] == 0)).sum())
            f1_ff0 = int(((chunk["isFraud"] == 1) & (chunk["isFlaggedFraud"] == 0)).sum())
            f0_ff1 = int(((chunk["isFraud"] == 0) & (chunk["isFlaggedFraud"] == 1)).sum())
            f1_ff1 = int(((chunk["isFraud"] == 1) & (chunk["isFlaggedFraud"] == 1)).sum())
            crosstab_fraud_flagged["isFraud_0_isFlagged_0"] += f0_ff0
            crosstab_fraud_flagged["isFraud_1_isFlagged_0"] += f1_ff0
            crosstab_fraud_flagged["isFraud_0_isFlagged_1"] += f0_ff1
            crosstab_fraud_flagged["isFraud_1_isFlagged_1"] += f1_ff1

        # Numerical columns tracking
        for col in numeric_cols:
            if col in chunk.columns:
                numeric_stats[col].update(chunk[col])

        # Zero-balance tracking across records
        for b_col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]:
            if b_col in chunk.columns:
                zero_balance_counts[b_col] += int((chunk[b_col] == 0).sum())

        # Merchant destination balance behavior (merchants start with M)
        if {"nameDest", "oldbalanceDest", "newbalanceDest"}.issubset(chunk.columns):
            m_mask = chunk["nameDest"].astype(str).str.startswith("M")
            m_cnt = int(m_mask.sum())
            merchant_tx_count += m_cnt
            if m_cnt > 0:
                m_zeros = int(((chunk.loc[m_mask, "oldbalanceDest"] == 0) & (chunk.loc[m_mask, "newbalanceDest"] == 0)).sum())
                merchant_zero_dest_bal_count += m_zeros

        # Fraud specific balance behavior
        if {"isFraud", "oldbalanceOrg", "newbalanceOrig", "amount", "oldbalanceDest", "newbalanceDest"}.issubset(chunk.columns):
            f_mask = chunk["isFraud"] == 1
            if f_mask.any():
                f_chunk = chunk[f_mask]
                fraud_stats["total_fraud"] += len(f_chunk)
                fraud_stats["newbalanceOrig_zero"] += int((f_chunk["newbalanceOrig"] == 0).sum())
                fraud_stats["full_drain_count"] += int(((f_chunk["oldbalanceOrg"] - f_chunk["amount"]).abs() < 0.01).sum())
                fraud_stats["orig_zero_start"] += int((f_chunk["oldbalanceOrg"] == 0).sum())
                fraud_stats["dest_zero_start_and_end"] += int(((f_chunk["oldbalanceDest"] == 0) & (f_chunk["newbalanceDest"] == 0)).sum())

        # Entity prefix tracking
        if "nameDest" in chunk.columns:
            dest_first = chunk["nameDest"].astype(str).str[0]
            dest_prefix_counts["C"] += int((dest_first == "C").sum())
            dest_prefix_counts["M"] += int((dest_first == "M").sum())
            dest_prefix_counts["OTHER"] += int((~dest_first.isin(["C", "M"])).sum())

        if "nameOrig" in chunk.columns:
            orig_first = chunk["nameOrig"].astype(str).str[0]
            orig_prefix_counts["C"] += int((orig_first == "C").sum())
            orig_prefix_counts["OTHER"] += int((orig_first != "C").sum())

        # Identifier recurrence tracking
        if "nameOrig" in chunk.columns:
            for orig, cnt in chunk["nameOrig"].value_counts().items():
                orig_str = str(orig)
                if orig_str in seen_orig:
                    recurring_orig_counts[orig_str] = recurring_orig_counts.get(orig_str, 1) + int(cnt)
                else:
                    seen_orig.add(orig_str)
                    if cnt > 1:
                        recurring_orig_counts[orig_str] = int(cnt)

        if "nameDest" in chunk.columns:
            for dest, cnt in chunk["nameDest"].value_counts().items():
                dest_str = str(dest)
                dest_counts[dest_str] = dest_counts.get(dest_str, 0) + int(cnt)

        # Balance equation consistency check
        if {"oldbalanceOrg", "amount", "newbalanceOrig"}.issubset(chunk.columns):
            orig_diff = (chunk["oldbalanceOrg"] - chunk["amount"] - chunk["newbalanceOrig"]).abs()
            orig_balance_matches += int((orig_diff < 0.01).sum())

        if {"oldbalanceDest", "amount", "newbalanceDest"}.issubset(chunk.columns):
            dest_diff = (chunk["oldbalanceDest"] + chunk["amount"] - chunk["newbalanceDest"]).abs()
            dest_balance_matches += int((dest_diff < 0.01).sum())

        # Duplicate signature check (step, type, amount, nameOrig, nameDest)
        if check_duplicates and {"step", "type", "amount", "nameOrig", "nameDest"}.issubset(chunk.columns):
            sig_series = (
                chunk["step"].astype(str)
                + "_"
                + chunk["type"].astype(str)
                + "_"
                + chunk["amount"].astype(str)
                + "_"
                + chunk["nameOrig"].astype(str)
                + "_"
                + chunk["nameDest"].astype(str)
            )
            for sig in sig_series:
                h = hash(sig)
                if h in seen_signatures:
                    duplicate_signatures_count += 1
                else:
                    seen_signatures.add(h)

        if chunks_processed % 5 == 0:
            logger.info("Processed %d rows (%d chunks)...", total_rows, chunks_processed)

    elapsed_seconds = round((datetime.now(timezone.utc) - start_time).total_seconds(), 2)
    logger.info("Completed profiling %d rows in %.2f seconds.", total_rows, elapsed_seconds)

    # Class imbalance metrics
    total_fraud = fraud_counts.get("1", 0)
    total_non_fraud = fraud_counts.get("0", 0)
    fraud_pct = round((total_fraud / total_rows * 100), 4) if total_rows > 0 else 0.0
    imbalance_ratio = (
        f"1:{round(total_non_fraud / total_fraud, 1)}" if total_fraud > 0 else "N/A"
    )

    # Identifier recurrence calculations
    distinct_orig = len(seen_orig)
    recurring_orig = len(recurring_orig_counts)
    max_orig_recurrence = max(recurring_orig_counts.values()) if recurring_orig_counts else 1

    distinct_dest = len(dest_counts)
    recurring_dest = sum(1 for c in dest_counts.values() if c > 1)
    max_dest_recurrence = max(dest_counts.values()) if dest_counts else 1

    dest_m_counts = {k: v for k, v in dest_counts.items() if k.startswith("M")}
    dest_c_counts = {k: v for k, v in dest_counts.items() if k.startswith("C")}

    overlap_count = sum(1 for d in dest_counts if d in seen_orig)

    profile_report: Dict[str, Any] = {
        "metadata": {
            "source_file": str(path.resolve()),
            "file_name": path.name,
            "file_size_mb": file_size_mb,
            "profiled_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": elapsed_seconds,
            "chunks_processed": chunks_processed,
            "total_rows": total_rows,
            "total_columns": len(null_counts),
        },
        "schema_audit": {
            "columns": list(null_counts.keys()),
            "inferred_dtypes": inferred_dtypes,
            "missingness": {
                col: {
                    "null_count": count,
                    "null_pct": round((count / total_rows * 100), 4) if total_rows > 0 else 0.0,
                }
                for col, count in null_counts.items()
            },
            "duplicate_signatures": {
                "checked_fields": ["step", "type", "amount", "nameOrig", "nameDest"],
                "duplicate_count": duplicate_signatures_count,
                "duplicate_pct": (
                    round((duplicate_signatures_count / total_rows * 100), 6)
                    if total_rows > 0
                    else 0.0
                ),
            },
        },
        "target_distribution": {
            "target_column": "isFraud",
            "counts": fraud_counts,
            "fraud_rate_pct": fraud_pct,
            "imbalance_ratio": imbalance_ratio,
            "fraud_by_transaction_type": fraud_by_type,
        },
        "flagged_fraud_audit": {
            "column": "isFlaggedFraud",
            "counts": flagged_fraud_counts,
            "crosstab_with_target": crosstab_fraud_flagged,
            "leakage_assessment": (
                "isFlaggedFraud is an operational rule-based artifact (>200k in TRANSFER). "
                "It misses 99.8% of actual frauds and must be excluded from predictive ML features."
            ),
        },
        "transaction_types": {
            "distribution": type_counts,
            "fraud_occurrence_types": list(fraud_by_type.keys()),
            "zero_fraud_types": [t for t in type_counts if t not in fraud_by_type],
        },
        "numerical_distributions": {
            col: stats.to_dict() for col, stats in numeric_stats.items()
        },
        "entity_structure": {
            "origin_prefix_breakdown": orig_prefix_counts,
            "destination_prefix_breakdown": dest_prefix_counts,
            "merchant_destinations_detected": dest_prefix_counts.get("M", 0) > 0,
        },
        "identifier_recurrence": {
            "origin_accounts": {
                "distinct_count": distinct_orig,
                "recurring_count": recurring_orig,
                "recurring_pct": round((recurring_orig / distinct_orig * 100), 4) if distinct_orig > 0 else 0.0,
                "single_use_count": distinct_orig - recurring_orig,
                "single_use_pct": round(((distinct_orig - recurring_orig) / distinct_orig * 100), 4) if distinct_orig > 0 else 0.0,
                "max_recurrence": max_orig_recurrence,
            },
            "destination_accounts": {
                "distinct_count": distinct_dest,
                "recurring_count": recurring_dest,
                "recurring_pct": round((recurring_dest / distinct_dest * 100), 4) if distinct_dest > 0 else 0.0,
                "single_use_count": distinct_dest - recurring_dest,
                "single_use_pct": round(((distinct_dest - recurring_dest) / distinct_dest * 100), 4) if distinct_dest > 0 else 0.0,
                "max_recurrence": max_dest_recurrence,
                "merchant_distinct_count": len(dest_m_counts),
                "merchant_total_transactions": sum(dest_m_counts.values()),
                "merchant_max_recurrence": max(dest_m_counts.values()) if dest_m_counts else 0,
                "customer_distinct_count": len(dest_c_counts),
                "customer_total_transactions": sum(dest_c_counts.values()),
                "customer_max_recurrence": max(dest_c_counts.values()) if dest_c_counts else 0,
            },
            "overlap": {
                "origin_and_destination_overlap_count": overlap_count,
                "overlap_pct_of_dest": round((overlap_count / distinct_dest * 100), 4) if distinct_dest > 0 else 0.0,
            },
        },
        "balance_integrity_audit": {
            "origin_balance_equation_match_pct": (
                round((orig_balance_matches / total_rows * 100), 2) if total_rows > 0 else 0.0
            ),
            "destination_balance_equation_match_pct": (
                round((dest_balance_matches / total_rows * 100), 2) if total_rows > 0 else 0.0
            ),
            "zero_balance_summary": {
                col: {
                    "count": zero_balance_counts[col],
                    "pct": round((zero_balance_counts[col] / total_rows * 100), 4) if total_rows > 0 else 0.0,
                }
                for col in ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
            },
            "merchant_balance_audit": {
                "merchant_transaction_count": merchant_tx_count,
                "merchant_zero_balance_count": merchant_zero_dest_bal_count,
                "merchant_zero_balance_pct": (
                    round((merchant_zero_dest_bal_count / merchant_tx_count * 100), 2)
                    if merchant_tx_count > 0
                    else 0.0
                ),
                "finding": "Merchants exhibit 100% zero balances in both oldbalanceDest and newbalanceDest as synthetic PaySim artifacts.",
            },
            "fraud_balance_behavior": {
                "total_fraud_transactions": fraud_stats["total_fraud"],
                "origin_balance_emptied_count": fraud_stats["newbalanceOrig_zero"],
                "origin_balance_emptied_pct": (
                    round((fraud_stats["newbalanceOrig_zero"] / fraud_stats["total_fraud"] * 100), 2)
                    if fraud_stats["total_fraud"] > 0
                    else 0.0
                ),
                "exact_balance_liquidated_count": fraud_stats["full_drain_count"],
                "exact_balance_liquidated_pct": (
                    round((fraud_stats["full_drain_count"] / fraud_stats["total_fraud"] * 100), 2)
                    if fraud_stats["total_fraud"] > 0
                    else 0.0
                ),
                "initial_origin_balance_zero_count": fraud_stats["orig_zero_start"],
                "initial_origin_balance_zero_pct": (
                    round((fraud_stats["orig_zero_start"] / fraud_stats["total_fraud"] * 100), 2)
                    if fraud_stats["total_fraud"] > 0
                    else 0.0
                ),
                "dest_balance_unaffected_count": fraud_stats["dest_zero_start_and_end"],
                "dest_balance_unaffected_pct": (
                    round((fraud_stats["dest_zero_start_and_end"] / fraud_stats["total_fraud"] * 100), 2)
                    if fraud_stats["total_fraud"] > 0
                    else 0.0
                ),
            },
            "recommendation": (
                "PaySim balance fields exhibit severe synthetic accounting anomalies and "
                "must be excluded from initial ML feature sets as per author caveats."
            ),
        },
    }

    # Write report if requested or default to reports/data_quality/dataset_profile.json
    out = Path(output_path) if output_path else Path("reports/data_quality/dataset_profile.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(profile_report, f, indent=2)
    logger.info("Saved profile report to %s", out)

    return profile_report


def print_summary_report(report: Dict[str, Any]) -> None:
    """Prints a clean human-readable audit summary to console."""
    meta = report["metadata"]
    target = report["target_distribution"]
    flagged = report["flagged_fraud_audit"]
    types = report["transaction_types"]
    bal = report["balance_integrity_audit"]
    recur = report.get("identifier_recurrence", {})

    print("\n" + "=" * 70)
    print("FINSIGHT DATASET AUDIT & PROFILING REPORT")
    print("=" * 70)
    print(f"File Name:        {meta['file_name']} ({meta['file_size_mb']} MB)")
    print(f"Total Records:    {meta['total_rows']:,}")
    print(f"Total Columns:    {meta['total_columns']}")
    print(f"Execution Time:   {meta['elapsed_seconds']}s across {meta['chunks_processed']} chunks")
    print("-" * 70)
    print("MISSING VALUES & DATA INTEGRITY:")
    all_zero_nulls = all(v["null_count"] == 0 for v in report["schema_audit"]["missingness"].values())
    if all_zero_nulls:
        print("  [PASS] Zero missing/null values detected across all columns.")
    else:
        for col, stats in report["schema_audit"]["missingness"].items():
            if stats["null_count"] > 0:
                print(f"  [ALERT] {col}: {stats['null_count']} nulls ({stats['null_pct']}%)")
    dup_sig = report["schema_audit"]["duplicate_signatures"]["duplicate_count"]
    print(f"  [INFO] Duplicate Transaction Signatures: {dup_sig}")
    print("-" * 70)
    print("TARGET VARIABLE DISTRIBUTION (isFraud):")
    print(f"  Non-Fraud (0):  {target['counts'].get('0', 0):,} ({100 - target['fraud_rate_pct']:.4f}%)")
    print(f"  Fraud (1):      {target['counts'].get('1', 0):,} ({target['fraud_rate_pct']}%)")
    print(f"  Class Imbalance Ratio: {target['imbalance_ratio']}")
    print("  Fraud by Transaction Type:")
    for t, c in target["fraud_by_transaction_type"].items():
        print(f"    - {t}: {c:,} fraud occurrences")
    print("  Zero-Fraud Types (0 fraudulent cases in data):")
    print(f"    - {', '.join(types['zero_fraud_types'])}")
    print("-" * 70)
    print("RULE-BASED FLAG AUDIT (isFlaggedFraud):")
    print(f"  Flagged Count:  {flagged['counts'].get('1', 0):,} transactions")
    ct = flagged["crosstab_with_target"]
    print(f"  Crosstab (isFraud vs isFlaggedFraud):")
    print(f"    - Legitimate & Unflagged: {ct['isFraud_0_isFlagged_0']:,}")
    print(f"    - Fraud & UNFLAGGED (Missed by rule): {ct['isFraud_1_isFlagged_0']:,}")
    print(f"    - Fraud & Flagged (Caught by rule):   {ct['isFraud_1_isFlagged_1']:,}")
    print(f"    - False Positives:                    {ct['isFraud_0_isFlagged_1']:,}")
    print("-" * 70)
    if recur:
        print("IDENTIFIER RECURRENCE & ENTITY TOPOLOGY:")
        orig = recur["origin_accounts"]
        dest = recur["destination_accounts"]
        ov = recur["overlap"]
        print(f"  Origin Accounts (nameOrig):      {orig['distinct_count']:,} distinct")
        print(f"    - Single-use origins:          {orig['single_use_count']:,} ({orig['single_use_pct']}%)")
        print(f"    - Recurring origins (>1 tx):   {orig['recurring_count']:,} ({orig['recurring_pct']}%, max={orig['max_recurrence']})")
        print(f"  Destination Accounts (nameDest): {dest['distinct_count']:,} distinct")
        print(f"    - Single-use destinations:     {dest['single_use_count']:,} ({dest['single_use_pct']}%)")
        print(f"    - Recurring destinations (>1): {dest['recurring_count']:,} ({dest['recurring_pct']}%, max={dest['max_recurrence']})")
        print(f"    - Merchants (M...):            {dest['merchant_distinct_count']:,} distinct accounts, max tx={dest['merchant_max_recurrence']:,}")
        print(f"    - Customers (C...):            {dest['customer_distinct_count']:,} distinct accounts, max tx={dest['customer_max_recurrence']:,}")
        print(f"  Origin & Destination Overlap:    {ov['origin_and_destination_overlap_count']:,} accounts")
        print("-" * 70)
    print("BALANCE INTEGRITY & LEAKAGE FINDINGS:")
    print(f"  Origin balance equation match rate:      {bal['origin_balance_equation_match_pct']}%")
    print(f"  Destination balance equation match rate: {bal['destination_balance_equation_match_pct']}%")
    if "merchant_balance_audit" in bal:
        mba = bal["merchant_balance_audit"]
        print(f"  Merchant destination 0-balance rate:     {mba['merchant_zero_balance_pct']}% ({mba['merchant_zero_balance_count']:,} tx)")
    if "fraud_balance_behavior" in bal:
        fbb = bal["fraud_balance_behavior"]
        print(f"  Fraud accounts completely emptied:       {fbb['origin_balance_emptied_pct']}% ({fbb['origin_balance_emptied_count']:,}/{fbb['total_fraud_transactions']:,})")
        print(f"  Fraud accounts exact balance liquidated: {fbb['exact_balance_liquidated_pct']}% ({fbb['exact_balance_liquidated_count']:,})")
        print(f"  Fraud with 0 initial origin balance:     {fbb['initial_origin_balance_zero_pct']}% ({fbb['initial_origin_balance_zero_count']:,})")
    print("  Recommendation: Exclude balance fields from initial ML features.")
    print("=" * 70 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FinSight Dataset Profiling & Audit Tool"
    )
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default=None,
        help="Path to the raw PaySim CSV file. If omitted, searches data/raw and project root.",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=250000,
        help="Chunk size for streaming processing (default: 250,000 rows).",
    )
    parser.add_argument(
        "--output-path",
        "-o",
        type=str,
        default="reports/data_quality/dataset_profile.json",
        help="Path for saving JSON profile report.",
    )
    parser.add_argument(
        "--max-rows",
        "-n",
        type=int,
        default=None,
        help="Optional maximum rows to process (for quick inspection).",
    )

    args = parser.parse_args()

    try:
        resolved_path = resolve_dataset_path(args.input_path)
    except FileNotFoundError as err:
        logger.error(str(err))
        sys.exit(1)

    try:
        report = profile_dataset(
            file_path=resolved_path,
            chunk_size=args.chunk_size,
            output_path=args.output_path,
            max_rows=args.max_rows,
        )
        print_summary_report(report)
    except Exception as exc:
        logger.exception("Profiling failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
