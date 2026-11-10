"""
FinSight — PostgreSQL Bulk Loader Module.

Milestone 2: Memory-safe chunked loader for transferring validated transaction
facts from compressed CSV into PostgreSQL 15+ using streaming COPY.

Includes connection management, safe rerun guards against duplication,
explicit transaction control, schema validation, and offline diagnostic checks.
"""

from __future__ import annotations

import argparse
import io
import logging
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# Ensure package root (src) is on sys.path
_src_dir = Path(__file__).resolve().parent.parent
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

# Automatically load environment variables from .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Optional psycopg2 import for environments with/without driver
try:
    import psycopg2
    from psycopg2 import sql
    PSYCOPG2_AVAILABLE = True
except ImportError:
    psycopg2 = None
    sql = None
    PSYCOPG2_AVAILABLE = False

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
logger = logging.getLogger("finsight.loader")

# Expected columns in data/processed/transactions.csv.gz
EXPECTED_CSV_COLUMNS: List[str] = [
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
    "transaction_hour",
    "transaction_day",
]

# Mapping from CSV headers to PostgreSQL table columns
CSV_TO_DB_COLUMNS: Dict[str, str] = {
    "step": "step",
    "type": "type",
    "amount": "amount",
    "nameOrig": "name_orig",
    "oldbalanceOrg": "oldbalance_org",
    "newbalanceOrig": "newbalance_orig",
    "nameDest": "name_dest",
    "oldbalanceDest": "oldbalance_dest",
    "newbalanceDest": "newbalance_dest",
    "isFraud": "is_fraud",
    "isFlaggedFraud": "is_flagged_fraud",
    "transaction_hour": "transaction_hour",
    "transaction_day": "transaction_day",
}

DEFAULT_TABLE_NAME = "fct_transactions"


@dataclass
class DatabaseConfig:
    """Database connection parameters extracted from environment or CLI."""
    host: str = "localhost"
    port: int = 5432
    name: str = "finsight_db"
    user: str = "postgres"
    password: str = ""
    schema: str = "public"

    @classmethod
    def from_env(cls) -> DatabaseConfig:
        """Construct config from standard environment variables."""
        return cls(
            host=os.getenv("DB_HOST", "localhost"),
            port=int(os.getenv("DB_PORT", "5432")),
            name=os.getenv("DB_NAME", "finsight_db"),
            user=os.getenv("DB_USER", os.getenv("POSTGRES_USER", "postgres")),
            password=os.getenv("DB_PASSWORD", os.getenv("POSTGRES_PASSWORD", "")),
            schema=os.getenv("DB_SCHEMA", "public"),
        )

    def sanitized_connection_string(self) -> str:
        """Return connection target without leaking password."""
        return f"postgresql://{self.user}:****@{self.host}:{self.port}/{self.name}?schema={self.schema}"


class DatabaseConnectionError(Exception):
    """Raised when connecting to PostgreSQL fails."""


class TableExistsError(Exception):
    """Raised when target table already contains data and rerun safeguard triggers."""


def validate_input_schema(csv_path: Path) -> List[str]:
    """Inspect CSV header without loading full dataset into memory.

    Returns:
        List of verified column names.
    Raises:
        FileNotFoundError: If CSV does not exist.
        ValueError: If required columns are missing.
    """
    if not csv_path.exists():
        raise FileNotFoundError(f"Processed transaction dataset not found at: {csv_path}")

    # Read header only (0 rows)
    header_df = pd.read_csv(csv_path, nrows=0)
    actual_columns = header_df.columns.tolist()

    missing_cols = set(EXPECTED_CSV_COLUMNS) - set(actual_columns)
    if missing_cols:
        raise ValueError(
            f"Input CSV schema is missing required columns: {sorted(missing_cols)}\n"
            f"Expected: {EXPECTED_CSV_COLUMNS}\nFound: {actual_columns}"
        )

    return actual_columns


def check_postgres_connection(config: DatabaseConfig) -> Tuple[bool, str]:
    """Test connection to PostgreSQL instance.

    Returns:
        Tuple of (is_connected: bool, message: str)
    """
    if not PSYCOPG2_AVAILABLE:
        return False, "psycopg2-binary is not installed. Install via: pip install psycopg2-binary"

    try:
        conn = psycopg2.connect(
            host=config.host,
            port=config.port,
            dbname=config.name,
            user=config.user,
            password=config.password,
            connect_timeout=3,
        )
        conn.close()
        return True, f"Successfully connected to PostgreSQL at {config.host}:{config.port}/{config.name}"
    except Exception as exc:
        return False, f"PostgreSQL connection failed to {config.sanitized_connection_string()}: {exc}"


def format_chunk_for_copy(chunk_df: pd.DataFrame) -> io.StringIO:
    """Format DataFrame chunk as an in-memory CSV string buffer for COPY.

    Reorders and renames columns to match target database schema.
    """
    ordered_cols = list(CSV_TO_DB_COLUMNS.keys())
    subset = chunk_df[ordered_cols].copy()

    buffer = io.StringIO()
    subset.to_csv(buffer, index=False, header=False, na_rep="\\N")
    buffer.seek(0)
    return buffer


def execute_schema_ddl(conn, ddl_path: Path) -> None:
    """Apply DDL script to create tables and indexes."""
    if not ddl_path.exists():
        raise FileNotFoundError(f"DDL script not found at {ddl_path}")

    with open(ddl_path, "r", encoding="utf-8") as f:
        ddl_sql = f.read()

    with conn.cursor() as cur:
        logger.info("Applying schema DDL from: %s", ddl_path)
        cur.execute(ddl_sql)
    conn.commit()
    logger.info("Schema DDL applied successfully.")


def load_dataset(
    csv_path: str | Path = "data/processed/transactions.csv.gz",
    config: Optional[DatabaseConfig] = None,
    table_name: str = DEFAULT_TABLE_NAME,
    chunk_size: int = 100000,
    max_rows: Optional[int] = None,
    truncate: bool = False,
    create_schema: bool = False,
    ddl_path: Optional[str | Path] = "sql/schema/01_schema_definition.sql",
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Stream transaction data from compressed CSV into PostgreSQL.

    Args:
        csv_path: Path to input compressed CSV.
        config: Database connection configuration.
        table_name: Destination PostgreSQL table.
        chunk_size: Number of records to stream per batch.
        max_rows: Optional maximum rows to load (useful for testing samples).
        truncate: If True, resets existing table before loading.
        create_schema: If True, executes schema DDL prior to load.
        ddl_path: Path to DDL script if create_schema is True.
        dry_run: If True, validates CSV and types without writing to database.

    Returns:
        Dict of execution summary and row metrics.
    """
    input_file = Path(csv_path)
    validate_input_schema(input_file)

    if dry_run:
        logger.info("[DRY RUN] Schema validated successfully. Input file: %s", input_file)
        rows_counted = 0
        chunks = 0
        for chunk in pd.read_csv(input_file, chunksize=chunk_size, nrows=max_rows):
            chunks += 1
            rows_counted += len(chunk)
        logger.info("[DRY RUN] Read %d chunks, %d total records. Database write skipped.", chunks, rows_counted)
        return {
            "status": "DRY_RUN_SUCCESS",
            "rows_inspected": rows_counted,
            "chunks_inspected": chunks,
            "table_name": table_name,
            "dry_run": True,
        }

    if not PSYCOPG2_AVAILABLE:
        raise DatabaseConnectionError(
            "psycopg2-binary driver is required for database loading.\n"
            "Install it via: py -m pip install psycopg2-binary"
        )

    db_cfg = config or DatabaseConfig.from_env()
    connected, conn_msg = check_postgres_connection(db_cfg)
    if not connected:
        raise DatabaseConnectionError(
            f"Cannot connect to database:\n  {conn_msg}\n"
            "Please ensure PostgreSQL 15+ is running and check your connection settings."
        )

    start_time = time.perf_counter()
    logger.info("Connected to database: %s", db_cfg.sanitized_connection_string())

    conn = psycopg2.connect(
        host=db_cfg.host,
        port=db_cfg.port,
        dbname=db_cfg.name,
        user=db_cfg.user,
        password=db_cfg.password,
    )

    try:
        with conn.cursor() as cur:
            # Check if target table exists
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = %s AND table_name = %s
                );
            """, (db_cfg.schema, table_name))
            table_exists = cur.fetchone()[0]

            if table_exists:
                # RERUN SAFEGUARD: Check existing rows to prevent silent duplication or data loss
                cur.execute(f"SELECT COUNT(*) FROM {table_name};")
                existing_rows = cur.fetchone()[0]

                if existing_rows > 0:
                    if not truncate:
                        raise TableExistsError(
                            f"Target table '{table_name}' already contains {existing_rows:,} records.\n"
                            "To prevent duplicate loads or accidental data loss, rerun was safely halted.\n"
                            "Specify --truncate to reset the table and reload clean records."
                        )
                    logger.warning(
                        "Resetting table '%s' (truncating %d existing rows) as requested by --truncate...",
                        table_name, existing_rows
                    )
                    cur.execute(f"TRUNCATE TABLE {table_name} RESTART IDENTITY;")
                    conn.commit()

        # Schema initialization (only if requested and table does not exist)
        if create_schema and ddl_path:
            if not table_exists:
                execute_schema_ddl(conn, Path(ddl_path))
        elif not table_exists:
            raise ValueError(
                f"Target table '{db_cfg.schema}.{table_name}' does not exist in database.\n"
                "Run with --create-schema or execute sql/schema/01_schema_definition.sql first."
            )

        # Database target columns for COPY
        db_cols = [CSV_TO_DB_COLUMNS[c] for c in EXPECTED_CSV_COLUMNS]
        cols_sql = ", ".join(db_cols)
        copy_sql = f"COPY {table_name} ({cols_sql}) FROM STDIN WITH (FORMAT csv, HEADER false)"

        total_loaded = 0
        chunks_processed = 0

        logger.info("Beginning streaming COPY into '%s' (chunk size: %d rows)...", table_name, chunk_size)

        reader = pd.read_csv(
            input_file,
            chunksize=chunk_size,
            nrows=max_rows,
            dtype={
                "step": int,
                "type": str,
                "amount": float,
                "nameOrig": str,
                "oldbalanceOrg": float,
                "newbalanceOrig": float,
                "nameDest": str,
                "oldbalanceDest": float,
                "newbalanceDest": float,
                "isFraud": int,
                "isFlaggedFraud": int,
                "transaction_hour": int,
                "transaction_day": int,
            },
        )

        with conn.cursor() as cur:
            for chunk_df in reader:
                chunks_processed += 1
                chunk_len = len(chunk_df)

                # Format chunk as CSV stream
                buffer = format_chunk_for_copy(chunk_df)

                # Execute fast COPY
                cur.copy_expert(copy_sql, buffer)
                total_loaded += chunk_len

                elapsed = time.perf_counter() - start_time
                rate = round(total_loaded / elapsed, 0) if elapsed > 0 else 0
                logger.info(
                    "Chunk %d loaded: +%d rows (cumulative: %d rows, %.1fs elapsed, ~%.0f rows/sec)",
                    chunks_processed, chunk_len, total_loaded, elapsed, rate
                )

        # Commit transaction atomically
        conn.commit()
        logger.info("Transaction committed successfully.")

        # Post-load verification
        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*), SUM(amount) FROM {table_name};")
            verified_count, total_amount = cur.fetchone()

        elapsed_total = round(time.perf_counter() - start_time, 2)
        logger.info("PostgreSQL bulk load complete: %d rows verified in %.2fs", verified_count, elapsed_total)

        return {
            "status": "SUCCESS",
            "table_name": table_name,
            "rows_loaded": total_loaded,
            "verified_rows": verified_count,
            "total_amount": float(total_amount) if total_amount else 0.0,
            "chunks_processed": chunks_processed,
            "elapsed_seconds": elapsed_total,
            "dry_run": False,
        }

    except Exception as exc:
        conn.rollback()
        logger.error("Load failed. Database transaction rolled back: %s", exc)
        raise
    finally:
        conn.close()


def main() -> None:
    """CLI entry point for PostgreSQL transaction loader."""
    parser = argparse.ArgumentParser(
        description="FinSight PostgreSQL Transaction Fact Loader"
    )
    parser.add_argument(
        "--input-path",
        "-i",
        type=str,
        default="data/processed/transactions.csv.gz",
        help="Path to clean processed CSV.gz (default: data/processed/transactions.csv.gz).",
    )
    parser.add_argument(
        "--table-name",
        "-t",
        type=str,
        default=DEFAULT_TABLE_NAME,
        help=f"Target table name in database (default: {DEFAULT_TABLE_NAME}).",
    )
    parser.add_argument(
        "--chunk-size",
        "-c",
        type=int,
        default=100000,
        help="Streaming chunk size (default: 100,000 rows).",
    )
    parser.add_argument(
        "--max-rows",
        "-n",
        type=int,
        default=None,
        help="Optional limit on rows to load (useful for testing samples).",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Reset and truncate target table before loading clean data.",
    )
    parser.add_argument(
        "--create-schema",
        action="store_true",
        help="Execute DDL script to create tables and indexes before loading.",
    )
    parser.add_argument(
        "--ddl-path",
        type=str,
        default="sql/schema/01_schema_definition.sql",
        help="Path to schema DDL file (default: sql/schema/01_schema_definition.sql).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect input CSV and validate schema without connecting to database.",
    )
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Test connection to PostgreSQL and exit.",
    )

    args = parser.parse_args()
    config = DatabaseConfig.from_env()

    if args.check_connection:
        ok, msg = check_postgres_connection(config)
        print(f"Connection test result: {'PASSED' if ok else 'FAILED'}")
        print(msg)
        sys.exit(0 if ok else 1)

    try:
        load_dataset(
            csv_path=args.input_path,
            config=config,
            table_name=args.table_name,
            chunk_size=args.chunk_size,
            max_rows=args.max_rows,
            truncate=args.truncate,
            create_schema=args.create_schema,
            ddl_path=args.ddl_path,
            dry_run=args.dry_run,
        )
    except TableExistsError as err:
        logger.error(str(err))
        sys.exit(1)
    except DatabaseConnectionError as err:
        logger.error(str(err))
        sys.exit(2)
    except Exception as exc:
        logger.exception("Loader execution failed: %s", exc)
        sys.exit(3)


if __name__ == "__main__":
    main()
