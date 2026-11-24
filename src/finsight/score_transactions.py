"""
FinSight — P3 scoring store: write champion scores and reason codes to PostgreSQL.

Two modes, which produce identical scores:

* ``bulk``   — score steps [min_step, max_step] in one run, with recipient history computed over the
               full chronological frame (``features.compute_recipient_prior_in_count``).
* ``hourly`` — score one step at a time, as a live system would: recipient counts come from a running
               state built from strictly earlier steps and updated after each step closes. Each hour is
               its own run.

Guarantees:

* Fraud labels are never loaded (the processed file is read without ``isFraud``).
* The champion is loaded only through ``finsight.champion`` (hash-checked).
* ``transaction_id`` = row number in the processed file; before writing, every scored row is checked
  against ``fct_transactions`` (step, type, amount, sender, recipient).
* Each run is one database transaction: scores, reason codes and the run-log row are committed together
  or not at all. Re-scoring a step range already scored by the same model is refused unless ``--replace``.
* Writes require ``--confirm-database <name>`` equal to the configured database, so a run cannot reach
  the wrong database by accident.

Every score carries ``evaluation_period``: training (1-550), calibration (551-650) or
sealed_test_reported (651-743). Stored scores are operational output, not new performance evidence.
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.alert_explanations import explain_rows  # noqa: E402
from finsight.champion import load_champion, operating_threshold  # noqa: E402
from finsight.features import RECIPIENT_PRIOR_IN_COUNT, compute_recipient_prior_in_count  # noqa: E402
from finsight.train import build_features, file_sha256  # noqa: E402

logger = logging.getLogger("finsight.score_transactions")

INPUT_COLUMNS: Tuple[str, ...] = ("step", "type", "amount", "transaction_hour", "nameOrig", "nameDest")
LABEL_COLUMNS: Tuple[str, ...] = ("isFraud", "is_fraud", "isFlaggedFraud", "is_flagged_fraud")
MAX_STEP = 743


class ScoringError(RuntimeError):
    """Raised when a scoring pre-condition fails; nothing is written."""


class AlreadyScoredError(ScoringError):
    """Raised when the step range was already scored by this model and --replace was not given."""


# ------------------------------------------------------------------------------
# Data (no labels)
# ------------------------------------------------------------------------------
def period_label(step: int) -> str:
    if step <= 550:
        return "training"
    if step <= 650:
        return "calibration"
    return "sealed_test_reported"


def load_unlabelled(data_path: Path, max_step: int = MAX_STEP) -> pd.DataFrame:
    """Processed transactions up to `max_step`, WITHOUT fraud labels; index = transaction_id (row number)."""
    df = pd.read_csv(data_path, usecols=list(INPUT_COLUMNS))
    assert not any(c in df.columns for c in LABEL_COLUMNS)
    df.index = pd.RangeIndex(1, len(df) + 1, name="transaction_id")
    return df[df["step"] <= max_step]


def _check_no_labels(df: pd.DataFrame) -> None:
    present = [c for c in LABEL_COLUMNS if c in df.columns]
    if present:
        raise ScoringError(f"the scorer must not receive label columns: {present}")


# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------
def _score_rows(rows: pd.DataFrame, champion: Dict[str, Any]) -> np.ndarray:
    X = build_features(rows, champion["config"])[champion["feature_names"]]
    return champion["model"].predict_proba(X)[:, 1]


def _score_table(rows: pd.DataFrame, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    return pd.DataFrame({
        "transaction_id": rows.index.to_numpy(dtype=np.int64),
        "step": rows["step"].to_numpy(dtype=np.int64),
        "evaluation_period": [period_label(int(s)) for s in rows["step"].to_numpy()],
        "recipient_prior_in_count": rows[RECIPIENT_PRIOR_IN_COUNT].to_numpy(dtype=np.int64),
        "risk_score": scores.astype(float),
        "operating_threshold": float(threshold),
        "is_alert": scores >= threshold,
    })


def bulk_score(df: pd.DataFrame, champion: Dict[str, Any], min_step: int, max_step: int
               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Score steps [min_step, max_step]; history over the full frame (must start at step 1)."""
    _check_no_labels(df)
    if int(df["step"].min()) != 1:
        raise ScoringError("bulk scoring needs the frame from step 1 so recipient history is complete")
    hist = df[df["step"] <= max_step].assign(
        **{RECIPIENT_PRIOR_IN_COUNT: lambda d: compute_recipient_prior_in_count(d)})
    rows = hist[hist["step"] >= min_step]
    scores = _score_rows(rows, champion)
    return rows, _score_table(rows, scores, operating_threshold(champion))


def hourly_score(df: pd.DataFrame, champion: Dict[str, Any], min_step: int, max_step: int
                 ) -> Iterable[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    """Yield (step, rows, score table) one hour at a time from a running recipient-count state."""
    _check_no_labels(df)
    dest_col = "nameDest"
    counts: Dict[str, int] = df.loc[df["step"] < min_step, dest_col].value_counts().to_dict()
    threshold = operating_threshold(champion)
    for s in range(min_step, max_step + 1):
        rows = df[df["step"] == s]
        if len(rows) == 0:
            continue
        prior = rows[dest_col].map(counts).fillna(0).astype(np.int64)              # strictly earlier steps only
        rows = rows.assign(**{RECIPIENT_PRIOR_IN_COUNT: prior.to_numpy()})
        yield s, rows, _score_table(rows, _score_rows(rows, champion), threshold)
        for d, n in rows[dest_col].value_counts().items():                         # the hour closes
            counts[d] = counts.get(d, 0) + int(n)


def reason_records(rows: pd.DataFrame, table: pd.DataFrame, champion: Dict[str, Any]) -> List[Dict[str, Any]]:
    alert = table["is_alert"].to_numpy()
    recs = explain_rows(rows[alert], table["risk_score"].to_numpy()[alert], operating_threshold(champion), champion)
    for r in recs:
        r["transaction_id"] = int(r["row_index"])
    return recs


# ------------------------------------------------------------------------------
# Database
# ------------------------------------------------------------------------------
def verify_mapping(conn, rows: pd.DataFrame, chunk: int = 500_000) -> int:
    """Check every row's transaction_id against fct_transactions; raise on any mismatch."""
    ids = rows.index.to_numpy(dtype=np.int64)
    lo, hi = int(ids.min()), int(ids.max())
    checked = 0
    with conn.cursor() as cur:
        for a in range(lo, hi + 1, chunk):
            b = min(a + chunk - 1, hi)
            cur.execute("SELECT transaction_id, step, type, amount, name_orig, name_dest FROM fct_transactions "
                        "WHERE transaction_id BETWEEN %s AND %s ORDER BY transaction_id", (a, b))
            db = pd.DataFrame(cur.fetchall(), columns=["transaction_id", "step", "type", "amount", "nameOrig", "nameDest"])
            part = rows.loc[(rows.index >= a) & (rows.index <= b)]
            db = db.set_index("transaction_id").reindex(part.index)
            ok = (db["step"].to_numpy() == part["step"].to_numpy()) & (db["type"].to_numpy() == part["type"].to_numpy()) \
                & (np.abs(db["amount"].astype(float).to_numpy() - part["amount"].to_numpy()) < 0.005) \
                & (db["nameOrig"].to_numpy() == part["nameOrig"].to_numpy()) \
                & (db["nameDest"].to_numpy() == part["nameDest"].to_numpy())
            if not ok.all():
                bad = part.index[~ok][:5].tolist()
                raise ScoringError(f"transaction mapping mismatch for transaction_id {bad}; nothing written")
            checked += len(part)
    return checked


def _copy_scores(cur, table: pd.DataFrame, run_id: str, model_sha: str) -> None:
    buf = io.StringIO()
    out = table.assign(model_artifact_sha256=model_sha, run_id=run_id)[
        ["transaction_id", "model_artifact_sha256", "run_id", "step", "evaluation_period", "recipient_prior_in_count",
         "risk_score", "operating_threshold", "is_alert"]]
    out.to_csv(buf, index=False, header=False, float_format="%.17g")
    buf.seek(0)
    cur.copy_expert("COPY fct_model_scores (transaction_id, model_artifact_sha256, run_id, step, evaluation_period, "
                    "recipient_prior_in_count, risk_score, operating_threshold, is_alert) FROM STDIN WITH (FORMAT csv)", buf)


def _insert_reasons(cur, recs: List[Dict[str, Any]], run_id: str, model_sha: str) -> None:
    from psycopg2.extras import execute_values

    values = [(r["transaction_id"], model_sha, run_id, json.dumps(r["model_explanation"]["top_contributors"]),
               r["model_explanation"]["summary"], r["model_explanation"]["disclaimer"],
               r["model_explanation"]["base_value_log_odds"], r["model_explanation"]["margin_log_odds"],
               r["model_explanation"]["additivity_error"], r["rule_conditions"]["rule_ids"],
               json.dumps(r["rule_conditions"]["rules_met"]), r["rule_conditions"]["rule_score"],
               r["rule_conditions"]["rule_tier"], r["rule_conditions"]["note"]) for r in recs]
    if values:
        execute_values(cur, "INSERT INTO fct_alert_reasons (transaction_id, model_artifact_sha256, run_id, top_contributors, "
                            "model_summary, model_disclaimer, base_value_log_odds, margin_log_odds, additivity_error, rule_ids, "
                            "rule_conditions, rule_score, rule_tier, rule_note) VALUES %s", values, page_size=1000)


def write_run(conn, table: pd.DataFrame, recs: List[Dict[str, Any]], meta: Dict[str, Any], replace: bool = False) -> str:
    """Write one run atomically. Returns the run_id. On any error the transaction is rolled back."""
    run_id = meta.get("run_id") or f"run_{time.strftime('%Y%m%dT%H%M%S', time.gmtime())}_{uuid.uuid4().hex[:8]}"
    a, b = int(table["step"].min()), int(table["step"].max())
    sha = meta["model_artifact_sha256"]
    if int(table["is_alert"].sum()) != len(recs):
        raise ScoringError("every alert needs exactly one reason record")
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM fct_model_scores WHERE model_artifact_sha256 = %s AND step BETWEEN %s AND %s",
                        (sha, a, b))
            existing = cur.fetchone()[0]
            if existing and not replace:
                raise AlreadyScoredError(f"steps {a}-{b} already have {existing} scores from this model; use --replace")
            if existing:
                cur.execute("DELETE FROM fct_alert_reasons r USING fct_model_scores s WHERE r.transaction_id = s.transaction_id "
                            "AND r.model_artifact_sha256 = s.model_artifact_sha256 AND s.model_artifact_sha256 = %s "
                            "AND s.step BETWEEN %s AND %s", (sha, a, b))
                cur.execute("DELETE FROM fct_model_scores WHERE model_artifact_sha256 = %s AND step BETWEEN %s AND %s", (sha, a, b))
            _copy_scores(cur, table, run_id, sha)
            _insert_reasons(cur, recs, run_id, sha)
            cur.execute("INSERT INTO scoring_run_log (run_id, model_artifact, model_artifact_sha256, config_name, mode, min_step, "
                        "max_step, operating_threshold, rows_scored, alerts, data_sha256, code_commit, started_at, completed_at, "
                        "status) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s, now(), 'SUCCESS')",
                        (run_id, meta["model_artifact"], sha, meta["config_name"], meta["mode"], a, b,
                         float(table["operating_threshold"].iloc[0]), len(table), int(table["is_alert"].sum()),
                         meta["data_sha256"], meta["code_commit"], meta["started_at"]))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    return run_id


def code_commit() -> str:
    head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
    return head + ("+dirty" if dirty else "")


def main() -> None:
    from finsight.load_to_postgres import DatabaseConfig
    import psycopg2

    parser = argparse.ArgumentParser(description="Write champion scores and reason codes to PostgreSQL")
    parser.add_argument("--mode", choices=["bulk", "hourly"], required=True)
    parser.add_argument("--min-step", type=int, default=1)
    parser.add_argument("--max-step", type=int, default=MAX_STEP)
    parser.add_argument("--replace", action="store_true", help="Re-score steps already scored by this model")
    parser.add_argument("--confirm-database", required=True, help="Must equal the configured DB_NAME")
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = DatabaseConfig.from_env()
    if args.confirm_database != cfg.name:
        raise SystemExit(f"--confirm-database {args.confirm_database!r} does not match configured database {cfg.name!r}")
    if not 1 <= args.min_step <= args.max_step <= MAX_STEP:
        raise SystemExit("invalid step range")
    logger.info("target %s:%s/%s", cfg.host, cfg.port, cfg.name)
    champ = load_champion()
    df = load_unlabelled(Path(args.data_path), args.max_step)
    meta = {"model_artifact": champ["entry"]["artifact"], "model_artifact_sha256": champ["entry"]["artifact_sha256"],
            "config_name": champ["entry"]["config_name"], "mode": args.mode,
            "data_sha256": file_sha256(Path(args.data_path)), "code_commit": code_commit()}
    conn = psycopg2.connect(host=cfg.host, port=cfg.port, dbname=cfg.name, user=cfg.user, password=cfg.password)
    try:
        if args.mode == "bulk":
            t0 = time.time()
            meta["started_at"] = pd.Timestamp.now('UTC').to_pydatetime()
            rows, table = bulk_score(df, champ, args.min_step, args.max_step)
            checked = verify_mapping(conn, rows)
            recs = reason_records(rows, table, champ)
            run_id = write_run(conn, table, recs, meta, args.replace)
            logger.info("run %s: %d rows (mapping verified %d), %d alerts, %.0fs", run_id, len(table), checked,
                        int(table["is_alert"].sum()), time.time() - t0)
        else:
            for s, rows, table in hourly_score(df, champ, args.min_step, args.max_step):
                meta["started_at"] = pd.Timestamp.now('UTC').to_pydatetime()
                verify_mapping(conn, rows)
                run_id = write_run(conn, table, reason_records(rows, table, champ), meta, args.replace)
                logger.info("step %d -> run %s: %d rows, %d alerts", s, run_id, len(table), int(table["is_alert"].sum()))
    finally:
        conn.close()


if __name__ == "__main__":
    main()
