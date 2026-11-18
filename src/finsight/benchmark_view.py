"""
FinSight — Benchmark view generator (P0-4).

Regenerates `sql/views/04_vw_risk_detection_benchmark.sql` and
`reports/evaluation/benchmark_rows.json` from the actual model artifacts, the
current rule engine and the processed dataset. The view used to be hand-typed
numbers; it is now always written by this script and must not be edited by hand.

Partitions (locked final test, steps >= 651, is excluded):
* validation            451–550  — used for threshold selection and experiment ranking.
* development_holdout   551–650  — out-of-time for all models trained on steps <= 450,
                                   but previously inspected (part of the old holdout).

Usage:
    py src/finsight/benchmark_view.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.evaluate import _confusion, assert_development_steps, simulated_days
from finsight.features import extract_features
from finsight.risk_rules import RiskEngine
from finsight.train import MODEL_CONFIGS, build_features, file_sha256

logger = logging.getLogger("finsight.benchmark_view")

PARTITIONS = {
    "validation": (451, 550),
    "development_holdout": (551, 650),
}

VIEW_HEADER = """-- ==============================================================================
-- FinSight Analytical Mart: Risk Detection Benchmark Performance
-- View Name: vw_risk_detection_benchmark
--
-- GENERATED FILE — DO NOT EDIT BY HAND.
-- Regenerate with: py src/finsight/benchmark_view.py
-- Generated (UTC): {generated}
-- Data file SHA-256: {data_sha}
-- Source rows: reports/evaluation/benchmark_rows.json
--
-- Grain: one row per evaluation partition x system x operating point.
-- Partitions: validation = steps 451-550; development_holdout = steps 551-650.
-- The locked final test (steps >= 651) is deliberately excluded.
-- Fraud prevalence differs between partitions, so precision/F1 are comparable
-- only within a partition. Rule flags and model alerts are not confirmed fraud.
-- average_precision / roc_auc are NULL for discrete rule tiers.
-- ==============================================================================
"""


def _row(partition: str, system: str, operating_point: str, y: np.ndarray, pred: np.ndarray,
         steps: np.ndarray, score: Optional[np.ndarray]) -> Dict[str, Any]:
    cm = _confusion(y, pred)
    return {
        "evaluation_partition": partition,
        "system_name": system,
        "operating_point": operating_point,
        "true_positives": cm["true_positives"],
        "false_positives": cm["false_positives"],
        "false_negatives": cm["false_negatives"],
        "true_negatives": cm["true_negatives"],
        "alerts_per_day": round(cm["alerts"] / simulated_days(steps), 1),
        "average_precision": None if score is None else round(float(average_precision_score(y, score)), 4),
        "roc_auc": None if score is None else round(float(roc_auc_score(y, score)), 4),
    }


def compute_rows(data_path: Path, champion_path: Path, lr_path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(data_path, usecols=["step", "type", "amount", "nameDest", "transaction_hour", "isFraud"])
    champion = joblib.load(champion_path)
    lr = joblib.load(lr_path)
    champ_cfg = MODEL_CONFIGS["hgb_baseline8_balanced"]
    engine = RiskEngine()
    rows: List[Dict[str, Any]] = []
    for part, (lo, hi) in PARTITIONS.items():
        d = df[(df["step"] >= lo) & (df["step"] <= hi)]
        steps = d["step"].to_numpy()
        assert_development_steps(steps, "benchmark view")
        y = d["isFraud"].to_numpy(dtype=int)

        s = champion["model"].predict_proba(build_features(d, champ_cfg)[champion["feature_names"]])[:, 1]
        t = float(champion["selected_threshold"])
        rows.append(_row(part, "HistGradientBoosting (frozen champion, 8 features)",
                         f"t = {t:.4f} (validation F1-optimal)", y, s >= t, steps, s))

        lr_X = extract_features(d, include_merchant_flag=True)[lr["feature_names"]]
        s_lr = lr["model"].predict_proba(lr_X)[:, 1]
        t_lr = float(lr["selected_threshold"])
        rows.append(_row(part, "Logistic Regression (legacy 10-feature artifact)",
                         f"t = {t_lr:.4f} (validation F1-optimal)", y, s_lr >= t_lr, steps, s_lr))

        r = engine.evaluate_dataframe(d)
        tier = r["risk_tier"].to_numpy()
        rows.append(_row(part, "Rule Engine (HIGH tier)", "score >= 70", y, tier == "HIGH", steps, None))
        rows.append(_row(part, "Rule Engine (MEDIUM + HIGH tiers)", "score >= 30", y, tier != "LOW", steps, None))

        prior = np.full(len(y), 0.0)
        rows.append(_row(part, "Dummy baseline (never alerts)", "no alerts", y, prior > 0.5, steps, None))
        rows[-1]["average_precision"] = round(float(y.mean()), 4)
        rows[-1]["roc_auc"] = 0.5
    return rows


def _sql_literal(v: Any) -> str:
    if v is None:
        return "NULL"
    if isinstance(v, str):
        return "'" + v.replace("'", "''") + "'"
    return repr(v)


def render_view_sql(rows: List[Dict[str, Any]], generated: str, data_sha: str) -> str:
    cols = ["evaluation_partition", "system_name", "operating_point", "true_positives", "false_positives",
            "false_negatives", "true_negatives", "alerts_per_day", "average_precision", "roc_auc"]
    values = ",\n".join(
        "        (" + ", ".join(_sql_literal(r[c]) for c in cols) + ")" for r in rows
    )
    body = f"""
CREATE OR REPLACE VIEW vw_risk_detection_benchmark AS
WITH raw_benchmark_data (
    {", ".join(cols)}
) AS (
    VALUES
{values}
)
SELECT
    evaluation_partition,
    system_name,
    operating_point,
    true_positives,
    false_positives,
    false_negatives,
    true_negatives,
    (true_positives + false_positives + false_negatives + true_negatives) AS total_evaluation_population,
    (true_positives + false_positives) AS alert_count,
    ROUND(COALESCE(CAST(true_positives + false_positives AS NUMERIC)
        / NULLIF(true_positives + false_positives + false_negatives + true_negatives, 0) * 100, 0.0), 4) AS alert_rate_pct,
    ROUND(COALESCE(CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_positives, 0), 0.0), 4) AS precision,
    ROUND(COALESCE(CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_negatives, 0), 0.0), 4) AS recall,
    ROUND(COALESCE(2.0 * true_positives / NULLIF(2.0 * true_positives + false_positives + false_negatives, 0), 0.0), 4) AS f1_score,
    -- Keep the existing view's column types (unconstrained NUMERIC). PostgreSQL's
    -- CREATE OR REPLACE VIEW rejects any type/typmod change on existing columns;
    -- new columns may only be appended at the end.
    ROUND(average_precision, 4) AS average_precision,
    ROUND(roc_auc, 4) AS roc_auc,
    ROUND(alerts_per_day, 1) AS alerts_per_day
FROM raw_benchmark_data
ORDER BY evaluation_partition ASC, f1_score DESC;

COMMENT ON VIEW vw_risk_detection_benchmark IS
'Generated by src/finsight/benchmark_view.py. Validation (451-550) and development holdout (551-650) only; locked final test excluded.';
"""
    return VIEW_HEADER.format(generated=generated, data_sha=data_sha) + body


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the risk detection benchmark view")
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--champion-path", default="models/hist_gradient_boosting_no_day.joblib")
    parser.add_argument("--lr-path", default="models/logistic_regression.joblib")
    parser.add_argument("--view-path", default="sql/views/04_vw_risk_detection_benchmark.sql")
    parser.add_argument("--rows-path", default="reports/evaluation/benchmark_rows.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_sha = file_sha256(Path(args.data_path))
    rows = compute_rows(Path(args.data_path), Path(args.champion_path), Path(args.lr_path))
    Path(args.rows_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rows_path).write_text(json.dumps({
        "generated_utc": generated, "data_sha256": data_sha,
        "champion_artifact_sha256": file_sha256(Path(args.champion_path)),
        "partitions": PARTITIONS, "rows": rows,
    }, indent=2), encoding="utf-8")
    Path(args.view_path).write_text(render_view_sql(rows, generated, data_sha), encoding="utf-8")
    logger.info("Wrote %d rows to %s and %s", len(rows), args.view_path, args.rows_path)


if __name__ == "__main__":
    main()
