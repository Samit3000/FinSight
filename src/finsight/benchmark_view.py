"""
FinSight — Benchmark view generator (P0-4).

Regenerates `sql/views/04_vw_risk_detection_benchmark.sql` and
`reports/evaluation/benchmark_rows.json` from the actual model artifacts, the
current rule engine and the processed dataset. The view used to be hand-typed
numbers; it is now always written by this script and must not be edited by hand.

Partitions:
* validation            451–550  — scored here; used for threshold selection and experiment ranking.
* development_holdout   551–650  — scored here; out-of-time for models trained on steps <= 450,
                                   but previously inspected (part of the old holdout).
* sealed_final_test     651–743  — NOT scored here. Rows are copied from the recorded one-time P1.5
                                   result (reports/final_test/p15_final_test.json); the sealed labels
                                   are never read again.

The active champion (E4, models/champion.json) was trained on steps 1-550 and calibrated on 551-650,
so its only held-out rows are the sealed ones. The development partitions show the previous champion,
the legacy logistic regression, the rule engine and a dummy baseline.

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

from finsight.champion import REGISTRY_PATH, read_registry
from finsight.evaluate import _confusion, assert_development_steps, simulated_days
from finsight.features import extract_features
from finsight.risk_rules import RiskEngine
from finsight.train import MODEL_CONFIGS, build_features, file_sha256

logger = logging.getLogger("finsight.benchmark_view")

PARTITIONS = {
    "validation": (451, 550),
    "development_holdout": (551, 650),
}
SEALED_PARTITION = ("sealed_final_test", (651, 743))
P15_RESULT = Path("reports/final_test/p15_final_test.json")

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
-- Partitions: validation = steps 451-550; development_holdout = steps 551-650 (scored by the script);
-- sealed_final_test = steps 651-743, copied from the recorded one-time P1.5 result, never re-scored.
-- Active champion: E4 (models/champion.json); it appears only in sealed_final_test, its one held-out period.
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
        rows.append(_row(part, "HistGradientBoosting (previous champion, 8 features)",
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


def _sealed_row(system: str, operating_point: str, tp: float, fp: float, fn: float, rows: int,
                alerts_per_day: float, ap: Optional[float], roc: Optional[float]) -> Dict[str, Any]:
    tp, fp, fn = int(round(tp)), int(round(fp)), int(round(fn))
    return {"evaluation_partition": SEALED_PARTITION[0], "system_name": system, "operating_point": operating_point,
            "true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "true_negatives": int(rows - tp - fp - fn), "alerts_per_day": round(float(alerts_per_day), 1),
            "average_precision": ap, "roc_auc": roc}


def sealed_rows(p15: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rows for steps 651-743 from the recorded P1.5 result (no scoring, no label access)."""
    S, P = p15["systems"], p15["policies"]
    n = int(S["E4"]["rows"])
    t_e4 = p15["meta"]["training"]["E4"]["thresholds"]["100"]
    t_e0 = p15["meta"]["training"]["E0"]["thresholds"]["100"]
    e4, e0, rule = P["E4 B @100"], P["E0 B @100"], p15["rule_engine_high"]
    ch = S["champion_artifact"]["at_threshold"]
    out = [
        _sealed_row("HistGradientBoosting E4 (active champion, recipient count)",
                    f"t = {t_e4:.4f} (100 alerts/day, set on steps 551-650)", e4["true_positives"],
                    e4["false_positives"], e4["false_negatives"], n, e4["alerts_per_day"],
                    S["E4"]["average_precision"], S["E4"]["roc_auc"]),
        _sealed_row("HistGradientBoosting (champion configuration retrained on steps 1-550)",
                    f"t = {t_e0:.4f} (100 alerts/day, set on steps 551-650)", e0["true_positives"],
                    e0["false_positives"], e0["false_negatives"], n, e0["alerts_per_day"],
                    S["E0"]["average_precision"], S["E0"]["roc_auc"]),
        _sealed_row("HistGradientBoosting (previous champion, 8 features)",
                    f"t = {p15['meta']['champion_threshold']:.4f} (validation F1-optimal)", ch["true_positives"],
                    ch["false_positives"], ch["false_negatives"], n, ch["alerts_per_day"],
                    S["champion_artifact"]["average_precision"], S["champion_artifact"]["roc_auc"]),
        _sealed_row("Rule Engine (HIGH tier)", "score >= 70", rule["true_positives"], rule["false_positives"],
                    rule["false_negatives"], n, rule["alerts_per_day"], None, None),
    ]
    if out[2]["true_negatives"] != ch["true_negatives"]:
        raise ValueError("sealed row population does not reconcile with the P1.5 confusion matrix")
    return out


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
'Generated by src/finsight/benchmark_view.py. Validation (451-550) and development holdout (551-650) scored by the script; sealed final test (651-743) copied from the recorded P1.5 result.';
"""
    return VIEW_HEADER.format(generated=generated, data_sha=data_sha) + body


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate the risk detection benchmark view")
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--champion-path", default=None,
                        help="Previous champion artifact (default: models/champion.json 'previous')")
    parser.add_argument("--p15-path", default=str(P15_RESULT))
    parser.add_argument("--lr-path", default="models/logistic_regression.joblib")
    parser.add_argument("--view-path", default="sql/views/04_vw_risk_detection_benchmark.sql")
    parser.add_argument("--rows-path", default="reports/evaluation/benchmark_rows.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    generated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data_sha = file_sha256(Path(args.data_path))
    registry = read_registry(REGISTRY_PATH)
    prev_path = Path(args.champion_path or registry["previous"]["artifact"])
    p15 = json.loads(Path(args.p15_path).read_text(encoding="utf-8"))
    rows = compute_rows(Path(args.data_path), prev_path, Path(args.lr_path)) + sealed_rows(p15)
    Path(args.rows_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rows_path).write_text(json.dumps({
        "generated_utc": generated, "data_sha256": data_sha,
        "previous_champion_artifact_sha256": file_sha256(prev_path),
        "active_champion_artifact_sha256": registry["active"]["artifact_sha256"],
        "sealed_rows_source": {"path": Path(args.p15_path).as_posix(), "sha256": file_sha256(Path(args.p15_path))},
        "partitions": {**PARTITIONS, SEALED_PARTITION[0]: SEALED_PARTITION[1]}, "rows": rows,
    }, indent=2), encoding="utf-8")
    Path(args.view_path).write_text(render_view_sql(rows, generated, data_sha), encoding="utf-8")
    logger.info("Wrote %d rows to %s and %s", len(rows), args.view_path, args.rows_path)


if __name__ == "__main__":
    main()
