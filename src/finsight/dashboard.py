"""
FinSight — data bundle for the interactive dashboard (Milestone 7).

The dashboard is a static front end in `dashboard/` (index.html, app.css, app.js: no build step, no
network access, works from the file system). This module produces its only data file,
`dashboard/data.js` (`window.FINSIGHT_DATA = {...}`), from a READ-ONLY database session.

Sources:
* Descriptive dataset views 01 and 03 (they contain the dataset's own labelled fraud counts).
* Model evaluation: only `vw_risk_detection_benchmark` (development partitions scored by code; sealed-test
  rows are the recorded one-time P1.5 result). Nothing here computes new performance figures.
* Workload, alert mix, reasons, score distribution and the alert queue: label-free views 05-10.
* Model registry (`models/champion.json`), the rebuild verification manifest and the ingestion run report.

Usage (database chosen by DB_HOST / DB_PORT / DB_NAME; nothing is written to it):
    py src/finsight/dashboard.py --out dashboard
"""

from __future__ import annotations

import argparse
import datetime as _dt
import decimal
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

logger = logging.getLogger("finsight.dashboard")

MODEL_NOTE = "Model attribution shows how inputs moved a model score; it is not evidence that a transaction is fraudulent."
RULE_NOTE = "Rule conditions are deterministic checks shown for context; they do not raise alerts."
LABEL_COLUMNS = ("is_fraud", "is_flagged_fraud", "isFraud")


def _plain(v: Any) -> Any:
    if isinstance(v, decimal.Decimal):
        return float(v)
    if isinstance(v, (_dt.datetime, _dt.date)):
        return v.isoformat()
    return v


def _rows(cur, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    leaked = [c for c in cols if c in LABEL_COLUMNS]
    if leaked:
        raise ValueError(f"dashboard queries must not return label columns: {leaked}")
    return [{c: _plain(v) for c, v in zip(cols, r)} for r in cur.fetchall()]


def collect(conn, registry: Dict[str, Any], extras: Dict[str, Any] | None = None) -> Dict[str, Any]:
    """Everything the front end shows. The session is set read-only first and verified."""
    conn.set_session(readonly=True)
    active = registry["active"]
    sha = active["artifact_sha256"]
    with conn.cursor() as cur:
        cur.execute("SHOW transaction_read_only")
        if cur.fetchone()[0] != "on":
            raise RuntimeError("dashboard collection requires a read-only session")
        d: Dict[str, Any] = {}
        d["daily"] = _rows(cur, "SELECT simulation_day AS day, transaction_count AS transactions, gross_transaction_value AS value, "
                                "labelled_fraud_transaction_count AS labelled_fraud FROM vw_daily_transaction_kpis ORDER BY 1")
        d["rails"] = _rows(cur, "SELECT payment_rail, transaction_count, gross_transaction_value, rail_volume_share_pct, "
                                "labelled_fraud_count, labelled_fraud_rate_pct FROM vw_payment_rail_performance "
                                "ORDER BY transaction_count DESC")
        d["benchmark"] = _rows(cur, "SELECT evaluation_partition, system_name, operating_point, alerts_per_day, precision, recall, "
                                    "f1_score, average_precision, roc_auc, true_positives, false_positives, false_negatives, "
                                    "true_negatives FROM vw_risk_detection_benchmark")
        d["workload_daily"] = _rows(cur, "SELECT simulated_day AS day, evaluation_period AS period, hours_covered, is_full_day, "
                                         "transactions_scored AS scored, alerts, first_time_recipient_alerts AS first_time "
                                         "FROM vw_alert_workload_daily WHERE model_artifact_sha256 = %s ORDER BY 1, 2", (sha,))
        d["workload_hourly"] = _rows(cur, "SELECT step, simulated_day AS day, hour_of_day AS hour, evaluation_period AS period, "
                                          "transactions_scored AS scored, alerts FROM vw_alert_workload_hourly "
                                          "WHERE model_artifact_sha256 = %s ORDER BY step", (sha,))
        d["mix"] = _rows(cur, "SELECT evaluation_period AS period, payment_type AS type, hour_band, amount_band, alerts, "
                              "first_time_recipient_alerts AS first_time FROM vw_alert_mix WHERE model_artifact_sha256 = %s", (sha,))
        d["reasons"] = _rows(cur, "SELECT evaluation_period AS period, reason_source AS source, reason, alerts "
                                  "FROM vw_alert_reason_summary WHERE model_artifact_sha256 = %s", (sha,))
        d["score_distribution"] = _rows(cur, "SELECT evaluation_period AS period, score_bin_start AS bin, transactions "
                                             "FROM vw_score_distribution WHERE model_artifact_sha256 = %s ORDER BY 1, 2", (sha,))
        d["alerts"] = _rows(cur, "SELECT transaction_id AS id, step, simulated_day AS day, transaction_hour AS hour, type, amount, "
                                 "name_dest AS recipient, evaluation_period AS period, recipient_prior_in_count AS recipient_prior, "
                                 "risk_score AS score, top_contributors, model_summary, rule_ids, rule_conditions, rule_score, rule_tier "
                                 "FROM vw_alert_queue WHERE model_artifact_sha256 = %s ORDER BY step, risk_score DESC", (sha,))
        d["runs"] = _rows(cur, "SELECT run_id, mode, min_step, max_step, rows_scored, alerts, completed_at, code_commit, data_sha256 "
                               "FROM scoring_run_log WHERE model_artifact_sha256 = %s ORDER BY completed_at, min_step", (sha,))
    conn.rollback()
    for a in d["alerts"]:
        # Compact but lossless for display: the summary sentence is rebuilt in the browser from the contributors, and
        # rule names come from one lookup table instead of being repeated on every alert.
        a["rule_ids"] = [r for r in (a["rule_ids"] or []) if r != "none"]
        a.pop("model_summary", None)
        a["rule_conditions"] = [{"rule_id": c["rule_id"], "points": c["points"], "message": c["message"]}
                                for c in (a["rule_conditions"] or [])]
        a["top_contributors"] = [{"feature": c["feature"], "value": c["value"], "contribution": c["contribution_log_odds"],
                                  "description": c["description"]} for c in a["top_contributors"]]
        a["score"] = round(float(a["score"]), 6)
    d["rule_names"] = {"RULE_001_HIGH_VALUE": "High value", "RULE_002_DRAINAGE_RAIL": "Drainage rail",
                       "RULE_003_OVERNIGHT_DRAINAGE": "Overnight", "RULE_004_LARGE_ROUND_AMOUNT": "Round amount"}
    d["registry"] = {"active": active, "previous": registry.get("previous")}
    d["notes"] = {"model": MODEL_NOTE, "rule": RULE_NOTE}
    d.update(extras or {})
    return d


def load_extras(root: Path = Path(".")) -> Dict[str, Any]:
    """Local, non-database facts shown on the Data & pipeline tab."""
    out: Dict[str, Any] = {}
    run = root / "reports/data_quality/pipeline_run.json"
    if run.exists():
        r = json.loads(run.read_text(encoding="utf-8"))
        out["ingestion"] = {"status": r.get("status"), "completed_at": r.get("completed_at"), "elapsed_seconds": r.get("elapsed_seconds"),
                            "rows": r.get("row_accounting", {}), "duplicates": r.get("duplicate_signatures_detected"),
                            "fraud_rate_pct": r.get("target_distribution", {}).get("fraud_rate_pct"),
                            "chunks": r.get("chunks_processed"), "chunk_size": r.get("chunk_size")}
    man = root / "models/candidates/hgb_e4_final.manifest.json"
    if man.exists():
        m = json.loads(man.read_text(encoding="utf-8"))
        v = m.get("verification", {})
        out["rebuild"] = {"config": m.get("config_name"), "rows_compared": v.get("rows_compared"),
                          "max_abs_score_difference": v.get("max_abs_score_difference"), "reproduced": v.get("reproduced"),
                          "alerts": v.get("alerts_reference")}
    return out


def wilson(k: float, n: float, z: float = 1.959964) -> List[float] | None:
    """95% Wilson score interval for a proportion k/n (expected counts are rounded to whole rows)."""
    k, n = round(k), round(n)
    if n <= 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return [round(max(0.0, mid - half), 4), round(min(1.0, mid + half), 4)]


def confusion_metrics(tp: float, fp: float, fn: float, tn: float) -> Dict[str, Any]:
    """Every threshold metric from one confusion matrix. Nothing here reads labels: the counts are recorded results."""
    n, pos, alerts = tp + fp + fn + tn, tp + fn, tp + fp
    div = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    precision, recall, specificity = div(tp, alerts), div(tp, pos), div(tn, fp + tn)
    mcc_den = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** 0.5
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn, "rows": n, "alerts": alerts,
        "accuracy": div(tp + tn, n), "precision": precision, "recall": recall, "specificity": specificity,
        "false_positive_rate": div(fp, fp + tn),
        "balanced_accuracy": round((recall + specificity) / 2, 4) if recall is not None and specificity is not None else None,
        "f1": div(2 * tp, 2 * tp + fp + fn), "mcc": round((tp * tn - fp * fn) / mcc_den, 4) if mcc_den else 0.0,
        "precision_ci95": wilson(tp, alerts), "recall_ci95": wilson(tp, pos),
    }


def _read(root: Path, rel: str) -> Dict[str, Any] | None:
    p = root / rel
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def model_card(root: Path, registry: Dict[str, Any]) -> Dict[str, Any]:
    """Hyperparameters read from the saved champion artifact itself, after checking its SHA-256 against the registry."""
    import hashlib
    import joblib

    active = registry["active"]
    path = root / active["artifact"]
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != active["artifact_sha256"]:
        raise ValueError(f"{path} does not match the registry SHA-256")
    obj = joblib.load(path)
    est = obj.get("model", obj) if isinstance(obj, dict) else obj
    params = est.get_params()
    keep = ("loss", "learning_rate", "max_iter", "max_leaf_nodes", "max_depth", "min_samples_leaf", "l2_regularization",
            "max_bins", "early_stopping", "class_weight", "random_state")
    feats = list(getattr(est, "feature_names_in_", [])) or (obj.get("features") if isinstance(obj, dict) else [])
    return {"estimator": type(est).__name__, "sklearn_params": {k: params.get(k) for k in keep if k in params},
            "n_iter": int(getattr(est, "n_iter_", 0)), "features": [str(f) for f in feats], "artifact_sha256": sha}


def load_study(root: Path, registry: Dict[str, Any]) -> Dict[str, Any]:
    """The ML study as recorded: development folds (P1.3), the one-time sealed test (P1.5), SHAP importance and the
    development-fold curves. Only saved results are read; accuracy, MCC etc. are derived from recorded counts."""
    s1, s3 = _read(root, "reports/evaluation/p13_screening.json"), _read(root, "reports/evaluation/p13_confirmation.json")
    p15 = _read(root, "reports/final_test/p15_final_test.json")
    out: Dict[str, Any] = {}
    if s1 and s3:
        keep = ("rows", "labelled_fraud", "fraud_prevalence_pct", "steps", "average_precision", "average_precision_ci95",
                "roc_auc", "train_rows", "train_fraud", "n_iter")
        dev: Dict[str, Any] = {}
        for src in (s1, s3):
            for lab, folds in src["results"].items():
                for f, r in folds.items():
                    dev.setdefault(lab, {})[f] = {**{k: r.get(k) for k in keep},
                                                  "at_budget": {b: confusion_metrics(*(r["at_alert_budget"][f"{b}_per_day"][k] for k in ("true_positives", "false_positives", "false_negatives", "true_negatives")))
                                                                for b in (50, 100, 200)}}
        diffs: Dict[str, Any] = {}
        for src in (s1, s3):
            for name, d in src["paired_ap_differences"].items():
                diffs.setdefault(name, {"isolates": d["isolates"]}).update({f: v for f, v in d.items() if f != "isolates"})
        out["development"] = {"configs": s1["meta"]["configs"], "results": dev, "paired_ap_differences": diffs,
                              "shortlist_rule": s1.get("shortlist_rule"), "shortlist": s1.get("shortlist"),
                              "confirmation_rule": s3.get("confirmation_rule"), "confirmation_decision": s3.get("confirmation_decision"),
                              "disclosure": s3["meta"].get("disclosure")}
    if p15:
        systems = {}
        for lab, r in p15["systems"].items():
            sysd = {k: r.get(k) for k in ("rows", "labelled_fraud", "fraud_prevalence_pct", "simulated_days", "average_precision",
                                          "average_precision_ci95", "roc_auc")}
            sysd["at_budget"] = {b: {**confusion_metrics(*(r["at_alert_budget"][f"{b}_per_day"][k] for k in ("true_positives", "false_positives", "false_negatives", "true_negatives"))),
                                     "score_cutoff": r["at_alert_budget"][f"{b}_per_day"]["score_cutoff"]} for b in (50, 100, 200)}
            sysd["breakdown_100"] = r.get("breakdown_at_100_per_day")
            if "at_threshold" in r:
                t = r["at_threshold"]
                sysd["at_threshold"] = {**confusion_metrics(t["true_positives"], t["false_positives"], t["false_negatives"], t["true_negatives"]),
                                        "alerts_per_day": t["alerts_per_day"]}
            systems[lab] = sysd
        e4 = systems.get("E4")
        bench = _read(root, "reports/evaluation/benchmark_rows.json") or {"rows": []}
        op = next((r for r in bench["rows"] if r["evaluation_partition"] == "sealed_final_test" and "active champion" in r["system_name"]), None)
        if e4 and op:      # the deployed fixed threshold (P1.6 benchmark row, same sealed run)
            e4["at_threshold"] = {**confusion_metrics(op["true_positives"], op["false_positives"], op["false_negatives"], op["true_negatives"]),
                                  "alerts_per_day": op["alerts_per_day"], "operating_point": op["operating_point"]}
        if e4:
            n, pos = e4["rows"], e4["labelled_fraud"]
            systems["dummy"] = {"rows": n, "labelled_fraud": pos, "average_precision": round(pos / n, 4), "roc_auc": 0.5,
                                "at_threshold": {**confusion_metrics(0, 0, pos, n - pos), "alerts_per_day": 0.0}}
            rh = p15["rule_engine_high"]
            systems["rules_high"] = {"rows": n, "labelled_fraud": pos, "average_precision": None, "roc_auc": None,
                                     "at_threshold": {**confusion_metrics(rh["true_positives"], rh["false_positives"], rh["false_negatives"],
                                                                          n - pos - rh["false_positives"]), "alerts_per_day": rh["alerts_per_day"]}}
        rule_text = (root / "reports/final_test/p15_final_test.md").read_text(encoding="utf-8")
        rule = next((l.split("Rule (pre-registered):", 1)[1].strip() for l in rule_text.splitlines() if "Rule (pre-registered):" in l), None)
        out["sealed"] = {"systems": systems, "paired_ap_difference_E4_minus_E0": p15["paired_ap_difference_E4_minus_E0"],
                         "decision": p15["decision"], "rule": rule,
                         "meta": {k: p15["meta"].get(k) for k in ("generated_utc", "head", "prereg_commit", "data_sha256")}}
    shap = root / "reports/explainability/global_shap_importance.csv"
    if shap.exists():
        lines = shap.read_text(encoding="utf-8").strip().splitlines()[1:]
        out["shap"] = [{"feature": f, "mean_abs_shap": float(m), "share_pct": float(s)} for _, f, m, s in (l.split(",") for l in lines)]
        out["shap_meta"] = {k: v for k, v in (_read(root, "reports/explainability/explainability_metadata.json") or {}).items()
                            if k in ("sample_size", "sample_fraud_count", "base_value_probability", "max_reconstruction_error")}
    curves = _read(root, "reports/evaluation/dev_curves.json")
    if curves:
        out["curves"] = curves
    out["model_card"] = model_card(root, registry)
    return out


def write_bundle(data: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "data.js"
    payload = json.dumps(data, separators=(",", ":"), default=_plain)
    path.write_text("// Generated by src/finsight/dashboard.py — do not edit by hand.\nwindow.FINSIGHT_DATA = " + payload + ";\n",
                    encoding="utf-8")
    return path


def main() -> None:
    from finsight.champion import read_registry
    from finsight.load_to_postgres import DatabaseConfig
    import psycopg2

    parser = argparse.ArgumentParser(description="Build the FinSight dashboard data bundle (read-only)")
    parser.add_argument("--out", default="dashboard")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    cfg = DatabaseConfig.from_env()
    conn = psycopg2.connect(host=cfg.host, port=cfg.port, dbname=cfg.name, user=cfg.user, password=cfg.password)
    try:
        registry = read_registry()
        data = collect(conn, registry, {**load_extras(), "study": load_study(Path("."), registry)})
    finally:
        conn.close()
    data["meta"] = {"generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
                    "source": f"{cfg.host}:{cfg.port}/{cfg.name}"}
    path = write_bundle(data, Path(args.out))
    logger.info("wrote %s (%d alerts, %.1f MB) from %s", path, len(data["alerts"]), path.stat().st_size / 1e6, data["meta"]["source"])


if __name__ == "__main__":
    main()
