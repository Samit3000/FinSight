"""
FinSight — Reproducible Model Training (P0).

One named configuration -> one model bundle with full provenance. This is the only
supported way to (re)create a FinSight model artifact.

* Trains on steps <= train_max_step (development data only; locked final test refused).
* Selects the decision threshold on the validation window with the configured policy.
* Writes `<out_dir>/<config_name>.joblib` plus `<config_name>.manifest.json`
  (artifact SHA-256, data SHA-256, library versions, metrics).
* Refuses to overwrite an existing file. Default output is `models/candidates/`, so
  promoted artifacts in `models/` are never touched.

The bundle keeps the same keys as `models.save_model_artifact` (model, feature_names,
selected_threshold, model_name, saved_at_utc, metadata), so `models.evaluate_frozen_model`
and `explainability.py` load it unchanged.

Usage:
    py src/finsight/train.py --config hgb_baseline8_balanced \
        --verify-against models/hist_gradient_boosting_no_day.joblib
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.evaluate import FINAL_TEST_MIN_STEP, assert_development_steps, development_frame
from finsight.features import extract_features
from finsight.ml_experiments import BASELINE_8_FEATURES, extract_extended_features
from finsight.models import find_optimal_threshold

logger = logging.getLogger("finsight.train")

_HGB_DEFAULTS: Dict[str, Any] = {
    "max_iter": 50, "learning_rate": 0.1, "max_leaf_nodes": 31,
    "min_samples_leaf": 20, "l2_regularization": 0.0, "random_state": 42,
}

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    # Exact configuration of the frozen champion models/hist_gradient_boosting_no_day.joblib
    "hgb_baseline8_balanced": {
        "model_name": "HistGradientBoosting No Day",
        "family": "hist_gradient_boosting",
        "feature_set": "baseline_8",
        # The frozen champion was trained via models.py, which loads `amount` as float32 and
        # computes log1p in float32. Reproducing it bit-for-bit requires the same input path.
        "amount_dtype": "float32",
        "class_weight": "balanced",
        "params": dict(_HGB_DEFAULTS),
        "train_max_step": 450,
        "threshold_policy": {"name": "validation_f1", "val_min_step": 451, "val_max_step": 550},
        "selection_history": (
            "Not independent of folds F2 and F3. The 8-feature set (dropping transaction_day) was chosen "
            "by comparing AP on the old holdout (steps 551-743, which contains F3's test window 551-650), "
            "and the decision threshold and earlier hyperparameter choices used validation steps 451-550 "
            "(F2's test window). F2 and F3 results are therefore optimistic for this configuration; F1 "
            "(steps 351-450) is the least affected."
        ),
    },
    # EXP-07: candidate nominated on validation; awaiting walk-forward evaluation (P1).
    "hgb_baseline8_cw50": {
        "model_name": "HistGradientBoosting No Day (class weight 1:50)",
        "family": "hist_gradient_boosting",
        "feature_set": "baseline_8",
        # EXP-07 was trained by ml_experiments.py on float64 amounts.
        "amount_dtype": "float64",
        "class_weight": {0: 1.0, 1: 50.0},
        "params": dict(_HGB_DEFAULTS),
        "train_max_step": 450,
        "threshold_policy": {"name": "validation_f1", "val_min_step": 451, "val_max_step": 550},
        "selection_history": (
            "Not independent of folds F2 and F3. The 1:50 class weight was chosen as the best of 14 "
            "experiments on validation steps 451-550 (F2's test window), and the old promotion gate then "
            "compared it on steps 551-743 (containing F3's test window). F2 and F3 results are optimistic "
            "for this configuration; F1 (steps 351-450) is the least affected."
        ),
    },
}


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def build_features(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Build the configured feature matrix, honouring the config's numeric input path."""
    if cfg.get("amount_dtype") == "float32":
        if cfg["feature_set"] != "baseline_8":
            raise ValueError("float32 input path is only defined for baseline_8")
        legacy = df[["step", "type", "amount"]].assign(amount=df["amount"].astype("float32"))
        return extract_features(legacy, include_merchant_flag=False)[list(BASELINE_8_FEATURES)]
    return extract_extended_features(df, cfg["feature_set"])


def fit_model(cfg: Dict[str, Any], X: pd.DataFrame, y: np.ndarray) -> HistGradientBoostingClassifier:
    if cfg["family"] != "hist_gradient_boosting":
        raise ValueError(f"Unsupported family: {cfg['family']}")
    model = HistGradientBoostingClassifier(class_weight=cfg["class_weight"], **cfg["params"])
    model.fit(X, y)
    return model


def train_bundle(
    config_name: str,
    data_path: Path = Path("data/processed/transactions.csv.gz"),
    out_dir: Path = Path("models/candidates"),
    overwrite: bool = False,
) -> Path:
    cfg = MODEL_CONFIGS[config_name]
    out_dir = Path(out_dir)
    artifact = out_dir / f"{config_name}.joblib"
    manifest_path = out_dir / f"{config_name}.manifest.json"
    for p in (artifact, manifest_path):
        if p.exists() and not overwrite:
            raise FileExistsError(f"{p} already exists; refusing to overwrite (pass --overwrite deliberately).")

    df = development_frame(pd.read_csv(data_path, usecols=["step", "type", "amount", "transaction_hour", "isFraud"]))
    train = df[df["step"] <= cfg["train_max_step"]]
    pol = cfg["threshold_policy"]
    val = df[(df["step"] >= pol["val_min_step"]) & (df["step"] <= pol["val_max_step"])]
    assert_development_steps(train["step"], "training")
    assert_development_steps(val["step"], "threshold selection")
    if train["step"].max() >= val["step"].min():
        raise ValueError("Training window must end before the validation window starts.")

    X_train, y_train = build_features(train, cfg), train["isFraud"].to_numpy(dtype=int)
    t0 = time.time()
    model = fit_model(cfg, X_train, y_train)
    fit_seconds = round(time.time() - t0, 1)

    X_val, y_val = build_features(val, cfg), val["isFraud"].to_numpy(dtype=int)
    val_scores = model.predict_proba(X_val)[:, 1]
    if pol["name"] != "validation_f1":
        raise ValueError(f"Unsupported threshold policy: {pol['name']}")
    threshold, val_eval = find_optimal_threshold(y_val, val_scores, objective="f1")

    metadata = {
        "config_name": config_name,
        "config": cfg,
        "feature_names": list(X_train.columns),
        "train_step_range": [int(train["step"].min()), int(train["step"].max())],
        "training_rows": int(len(train)),
        "training_frauds": int(y_train.sum()),
        "validation_step_range": [int(val["step"].min()), int(val["step"].max())],
        "threshold_policy": pol,
        "validation_metrics": {
            "average_precision": round(float(average_precision_score(y_val, val_scores)), 4),
            "roc_auc": round(float(roc_auc_score(y_val, val_scores)), 4),
            **val_eval["metrics"],
            "alerts": val_eval["predicted_positive_count"],
        },
        "final_test_min_step": FINAL_TEST_MIN_STEP,
        "score_semantics": "risk score from class-weighted training; not a calibrated probability",
        "data_path": str(data_path),
        "data_sha256": file_sha256(Path(data_path)),
        "library_versions": {
            "python": platform.python_version(), "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__, "numpy": np.__version__, "joblib": joblib.__version__,
        },
        "fit_seconds": fit_seconds,
    }
    bundle = {
        "model_name": cfg["model_name"],
        "model": model,
        "feature_names": list(X_train.columns),
        "selected_threshold": float(threshold),
        "saved_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "metadata": metadata,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, artifact)
    manifest = {"artifact": artifact.as_posix(), "artifact_sha256": file_sha256(artifact),
                "saved_at_utc": bundle["saved_at_utc"], "selected_threshold": float(threshold), **metadata}
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    logger.info("Saved %s (threshold=%.4f, val AP=%.4f)", artifact, threshold,
                metadata["validation_metrics"]["average_precision"])
    return artifact


def verify_against(candidate: Path, reference: Path, data_path: Path) -> Dict[str, Any]:
    """Compare two bundles on development rows only (no locked-test labels are read)."""
    a, b = joblib.load(candidate), joblib.load(reference)
    cfg = a["metadata"]["config"]
    if a["feature_names"] != b["feature_names"]:
        return {"reproduced": False, "reason": "feature lists differ"}
    df = development_frame(pd.read_csv(data_path, usecols=["step", "type", "amount", "transaction_hour", "isFraud"]))
    X = build_features(df, cfg)[a["feature_names"]]
    sa, sb = a["model"].predict_proba(X)[:, 1], b["model"].predict_proba(X)[:, 1]
    max_diff = float(np.max(np.abs(sa - sb)))
    same_threshold = abs(float(a["selected_threshold"]) - float(b["selected_threshold"])) < 1e-9
    alerts_a = int((sa >= a["selected_threshold"]).sum())
    alerts_b = int((sb >= b["selected_threshold"]).sum())
    return {
        "rows_compared": int(len(df)), "steps_compared": [int(df.step.min()), int(df.step.max())],
        "max_abs_score_difference": max_diff, "same_threshold": same_threshold,
        "alerts_candidate": alerts_a, "alerts_reference": alerts_b,
        "reproduced": bool(max_diff < 1e-6 and same_threshold and alerts_a == alerts_b),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="FinSight reproducible model training")
    parser.add_argument("--config", required=True, choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--out-dir", default="models/candidates")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verify-against", default=None, help="Reference bundle to compare against")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    path = train_bundle(args.config, Path(args.data_path), Path(args.out_dir), overwrite=args.overwrite)
    if args.verify_against:
        result = verify_against(path, Path(args.verify_against), Path(args.data_path))
        print(json.dumps(result, indent=2))
        manifest_path = path.with_suffix(".manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["verification"] = {"reference": args.verify_against, **result}
        manifest_path.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
        if not result["reproduced"]:
            sys.exit(4)


if __name__ == "__main__":
    main()
