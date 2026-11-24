"""
FinSight — ROC, precision-recall and reliability curves for E0-E4 on the development folds.

Figures only: no model is selected, promoted, retrained for use, or saved. Each fold refits E0-E4 exactly as
``compare_candidates.run_stage`` did in P1.3 (same configs, same fold masks, same seed), then checks that the
recomputed AP and ROC-AUC equal the recorded P1.3 values to 4 decimal places. If any value differs, nothing is
written.

The sealed steps 651-743 are never read: data goes through ``train.load_development_data`` (which drops them
first) and every fold's test steps pass ``evaluate.assert_development_steps``.

    py src/finsight/dev_curves.py          # writes reports/evaluation/dev_curves.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.compare_candidates import CANDIDATE_SET, _fold  # noqa: E402
from finsight.evaluate import FINAL_TEST_MIN_STEP, assert_development_steps, fold_masks  # noqa: E402
from finsight.train import MODEL_CONFIGS, build_features, file_sha256, fit_model, load_development_data  # noqa: E402

logger = logging.getLogger("finsight.dev_curves")

FOLDS: Tuple[str, ...] = ("F1", "F2", "F3")
RECORDED = {"F1": "reports/evaluation/p13_screening.json", "F2": "reports/evaluation/p13_screening.json",
            "F3": "reports/evaluation/p13_confirmation.json"}
RELIABILITY_BINS = np.array([0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, .99, 1.0000001])


class ReproductionError(RuntimeError):
    """Recomputed metrics do not match the recorded P1.3 results."""


def thin(x: np.ndarray, y: np.ndarray, tol: float = 0.002) -> List[List[float]]:
    """Keep the curve's shape with few points: a point is kept when x or y moved by more than tol since the
    last kept point. First and last points are always kept."""
    keep = [0]
    for i in range(1, len(x) - 1):
        j = keep[-1]
        if abs(x[i] - x[j]) > tol or abs(y[i] - y[j]) > tol:
            keep.append(i)
    keep.append(len(x) - 1)
    return [[round(float(x[i]), 5), round(float(y[i]), 5)] for i in keep]


def curves(y: np.ndarray, score: np.ndarray) -> Dict[str, Any]:
    prec, rec, _ = precision_recall_curve(y, score)
    order = np.argsort(rec, kind="stable")              # recall ascending for drawing
    fpr, tpr, _ = roc_curve(y, score, drop_intermediate=True)
    idx = np.digitize(score, RELIABILITY_BINS) - 1
    rel = []
    for b in range(len(RELIABILITY_BINS) - 1):
        m = idx == b
        if m.any():
            rel.append({"bin": [float(RELIABILITY_BINS[b]), round(float(min(RELIABILITY_BINS[b + 1], 1.0)), 2)],
                        "rows": int(m.sum()), "mean_score": round(float(score[m].mean()), 4),
                        "fraud_rate": round(float(y[m].mean()), 4)})
    return {"average_precision": round(float(average_precision_score(y, score)), 4),
            "roc_auc": round(float(roc_auc_score(y, score)), 4),
            "pr": thin(rec[order], prec[order]), "roc": thin(fpr, tpr), "reliability": rel}


def recorded_metrics(root: Path) -> Dict[str, Dict[str, Tuple[float, float]]]:
    out: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for f, path in RECORDED.items():
        res = json.loads((root / path).read_text(encoding="utf-8"))["results"]
        out[f] = {lab: (res[lab][f]["average_precision"], res[lab][f]["roc_auc"]) for lab in res if f in res[lab]}
    return out


def check_reproduction(computed: Dict[str, Dict[str, Dict[str, Any]]], recorded: Dict[str, Dict[str, Tuple[float, float]]]) -> List[str]:
    problems = []
    for f, labs in recorded.items():
        for lab, (ap, auc) in labs.items():
            c = computed.get(lab, {}).get(f)
            if c is None:
                problems.append(f"{lab} {f}: not computed")
            elif (c["average_precision"], c["roc_auc"]) != (ap, auc):
                problems.append(f"{lab} {f}: AP {c['average_precision']} vs {ap}, AUC {c['roc_auc']} vs {auc}")
    return problems


def run(df, labels: Sequence[str] = tuple(CANDIDATE_SET), folds: Sequence[str] = FOLDS) -> Dict[str, Any]:
    assert_development_steps(df["step"], "dev_curves input")       # refuses any row from steps 651-743
    out: Dict[str, Dict[str, Any]] = {lab: {} for lab in labels}
    meta: Dict[str, Any] = {}
    for fname in folds:
        fold = _fold(fname)
        tr_m, te_m = fold_masks(df["step"], fold)
        train, test = df[tr_m], df[te_m]
        assert_development_steps(test["step"], f"{fname} test")
        y_tr, y_te = train["isFraud"].to_numpy(dtype=int), test["isFraud"].to_numpy(dtype=int)
        meta[fname] = {"train_steps": [1, fold.train_max_step], "test_steps": [fold.test_min_step, fold.test_max_step],
                       "test_rows": int(len(test)), "test_fraud": int(y_te.sum()),
                       "prevalence": round(float(y_te.mean()), 5)}
        for lab in labels:
            cfg = MODEL_CONFIGS[CANDIDATE_SET[lab]]
            t0 = time.time()
            model = fit_model(cfg, build_features(train, cfg), y_tr)
            score = model.predict_proba(build_features(test, cfg))[:, 1]
            out[lab][fname] = curves(y_te, score)
            logger.info("%s %s AP %.4f AUC %.4f (%.0fs)", lab, fname, out[lab][fname]["average_precision"],
                        out[lab][fname]["roc_auc"], time.time() - t0)
    return {"systems": out, "folds": meta}


def main() -> None:
    parser = argparse.ArgumentParser(description="Development-fold ROC / PR / reliability curves (figures only)")
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--out", default="reports/evaluation/dev_curves.json")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    t0 = time.time()
    df = load_development_data(Path(args.data_path), MODEL_CONFIGS[CANDIDATE_SET["E1"]])
    assert df["step"].max() < FINAL_TEST_MIN_STEP
    result = run(df)
    problems = check_reproduction(result["systems"], recorded_metrics(Path(".")))
    if problems:
        raise ReproductionError("recomputed metrics differ from P1.3; nothing written:\n  " + "\n  ".join(problems))
    result["meta"] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_sha256": file_sha256(Path(args.data_path)), "max_step_loaded": int(df["step"].max()),
        "configs": dict(CANDIDATE_SET), "reproduction_check": "AP and ROC-AUC equal recorded P1.3 values to 4 dp",
        "purpose": "figures only; no selection, promotion or saved model", "runtime_seconds": round(time.time() - t0),
    }
    Path(args.out).write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    logger.info("wrote %s (%.0f KB) in %.0fs", args.out, Path(args.out).stat().st_size / 1e3, time.time() - t0)


if __name__ == "__main__":
    main()
