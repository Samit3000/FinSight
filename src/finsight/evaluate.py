"""
FinSight — Locked Evaluation Protocol (P0).

Single source of truth for how FinSight models are evaluated.

Protocol
--------
* Development data: steps 1–650. All training, tuning, threshold and feature decisions
  use development data only.
* Locked final test: steps 651–743 (80,831 transactions, 1,022 labelled fraud).
  It may only be scored once per promoted model, through `score_locked_final_test`,
  which requires an explicit acknowledgement. Every tuning helper in this module
  refuses rows from the locked period.
  Disclosure: this window was previously visible inside the old holdout (steps >= 551)
  in aggregate analyses and in the EXP-07 gate, so it is not pristine.
* Walk-forward folds (expanding window, train strictly before test):
      F1: train <= 350 -> test 351–450
      F2: train <= 450 -> test 451–550
      F3: train <= 550 -> test 551–650
* Models are compared at a fixed review capacity (alerts per simulated day), not at
  F1-optimal thresholds, because PaySim fraud prevalence rises ~13x across the
  simulation (legitimate volume collapses after day 17 while fraud stays ~260/day).
  Precision and F1 are therefore only comparable within a fold.
* Uncertainty: step-block bootstrap (resampling whole simulation hours) for AP and
  for paired AP differences between models.

Model scores from class-weighted training are risk scores, not calibrated probabilities.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

logger = logging.getLogger("finsight.evaluate")

STEPS_PER_DAY: int = 24
FINAL_TEST_MIN_STEP: int = 651
DEFAULT_ALERT_BUDGETS_PER_DAY: tuple[int, ...] = (25, 50, 100, 200)
DEFAULT_BOOTSTRAP_ITERATIONS: int = 300


class LockedTestAccessError(RuntimeError):
    """Raised when development-only code receives rows from the locked final test."""


@dataclass(frozen=True)
class Fold:
    name: str
    train_max_step: int
    test_min_step: int
    test_max_step: int

    def __post_init__(self) -> None:
        if not self.train_max_step < self.test_min_step <= self.test_max_step:
            raise ValueError(f"Fold {self.name}: train must end before test starts.")
        if self.test_max_step >= FINAL_TEST_MIN_STEP:
            raise LockedTestAccessError(
                f"Fold {self.name} reaches step {self.test_max_step}; folds must end before {FINAL_TEST_MIN_STEP}."
            )


WALK_FORWARD_FOLDS: tuple[Fold, ...] = (
    Fold("F1", train_max_step=350, test_min_step=351, test_max_step=450),
    Fold("F2", train_max_step=450, test_min_step=451, test_max_step=550),
    Fold("F3", train_max_step=550, test_min_step=551, test_max_step=650),
)


# ------------------------------------------------------------------------------
# Guards
# ------------------------------------------------------------------------------
def assert_development_steps(steps: Sequence[int] | np.ndarray | pd.Series, context: str = "") -> None:
    """Raise if any step belongs to the locked final test period."""
    arr = np.asarray(steps)
    if arr.size and int(arr.max()) >= FINAL_TEST_MIN_STEP:
        n = int((arr >= FINAL_TEST_MIN_STEP).sum())
        raise LockedTestAccessError(
            f"{context or 'Development-only operation'} received {n} rows from the locked final test "
            f"(step >= {FINAL_TEST_MIN_STEP})."
        )


def development_frame(df: pd.DataFrame, step_col: str = "step") -> pd.DataFrame:
    """Return only development rows (step < FINAL_TEST_MIN_STEP)."""
    return df[df[step_col] < FINAL_TEST_MIN_STEP]


def fold_masks(steps: pd.Series, fold: Fold) -> tuple[pd.Series, pd.Series]:
    train = steps <= fold.train_max_step
    test = (steps >= fold.test_min_step) & (steps <= fold.test_max_step)
    return train, test


# ------------------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------------------
def simulated_days(steps: Sequence[int] | np.ndarray) -> float:
    """Length of the evaluated period in simulated days (distinct hourly steps / 24)."""
    return len(np.unique(np.asarray(steps))) / STEPS_PER_DAY


def confusion_at_threshold(y: np.ndarray, score: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = score >= threshold
    return _confusion(y, pred)


def _confusion(y: np.ndarray, pred: np.ndarray) -> Dict[str, Any]:
    y = np.asarray(y).astype(int)
    pred = np.asarray(pred).astype(bool)
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    alerts = tp + fp
    precision = tp / alerts if alerts else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "alerts": alerts, "true_positives": tp, "false_positives": fp,
        "false_negatives": fn, "true_negatives": tn,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4),
    }


def metrics_at_alert_budget(
    y: np.ndarray, score: np.ndarray, steps: np.ndarray, alerts_per_day: float
) -> Dict[str, Any]:
    """Review exactly K alerts, K = alerts_per_day x simulated days in the period.

    Tie handling: tree models emit few distinct scores, so many rows can share the
    K-th score. Every row scoring strictly above the cut-off is reviewed; the remaining
    review slots are filled from the rows tied at the cut-off, and their outcome is the
    expected value under random tie-breaking (tied fraud share x slots). Counts can
    therefore be fractional. This removes any dependence on row order. The cut-off is
    retrospective over the whole period (a ranking measure, not an online daily cap).
    """
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    days = simulated_days(steps)
    k = int(round(alerts_per_day * days))
    k = max(0, min(k, len(score)))
    positives = int(y.sum())
    negatives = int(len(y) - positives)
    out: Dict[str, Any] = {"alerts_per_day_budget": alerts_per_day, "alerts": k}
    if k == 0:
        tp = fp = 0.0
        out.update({"score_cutoff": None, "rows_tied_at_cutoff": 0, "tied_rows_reviewed": 0})
    else:
        cutoff = float(np.sort(score)[::-1][k - 1])
        above = score > cutoff
        tied = score == cutoff
        n_above, n_tied = int(above.sum()), int(tied.sum())
        slots = k - n_above
        tied_fraud = int(y[tied].sum())
        tp = float(y[above].sum()) + slots * tied_fraud / n_tied
        fp = float(n_above - y[above].sum()) + slots * (n_tied - tied_fraud) / n_tied
        out.update({"score_cutoff": round(cutoff, 6), "rows_tied_at_cutoff": n_tied, "tied_rows_reviewed": slots})
    fn = positives - tp
    tn = negatives - fp
    precision = tp / k if k else 0.0
    recall = tp / positives if positives else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out.update({
        "tie_policy": "expected value under random tie-breaking",
        "true_positives": round(tp, 2), "false_positives": round(fp, 2),
        "false_negatives": round(fn, 2), "true_negatives": round(tn, 2),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4),
    })
    return out


def _step_block_indices(steps: np.ndarray) -> List[np.ndarray]:
    _, inverse = np.unique(steps, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    bounds = np.flatnonzero(np.diff(inverse[order])) + 1
    return np.split(order, bounds)


def bootstrap_ap_ci(
    y: np.ndarray, score: np.ndarray, steps: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS, seed: int = 0,
) -> List[float]:
    """95% step-block bootstrap interval for average precision."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    blocks = _step_block_indices(np.asarray(steps))
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if y[idx].sum() == 0:
            continue
        stats.append(average_precision_score(y[idx], score[idx]))
    return [round(float(v), 4) for v in np.percentile(stats, [2.5, 97.5])]


def bootstrap_ap_difference(
    y: np.ndarray, score_a: np.ndarray, score_b: np.ndarray, steps: np.ndarray,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS, seed: int = 0,
) -> Dict[str, Any]:
    """Paired step-block bootstrap of AP(a) - AP(b) on the same rows."""
    y = np.asarray(y).astype(int)
    a = np.asarray(score_a, dtype=float)
    b = np.asarray(score_b, dtype=float)
    blocks = _step_block_indices(np.asarray(steps))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if y[idx].sum() == 0:
            continue
        diffs.append(average_precision_score(y[idx], a[idx]) - average_precision_score(y[idx], b[idx]))
    diffs = np.asarray(diffs)
    return {
        "point": round(float(average_precision_score(y, a) - average_precision_score(y, b)), 4),
        "ci95": [round(float(v), 4) for v in np.percentile(diffs, [2.5, 97.5])],
        "share_of_resamples_not_better": round(float((diffs <= 0).mean()), 4),
    }


def evaluate_scores(
    y: np.ndarray, score: np.ndarray, steps: np.ndarray,
    alert_budgets_per_day: Sequence[float] = DEFAULT_ALERT_BUDGETS_PER_DAY,
    threshold: Optional[float] = None,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> Dict[str, Any]:
    """Full metric block for one scored period (development rows only)."""
    assert_development_steps(steps, "evaluate_scores")
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=float)
    steps = np.asarray(steps)
    days = simulated_days(steps)
    out: Dict[str, Any] = {
        "rows": int(len(y)),
        "labelled_fraud": int(y.sum()),
        "fraud_prevalence_pct": round(float(y.mean() * 100), 4),
        "steps": [int(steps.min()), int(steps.max())],
        "simulated_days": round(days, 3),
        "average_precision": round(float(average_precision_score(y, score)), 4),
        "average_precision_ci95": bootstrap_ap_ci(y, score, steps, n_boot=n_boot),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "mean_score": round(float(score.mean()), 4),
        "at_alert_budget": {
            f"{b:g}_per_day": metrics_at_alert_budget(y, score, steps, b) for b in alert_budgets_per_day
        },
    }
    if threshold is not None:
        cm = confusion_at_threshold(y, score, threshold)
        cm["threshold"] = threshold
        cm["alerts_per_day"] = round(cm["alerts"] / days, 1)
        out["at_threshold"] = cm
    return out


# ------------------------------------------------------------------------------
# Walk-forward driver
# ------------------------------------------------------------------------------
FitFn = Callable[[pd.DataFrame, np.ndarray], Any]
ScoreFn = Callable[[Any, pd.DataFrame], np.ndarray]


def run_walk_forward(
    df: pd.DataFrame,
    fit_fn: FitFn,
    score_fn: ScoreFn,
    folds: Sequence[Fold] = WALK_FORWARD_FOLDS,
    target_col: str = "isFraud",
    step_col: str = "step",
    alert_budgets_per_day: Sequence[float] = DEFAULT_ALERT_BUDGETS_PER_DAY,
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> Dict[str, Any]:
    """Fit on each fold's training window, score its test window, return per-fold metrics.

    `fit_fn(train_df, y_train) -> model` and `score_fn(model, df) -> scores`.
    Locked-test rows are removed before anything else happens.
    """
    dev = development_frame(df, step_col)
    results: Dict[str, Any] = {"folds": {}, "fold_definitions": [asdict(f) for f in folds]}
    scores_by_fold: Dict[str, np.ndarray] = {}
    for fold in folds:
        train_mask, test_mask = fold_masks(dev[step_col], fold)
        train_df, test_df = dev[train_mask], dev[test_mask]
        assert train_df[step_col].max() < test_df[step_col].min()
        t0 = time.time()
        model = fit_fn(train_df, train_df[target_col].to_numpy(dtype=int))
        fit_s = time.time() - t0
        scores = np.asarray(score_fn(model, test_df), dtype=float)
        block = evaluate_scores(
            test_df[target_col].to_numpy(dtype=int), scores, test_df[step_col].to_numpy(),
            alert_budgets_per_day=alert_budgets_per_day, n_boot=n_boot,
        )
        block["train_rows"] = int(len(train_df))
        block["train_fraud"] = int(train_df[target_col].sum())
        block["fit_seconds"] = round(fit_s, 1)
        results["folds"][fold.name] = block
        scores_by_fold[fold.name] = scores
        logger.info("%s: AP=%.4f (CI %s), prevalence=%.4f%%", fold.name, block["average_precision"],
                    block["average_precision_ci95"], block["fraud_prevalence_pct"])
    aps = [results["folds"][f.name]["average_precision"] for f in folds]
    results["summary"] = {
        "average_precision_by_fold": dict(zip([f.name for f in folds], aps)),
        "average_precision_min": min(aps),
        "average_precision_mean": round(float(np.mean(aps)), 4),
    }
    results["_scores"] = scores_by_fold
    return results


def compare_on_folds(
    df: pd.DataFrame, scores_a: Mapping[str, np.ndarray], scores_b: Mapping[str, np.ndarray],
    folds: Sequence[Fold] = WALK_FORWARD_FOLDS, target_col: str = "isFraud", step_col: str = "step",
    n_boot: int = DEFAULT_BOOTSTRAP_ITERATIONS,
) -> Dict[str, Any]:
    """Paired AP-difference (a - b) per fold for two models scored by run_walk_forward."""
    dev = development_frame(df, step_col)
    out = {}
    for fold in folds:
        _, test_mask = fold_masks(dev[step_col], fold)
        t = dev[test_mask]
        out[fold.name] = bootstrap_ap_difference(
            t[target_col].to_numpy(dtype=int), scores_a[fold.name], scores_b[fold.name],
            t[step_col].to_numpy(), n_boot=n_boot,
        )
    return out


# ------------------------------------------------------------------------------
# Locked final test (deliberately awkward to call)
# ------------------------------------------------------------------------------
def score_locked_final_test(
    df: pd.DataFrame, score_fn: Callable[[pd.DataFrame], np.ndarray], *,
    i_understand_this_is_the_single_final_evaluation: bool = False,
    target_col: str = "isFraud", step_col: str = "step",
    alert_budgets_per_day: Sequence[float] = DEFAULT_ALERT_BUDGETS_PER_DAY,
    threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """Score the locked final test once for a model that has already been promoted."""
    if not i_understand_this_is_the_single_final_evaluation:
        raise LockedTestAccessError(
            "The locked final test may only be scored once per promoted model. "
            "Pass i_understand_this_is_the_single_final_evaluation=True deliberately."
        )
    test = df[df[step_col] >= FINAL_TEST_MIN_STEP]
    y = test[target_col].to_numpy(dtype=int)
    score = np.asarray(score_fn(test), dtype=float)
    steps = test[step_col].to_numpy()
    days = simulated_days(steps)
    out = {
        "rows": int(len(y)), "labelled_fraud": int(y.sum()),
        "fraud_prevalence_pct": round(float(y.mean() * 100), 4),
        "simulated_days": round(days, 3),
        "average_precision": round(float(average_precision_score(y, score)), 4),
        "roc_auc": round(float(roc_auc_score(y, score)), 4),
        "at_alert_budget": {f"{b:g}_per_day": metrics_at_alert_budget(y, score, steps, b) for b in alert_budgets_per_day},
    }
    if threshold is not None:
        cm = confusion_at_threshold(y, score, threshold)
        cm["alerts_per_day"] = round(cm["alerts"] / days, 1)
        out["at_threshold"] = cm
    return out


# ------------------------------------------------------------------------------
# Report rendering
# ------------------------------------------------------------------------------
def render_walk_forward_markdown(title: str, results: Dict[str, Any], meta: Dict[str, Any]) -> str:
    lines = [
        f"# {title}",
        "",
        "> Generated by `src/finsight/evaluate.py`. Do not edit by hand.",
        "",
        f"- Generated (UTC): {meta.get('generated_utc')}",
        f"- Model config: `{meta.get('config_name')}`",
        f"- Data file SHA-256: `{meta.get('data_sha256')}`",
        f"- Locked final test: steps >= {FINAL_TEST_MIN_STEP} — **not scored** in this report.",
        "- Scores are risk scores from class-weighted training, not calibrated probabilities.",
        "- Prevalence differs by fold, so precision/F1 are only comparable within a fold.",
        "- Confidence intervals: 95% step-block bootstrap (whole simulated hours resampled). They cover "
        "test-sample variation only, not training variation, and treat hours as exchangeable "
        "(the day/night pattern is not modelled).",
        "",
        "## Selection disclosure",
        "",
        meta.get("selection_history") or "No selection history recorded for this configuration.",
        "",
        "## Per-fold results",
        "",
        "| Fold | Train steps | Test steps | Test rows | Fraud | Prevalence | Days | AP | AP 95% CI | ROC-AUC |",
        "|---|---|---|---:|---:|---:|---:|---:|---|---:|",
    ]
    defs = {d["name"]: d for d in results["fold_definitions"]}
    for name, b in results["folds"].items():
        d = defs[name]
        lines.append(
            f"| {name} | 1–{d['train_max_step']} | {d['test_min_step']}–{d['test_max_step']} | {b['rows']:,} | "
            f"{b['labelled_fraud']:,} | {b['fraud_prevalence_pct']:.4f}% | {b['simulated_days']:.2f} | "
            f"{b['average_precision']:.4f} | {b['average_precision_ci95'][0]:.4f}–{b['average_precision_ci95'][1]:.4f} | "
            f"{b['roc_auc']:.4f} |"
        )
    lines += ["", "## Fixed review capacity (top-K by score, K = alerts/day × days)", "",
              "Rows tied at the cut-off score are counted at their expected value under random tie-breaking, "
              "so caught / false-alarm / missed counts can be fractional. The cut-off is retrospective over the "
              "whole fold (a ranking measure, not an online daily cap).", "",
              "| Fold | Alerts/day | Alerts | Caught | False alarms | Missed | Precision | Recall | Rows tied at cut-off |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, b in results["folds"].items():
        for key, m in b["at_alert_budget"].items():
            lines.append(
                f"| {name} | {m['alerts_per_day_budget']:g} | {m['alerts']:,} | {m['true_positives']:,.1f} | "
                f"{m['false_positives']:,.1f} | {m['false_negatives']:,.1f} | {m['precision']:.4f} | {m['recall']:.4f} | "
                f"{m['rows_tied_at_cutoff']:,} |"
            )
    s = results["summary"]
    lines += ["", "## Summary", "",
              f"- AP by fold: {s['average_precision_by_fold']}",
              f"- Worst-fold AP: {s['average_precision_min']:.4f}; mean AP: {s['average_precision_mean']:.4f}", ""]
    return "\n".join(lines)


def main() -> None:
    from finsight.train import MODEL_CONFIGS, build_features, fit_model, file_sha256, load_development_data

    parser = argparse.ArgumentParser(description="FinSight walk-forward evaluation (development data only)")
    parser.add_argument("--config", default="hgb_baseline8_balanced", choices=sorted(MODEL_CONFIGS))
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--reports-dir", default="reports/evaluation")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    cfg = MODEL_CONFIGS[args.config]
    df = load_development_data(Path(args.data_path), cfg)  # history computed before fold slicing
    results = run_walk_forward(
        df,
        fit_fn=lambda tr, y: fit_model(cfg, build_features(tr, cfg), y),
        score_fn=lambda m, te: m.predict_proba(build_features(te, cfg))[:, 1],
        n_boot=args.n_boot,
    )
    results.pop("_scores")
    meta = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "config_name": args.config,
        "config": cfg,
        "selection_history": cfg.get("selection_history"),
        "data_sha256": file_sha256(Path(args.data_path)),
        "final_test_min_step": FINAL_TEST_MIN_STEP,
    }
    out_dir = Path(args.reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"walk_forward_{args.config}"
    (out_dir / f"{stem}.json").write_text(json.dumps({"meta": meta, **results}, indent=2, default=str), encoding="utf-8")
    (out_dir / f"{stem}.md").write_text(
        render_walk_forward_markdown(f"Walk-forward evaluation — {args.config}", results, meta), encoding="utf-8"
    )
    logger.info("Wrote %s/%s.{json,md}", out_dir, stem)


if __name__ == "__main__":
    main()
