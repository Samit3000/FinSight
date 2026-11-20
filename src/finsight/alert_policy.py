"""
FinSight — P1.4 alert policy for the selected configuration (development data only).

Question: can the P1.3 ranking result be run as a prospective policy, and does reserving part
of the daily alert budget for hours 06-23 help?

Per fold, the model (E4 = ``hgb_baseline7_rcpt_cw50``) is trained on steps <= T-100. Thresholds
are fixed on the calibration window (T-99 .. T) and then applied, unchanged and with the same
model, to the fold's test window. Policies, each at 50 / 100 / 200 alerts per simulated day:

* A  top-K over the whole test period (retrospective reference; the P1.3 measure)
* B  one fixed threshold that would have produced K alerts/day in the calibration window
* C_s  fixed thresholds per hour band, with share ``s`` of the budget reserved for hours 06-23

Two recorded stages, as in P1.3: ``--stage selection`` (F1, F2) applies SELECTION_RULE and writes
the chosen policy before F3 is touched; ``--stage confirmation`` reads it and reports F3 once.
Steps >= 651 are never read. No model artifact is written.
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
import pandas as pd

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.compare_candidates import selection_weights  # noqa: E402
from finsight.evaluate import (  # noqa: E402
    FINAL_TEST_MIN_STEP, STEPS_PER_DAY, WALK_FORWARD_FOLDS, Fold, assert_development_steps,
    metrics_at_alert_budget, simulated_days,
)
from finsight.train import MODEL_CONFIGS, build_features, file_sha256, fit_model, load_development_data  # noqa: E402

logger = logging.getLogger("finsight.alert_policy")

CONFIG_NAME = "hgb_baseline7_rcpt_cw50"   # E4, selected by the user after P1.3
CALIBRATION_STEPS = 100
BUDGETS: Tuple[int, ...] = (50, 100, 200)
PRIMARY_BUDGET = 100
DAY_SHARES: Tuple[float, ...] = (0.3, 0.5)
NIGHT_HOURS = (0, 5)                         # band 00-05; everything else is the reserved 06-23 band
VOLUME_TOLERANCE = 1.10
SELECTION_FOLDS: Tuple[str, ...] = ("F1", "F2")
CONFIRMATION_FOLDS: Tuple[str, ...] = ("F3",)
SEGMENTS: Tuple[str, ...] = ("00-05", "06-08", "09-17", "18-23", "06-23", "amount<100k")

SELECTION_RULE = (
    f"At {PRIMARY_BUDGET} alerts/day, policy C_s is adopted over B only if, in BOTH F1 and F2: "
    "(a) total recall of C_s >= B, (b) recall on hours 06-23 of C_s > B, and "
    f"(c) realised alerts of C_s <= {VOLUME_TOLERANCE:.2f} x B's (same review load). "
    "If several shares pass, the one with the higher mean total recall over F1 and F2 is chosen; "
    "if none passes, B is chosen."
)
DISCLOSURE = (
    "Fold models here are trained on steps <= T-100 (T = the fold's training end), so they differ from the "
    "P1.3 fold models and their numbers are not directly comparable. Thresholds are set only on the "
    "calibration window (T-99..T), which lies before the test window. F1 and F2 choose the policy; F3 was "
    "already used for P1.3 confirmation, so this is its second use and it is not independent. The locked "
    "test (651-743) is not read."
)


def _fold(name: str) -> Fold:
    return next(f for f in WALK_FORWARD_FOLDS if f.name == name)


def policy_windows(fold: Fold, calibration_steps: int = CALIBRATION_STEPS) -> Dict[str, Tuple[int, int]]:
    inner_max = fold.train_max_step - calibration_steps
    if inner_max < 1:
        raise ValueError(f"{fold.name}: training window too short for a {calibration_steps}-step calibration window")
    w = {"train": (1, inner_max), "calibration": (inner_max + 1, fold.train_max_step),
         "test": (fold.test_min_step, fold.test_max_step)}
    assert w["train"][1] < w["calibration"][0] <= w["calibration"][1] < w["test"][0]
    return w


def threshold_for_budget(score: np.ndarray, k: int) -> float:
    """Lowest threshold t such that at most the top-k scores (plus ties at t) satisfy score >= t."""
    score = np.asarray(score, dtype=float)
    if k <= 0 or len(score) == 0:
        return float("inf")
    return float(np.sort(score)[::-1][min(k, len(score)) - 1])


def is_night(hours: np.ndarray) -> np.ndarray:
    return (hours >= NIGHT_HOURS[0]) & (hours <= NIGHT_HOURS[1])


def calibrate(score: np.ndarray, steps: np.ndarray, hours: np.ndarray, budget: float) -> Dict[str, Any]:
    """Thresholds for B and every C_s from calibration-window scores only."""
    k = int(round(budget * simulated_days(steps)))
    night = is_night(hours)
    t_b = threshold_for_budget(score, k)
    out: Dict[str, Any] = {"calibration_alerts_target": k, "B": {"threshold": t_b}}
    b_alerts = score >= t_b
    out["B"]["calibration_share_06_23"] = round(float((b_alerts & ~night).sum() / max(b_alerts.sum(), 1)), 4)
    for s in DAY_SHARES:
        k_day = int(round(s * k))
        out[f"C_{s:g}"] = {"threshold_00_05": threshold_for_budget(score[night], k - k_day),
                           "threshold_06_23": threshold_for_budget(score[~night], k_day), "day_share": s}
    return out


def apply_policy(name: str, cal: Dict[str, Any], score: np.ndarray, hours: np.ndarray) -> np.ndarray:
    if name == "B":
        return (score >= cal["B"]["threshold"]).astype(float)
    c = cal[name]
    night = is_night(hours)
    return np.where(night, score >= c["threshold_00_05"], score >= c["threshold_06_23"]).astype(float)


def daily_alerts(w: np.ndarray, steps: np.ndarray) -> Dict[str, Any]:
    day = (steps - 1) // STEPS_PER_DAY + 1
    per_day: Dict[int, float] = {}
    full: List[float] = []
    for d in np.unique(day):
        m = day == d
        per_day[int(d)] = round(float(w[m].sum()), 1)
        if len(np.unique(steps[m])) == STEPS_PER_DAY:
            full.append(per_day[int(d)])
    return {"per_day": per_day, "full_days": len(full),
            "full_day_min": min(full) if full else None,
            "full_day_median": float(np.median(full)) if full else None,
            "full_day_max": max(full) if full else None}


def policy_metrics(w: np.ndarray, y: np.ndarray, steps: np.ndarray, hours: np.ndarray, amount: np.ndarray) -> Dict[str, Any]:
    alerts = float(w.sum())
    tp = float((w * y).sum())
    fp = alerts - tp
    pos = int(y.sum())
    precision = tp / alerts if alerts else 0.0
    recall = tp / pos if pos else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    masks = {
        "00-05": is_night(hours), "06-08": (hours >= 6) & (hours <= 8), "09-17": (hours >= 9) & (hours <= 17),
        "18-23": hours >= 18, "06-23": ~is_night(hours), "amount<100k": amount < 1e5,
    }
    seg = {}
    for k, m in masks.items():
        f = int(y[m].sum())
        seg[k] = {"fraud": f, "recall": round(float((w * y)[m].sum() / f), 4) if f else None}
    return {
        "alerts": round(alerts, 1), "alerts_per_day": round(alerts / simulated_days(steps), 1),
        "true_positives": round(tp, 1), "false_positives": round(fp, 1), "false_negatives": round(pos - tp, 1),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1_score": round(f1, 4),
        "fp_share_00_05": round(float((w * (1 - y))[masks["00-05"]].sum() / fp), 4) if fp else None,
        "segment_recall": seg, "daily": daily_alerts(w, steps),
    }


def run_fold(df: pd.DataFrame, fold_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    fold = _fold(fold_name)
    win = policy_windows(fold)
    st = df["step"]
    parts = {k: df[(st >= lo) & (st <= hi)] for k, (lo, hi) in win.items()}
    for k, p in parts.items():
        assert_development_steps(p["step"], f"{fold_name} {k}")
    t0 = time.time()
    model = fit_model(cfg, build_features(parts["train"], cfg), parts["train"]["isFraud"].to_numpy(dtype=int))
    fit_s = round(time.time() - t0, 1)
    sc = {k: model.predict_proba(build_features(parts[k], cfg))[:, 1] for k in ("calibration", "test")}
    te, ca = parts["test"], parts["calibration"]
    y, steps = te["isFraud"].to_numpy(dtype=int), te["step"].to_numpy()
    hours, amount = te["transaction_hour"].to_numpy(), te["amount"].to_numpy()
    out: Dict[str, Any] = {
        "windows": win, "fit_seconds": fit_s, "train_rows": int(len(parts["train"])),
        "train_fraud": int(parts["train"]["isFraud"].sum()),
        "calibration_rows": int(len(ca)), "calibration_simulated_days": round(simulated_days(ca["step"]), 3),
        "test_rows": int(len(te)), "test_fraud": int(y.sum()), "test_simulated_days": round(simulated_days(steps), 3),
        "budgets": {},
    }
    for b in BUDGETS:
        cal = calibrate(sc["calibration"], ca["step"].to_numpy(), ca["transaction_hour"].to_numpy(), b)
        k = int(round(b * simulated_days(steps)))
        pol = {"A": policy_metrics(selection_weights(sc["test"], k), y, steps, hours, amount)}
        pol["A"]["check_vs_evaluate"] = metrics_at_alert_budget(y, sc["test"], steps, b)["recall"]
        for name in ["B"] + [f"C_{s:g}" for s in DAY_SHARES]:
            pol[name] = policy_metrics(apply_policy(name, cal, sc["test"], hours), y, steps, hours, amount)
        out["budgets"][str(b)] = {"calibration": cal, "policies": pol}
    logger.info("%s done (fit %.0fs)", fold_name, fit_s)
    return out


def apply_selection_rule(folds: Dict[str, Dict[str, Any]], fold_names: Sequence[str] = SELECTION_FOLDS) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    passing = []
    for s in DAY_SHARES:
        name = f"C_{s:g}"
        rows = []
        for f in fold_names:
            P = folds[f]["budgets"][str(PRIMARY_BUDGET)]["policies"]
            c, b = P[name], P["B"]
            rows.append({
                "fold": f,
                "recall_not_lower": c["recall"] >= b["recall"],
                "recall_06_23_higher": c["segment_recall"]["06-23"]["recall"] > b["segment_recall"]["06-23"]["recall"],
                "volume_within_tolerance": c["alerts"] <= VOLUME_TOLERANCE * b["alerts"],
                "recall": c["recall"], "recall_B": b["recall"],
            })
        ok = all(r["recall_not_lower"] and r["recall_06_23_higher"] and r["volume_within_tolerance"] for r in rows)
        checks[name] = {"passes": ok, "checks": rows}
        if ok:
            passing.append((float(np.mean([r["recall"] for r in rows])), name))
    chosen = max(passing)[1] if passing else "B"
    return {"rule": SELECTION_RULE, "checks": checks, "chosen_policy": chosen}


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def _r(v: Any) -> str:
    return "–" if v is None else f"{v:.3f}"


def render_markdown(stage: str, out: Dict[str, Any]) -> str:
    title = "P1.4 alert policy — selection (F1, F2)" if stage == "selection" else "P1.4 alert policy — confirmation (F3)"
    L = [f"# {title}", "",
         f"Config `{CONFIG_NAME}` (E4) · generated {out['meta']['generated_utc']} · data sha256 "
         f"`{out['meta']['data_sha256'][:12]}…` · locked test (≥{FINAL_TEST_MIN_STEP}) not read.", "",
         "## Disclosure", "", DISCLOSURE, "",
         "Policies: **A** top-K over the whole test period (retrospective reference) · **B** one fixed threshold set on "
         "the calibration window · **C_s** fixed thresholds per hour band with share s of the budget reserved for 06-23.", ""]
    if stage == "confirmation":
        L += [f"Chosen policy carried over unchanged from the selection report: **{out['chosen_policy']}**. "
              "The other policies are shown for reference only: they were not chosen on F1/F2, and results on F3 "
              "cannot change the choice.", ""]
    for f, r in out["folds"].items():
        w = r["windows"]
        L += [f"## Fold {f}", "",
              f"Model trained on steps {w['train'][0]}–{w['train'][1]} ({r['train_rows']:,} rows, {r['train_fraud']:,} fraud); "
              f"thresholds set on {w['calibration'][0]}–{w['calibration'][1]} ({r['calibration_rows']:,} rows); "
              f"test {w['test'][0]}–{w['test'][1]} ({r['test_rows']:,} rows, {r['test_fraud']:,} fraud, "
              f"{r['test_simulated_days']:.2f} simulated days).", "",
              "| Budget | Policy | Alerts/day | Full-day alerts min / median / max | TP | FP | FN | Precision | Recall | F1 "
              "| R 00-05 | R 06-08 | R 09-17 | R 18-23 | R 06-23 | R <100k | FP share 00-05 |",
              "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for b, blk in r["budgets"].items():
            for name, m in blk["policies"].items():
                d, sr = m["daily"], m["segment_recall"]
                L.append(
                    f"| {b} | {name} | {m['alerts_per_day']:.1f} | {d['full_day_min']:.0f} / {d['full_day_median']:.0f} / "
                    f"{d['full_day_max']:.0f} | {m['true_positives']:.1f} | {m['false_positives']:.1f} | "
                    f"{m['false_negatives']:.1f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1_score']:.3f} | "
                    + " | ".join(_r(sr[s]["recall"]) for s in SEGMENTS) + f" | {_r(m['fp_share_00_05'])} |")
        L += ["", "Calibration (thresholds and B's share of calibration alerts in 06-23):", "",
              "| Budget | B threshold | B share 06-23 | " + " | ".join(
                  f"C_{s:g} thresholds (00-05 / 06-23)" for s in DAY_SHARES) + " |",
              "|---:|---:|---:|" + "---|" * len(DAY_SHARES)]
        for b, blk in r["budgets"].items():
            c = blk["calibration"]
            L.append(f"| {b} | {c['B']['threshold']:.4f} | {c['B']['calibration_share_06_23']:.2f} | " + " | ".join(
                f"{c[f'C_{s:g}']['threshold_00_05']:.4f} / {c[f'C_{s:g}']['threshold_06_23']:.4f}" for s in DAY_SHARES) + " |")
        L.append("")
    if stage == "selection":
        sel = out["selection"]
        L += ["## Selection", "", f"Rule (declared before F3): {sel['rule']}", "",
              "| Policy | Fold | Recall vs B | Recall not lower | 06-23 recall higher | Alerts within tolerance |",
              "|---|---|---|---|---|---|"]
        for name, c in sel["checks"].items():
            for row in c["checks"]:
                L.append(f"| {name} | {row['fold']} | {row['recall']:.3f} vs {row['recall_B']:.3f} | "
                         f"{row['recall_not_lower']} | {row['recall_06_23_higher']} | {row['volume_within_tolerance']} |")
        L += ["", f"**Chosen policy:** {sel['chosen_policy']}.", ""]
    L += ["## Notes", "",
          "- B and C counts are deterministic (every row at or above its threshold is alerted). A's counts are expected "
          "values under random tie-breaking, as in `evaluate.py`.",
          "- Daily alert counts use whole simulated days only (24 steps inside the test window); partial days at the "
          "window edges are listed in the JSON report.",
          "- No confidence intervals: this stage compares operating policies, not models.",
          "- Scores are risk scores, not calibrated probabilities.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1.4 alert policy (development folds only)")
    parser.add_argument("--stage", required=True, choices=["selection", "confirmation"])
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--reports-dir", default="reports/evaluation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out_dir = Path(args.reports_dir)
    sel_path = out_dir / "p14_selection.json"
    if args.stage == "selection":
        folds, target = SELECTION_FOLDS, sel_path
    else:
        if not sel_path.exists():
            raise SystemExit("Run --stage selection first; the policy must be chosen before F3 is evaluated.")
        chosen = json.loads(sel_path.read_text(encoding="utf-8"))["selection"]["chosen_policy"]
        folds, target = CONFIRMATION_FOLDS, out_dir / "p14_confirmation.json"
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite a recorded stage.")

    t0 = time.time()
    cfg = MODEL_CONFIGS[CONFIG_NAME]
    df = load_development_data(Path(args.data_path), cfg)
    assert df["step"].max() < FINAL_TEST_MIN_STEP
    out: Dict[str, Any] = {"folds": {f: run_fold(df, f, cfg) for f in folds}}
    out["meta"] = {"stage": args.stage, "config_name": CONFIG_NAME, "config": cfg,
                   "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "data_sha256": file_sha256(Path(args.data_path)), "max_step_loaded": int(df["step"].max()),
                   "calibration_steps": CALIBRATION_STEPS, "disclosure": DISCLOSURE,
                   "runtime_seconds": round(time.time() - t0)}
    if args.stage == "selection":
        out["selection"] = apply_selection_rule(out["folds"])
    else:
        out["chosen_policy"] = chosen
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    target.with_suffix(".md").write_text(render_markdown(args.stage, out), encoding="utf-8")
    logger.info("wrote %s (+ .md) in %.0fs", target, time.time() - t0)


if __name__ == "__main__":
    main()
