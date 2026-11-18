"""
FinSight — P1.4.2 priority review queue with deadlines (simulation, development data only).

Real-time design evaluated here (the policy P1.4.1 recommended investigating):

* Every transaction is scored when it arrives. Transactions scoring at or above a *floor*
  threshold enter a review queue; nothing else is ever reviewed.
* Analysts review a fixed number of items per hour (the daily budget K spread evenly over
  24 hours: hourly pacing, no midnight reset). Each hour they take the highest-scoring open
  items, oldest first among equal scores.
* An item that has not been reviewed within its deadline (6, 12 or 24 hours) expires
  unreviewed.
* Optional escalation: items at or above an escalation threshold skip the queue and go to an
  automatic hold (step-up / hold decision). Holds use no analyst capacity but cost customer
  friction, so they are reported separately and are not part of the selection rule.

Time resolution is the hourly `step`. An item arriving in step s can be reviewed in the
review rounds at the end of steps s .. s + deadline - 1. Order within a step is unknown, so
items with an identical (score, arrival step) key share capacity in proportion to their
remaining mass (expected values), mirroring `evaluate.metrics_at_alert_budget`.

Stages (as in P1.3/P1.4): ``--stage selection`` (F1, F2) applies SELECTION_RULE and records the
choice; ``--stage confirmation`` reads it and reports F3 once. Steps >= 651 are never read and
no model artifact is written.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.alert_policy import CONFIG_NAME, _fold, is_night, policy_metrics, policy_windows, threshold_for_budget  # noqa: E402
from finsight.compare_candidates import selection_weights  # noqa: E402
from finsight.evaluate import FINAL_TEST_MIN_STEP, STEPS_PER_DAY, assert_development_steps, simulated_days  # noqa: E402
from finsight.train import MODEL_CONFIGS, build_features, file_sha256, fit_model, load_development_data  # noqa: E402

logger = logging.getLogger("finsight.review_queue")

BUDGETS: Tuple[int, ...] = (50, 100, 200)
PRIMARY_BUDGET = 100
DEADLINES: Tuple[int, ...] = (6, 12, 24)
FLOOR_MULTIPLE = 4            # queue floor: calibration threshold for FLOOR_MULTIPLE x K alerts/day
ESCALATION_PER_DAY = 10       # escalation threshold: calibration threshold for 10 holds/day
WARMUP_STEPS = 24             # queue is warmed up on the last 24 calibration steps (scores only)
RECALL_TOLERANCE = 0.01
LOAD_CV_LIMIT = 0.05
SELECTION_FOLDS: Tuple[str, ...] = ("F1", "F2")
CONFIRMATION_FOLDS: Tuple[str, ...] = ("F3",)

SELECTION_RULE = (
    f"At {PRIMARY_BUDGET} reviews/day, among the queues Q6, Q12 and Q24 without escalation, choose the shortest "
    f"deadline whose recall is within {RECALL_TOLERANCE:.2f} of the best queue's recall in BOTH F1 and F2. "
    "The chosen queue replaces fixed threshold B as the reference policy only if, in BOTH F1 and F2, its recall "
    f">= the first-come capped policy D2's recall and its whole-day review load has CV <= {LOAD_CV_LIMIT:.2f}; "
    "otherwise B stays the reference."
)
DISCLOSURE = (
    "No fold is independent. F1 and F2 were used for feature discovery (P1.1), candidate screening (P1.3), policy "
    "selection (P1.4) and the P1.4.1 investigation; F3 was used for P1.3 and P1.4 confirmation and in P1.4.1, so this "
    "is its fourth use. The queue design itself was proposed after seeing P1.4.1 results on F1-F3. Models are the "
    "P1.4 fold models (E4 trained on steps <= T-100); floor and escalation thresholds come from the calibration "
    "window (T-99..T). The locked test (651-743) is not read. PaySim has no settlement timing, so the cost of a "
    "review delay cannot be measured: a fraud reviewed 20 hours late counts as detected here."
)


# ------------------------------------------------------------------------------
# Policy mechanics
# ------------------------------------------------------------------------------
def hourly_capacities(per_day: int) -> np.ndarray:
    """Integer reviews per hour (index = hour of day), summing exactly to `per_day`."""
    edges = np.floor(np.arange(STEPS_PER_DAY + 1) * per_day / STEPS_PER_DAY).astype(int)
    return np.diff(edges)


def simulate_queue(
    score: np.ndarray, steps: np.ndarray, per_day: int, deadline: int, floor: float,
    escalate_at: Optional[float] = None, warmup_score: Optional[np.ndarray] = None,
    warmup_steps: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Simulate the queue; return per-row arrays for the scored rows (warm-up rows excluded).

    reviewed: expected reviewed mass in [0, 1]; delay_hours: mass-weighted delay (divide by reviewed);
    held: escalated to automatic hold; expired: mass that hit its deadline; pending: mass still queued
    when the window ends; below_floor: never queued; reviews_per_step: reviews used in each round
    (warm-up rounds included).
    """
    score = np.asarray(score, dtype=float)
    steps = np.asarray(steps, dtype=int)
    if deadline < 1:
        raise ValueError("deadline must be at least 1 hour")
    n = len(score)
    ws = np.asarray(warmup_score if warmup_score is not None else [], dtype=float)
    wt = np.asarray(warmup_steps if warmup_steps is not None else [], dtype=int)
    if len(wt) and len(steps) and wt.max() >= steps.min():
        raise ValueError("warm-up rows must precede the scored rows")
    all_score = np.concatenate([ws, score])
    all_steps = np.concatenate([wt, steps])
    row = np.concatenate([np.full(len(ws), -1), np.arange(n)])   # -1 = warm-up row (not reported)
    caps = hourly_capacities(per_day)

    reviewed = np.zeros(n); delay = np.zeros(n); expired = np.zeros(n); pending = np.zeros(n)
    held = np.zeros(n, dtype=bool)
    below_floor = np.zeros(n, dtype=bool)
    by_step: Dict[int, np.ndarray] = {}
    order = np.argsort(all_steps, kind="stable")
    bounds = np.flatnonzero(np.diff(all_steps[order])) + 1
    for grp in np.split(order, bounds):
        by_step[int(all_steps[grp[0]])] = grp

    q_score = np.empty(0); q_arr = np.empty(0, dtype=int); q_mass = np.empty(0); q_row = np.empty(0, dtype=int)
    reviews_per_step: Dict[int, float] = {}
    for s in range(int(all_steps.min()), int(all_steps.max()) + 1):
        grp = by_step.get(s, np.empty(0, dtype=int))
        sc = all_score[grp]
        if escalate_at is not None:
            esc = sc >= escalate_at
            r = row[grp[esc]]
            held[r[r >= 0]] = True
            grp, sc = grp[~esc], sc[~esc]
        ok = sc >= floor
        r = row[grp[~ok]]
        below_floor[r[r >= 0]] = True
        q_score = np.concatenate([q_score, sc[ok]])
        q_arr = np.concatenate([q_arr, np.full(int(ok.sum()), s)])
        q_mass = np.concatenate([q_mass, np.ones(int(ok.sum()))])
        q_row = np.concatenate([q_row, row[grp[ok]]])
        # expire items whose last eligible round (arrival + deadline - 1) has passed
        exp = q_arr <= s - deadline
        if exp.any():
            m = q_row[exp] >= 0
            np.add.at(expired, q_row[exp][m], q_mass[exp][m])
            keep = ~exp
            q_score, q_arr, q_mass, q_row = q_score[keep], q_arr[keep], q_mass[keep], q_row[keep]
        cap = float(caps[(s - 1) % STEPS_PER_DAY])
        if cap > 0 and len(q_mass):
            o = np.lexsort((q_arr, -q_score))     # highest score first, oldest first among ties
            ks, ka = q_score[o], q_arr[o]
            new_key = np.r_[True, (ks[1:] != ks[:-1]) | (ka[1:] != ka[:-1])]
            starts = np.flatnonzero(new_key)
            ends = np.r_[starts[1:], len(o)]
            for a, b in zip(starts, ends):
                members = o[a:b]
                gm = q_mass[members].sum()
                take = min(1.0, cap / gm)
                rv = take * q_mass[members]
                m = q_row[members] >= 0
                np.add.at(reviewed, q_row[members][m], rv[m])
                np.add.at(delay, q_row[members][m], rv[m] * (s - q_arr[members][m]))
                q_mass[members] -= rv
                cap -= take * gm
                if cap <= 1e-12:
                    break
            reviews_per_step[s] = float(caps[(s - 1) % STEPS_PER_DAY]) - max(cap, 0.0)
            keep = q_mass > 1e-12
            q_score, q_arr, q_mass, q_row = q_score[keep], q_arr[keep], q_mass[keep], q_row[keep]
    m = q_row >= 0
    np.add.at(pending, q_row[m], q_mass[m])
    return {"reviewed": reviewed, "delay_hours": delay, "expired": expired, "pending": pending,
            "held": held, "below_floor": below_floor, "reviews_per_step": reviews_per_step}


def first_come_cap(eligible: np.ndarray, steps: np.ndarray, per_day: int) -> np.ndarray:
    """P1.4.1 policy D2: eligible alerts in arrival order until the day's budget is used (midnight reset)."""
    steps = np.asarray(steps, dtype=int)
    day = (steps - 1) // STEPS_PER_DAY + 1
    steps_in_day = {d: len(np.unique(steps[day == d])) for d in np.unique(day)}
    w = np.zeros(len(steps))
    remaining: Dict[int, float] = {}
    order = np.argsort(steps, kind="stable")
    for idx in np.split(order, np.flatnonzero(np.diff(steps[order])) + 1):
        d = int(day[idx[0]])
        remaining.setdefault(d, per_day * steps_in_day[d] / STEPS_PER_DAY)
        el = idx[eligible[idx]]
        if len(el) == 0 or remaining[d] <= 0:
            continue
        take = min(1.0, remaining[d] / len(el))
        w[el] = take
        remaining[d] -= take * len(el)
    return w


def daily_top_k(score: np.ndarray, steps: np.ndarray, per_day: int) -> np.ndarray:
    """P1.4.1 policy D1: each day's top-K (pro-rated on partial days); needs the whole day's scores."""
    steps = np.asarray(steps, dtype=int)
    day = (steps - 1) // STEPS_PER_DAY + 1
    w = np.zeros(len(score))
    for d in np.unique(day):
        m = day == d
        w[m] = selection_weights(score[m], int(round(per_day * len(np.unique(steps[m])) / STEPS_PER_DAY)))
    return w


def load_cv(per_day: Dict[int, float], steps: np.ndarray) -> Optional[float]:
    """Coefficient of variation of review counts over whole days (24 steps) inside the window."""
    steps = np.unique(np.asarray(steps, dtype=int))
    day = (steps - 1) // STEPS_PER_DAY + 1
    full = [per_day[int(d)] for d in np.unique(day) if (day == d).sum() == STEPS_PER_DAY and int(d) in per_day]
    arr = np.asarray(full, dtype=float)
    return round(float(arr.std() / arr.mean()), 3) if len(arr) and arr.mean() > 0 else None


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> Optional[float]:
    m = weights > 0
    if not m.any():
        return None
    v, w = values[m], weights[m]
    o = np.argsort(v)
    c = np.cumsum(w[o]) / w.sum()
    return float(v[o][np.searchsorted(c, q)])


def queue_diagnostics(sim: Dict[str, np.ndarray], y: np.ndarray, steps: np.ndarray) -> Dict[str, Any]:
    rv, dl = sim["reviewed"], sim["delay_hours"]
    per_item_delay = np.divide(dl, rv, out=np.zeros_like(dl), where=rv > 0)
    days = simulated_days(steps)
    held = sim["held"].astype(float)
    return {
        "fraud_expired": round(float((sim["expired"] * y).sum()), 1),
        "fraud_pending_at_window_end": round(float((sim["pending"] * y).sum()), 1),
        "fraud_below_floor": int((sim["below_floor"] & (y == 1)).sum()),
        "items_expired": round(float(sim["expired"].sum()), 1),
        "fraud_delay_hours_median": _weighted_quantile(per_item_delay, rv * y, 0.5),
        "fraud_delay_hours_p90": _weighted_quantile(per_item_delay, rv * y, 0.9),
        "fraud_reviewed_within_1h_share": round(float((rv * y * (per_item_delay < 1)).sum() / max((rv * y).sum(), 1e-9)), 4),
        "holds_per_day": round(float(held.sum() / days), 1),
        "false_holds_per_day": round(float((held * (1 - y)).sum() / days), 2),
        "fraud_held": int((held * y).sum()),
    }


# ------------------------------------------------------------------------------
# Fold run
# ------------------------------------------------------------------------------
def run_fold(df: pd.DataFrame, fold_name: str, cfg: Dict[str, Any]) -> Dict[str, Any]:
    fold = _fold(fold_name)
    win = policy_windows(fold)
    st = df["step"]
    parts = {k: df[(st >= lo) & (st <= hi)] for k, (lo, hi) in win.items()}
    for k, p in parts.items():
        assert_development_steps(p["step"], f"{fold_name} {k}")
    t0 = time.time()
    model = fit_model(cfg, build_features(parts["train"], cfg), parts["train"]["isFraud"].to_numpy(dtype=int))
    ca, te = parts["calibration"], parts["test"]
    s_cal = model.predict_proba(build_features(ca, cfg))[:, 1]
    s = model.predict_proba(build_features(te, cfg))[:, 1]
    fit_s = round(time.time() - t0, 1)
    cal_steps = ca["step"].to_numpy()
    y, steps = te["isFraud"].to_numpy(dtype=int), te["step"].to_numpy()
    hours, amount = te["transaction_hour"].to_numpy(), te["amount"].to_numpy()
    warm = cal_steps > cal_steps.max() - WARMUP_STEPS
    cal_days = simulated_days(cal_steps)
    t_esc = threshold_for_budget(s_cal, int(round(ESCALATION_PER_DAY * cal_days)))
    out: Dict[str, Any] = {"windows": win, "fit_seconds": fit_s, "test_rows": int(len(te)),
                           "test_fraud": int(y.sum()), "escalation_threshold": t_esc, "budgets": {}}
    for K in BUDGETS:
        t_b = threshold_for_budget(s_cal, int(round(K * cal_days)))
        floor = threshold_for_budget(s_cal, int(round(FLOOR_MULTIPLE * K * cal_days)))
        pol: Dict[str, Any] = {}
        pol["B"] = policy_metrics((s >= t_b).astype(float), y, steps, hours, amount)
        pol["D1"] = policy_metrics(daily_top_k(s, steps, K), y, steps, hours, amount)
        pol["D2"] = policy_metrics(first_come_cap(s >= t_b, steps, K), y, steps, hours, amount)
        for d in DEADLINES:
            for esc in (False, True):
                sim = simulate_queue(s, steps, K, d, floor, escalate_at=t_esc if esc else None,
                                     warmup_score=s_cal[warm], warmup_steps=cal_steps[warm])
                name = f"Q{d}" + ("+esc" if esc else "")
                m = policy_metrics(sim["reviewed"], y, steps, hours, amount)
                m["queue"] = queue_diagnostics(sim, y, steps)
                if esc:
                    det = np.maximum(sim["reviewed"], sim["held"].astype(float))
                    m["detected_incl_holds"] = policy_metrics(det, y, steps, hours, amount)
                pol[name] = m
        for m in pol.values():
            m["load_cv"] = load_cv(m["daily"]["per_day"], steps)
        out["budgets"][str(K)] = {"threshold_B": t_b, "queue_floor": floor, "policies": pol}
    logger.info("%s done (fit %.0fs)", fold_name, fit_s)
    return out


def apply_selection_rule(folds: Dict[str, Dict[str, Any]], fold_names: Sequence[str] = SELECTION_FOLDS) -> Dict[str, Any]:
    names = [f"Q{d}" for d in DEADLINES]
    rec = {f: {n: folds[f]["budgets"][str(PRIMARY_BUDGET)]["policies"][n]["recall"] for n in names} for f in fold_names}
    best = {f: max(rec[f].values()) for f in fold_names}
    near = [n for n in names if all(rec[f][n] >= best[f] - RECALL_TOLERANCE for f in fold_names)]
    chosen = near[0]  # names are ordered by deadline; the best queue always qualifies
    adopt_checks = []
    for f in fold_names:
        P = folds[f]["budgets"][str(PRIMARY_BUDGET)]["policies"]
        cv = P[chosen]["load_cv"]
        adopt_checks.append({"fold": f, "recall": P[chosen]["recall"], "recall_D2": P["D2"]["recall"],
                             "recall_B": P["B"]["recall"], "recall_not_below_D2": P[chosen]["recall"] >= P["D2"]["recall"],
                             "load_cv": cv, "load_stable": cv is not None and cv <= LOAD_CV_LIMIT})
    adopt = all(c["recall_not_below_D2"] and c["load_stable"] for c in adopt_checks)
    return {"rule": SELECTION_RULE, "recall_by_fold": rec, "within_tolerance": near, "chosen_queue": chosen,
            "adoption_checks": adopt_checks, "reference_policy": chosen if adopt else "B"}


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def _f(v: Any, fmt: str = "{:.3f}") -> str:
    return "–" if v is None else fmt.format(v)


def render_markdown(stage: str, out: Dict[str, Any]) -> str:
    title = ("P1.4.2 review queue — selection (F1, F2)" if stage == "selection"
             else "P1.4.2 review queue — confirmation (F3)")
    L = [f"# {title}", "",
         f"Config `{CONFIG_NAME}` (E4) · generated {out['meta']['generated_utc']} · data sha256 "
         f"`{out['meta']['data_sha256'][:12]}…` · locked test (≥{FINAL_TEST_MIN_STEP}) not read.", "",
         "## Disclosure", "", DISCLOSURE, "",
         "Policies: **B** fixed threshold (P1.4 reference, uncapped) · **D1** each day's top-K (needs the whole day; "
         "upper bound) · **D2** B's threshold + first-come daily cap (P1.4.1) · **Qd** priority queue, K/24 reviews per "
         f"hour, deadline d hours, floor = calibration threshold for {FLOOR_MULTIPLE}×K/day · **+esc** adds automatic "
         f"holds above the calibration threshold for {ESCALATION_PER_DAY} holds/day (no analyst capacity used).", ""]
    if stage == "confirmation":
        L += [f"Choice carried over unchanged from the selection report: queue **{out['chosen_queue']}**, reference "
              f"policy **{out['reference_policy']}**. Other policies are shown for reference only; F3 cannot change "
              "the choice.", ""]
    for f, r in out["folds"].items():
        w = r["windows"]
        L += [f"## Fold {f}", "",
              f"Model trained on {w['train'][0]}–{w['train'][1]}; thresholds from {w['calibration'][0]}–"
              f"{w['calibration'][1]}; test {w['test'][0]}–{w['test'][1]} ({r['test_rows']:,} rows, {r['test_fraud']:,} "
              "fraud).", "",
              "| K | Policy | Reviews/day | Whole-day min / median / max | CV | Precision | Recall | F1 | R 00-05 | "
              "R 06-23 | R 09-17 | R <100k | FP share 00-05 |",
              "|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for K, blk in r["budgets"].items():
            for name, m in blk["policies"].items():
                d, sr = m["daily"], m["segment_recall"]
                L.append(f"| {K} | {name} | {m['alerts_per_day']:.1f} | {_f(d['full_day_min'], '{:.0f}')} / "
                         f"{_f(d['full_day_median'], '{:.0f}')} / {_f(d['full_day_max'], '{:.0f}')} | "
                         f"{_f(m.get('load_cv'))} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1_score']:.3f} | "
                         f"{_f(sr['00-05']['recall'])} | {_f(sr['06-23']['recall'])} | {_f(sr['09-17']['recall'])} | "
                         f"{_f(sr['amount<100k']['recall'])} | {_f(m['fp_share_00_05'])} |")
        L += ["", "Queue diagnostics (fraud counts are expected values):", "",
              "| K | Policy | Fraud expired | Fraud pending at end | Fraud below floor | Median / p90 review delay of "
              "fraud (h) | Fraud reviewed within the hour | Holds/day | False holds/day | Fraud held | Recall incl. holds |",
              "|---:|---|---:|---:|---:|---|---:|---:|---:|---:|---:|"]
        for K, blk in r["budgets"].items():
            for name, m in blk["policies"].items():
                q = m.get("queue")
                if not q:
                    continue
                inc = m.get("detected_incl_holds")
                L.append(f"| {K} | {name} | {q['fraud_expired']:.1f} | {q['fraud_pending_at_window_end']:.1f} | "
                         f"{q['fraud_below_floor']} | {_f(q['fraud_delay_hours_median'], '{:.0f}')} / "
                         f"{_f(q['fraud_delay_hours_p90'], '{:.0f}')} | {q['fraud_reviewed_within_1h_share']:.2f} | "
                         f"{q['holds_per_day']:.1f} | {q['false_holds_per_day']:.2f} | {q['fraud_held']} | "
                         f"{_f(inc['recall'] if inc else None)} |")
        L.append("")
    if stage == "selection":
        sel = out["selection"]
        L += ["## Selection", "", f"Rule (declared before F3): {sel['rule']}", "",
              "| Fold | " + " | ".join(sel["recall_by_fold"][next(iter(sel["recall_by_fold"]))]) + " |",
              "|---|" + "---:|" * len(DEADLINES)]
        for f, rr in sel["recall_by_fold"].items():
            L.append(f"| {f} | " + " | ".join(f"{v:.3f}" for v in rr.values()) + " |")
        L += ["", f"Within {RECALL_TOLERANCE:.2f} of the best in both folds: {', '.join(sel['within_tolerance'])}. "
              f"**Chosen queue: {sel['chosen_queue']}.**", "",
              "| Fold | Chosen recall | D2 recall | B recall | Not below D2 | Load CV | Load stable |", "|---|---:|---:|---:|---|---:|---|"]
        for c in sel["adoption_checks"]:
            L.append(f"| {c['fold']} | {c['recall']:.3f} | {c['recall_D2']:.3f} | {c['recall_B']:.3f} | "
                     f"{c['recall_not_below_D2']} | {_f(c['load_cv'])} | {c['load_stable']} |")
        L += ["", f"**Reference policy after P1.4.2: {sel['reference_policy']}.**", ""]
    L += ["## Notes", "",
          "- Review counts and recall are expected values where capacity is split among items with identical score "
          "and arrival hour.",
          "- CV is the standard deviation of whole-day review counts divided by their mean (whole days inside the test "
          "window only).",
          "- Escalation results are shown for the business decision only; they are not part of the selection rule.",
          "- No confidence intervals: this compares operating policies, not models.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1.4.2 priority review queue (development folds only)")
    parser.add_argument("--stage", required=True, choices=["selection", "confirmation"])
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--reports-dir", default="reports/evaluation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out_dir = Path(args.reports_dir)
    sel_path = out_dir / "p142_selection.json"
    if args.stage == "selection":
        folds, target = SELECTION_FOLDS, sel_path
    else:
        if not sel_path.exists():
            raise SystemExit("Run --stage selection first; the queue must be chosen before F3 is evaluated.")
        sel = json.loads(sel_path.read_text(encoding="utf-8"))["selection"]
        folds, target = CONFIRMATION_FOLDS, out_dir / "p142_confirmation.json"
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite a recorded stage.")

    t0 = time.time()
    cfg = MODEL_CONFIGS[CONFIG_NAME]
    df = load_development_data(Path(args.data_path), cfg)
    assert df["step"].max() < FINAL_TEST_MIN_STEP
    out: Dict[str, Any] = {"folds": {}}
    for f in folds:
        out["folds"][f] = run_fold(df, f, cfg)
    out["meta"] = {"stage": args.stage, "config_name": CONFIG_NAME,
                   "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "data_sha256": file_sha256(Path(args.data_path)), "max_step_loaded": int(df["step"].max()),
                   "floor_multiple": FLOOR_MULTIPLE, "escalation_per_day": ESCALATION_PER_DAY,
                   "warmup_steps": WARMUP_STEPS, "disclosure": DISCLOSURE, "runtime_seconds": round(time.time() - t0)}
    if args.stage == "selection":
        out["selection"] = apply_selection_rule(out["folds"])
    else:
        out["chosen_queue"], out["reference_policy"] = sel["chosen_queue"], sel["reference_policy"]
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    target.with_suffix(".md").write_text(render_markdown(args.stage, out), encoding="utf-8")
    logger.info("wrote %s (+ .md) in %.0fs", target, time.time() - t0)


if __name__ == "__main__":
    main()
