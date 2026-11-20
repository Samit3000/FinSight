"""
FinSight — P2 rule-engine backstop study (development data only).

Question: do explainable rules catch fraud that the champion (E4) misses, better than simply giving E4
the same extra review capacity?

* E4 alerts: temporary fold copies of the E4 configuration (trained on steps <= 350 / 450 / 550, as in
  P1.3), alerting on the top 100 transactions per simulated day of each fold window (ties at expected
  value). The champion artifact and registry are never touched.
* Rule alerts: HIGH tier of a rule configuration; "extra" alerts are rule alerts E4 did not raise.
* Comparison: frauds caught by the extra rule alerts vs frauds E4 catches when given the same total
  number of extra review slots (its next-ranked transactions).

Stages (each recorded once, never overwritten): ``--stage selection`` uses F1 and F2 only and applies
SELECTION_RULE; ``--stage confirmation`` reads the recorded selection and evaluates F3 once (the selected
configuration, if any, plus the baseline rules). ``--stage render`` rewrites the markdown from recorded JSON.
Steps >= 651 (the retired sealed test) are never read. Nothing here changes the rule engine's defaults.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.compare_candidates import selection_weights  # noqa: E402
from finsight.evaluate import FINAL_TEST_MIN_STEP, STEPS_PER_DAY, simulated_days  # noqa: E402
from finsight.train import MODEL_CONFIGS, build_features, fit_model, load_development_data  # noqa: E402

logger = logging.getLogger("finsight.rule_tuning")

E4 = "hgb_baseline7_rcpt_cw50"
FOLDS: Dict[str, Tuple[int, int, int]] = {"F1": (350, 351, 450), "F2": (450, 451, 550), "F3": (550, 551, 650)}
SELECTION_FOLDS = ("F1", "F2")
BASE_ALERTS_PER_DAY = 100
CAP_PER_DAY = 30
MIN_EXTRA_FRAUD_PER_DAY = 1.0
BASELINE = "hv200k_nonew_cut70"
REPORT_STEM = "p2_rule_backstop"

SELECTION_RULE = (
    "Among configurations that, in BOTH F1 and F2: (a) add at most 30 extra alerts beyond E4's 100/day on EVERY "
    "whole simulated day (strict daily cap, not an average), (b) catch more frauds that E4 missed than E4 itself "
    "catches when given the same total number of extra review slots (its next-ranked transactions), and (c) catch at "
    "least 1.0 extra fraud per day on average; choose the one with the most extra frauds over F1+F2. Ties: higher "
    "extra precision, then fewer active rules. If none qualifies, recommend no backstop alerts (rules give reason "
    "codes only)."
)
CONFIRMATION_RULE = "The selected configuration is confirmed only if conditions (a) and (b) also hold on F3."
DISCLOSURE = (
    "F1 and F2 select; F3 is used once here but has been used many times before (P1.1 discovery, P1.3 and P1.4 "
    "confirmations, P1.4.1, P1.4.2, and the champion's threshold calibration on steps 551-650), so it is not "
    "independent evidence. The retired sealed test (steps 651-743) is not read."
)


def grid() -> Dict[str, Dict[str, Any]]:
    """72 configurations: high-value threshold x optional first-time-recipient rule x HIGH cutoff."""
    out: Dict[str, Dict[str, Any]] = {}
    news = [None] + [(a, p) for a in (250_000, 500_000, 1_000_000, 2_000_000) for p in (25, 45)]
    for hv, new, cut in itertools.product([200_000, 500_000, 1_000_000, 2_000_000], news, [70, 80]):
        name = f"hv{hv // 1000}k_" + (f"new{new[0] // 1000}k@{new[1]}" if new else "nonew") + f"_cut{cut}"
        out[name] = {"hv": hv, "new": new, "cut": cut}
    return out


def rule_high(df: pd.DataFrame, hv: float, new: Optional[Tuple[float, int]], cut: int) -> Tuple[np.ndarray, int]:
    """HIGH-tier mask. With hv=200k, new=None, cut=70 this equals RiskEngine's default HIGH tier."""
    amt, typ, hr = df["amount"].to_numpy(), df["type"].to_numpy(), df["transaction_hour"].to_numpy()
    rail = np.isin(typ, ["TRANSFER", "CASH_OUT"])
    s = (35 * (amt >= hv) + 25 * rail + 25 * (np.isin(hr, [1, 2, 3, 4, 5]) & rail & (amt >= 10_000))
         + 15 * ((amt >= 50_000) & rail & (np.fmod(amt, 5000.0) == 0.0)))
    n_rules = 4
    if new:
        a, p = new
        s = s + p * (rail & (df["recipient_prior_in_count"].to_numpy() == 0) & (amt >= a))
        n_rules = 5
    return np.minimum(s, 100) >= cut, n_rules


def extra_alert_accounting(y: np.ndarray, steps: np.ndarray, score: np.ndarray, rule_alert: np.ndarray,
                           base_per_day: float = BASE_ALERTS_PER_DAY) -> Dict[str, Any]:
    """Extra alerts and fraud from rules beyond E4's top-K, vs E4 given the same extra capacity."""
    days = simulated_days(steps)
    k = int(round(base_per_day * days))
    w = selection_weights(score, k)
    h = rule_alert.astype(float)
    extra = h * (1 - w)
    extra_total = float(extra.sum())
    extra_fraud = float((extra * y).sum())
    w_more = selection_weights(score, k + int(round(extra_total)))
    e4_more = float(((w_more - w) * y).sum())
    day = (steps - 1) // STEPS_PER_DAY + 1
    per_day: Dict[int, float] = {}
    full: List[float] = []
    for d in np.unique(day):
        m = day == d
        per_day[int(d)] = round(float(extra[m].sum()), 1)
        if len(np.unique(steps[m])) == STEPS_PER_DAY:
            full.append(per_day[int(d)])
    fa = np.asarray(full) if full else np.asarray([np.nan])
    return {
        "days": round(days, 3), "full_days": len(full),
        "extra_alerts_per_day_mean": round(extra_total / days, 1),
        "extra_alerts_full_day_min": float(np.nanmin(fa)), "extra_alerts_full_day_median": float(np.nanmedian(fa)),
        "extra_alerts_full_day_max": float(np.nanmax(fa)), "extra_alerts_per_day": per_day,
        "extra_fraud": round(extra_fraud, 1), "extra_fraud_per_day": round(extra_fraud / days, 2),
        "extra_precision": round(extra_fraud / extra_total, 3) if extra_total else None,
        "e4_same_capacity_fraud": round(e4_more, 1),
        "beats_e4_same_capacity": bool(extra_fraud > e4_more),
        "e4_recall": round(float((w * y).sum() / y.sum()), 4),
        "combined_recall": round(float((np.maximum(w, h) * y).sum() / y.sum()), 4),
        "e4_plus_capacity_recall": round(float((w_more * y).sum() / y.sum()), 4),
        "_weights": (w, extra),
    }


def evaluate(df: pd.DataFrame, te: np.ndarray, score: np.ndarray, rule_alert: np.ndarray) -> Dict[str, Any]:
    y = df["isFraud"].to_numpy()[te]
    st = df["step"].to_numpy()[te]
    hr = df["transaction_hour"].to_numpy()[te]
    amt = df["amount"].to_numpy()[te]
    h = rule_alert[te]
    r = extra_alert_accounting(y, st, score, h)
    w, extra = r.pop("_weights")
    tp_rule = float((h * y).sum())
    r.update({
        "rule_standalone_alerts_per_day": round(float(h.sum()) / simulated_days(st), 1),
        "rule_standalone_precision": round(tp_rule / h.sum(), 3) if h.sum() else None,
        "rule_standalone_recall": round(tp_rule / y.sum(), 4),
        "extra_fraud_by_hour": {b: round(float((extra * y)[(hr >= lo) & (hr <= hi)].sum()), 1)
                                for b, (lo, hi) in {"00-05": (0, 5), "06-08": (6, 8), "09-17": (9, 17), "18-23": (18, 23)}.items()},
        "extra_fraud_by_amount": {b: round(float((extra * y)[(amt >= lo) & (amt < hi)].sum()), 1)
                                  for b, (lo, hi) in {"<500k": (0, 5e5), "500k-1M": (5e5, 1e6), ">=1M": (1e6, np.inf)}.items()},
    })
    return r


def checks(r: Dict[str, Any]) -> Dict[str, bool]:
    return {"cap_every_full_day": r["extra_alerts_full_day_max"] <= CAP_PER_DAY,
            "beats_e4_same_capacity": r["beats_e4_same_capacity"],
            "min_extra_fraud": r["extra_fraud_per_day"] >= MIN_EXTRA_FRAUD_PER_DAY}


def choose(results: Dict[str, Dict[str, Any]]) -> Tuple[List[str], Optional[str]]:
    qual = [n for n, r in results.items() if all(all(r["folds"][f]["checks"].values()) for f in SELECTION_FOLDS)]

    def key(n: str):
        r = results[n]["folds"]
        tot = sum(r[f]["extra_fraud"] for f in SELECTION_FOLDS)
        alerts = sum(r[f]["extra_alerts_per_day_mean"] * r[f]["days"] for f in SELECTION_FOLDS)
        return (tot, tot / max(alerts, 1e-9), -results[n]["n_rules"])

    return qual, (max(qual, key=key) if qual else None)


def _e4_fold(df: pd.DataFrame, fold: str) -> Tuple[np.ndarray, np.ndarray]:
    tmax, lo, hi = FOLDS[fold]
    st = df["step"].to_numpy()
    tr, te = st <= tmax, (st >= lo) & (st <= hi)
    cfg = MODEL_CONFIGS[E4]
    model = fit_model(cfg, build_features(df[tr], cfg), df["isFraud"].to_numpy()[tr])
    return te, model.predict_proba(build_features(df[te], cfg))[:, 1]


def run_selection(df: pd.DataFrame) -> Dict[str, Any]:
    assert df["step"].max() < FINAL_TEST_MIN_STEP
    fold_scores = {f: _e4_fold(df, f) for f in SELECTION_FOLDS}
    results: Dict[str, Any] = {}
    for name, g in grid().items():
        H, n_rules = rule_high(df, g["hv"], g["new"], g["cut"])
        results[name] = {"params": g, "n_rules": n_rules, "folds": {}}
        for f, (te, s) in fold_scores.items():
            r = evaluate(df, te, s, H)
            r["checks"] = checks(r)
            results[name]["folds"][f] = r
    qual, chosen = choose(results)
    return {"selection_rule": SELECTION_RULE, "confirmation_rule": CONFIRMATION_RULE, "folds_used": list(SELECTION_FOLDS),
            "max_step_loaded": int(df["step"].max()), "qualifying": qual, "chosen": chosen, "results": results}


def run_confirmation(df: pd.DataFrame, selection: Dict[str, Any], selection_sha: str) -> Dict[str, Any]:
    assert df["step"].max() < FINAL_TEST_MIN_STEP
    te, s = _e4_fold(df, "F3")
    g = grid()
    res = {}
    for n in [x for x in (selection["chosen"], BASELINE) if x]:
        H, _ = rule_high(df, g[n]["hv"], g[n]["new"], g[n]["cut"])
        r = evaluate(df, te, s, H)
        r["checks"] = checks(r)
        res[n] = r
    chosen = selection["chosen"]
    confirmed = bool(chosen) and res[chosen]["checks"]["cap_every_full_day"] and res[chosen]["checks"]["beats_e4_same_capacity"]
    return {"selection_sha256": selection_sha, "chosen_from_selection": chosen, "confirmation_rule": CONFIRMATION_RULE,
            "confirmed": confirmed, "max_step_loaded": int(df["step"].max()), "results": res}


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def _row(name: str, fold: str, r: Dict[str, Any]) -> str:
    ok = all(r["checks"].values()) if fold != "F3" else (r["checks"]["cap_every_full_day"] and r["checks"]["beats_e4_same_capacity"])
    return (f"| {name} | {fold} | {r['extra_alerts_per_day_mean']:.1f} | {r['extra_alerts_full_day_min']:.0f} / "
            f"{r['extra_alerts_full_day_median']:.0f} / {r['extra_alerts_full_day_max']:.0f} | {r['extra_fraud']:.1f} | "
            f"{r['e4_same_capacity_fraud']:.1f} | {r['combined_recall']:.3f} | {r['e4_plus_capacity_recall']:.3f} | "
            f"{'yes' if ok else 'no'} |")


def render_markdown(selection: Dict[str, Any], confirmation: Optional[Dict[str, Any]]) -> str:
    R = selection["results"]
    L = ["# P2 — Rule-engine backstop study", "",
         "**Question:** do rules catch fraud that E4 misses, better than simply giving E4 the same extra review capacity?", ""]
    verdict = ("No. No configuration met the pre-declared rule; giving E4 the same extra review slots catches more fraud. "
               "Rules are kept for reason codes only." if not selection["chosen"] else f"Selected: {selection['chosen']}.")
    if confirmation is not None:
        verdict += (" F3 confirmation: confirmed." if confirmation["confirmed"] else
                    " F3 confirmation: nothing was selected, so F3 (used once) reports the baseline rules only; they also "
                    "lose to E4 with the same extra slots." if not confirmation["chosen_from_selection"] else
                    " F3 confirmation: not confirmed.")
    L += [f"**Answer:** {verdict}", "", "## Protocol", "", f"Selection rule (fixed before F3): {selection['selection_rule']}", "",
          f"Confirmation rule: {selection['confirmation_rule']}", "", DISCLOSURE, "",
          "E4 alerts come from fold copies of the E4 configuration at 100 alerts per simulated day; the champion is untouched.", "",
          "## Key configurations", "",
          "| Configuration | Fold | Extra alerts/day (mean) | Whole-day min / median / max | Extra fraud caught | E4 with same extra slots | Combined recall | E4 + slots recall | Passes |",
          "|---|---|---:|---|---:|---:|---:|---:|---|"]
    shown = [BASELINE, "hv500k_nonew_cut70", "hv2000k_nonew_cut70", "hv500k_new1000k@25_cut70", "hv200k_new2000k@25_cut70"]
    for n in shown:
        for f in SELECTION_FOLDS:
            L.append(_row(n, f, R[n]["folds"][f]))
        if confirmation and n in confirmation["results"]:
            L.append(_row(n, "F3 (confirmation)", confirmation["results"][n]))
    n_cap = {f: sum(r["folds"][f]["checks"]["cap_every_full_day"] for r in R.values()) for f in SELECTION_FOLDS}
    n_beat = {f: sum(r["folds"][f]["checks"]["beats_e4_same_capacity"] for r in R.values()) for f in SELECTION_FOLDS}
    L += ["", f"Across all {len(R)} configurations: daily cap met on every whole day by {n_cap['F1']} (F1) and {n_cap['F2']} (F2); "
          f"beat E4 with the same extra slots in {n_beat['F1']} (F1) and {n_beat['F2']} (F2). Qualifying in both folds: "
          f"{len(selection['qualifying'])}.", "",
          "## All configurations (F1 | F2: mean extra alerts/day, max day, extra fraud vs E4 with same slots)", "",
          "| Configuration | F1 | F2 |", "|---|---|---|"]
    for n, r in R.items():
        c = [f"{r['folds'][f]['extra_alerts_per_day_mean']:.1f}/day, max {r['folds'][f]['extra_alerts_full_day_max']:.0f}, "
             f"{r['folds'][f]['extra_fraud']:.0f} vs {r['folds'][f]['e4_same_capacity_fraud']:.0f}" for f in SELECTION_FOLDS]
        L.append(f"| {n} | {c[0]} | {c[1]} |")
    L += ["", "## Notes", "",
          "- Counts are expected values where ties at E4's cut-off score are shared.",
          "- Rules use no balance fields, IDs or flags; the first-time-recipient condition uses the leakage-safe recipient count.",
          "- No confidence intervals: this compares review policies on fixed folds.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="P2 rule-engine backstop study (development folds only)")
    parser.add_argument("--stage", required=True, choices=["selection", "confirmation", "render"])
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--reports-dir", default="reports/evaluation")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out_dir = Path(args.reports_dir)
    sel_path = out_dir / f"{REPORT_STEM}_selection.json"
    conf_path = out_dir / f"{REPORT_STEM}_confirmation.json"
    if args.stage == "render":
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        conf = json.loads(conf_path.read_text(encoding="utf-8")) if conf_path.exists() else None
        (out_dir / f"{REPORT_STEM}.md").write_text(render_markdown(sel, conf), encoding="utf-8")
        return
    target = sel_path if args.stage == "selection" else conf_path
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite a recorded stage.")
    if args.stage == "confirmation" and not sel_path.exists():
        raise SystemExit("Run --stage selection first; the choice must be recorded before F3 is evaluated.")
    df = load_development_data(Path(args.data_path), MODEL_CONFIGS[E4]).reset_index(drop=True)
    if args.stage == "selection":
        out = run_selection(df)
    else:
        out = run_confirmation(df, json.loads(sel_path.read_text(encoding="utf-8")),
                               hashlib.sha256(sel_path.read_bytes()).hexdigest())
    out["generated_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    logger.info("wrote %s", target)


if __name__ == "__main__":
    main()
