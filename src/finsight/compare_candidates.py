"""
FinSight — P1.3 candidate comparison (development data only).

Two stages, run in order:

1. ``--stage screening``: every config in CANDIDATE_SET on folds F1 and F2. Applies the
   pre-declared shortlist rule (SHORTLIST_RULE) and writes the shortlist to the screening
   report *before* F3 is touched.
2. ``--stage confirmation``: reads that shortlist and evaluates only E0, the comparison
   baseline E2 and the shortlisted candidates on F3. The shortlist cannot be changed here.

The locked final test (steps 651-743) is never read: data goes through
``train.load_development_data`` (which drops it first) and every metric goes through
``evaluate.evaluate_scores`` (which raises on it). No model artifact is written.

Metric conventions match ``evaluate.py``: fixed alert budgets per simulated day with
tie-aware expected values, AP with a 95% step-block bootstrap interval, and paired
step-block bootstrap intervals for AP differences on the same rows.
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

from finsight.evaluate import (  # noqa: E402
    DEFAULT_BOOTSTRAP_ITERATIONS, FINAL_TEST_MIN_STEP, WALK_FORWARD_FOLDS, Fold,
    assert_development_steps, bootstrap_ap_difference, evaluate_scores, fold_masks, simulated_days,
)
from finsight.features import RECIPIENT_PRIOR_IN_COUNT  # noqa: E402
from finsight.train import MODEL_CONFIGS, build_features, fit_model, file_sha256, load_development_data  # noqa: E402

logger = logging.getLogger("finsight.compare_candidates")

# Label -> training config. E0 is the champion configuration (not the frozen artifact).
CANDIDATE_SET: Dict[str, str] = {
    "E0": "hgb_baseline8_balanced",
    "E1": "hgb_baseline8_rcpt_balanced",
    "E2": "hgb_baseline8_cw50_f32",
    "E3": "hgb_baseline8_rcpt_cw50",
    "E4": "hgb_baseline7_rcpt_cw50",
}
CANDIDATES: Tuple[str, ...] = ("E1", "E3", "E4")   # eligible for the shortlist
BASELINES: Tuple[str, ...] = ("E0", "E2")          # always evaluated as comparators

# (a, b, what the difference isolates) — each pair differs by exactly one change.
COMPARISONS: Tuple[Tuple[str, str, str], ...] = (
    ("E1", "E0", "recipient count (balanced weights)"),
    ("E2", "E0", "class weight 1:50 (no recipient count)"),
    ("E3", "E1", "class weight 1:50 (with recipient count)"),
    ("E3", "E2", "recipient count (class weight 1:50)"),
    ("E4", "E3", "removing log1p_amount"),
    ("E3", "E0", "recipient count + class weight 1:50 combined"),
    ("E4", "E0", "recipient count + class weight 1:50 - log1p_amount combined"),
)

SCREENING_FOLDS: Tuple[str, ...] = ("F1", "F2")
CONFIRMATION_FOLDS: Tuple[str, ...] = ("F3",)
ALERT_BUDGETS: Tuple[int, ...] = (50, 100, 200)
BREAKDOWN_BUDGET = 100
HOUR_BANDS: Tuple[Tuple[str, int, int], ...] = (("00-05", 0, 5), ("06-08", 6, 8), ("09-17", 9, 17), ("18-23", 18, 23))
RECIPIENT_BUCKETS: Tuple[Tuple[str, int, float], ...] = (("0", 0, 0), ("1-5", 1, 5), ("6-20", 6, 20), ("21+", 21, np.inf))

SHORTLIST_RULE = (
    "A candidate (E1, E3, E4) is shortlisted for F3 confirmation only if, in BOTH F1 and F2: "
    "(a) the paired 95% CI of AP(candidate) - AP(E0) lies entirely above 0, and "
    "(b) tie-aware recall at 100 and at 200 alerts/day is not lower than E0's."
)
CONFIRMATION_RULE = (
    "A shortlisted candidate is confirmed on F3 only if the paired 95% CI of AP(candidate) - AP(E0) lies "
    "entirely above 0 and tie-aware recall at 100 and 200 alerts/day is not lower than E0's on F3. "
    "Confirmation is not promotion: promotion and the single locked-test evaluation need separate approval."
)
DISCLOSURE = (
    "F1 and F2 are screening folds. F3 is the least-used fold, not an independent one: the recipient count "
    "was discovered by analysing the frozen champion's errors on steps 451-650 (which includes F3's test "
    "window), the champion configuration E0 was chosen using steps 551-743, and the 1:50 class weight was "
    "chosen on 451-550 and compared on 551-743. F2 is the most reused fold. The locked test (651-743) was not "
    "read in P1.3 (no labels, predictions or metrics) and gives no evidence here."
)


def _fold(name: str) -> Fold:
    return next(f for f in WALK_FORWARD_FOLDS if f.name == name)


def selection_weights(score: np.ndarray, k: int) -> np.ndarray:
    """Probability each row is reviewed when the top-k rows are reviewed with random tie-breaking.

    Mirrors the tie policy of `evaluate.metrics_at_alert_budget`.
    """
    score = np.asarray(score, dtype=float)
    w = np.zeros(len(score))
    if k <= 0:
        return w
    k = min(k, len(score))
    cutoff = np.sort(score)[::-1][k - 1]
    above, tied = score > cutoff, score == cutoff
    w[above] = 1.0
    w[tied] = (k - above.sum()) / tied.sum()
    return w


def segment_breakdown(y: np.ndarray, w: np.ndarray, seg: np.ndarray, labels: Sequence[str]) -> List[Dict[str, Any]]:
    rows = []
    for lab in labels:
        m = seg == lab
        fraud = int(y[m].sum())
        tp = float((w * y)[m].sum())
        fp = float((w * (1 - y))[m].sum())
        rows.append({
            "segment": lab, "rows": int(m.sum()), "fraud": fraud,
            "true_positives": round(tp, 2), "false_negatives": round(fraud - tp, 2), "false_positives": round(fp, 2),
            "recall": round(tp / fraud, 4) if fraud else None,
        })
    return rows


def hour_band(hours: np.ndarray) -> np.ndarray:
    out = np.empty(len(hours), dtype=object)
    for lab, lo, hi in HOUR_BANDS:
        out[(hours >= lo) & (hours <= hi)] = lab
    return out


def recipient_bucket(counts: np.ndarray) -> np.ndarray:
    out = np.empty(len(counts), dtype=object)
    for lab, lo, hi in RECIPIENT_BUCKETS:
        out[(counts >= lo) & (counts <= hi)] = lab
    return out


def breakdowns(y: np.ndarray, score: np.ndarray, steps: np.ndarray, hours: np.ndarray, rcpt: np.ndarray,
               budget: float = BREAKDOWN_BUDGET) -> Dict[str, Any]:
    k = int(round(budget * simulated_days(steps)))
    w = selection_weights(score, k)
    return {
        "alerts_per_day_budget": budget,
        "by_hour_band": segment_breakdown(y, w, hour_band(hours), [b[0] for b in HOUR_BANDS]),
        "by_recipient_prior_count": segment_breakdown(y, w, recipient_bucket(rcpt), [b[0] for b in RECIPIENT_BUCKETS]),
    }


def apply_shortlist_rule(blocks: Dict[str, Dict[str, Any]], diffs: Dict[str, Dict[str, Any]],
                         folds: Sequence[str] = SCREENING_FOLDS) -> Dict[str, Any]:
    """Return {candidate: {"shortlisted": bool, "checks": [...]}} from screening results."""
    out: Dict[str, Any] = {}
    for c in CANDIDATES:
        checks = []
        for f in folds:
            ci_low = diffs[f"{c}-E0"][f]["ci95"][0]
            r = {b: blocks[c][f]["at_alert_budget"][f"{b}_per_day"]["recall"] for b in (100, 200)}
            r0 = {b: blocks["E0"][f]["at_alert_budget"][f"{b}_per_day"]["recall"] for b in (100, 200)}
            checks.append({"fold": f, "dAP_ci_low": ci_low, "dAP_ci_above_0": ci_low > 0,
                           "recall_100_not_lower": r[100] >= r0[100], "recall_200_not_lower": r[200] >= r0[200]})
        ok = all(ch["dAP_ci_above_0"] and ch["recall_100_not_lower"] and ch["recall_200_not_lower"] for ch in checks)
        out[c] = {"shortlisted": bool(ok), "checks": checks}
    return out


def _bootstrap_jobs(jobs: List[Tuple], n_jobs: int) -> List[Any]:
    from joblib import Parallel, delayed

    def run(kind, *args):
        return evaluate_scores(*args[:3], alert_budgets_per_day=ALERT_BUDGETS, n_boot=args[3]) if kind == "block" \
            else bootstrap_ap_difference(*args[:4], n_boot=args[4])

    return Parallel(n_jobs=n_jobs)(delayed(run)(*j) for j in jobs)


def run_stage(df: pd.DataFrame, labels: Sequence[str], folds: Sequence[str], n_boot: int, n_jobs: int) -> Dict[str, Any]:
    scores: Dict[str, Dict[str, np.ndarray]] = {lab: {} for lab in labels}
    fold_data: Dict[str, Dict[str, np.ndarray]] = {}
    fit_info: Dict[str, Dict[str, Any]] = {lab: {} for lab in labels}
    for fname in folds:
        fold = _fold(fname)
        tr_m, te_m = fold_masks(df["step"], fold)
        train, test = df[tr_m], df[te_m]
        assert_development_steps(test["step"], f"{fname} test")
        assert train["step"].max() < test["step"].min()
        fold_data[fname] = {
            "y": test["isFraud"].to_numpy(dtype=int), "steps": test["step"].to_numpy(),
            "hours": test["transaction_hour"].to_numpy(), "rcpt": test[RECIPIENT_PRIOR_IN_COUNT].to_numpy(),
        }
        y_train = train["isFraud"].to_numpy(dtype=int)
        for lab in labels:
            cfg = MODEL_CONFIGS[CANDIDATE_SET[lab]]
            t0 = time.time()
            X_train = build_features(train, cfg)
            model = fit_model(cfg, X_train, y_train)
            scores[lab][fname] = model.predict_proba(build_features(test, cfg))[:, 1]
            fit_info[lab][fname] = {"fit_seconds": round(time.time() - t0, 1), "features": list(X_train.columns),
                                    "n_iter": int(model.n_iter_), "train_rows": int(len(train)),
                                    "train_fraud": int(y_train.sum())}
            logger.info("%s %s fitted in %.0fs", lab, fname, time.time() - t0)

    jobs, keys = [], []
    for lab in labels:
        for f in folds:
            d = fold_data[f]
            jobs.append(("block", d["y"], scores[lab][f], d["steps"], n_boot)); keys.append(("block", lab, f))
    comps = [c for c in COMPARISONS if c[0] in labels and c[1] in labels]
    for a, b, _ in comps:
        for f in folds:
            d = fold_data[f]
            jobs.append(("diff", d["y"], scores[a][f], scores[b][f], d["steps"], n_boot)); keys.append(("diff", f"{a}-{b}", f))
    t0 = time.time()
    results = _bootstrap_jobs(jobs, n_jobs)
    logger.info("bootstrap jobs (%d) done in %.0fs", len(jobs), time.time() - t0)

    blocks: Dict[str, Dict[str, Any]] = {lab: {} for lab in labels}
    diffs: Dict[str, Dict[str, Any]] = {f"{a}-{b}": {"isolates": why} for a, b, why in comps}
    for (kind, name, f), res in zip(keys, results):
        if kind == "block":
            d = fold_data[f]
            res["breakdown_at_100_per_day"] = breakdowns(d["y"], scores[name][f], d["steps"], d["hours"], d["rcpt"])
            res.update(fit_info[name][f])
            blocks[name][f] = res
        else:
            diffs[name][f] = res
    return {"labels": list(labels), "folds": list(folds), "results": blocks, "paired_ap_differences": diffs}


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def render_markdown(stage: str, out: Dict[str, Any]) -> str:
    folds, labels, R = out["folds"], out["labels"], out["results"]
    title = "P1.3 screening (folds F1, F2)" if stage == "screening" else "P1.3 confirmation (fold F3)"
    L = [f"# {title}", "", f"Generated {out['meta']['generated_utc']} · data sha256 `{out['meta']['data_sha256'][:12]}…` · "
         f"{out['meta']['n_boot']} step-block bootstrap resamples · locked test (≥{FINAL_TEST_MIN_STEP}) not read.", "",
         "## Disclosure", "", DISCLOSURE, ""]
    if stage == "screening":
        L += ["## Results are screening results",
              "", "These folds were used to discover and choose the candidates. Treat them as optimistic.", ""]
    else:
        L += ["## Results are confirmation results", "",
              f"Shortlist carried over unchanged from the screening report: {out['shortlist']}.", "",
              "F3 is not fully independent (see disclosure).", ""]
    L += ["## Configurations", "", "| Label | Config | Features | Class weight |", "|---|---|---|---|"]
    for lab in labels:
        cfg = MODEL_CONFIGS[CANDIDATE_SET[lab]]
        feats = R[lab][folds[0]]["features"]
        L.append(f"| {lab} | `{CANDIDATE_SET[lab]}` | {len(feats)}: {', '.join(feats)} | {cfg['class_weight']} |")
    L += ["", "All use the champion's float32 amount path and hyperparameters (max_iter 50, lr 0.1, 31 leaves, seed 42).", ""]
    for f in folds:
        b0 = R[labels[0]][f]
        L += [f"## Fold {f} (test steps {b0['steps'][0]}–{b0['steps'][1]}, {b0['rows']:,} rows, "
              f"{b0['labelled_fraud']:,} fraud, prevalence {b0['fraud_prevalence_pct']:.3f}%, "
              f"{b0['simulated_days']:.2f} simulated days)", "",
              "| | AP | AP 95% CI | ROC-AUC | Budget | Alerts | TP | FP | FN | Precision | Recall | F1 |",
              "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for lab in labels:
            r = R[lab][f]
            for i, bud in enumerate(ALERT_BUDGETS):
                m = r["at_alert_budget"][f"{bud}_per_day"]
                head = (f"| **{lab}** | {r['average_precision']:.4f} | {r['average_precision_ci95'][0]:.3f}–"
                        f"{r['average_precision_ci95'][1]:.3f} | {r['roc_auc']:.4f} ") if i == 0 else "| | | | "
                L.append(head + f"| {bud}/day | {m['alerts']:,} | {m['true_positives']:.1f} | {m['false_positives']:.1f} | "
                         f"{m['false_negatives']:.1f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1_score']:.3f} |")
        L += ["", f"Paired AP differences on {f} (same rows, step-block bootstrap):", "",
              "| Comparison | Isolates | ΔAP | 95% CI | Share of resamples not better |", "|---|---|---:|---|---:|"]
        for name, d in out["paired_ap_differences"].items():
            if f in d:
                L.append(f"| {name.replace('-', ' vs ')} | {d['isolates']} | {d[f]['point']:+.4f} | "
                         f"{d[f]['ci95'][0]:+.4f} to {d[f]['ci95'][1]:+.4f} | {d[f]['share_of_resamples_not_better']:.2f} |")
        for key, title_ in (("by_hour_band", "transaction hour"), ("by_recipient_prior_count", "recipient prior count")):
            segs = [s["segment"] for s in R[labels[0]][f]["breakdown_at_100_per_day"][key]]
            L += ["", f"Errors at 100 alerts/day by {title_} on {f} (FN / FP, expected values; fraud per segment in header):", "",
                  "| | " + " | ".join(f"{s['segment']} ({s['fraud']} fraud, {s['rows']:,} rows)"
                                      for s in R[labels[0]][f]["breakdown_at_100_per_day"][key]) + " |",
                  "|---|" + "---:|" * len(segs)]
            for lab in labels:
                cells = [f"{s['false_negatives']:.0f} / {s['false_positives']:.0f}"
                         for s in R[lab][f]["breakdown_at_100_per_day"][key]]
                L.append(f"| {lab} | " + " | ".join(cells) + " |")
        L.append("")
    if stage == "screening":
        L += ["## Shortlist", "", f"Rule (declared before any F3 evaluation): {SHORTLIST_RULE}", "",
              "| Candidate | Fold | ΔAP CI low vs E0 | CI above 0 | Recall@100 not lower | Recall@200 not lower |",
              "|---|---|---:|---|---|---|"]
        for c, s in out["shortlist_decision"].items():
            for ch in s["checks"]:
                L.append(f"| {c} | {ch['fold']} | {ch['dAP_ci_low']:+.4f} | {ch['dAP_ci_above_0']} | "
                         f"{ch['recall_100_not_lower']} | {ch['recall_200_not_lower']} |")
        L += ["", f"**Shortlisted for F3:** {', '.join(out['shortlist']) or 'none'}.", ""]
    else:
        L += ["## Confirmation decision", "", f"Rule: {CONFIRMATION_RULE}", "",
              "| Candidate | ΔAP vs E0 | 95% CI | Recall@100 vs E0 | Recall@200 vs E0 | Confirmed |", "|---|---:|---|---|---|---|"]
        for c, s in out["confirmation_decision"].items():
            L.append(f"| {c} | {s['dAP']:+.4f} | {s['ci95'][0]:+.4f} to {s['ci95'][1]:+.4f} | {s['recall_100']} | "
                     f"{s['recall_200']} | **{s['confirmed']}** |")
        L.append("")
    L += ["## Notes", "",
          "- Precision, recall and F1 are reported at fixed review capacities (alerts per simulated day), the protocol's "
          "operating points. No per-fold decision threshold is reported: no window exists that is disjoint from both a "
          "fold's training and test data without changing the model, and threshold policy is P1.4.",
          "- Confidence intervals are given for AP and paired AP differences (test-sample variation only, not "
          "training variation). Budget metrics have no interval in the current protocol.",
          "- Budget counts are expected values under random tie-breaking at the cut-off score, so they can be fractional.",
          "- Scores are risk scores, not calibrated probabilities.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="P1.3 candidate comparison (development folds only)")
    parser.add_argument("--stage", required=True, choices=["screening", "confirmation"])
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--reports-dir", default="reports/evaluation")
    parser.add_argument("--n-boot", type=int, default=DEFAULT_BOOTSTRAP_ITERATIONS)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    out_dir = Path(args.reports_dir)
    screening_path = out_dir / "p13_screening.json"

    if args.stage == "screening":
        labels, folds = list(CANDIDATE_SET), list(SCREENING_FOLDS)
        target = screening_path
    else:
        if not screening_path.exists():
            raise SystemExit("Run --stage screening first; the shortlist must exist before F3 is evaluated.")
        screening = json.loads(screening_path.read_text(encoding="utf-8"))
        shortlist = screening["shortlist"]
        if not shortlist:
            raise SystemExit("Screening shortlisted no candidate; nothing to confirm.")
        labels = [lab for lab in CANDIDATE_SET if lab in BASELINES or lab in shortlist]
        folds = list(CONFIRMATION_FOLDS)
        target = out_dir / "p13_confirmation.json"
    if target.exists():
        raise SystemExit(f"{target} exists; refusing to overwrite a recorded stage.")

    t0 = time.time()
    cfg_any = MODEL_CONFIGS[CANDIDATE_SET["E1"]]  # history-bearing config: loads nameDest and adds the count
    df = load_development_data(Path(args.data_path), cfg_any)
    assert df["step"].max() < FINAL_TEST_MIN_STEP
    logger.info("loaded %s development rows in %.0fs", f"{len(df):,}", time.time() - t0)

    out = run_stage(df, labels, folds, args.n_boot, args.n_jobs)
    out["meta"] = {
        "stage": args.stage, "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_sha256": file_sha256(Path(args.data_path)), "n_boot": args.n_boot,
        "final_test_min_step": FINAL_TEST_MIN_STEP, "max_step_loaded": int(df["step"].max()),
        "configs": {lab: CANDIDATE_SET[lab] for lab in labels}, "disclosure": DISCLOSURE,
        "runtime_seconds": None,
    }
    if args.stage == "screening":
        out["shortlist_rule"] = SHORTLIST_RULE
        out["shortlist_decision"] = apply_shortlist_rule(out["results"], out["paired_ap_differences"])
        out["shortlist"] = [c for c, s in out["shortlist_decision"].items() if s["shortlisted"]]
    else:
        out["shortlist"] = shortlist
        out["confirmation_rule"] = CONFIRMATION_RULE
        dec = {}
        for c in shortlist:
            d = out["paired_ap_differences"][f"{c}-E0"]["F3"]
            r = {b: out["results"][c]["F3"]["at_alert_budget"][f"{b}_per_day"]["recall"] for b in (100, 200)}
            r0 = {b: out["results"]["E0"]["F3"]["at_alert_budget"][f"{b}_per_day"]["recall"] for b in (100, 200)}
            dec[c] = {"dAP": d["point"], "ci95": d["ci95"],
                      "recall_100": f"{r[100]:.3f} vs {r0[100]:.3f}", "recall_200": f"{r[200]:.3f} vs {r0[200]:.3f}",
                      "confirmed": bool(d["ci95"][0] > 0 and r[100] >= r0[100] and r[200] >= r0[200])}
        out["confirmation_decision"] = dec
    out["meta"]["runtime_seconds"] = round(time.time() - t0)
    out_dir.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    target.with_suffix(".md").write_text(render_markdown(args.stage, out), encoding="utf-8")
    logger.info("wrote %s (+ .md) in %.0fs", target, time.time() - t0)


if __name__ == "__main__":
    main()
