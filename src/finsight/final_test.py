"""
FinSight — P1.5 one-time evaluation on the sealed final test (steps 651-743).

Two stages, each run once and in order:

1. ``--stage preregister`` writes ``reports/final_test/p15_preregistration.json``: the code commit,
   the hashes of every configuration used, the data hash, the champion artifact hash, the
   windows, the policy parameters and the decision rule. It reads no locked label. The
   manifest must then be committed.
2. ``--stage run`` refuses unless the working tree is clean, the manifest is committed and
   unchanged, no source file changed since the pre-registered commit, every hash still
   matches and no result exists yet. It then trains, scores the sealed test once and applies
   the pre-registered decision rule.

Design (option A, approved): every model is trained on steps 1-550; thresholds and the review
queue are calibrated on steps 551-650 (scores only); the sealed steps 651-743 are scored once.
The model that is evaluated is exactly the model that would be shipped.

Locked labels are separated from the features at load time and read in one function only,
``_evaluate_sealed``, which scores through ``evaluate.score_locked_final_test``. Recipient history
for the sealed steps uses recipient and step only, never labels. Nothing here writes to the
database or to ``models/``; the E4 bundle goes to the git-ignored ``models/candidates/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from sklearn.metrics import average_precision_score  # noqa: E402

from finsight.alert_policy import policy_metrics, threshold_for_budget  # noqa: E402
from finsight.compare_candidates import breakdowns  # noqa: E402
from finsight.evaluate import (  # noqa: E402
    DEFAULT_BOOTSTRAP_ITERATIONS, FINAL_TEST_MIN_STEP, bootstrap_ap_ci, bootstrap_ap_difference,
    score_locked_final_test, simulated_days,
)
from finsight.features import RECIPIENT_PRIOR_IN_COUNT  # noqa: E402
from finsight.review_queue import FLOOR_MULTIPLE, WARMUP_STEPS, queue_diagnostics, simulate_queue  # noqa: E402
from finsight.risk_rules import RiskEngine  # noqa: E402
from finsight.train import MODEL_CONFIGS, add_history_features, build_features, file_sha256, fit_model  # noqa: E402

logger = logging.getLogger("finsight.final_test")

CANDIDATE = "hgb_baseline7_rcpt_cw50"      # E4
BASELINE = "hgb_baseline8_balanced"        # E0: the champion configuration, retrained the same way
CHAMPION_ARTIFACT = Path("models/hist_gradient_boosting_no_day.joblib")
TRAIN_MAX_STEP = 550
CALIBRATION = (551, 650)
SEALED = (FINAL_TEST_MIN_STEP, 743)
BUDGETS: Tuple[int, ...] = (50, 100, 200)
DECISION_BUDGETS: Tuple[int, ...] = (100, 200)
QUEUE_DEADLINE = 12
RULE_HIGH_SCORE = 70
REPORT_DIR = Path("reports/final_test")
PREREG_FILE = REPORT_DIR / "p15_preregistration.json"
RESULT_FILE = REPORT_DIR / "p15_final_test.json"
BUNDLE_FILE = Path("models/candidates/p15_e4_final.joblib")
SOURCE_PATHS: Tuple[str, ...] = ("src",)
P13_CONFIRMATION = Path("reports/evaluation/p13_confirmation.json")
P13_LABELS = {"E4": CANDIDATE, "E0": BASELINE}

DECISION_RULE = (
    "Promote E4 (hgb_baseline7_rcpt_cw50) over the champion configuration E0 (hgb_baseline8_balanced), both "
    "trained on steps 1-550, only if on the sealed steps 651-743: (a) the 95% paired step-block bootstrap "
    "interval of AP(E4) - AP(E0) lies entirely above 0, and (b) tie-aware recall at 100 and at 200 alerts per "
    "simulated day of E4 is not lower than E0's. Otherwise keep the champion. The frozen champion artifact, the "
    "rule engine and the alert policies are reported for context and cannot change the decision. No setting is "
    "changed after the sealed labels are read."
)
DISCLOSURE = (
    "The sealed steps are not pristine: before P0 they were inside the old holdout (steps 551-743), which was used "
    "to choose the champion's 8-feature set and to compare the 1:50 class weight (EXP-07). They were never used to "
    "discover or screen the recipient feature (P1.1-P1.3 read steps <= 650 only). Development folds F1-F3 were "
    "reused many times; this run is the only use of steps 651-743 in P1."
)


class FinalTestGuardError(RuntimeError):
    """Raised when a pre-condition of the one-time sealed evaluation is not met."""


# ------------------------------------------------------------------------------
# Hashes and repository state
# ------------------------------------------------------------------------------
def config_sha256(cfg: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def frozen_settings() -> Dict[str, Any]:
    """Everything the decision depends on, in a form that can be hashed and compared."""
    return {
        "candidate": CANDIDATE, "baseline": BASELINE,
        "config_sha256": {n: config_sha256(MODEL_CONFIGS[n]) for n in (CANDIDATE, BASELINE)},
        "train_max_step": TRAIN_MAX_STEP, "calibration_steps": list(CALIBRATION), "sealed_steps": list(SEALED),
        "budgets": list(BUDGETS), "decision_budgets": list(DECISION_BUDGETS),
        "queue": {"deadline_hours": QUEUE_DEADLINE, "floor_multiple": FLOOR_MULTIPLE, "warmup_steps": WARMUP_STEPS},
        "rule_engine_high_score": RULE_HIGH_SCORE, "n_boot": DEFAULT_BOOTSTRAP_ITERATIONS, "bootstrap_seed": 0,
        "decision_rule": DECISION_RULE,
    }


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=False)


def repo_state(prereg_commit: Optional[str] = None) -> Dict[str, Any]:
    head = _git("rev-parse", "HEAD").stdout.strip()
    state: Dict[str, Any] = {
        "head": head,
        "clean": _git("status", "--porcelain").stdout.strip() == "",
        "prereg_tracked": _git("ls-files", "--error-unmatch", str(PREREG_FILE).replace("\\", "/")).returncode == 0,
    }
    if prereg_commit:
        state["prereg_commit_is_ancestor"] = _git("merge-base", "--is-ancestor", prereg_commit, head).returncode == 0
        state["source_unchanged_since_prereg"] = _git("diff", "--quiet", prereg_commit, head, "--", *SOURCE_PATHS).returncode == 0
    return state


def check_run_preconditions(manifest: Dict[str, Any], state: Dict[str, Any], current: Dict[str, Any],
                            data_sha: str, champion_sha: str, result_exists: bool) -> None:
    """Raise FinalTestGuardError unless every pre-condition of the sealed run holds."""
    problems: List[str] = []
    if result_exists:
        problems.append(f"{RESULT_FILE} or {BUNDLE_FILE} already exists: the sealed test has been run")
    if not state.get("clean"):
        problems.append("working tree has uncommitted changes")
    if not state.get("prereg_tracked"):
        problems.append("the pre-registration manifest is not committed")
    if not state.get("prereg_commit_is_ancestor", False):
        problems.append("the pre-registered commit is not an ancestor of HEAD")
    if not state.get("source_unchanged_since_prereg", False):
        problems.append("source code changed since the pre-registered commit")
    if manifest.get("settings") != current:
        problems.append("frozen settings differ from the pre-registration")
    if manifest.get("data_sha256") != data_sha:
        problems.append("data file differs from the pre-registration")
    if manifest.get("champion_artifact_sha256") != champion_sha:
        problems.append("champion artifact differs from the pre-registration")
    if problems:
        raise FinalTestGuardError("Sealed test refused: " + "; ".join(problems))


def expected_reproduction() -> Dict[str, float]:
    """AP on steps 551-650 that P1.3 recorded for the same models (trained on steps 1-550)."""
    p13 = json.loads(P13_CONFIRMATION.read_text(encoding="utf-8"))
    out = {}
    for label, name in P13_LABELS.items():
        if p13["meta"]["configs"].get(label) != name:
            raise FinalTestGuardError(f"P1.3 label {label} is not {name}")
        f3 = p13["results"][label]["F3"]
        if f3["steps"] != list(CALIBRATION):
            raise FinalTestGuardError("P1.3 F3 window does not match the calibration window")
        out[label] = float(f3["average_precision"])
    return out


def check_reproduction(actual: Dict[str, float], expected: Dict[str, float]) -> None:
    """Refuse before any sealed label is read unless the models reproduce P1.3's F3 results."""
    bad = {k: (round(actual[k], 4), expected[k]) for k in expected if round(actual[k], 4) != round(expected[k], 4)}
    if bad:
        raise FinalTestGuardError(f"Models do not reproduce P1.3 F3 average precision {bad}; sealed labels not read.")


# ------------------------------------------------------------------------------
# Decision rule
# ------------------------------------------------------------------------------
def decide(dap_ci95: Sequence[float], recall_candidate: Dict[int, float], recall_baseline: Dict[int, float]) -> Dict[str, Any]:
    checks = {"dAP_ci_above_0": bool(dap_ci95[0] > 0)}
    for b in DECISION_BUDGETS:
        checks[f"recall_{b}_not_lower"] = bool(recall_candidate[b] >= recall_baseline[b])
    promote = all(checks.values())
    return {"checks": checks, "promote_e4": promote,
            "decision": "PROMOTE E4 (pending separate promotion approval)" if promote else "KEEP CHAMPION"}


# ------------------------------------------------------------------------------
# Data and scoring (no locked labels read here)
# ------------------------------------------------------------------------------
def load_frames(data_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return (development rows with labels, sealed rows WITHOUT labels, sealed labels kept apart)."""
    cols = ["step", "type", "amount", "transaction_hour", "isFraud", "nameDest"]
    df = pd.read_csv(data_path, usecols=cols)
    df = add_history_features(df, MODEL_CONFIGS[CANDIDATE])   # recipient + step only; labels never read
    sealed_mask = df["step"] >= SEALED[0]
    dev = df[~sealed_mask].copy()
    sealed = df[sealed_mask]
    sealed_labels = sealed["isFraud"].copy()
    sealed = sealed.drop(columns=["isFraud"])
    return dev, sealed, sealed_labels


def train_and_score(dev: pd.DataFrame, sealed: pd.DataFrame, cfg: Dict[str, Any]) -> Dict[str, Any]:
    if "isFraud" in sealed.columns:
        raise FinalTestGuardError("sealed features must not carry labels")
    train = dev[dev["step"] <= TRAIN_MAX_STEP]
    cal = dev[(dev["step"] >= CALIBRATION[0]) & (dev["step"] <= CALIBRATION[1])]
    t0 = time.time()
    model = fit_model(cfg, build_features(train, cfg), train["isFraud"].to_numpy(dtype=int))
    s_cal = model.predict_proba(build_features(cal, cfg))[:, 1]
    s_sealed = model.predict_proba(build_features(sealed, cfg))[:, 1]
    cal_steps = cal["step"].to_numpy()
    cal_days = simulated_days(cal_steps)
    thresholds = {b: threshold_for_budget(s_cal, int(round(b * cal_days))) for b in BUDGETS}
    floors = {b: threshold_for_budget(s_cal, int(round(FLOOR_MULTIPLE * b * cal_days))) for b in BUDGETS}
    warm = cal_steps > cal_steps.max() - WARMUP_STEPS
    cal_ap = float(average_precision_score(cal["isFraud"].to_numpy(dtype=int), s_cal))  # development labels only
    return {"model": model, "calibration_ap": cal_ap, "features": list(build_features(train.head(1), cfg).columns),
            "scores": s_sealed, "thresholds": thresholds, "floors": floors,
            "warm_scores": s_cal[warm], "warm_steps": cal_steps[warm],
            "train_rows": int(len(train)), "train_fraud": int(train["isFraud"].sum()),
            "fit_seconds": round(time.time() - t0, 1)}


# ------------------------------------------------------------------------------
# The only place that reads sealed labels
# ------------------------------------------------------------------------------
def _evaluate_sealed(sealed: pd.DataFrame, sealed_labels: pd.Series, systems: Dict[str, np.ndarray],
                     runs: Dict[str, Dict[str, Any]], champion_threshold: float, rule_high: np.ndarray) -> Dict[str, Any]:
    labelled = sealed.assign(isFraud=sealed_labels.to_numpy())
    y = sealed_labels.to_numpy(dtype=int)
    steps = sealed["step"].to_numpy()
    hours = sealed["transaction_hour"].to_numpy()
    amount = sealed["amount"].to_numpy()
    rcpt = sealed[RECIPIENT_PRIOR_IN_COUNT].to_numpy()
    out: Dict[str, Any] = {"systems": {}}
    for name, score in systems.items():
        res = score_locked_final_test(
            labelled, lambda t, s=score: s[labelled.index.get_indexer(t.index)],
            i_understand_this_is_the_single_final_evaluation=True, alert_budgets_per_day=BUDGETS,
            threshold=champion_threshold if name == "champion_artifact" else None,
        )
        res["average_precision_ci95"] = bootstrap_ap_ci(y, score, steps)
        res["breakdown_at_100_per_day"] = breakdowns(y, score, steps, hours, rcpt)
        out["systems"][name] = res
    out["paired_ap_difference_E4_minus_E0"] = bootstrap_ap_difference(y, systems["E4"], systems["E0"], steps)
    out["rule_engine_high"] = policy_metrics(rule_high.astype(float), y, steps, hours, amount)
    pol: Dict[str, Any] = {}
    for name in ("E4", "E0"):
        r = runs[name]
        for b in BUDGETS:
            fixed = (r["scores"] >= r["thresholds"][b]).astype(float)
            pol[f"{name} B @{b}"] = policy_metrics(fixed, y, steps, hours, amount)
            sim = simulate_queue(r["scores"], steps, b, QUEUE_DEADLINE, r["floors"][b],
                                 warmup_score=r["warm_scores"], warmup_steps=r["warm_steps"])
            m = policy_metrics(sim["reviewed"], y, steps, hours, amount)
            m["queue"] = queue_diagnostics(sim, y, steps)
            pol[f"{name} Q12 @{b}"] = m
    out["policies"] = pol
    rec = {n: {b: out["systems"][n]["at_alert_budget"][f"{b}_per_day"]["recall"] for b in DECISION_BUDGETS} for n in ("E4", "E0")}
    out["decision"] = decide(out["paired_ap_difference_E4_minus_E0"]["ci95"], rec["E4"], rec["E0"])
    return out


# ------------------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------------------
def render_markdown(result: Dict[str, Any]) -> str:
    S, d = result["systems"], result["decision"]
    L = ["# P1.5 sealed final test (steps 651-743)", "",
         f"Run {result['meta']['generated_utc']} · code `{result['meta']['head'][:10]}` · pre-registered at "
         f"`{result['meta']['prereg_commit'][:10]}` · data sha256 `{result['meta']['data_sha256'][:12]}…`", "",
         f"## Decision: {d['decision']}", "", f"Rule (pre-registered): {DECISION_RULE}", "",
         "| Check | Result |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in d["checks"].items()]
    dd = result["paired_ap_difference_E4_minus_E0"]
    L += ["", f"AP(E4) - AP(E0) = {dd['point']:+.4f}, 95% CI {dd['ci95'][0]:+.4f} to {dd['ci95'][1]:+.4f}.", "",
          "## Disclosure", "", DISCLOSURE, "",
          "## Systems on the sealed steps", "",
          "| System | AP | AP 95% CI | ROC-AUC | Budget | Alerts | TP | FP | FN | Precision | Recall | F1 |",
          "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, r in S.items():
        for i, b in enumerate(BUDGETS):
            m = r["at_alert_budget"][f"{b}_per_day"]
            head = (f"| **{name}** | {r['average_precision']:.4f} | {r['average_precision_ci95'][0]:.3f}–"
                    f"{r['average_precision_ci95'][1]:.3f} | {r['roc_auc']:.4f} ") if i == 0 else "| | | | "
            L.append(head + f"| {b}/day | {m['alerts']:,} | {m['true_positives']:.1f} | {m['false_positives']:.1f} | "
                     f"{m['false_negatives']:.1f} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1_score']:.3f} |")
    ch = S["champion_artifact"].get("at_threshold")
    if ch:
        L += ["", f"Frozen champion at its threshold {result['meta']['champion_threshold']}: {ch['alerts_per_day']} alerts/day, "
                  f"precision {ch['precision']}, recall {ch['recall']}."]
    rh = result["rule_engine_high"]
    L += [f"Rule engine HIGH tier: {rh['alerts_per_day']} alerts/day, precision {rh['precision']:.3f}, recall {rh['recall']:.3f}.", "",
          "## Alert policies (context)", "",
          "| Policy | Alerts/day | Whole-day min / median / max | Precision | Recall | R 00-05 | R 06-23 | R <100k |",
          "|---|---:|---|---:|---:|---:|---:|---:|"]
    for name, m in result["policies"].items():
        dl, sr = m["daily"], m["segment_recall"]
        rng = (f"{dl['full_day_min']:.0f} / {dl['full_day_median']:.0f} / {dl['full_day_max']:.0f}"
               if dl["full_day_min"] is not None else "–")
        L.append(f"| {name} | {m['alerts_per_day']:.1f} | {rng} | {m['precision']:.3f} | {m['recall']:.3f} | "
                 f"{sr['00-05']['recall']} | {sr['06-23']['recall']} | {sr['amount<100k']['recall']} |")
    L += ["", "## Notes", "",
          "- Every model was trained on steps 1-550; thresholds and the queue were calibrated on steps 551-650 (scores only).",
          "- Budget counts are expected values under random tie-breaking at the cut-off score.",
          "- Intervals reflect test-sample variation only, not training variation.",
          "- Promotion is a separate step (P1.6) and needs its own approval.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------------------
# Stages
# ------------------------------------------------------------------------------
def preregister(data_path: Path) -> Path:
    if PREREG_FILE.exists():
        raise FinalTestGuardError(f"{PREREG_FILE} already exists; pre-registration is done once.")
    state = repo_state()
    if not state["clean"]:
        raise FinalTestGuardError("Commit the code before pre-registering (working tree has changes).")
    manifest = {
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "code_commit": state["head"], "settings": frozen_settings(),
        "configs": {n: MODEL_CONFIGS[n] for n in (CANDIDATE, BASELINE)},
        "data_path": str(data_path), "data_sha256": file_sha256(data_path),
        "champion_artifact": str(CHAMPION_ARTIFACT), "champion_artifact_sha256": file_sha256(CHAMPION_ARTIFACT),
        "expected_calibration_ap": expected_reproduction(),
        "disclosure": DISCLOSURE, "locked_labels_read": False,
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    PREREG_FILE.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return PREREG_FILE


def run(data_path: Path) -> Path:
    if not PREREG_FILE.exists():
        raise FinalTestGuardError("Pre-register first (--stage preregister) and commit the manifest.")
    manifest = json.loads(PREREG_FILE.read_text(encoding="utf-8"))
    state = repo_state(manifest["code_commit"])
    check_run_preconditions(manifest, state, frozen_settings(), file_sha256(data_path),
                            file_sha256(CHAMPION_ARTIFACT), RESULT_FILE.exists() or BUNDLE_FILE.exists())
    t0 = time.time()
    dev, sealed, sealed_labels = load_frames(data_path)
    runs = {"E4": train_and_score(dev, sealed, MODEL_CONFIGS[CANDIDATE]),
            "E0": train_and_score(dev, sealed, MODEL_CONFIGS[BASELINE])}
    check_reproduction({k: r["calibration_ap"] for k, r in runs.items()}, manifest["expected_calibration_ap"])
    champion = joblib.load(CHAMPION_ARTIFACT)
    s_champ = champion["model"].predict_proba(
        build_features(sealed, MODEL_CONFIGS[BASELINE])[champion["feature_names"]])[:, 1]
    rule_high = (RiskEngine().evaluate_dataframe(sealed)["risk_score"].to_numpy() >= RULE_HIGH_SCORE)
    systems = {"E4": runs["E4"]["scores"], "E0": runs["E0"]["scores"], "champion_artifact": s_champ}
    result = _evaluate_sealed(sealed, sealed_labels, systems, runs, float(champion["selected_threshold"]), rule_high)
    result["meta"] = {
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "head": state["head"],
        "prereg_commit": manifest["code_commit"], "data_sha256": manifest["data_sha256"],
        "champion_threshold": float(champion["selected_threshold"]), "settings": frozen_settings(),
        "training": {n: {k: r[k] for k in ("features", "train_rows", "train_fraud", "fit_seconds", "thresholds", "floors",
                                            "calibration_ap")}
                     for n, r in runs.items()},
        "disclosure": DISCLOSURE, "runtime_seconds": round(time.time() - t0),
    }
    # Write the result first: if anything below fails, the one-time result is already on disk.
    RESULT_FILE.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    RESULT_FILE.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    BUNDLE_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model_name": MODEL_CONFIGS[CANDIDATE]["model_name"], "model": runs["E4"]["model"],
                 "feature_names": runs["E4"]["features"], "selected_threshold": float(runs["E4"]["thresholds"][100]),
                 "thresholds_by_budget": runs["E4"]["thresholds"], "queue_floors_by_budget": runs["E4"]["floors"],
                 "saved_at_utc": result["meta"]["generated_utc"],
                 "metadata": {"config_name": CANDIDATE, "config": MODEL_CONFIGS[CANDIDATE], "train_step_range": [1, TRAIN_MAX_STEP],
                              "calibration_steps": list(CALIBRATION), "p15_decision": result["decision"]}}, BUNDLE_FILE)
    result["meta"]["e4_bundle"] = {"path": BUNDLE_FILE.as_posix(), "sha256": file_sha256(BUNDLE_FILE)}
    RESULT_FILE.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    RESULT_FILE.with_suffix(".md").write_text(render_markdown(result), encoding="utf-8")
    return RESULT_FILE


def main() -> None:
    parser = argparse.ArgumentParser(description="P1.5 one-time sealed final test")
    parser.add_argument("--stage", required=True, choices=["preregister", "run"])
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    path = preregister(Path(args.data_path)) if args.stage == "preregister" else run(Path(args.data_path))
    logger.info("wrote %s", path)


if __name__ == "__main__":
    main()
