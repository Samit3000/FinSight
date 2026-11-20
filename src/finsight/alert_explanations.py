"""
FinSight — P2.3 reason codes for every champion alert.

For each transaction the active champion (E4) alerts on, one record with two clearly separated parts:

* ``model_explanation`` — E4's risk score and operating threshold, and the top three TreeSHAP
  contributors (feature, value, contribution in log-odds, direction). These describe how the model's
  inputs moved this score relative to the model's average; they are **not** evidence that the
  transaction is fraudulent.
* ``rule_conditions`` — every rule of the explainable rule engine whose condition the transaction meets,
  with its points and message, or an explicit "none". The P2 study found that rule-based extra alerts do
  not beat giving E4 the same review capacity, so rules here explain; they do not raise alerts.

Labels are never read: alerts depend only on the champion's score. Recipient history is computed by
``finsight.champion.score_frame`` over the full chronological frame (which must start at step 1).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

_src_root = str(Path(__file__).resolve().parent.parent)
if _src_root not in sys.path:
    sys.path.insert(0, _src_root)

from finsight.champion import load_champion, operating_threshold, score_frame  # noqa: E402
from finsight.evaluate import FINAL_TEST_MIN_STEP, STEPS_PER_DAY  # noqa: E402
from finsight.risk_rules import RiskEngine  # noqa: E402
from finsight.train import add_history_features, build_features  # noqa: E402

logger = logging.getLogger("finsight.alert_explanations")

MODEL_DISCLAIMER = (
    "Model attribution: shows how each input moved this transaction's model score relative to the model's "
    "average (log-odds). It is not evidence that the transaction is fraudulent and does not establish a cause."
)
RULE_NOTE = (
    "Rule conditions met: deterministic checks from the rule engine, shown for context. They do not raise "
    "alerts in FinSight and are not evidence of fraud on their own."
)
CAUSAL_PHRASES = (
    "is fraud because", "proves", "proof of fraud", "caused", "because it is fraud", "confirms fraud",
    "is fraudulent because", "evidence of fraud.", "guilty",
)

_FEATURE_TEXT = {
    "amount": lambda v: f"amount ${v:,.2f}",
    "transaction_hour": lambda v: f"hour {int(v):02d}:00",
    "recipient_prior_in_count": lambda v: ("first-time recipient (no earlier payments received)" if int(v) == 0
                                           else f"recipient with {int(v)} earlier payment{'s' if int(v) != 1 else ''}"),
}


def _feature_phrase(feature: str, value: float) -> str:
    if feature in _FEATURE_TEXT:
        return _FEATURE_TEXT[feature](value)
    if feature.startswith("type_"):
        rail = feature[5:]
        return f"type is {rail}" if value >= 0.5 else f"type is not {rail}"
    return f"{feature} = {value}"


def assert_non_causal(text: str) -> None:
    low = text.lower()
    hits = [p for p in CAUSAL_PHRASES if p in low]
    if hits:
        raise ValueError(f"causal wording not allowed in explanations: {hits}")


def top_contributors(shap_row: np.ndarray, values: Sequence[float], feature_names: Sequence[str], k: int = 3) -> List[Dict[str, Any]]:
    """Top-k features by absolute SHAP value (ties broken by feature order)."""
    order = sorted(range(len(feature_names)), key=lambda i: (-abs(float(shap_row[i])), i))[:k]
    return [{"feature": feature_names[i], "value": float(values[i]), "contribution_log_odds": round(float(shap_row[i]), 4),
             "direction": "raised the score" if shap_row[i] > 0 else "lowered the score",
             "description": _feature_phrase(feature_names[i], float(values[i]))} for i in order]


def explain_alerts(frame: pd.DataFrame, min_step: int, max_step: int, alerts_per_day: Optional[int] = None,
                   champion: Optional[Dict[str, Any]] = None, top_k: int = 3) -> List[Dict[str, Any]]:
    """Reason-code records for every champion alert with min_step <= step <= max_step.

    `frame` must be the full chronological frame from step 1 (history), without steps >= 651.
    """
    import shap

    if max_step >= FINAL_TEST_MIN_STEP or int(frame["step"].max()) >= FINAL_TEST_MIN_STEP:
        raise ValueError(f"steps >= {FINAL_TEST_MIN_STEP} (retired sealed test) are not allowed")
    champ = champion or load_champion()
    cfg, model, names = champ["config"], champ["model"], champ["feature_names"]
    threshold = operating_threshold(champ, alerts_per_day)
    frame = frame.drop(columns=[c for c in ("isFraud", "is_fraud") if c in frame.columns])
    scores = score_frame(champ, frame)
    window = ((frame["step"] >= min_step) & (frame["step"] <= max_step)).to_numpy()
    alert = window & (scores >= threshold)
    if not alert.any():
        return []
    hist = add_history_features(frame, cfg)
    rows = hist[alert]
    X = build_features(rows, cfg)[names]
    explainer = shap.TreeExplainer(model)
    sv = np.asarray(explainer.shap_values(X))
    sv = sv if sv.ndim == 2 else sv[..., 1]
    base = float(np.atleast_1d(explainer.expected_value)[0])
    margin = model.decision_function(X)
    engine = RiskEngine()
    alert_scores = scores[alert]
    out: List[Dict[str, Any]] = []
    for i, (idx, row) in enumerate(rows.iterrows()):
        contrib = top_contributors(sv[i], X.iloc[i].to_numpy(dtype=float), names, top_k)
        rule = engine.evaluate_transaction(row)
        summary = (f"Model score {alert_scores[i]:.4f} reached the alert threshold {threshold:.4f}. The inputs that moved the "
                   f"score most were: " + "; ".join(f"{c['description']} ({c['direction']}, {c['contribution_log_odds']:+.2f})"
                                                   for c in contrib) + ".")
        assert_non_causal(summary)
        rec = {
            "step": int(row["step"]), "simulated_day": int((row["step"] - 1) // STEPS_PER_DAY + 1),
            "hour": int(row["transaction_hour"]), "type": str(row["type"]), "amount": float(row["amount"]),
            "recipient": str(row.get("nameDest", row.get("name_dest", ""))),
            "alert_source": "champion model (E4)",
            "model_explanation": {
                "score": round(float(alert_scores[i]), 6), "threshold": round(threshold, 6),
                "base_value_log_odds": round(base, 4), "margin_log_odds": round(float(margin[i]), 4),
                "top_contributors": contrib, "summary": summary, "disclaimer": MODEL_DISCLAIMER,
                "additivity_error": float(abs(base + sv[i].sum() - margin[i])),
            },
            "rule_conditions": {
                "rules_met": [{"rule_id": rc["rule_id"], "name": rc["name"], "points": rc["points"], "message": rc["message"]}
                              for rc in rule.reason_codes],
                "rule_ids": rule.triggered_rules or ["none"],
                "rule_score": rule.risk_score, "rule_tier": rule.risk_tier, "note": RULE_NOTE,
            },
        }
        out.append(rec)
    return out


def render_markdown(records: List[Dict[str, Any]], meta: Dict[str, Any]) -> str:
    L = ["# Alert reason codes — sample", "",
         f"Champion `{meta['artifact']}` · steps {meta['min_step']}–{meta['max_step']} (simulated day {meta['day']}) · "
         f"threshold {meta['threshold']:.4f} · {len(records)} alerts · generated {meta['generated_utc']}", "",
         "> " + MODEL_DISCLAIMER, ">", "> " + RULE_NOTE, "",
         "Steps 551–650 are the champion's threshold-calibration window, so this sample illustrates the format; it is "
         "not a performance estimate. Labels were not read.", "",
         "| Step | Hour | Type | Amount | Score | Top model contributors (log-odds) | Rule conditions met |",
         "|---:|---:|---|---:|---:|---|---|"]
    for r in records:
        m, rc = r["model_explanation"], r["rule_conditions"]
        contrib = "; ".join(f"{c['description']} {c['contribution_log_odds']:+.2f}" for c in m["top_contributors"])
        L.append(f"| {r['step']} | {r['hour']:02d} | {r['type']} | ${r['amount']:,.0f} | {m['score']:.4f} | {contrib} | "
                 f"{', '.join(rc['rule_ids'])} |")
    n_rules = sum(1 for r in records if r["rule_conditions"]["rule_ids"] != ["none"])
    L += ["", f"{n_rules} of {len(records)} alerts meet at least one rule condition.", ""]
    return "\n".join(L)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reason codes for champion alerts on one development day")
    parser.add_argument("--day", type=int, default=25, help="Simulated day (must end before step 651)")
    parser.add_argument("--data-path", default="data/processed/transactions.csv.gz")
    parser.add_argument("--out-dir", default="reports/explainability")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    lo, hi = (args.day - 1) * STEPS_PER_DAY + 1, args.day * STEPS_PER_DAY
    if hi >= FINAL_TEST_MIN_STEP:
        raise SystemExit(f"day {args.day} reaches step {hi}; the sealed test (>= {FINAL_TEST_MIN_STEP}) is retired")
    cols = ["step", "type", "amount", "transaction_hour", "nameDest"]
    frame = pd.read_csv(args.data_path, usecols=cols)
    frame = frame[frame["step"] <= hi].reset_index(drop=True)
    champ = load_champion()
    records = explain_alerts(frame, lo, hi, champion=champ)
    meta = {"artifact": champ["entry"]["artifact"], "artifact_sha256": champ["entry"]["artifact_sha256"],
            "min_step": lo, "max_step": hi, "day": args.day, "threshold": operating_threshold(champ),
            "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "alerts": len(records)}
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "alert_reason_codes_sample.json").write_text(json.dumps({"meta": meta, "alerts": records}, indent=1), encoding="utf-8")
    (out / "alert_reason_codes_sample.md").write_text(render_markdown(records, meta), encoding="utf-8")
    logger.info("wrote %d alert records for day %d", len(records), args.day)


if __name__ == "__main__":
    main()
