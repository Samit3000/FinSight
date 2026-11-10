"""
FinSight — Explainable Risk Rule Engine.

Milestone 3: Transparent, deterministic rule-based scoring baseline for transaction
monitoring and payment risk intelligence.

Core Governance Principles:
1. 100% Deterministic: Evaluates transactions purely on transparent business criteria.
2. Label Isolation: NEVER uses `is_fraud` in rule conditions or score computation.
   Ground-truth labels are evaluated separately after scoring.
3. Investigation Signals: Rule scores represent prioritized investigative cues,
   NOT definitive determinations of criminal intent.
4. Configurable Thresholds: All parameters, weights, and cutoffs are fully configurable
   and their empirical limitations explicitly documented.
5. Actionable Reason Codes: Every triggered rule generates a plain-English explanation
   detailing the exact parameter values and thresholds that fired.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("finsight.risk_rules")

# Supported risk tiers and operational action recommendations
RISK_TIERS: Tuple[str, ...] = ("LOW", "MEDIUM", "HIGH")
RECOMMENDED_ACTIONS: Dict[str, str] = {
    "LOW": "AUTO_APPROVE",
    "MEDIUM": "MANUAL_REVIEW",
    "HIGH": "FLAG_SUSPICIOUS",
}


@dataclass(frozen=True)
class RuleDefinition:
    """Specification of an individual explainable risk rule."""
    rule_id: str
    name: str
    description: str
    category: str
    severity: str
    risk_points: int
    enabled: bool = True
    parameters: Dict[str, Any] = field(default_factory=dict)
    rationale: str = ""
    limitations: str = ""


@dataclass
class RuleResult:
    """Outcome of evaluating a single rule against a transaction."""
    rule_id: str
    name: str
    triggered: bool
    points: int
    severity: str
    message: str


@dataclass
class EvaluationResult:
    """Aggregate risk assessment outcome for a single transaction."""
    risk_score: int
    risk_tier: str
    action_recommended: str
    triggered_rule_count: int
    triggered_rules: List[str]
    reason_codes: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "risk_tier": self.risk_tier,
            "action_recommended": self.action_recommended,
            "triggered_rule_count": self.triggered_rule_count,
            "triggered_rules": self.triggered_rules,
            "reason_codes": self.reason_codes,
        }


@dataclass
class RiskEngineConfig:
    """Configuration container for risk scoring framework and rule definitions."""
    baseline_score: int = 0
    max_score: int = 100
    low_cutoff: int = 29      # 0 - 29: LOW
    medium_cutoff: int = 69   # 30 - 69: MEDIUM, 70 - 100: HIGH
    rules: List[RuleDefinition] = field(default_factory=list)

    @classmethod
    def default_config(cls) -> RiskEngineConfig:
        """Construct the canonical production-tested rule set."""
        rules = [
            RuleDefinition(
                rule_id="RULE_001_HIGH_VALUE",
                name="Excessive Single Transaction Amount",
                description="Flags transactions exceeding single-transaction monetary threshold.",
                category="VALUE_ANOMALY",
                severity="HIGH",
                risk_points=35,
                enabled=True,
                parameters={"threshold_amount": 200000.00},
                rationale=(
                    "In PaySim, fraudulent transfers and cash-outs average ~$1.47M, compared to ~$173k "
                    "for legitimate cash-outs. High-value transactions present outsized monetary loss exposure."
                ),
                limitations=(
                    "Large transactions occur legitimately for payroll, B2B procurement, or asset purchases. "
                    "Value alone has a significant false-positive rate."
                ),
            ),
            RuleDefinition(
                rule_id="RULE_002_DRAINAGE_RAIL",
                name="Fund Drainage Rail Exposure",
                description="Flags transactions utilizing CASH_OUT or TRANSFER payment rails.",
                category="RAIL_EXPOSURE",
                severity="MEDIUM",
                risk_points=25,
                enabled=True,
                parameters={"high_risk_rails": ["TRANSFER", "CASH_OUT"]},
                rationale=(
                    "Empirical data in PaySim shows 100% of simulated fraud transactions occur via TRANSFER "
                    "or CASH_OUT rails. PAYMENT, CASH_IN, and DEBIT exhibit zero simulated fraud."
                ),
                limitations=(
                    "Over 99% of TRANSFER and CASH_OUT volume is legitimate consumer activity. "
                    "Rail type alone cannot confirm illicit intent."
                ),
            ),
            RuleDefinition(
                rule_id="RULE_003_OVERNIGHT_DRAINAGE",
                name="Off-Hours Drainage Activity",
                description="Flags transfers or cash-outs initiated during low-volume overnight hours.",
                category="TEMPORAL_ANOMALY",
                severity="HIGH",
                risk_points=25,
                enabled=True,
                parameters={
                    "overnight_hours": [1, 2, 3, 4, 5],
                    "min_amount": 10000.00,
                    "target_rails": ["TRANSFER", "CASH_OUT"],
                },
                rationale=(
                    "Human consumer payment volumes drop by >99% between 1 AM and 5 AM. "
                    "Automated bot scripts and unauthorized account takeovers disproportionately run off-hours."
                ),
                limitations=(
                    "Night-shift workers, automated subscription debits, and cross-border users "
                    "frequently transact overnight without malicious intent."
                ),
            ),
            RuleDefinition(
                rule_id="RULE_004_LARGE_ROUND_AMOUNT",
                name="Elevated Round Amount Disparity",
                description="Flags large round-denomination disbursements on drainage rails.",
                category="STRUCTURAL_ANOMALY",
                severity="MEDIUM",
                risk_points=15,
                enabled=True,
                parameters={
                    "min_amount": 50000.00,
                    "round_modulus": 5000.00,
                    "target_rails": ["TRANSFER", "CASH_OUT"],
                },
                rationale=(
                    "Fraudulent account liquidations frequently test round denomination maximum limits "
                    "(e.g. $50k, $100k daily caps) rather than purchase-exact odd cents."
                ),
                limitations=(
                    "Wholesale funding, round-number inter-account transfers, and real estate deposits "
                    "frequently transact in clean multiples."
                ),
            ),
        ]
        return cls(
            baseline_score=0,
            max_score=100,
            low_cutoff=29,
            medium_cutoff=69,
            rules=rules,
        )


class RiskEngine:
    """Deterministic, explainable transaction risk evaluation engine."""

    def __init__(self, config: Optional[RiskEngineConfig] = None) -> None:
        """Initialize engine with provided or default configuration."""
        self.config = config or RiskEngineConfig.default_config()

    def get_tier(self, score: int) -> str:
        """Map composite risk score to operational risk tier."""
        if score <= self.config.low_cutoff:
            return "LOW"
        elif score <= self.config.medium_cutoff:
            return "MEDIUM"
        else:
            return "HIGH"

    def get_action(self, tier: str) -> str:
        """Map risk tier to standard recommended operational action."""
        return RECOMMENDED_ACTIONS.get(tier, "MANUAL_REVIEW")

    def evaluate_transaction(
        self,
        transaction: Union[Dict[str, Any], pd.Series],
    ) -> EvaluationResult:
        """Evaluate a single transaction deterministically against all active rules.

        Args:
            transaction: Key-value mapping or Series with transaction fields:
                - amount: float (non-negative)
                - type: str (rail name)
                - transaction_hour: int (optional, derived from step if missing)
                - step: int (optional)

        Returns:
            EvaluationResult: Detailed scoring report and audit trail.
        """
        # Extract fields safely (supporting both dict and pd.Series)
        def get_val(key_list: Sequence[str], default: Any = None) -> Any:
            for k in key_list:
                if isinstance(transaction, dict) and k in transaction:
                    return transaction[k]
                elif hasattr(transaction, "index") and k in transaction.index:
                    return transaction[k]
            return default

        amount_raw = get_val(["amount"], 0.0)
        try:
            amount = float(amount_raw)
        except (ValueError, TypeError):
            amount = 0.0

        type_raw = get_val(["type"], "")
        tx_type = str(type_raw).strip().upper()

        step_raw = get_val(["step"], 1)
        try:
            step = int(step_raw)
        except (ValueError, TypeError):
            step = 1

        hour_raw = get_val(["transaction_hour"])
        if hour_raw is not None:
            try:
                hour = int(hour_raw)
            except (ValueError, TypeError):
                hour = (step - 1) % 24
        else:
            hour = (step - 1) % 24

        triggered_rule_ids: List[str] = []
        reason_codes: List[Dict[str, Any]] = []
        total_points = self.config.baseline_score

        for rule in self.config.rules:
            if not rule.enabled:
                continue

            triggered = False
            msg = ""

            if rule.rule_id == "RULE_001_HIGH_VALUE":
                thresh = float(rule.parameters.get("threshold_amount", 200000.00))
                if amount >= thresh:
                    triggered = True
                    msg = (
                        f"Transaction amount ${amount:,.2f} meets or exceeds the single transaction "
                        f"monitoring threshold of ${thresh:,.2f}."
                    )

            elif rule.rule_id == "RULE_002_DRAINAGE_RAIL":
                target_rails = [r.upper() for r in rule.parameters.get("high_risk_rails", ["TRANSFER", "CASH_OUT"])]
                if tx_type in target_rails:
                    triggered = True
                    msg = f"Transaction rail '{tx_type}' matches high-risk fund drainage channel."

            elif rule.rule_id == "RULE_003_OVERNIGHT_DRAINAGE":
                overnight_hours = rule.parameters.get("overnight_hours", [1, 2, 3, 4, 5])
                min_amt = float(rule.parameters.get("min_amount", 10000.00))
                target_rails = [r.upper() for r in rule.parameters.get("target_rails", ["TRANSFER", "CASH_OUT"])]
                if hour in overnight_hours and tx_type in target_rails and amount >= min_amt:
                    triggered = True
                    msg = (
                        f"Disbursement of ${amount:,.2f} via '{tx_type}' executed overnight "
                        f"at hour {hour:02d}:00 (dormancy monitoring window)."
                    )

            elif rule.rule_id == "RULE_004_LARGE_ROUND_AMOUNT":
                min_amt = float(rule.parameters.get("min_amount", 50000.00))
                modulus = float(rule.parameters.get("round_modulus", 5000.00))
                target_rails = [r.upper() for r in rule.parameters.get("target_rails", ["TRANSFER", "CASH_OUT"])]
                if amount >= min_amt and tx_type in target_rails and (amount % modulus == 0):
                    triggered = True
                    msg = (
                        f"Large round amount ${amount:,.2f} on '{tx_type}' is an exact multiple of "
                        f"${modulus:,.2f}, indicating possible limit ceiling extraction."
                    )

            if triggered:
                total_points += rule.risk_points
                triggered_rule_ids.append(rule.rule_id)
                reason_codes.append({
                    "rule_id": rule.rule_id,
                    "name": rule.name,
                    "severity": rule.severity,
                    "points": rule.risk_points,
                    "message": msg,
                })

        final_score = min(self.config.max_score, total_points)
        tier = self.get_tier(final_score)
        action = self.get_action(tier)

        return EvaluationResult(
            risk_score=final_score,
            risk_tier=tier,
            action_recommended=action,
            triggered_rule_count=len(triggered_rule_ids),
            triggered_rules=triggered_rule_ids,
            reason_codes=reason_codes,
        )

    def evaluate_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fast vectorized batch evaluation over a pandas DataFrame.

        Computes risk scores, tiers, actions, and reason lists while preserving
        exact row counts, ordering, and index alignment.

        Args:
            df: Input DataFrame containing transaction records.

        Returns:
            pd.DataFrame: Evaluation results DataFrame with columns:
                - `risk_score` (int 0-100)
                - `risk_tier` (LOW, MEDIUM, HIGH)
                - `action_recommended` (AUTO_APPROVE, MANUAL_REVIEW, FLAG_SUSPICIOUS)
                - `triggered_rule_count` (int)
                - `triggered_rules` (List[str] as comma-separated string)
                - `reasons_summary` (str)
        """
        n_rows = len(df)
        if n_rows == 0:
            return pd.DataFrame(
                columns=[
                    "risk_score",
                    "risk_tier",
                    "action_recommended",
                    "triggered_rule_count",
                    "triggered_rules",
                    "reasons_summary",
                ],
                index=df.index,
            )

        # Parse amounts safely
        amount_col = "amount" if "amount" in df.columns else None
        if amount_col is None:
            raise KeyError("DataFrame must contain 'amount' column.")
        amounts = pd.to_numeric(df[amount_col], errors="coerce").fillna(0.0).values

        # Parse transaction types safely
        if "type" not in df.columns:
            raise KeyError("DataFrame must contain 'type' column.")
        types = df["type"].astype(str).str.strip().str.upper().values

        # Parse or derive hours
        if "transaction_hour" in df.columns:
            hours = pd.to_numeric(df["transaction_hour"], errors="coerce").fillna(0).astype(int).values
        elif "step" in df.columns:
            steps = pd.to_numeric(df["step"], errors="coerce").fillna(1).astype(int).values
            hours = (steps - 1) % 24
        else:
            hours = np.zeros(n_rows, dtype=int)

        scores = np.full(n_rows, self.config.baseline_score, dtype=int)
        rule_hits = [[] for _ in range(n_rows)]
        reason_summaries = [[] for _ in range(n_rows)]

        for rule in self.config.rules:
            if not rule.enabled:
                continue

            mask = np.zeros(n_rows, dtype=bool)

            if rule.rule_id == "RULE_001_HIGH_VALUE":
                thresh = float(rule.parameters.get("threshold_amount", 200000.00))
                mask = amounts >= thresh

            elif rule.rule_id == "RULE_002_DRAINAGE_RAIL":
                target_rails = set(r.upper() for r in rule.parameters.get("high_risk_rails", ["TRANSFER", "CASH_OUT"]))
                mask = np.isin(types, list(target_rails))

            elif rule.rule_id == "RULE_003_OVERNIGHT_DRAINAGE":
                overnight_hours = set(rule.parameters.get("overnight_hours", [1, 2, 3, 4, 5]))
                min_amt = float(rule.parameters.get("min_amount", 10000.00))
                target_rails = set(r.upper() for r in rule.parameters.get("target_rails", ["TRANSFER", "CASH_OUT"]))
                mask = (
                    np.isin(hours, list(overnight_hours))
                    & np.isin(types, list(target_rails))
                    & (amounts >= min_amt)
                )

            elif rule.rule_id == "RULE_004_LARGE_ROUND_AMOUNT":
                min_amt = float(rule.parameters.get("min_amount", 50000.00))
                modulus = float(rule.parameters.get("round_modulus", 5000.00))
                target_rails = set(r.upper() for r in rule.parameters.get("target_rails", ["TRANSFER", "CASH_OUT"]))
                mask = (
                    (amounts >= min_amt)
                    & np.isin(types, list(target_rails))
                    & (np.fmod(amounts, modulus) == 0.0)
                )

            if np.any(mask):
                scores[mask] += rule.risk_points
                hit_indices = np.where(mask)[0]
                for idx in hit_indices:
                    rule_hits[idx].append(rule.rule_id)
                    reason_summaries[idx].append(f"{rule.rule_id}: {rule.name}")

        clamped_scores = np.minimum(self.config.max_score, scores)

        # Vectorized tier assignment
        tiers = np.where(
            clamped_scores <= self.config.low_cutoff,
            "LOW",
            np.where(clamped_scores <= self.config.medium_cutoff, "MEDIUM", "HIGH"),
        )

        actions = np.where(
            tiers == "LOW",
            "AUTO_APPROVE",
            np.where(tiers == "MEDIUM", "MANUAL_REVIEW", "FLAG_SUSPICIOUS"),
        )

        results_df = pd.DataFrame(index=df.index)
        results_df["risk_score"] = clamped_scores
        results_df["risk_tier"] = tiers
        results_df["action_recommended"] = actions
        results_df["triggered_rule_count"] = [len(h) for h in rule_hits]
        results_df["triggered_rules"] = [";".join(h) if h else "NONE" for h in rule_hits]
        results_df["reasons_summary"] = [" | ".join(r) if r else "No rules triggered" for r in reason_summaries]

        return results_df

    def evaluate_against_ground_truth(
        self,
        evaluation_df: pd.DataFrame,
        y_true: Union[pd.Series, Sequence[int]],
        flagged_tier: str = "HIGH",
        min_score: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Benchmark rule engine operational flags against external ground-truth labels.

        Strict Separation Note:
        Ground-truth labels are NEVER passed into rule evaluation; this audit method
        is strictly called post-hoc to measure detection metrics.

        Args:
            evaluation_df: Output DataFrame from `evaluate_dataframe`.
            y_true: Binary ground-truth labels (0 = legitimate, 1 = fraud).
            flagged_tier: Risk tier considered positive alert (default: 'HIGH', score >= 70).
            min_score: Optional explicit numeric score cutoff. If provided, overrides flagged_tier.

        Returns:
            Dict containing confusion matrix, precision, recall, F1 score,
            tier distributions, and action recommendation counts.
        """
        y_arr = np.asarray(y_true, dtype=int)
        if len(evaluation_df) != len(y_arr):
            raise ValueError(
                f"Row mismatch: evaluations ({len(evaluation_df)}) != ground truth ({len(y_arr)})"
            )

        total = len(y_arr)
        total_frauds = int(np.sum(y_arr == 1))

        # Determine decision threshold
        if min_score is not None:
            threshold_desc = f"risk_score >= {min_score}"
            y_pred = (evaluation_df["risk_score"] >= min_score).astype(int).values
        else:
            threshold_desc = f"risk_tier == '{flagged_tier}' (score >= {self.config.medium_cutoff + 1 if flagged_tier == 'HIGH' else self.config.low_cutoff + 1})"
            y_pred = (evaluation_df["risk_tier"] == flagged_tier).astype(int).values

        tp = int(np.sum((y_pred == 1) & (y_arr == 1)))
        fp = int(np.sum((y_pred == 1) & (y_arr == 0)))
        tn = int(np.sum((y_pred == 0) & (y_arr == 0)))
        fn = int(np.sum((y_pred == 0) & (y_arr == 1)))

        total_flagged = int(np.sum(y_pred == 1))

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )

        tier_counts = evaluation_df["risk_tier"].value_counts().to_dict()
        tier_distribution = {
            t: {
                "count": int(tier_counts.get(t, 0)),
                "pct": round(int(tier_counts.get(t, 0)) / total * 100, 4) if total > 0 else 0.0,
            }
            for t in ["LOW", "MEDIUM", "HIGH"]
        }

        action_counts = evaluation_df["action_recommended"].value_counts().to_dict()
        action_recommendations = {
            a: {
                "count": int(action_counts.get(a, 0)),
                "pct": round(int(action_counts.get(a, 0)) / total * 100, 4) if total > 0 else 0.0,
            }
            for a in ["AUTO_APPROVE", "MANUAL_REVIEW", "FLAG_SUSPICIOUS"]
        }

        return {
            "total_transactions": total,
            "total_frauds": total_frauds,
            "decision_threshold": threshold_desc,
            "total_flagged_high_risk": total_flagged,
            "flagged_rate_pct": round(total_flagged / total * 100, 4) if total > 0 else 0.0,
            "tier_distribution": tier_distribution,
            "action_recommendations": action_recommendations,
            "confusion_matrix": {
                "true_positives": tp,
                "false_positives": fp,
                "true_negatives": tn,
                "false_negatives": fn,
            },
            "metrics": {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
            },
            "interpretation_notes": (
                "Rule engine flags represent operational alerts for investigation, not automated block decisions. "
                "Low precision reflects deliberate trade-off to capture outsized monetary loss without ML optimization."
            ),
        }
