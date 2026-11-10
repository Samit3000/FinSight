"""
FinSight — Milestone 3 Explainable Risk Rule Engine Automated Tests.

Tests rule determinism, scoring calculations, reason code explanations,
boundary conditions, label isolation, and batch evaluation equivalence.
"""

import pandas as pd
import pytest

from finsight.risk_rules import (
    RECOMMENDED_ACTIONS,
    RISK_TIERS,
    EvaluationResult,
    RiskEngine,
    RiskEngineConfig,
    RuleDefinition,
)


@pytest.fixture
def engine() -> RiskEngine:
    """Fixture providing initialized RiskEngine with default configuration."""
    return RiskEngine()


def test_zero_rule_trigger_normal_payment(engine: RiskEngine):
    """Verify standard low-value payment triggers zero rules and auto-approves."""
    txn = {
        "amount": 45.00,
        "type": "PAYMENT",
        "transaction_hour": 14,
        "step": 15,
    }
    result = engine.evaluate_transaction(txn)

    assert result.risk_score == 0
    assert result.risk_tier == "LOW"
    assert result.action_recommended == "AUTO_APPROVE"
    assert result.triggered_rule_count == 0
    assert result.triggered_rules == []
    assert result.reason_codes == []


def test_rule_score_determinism(engine: RiskEngine):
    """Verify evaluating identical transaction repeatedly yields identical scores and reasons."""
    txn = {
        "amount": 350000.00,
        "type": "TRANSFER",
        "transaction_hour": 3,
        "step": 4,
    }
    res1 = engine.evaluate_transaction(txn)
    res2 = engine.evaluate_transaction(txn)

    assert res1.risk_score == res2.risk_score
    assert res1.risk_tier == res2.risk_tier
    assert res1.triggered_rules == res2.triggered_rules
    assert res1.reason_codes == res2.reason_codes


def test_single_rule_drainage_rail(engine: RiskEngine):
    """Verify modest CASH_OUT triggers only the drainage rail rule (25 points -> LOW)."""
    txn = {
        "amount": 250.00,
        "type": "CASH_OUT",
        "transaction_hour": 15,
        "step": 16,
    }
    result = engine.evaluate_transaction(txn)

    assert result.risk_score == 25
    assert result.risk_tier == "LOW"
    assert result.action_recommended == "AUTO_APPROVE"
    assert result.triggered_rule_count == 1
    assert result.triggered_rules == ["RULE_002_DRAINAGE_RAIL"]
    assert "CASH_OUT" in result.reason_codes[0]["message"]


def test_multi_rule_high_value_drainage(engine: RiskEngine):
    """Verify large daytime transfer triggers high value and drainage rail (35 + 25 = 60 -> MEDIUM)."""
    txn = {
        "amount": 245123.45,
        "type": "TRANSFER",
        "transaction_hour": 12,
        "step": 13,
    }
    result = engine.evaluate_transaction(txn)

    assert result.risk_score == 60  # 35 (high value) + 25 (drainage rail)
    assert result.risk_tier == "MEDIUM"
    assert result.action_recommended == "MANUAL_REVIEW"
    assert result.triggered_rule_count == 2
    assert result.triggered_rules == ["RULE_001_HIGH_VALUE", "RULE_002_DRAINAGE_RAIL"]


def test_critical_overnight_all_rules_trigger_high(engine: RiskEngine):
    """Verify massive overnight round transfer hits all rules and clamps to 100 (HIGH)."""
    txn = {
        "amount": 500000.00,  # Hits RULE_001 (>=200k) and RULE_004 (mod 5000 == 0)
        "type": "TRANSFER",    # Hits RULE_002
        "transaction_hour": 3, # Hits RULE_003 (overnight 1-5, amount >= 10k)
        "step": 4,
    }
    result = engine.evaluate_transaction(txn)

    # 35 (high value) + 25 (drainage) + 25 (overnight) + 15 (round) = 100
    assert result.risk_score == 100
    assert result.risk_tier == "HIGH"
    assert result.action_recommended == "FLAG_SUSPICIOUS"
    assert result.triggered_rule_count == 4
    assert len(result.reason_codes) == 4


def test_label_isolation_ignores_is_fraud(engine: RiskEngine):
    """Verify risk engine produces identical results regardless of is_fraud value."""
    txn_fraud_0 = {
        "amount": 100.00,
        "type": "PAYMENT",
        "transaction_hour": 12,
        "is_fraud": 0,
        "isFraud": 0,
    }
    txn_fraud_1 = {
        "amount": 100.00,
        "type": "PAYMENT",
        "transaction_hour": 12,
        "is_fraud": 1,
        "isFraud": 1,
    }

    res0 = engine.evaluate_transaction(txn_fraud_0)
    res1 = engine.evaluate_transaction(txn_fraud_1)

    assert res0.risk_score == res1.risk_score == 0
    assert res0.risk_tier == res1.risk_tier == "LOW"


def test_boundary_threshold_amount(engine: RiskEngine):
    """Verify behavior exactly at, slightly below, and slightly above threshold."""
    thresh = 200000.00

    # 1 cent below threshold
    txn_below = {"amount": thresh - 0.01, "type": "DEBIT", "transaction_hour": 12}
    # Exactly at threshold
    txn_at = {"amount": thresh, "type": "DEBIT", "transaction_hour": 12}
    # 1 cent above threshold
    txn_above = {"amount": thresh + 0.01, "type": "DEBIT", "transaction_hour": 12}

    res_below = engine.evaluate_transaction(txn_below)
    res_at = engine.evaluate_transaction(txn_at)
    res_above = engine.evaluate_transaction(txn_above)

    assert "RULE_001_HIGH_VALUE" not in res_below.triggered_rules
    assert "RULE_001_HIGH_VALUE" in res_at.triggered_rules
    assert "RULE_001_HIGH_VALUE" in res_above.triggered_rules


def test_extreme_and_invalid_values(engine: RiskEngine):
    """Verify handling of zero amounts, negative values, and extreme numbers."""
    # Zero amount
    res_zero = engine.evaluate_transaction({"amount": 0.0, "type": "PAYMENT"})
    assert res_zero.risk_score == 0

    # Negative amount handled gracefully (coerced to 0.0)
    res_neg = engine.evaluate_transaction({"amount": -100.0, "type": "PAYMENT"})
    assert res_neg.risk_score == 0

    # Extremely large amount
    res_huge = engine.evaluate_transaction({"amount": 1e12, "type": "TRANSFER", "transaction_hour": 2})
    assert res_huge.risk_score <= 100
    assert res_huge.risk_tier == "HIGH"


def test_evaluate_dataframe_equivalence(engine: RiskEngine):
    """Verify vectorized evaluate_dataframe matches row-by-row evaluate_transaction."""
    df = pd.DataFrame([
        {"amount": 50.00, "type": "PAYMENT", "transaction_hour": 10},
        {"amount": 1500.00, "type": "CASH_OUT", "transaction_hour": 14},
        {"amount": 300000.00, "type": "TRANSFER", "transaction_hour": 3},
    ])

    batch_res = engine.evaluate_dataframe(df)

    assert len(batch_res) == 3
    for i in range(len(df)):
        single_res = engine.evaluate_transaction(df.iloc[i])
        assert batch_res["risk_score"].iloc[i] == single_res.risk_score
        assert batch_res["risk_tier"].iloc[i] == single_res.risk_tier
        assert batch_res["action_recommended"].iloc[i] == single_res.action_recommended
        assert batch_res["triggered_rule_count"].iloc[i] == single_res.triggered_rule_count


def test_evaluate_against_ground_truth(engine: RiskEngine):
    """Verify post-hoc ground truth evaluation metrics computation."""
    df = pd.DataFrame([
        {"amount": 50.00, "type": "PAYMENT", "transaction_hour": 10},     # Score 0 (LOW)
        {"amount": 1500.00, "type": "CASH_OUT", "transaction_hour": 14},  # Score 25 (LOW)
        {"amount": 500000.00, "type": "TRANSFER", "transaction_hour": 3}, # Score 100 (HIGH)
    ])
    eval_df = engine.evaluate_dataframe(df)
    y_true = pd.Series([0, 0, 1])  # 3rd row is actual fraud

    metrics = engine.evaluate_against_ground_truth(eval_df, y_true, flagged_tier="HIGH")

    assert metrics["total_transactions"] == 3
    assert metrics["total_frauds"] == 1
    assert metrics["total_flagged_high_risk"] == 1
    assert metrics["tier_distribution"]["LOW"]["count"] == 2
    assert metrics["tier_distribution"]["HIGH"]["count"] == 1
    assert metrics["action_recommendations"]["AUTO_APPROVE"]["count"] == 2
    assert metrics["action_recommendations"]["FLAG_SUSPICIOUS"]["count"] == 1
    assert metrics["confusion_matrix"]["true_positives"] == 1
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["metrics"]["precision"] == 1.0
    assert metrics["metrics"]["recall"] == 1.0
    assert metrics["metrics"]["f1_score"] == 1.0


def test_evaluate_against_ground_truth_min_score(engine: RiskEngine):
    """Verify ground truth evaluation using explicit min_score cutoff."""
    df = pd.DataFrame([
        {"amount": 50.00, "type": "PAYMENT", "transaction_hour": 10},     # Score 0 (LOW)
        {"amount": 1500.00, "type": "CASH_OUT", "transaction_hour": 14},  # Score 25 (LOW)
        {"amount": 500000.00, "type": "TRANSFER", "transaction_hour": 3}, # Score 100 (HIGH)
    ])
    eval_df = engine.evaluate_dataframe(df)
    y_true = pd.Series([0, 1, 1])

    # Threshold min_score = 25 captures CASH_OUT (25) and TRANSFER (100)
    metrics = engine.evaluate_against_ground_truth(eval_df, y_true, min_score=25)

    assert metrics["decision_threshold"] == "risk_score >= 25"
    assert metrics["total_flagged_high_risk"] == 2
    assert metrics["confusion_matrix"]["true_positives"] == 2
    assert metrics["confusion_matrix"]["false_positives"] == 0
    assert metrics["metrics"]["recall"] == 1.0
