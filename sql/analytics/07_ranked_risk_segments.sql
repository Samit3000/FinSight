-- ==============================================================================
-- Query 07: Ranked Risk Segments by Simulated Fraud Rate with Sample Guardrails
-- Purpose: Rank temporal segments (day of simulation + hour of day) by simulated
-- fraud rate, prominently exposing transaction counts to prevent small-denominator
-- distortion (e.g. 1 fraud out of 2 transactions looking artificially critical).
-- ==============================================================================

WITH segment_metrics AS (
    SELECT
        transaction_day,
        transaction_hour,
        type AS transaction_type,
        COUNT(*) AS total_transactions,
        COUNT(*) FILTER (WHERE is_fraud = 1) AS simulated_fraud_count,
        ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1), 0.00), 2) AS simulated_fraud_amount,
        ROUND(
            COALESCE(
                (CAST(COUNT(*) FILTER (WHERE is_fraud = 1) AS NUMERIC) / NULLIF(COUNT(*), 0)) * 100,
                0.0
            ),
            4
        ) AS simulated_fraud_rate_pct
    FROM fct_transactions
    WHERE type IN ('CASH_OUT', 'TRANSFER')
    GROUP BY transaction_day, transaction_hour, type
)
SELECT
    DENSE_RANK() OVER (ORDER BY simulated_fraud_rate_pct DESC, simulated_fraud_count DESC) AS risk_rank,
    transaction_day,
    transaction_hour,
    transaction_type,
    simulated_fraud_rate_pct,
    simulated_fraud_count,
    total_transactions,
    simulated_fraud_amount,
    -- Sample size reliability flag
    CASE
        WHEN total_transactions >= 500 THEN 'HIGH_SAMPLE_STABLE'
        WHEN total_transactions >= 50 THEN 'MODERATE_SAMPLE'
        ELSE 'LOW_SAMPLE_VOLATILE'
    END AS sample_reliability
FROM segment_metrics
WHERE simulated_fraud_count > 0
ORDER BY simulated_fraud_rate_pct DESC, simulated_fraud_count DESC
LIMIT 50;
