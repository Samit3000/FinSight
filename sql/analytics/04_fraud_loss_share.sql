-- ==============================================================================
-- Query 04: Simulated Fraud Value and Financial Share by Transaction Rail
-- Purpose: Measure total financial exposure associated with simulated fraud
-- relative to gross processed volume.
-- ==============================================================================

SELECT
    type AS transaction_type,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1), 0.00), 2) AS simulated_fraud_amount,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 0), 0.00), 2) AS simulated_legitimate_amount,
    ROUND(
        COALESCE(
            (SUM(amount) FILTER (WHERE is_fraud = 1) / NULLIF(SUM(amount), 0)) * 100,
            0.0
        ),
        4
    ) AS fraud_amount_share_pct,
    ROUND(
        COALESCE(
            AVG(amount) FILTER (WHERE is_fraud = 1),
            0.00
        ),
        2
    ) AS avg_fraud_transaction_amount,
    ROUND(
        COALESCE(
            AVG(amount) FILTER (WHERE is_fraud = 0),
            0.00
        ),
        2
    ) AS avg_legitimate_transaction_amount
FROM fct_transactions
GROUP BY type
ORDER BY simulated_fraud_amount DESC;
