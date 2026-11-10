-- ==============================================================================
-- Query 06: Simulated Fraud Rates by Hour of Day and Transaction Type
-- Purpose: Identify temporal risk hotspots across high-risk transaction types.
-- Caveat: Uses NULLIF to prevent division by zero; rates reflect simulated labels.
-- ==============================================================================

SELECT
    transaction_hour,
    type AS transaction_type,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (WHERE is_fraud = 1) AS simulated_fraud_count,
    ROUND(
        COALESCE(
            (CAST(COUNT(*) FILTER (WHERE is_fraud = 1) AS NUMERIC) / NULLIF(COUNT(*), 0)) * 100,
            0.0
        ),
        4
    ) AS simulated_fraud_rate_pct,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1), 0.00), 2) AS simulated_fraud_amount
FROM fct_transactions
WHERE type IN ('CASH_OUT', 'TRANSFER')
GROUP BY transaction_hour, type
ORDER BY transaction_hour ASC, type ASC;
