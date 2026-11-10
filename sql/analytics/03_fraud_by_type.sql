-- ==============================================================================
-- Query 03: Simulated Fraud Counts and Rates by Transaction Type
-- Purpose: Quantify fraud incidence across payment rails.
-- Caveat: is_fraud represents simulated PaySim labels, not real-world confirmed fraud.
-- ==============================================================================

SELECT
    type AS transaction_type,
    COUNT(*) AS total_transactions,
    COUNT(*) FILTER (WHERE is_fraud = 1) AS simulated_fraud_count,
    COUNT(*) FILTER (WHERE is_fraud = 0) AS simulated_legitimate_count,
    ROUND(
        COALESCE(
            (CAST(COUNT(*) FILTER (WHERE is_fraud = 1) AS NUMERIC) / NULLIF(COUNT(*), 0)) * 100,
            0.0
        ),
        4
    ) AS simulated_fraud_rate_pct,
    -- Flag whether the rail contains simulated fraud
    CASE
        WHEN COUNT(*) FILTER (WHERE is_fraud = 1) > 0 THEN 'FRAUD_EXPOSED'
        ELSE 'ZERO_SIMULATED_FRAUD'
    END AS risk_exposure_category
FROM fct_transactions
GROUP BY type
ORDER BY simulated_fraud_count DESC, total_transactions DESC;
