-- ==============================================================================
-- Query 01: Executive Summary & Platform Totals
-- Purpose: High-level KPI summary of transactional volume, financial value,
-- entity cardinality, and simulated fraud prevalence across the entire dataset.
-- ==============================================================================

SELECT
    COUNT(*) AS total_transactions,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    COUNT(DISTINCT name_orig) AS distinct_origin_accounts,
    COUNT(DISTINCT name_dest) AS distinct_destination_accounts,
    COUNT(*) FILTER (WHERE is_fraud = 1) AS total_simulated_fraud_transactions,
    ROUND(
        COALESCE(
            (CAST(COUNT(*) FILTER (WHERE is_fraud = 1) AS NUMERIC) / NULLIF(COUNT(*), 0)) * 100,
            0.0
        ),
        4
    ) AS simulated_fraud_rate_pct,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1), 0.00), 2) AS total_simulated_fraud_amount,
    ROUND(
        COALESCE(
            (SUM(amount) FILTER (WHERE is_fraud = 1) / NULLIF(SUM(amount), 0)) * 100,
            0.0
        ),
        4
    ) AS simulated_fraud_amount_share_pct
FROM fct_transactions;
