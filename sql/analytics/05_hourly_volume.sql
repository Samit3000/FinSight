-- ==============================================================================
-- Query 05: Diurnal Transaction Volume and Value Trends
-- Purpose: Analyze platform usage patterns across the 24-hour cycle.
-- Assumption: transaction_hour is derived from step as ((step - 1) % 24).
-- ==============================================================================

SELECT
    transaction_hour,
    COUNT(*) AS transaction_count,
    ROUND(
        (CAST(COUNT(*) AS NUMERIC) / NULLIF(SUM(COUNT(*)) OVER (), 0)) * 100,
        2
    ) AS hourly_volume_share_pct,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(AVG(amount), 2) AS avg_amount,
    COUNT(DISTINCT name_orig) AS distinct_originators
FROM fct_transactions
GROUP BY transaction_hour
ORDER BY transaction_hour ASC;
