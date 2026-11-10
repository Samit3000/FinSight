-- ==============================================================================
-- Query 02: Transaction Volume & Financial Value by Rail / Type
-- Purpose: Segment platform activity across PaySim transaction categories
-- to evaluate volume and monetary concentration.
-- ==============================================================================

SELECT
    type AS transaction_type,
    COUNT(*) AS transaction_count,
    ROUND(
        (CAST(COUNT(*) AS NUMERIC) / NULLIF(SUM(COUNT(*)) OVER (), 0)) * 100,
        2
    ) AS volume_share_pct,
    ROUND(SUM(amount), 2) AS total_amount,
    ROUND(
        (CAST(SUM(amount) AS NUMERIC) / NULLIF(SUM(SUM(amount)) OVER (), 0)) * 100,
        2
    ) AS amount_share_pct,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    ROUND(MIN(amount), 2) AS min_amount,
    ROUND(MAX(amount), 2) AS max_amount
FROM fct_transactions
GROUP BY type
ORDER BY transaction_count DESC;
