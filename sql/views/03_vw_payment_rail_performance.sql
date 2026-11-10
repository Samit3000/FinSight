-- ==============================================================================
-- FinSight Analytical Mart: Payment Rail Performance & Risk Distribution
-- View Name: vw_payment_rail_performance
-- Grain: One row per payment rail / transaction type (5 rows).
--
-- Description:
-- Performance, financial throughput, and fraud concentration across all 5
-- payment rails (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER).
--
-- Financial & Risk Semantics:
-- - gross_transaction_value: Total transaction volume processed on this rail.
-- - labelled_fraud_count: Count of simulated fraud events on this rail.
--   Note: In PaySim, simulated fraud is strictly restricted to CASH_OUT and TRANSFER.
-- - gross_labelled_fraud_value: Total face value of fraudulent transactions on this rail.
-- - labelled_fraud_rate_pct: Percentage of transactions on this rail that are fraud.
-- - fraud_value_exposure_share_pct: Share of this rail's dollar volume involved in fraud.
-- - rail_volume_share_pct: Proportion of total platform transactions processed on this rail.
-- - rail_value_share_pct: Proportion of total platform dollar volume processed on this rail.
--
-- Source: fct_transactions
-- Idempotency: CREATE OR REPLACE VIEW
-- ==============================================================================

CREATE OR REPLACE VIEW vw_payment_rail_performance AS
SELECT
    type AS payment_rail,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS gross_transaction_value,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    COUNT(*) FILTER (WHERE is_fraud = 1) AS labelled_fraud_count,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1), 0.00), 2) AS gross_labelled_fraud_value,
    ROUND(
        COALESCE(
            (CAST(COUNT(*) FILTER (WHERE is_fraud = 1) AS NUMERIC) / NULLIF(COUNT(*), 0)) * 100,
            0.0
        ),
        4
    ) AS labelled_fraud_rate_pct,
    ROUND(
        COALESCE(
            (SUM(amount) FILTER (WHERE is_fraud = 1) / NULLIF(SUM(amount), 0)) * 100,
            0.0
        ),
        4
    ) AS fraud_value_exposure_share_pct,
    ROUND(
        COALESCE(
            (CAST(COUNT(*) AS NUMERIC) / NULLIF(SUM(COUNT(*)) OVER (), 0)) * 100,
            0.0
        ),
        4
    ) AS rail_volume_share_pct,
    ROUND(
        COALESCE(
            (SUM(amount) / NULLIF(SUM(SUM(amount)) OVER (), 0)) * 100,
            0.0
        ),
        4
    ) AS rail_value_share_pct
FROM fct_transactions
GROUP BY type
ORDER BY gross_transaction_value DESC;

COMMENT ON VIEW vw_payment_rail_performance IS
'Payment rail throughput, average transaction size, and simulated fraud exposure breakdown across all 5 payment types.';
