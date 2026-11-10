-- ==============================================================================
-- FinSight Analytical Mart: Daily Transaction & Fraud KPIs
-- View Name: vw_daily_transaction_kpis
-- Grain: One row per simulation day (transaction_day, 1 to 31).
--
-- Description:
-- Aggregates transaction volumes, processed monetary values, and simulated
-- fraud prevalence across the 31-day simulation period.
--
-- Financial Semantics & Metrics:
-- - gross_transaction_value: Total payment volume across all rails (GTV).
-- - gross_labelled_fraud_value: Face value of all transactions with is_fraud = 1.
--   Note: Represents event-level gross exposure; does not account for paired
--   TRANSFER -> CASH_OUT simulator attack legs or post-fraud recovery.
-- - fraud_transfer_face_value: Stage-1 infiltration face value (TRANSFER & is_fraud=1).
-- - fraud_transaction_rate_pct: Ratio of fraud transactions to total transactions
--   processed on that simulation day (denominator: transaction_count).
--
-- Source: fct_transactions
-- Idempotency: CREATE OR REPLACE VIEW
-- ==============================================================================

CREATE OR REPLACE VIEW vw_daily_transaction_kpis AS
SELECT
    transaction_day AS simulation_day,
    COUNT(*) AS transaction_count,
    ROUND(SUM(amount), 2) AS gross_transaction_value,
    ROUND(AVG(amount), 2) AS avg_transaction_amount,
    COUNT(*) FILTER (WHERE is_fraud = 1) AS labelled_fraud_transaction_count,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1), 0.00), 2) AS gross_labelled_fraud_value,
    ROUND(
        COALESCE(
            (CAST(COUNT(*) FILTER (WHERE is_fraud = 1) AS NUMERIC) / NULLIF(COUNT(*), 0)) * 100,
            0.0
        ),
        4
    ) AS fraud_transaction_rate_pct,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1 AND type = 'TRANSFER'), 0.00), 2) AS fraud_transfer_face_value,
    ROUND(
        COALESCE(
            (SUM(amount) FILTER (WHERE is_fraud = 1) / NULLIF(SUM(amount), 0)) * 100,
            0.0
        ),
        4
    ) AS fraud_value_exposure_share_pct
FROM fct_transactions
GROUP BY transaction_day
ORDER BY simulation_day ASC;

COMMENT ON VIEW vw_daily_transaction_kpis IS
'Daily aggregated transaction volumes, gross financial values, and simulated fraud exposure rates across the 31-day simulation period.';
