-- ==============================================================================
-- FinSight Analytical Mart: Hourly Risk Dynamics
-- View Name: vw_hourly_risk_dynamics
-- Grain: One row per simulated hour (step, 1 to 743).
--
-- Description:
-- Hourly transaction activity, monetary throughput, and simulated fraud
-- prevalence across all 743 simulation hours.
--
-- Temporal Semantics & Simulation Constraints:
-- - step represents 1-indexed simulation hours (1 to 743, spanning 31 days).
-- - simulation_day represents (((step - 1) / 24) + 1).
-- - hour_of_day represents ((step - 1) % 24), simulating a 0-23 diurnal cycle.
-- - CRITICAL: step-derived hours are synthetic simulation intervals; they MUST
--   NOT be interpreted as UTC or local calendar timestamps, nor do they reflect
--   a real-world timezone.
--
-- Model Alert Telemetry:
-- - Model scoring is executed in the isolated ML evaluation partition (step >= 551).
-- - In the absence of a materialized transaction-level model scoring table,
--   alert metrics are not joined on unverified keys, preventing data fabrication.
--
-- Source: fct_transactions
-- Idempotency: CREATE OR REPLACE VIEW
-- ==============================================================================

CREATE OR REPLACE VIEW vw_hourly_risk_dynamics AS
SELECT
    step AS simulated_hour_step,
    transaction_day AS simulation_day,
    transaction_hour AS hour_of_day,
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
    ) AS labelled_fraud_prevalence_pct,
    ROUND(COALESCE(SUM(amount) FILTER (WHERE is_fraud = 1 AND type = 'TRANSFER'), 0.00), 2) AS fraud_transfer_face_value
FROM fct_transactions
GROUP BY step, transaction_day, transaction_hour
ORDER BY step ASC;

COMMENT ON VIEW vw_hourly_risk_dynamics IS
'Hourly transaction volumes, monetary values, and simulated fraud prevalence across all 743 simulation hours. Note: step is a simulation counter, not a real-world timestamp.';
