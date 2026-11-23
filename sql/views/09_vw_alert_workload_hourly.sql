-- ==============================================================================
-- FinSight Milestone 7 — Hourly alert workload
-- View Name: vw_alert_workload_hourly
--
-- Grain: one row per model x simulated hour (step). Scored transactions and alerts per hour, with the
-- hour's evaluation period. Operational only: no fraud labels, no performance claims.
-- Idempotency: CREATE OR REPLACE VIEW.
-- ==============================================================================

CREATE OR REPLACE VIEW vw_alert_workload_hourly AS
SELECT
    s.model_artifact_sha256,
    s.step,
    ((s.step - 1) / 24) + 1         AS simulated_day,
    (s.step - 1) % 24               AS hour_of_day,
    s.evaluation_period,
    COUNT(*)                        AS transactions_scored,
    SUM(s.is_alert::INT)            AS alerts
FROM fct_model_scores AS s
GROUP BY s.model_artifact_sha256, s.step, s.evaluation_period;

COMMENT ON VIEW vw_alert_workload_hourly IS
'Scored transactions and alerts per simulated hour and evaluation period. No fraud labels.';
