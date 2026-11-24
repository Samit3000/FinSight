-- ==============================================================================
-- FinSight Milestone 6 — Daily alert workload
-- View Name: vw_alert_workload_daily
--
-- Grain: one row per model x simulated day x evaluation period (a day that spans two periods has two rows).
-- Operational workload only: counts of scored transactions and alerts. No fraud labels, no performance claims.
-- Idempotency: CREATE OR REPLACE VIEW.
-- ==============================================================================

CREATE OR REPLACE VIEW vw_alert_workload_daily AS
SELECT
    s.model_artifact_sha256,
    ((s.step - 1) / 24) + 1                                   AS simulated_day,
    s.evaluation_period,
    COUNT(DISTINCT s.step)                                     AS hours_covered,
    (COUNT(DISTINCT s.step) = 24)                              AS is_full_day,
    COUNT(*)                                                   AS transactions_scored,
    SUM(s.is_alert::INT)                                       AS alerts,
    ROUND(100.0 * SUM(s.is_alert::INT) / NULLIF(COUNT(*), 0), 4) AS alert_rate_pct,
    SUM((s.is_alert AND t.type = 'TRANSFER')::INT)             AS transfer_alerts,
    SUM((s.is_alert AND t.type = 'CASH_OUT')::INT)             AS cash_out_alerts,
    SUM((s.is_alert AND s.recipient_prior_in_count = 0)::INT)  AS first_time_recipient_alerts,
    MIN(s.operating_threshold)                                 AS operating_threshold
FROM fct_model_scores AS s
JOIN fct_transactions AS t ON t.transaction_id = s.transaction_id
GROUP BY s.model_artifact_sha256, ((s.step - 1) / 24) + 1, s.evaluation_period;

COMMENT ON VIEW vw_alert_workload_daily IS
'Alerts and scored transactions per simulated day and evaluation period. No fraud labels; workload only.';
