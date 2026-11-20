-- ==============================================================================
-- FinSight Milestone 6 — Alert mix by payment type, hour band and amount band
-- View Name: vw_alert_mix
--
-- Grain: one row per model x evaluation period x payment type x hour band x amount band that has alerts.
-- Operational only: alert counts. No fraud labels, no performance claims.
-- Idempotency: CREATE OR REPLACE VIEW.
-- ==============================================================================

CREATE OR REPLACE VIEW vw_alert_mix AS
SELECT
    s.model_artifact_sha256,
    s.evaluation_period,
    t.type                                              AS payment_type,
    CASE
        WHEN t.transaction_hour BETWEEN 0 AND 5  THEN '00-05'
        WHEN t.transaction_hour BETWEEN 6 AND 8  THEN '06-08'
        WHEN t.transaction_hour BETWEEN 9 AND 17 THEN '09-17'
        ELSE '18-23'
    END                                                 AS hour_band,
    CASE
        WHEN t.amount < 100000   THEN '<100k'
        WHEN t.amount < 500000   THEN '100k-500k'
        WHEN t.amount < 1000000  THEN '500k-1M'
        ELSE '>=1M'
    END                                                 AS amount_band,
    COUNT(*)                                            AS alerts,
    SUM((s.recipient_prior_in_count = 0)::INT)          AS first_time_recipient_alerts,
    ROUND(AVG(s.risk_score)::NUMERIC, 4)                AS avg_risk_score
FROM fct_model_scores AS s
JOIN fct_transactions AS t ON t.transaction_id = s.transaction_id
WHERE s.is_alert
GROUP BY 1, 2, 3, 4, 5;

COMMENT ON VIEW vw_alert_mix IS
'Alert counts by evaluation period, payment type, hour band and amount band. No fraud labels.';
