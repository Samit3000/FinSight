-- ==============================================================================
-- FinSight P3 — Alert queue view
-- View Name: vw_alert_queue
--
-- One row per alert per model, with the transaction, its risk score and its reason codes.
-- Operational view: it deliberately contains NO fraud labels, and it makes no performance claim.
-- evaluation_period tells the reader which hours the model trained on (training), which set its
-- threshold (calibration) and which were the one-time sealed test (sealed_test_reported).
-- Idempotency: CREATE OR REPLACE VIEW (new view; first created in P3).
-- ==============================================================================

CREATE OR REPLACE VIEW vw_alert_queue AS
SELECT
    s.transaction_id,
    s.model_artifact_sha256,
    s.run_id,
    t.step,
    t.transaction_day AS simulated_day,
    t.transaction_hour,
    t.type,
    t.amount,
    t.name_orig,
    t.name_dest,
    s.evaluation_period,
    s.recipient_prior_in_count,
    s.risk_score,
    s.operating_threshold,
    r.top_contributors,
    r.model_summary,
    r.model_disclaimer,
    r.rule_ids,
    r.rule_conditions,
    r.rule_score,
    r.rule_tier,
    r.rule_note
FROM fct_model_scores AS s
JOIN fct_transactions AS t ON t.transaction_id = s.transaction_id
JOIN fct_alert_reasons AS r
  ON r.transaction_id = s.transaction_id AND r.model_artifact_sha256 = s.model_artifact_sha256
WHERE s.is_alert;

COMMENT ON VIEW vw_alert_queue IS
'Alerts with reason codes. No fraud labels; model attribution is not evidence of fraud. evaluation_period marks training, calibration and sealed_test_reported hours.';
