-- ==============================================================================
-- FinSight Milestone 6 — How often each reason appears on alerts
-- View Name: vw_alert_reason_summary
--
-- Grain: one row per model x evaluation period x reason_source x reason.
--   reason_source = 'top_model_contributor' : the feature with the largest SHAP contribution on the alert
--                   (model attribution, not evidence of fraud)
--   reason_source = 'rule_condition'        : a rule condition the alert meets ('none' when no rule is met)
-- The two sources are counted separately and must not be added together.
-- No fraud labels, no performance claims. Idempotency: CREATE OR REPLACE VIEW.
-- ==============================================================================

CREATE OR REPLACE VIEW vw_alert_reason_summary AS
SELECT
    r.model_artifact_sha256,
    s.evaluation_period,
    'top_model_contributor'::TEXT                       AS reason_source,
    r.top_contributors -> 0 ->> 'feature'               AS reason,
    COUNT(*)                                            AS alerts
FROM fct_alert_reasons AS r
JOIN fct_model_scores AS s USING (transaction_id, model_artifact_sha256)
GROUP BY 1, 2, 3, 4
UNION ALL
SELECT
    r.model_artifact_sha256,
    s.evaluation_period,
    'rule_condition'::TEXT                              AS reason_source,
    rule_id                                             AS reason,
    COUNT(*)                                            AS alerts
FROM fct_alert_reasons AS r
JOIN fct_model_scores AS s USING (transaction_id, model_artifact_sha256)
CROSS JOIN LATERAL UNNEST(r.rule_ids) AS rule_id
GROUP BY 1, 2, 3, 4;

COMMENT ON VIEW vw_alert_reason_summary IS
'Counts of alerts by top model contributor and, separately, by rule condition met. Sources must not be summed together. No fraud labels.';
