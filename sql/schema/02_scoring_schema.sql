-- ==============================================================================
-- FinSight P3 — Model scoring store
-- Script: 02_scoring_schema.sql
--
-- NON-DESTRUCTIVE and IDEMPOTENT: only creates objects that do not exist yet. It never
-- drops, truncates or alters existing tables (fct_transactions, dq_audit_log) or views.
-- Rollback (manual only): sql/admin/drop_scoring_objects.sql
--
-- Contents:
--   scoring_run_log    one row per successful scoring run (failed runs roll back completely)
--   fct_model_scores   one row per transaction per model (risk score, alert flag, period label)
--   fct_alert_reasons  one row per alert per model (model attribution and rule conditions, kept separate)
--
-- Fraud labels are never stored here; the scorer does not read them.
-- ==============================================================================

CREATE TABLE IF NOT EXISTS scoring_run_log (
    run_id                  VARCHAR(64) PRIMARY KEY,
    model_artifact          VARCHAR(256) NOT NULL,
    model_artifact_sha256   CHAR(64) NOT NULL,
    config_name             VARCHAR(64) NOT NULL,
    mode                    VARCHAR(16) NOT NULL CHECK (mode IN ('bulk', 'hourly')),
    min_step                INT NOT NULL CHECK (min_step >= 1),
    max_step                INT NOT NULL,
    operating_threshold     DOUBLE PRECISION NOT NULL,
    rows_scored             BIGINT NOT NULL CHECK (rows_scored >= 0),
    alerts                  BIGINT NOT NULL CHECK (alerts >= 0),
    data_sha256             CHAR(64) NOT NULL,
    code_commit             VARCHAR(48) NOT NULL,
    started_at              TIMESTAMPTZ NOT NULL,
    completed_at            TIMESTAMPTZ NOT NULL,
    status                  VARCHAR(16) NOT NULL CHECK (status = 'SUCCESS'),
    CHECK (max_step >= min_step)
);

COMMENT ON TABLE scoring_run_log IS
'One row per successful scoring run. A run is a single transaction: if it fails, neither scores nor this row are kept.';

CREATE TABLE IF NOT EXISTS fct_model_scores (
    transaction_id              BIGINT NOT NULL REFERENCES fct_transactions (transaction_id),
    model_artifact_sha256       CHAR(64) NOT NULL,
    run_id                      VARCHAR(64) NOT NULL
                                REFERENCES scoring_run_log (run_id) DEFERRABLE INITIALLY DEFERRED,
    step                        INT NOT NULL CHECK (step >= 1),
    evaluation_period           VARCHAR(24) NOT NULL,
    recipient_prior_in_count    INT NOT NULL CHECK (recipient_prior_in_count >= 0),
    risk_score                  DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    operating_threshold         DOUBLE PRECISION NOT NULL,
    is_alert                    BOOLEAN NOT NULL,
    PRIMARY KEY (transaction_id, model_artifact_sha256),
    CHECK (is_alert = (risk_score >= operating_threshold)),
    CHECK (
        (step <= 550 AND evaluation_period = 'training')
        OR (step BETWEEN 551 AND 650 AND evaluation_period = 'calibration')
        OR (step >= 651 AND evaluation_period = 'sealed_test_reported')
    )
);

CREATE INDEX IF NOT EXISTS idx_scores_model_step ON fct_model_scores (model_artifact_sha256, step);
CREATE INDEX IF NOT EXISTS idx_scores_alerts ON fct_model_scores (model_artifact_sha256, step) WHERE is_alert;
CREATE INDEX IF NOT EXISTS idx_scores_run ON fct_model_scores (run_id);

COMMENT ON TABLE fct_model_scores IS
'Champion risk scores (class-weighted, not calibrated probabilities). evaluation_period: training = steps 1-550 (the model learned from them, scores optimistic); calibration = 551-650 (threshold set here); sealed_test_reported = 651-743 (performance already reported once in P1.5; never used for selection).';

CREATE TABLE IF NOT EXISTS fct_alert_reasons (
    transaction_id          BIGINT NOT NULL,
    model_artifact_sha256   CHAR(64) NOT NULL,
    run_id                  VARCHAR(64) NOT NULL
                            REFERENCES scoring_run_log (run_id) DEFERRABLE INITIALLY DEFERRED,
    -- Model attribution (TreeSHAP); never causal evidence
    top_contributors        JSONB NOT NULL,
    model_summary           TEXT NOT NULL,
    model_disclaimer        TEXT NOT NULL,
    base_value_log_odds     DOUBLE PRECISION NOT NULL,
    margin_log_odds         DOUBLE PRECISION NOT NULL,
    additivity_error        DOUBLE PRECISION NOT NULL CHECK (additivity_error < 1e-6),
    -- Rule conditions met (deterministic checks; they do not raise alerts)
    rule_ids                TEXT[] NOT NULL,
    rule_conditions         JSONB NOT NULL,
    rule_score              INT NOT NULL,
    rule_tier               VARCHAR(8) NOT NULL CHECK (rule_tier IN ('LOW', 'MEDIUM', 'HIGH')),
    rule_note               TEXT NOT NULL,
    PRIMARY KEY (transaction_id, model_artifact_sha256),
    FOREIGN KEY (transaction_id, model_artifact_sha256)
        REFERENCES fct_model_scores (transaction_id, model_artifact_sha256),
    CHECK (jsonb_typeof(top_contributors) = 'array' AND jsonb_array_length(top_contributors) = 3)
);

CREATE INDEX IF NOT EXISTS idx_reasons_run ON fct_alert_reasons (run_id);

COMMENT ON TABLE fct_alert_reasons IS
'Reason codes for each alert: model attribution (top three TreeSHAP contributors, with a non-causal disclaimer) and, separately, the rule conditions met.';
