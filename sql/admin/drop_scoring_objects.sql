-- ==============================================================================
-- FinSight P3 — MANUAL ROLLBACK of the scoring store
-- Script: drop_scoring_objects.sql
--
-- DESTRUCTIVE: removes ONLY the objects created by sql/schema/02_scoring_schema.sql and
-- sql/views/05_vw_alert_queue.sql. It never touches fct_transactions, dq_audit_log or the
-- other analytical views. Run by hand only, after deliberate approval.
-- ==============================================================================

BEGIN;
DROP VIEW IF EXISTS vw_alert_queue;
DROP TABLE IF EXISTS fct_alert_reasons;
DROP TABLE IF EXISTS fct_model_scores;
DROP TABLE IF EXISTS scoring_run_log;
COMMIT;
