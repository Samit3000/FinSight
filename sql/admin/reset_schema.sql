-- ==============================================================================
-- FinSight — DESTRUCTIVE SCHEMA RESET (manual use only)
--
-- WARNING: This permanently deletes every loaded transaction (6.36M rows when
-- fully loaded), the data-quality audit log, and all analytical views that
-- depend on fct_transactions. There is no undo.
--
-- Nothing in the FinSight codebase executes this file. Run it only by hand,
-- only against a database you intend to rebuild, and then re-apply
-- sql/schema/01_schema_definition.sql, reload data, and re-create sql/views/*.
--
--   psql -h <host> -U <user> -d <database> -v ON_ERROR_STOP=1 -f sql/admin/reset_schema.sql
-- ==============================================================================

BEGIN;
DROP VIEW IF EXISTS vw_risk_detection_benchmark;
DROP VIEW IF EXISTS vw_payment_rail_performance;
DROP VIEW IF EXISTS vw_hourly_risk_dynamics;
DROP VIEW IF EXISTS vw_daily_transaction_kpis;
DROP TABLE IF EXISTS fct_transactions;
DROP TABLE IF EXISTS dq_audit_log;
COMMIT;
