-- ==============================================================================
-- FinSight Platform — PostgreSQL 15+ Core Schema Definition
-- Milestone 2: Transaction-Level Warehouse Model & Analytics Schema
-- ==============================================================================

-- Drop tables in reverse dependency order if recreating
DROP TABLE IF EXISTS fct_risk_evaluations CASCADE;
DROP TABLE IF EXISTS fct_transactions CASCADE;
DROP TABLE IF EXISTS dim_customers CASCADE;
DROP TABLE IF EXISTS dim_merchants CASCADE;
DROP TABLE IF EXISTS dq_quarantine_records CASCADE;
DROP TABLE IF EXISTS dq_audit_log CASCADE;

-- ------------------------------------------------------------------------------
-- 1. TRANSACTION FACT TABLE: fct_transactions
-- Grain: One record represents exactly one financial transaction event.
-- Primary Key Strategy: BIGSERIAL surrogate key (transaction_id) to guarantee
-- unique transaction addressability without assuming account IDs are unique.
-- ------------------------------------------------------------------------------
CREATE TABLE fct_transactions (
    transaction_id BIGSERIAL PRIMARY KEY,
    step INT NOT NULL CHECK (step >= 1),
    type VARCHAR(32) NOT NULL CHECK (type IN ('CASH_OUT', 'TRANSFER', 'PAYMENT', 'CASH_IN', 'DEBIT')),
    amount NUMERIC(16, 2) NOT NULL CHECK (amount >= 0.00),
    name_orig VARCHAR(64) NOT NULL,
    oldbalance_org NUMERIC(16, 2) NOT NULL,
    newbalance_orig NUMERIC(16, 2) NOT NULL,
    name_dest VARCHAR(64) NOT NULL,
    oldbalance_dest NUMERIC(16, 2) NOT NULL,
    newbalance_dest NUMERIC(16, 2) NOT NULL,
    is_fraud INT NOT NULL CHECK (is_fraud IN (0, 1)),
    is_flagged_fraud INT NOT NULL CHECK (is_flagged_fraud IN (0, 1)),
    transaction_hour INT NOT NULL CHECK (transaction_hour BETWEEN 0 AND 23),
    transaction_day INT NOT NULL CHECK (transaction_day >= 1),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- ------------------------------------------------------------------------------
-- Performance Indexes Tailored for Analytics Queries:
-- 1. idx_fct_type: Accelerates volume and value aggregation by transaction rail.
-- 2. idx_fct_fraud: Fast lookup and filtering for simulated fraud cases.
-- 3. idx_fct_type_fraud: Composite index for type-level fraud rate calculations.
-- 4. idx_fct_hour: Hourly volume trends.
-- 5. idx_fct_hour_type_fraud: Hourly risk breakdown across rails.
-- 6. idx_fct_name_orig / idx_fct_name_dest: Distinct account aggregation.
-- ------------------------------------------------------------------------------
CREATE INDEX idx_fct_type ON fct_transactions (type);
CREATE INDEX idx_fct_fraud ON fct_transactions (is_fraud);
CREATE INDEX idx_fct_type_fraud ON fct_transactions (type, is_fraud);
CREATE INDEX idx_fct_hour ON fct_transactions (transaction_hour);
CREATE INDEX idx_fct_hour_type_fraud ON fct_transactions (transaction_hour, type, is_fraud);
CREATE INDEX idx_fct_name_orig ON fct_transactions (name_orig);
CREATE INDEX idx_fct_name_dest ON fct_transactions (name_dest);

-- ------------------------------------------------------------------------------
-- Table & Column Documentation:
-- Explicitly records the analytical utility of fields and documents the ML boundary.
-- ------------------------------------------------------------------------------
COMMENT ON TABLE fct_transactions IS 
'Curated transaction facts from PaySim financial logs. Each row represents one transaction event with validated schema and derived temporal attributes.';

COMMENT ON COLUMN fct_transactions.transaction_id IS 
'Surrogate auto-incrementing primary key guaranteeing unique row addressability.';

COMMENT ON COLUMN fct_transactions.step IS 
'1-indexed simulation hour (1 to 744, representing a 31-day simulation period).';

COMMENT ON COLUMN fct_transactions.transaction_hour IS 
'Derived hour of day (0-23) calculated as ((step - 1) % 24). Documented simulation assumption; no calendar timestamps fabricated.';

COMMENT ON COLUMN fct_transactions.transaction_day IS 
'Derived simulation day (1-31) calculated as (((step - 1) / 24) + 1).';

COMMENT ON COLUMN fct_transactions.name_orig IS 
'Origin account identifier. Retained for entity analytics and distinct customer metrics. Excluded from initial ML features due to 99.85% single-use sparsity.';

COMMENT ON COLUMN fct_transactions.name_dest IS 
'Destination account identifier (C-prefix for customers, M-prefix for merchants). Retained for network analytics. Excluded from initial ML features.';

COMMENT ON COLUMN fct_transactions.oldbalance_org IS 
'Origin initial balance. Retained for balance discrepancy audits. Excluded from initial ML features due to synthetic simulator artifacts (33% zero balances).';

COMMENT ON COLUMN fct_transactions.newbalance_orig IS 
'Origin post-transaction balance. Excluded from initial ML features due to 98% complete liquidation leakage in fraudulent records.';

COMMENT ON COLUMN fct_transactions.oldbalance_dest IS 
'Destination initial balance. Excluded from initial ML features due to 100% zero-balance synthetic artifact for merchant destinations.';

COMMENT ON COLUMN fct_transactions.newbalance_dest IS 
'Destination post-transaction balance. Excluded from initial ML features due to synthetic merchant zero-balance artifact.';

COMMENT ON COLUMN fct_transactions.is_fraud IS 
'Simulated fraud ground-truth label (1=fraud, 0=legitimate). Preserved strictly as the model evaluation target, never as a predictor.';

COMMENT ON COLUMN fct_transactions.is_flagged_fraud IS 
'Pre-existing rule-based flag (>200,000 threshold in TRANSFER). Kept strictly distinct from is_fraud. Excluded from ML features to prevent rule leakage.';

-- ------------------------------------------------------------------------------
-- 2. PIPELINE EXECUTION AUDIT LOG: dq_audit_log
-- Audit log tracking batch ingestion runs, row accounting, and durations.
-- ------------------------------------------------------------------------------
CREATE TABLE dq_audit_log (
    batch_id VARCHAR(64) PRIMARY KEY,
    source_file VARCHAR(256) NOT NULL,
    total_rows_read INT NOT NULL,
    clean_rows_loaded INT NOT NULL,
    quarantine_rows INT NOT NULL DEFAULT 0,
    duplicate_rows INT NOT NULL DEFAULT 0,
    execution_status VARCHAR(20) NOT NULL CHECK (execution_status IN ('SUCCESS', 'PARTIAL_SUCCESS', 'FAILED')),
    started_at TIMESTAMPTZ NOT NULL,
    completed_at TIMESTAMPTZ NOT NULL,
    duration_seconds NUMERIC(8, 2) NOT NULL
);

COMMENT ON TABLE dq_audit_log IS 
'Audit ledger recording batch execution telemetry, row accounting, and runtime duration.';
