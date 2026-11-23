-- ==============================================================================
-- FinSight Milestone 7 — Champion score distribution (for capacity planning)
-- View Name: vw_score_distribution
--
-- Grain: one row per model x evaluation period x score bin. Bins are 0.0005 wide from 0.5 to 1.0; every
-- score below 0.5 falls in one bin starting at 0. With these counts, the number of alerts any threshold
-- would produce can be read off directly. Workload only: no fraud labels, so no detection claims.
-- Idempotency: CREATE OR REPLACE VIEW.
-- ==============================================================================

CREATE OR REPLACE VIEW vw_score_distribution AS
SELECT
    s.model_artifact_sha256,
    s.evaluation_period,
    CASE WHEN s.risk_score < 0.5 THEN 0.0
         ELSE FLOOR(s.risk_score * 2000) / 2000.0 END       AS score_bin_start,
    COUNT(*)                                                 AS transactions,
    COUNT(DISTINCT s.step)                                   AS hours_with_scores
FROM fct_model_scores AS s
GROUP BY 1, 2, 3;

COMMENT ON VIEW vw_score_distribution IS
'Count of scored transactions per 0.0005-wide score bin (scores below 0.5 in one bin) and evaluation period. No fraud labels; used to estimate alert volume at any threshold.';
