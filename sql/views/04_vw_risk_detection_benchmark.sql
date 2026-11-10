-- ==============================================================================
-- FinSight Analytical Mart: Risk Detection Benchmark Performance
-- View Name: vw_risk_detection_benchmark
-- Grain: One row per evaluation partition, system/model identifier, and operating point.
--
-- Description:
-- Performance comparison of supervised machine learning models, heuristic rule
-- tiers, and naive baselines evaluated on identical chronological partitions.
--
-- Methodological Integrity:
-- - All metrics are computed from transaction-level evaluation records.
-- - Confusion matrices represent complete, non-aggregated evaluation populations:
--   * Holdout Partition (step >= 551): Exactly 196,601 transactions (2,148 frauds).
--   * Validation Partition (451 <= step <= 550): Exactly 151,302 transactions (1,008 frauds).
-- - TP + FP + FN + TN strictly equals the total partition population for every row.
-- - Average Precision and ROC-AUC are populated ONLY for supervised ML models
--   generating complete continuous risk probabilities; set to NULL for discrete rules.
--
-- Source: Reproducible model evaluation outputs & verified rule audit records.
-- Idempotency: CREATE OR REPLACE VIEW
-- ==============================================================================

CREATE OR REPLACE VIEW vw_risk_detection_benchmark AS
WITH raw_benchmark_data (
    evaluation_partition,
    system_name,
    operating_point,
    true_positives,
    false_positives,
    false_negatives,
    true_negatives,
    average_precision,
    roc_auc
) AS (
    VALUES
        -- 1. Frozen Champion: HistGradientBoosting (No Day) on Holdout
        ('holdout', 'HistGradientBoosting (Champion No Day)', 't = 0.9660 (Val F1 Optimal)', 918, 672, 1230, 193781, 0.4479, 0.9487),
        -- 2. Frozen Champion: HistGradientBoosting (No Day) on Validation
        ('validation', 'HistGradientBoosting (Champion No Day)', 't = 0.9660 (Val F1 Optimal)', 435, 863, 573, 149431, 0.3115, 0.9509),
        -- 3. Baseline Model: HistGradientBoosting (9-feat) on Holdout
        ('holdout', 'HistGradientBoosting (Baseline 9-feat)', 't = 0.9900 (Val Capacity Constrained)', 737, 331, 1411, 194122, 0.3836, 0.9185),
        -- 4. Baseline Model: HistGradientBoosting (9-feat) on Holdout (Unconstrained)
        ('holdout', 'HistGradientBoosting (Baseline 9-feat)', 't = 0.9830 (Val F1 Optimal)', 845, 551, 1303, 193902, 0.3836, 0.9185),
        -- 5. Linear Baseline: Logistic Regression (Balanced) on Holdout
        ('holdout', 'Logistic Regression (Balanced)', 't = 0.9009 (Val F1 Optimal)', 626, 1950, 1522, 192503, 0.1805, 0.9152),
        -- 6. Heuristic Rule Engine: HIGH Tier (Score >= 70) on Holdout
        ('holdout', 'Heuristic Rule Engine (HIGH Tier)', 'Score >= 70 (HIGH Tier)', 402, 156, 1746, 194297, NULL, NULL),
        -- 7. Heuristic Rule Engine: MED+HIGH Tier (Score >= 30) on Holdout
        ('holdout', 'Heuristic Rule Engine (MED+HIGH Tier)', 'Score >= 30 (MED+HIGH Tier)', 1717, 47213, 431, 147240, NULL, NULL),
        -- 8. Naive Baseline: Dummy Majority Classifier on Holdout
        ('holdout', 'Dummy Majority Baseline', 't = 0.5000 (Always Negative)', 0, 0, 2148, 194453, 0.0109, 0.5000)
)
SELECT
    evaluation_partition,
    system_name,
    operating_point,
    true_positives,
    false_positives,
    false_negatives,
    true_negatives,
    (true_positives + false_positives + false_negatives + true_negatives) AS total_evaluation_population,
    (true_positives + false_positives) AS alert_count,
    ROUND(
        COALESCE(
            (CAST(true_positives + false_positives AS NUMERIC) / 
             NULLIF(true_positives + false_positives + false_negatives + true_negatives, 0)) * 100,
            0.0
        ),
        4
    ) AS alert_rate_pct,
    ROUND(
        COALESCE(
            CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_positives, 0),
            0.0
        ),
        4
    ) AS precision,
    ROUND(
        COALESCE(
            CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_negatives, 0),
            0.0
        ),
        4
    ) AS recall,
    ROUND(
        COALESCE(
            2.0 * (CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_positives, 0)) * 
                  (CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_negatives, 0)) / 
            NULLIF(
                (CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_positives, 0)) + 
                (CAST(true_positives AS NUMERIC) / NULLIF(true_positives + false_negatives, 0)),
                0
            ),
            0.0
        ),
        4
    ) AS f1_score,
    ROUND(average_precision, 4) AS average_precision,
    ROUND(roc_auc, 4) AS roc_auc
FROM raw_benchmark_data
ORDER BY evaluation_partition ASC, f1_score DESC;

COMMENT ON VIEW vw_risk_detection_benchmark IS
'Comparative evaluation benchmarks across machine learning models, heuristic rule tiers, and baselines on frozen validation and holdout partitions.';
