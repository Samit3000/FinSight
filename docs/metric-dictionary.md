# FinSight — Business & Analytical Metric Dictionary

**Document Version:** 1.0.0  
**Domain:** Digital Payments, Fraud Risk, and Payment Operations  
**Target Systems:** PostgreSQL 15+ & Microsoft Power BI  

---

## 1. Metric Catalog Overview

The FinSight metric catalog defines standardized business KPIs, operational payment performance indicators, risk monitoring ratios, and data pipeline health metrics. Each metric is specified with its business rationale, mathematical formulation, SQL query template, and reporting grain.

---

## 2. Payment Performance & Business KPIs

### 2.1. Gross Payment Volume (GPV)
- **Business Description**: The total aggregate monetary value of successfully settled transactions processed across the platform.
- **Reporting Grain**: Daily, Weekly, Monthly, By Payment Method, By Currency.
- **Formula**:
  $$\text{GPV} = \sum_{\text{status} = \text{'SUCCESS'}} \text{amount}$$
- **SQL Implementation**:
  ```sql
  SELECT 
      DATE_TRUNC('day', timestamp) AS transaction_date,
      currency,
      SUM(amount) AS gross_payment_volume
  FROM fct_transactions
  WHERE status = 'SUCCESS'
  GROUP BY 1, 2;
  ```
- **Interpretation**: Core top-line growth indicator for digital payment networks.

---

### 2.2. Total Transaction Volume (TTV / Transaction Count)
- **Business Description**: The total count of all attempted transactions, regardless of final outcome status.
- **Reporting Grain**: Overall, Daily, By Payment Rail.
- **Formula**:
  $$\text{Total Transactions} = \text{COUNT}(\text{transaction\_id})$$
- **SQL Implementation**:
  ```sql
  SELECT 
      DATE_TRUNC('day', timestamp) AS transaction_date,
      payment_method,
      COUNT(transaction_id) AS total_transactions
  FROM fct_transactions
  GROUP BY 1, 2;
  ```

---

### 2.3. Transaction Success Rate (Authorization Rate)
- **Business Description**: The proportion of attempted transactions that successfully settle without processor declines, cancellations, or system errors.
- **Reporting Grain**: Daily, By Payment Rail, By Merchant Category.
- **Formula**:
  $$\text{Success Rate (\%)} = \left( \frac{\text{Count of Successful Transactions}}{\text{Total Attempted Transactions}} \right) \times 100$$
- **SQL Implementation**:
  ```sql
  SELECT 
      payment_method,
      COUNT(transaction_id) AS total_attempts,
      COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END) AS successful_transactions,
      ROUND(
          COUNT(CASE WHEN status = 'SUCCESS' THEN 1 END)::NUMERIC / 
          NULLIF(COUNT(transaction_id), 0) * 100, 2
      ) AS success_rate_pct
  FROM fct_transactions
  GROUP BY payment_method;
  ```
- **Benchmark**: High-performing cards/UPI rails typically exhibit 88%–95%+ authorization rates. Drops below 85% signal gateway downtime or aggressive merchant fraud rules.

---

### 2.4. Average Order Value (AOV)
- **Business Description**: The average monetary value of each successful transaction.
- **Reporting Grain**: Overall, By Merchant, By Payment Method.
- **Formula**:
  $$\text{AOV} = \frac{\text{GPV}}{\text{Count of Successful Transactions}}$$
- **SQL Implementation**:
  ```sql
  SELECT 
      merchant_id,
      ROUND(AVG(amount), 2) AS average_order_value,
      ROUND(STDDEV(amount), 2) AS aov_std_dev
  FROM fct_transactions
  WHERE status = 'SUCCESS'
  GROUP BY merchant_id;
  ```

---

### 2.5. Payment Rail Distribution (%)
- **Business Description**: The percentage share of total volume or count captured by each payment method (e.g., Credit Card, Debit Card, UPI, Digital Wallet).
- **Formula**:
  $$\text{Share}_{\text{rail}} = \left( \frac{\text{Volume}_{\text{rail}}}{\text{Total Volume}} \right) \times 100$$
- **SQL Implementation**:
  ```sql
  SELECT 
      payment_method,
      SUM(amount) AS rail_volume,
      ROUND(
          SUM(amount) / SUM(SUM(amount)) OVER () * 100, 2
      ) AS volume_share_pct
  FROM fct_transactions
  WHERE status = 'SUCCESS'
  GROUP BY payment_method;
  ```

---

## 3. Risk & Fraud Analytics Metrics

### 3.1. Flagged Transaction Rate
- **Business Description**: The percentage of evaluated transactions assigned to the `HIGH` risk tier (`risk_score >= 70`) or flagged for operational review.
- **Reporting Grain**: Daily, By Payment Method, By Merchant Category.
- **Formula**:
  $$\text{Flagged Rate (\%)} = \left( \frac{\text{Count of Flagged Transactions}}{\text{Total Evaluated Transactions}} \right) \times 100$$
- **SQL Implementation**:
  ```sql
  SELECT 
      DATE_TRUNC('day', t.timestamp) AS evaluation_date,
      COUNT(t.transaction_id) AS total_transactions,
      COUNT(CASE WHEN r.is_flagged = TRUE THEN 1 END) AS flagged_transactions,
      ROUND(
          COUNT(CASE WHEN r.is_flagged = TRUE THEN 1 END)::NUMERIC / 
          NULLIF(COUNT(t.transaction_id), 0) * 100, 2
      ) AS flagged_transaction_rate_pct
  FROM fct_transactions t
  INNER JOIN fct_risk_evaluations r ON t.transaction_id = r.transaction_id
  GROUP BY 1;
  ```
- **Benchmark**: In typical digital payments, healthy flagged rates range between 1.5% and 5.0%. Excessively high rates cause analyst fatigue and false positive customer friction.

---

### 3.2. Flagged Volume Exposure
- **Business Description**: The total monetary volume associated with high-risk flagged transactions.
- **Formula**:
  $$\text{Flagged Volume} = \sum_{\text{is\_flagged} = \text{TRUE}} \text{amount}$$
- **SQL Implementation**:
  ```sql
  SELECT 
      r.risk_tier,
      COUNT(t.transaction_id) AS transaction_count,
      SUM(t.amount) AS total_exposure_volume,
      ROUND(
          SUM(t.amount) / SUM(SUM(t.amount)) OVER () * 100, 2
      ) AS exposure_share_pct
  FROM fct_transactions t
  INNER JOIN fct_risk_evaluations r ON t.transaction_id = r.transaction_id
  GROUP BY r.risk_tier;
  ```

---

### 3.3. Rule Trigger Frequency & Share
- **Business Description**: Counts how frequently individual risk rules fire across all evaluated batches.
- **SQL Implementation**:
  ```sql
  SELECT 
      rule_id,
      COUNT(*) AS trigger_count
  FROM (
      SELECT UNNEST(triggered_rules) AS rule_id
      FROM fct_risk_evaluations
  ) unnested_rules
  GROUP BY rule_id
  ORDER BY trigger_count DESC;
  ```

---

### 3.4. High-Risk Merchant Concentration
- **Business Description**: Identifies merchants generating an outsized share of high-risk transactions.
- **SQL Implementation**:
  ```sql
  SELECT 
      t.merchant_id,
      COUNT(t.transaction_id) AS merchant_tx_count,
      COUNT(CASE WHEN r.is_flagged = TRUE THEN 1 END) AS high_risk_tx_count,
      ROUND(
          COUNT(CASE WHEN r.is_flagged = TRUE THEN 1 END)::NUMERIC / 
          NULLIF(COUNT(t.transaction_id), 0) * 100, 2
      ) AS merchant_risk_ratio_pct
  FROM fct_transactions t
  INNER JOIN fct_risk_evaluations r ON t.transaction_id = r.transaction_id
  GROUP BY t.merchant_id
  HAVING COUNT(t.transaction_id) >= 20
  ORDER BY merchant_risk_ratio_pct DESC
  LIMIT 10;
  ```

---

## 4. Data Quality & Pipeline Health Metrics

### 4.1. Data Ingestion Completeness Rate
- **Business Description**: Percentage of records from the raw file successfully validated and ingested into facts.
- **Formula**:
  $$\text{Completeness Rate (\%)} = \left( \frac{\text{clean\_rows\_loaded}}{\text{total\_rows\_read}} \right) \times 100$$
- **SQL Implementation**:
  ```sql
  SELECT 
      batch_id,
      total_rows_read,
      clean_rows_loaded,
      quarantine_rows,
      ROUND((clean_rows_loaded::NUMERIC / NULLIF(total_rows_read, 0)) * 100, 2) AS completeness_rate_pct
  FROM dq_audit_log
  ORDER BY started_at DESC;
  ```

---

### 4.2. Quarantine Defect Rate
- **Business Description**: Percentage of incoming rows that violated data quality contracts.
- **Formula**:
  $$\text{Defect Rate (\%)} = \left( \frac{\text{quarantine\_rows}}{\text{total\_rows\_read}} \right) \times 100$$
- **Target**: `< 0.5%` in production batch pipelines.
