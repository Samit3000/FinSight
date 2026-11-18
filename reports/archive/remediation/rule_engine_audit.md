# FinSight — Rule Engine & Benchmark Integrity Audit (Phase 6)

**Audit Execution Date:** 2026-10-19  
**Auditor:** Senior Fraud Analytics Specialist & ML Quality Auditor  
**Scope:** Deterministic rule definitions, overnight volume claim verification, standalone rule precision/recall, operational tier metrics, and ML head-to-head benchmarking  

---

## 1. Executive Summary

A comprehensive empirical audit of FinSight's explainable rule engine baseline (`src/finsight/risk_rules.py`) was conducted against the complete $6,362,620$-transaction dataset and the frozen $196,601$-transaction holdout partition.

### Key Audit Conclusions:
1. **The ">99% Overnight Drop" Claim is Empirically True:**
   Forensic analysis of the simulation's hourly transaction volume confirms that diurnal consumer activity drops from a peak of **647,814 transactions** at Hour 18 (6 PM) down to **1,241 transactions** at Hour 3 (3 AM)—an empirical drop of **99.81%** (and a **99.47%** drop compared to the 1 AM–5 AM overnight average).
2. **Extreme Overnight Fraud Concentration:**
   While legitimate consumer transactions collapse overnight, simulated fraud scripts run continuously ($\sim 300\text{--}375$ frauds every hour across all 24 hours). Consequently, the empirical fraud incidence rate jumps from **0.0528%** at 6 PM to **22.30%** at 4 AM (a **422× increase in fraud density**).
3. **High Standalone Precision on Heuristic Rules:**
   - `RULE_004_LARGE_ROUND_AMOUNT` achieves **80.92% precision** on holdout (106 true frauds out of 131 triggers).
   - `RULE_003_OVERNIGHT_DRAINAGE` achieves **59.57% precision** on holdout (448 true frauds out of 752 triggers).
4. **Benchmark Tier Performance on Holdout:**
   - **`HIGH` Risk Tier (Score $\ge 70$):** Precision **0.7204**, Recall **0.1872**, F1 **0.2971** ($558$ alerts, $402$ true frauds caught).
   - **`MEDIUM + HIGH` Risk Tiers (Score $\ge 30$):** Precision **0.0312**, Recall **0.7318**, F1 **0.0599** ($50,357$ alerts, $1,572$ true frauds caught).
5. **Head-to-Head Benchmark vs. Supervised ML:**
   On the identical holdout slice, the machine learning models capture **nearly 2× the recall** of the Rule Engine ($34.3\%\text{--}45.3\%$ vs $18.7\%$) at competitive precision ($69.0\%\text{--}53.7\%$). The rule engine remains preserved as an essential baseline benchmark.

---

## 2. Hourly Volume & Overnight Rationale Verification

The hourly distribution across all $6,362,620$ transactions in PaySim was tabulated to verify the documented business rationale:

| Hour of Day | Total Volume | Fraud Count | Fraud Incidence Rate (%) | Operational Activity Assessment |
| :---: | :---: | :---: | :---: | :--- |
| **00:00 (12 AM)** | 27,111 | 358 | 1.3205% | Evening winding down. |
| **01:00 (1 AM)** | 9,018 | 372 | 4.1251% | Overnight dormancy begins. |
| **02:00 (2 AM)** | 2,007 | 326 | 16.2431% | Extreme dormancy; automated bot activity. |
| **03:00 (3 AM)** | **1,241** | 274 | **22.0790%** | **Minimum diurnal volume (99.81% below peak).** |
| **04:00 (4 AM)** | 1,641 | 366 | **22.3035%** | **Peak fraud concentration window.** |
| **05:00 (5 AM)** | 3,420 | 358 | 10.4678% | Early morning dormancy window. |
| **06:00 (6 AM)** | 8,988 | 328 | 3.6493% | Early commuter activity begins. |
| **07:00 (7 AM)** | 26,915 | 368 | 1.3673% | Morning ramp-up. |
| **08:00 (8 AM)** | 283,518 | 341 | 0.1203% | Business day opens; volume surges 10×. |
| **09:00 (9 AM)** | 425,729 | 375 | 0.0881% | High retail business volume. |
| **10:00 (10 AM)**| 445,992 | 324 | 0.0726% | Steady-state business operations. |
| **11:00 (11 AM)**| 483,418 | 339 | 0.0701% | Peak morning commercial volume. |
| **12:00 (12 PM)**| 468,474 | 346 | 0.0739% | Midday lunch retail payments. |
| **13:00 (1 PM)** | 439,653 | 353 | 0.0803% | Afternoon commercial operations. |
| **14:00 (2 PM)** | 416,686 | 341 | 0.0818% | Steady afternoon activity. |
| **15:00 (3 PM)** | 441,612 | 345 | 0.0781% | Steady afternoon activity. |
| **16:00 (4 PM)** | 439,941 | 353 | 0.0802% | End of standard banking hours. |
| **17:00 (5 PM)** | 580,509 | 343 | 0.0591% | Evening commute / shopping surge. |
| **18:00 (6 PM)** | **647,814** | 342 | **0.0528%** | **Global peak volume across entire dataset.** |
| **19:00 (7 PM)** | 553,728 | 340 | 0.0614% | Evening retail and dining. |
| **20:00 (8 PM)** | 247,806 | 347 | 0.1400% | Retail volume slows. |
| **21:00 (9 PM)** | 194,555 | 351 | 0.1804% | Evening taper. |
| **22:00 (10 PM)**| 141,257 | 323 | 0.2287% | Late evening taper. |
| **23:00 (11 PM)**| 71,587 | 300 | 0.4191% | Pre-midnight taper. |

**Audit Verification Result:**
The documentation statement that human payment volume drops by $>99\%$ overnight is **100% verified and true**. Volume drops by **99.81%** from 6 PM to 3 AM. Because fraud generation is completely flat, the off-hours window (1 AM–5 AM) represents a legitimate, empirical high-risk signal in PaySim.

---

## 3. Standalone Rule Trigger Performance

Evaluated individually on the final holdout partition ($N = 196,601$, $y=1 \implies 2,148$):

| Rule ID & Name | Trigger Logic | Points | Holdout Triggers | Precision | Recall | F1 Score | Behavioral & Operational Assessment |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`RULE_001_HIGH_VALUE`** | $\text{amount} \ge \$200,000$ | 35 | 50,060 | 0.0288 | **0.6704** | 0.0552 | High recall signal ($67.0\%$ of all holdout fraud caught), but low standalone precision due to legitimate corporate transfers. |
| **`RULE_002_DRAINAGE_RAIL`** | $\text{rail} \in (\text{TRANSFER}, \text{CASH\_OUT})$ | 25 | 82,221 | 0.0261 | **1.0000** | 0.0509 | Catches $100.0\%$ of all simulated fraud in PaySim. Necessary prerequisite for risk accumulation. |
| **`RULE_003_OVERNIGHT_DRAINAGE`** | $1\le \text{hour} \le 5 \land \text{rail} \in (\text{TRF}, \text{CO}) \land \text{amt} \ge \$10\text{k}$ | 25 | 752 | **0.5957** | 0.2086 | 0.3090 | **High standalone precision (59.6%).** Exploits overnight dormancy on drainage channels. |
| **`RULE_004_LARGE_ROUND_AMOUNT`** | $\text{amt} \ge \$50\text{k} \land \text{amt} \pmod{5000} == 0 \land \text{TRF}/\text{CO}$ | 15 | 131 | **0.8092** | 0.0493 | 0.0930 | **Extremely high precision (80.9%).** Exploits round denomination limit ceiling extractions. |

---

## 4. Operational Tier Performance on Holdout Partition

The rule engine computes a composite score ($0\text{--}100$) by summing points from triggered rules:

$$\text{Composite Score} = \min\left(100, \sum \text{Rule Points}\right)$$

Tiers: `LOW` ($0\text{--}29$), `MEDIUM` ($30\text{--}69$), `HIGH` ($70\text{--}100$).

### A. High Risk Tier (Score $\ge 70$, Action: `FLAG_SUSPICIOUS`)
Requires triggering at least 3 rules (e.g. Rule 1 [35 pts] + Rule 2 [25 pts] + Rule 3 [25 pts] $= 85$ pts):
- **Alert Count:** **558**
- **True Positives (TP):** **402**
- **False Positives (FP):** **156**
- **False Negatives (FN):** **1,746**
- **True Negatives (TN):** **194,297**
- **Precision:** **0.7204** (72.04% of high-tier alerts are verified fraud)
- **Recall:** **0.1872** (Catches 18.72% of all holdout fraud)
- **F1 Score:** **0.2971**

### B. Medium + High Risk Tiers (Score $\ge 30$, Action: `MANUAL_REVIEW` or `FLAG_SUSPICIOUS`)
Requires triggering at least Rule 1 (35 pts) or Rule 2 (25 pts) + Rule 4 (15 pts) $= 40$ pts:
- **Alert Count:** **50,357**
- **True Positives (TP):** **1,572**
- **False Positives (FP):** **48,785**
- **False Negatives (FN):** **576**
- **True Negatives (TN):** **145,668**
- **Precision:** **0.0312** (3.12%)
- **Recall:** **0.7318** (Catches 73.18% of all holdout fraud)
- **F1 Score:** **0.0599**

---

## 5. Head-to-Head Comparison: Rule Engine vs. Supervised ML

Evaluated on the exact same unseen holdout slice ($N = 196,601$, $y=1 \implies 2,148$ frauds):

| Detection System | Decision Rule / Threshold | Alerts Generated | Precision | Recall | F1 Score | False Positive Count | True Positive Count |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Rule Engine (HIGH Tier)** | Score $\ge 70$ | **558** | **0.7204** | 0.1872 | 0.2971 | **156** | 402 |
| **Rule Engine (MED+HIGH Tier)**| Score $\ge 30$ | 50,357 | 0.0312 | **0.7318** | 0.0599 | 48,785 | 1,572 |
| **HGB Baseline (9 features)** | ML Score $\ge 0.9900$ | 1,068 | 0.6901 | 0.3431 | 0.4583 | 331 | 737 |
| **HGB Remediation (8 feat, No Day)**| ML Score $\ge 0.9900$ | 1,184 | 0.6723 | 0.3706 | 0.4776 | 388 | 796 |
| **HGB Remediation (8 feat, No Day)**| ML Score $\ge 0.9660$ (Val Opt) | 1,810 | 0.5369 | **0.4525** | **0.4912** | 838 | 972 |

### Key Architectural Takeaways:
1. **Complementary Strengths:**
   - The Rule Engine `HIGH` tier operates at the highest precision ($72.04\%$) with the lowest alert burden ($558$ alerts), making it ideal for immediate automated freeze interventions.
   - The ML model nearly **doubles detection recall** ($34.3\%\text{--}45.3\%$ vs $18.7\%$), uncovering non-linear relationships between transaction amount, hour, and payment rail that static rules miss.
2. **Clear Separation of Signals:**
   Rule scores ($0\text{--}100$) and ML scores ($0.0\text{--}1.0$) represent distinct orthogonal dimensions: business compliance rules vs. statistical pattern recognition. Both must be preserved in the analytics schema and reporting dashboards.
