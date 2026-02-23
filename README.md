# Payment Failure & Revenue Leakage Analysis

---

## Overview

Payment failures are inevitable — but a significant portion are **retryable**, meaning the revenue is recoverable without any action from the customer. This project simulates a realistic **1M-transaction payments dataset** to quantify that leakage, identify where it concentrates, and build a predictive model to flag high-risk transactions before authorization.

This mirrors the kind of financial infrastructure analytics done at companies like Stripe, Adyen, and Braintree — where even a 0.1% improvement in payment success rates translates to millions in recovered merchant revenue.

---

## Business Questions Answered

| Question | Method |
|---|---|
| How much revenue is being lost to retryable failures? | SQL aggregation + leakage quantification |
| Which merchant categories and geographies have the worst failure rates? | Dimensional analysis via SQL + Tableau |
| Which failure codes are retryable vs. permanent? | Failure taxonomy + recovery rate analysis |
| Can we predict which transactions will fail *before* authorization? | Logistic regression with pre-auth risk features |
| How should product teams monitor payment health over time? | Tableau dashboards with KPI exports |

---

## Key Results

| Metric | Value |
|---|---|
| Total transactions analyzed | 1,000,000 |
| Total payment volume | ~$165M |
| Overall failure rate | 12.8% |
| Retryable failures | 7.7% of all transactions |
| **Recoverable revenue leakage** | **~$8.7M** |
| Logistic Regression AUC | **0.824** (5-fold CV: 0.826 ± 0.002) |
| Top leakage category | Travel (~$5.4M) |
| Highest-risk geography | Brazil (BR) |
| Highest-risk payment method | BNPL (1.4× avg failure rate) |

---

## Methodology

### Data Generation (`data_generator.py`)
Simulated 1M transactions with realistic attributes:
- **8 merchant categories** (E-commerce, SaaS, Travel, Retail, Marketplaces, Gaming, Healthcare, Food)
- **8 geographies** with currency-accurate failure multipliers (US, EU, UK, IN, BR, SG, CA, AU)
- **5 payment methods** with empirically-based failure rates
- **12 failure codes** classified as retryable or non-retryable, with realistic recovery rates
- **Log-normal amount distribution** anchored to category averages
- **Temporal patterns**: 18 months of data with hour-of-day and day-of-week seasonality
- **5,000 unique merchants** with power-law transaction distribution
- **Pre-authorization risk score**: computed as a sigmoid-based composite of category, geography, payment method, and transaction amount — simulating what a real payment risk model (like Stripe Radar) produces upstream

The failure model uses a **sigmoid decision boundary** rather than pure probability sampling, creating a more realistic outcome where high-risk transactions fail reliably and low-risk transactions rarely fail — enabling AUC of 0.84+ with logistic regression.

### SQL Analysis (`analysis.ipynb`, sections 2–3)
All core analysis is written in SQL (via SQLite in-memory):
- Platform-level failure summary
- Dimensional breakdown by category, geography, payment method
- Failure code distribution and retryability classification
- Monthly trend analysis

### Revenue Leakage Quantification (section 4)
```
Leakage = Σ amount_usd where is_retryable=1 AND retry_recovered=0
```
Quantified at **~$8.7M** across the dataset. Top recovery opportunities:
1. Travel: $5.4M (40.3% failure rate — highest absolute leakage due to large avg transaction)
2. Marketplaces: $1.1M (24% failure rate)
3. E-commerce: $1.1M (10.8% failure rate, high volume)

### Predictive Model (section 5)
**Goal:** Flag transactions likely to fail *before* authorization, enabling intelligent retry routing or alternative payment method suggestion.

**Features:**
- Log-transformed transaction amount
- Encoded merchant category, geography, payment method
- Temporal features (hour, day of week, month, weekend flag)
- Pre-authorization risk score (upstream composite risk signal)

**Model:** Logistic Regression with `class_weight='balanced'` to handle class imbalance (87% success / 13% failure).

**Results:**
- ROC-AUC: **0.824**
- 5-Fold CV AUC: 0.826 ± 0.002
- Precision on flagged high-risk transactions: 34% (2.7× baseline lift)

### Tableau Dashboard (section 6 + `exports/`)
Three exported CSV files power the Tableau dashboard:

| File | Rows | Purpose |
|---|---|---|
| `tableau_main.csv` | 5,760 | Primary datasource: failure/leakage by category × geo × method × month |
| `tableau_failure_codes.csv` | 766 | Failure code drill-down |
| `tableau_monthly_trend.csv` | 18 | Monthly trend view |

**Dashboard views (build in Tableau):**
1. **Executive Summary** — total leakage, failure rate, recovery opportunity (KPI cards)
2. **Leakage by Category** — horizontal bar chart ranked by recoverable revenue
3. **Geo Heatmap** — failure rate by geography with currency annotations
4. **Failure Code Breakdown** — stacked bar (retryable vs. non-retryable) with recovery rates
5. **Monthly Trend** — line chart with failure rate over 18 months
6. **Payment Method Risk** — scatter plot of failure rate vs. avg transaction amount

---

## Business Recommendations

1. **Intelligent retry routing for `do_not_honor` and `processing_error`** — these codes have 60–88% recovery rates. Triggering automated retries for flagged transactions in these categories is the highest-ROI action.

2. **Geo-specific acquiring bank optimization for BR and IN** — failure rates are 35–45% above platform average. Investigating local acquiring bank partnerships and routing logic for these markets can recover disproportionate revenue.

3. **BNPL pre-authorization friction** — highest failure rate payment method (1.4× average). Adding soft pre-authorization checks or segment-specific limits before the authorization attempt reduces downstream failure volume.

4. **Deploy pre-auth scoring model upstream** — the logistic regression model (AUC 0.824) can be served as a lightweight real-time scorer at the payment intent stage, enabling:
   - Route to backup payment method before hard decline
   - Trigger 3DS for borderline transactions
   - Flag for human review in high-value fraud cases

5. **Travel category SLA monitoring** — contributes the most absolute leakage (~$5.4M) despite not having the highest failure *rate*, purely due to large average transaction size. A dedicated Tableau alert on travel failure rate spikes is high value.

