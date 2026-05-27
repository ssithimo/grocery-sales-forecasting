# 🛒 Stater Bros. Markets — December Sales Forecasting

> *For a 169-store grocery chain, a 20% forecast error in December represents $48M in planning uncertainty — here's how we quantified it.*

---

## 📈 Project Overview

Grocery retail operates on margins of 1–3% net profit, making forecast accuracy a direct operational cost driver and not just an academic metric. 
Understocking in December risks stockouts on high-demand holiday items and unrecoverable sales. Overstocking ties up working capital and forces markdowns that stress already thin margins.

This project develops a time series forecasting model for **Stater Bros. Markets** using 47 months of historical sales data (January 2022 – November 2025),to produce a reliable December 2025 forecast to support inventory planning and purchase decisions across 169 store locations.

---

## 🗂️ Dataset

**Source:** `staterbros.csv`  
**Description:** 47 months of monthly sales data  
**Column:**
- `Sales`: Monthly chain-wide sales in USD

**Data Split:**
- **Training set:** January 2022 – December 2024 (36 months)  
- **Validation set:** January 2025 – November 2025 (11 months)

---

## 🔍 Methods

Two forecasting models were developed and evaluated:

### 1. Holt-Winters Exponential Smoothing (ETS)
- Selected Model: **MNM** (Multiplicative Error, No Trend, Multiplicative Seasonality)
- Automatically fitted using R's `ets()` function
- Multiplicative seasonality reflects the reality that holiday demand spikes scale proportionally with the overall sales level
  
### 2. Moving Average Model
- 3-month trailing moving average (**MA(3)**)
- Implemented using `rollmean()` from the **zoo** package
- Used as a naive benchmark for comparison
  
---

## 📊 Model Evaluation Metrics

Each model was evaluated based on:
- **MAPE**: Mean Absolute Percentage Error  
- **RMSE**: Root Mean Squared Error  
- **Residual diagnostics**: Normality, constant variance, autocorrelation

---

## 📌 Key Results

| Model | Train MAPE | Test MAPE | Test RMSE |
|-------|-----------|-----------|-----------|
| Holt-Winters (MNM) | 10.87% | 20.39% | $127,910,148 |
| Moving Average MA(3) | 19.64% | 29.43% | $203,577,995 |

✅ **Best model:** Holt-Winters (MNM)

- Holt-Winters outperformed MA(3) on both MAPE and RMSE across the validation set and was selected as the production forecasting model.

**Residual diagnostics:** The Ljung-Box test returned a statistically 
significant result (Q* = 23.63, p = 0.001), indicating unexplained autocorrelation remains in the residuals. This is also present in the ACF, where lags 1 and 2 show threshold crossings. This finding directly motivates the incorporation of external variables in future model iterations —promotional calendars, holiday indicators, and local economic conditions are the highest-priority candidates.

---

## 📌 December 2006 Forecast & Business Impact

### Chain-Wide

| Metric | Value |
|--------|-------|
| December 2006 Forecast | $239,481,293 |
| Uncertainty Range (±20.39%) | ±$48,830,276 |
| Low Estimate | $190,651,017 |
| High Estimate | $288,311,569 |

### Per Store (169 locations)

| Metric | Value |
|--------|-------|
| Per-Store Forecast | $1,417,049 |
| Per-Store Uncertainty | ±$288,936 |
| Per-Store Range | $1,128,113 – $1,705,985 |

The $577,872 spread between the low and high per-store estimates represents meaningful inventory planning risk. In December — the highest-revenue month of the year — getting this wrong in either direction has an outsized margin impact relative to slower months.

---

## 📋 Actionable Planning Framework

Rather than treating the point forecast as a single target, buyers 
should apply the uncertainty range as a decision tool:

- **Committed inventory orders** → use low estimate ($1,128,113/store) 
  as the conservative floor for guaranteed purchase commitments
- **Baseline planning target** → use point forecast ($1,417,049/store) 
  for standard replenishment and staffing decisions
- **Contingency stock** → use high estimate ($1,705,985/store) to 
  inform safety stock levels for fast-moving seasonal and perishable SKUs

---

## 🧠 Recommendations

1. **Retrain monthly** — incorporate new sales data as they become available to keep the model current and close the train/test MAPE gap over time

2. **Add external variables** — the significant Ljung-Box result (p = 0.001) and ACF plot confirm unexplained patterns remain in the residuals; promotional calendars and holiday indicators are the highest-priority additions to improve forecast accuracy

3. **Extend the dataset** — 47 months is a limited training window; additional historical data would improve the model's ability to generalize and reduce validation MAPE

---

## 📁 Files
- `data/staterbros.csv` — Monthly sales data
- `notebooks/Grocery_Sales_HWModeling.Rmd` — Full analysis: EDA, model development, evaluation, residual diagnostics, and business impact calculations
- `Executive_summary_stater_brps.pdf` — One-page non-technical summary for stakeholders

---

## 🔧 Tools
R, RMarkdown, `forecast`, `zoo`, `ggplot2`
