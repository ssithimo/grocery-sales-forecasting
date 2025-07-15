# 🛒 Alomega Food Stores Sales Forecasting

## 📈 Project Overview

This project presents a time series forecasting analysis for **Alomega Food Stores**, a hypothetical Midwestern retail grocery chain. Using monthly sales data from **January 2003 to November 2006**, the goal is to develop robust statistical models that accurately predict future sales and support inventory planning and operational decisions.

---

## 🗂️ Dataset

**Source:** `MidtermData.csv`  
**Description:** 47 months of monthly sales data  
**Column:**
- `Sales`: Monthly total sales in USD

**Data Split:**
- **Training set:** January 2003 – December 2005 (36 months)  
- **Validation set:** January 2006 – November 2006 (11 months)

---

## 🔍 Methods

Two forecasting models were developed and evaluated:

### 1. Holt-Winters Exponential Smoothing (ETS)
- Selected Model: **MNM** (Multiplicative Error, No Trend, Multiplicative Seasonality)
- Automatically fitted using R's `ets()` function

### 2. Moving Average Model
- 3-month trailing moving average (**MA(3)**)
- Implemented using `rollmean()` from the **zoo** package

---

## 📊 Evaluation Metrics

Each model was evaluated based on:
- **MAPE**: Mean Absolute Percentage Error  
- **RMSE**: Root Mean Squared Error  
- **Residual diagnostics**: Normality, constant variance, autocorrelation

---

## 📌 Key Results

| Model         | MAPE (Train) | MAPE (Test) | Jan 2006 Forecast |
|---------------|--------------|-------------|-------------------|
| Holt-Winters  | 10.97%       | 19.80%      | $573,437.70       |
| MA(3)         | 19.64%       | 29.43%      | $311,000.00       |

- ✅ **Best model:** Holt-Winters (MNM)
- 📦 **December 2006 Forecast (Holt-Winters):** **$226,910.50**

---

## 📁 Files

- `alomega.csv`: Source data  
- `grocery_sales_HWModeling.Rmd`: Full analysis and plots  
- `README.md`: Project summary

---

## 🧠 Recommendations

- Update the model regularly as new sales data becomes available
- Explore external variables (e.g., promotions, weather, holidays) to improve predictive power
- Use the Holt-Winters model for operational forecasting, with regular monitoring of residuals for assumption violations

---

## 🔧 Tools Used

- R (base + `forecast`, `zoo`, `ggplot2`)
- RMarkdown for reproducible reporting

---

## 📬 Contact

For questions or suggestions, feel free to open an issue or contact the project maintainer.
