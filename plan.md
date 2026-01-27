# Implementation Plan - Sales & Demand Forecasting

## 1. Project Setup
- Goal: Forecast daily sales for an online retail business.
- Dataset: Online Retail.xlsx
- KPIs: RMSE, MAE, MAPE.

## 2. Methodology
- **Preprocessing**: 
    - Convert `InvoiceDate` to datetime.
    - Remove cancellations (negative Quantity).
    - Calculate `TotalRevenue = Quantity * UnitPrice`.
    - Handle outliers (e.g., bulk orders that aren't representative).
- **Aggregation**: Group by date to get daily total revenue.
- **Feature Engineering**:
    - Lag features (t-1, t-7, t-30).
    - Rolling window features (7-day and 30-day means).
    - Extraction of calendar features (Day of week, Month, etc.).
- **Models**:
    - Baseline: Simple Moving Average.
    - Advanced: XGBoost/Random Forest Regressor.
- **Evaluation**: Time-series split cross-validation.

## 3. Deliverables
- `sales_forecasting.ipynb`: Complete analysis and model.
- `requirements.txt`: Necessary libraries.
- `README.md`: Business context and results summary.
