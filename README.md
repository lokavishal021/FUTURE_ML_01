# 🚀 Professional Sales & Demand Forecasting System

A high-performance machine learning pipeline designed for **Online Retail** demand prediction. This system features a unified "One-Step" execution and **Power BI Integration** for enterprise-grade reporting.

---

## 🏗️ System Architecture
1.  **Preprocessing**: Cleans raw data and handles seasonality.
2.  **Training**: Optimizes an XGBoost model via Time-Series Cross-Validation.
3.  **Sync Engine**: Generates 7 high-impact graphs and a **Power BI Master Report**.

## 📊 Business Intelligence & Strategy
This system is designed to transform "Raw Data" into "Actionable Intelligence".

### **What the Forecast Means**
- **Unified Pipeline**: Connects your past sales directly to future predictions so you can see the expected trajectory of the business.
- **Risk Distribution**: The shaded confidence bands represent the 'Safe Zone'. If actual sales fall here, your business operations are stable.
- **Demand Peaks**: Automatically identifies the top 5 highest-demand days in the next 30 days.

### **Operational Action Plan**
- **Inventory Management**: Stock up on high-velocity items at least 7 days before a **Peak Alert**.
- **Staffing**: Align your most experienced staff to the **Critical Dates** identified by the AI.
- **Marketing**: Use predicted 'Valleys' (low-demand periods) to launch promotions and stabilize cash flow.

## ⚡ Quick Start
Run the master script to detect data changes and generate new forecasts:
```powershell
python main.py
```

## 📈 Interactive Web Dashboard (NEW)
We have added a professional real-time dashboard for enhanced analysis:
1. **Launch**:
   ```powershell
   streamlit run dashboard.py
   ```
2. **Features**:
   - **Hover-able Graphs**: Get exact sales figures by moving your mouse.
   - **One-Click Refresh**: Update your ML model directly from the web UI.
   - **Peak Alerts**: Instantly see your top 5 predicted high-demand days.

## 📊 Power BI Integration
This project is built to work with professional Business Intelligence tools.
1.  **Connect**: Open Power BI and "Get Data" from `data/powerbi_master_report.csv`.
2.  **Visualize**: Use the `Category` column to separate "Actual" vs "Forecast" in your charts.
3.  **Auto-Update**: Every time you run `python main.py`, the data refreshes. Just click **Refresh** in Power BI to see the new predictions!

---

## 🛠️ Installation
```powershell
pip install -r requirements.txt
```

## 📂 Project Structure
- 📄 `main.py`: Single entry point.
- 📁 `data/`: BI-ready datasets (`powerbi_master_report.csv`).
- 📁 `plots/`: 7 Premium analytical charts.
- 📁 `models/`: Trained ML "Brain" (Saved as .joblib).
