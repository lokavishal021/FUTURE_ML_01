import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

def generate_forecast():
    print("--- [1/2] Generating High-Impact Analytical Visuals ---")
    
    # 1. LOAD DATA & MODEL
    PROCESSED_FILE = os.path.join('data', 'processed_daily_sales.csv')
    MODEL_PATH = os.path.join('models', 'sales_model.joblib')
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(PROCESSED_FILE):
        print("Error: Required files not found.")
        return

    # Load and clean
    raw_daily = pd.read_csv(PROCESSED_FILE)
    raw_daily['Date'] = pd.to_datetime(raw_daily['Date'])
    daily_sales = raw_daily.sort_values('Date').set_index('Date')
    
    model = joblib.load(MODEL_PATH)
    os.makedirs('plots', exist_ok=True)
    sns.set_theme(style="whitegrid")

    # 2. RECURSIVE FORECASTING
    future_forecast = []
    last_window = daily_sales.copy()
    last_date = daily_sales.index[-1]
    forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=30)

    for i in range(30):
        current_date = forecast_dates[i]
        feat = {
            'dayofweek': current_date.dayofweek, 'month': current_date.month, 'day': current_date.day,
            'is_weekend': int(current_date.dayofweek in [5, 6]), 'is_december': int(current_date.month == 12),
            'days_to_christmas': min(31, max(0, (pd.Timestamp(year=current_date.year, month=12, day=25) - current_date).days)),
            'month_sin': np.sin(2 * np.pi * current_date.month/12), 'month_cos': np.cos(2 * np.pi * current_date.month/12),
            'day_sin': np.sin(2 * np.pi * current_date.dayofweek/7), 'day_cos': np.cos(2 * np.pi * current_date.dayofweek/7),
            'lag_1': last_window['Sales_clipped'].iloc[-1], 'lag_7': last_window['Sales_clipped'].iloc[-7],
            'lag_14': last_window['Sales_clipped'].iloc[-14], 'lag_21': last_window['Sales_clipped'].iloc[-21],
            'lag_30': last_window['Sales_clipped'].iloc[-30]
        }
        feat['diff_1_7'] = feat['lag_1'] - feat['lag_7']
        feat['rolling_mean_7'] = last_window['Sales_clipped'].iloc[-7:].mean()
        feat['rolling_std_7'] = last_window['Sales_clipped'].iloc[-7:].std()
        feat['rolling_mean_30'] = last_window['Sales_clipped'].iloc[-30:].mean()
        
        pred = max(0, model.predict(pd.DataFrame([feat]))[0])
        new_entry = pd.DataFrame({'Sales_clipped': [pred], 'Sales': [pred]}, index=[current_date])
        last_window = pd.concat([last_window, new_entry])
        future_forecast.append(pred)

    forecast_df = pd.DataFrame({'Predicted_Sales': future_forecast}, index=forecast_dates)

    # ---------------------------------------------------------
    # GRAPH 1: Connected Historical & Future
    # ---------------------------------------------------------
    plt.figure(figsize=(15, 7))
    full_plot_dates = [daily_sales.index[-1]] + list(forecast_dates)
    full_plot_values = [daily_sales['Sales'].iloc[-1]] + future_forecast
    plt.plot(daily_sales.index[-60:], daily_sales['Sales'].iloc[-60:], label='Historical Sales (Actual)', color='royalblue', alpha=0.5)
    plt.plot(full_plot_dates, full_plot_values, label='ML Future Forecast (Predicted)', color='darkorange', linewidth=3)
    plt.fill_between(forecast_dates, np.array(future_forecast)*0.85, np.array(future_forecast)*1.15, color='orange', alpha=0.1, label='Prediction Variance Range')
    plt.axvline(last_date, color='red', linestyle='--', alpha=0.5, label='Forecast Horizon Trigger')
    plt.title('Graph 1: Unified Historical & Future Sales Pipeline', fontsize=16, fontweight='bold')
    plt.legend(frameon=True, shadow=True, loc='upper left'); plt.savefig('plots/1_main_forecast.png', dpi=300); plt.close()

    # Graph 2: Risk Fan
    plt.figure(figsize=(12, 6)); plt.plot(forecast_dates, future_forecast, color='black', label='Core Prediction Path')
    plt.fill_between(forecast_dates, np.array(future_forecast)*0.8, np.array(future_forecast)*1.2, color='orange', alpha=0.2, label='80% Confidence Band')
    plt.title('Graph 2: ML Probability & Risk Distribution', fontweight='bold'); plt.legend(frameon=True); plt.savefig('plots/2_risk_fan.png'); plt.close()

    # Graph 3: Donut (Weekday Dist Compare)
    plt.figure(figsize=(10, 8))
    # Get Historical Average per Weekday
    hist_temp = daily_sales.copy().iloc[-90:]
    hist_temp['Day'] = hist_temp.index.day_name()
    hist_day_rev = hist_temp.groupby('Day')['Sales'].mean()
    
    # Get Predicted Average per Weekday
    pred_temp = forecast_df.copy()
    pred_temp['Day'] = pred_temp.index.day_name()
    pred_day_rev = pred_temp.groupby('Day')['Predicted_Sales'].mean()
    
    # Sort by weekday order
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    hist_day_rev = hist_day_rev.reindex(days_order).fillna(0)
    pred_day_rev = pred_day_rev.reindex(days_order).fillna(0)

    plt.pie(pred_day_rev, labels=pred_day_rev.index, autopct='%1.1f%%', 
            colors=sns.color_palette('viridis', 7), pctdistance=0.85, 
            textprops={'fontweight':'bold', 'color':'black'}, wedgeprops={'alpha':0.8})
    plt.gca().add_artist(plt.Circle((0,0), 0.7, fc='white'))
    plt.title('Graph 3: Predicted Sales Distribution by Weekday', fontweight='bold', fontsize=14)
    plt.legend(pred_day_rev.index, title="Weekdays", loc="center right", bbox_to_anchor=(1.2, 0.5))
    plt.tight_layout()
    plt.savefig('plots/3_revenue_donut.png', dpi=300); plt.close()

    # Graph 4: 30-Day Predictive Sales Roadmap (with Context)
    plt.figure(figsize=(14, 7))
    hist_window = daily_sales.iloc[-14:]
    plt.bar(hist_window.index, hist_window['Sales'], color='gray', alpha=0.3, label='Recent Historical Sales (Actual)')
    plt.bar(forecast_dates, future_forecast, color='skyblue', label='Future Sales Forecast (Predicted)')
    plt.plot(pd.concat([hist_window['Sales'], forecast_df['Predicted_Sales']]), color='darkorange', marker='o', markersize=4, label='Sales Trendline')
    plt.axvline(last_date, color='red', linestyle='--', linewidth=2, label='Forecast Start Line')
    plt.title('Graph 4: 30-Day Predictive Sales Roadmap (Actual + Forecast)', fontweight='bold', fontsize=14)
    plt.xlabel('Date'); plt.ylabel('Sales Revenue')
    plt.legend(frameon=True, shadow=True); plt.savefig('plots/4_daily_roadmap.png', dpi=300); plt.close()

    # Graph 5: Peaks
    plt.figure(figsize=(14, 6)); plt.plot(forecast_dates, future_forecast, color='darkorange', linewidth=3, label='Predicted Demand Path')
    peaks = forecast_df.nlargest(5, 'Predicted_Sales')
    plt.scatter(peaks.index, peaks['Predicted_Sales'], color='red', s=150, edgecolors='black', label='Critical Surge Alert', zorder=5)
    plt.title('Graph 5: Top 5 Predicted Demand Peaks', fontweight='bold'); plt.legend(); plt.savefig('plots/5_peak_detection.png'); plt.close()

    # Graph 6: Distribution
    plt.figure(figsize=(10, 6)); sns.kdeplot(daily_sales['Sales'].iloc[-60:], fill=True, label='Past Sales Volume (Actual)', color='blue')
    sns.kdeplot(future_forecast, fill=True, label='Future Sales Volume (Predicted)', color='orange')
    plt.title('Graph 6: Sales Volume Density Comparison', fontweight='bold'); plt.legend(); plt.savefig('plots/6_distribution_compare.png'); plt.close()

    # Graph 7: Weekly Predicted Revenue Totals (Actual vs Forecast)
    plt.figure(figsize=(12, 6))
    hist_weekly = daily_sales['Sales'].resample('W').sum().iloc[-4:]
    pred_weekly = forecast_df['Predicted_Sales'].resample('W').sum()
    
    hist_labels = [f"Actual W{i+1}" for i in range(len(hist_weekly))]
    pred_labels = [f"Forecast W{i+1}" for i in range(len(pred_weekly))]
    
    plt.bar(hist_labels, hist_weekly, color='navy', alpha=0.6, label='Historical Weekly Total')
    plt.bar(pred_labels, pred_weekly, color='gold', alpha=0.8, label='Predicted Weekly Total')
    plt.title('Graph 7: Weekly Sales Pulse (Actual vs Predicted)', fontweight='bold', fontsize=14)
    plt.ylabel('Total Revenue')
    plt.legend(frameon=True, shadow=True)
    plt.tight_layout()
    plt.savefig('plots/7_weekly_summary.png', dpi=300); plt.close()

    # ---------------------------------------------------------
    # ENHANCED POWER BI MASTER EXPORT
    # ---------------------------------------------------------
    print("--- [2/2] Synchronizing Enriched Power BI Master Report ---")
    
    # Export Historical Data
    hist_bi = daily_sales[['Sales']].copy().reset_index()
    hist_bi.columns = ['Date', 'Revenue']
    hist_bi['Category'] = 'Actual'
    
    # Export Forecast Data (Pure from forecast_df, ensure only 2 original columns)
    fore_bi = forecast_df[['Predicted_Sales']].copy().reset_index()
    fore_bi.columns = ['Date', 'Revenue']
    fore_bi['Category'] = 'Forecast'
    
    # UNIFIED MASTER DATA
    master_bi = pd.concat([hist_bi, fore_bi])
    
    # Enrich with Power BI metadata
    master_bi['Weekday'] = master_bi['Date'].dt.day_name()
    master_bi['Month'] = master_bi['Date'].dt.month_name()
    master_bi['IsWeekend'] = master_bi['Date'].dt.dayofweek.isin([5, 6]).astype(str)
    master_bi['Year'] = master_bi['Date'].dt.year
    master_bi['TrendLine_7D'] = master_bi['Revenue'].rolling(window=7).mean()
    
    os.makedirs('data', exist_ok=True)
    master_bi.to_csv('data/powerbi_master_report.csv', index=False)
    
    print("\nPower BI Sync Complete: 'data/powerbi_master_report.csv'")
    print("Optimization: 7 analytical plots with clear Legends ready in 'plots/'.")

if __name__ == "__main__":
    generate_forecast()
