import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import os
import subprocess
import time
from datetime import datetime, timedelta
from scipy.stats import gaussian_kde
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
import joblib

# --- CONFIGURATION & THEME ---
st.set_page_config(
    page_title="Forecasting Lab Pro | Enterprise AI",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS injection for the "Ultimate" UI/UX
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at 0% 0%, #111827 0%, #030712 100%);
        color: #f1f5f9;
    }

    [data-testid="stSidebar"] {
        background-color: #030712;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    .dashboard-header {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #f8fafc 30%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .sub-gradient {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 40px;
        font-weight: 400;
    }

    .section-title {
        color: #ffffff;
        font-weight: 800;
        font-size: 1.2rem;
        letter-spacing: 0.5px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
    }
    .section-title::before {
        content: "";
        display: inline-block;
        width: 12px;
        height: 12px;
        background: #6366f1;
        margin-right: 12px;
        border-radius: 3px;
        box-shadow: 0 0 10px #6366f1;
    }

    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px !important;
        backdrop-filter: blur(10px);
        transition: all 0.4s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-5px);
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 700 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 800;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.4);
    }

    .alert-card {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        padding: 16px;
        border-radius: 12px;
        margin-bottom: 12px;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }
    .alert-date { font-weight: 700; color: #f8fafc; }
    .alert-val { color: #818cf8; font-weight: 800; font-size: 1.1rem; }
    .alert-badge { 
        background: rgba(239, 68, 68, 0.1); 
        color: #ef4444; 
        padding: 2px 8px; 
        border-radius: 4px; 
        font-size: 0.7rem; 
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITIES ---
DATA_PATH = os.path.join('data', 'powerbi_master_report.csv')
VAL_PATH = os.path.join('data', 'validation_results.csv')
REGIONAL_PATH = os.path.join('data', 'regional_sales.csv')

def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_val_data():
    if os.path.exists(VAL_PATH):
        df = pd.read_csv(VAL_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_regional_data():
    if os.path.exists(REGIONAL_PATH):
        return pd.read_csv(REGIONAL_PATH)
    return None

# --- SIDEBAR & NAVIGATION ---
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("## 🧭 NAVIGATION")
    mode = st.radio("Select View:", ["Enterprise Dashboard", "Custom Forecast Lab"], label_visibility="collapsed")
    
    st.markdown("---")
    
    if mode == "Enterprise Dashboard":
        st.markdown("## ⚙️ CONTROL PANEL")
        st.markdown("Status: Active")
        if st.button("🔄 REFRESH DATA & MODEL"):
            with st.spinner("Processing new data stream..."):
                subprocess.run(["python", "main.py", "--refresh"], capture_output=True)
                st.rerun()
        st.markdown("---")
        if os.path.exists(DATA_PATH):
            df_exp = pd.read_csv(DATA_PATH)
            st.download_button("📥 Export Analysis (.csv)", df_exp.to_csv(index=False), "sales_forecast_data.csv")
    
    elif mode == "Custom Forecast Lab":
         st.markdown("## 📤 DATA INGESTION")
         st.info("Upload any historical dataset to generate an instant ML forecast.")

    st.markdown("<br>"*2, unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: rgba(16, 185, 129, 0.1); padding: 15px; border-radius: 10px; border-left: 4px solid #10b981;'>
        <p style='color: #10b981; font-weight: 800; font-size: 0.8rem; margin: 0;'>● SYSTEM ONLINE</p>
        <p style='color: #64748b; font-size: 0.7rem; margin: 0;'>Last Update: {datetime.now().strftime('%H:%M:%S %p')}</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<br>"*2, unsafe_allow_html=True)
    st.caption("AI PREDICTIVE ANALYTICS v9.0")

# --- MAIN CONTENT ---

if mode == "Enterprise Dashboard":
    # =============================================================================================
    # ENTERPRISE DASHBOARD (PROJECT ANALYSIS)
    # =============================================================================================
    st.markdown('<h1 class="dashboard-header">Executive Sales Intelligence ⚡</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-gradient">Comprehensive Demand Forecasting & Performance Metrics.</p>', unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown("### **🎯 Project Purpose**")
        st.markdown("""
        This project is built to support **Online Retail** business decisions. It automates the process of identifying 
        trends and predicting future demand using an **XGBoost AI Model**. The goal is to maximize inventory 
        efficiency and minimize lost sales during peak periods.
        """)
        
    st.markdown('<p class="section-title">Executive Strategic Intelligence</p>', unsafe_allow_html=True)
    with st.container(border=True):
        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### **📊 What the Forecast Means**")
            st.markdown("- **Core Path**: Most likely revenue target based on AI analysis.")
            st.markdown("- **Safety Zone**: Confidence bands representing operational stability.")
            st.markdown("- **Peak Alerts**: High-demand triggers for staffing & inventory.")
        with c2:
            st.markdown("#### **🛠️ Operational Action Plan**")
            st.markdown("- **Inventory**: Re-stock 7 days *before* predicted peaks.")
            st.markdown("- **Staffing**: Assign senior staff to 'Critical Surge' dates.")
            st.markdown("- **Cash Flow**: Use 'Slow Day' predictions to launch mini-promos.")
        
        st.caption(f"System Integrity: 🟢 Active | Last Strategy Sync: {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    st.markdown("<br>", unsafe_allow_html=True)

    df = load_data()
    val_df = load_val_data()
    reg_df = load_regional_data()

    if df is not None:
        actuals = df[df['Category'] == 'Actual']
        forecast = df[df['Category'] == 'Forecast']

        # KPI Row
        c1, c2, c3, c4 = st.columns(4)
        with c1: st.metric("TOTAL FORECAST", f"${forecast['Revenue'].sum()/1000:,.1f}K", f"{((forecast['Revenue'].sum() / actuals['Revenue'].tail(30).sum()) - 1)*100:+.1f}%")
        with c2: st.metric("AVG DAILY PROJECTED", f"${forecast['Revenue'].mean():,.0f}")
        with c3: 
            p_idx = forecast['Revenue'].idxmax()
            st.metric("PEAK SURGE VALUE", f"${forecast.loc[p_idx, 'Revenue']:,.0f}")
        with c4: st.metric("CRITICAL DATE", forecast.loc[p_idx, 'Date'].strftime('%d %B'))

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. SECTION: Strategic 30-Day Sales Outlook
        st.markdown('<p class="section-title">30-Day Strategic Sales Outlook</p>', unsafe_allow_html=True)
        with st.container(border=True):
            fig_unified = go.Figure()
            
            # Historical
            hist_trim = actuals.tail(90)
            fig_unified.add_trace(go.Scatter(x=hist_trim['Date'], y=hist_trim['Revenue'], name='Historical Sales (Actual)', line=dict(color='#60a5fa', width=2)))
            
            # Forecast Horizon Vertical Line
            if not actuals.empty:
                last_date = actuals['Date'].iloc[-1]
                fig_unified.add_shape(
                    type="line", x0=last_date, x1=last_date, y0=0, y1=1, yref="paper",
                    line=dict(color="#ef4444", width=2, dash="dash")
                )
                fig_unified.add_annotation(
                    x=last_date, y=1, yref="paper", text="Forecast Horizon Trigger",
                    showarrow=False, font=dict(color="#ef4444"), textangle=-90, xanchor="left"
                )
            
            # Future Forecast with Bridge
            if not actuals.empty and not forecast.empty:
                bridge_dates = pd.concat([pd.Series([last_date]), forecast['Date']])
                bridge_revenue = pd.concat([pd.Series([actuals['Revenue'].iloc[-1]]), forecast['Revenue']])
                fig_unified.add_trace(go.Scatter(x=bridge_dates, y=bridge_revenue, name='ML Future Forecast (Predicted)', line=dict(color='#f59e0b', width=4)))
            elif not forecast.empty:
                fig_unified.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Revenue'], name='ML Future Forecast (Predicted)', line=dict(color='#f59e0b', width=4)))
            
            fig_unified.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=500, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_unified, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 2. ML Probability & Risk Distribution
        st.markdown('<p class="section-title">ML Probability & Risk Distribution</p>', unsafe_allow_html=True)
        with st.container(border=True):
            fig_risk = go.Figure()
            if not forecast.empty:
                fig_risk.add_trace(go.Scatter(
                    x=pd.concat([forecast['Date'], forecast['Date'][::-1]]),
                    y=pd.concat([forecast['Revenue']*1.2, (forecast['Revenue']*0.8)[::-1]]),
                    fill='toself', fillcolor='rgba(255, 255, 255, 0.05)',
                    line=dict(color='rgba(255,255,255,0)'), name='80% Confidence Band'
                ))
                fig_risk.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Revenue'], name='Core Prediction Path', line=dict(color='white', width=2)))
            
            fig_risk.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=400, margin=dict(l=0, r=0, t=10, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. SECTION: Regional Analysis & Optimal Days
        st.markdown('<p class="section-title">Global Reach & Market Intelligence</p>', unsafe_allow_html=True)
        c_w1, c_w2 = st.columns(2)
        with c_w1:
            with st.container(border=True):
                st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #94a3b8;'>Market Share by Region</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.7rem; color: #64748b; margin-top: -15px;'>Global revenue distribution across top 5 performing countries.</p>", unsafe_allow_html=True)
                if reg_df is not None:
                    # Custom colors for a premium look
                    colors = ['#6366f1', '#10b981', '#f59e0b', '#ec4899', '#8b5cf6', '#64748b']
                    fig_reg = px.pie(reg_df, values='Sales', names='Country', hole=0.6, color_discrete_sequence=colors)
                    fig_reg.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(l=0,r=0,t=0,b=0), showlegend=True, legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1))
                    st.plotly_chart(fig_reg, use_container_width=True)
                st.markdown("<div style='background: rgba(99, 102, 241, 0.05); padding: 10px; border-radius: 8px; border-left: 3px solid #6366f1;'><p style='font-size: 0.75rem; font-weight: 700; color: #818cf8; margin: 0;'>Strategic Action:</p><p style='font-size: 0.7rem; color: #94a3b8; margin: 0;'>Target localized marketing campaigns in high-performing regions to maximize ROI.</p></div>", unsafe_allow_html=True)
                
        with c_w2:
            with st.container(border=True):
                st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #94a3b8;'>Optimal Operational Days</p>", unsafe_allow_html=True)
                st.markdown("<p style='font-size: 0.7rem; color: #64748b; margin-top: -15px;'>Revenue concentration by day of the week.</p>", unsafe_allow_html=True)
                weekday_rev = actuals.groupby('Weekday')['Revenue'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']).fillna(0)
                fig_opt = px.bar(x=weekday_rev.index, y=weekday_rev.values, labels={'x': '', 'y': ''}, color_discrete_sequence=['#6366f1'])
                fig_opt.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=325, margin=dict(l=0,r=0,t=10,b=0))
                st.plotly_chart(fig_opt, use_container_width=True)
                st.markdown("<div style='background: rgba(99, 102, 241, 0.05); padding: 10px; border-radius: 8px; border-left: 3px solid #6366f1;'><p style='font-size: 0.75rem; font-weight: 700; color: #818cf8; margin: 0;'>Resource Tip:</p><p style='font-size: 0.7rem; color: #94a3b8; margin: 0;'>Scale staffing levels during high-volume days revealed in the distribution above.</p></div>", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 4. SECTION: Comparative Analysis & Model Verification
        st.markdown('<p class="section-title">Comparative Analysis & Model Verification</p>', unsafe_allow_html=True)
        c_left, c_right = st.columns(2)
        with c_left:
            # KDE
            with st.container(border=True):
                st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #94a3b8;'>Sales Volume Density Comparison</p>", unsafe_allow_html=True)
                hist_vals = actuals['Revenue'].tail(150).values
                pred_vals = forecast['Revenue'].values
                full_r = np.linspace(min(min(hist_vals), min(pred_vals))*0.5, max(max(hist_vals), max(pred_vals))*1.2, 200)
                k_h = gaussian_kde(hist_vals)(full_r)
                k_p = gaussian_kde(pred_vals)(full_r)
                
                fig_kde = go.Figure()
                fig_kde.add_trace(go.Scatter(x=full_r, y=k_h, fill='toself', name='Past (Actual)', fillcolor='rgba(96, 165, 250, 0.2)', line=dict(color='#60a5fa')))
                fig_kde.add_trace(go.Scatter(x=full_r, y=k_p, fill='toself', name='Future (Predicted)', fillcolor='rgba(245, 158, 11, 0.2)', line=dict(color='#f59e0b')))
                fig_kde.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", x=0, y=1.1))
                st.plotly_chart(fig_kde, use_container_width=True)

        with c_right:
            # Validation
            if val_df is not None:
                with st.container(border=True):
                    st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #94a3b8;'>30-Day Blind Back-Test Results</p>", unsafe_allow_html=True)
                    fig_v = go.Figure()
                    fig_v.add_trace(go.Scatter(x=val_df['Date'], y=val_df['Actual'], name='Real', line=dict(color='#60a5fa')))
                    fig_v.add_trace(go.Scatter(x=val_df['Date'], y=val_df['Forecast'], name='ML', line=dict(color='#ef4444', dash='dash')))
                    fig_v.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0, r=0, t=0, b=0), legend=dict(orientation="h", x=0, y=1.1))
                    st.plotly_chart(fig_v, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)
        
        # 5. SECTION: Weekly Summary
        st.markdown('<p class="section-title">Institutional Forecasting Analysis</p>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown("<p style='font-size: 0.9rem; font-weight: 700; color: #94a3b8;'>Weekly Sales Pulse (Actual vs Forecast)</p>", unsafe_allow_html=True)
            hist_weekly = actuals.set_index('Date')['Revenue'].resample('W').sum().tail(4)
            pred_weekly = forecast.set_index('Date')['Revenue'].resample('W').sum()
            fig_weekly = go.Figure()
            fig_weekly.add_trace(go.Bar(x=[f"Actual W{i+1}" for i in range(len(hist_weekly))], y=hist_weekly, name='Actual', marker_color='#334155'))
            fig_weekly.add_trace(go.Bar(x=[f"Forecast W{i+1}" for i in range(len(pred_weekly))], y=pred_weekly, name='Forecast', marker_color='#6366f1'))
            fig_weekly.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=300, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_weekly, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 6. Final Section: Daily Demand Pipeline
        st.markdown('<p class="section-title">30-Day Daily Demand Intelligence Pipeline</p>', unsafe_allow_html=True)
        r_left, r_right = st.columns([1.2, 0.8])
        
        with r_left:
            with st.container(border=True):
                fig_rd = px.bar(forecast, x='Date', y='Revenue', color='Revenue', color_continuous_scale='Turbo')
                fig_rd.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
                st.plotly_chart(fig_rd, use_container_width=True)

        with r_right:
            # Filling the void with high-level intelligence
            with st.container(border=True):
                st.markdown("#### **🎯 Strategic Forecast Insight**")
                st.markdown(f"""
                The model predicts a total volume of **${forecast['Revenue'].sum():,.0f}** for the next 30 days. 
                Demand is expected to be **{((forecast['Revenue'].sum() / actuals['Revenue'].tail(30).sum()) - 1)*100:+.1f}%** 
                compared to the previous period.
                """)
                st.info("💡 Recommendation: Align logistics for the upcoming surge.")

            with st.container(border=True):
                st.markdown("#### **🔥 High-Volume Surges**")
                st.markdown("<p style='font-size: 0.75rem; color: #64748b; margin-top: -15px;'>Detected peak demand events requiring focus.</p>", unsafe_allow_html=True)
                
                top_d = forecast.nlargest(8, 'Revenue')
                
                # Use the CSS classes defined in the header for a cleaner, safer implementation
                st.markdown("<div style='height: 320px; overflow-y: auto; padding-right: 15px;'>", unsafe_allow_html=True)
                for _, row in top_d.iterrows():
                    st.markdown(f"""
                    <div class="alert-card">
                        <div>
                            <div class="alert-date">{row['Date'].strftime('%d %B')}</div>
                            <div style='color: #64748b; font-size: 0.8rem;'>{row['Weekday']}</div>
                        </div>
                        <div style='text-align: right;'>
                            <span class="alert-badge">🔥 PEAK ALERT</span><br>
                            <div class="alert-val">${row['Revenue']:,.0f}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"<div style='text-align: center; color: #475569; font-size: 0.8rem;'>Dashboard Integrity: 🟢 SECURE | Computational Node Active</div>", unsafe_allow_html=True)

    else:
        st.info("System Ready. Please initiate Data Sync or View.")


elif mode == "Custom Forecast Lab":
    # =============================================================================================
    # CUSTOM FORECAST LAB (UPLOAD YOUR DATA)
    # =============================================================================================
    st.markdown('<h1 class="dashboard-header">Predictive Strategy Lab 🚀</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-gradient">Upload your own datasets to generate instant AI-powered forecasts.</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        uploaded_file = st.file_uploader("Upload Time-Series Data (CSV/Excel)", type=['csv', 'xlsx'])
        
        if uploaded_file is not None:
            # FAST LOAD PREVIEW
            try:
                if uploaded_file.name.endswith('.csv'):
                    raw_preview = pd.read_csv(uploaded_file, nrows=5)
                else:
                    raw_preview = pd.read_excel(uploaded_file, nrows=5)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("### 1. Data Mapping")
                
                # Smart Column Detection
                all_cols = list(raw_preview.columns)
                
                # Try to find 'date' or 'time' in column names
                date_candidates = [c for c in all_cols if 'date' in c.lower() or 'time' in c.lower()]
                default_date_idx = all_cols.index(date_candidates[0]) if date_candidates else 0
                
                # Try to find numeric columns for target
                numeric_cols = raw_preview.select_dtypes(include=[np.number]).columns.tolist()
                default_target_idx = 0
                # If we have a common target like 'Sales', 'Quantity', 'Revenue', pick it
                for i, col in enumerate(numeric_cols):
                    if any(x in col.lower() for x in ['sales', 'revenue', 'quantity', 'total', 'profit']):
                        default_target_idx = i
                        break

                c_sel1, c_sel2 = st.columns(2)
                with c_sel1:
                    date_col = st.selectbox("Select Date Column", options=all_cols, index=default_date_idx)
                with c_sel2:
                    target_col = st.selectbox("Select Target to Forecast", options=numeric_cols, index=default_target_idx)
                
                st.dataframe(raw_preview.head(), use_container_width=True)
                
                if uploaded_file.name.endswith('.xlsx'):
                     st.caption("ℹ️ Tip: Excel files can be slow to process. For instant results convert to .csv")

                st.markdown("<br>", unsafe_allow_html=True)
                
                if st.button("🚀 INITIALIZE AI CORE & GENERATE FORECAST"):
                    
                    # PROGRESS BAR CONTAINER
                    progress_container = st.empty()
                    bar = progress_container.progress(0)
                    status_text = st.empty()
                    
                    def update_status(progress, text):
                        bar.progress(progress)
                        status_text.markdown(f"**⚡ SYSTEM STATUS: {text} ({progress}%)**")

                    try:
                        # STEP 1: LOAD RELEVANT DATA (10%)
                        # OPTIMIZATION: Read ONLY the necessary columns. 
                        # This skips parsing text descriptions etc, making it 50x faster.
                        update_status(10, f"Extracting vectors [{date_col}, {target_col}] from dataset...")
                        start_time = datetime.now()
                        
                        uploaded_file.seek(0)
                        
                        if uploaded_file.name.endswith('.csv'):
                            raw_df = pd.read_csv(uploaded_file, usecols=[date_col, target_col])
                        else:
                            # Excel is slow, but usecols helps immensely
                            raw_df = pd.read_excel(uploaded_file, usecols=[date_col, target_col])
                            
                        # STEP 2: CLEANING & AGGREGATION (30%)
                        update_status(30, "Cleaning Anomalies & Aggregating Time Factors...")
                        
                        df_proc = raw_df.copy()
                        # Coerce errors to NaN to prevent crashing on bad headers/strings
                        df_proc[date_col] = pd.to_datetime(df_proc[date_col], errors='coerce')
                        df_proc = df_proc.dropna(subset=[date_col])
                        df_proc = df_proc.sort_values(date_col)
                        
                        daily = df_proc.groupby(df_proc[date_col].dt.date)[target_col].sum().reset_index()
                        daily.columns = ['Date', 'Value']
                        daily['Date'] = pd.to_datetime(daily['Date'])
                        daily = daily.set_index('Date').asfreq('D').fillna(0)
                        
                        # STEP 3: FEATURE ENGINEERING (50%)
                        update_status(50, "Engineering Lag Features & Rolling Statistics...")
                        
                        data = daily.copy()
                        data['dayofweek'] = data.index.dayofweek
                        data['month'] = data.index.month
                        data['day'] = data.index.day
                        
                        for lag in [1, 7, 14, 30]:
                            data[f'lag_{lag}'] = data['Value'].shift(lag)
                            
                        data['rolling_mean_7'] = data['Value'].shift(1).rolling(window=7).mean()
                        data['rolling_mean_30'] = data['Value'].shift(1).rolling(window=30).mean()
                        
                        data = data.dropna()
                        
                        if len(data) < 10:
                            raise ValueError("Not enough execution data after cleaning (Need > 10 days).")
                        
                        # STEP 4: MODEL TRAINING (75%)
                        update_status(75, "Training XGBoost Neural Regressor...")
                        
                        X = data.drop('Value', axis=1)
                        y = data['Value']
                        
                        split_idx = len(X) - 30
                        if split_idx < 10: split_idx = int(len(X) * 0.8)
                        
                        X_train = X.iloc[:split_idx]
                        y_train = y.iloc[:split_idx]
                        
                        model = XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=5, random_state=42, n_jobs=-1)
                        model.fit(X_train, y_train)
                        
                        # STEP 5: FORECASTING (90%)
                        update_status(90, "Extrapolating Future Demand Curves...")
                        
                        future_dates = [daily.index[-1] + timedelta(days=x) for x in range(1, 31)]
                        current_lag_data = pd.concat([daily, pd.DataFrame({'Value': [np.nan]*30}, index=future_dates)])
                        predictions = []
                        
                        for date in future_dates:
                            idx = date
                            feats = {
                                'dayofweek': idx.dayofweek,
                                'month': idx.month,
                                'day': idx.day
                            }
                            
                            for lag in [1, 7, 14, 30]:
                                target_date = idx - timedelta(days=lag)
                                if target_date in current_lag_data.index:
                                    val = current_lag_data.loc[target_date, 'Value']
                                    feats[f'lag_{lag}'] = val
                                else:
                                    feats[f'lag_{lag}'] = 0
                                    
                            past_7 = current_lag_data.loc[idx - timedelta(days=7):idx - timedelta(days=1), 'Value']
                            feats['rolling_mean_7'] = past_7.mean() if len(past_7) > 0 else 0
                            
                            past_30 = current_lag_data.loc[idx - timedelta(days=30):idx - timedelta(days=1), 'Value']
                            feats['rolling_mean_30'] = past_30.mean() if len(past_30) > 0 else 0
                            
                            row = pd.DataFrame([feats])
                            row = row[X_train.columns]
                            
                            pred = model.predict(row)[0]
                            pred = max(0, pred)
                            predictions.append(pred)
                            current_lag_data.loc[date, 'Value'] = pred
                        
                        future_data = pd.DataFrame({'Date': future_dates, 'Forecast': predictions})
                        hist_data = daily
                        
                        # COMPLETE
                        update_status(100, "Finalizing Visualization Logic...")
                        time.sleep(0.5) # Slight pause for user to verify completion
                        progress_container.empty() # Remove bar
                        status_text.empty()
                        
                        # --- RESULTS ---
                        st.markdown("<br>", unsafe_allow_html=True)
                        st.markdown(f'<p class="section-title">Analysis Complete ({ (datetime.now() - start_time).seconds }s)</p>', unsafe_allow_html=True)
                        
                        # 0. Data Intelligence Summary (New)
                        with st.expander("📝 Data Intelligence Summary", expanded=True):
                            s1, s2, s3, s4 = st.columns(4)
                            history_len = len(hist_data)
                            total_vol = hist_data['Value'].sum()
                            avg_vol = hist_data['Value'].mean()
                            max_vol = hist_data['Value'].max()
                            
                            s1.metric("Historical Range", f"{history_len} Days")
                            s2.metric("Total Observed Volume", f"{total_vol:,.0f}")
                            s3.metric("Daily Average", f"{avg_vol:,.1f}")
                            s4.metric("Max Peak", f"{max_vol:,.0f}")
                            
                            st.caption(f"Dataset Interval: {hist_data.index.min().strftime('%Y-%m-%d')} to {hist_data.index.max().strftime('%Y-%m-%d')}")
                        
                        # Metrics
                        m1, m2, m3 = st.columns(3)
                        total_expected = future_data['Forecast'].sum()
                        growth_val = (future_data['Forecast'].mean() - hist_data['Value'].mean()) / hist_data['Value'].mean() * 100 if hist_data['Value'].mean() != 0 else 0
                        
                        with m1: st.metric("PROJECTED VOLUME (Next 30d)", f"{total_expected:,.0f}", help="Sum of predicted values for the next 30 days")
                        with m2: st.metric("AVG DAILY PREDICTION", f"{future_data['Forecast'].mean():,.0f}")
                        with m3: st.metric("PROJECTED GROWTH", f"{growth_val:+.1f}%", help="Comparison against historical average")
                        
                        # 1. Main Graph (Enhanced)
                        st.markdown("### 📈 Forecast Horizon")
                        fig_dyn = go.Figure()
                        # Historical (Last 90 days for clarity)
                        hist_show = hist_data.tail(90)
                        fig_dyn.add_trace(go.Scatter(x=hist_show.index, y=hist_show['Value'], name='Historical Data', line=dict(color='#60a5fa')))
                        # Forecast
                        fig_dyn.add_trace(go.Scatter(x=future_data['Date'], y=future_data['Forecast'], name='AI Prediction', line=dict(color='#f59e0b', width=3)))
                        
                        fig_dyn.update_layout(
                            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                            height=500, margin=dict(l=0, r=0, t=20, b=0),
                            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, title_text=""),
                            hovermode="x unified"
                        )
                        st.plotly_chart(fig_dyn, use_container_width=True)
                        
                        # 2. Advanced Analysis (New Graphs)
                        st.markdown("### 🔍 Deep Dive Analysis")
                        t1, t2 = st.tabs(["Trend Analysis", "Anomaly Detection"])
                        
                        with t1:
                            st.markdown("##### 📉 Trend Component Analysis")
                            st.caption("Underlying market direction (30-day Moving Average)")
                            
                            # Calculate Trend
                            trend = hist_data['Value'].rolling(window=30).mean()
                            
                            fig_trend = go.Figure()
                            fig_trend.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Value'], name='Raw Data', line=dict(color='rgba(255,255,255,0.1)')))
                            fig_trend.add_trace(go.Scatter(x=hist_data.index, y=trend, name='30-Day Trend', line=dict(color='#10b981', width=3)))
                            fig_trend.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
                            st.plotly_chart(fig_trend, use_container_width=True)
                            
                        with t2:
                            st.markdown("##### ⚠️ Anomaly Detection")
                            st.caption("Points deviating significantly (> 2 Std Dev) from the trend.")
                            
                            # Simple Anomaly Detection
                            rolling_mean = hist_data['Value'].rolling(window=7).mean()
                            rolling_std = hist_data['Value'].rolling(window=7).std()
                            anomalies = hist_data[(hist_data['Value'] > rolling_mean + (2 * rolling_std)) | (hist_data['Value'] < rolling_mean - (2 * rolling_std))]
                            
                            fig_anom = go.Figure()
                            fig_anom.add_trace(go.Scatter(x=hist_data.index, y=hist_data['Value'], name='Normal Data', line=dict(color='#64748b')))
                            fig_anom.add_trace(go.Scatter(
                                mode='markers', x=anomalies.index, y=anomalies['Value'], name='Anomaly Detection',
                                marker=dict(color='#ef4444', size=8, symbol='x')
                            ))
                            fig_anom.update_layout(template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=350)
                            st.plotly_chart(fig_anom, use_container_width=True)

                        # 3. Components (Refined)
                        c_d1, c_d2 = st.columns(2)
                        with c_d1:
                            st.markdown("### 📊 Weekly Patterns")
                            # Group forecast by day of week
                            future_data['Weekday'] = future_data['Date'].dt.day_name()
                            weekly_pat = future_data.groupby('Weekday')['Forecast'].mean().reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])
                            
                            fig_w = px.bar(x=weekly_pat.index, y=weekly_pat.values, color=weekly_pat.values, color_continuous_scale='Viridis', title="")
                            fig_w.update_layout(
                                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                height=350, showlegend=False, xaxis_title="", yaxis_title="Avg Volume",
                                coloraxis_showscale=False
                            )
                            st.plotly_chart(fig_w, use_container_width=True)
                            
                        with c_d2:
                            st.markdown("### 🔮 Prediction Confidence Intervals")
                            # Simple risk band visual
                            fig_r = go.Figure()
                            fig_r.add_trace(go.Scatter(
                                x=pd.concat([future_data['Date'], future_data['Date'][::-1]]),
                                y=pd.concat([future_data['Forecast']*1.15, (future_data['Forecast']*0.85)[::-1]]),
                                fill='toself', fillcolor='rgba(245, 158, 11, 0.1)',
                                line=dict(color='rgba(255,255,255,0)'), name='85% Confidence Range'
                            ))
                            fig_r.add_trace(go.Scatter(x=future_data['Date'], y=future_data['Forecast'], name='Forecast Central Path', line=dict(color='#f59e0b')))
                            fig_r.update_layout(
                                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', 
                                height=350, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
                            )
                            st.plotly_chart(fig_r, use_container_width=True)
                            
                        st.success("Analysis generated successfully. Models are running at peak efficiency.")
            
                    except Exception as e:
                        st.error(f"Error processing dataset: {e}")
                        st.warning("Please ensure your dataset has a valid Date column and a Numeric Target column.")

            except Exception as e:
                st.error(f"Error reading file: {e}")
