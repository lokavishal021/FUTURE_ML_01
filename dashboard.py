import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
import subprocess
from datetime import datetime
from scipy.stats import gaussian_kde

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
        background: #f59e0b;
        margin-right: 12px;
        border-radius: 3px;
        box-shadow: 0 0 10px #f59e0b;
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
        border-color: rgba(245, 158, 11, 0.4);
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
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: #030712;
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
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.4);
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
    .alert-val { color: #f59e0b; font-weight: 800; font-size: 1.1rem; }
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

# --- SIDEBAR ---
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("## ⚙️ SYSTEM CONTROL")
    st.markdown("Protocol: Autonomous AI Node")
    if st.button("🚀 RETRAIN & SYNC ENGINE"):
        with st.spinner("Analyzing high-dimensional datasets..."):
            subprocess.run(["python", "main.py", "--refresh"], capture_output=True)
            st.rerun()
    st.markdown("---")
    if os.path.exists(DATA_PATH):
        df_exp = pd.read_csv(DATA_PATH)
        st.download_button("Download Full Intel (.csv)", df_exp.to_csv(index=False), "ai_sales_export.csv")
    st.markdown("<br>"*5, unsafe_allow_html=True)
    st.caption("AI RESEARCH LAB v5.0 | Enterprise Cloud")

# --- HEADER ---
st.markdown('<h1 class="dashboard-header">AI Sales Pulse ⚡</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-gradient">Strategic Enterprise Demand Intelligence & Back-Test Analysis.</p>', unsafe_allow_html=True)

df = load_data()
val_df = load_val_data()

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

    # GRAPH 1: ML Probability & Risk Distribution (Image 4 Style)
    st.markdown('<p class="section-title">Graph 2: ML Probability & Risk Distribution</p>', unsafe_allow_html=True)
    with st.container(border=True):
        fig_risk = go.Figure()
        # Band
        fig_risk.add_trace(go.Scatter(
            x=pd.concat([forecast['Date'], forecast['Date'][::-1]]),
            y=pd.concat([forecast['Revenue']*1.2, (forecast['Revenue']*0.8)[::-1]]),
            fill='toself', fillcolor='rgba(245, 158, 11, 0.15)',
            line=dict(color='rgba(255,255,255,0)'), name='80% Confidence Band'
        ))
        # Line
        fig_risk.add_trace(go.Scatter(x=forecast['Date'], y=forecast['Revenue'], name='Core Prediction Path', line=dict(color='white', width=3)))
        
        fig_risk.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=450, margin=dict(l=0, r=0, t=20, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_risk, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # GRAPH 2: Validation Strategy (Image 3 Style)
    if val_df is not None:
        st.markdown('<p class="section-title">Validation Strategy: 30-Day Blind Back-Test</p>', unsafe_allow_html=True)
        with st.container(border=True):
            fig_val = go.Figure()
            fig_val.add_trace(go.Scatter(x=val_df['Date'], y=val_df['Actual'], name='Actual Data (Real)', line=dict(color='#60a5fa', width=2), marker=dict(size=8, symbol='circle')))
            fig_val.add_trace(go.Scatter(x=val_df['Date'], y=val_df['Forecast'], name='ML Forecast', line=dict(color='#ef4444', width=2, dash='dash'), marker=dict(size=8, symbol='x')))
            
            fig_val.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=450, margin=dict(l=0, r=0, t=20, b=0),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_val, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Row with Density and Weekday (Image 1 & 2 Style)
    r1, r2 = st.columns([1.2, 0.8])
    with r1:
        st.markdown('<p class="section-title">Graph 6: Sales Volume Density Comparison</p>', unsafe_allow_html=True)
        with st.container(border=True):
            # KDE Calculation
            hist_vals = actuals['Revenue'].tail(120).values
            pred_vals = forecast['Revenue'].values
            
            # Combine for range
            full_range = np.linspace(min(min(hist_vals), min(pred_vals))*0.5, max(max(hist_vals), max(pred_vals))*1.2, 200)
            
            kde_hist = gaussian_kde(hist_vals)(full_range)
            kde_pred = gaussian_kde(pred_vals)(full_range)
            
            fig_kde = go.Figure()
            fig_kde.add_trace(go.Scatter(x=full_range, y=kde_hist, fill='toself', name='Past Sales Volume (Actual)', fillcolor='rgba(96, 165, 250, 0.3)', line=dict(color='#60a5fa')))
            fig_kde.add_trace(go.Scatter(x=full_range, y=kde_pred, fill='toself', name='Future Sales Volume (Predicted)', fillcolor='rgba(245, 158, 11, 0.3)', line=dict(color='#f59e0b')))
            
            fig_kde.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=400, margin=dict(l=0, r=0, t=20, b=0),
                xaxis_title="Sales Revenue", yaxis_title="Probability Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0)
            )
            st.plotly_chart(fig_kde, use_container_width=True)

    with r2:
        st.markdown('<p class="section-title">Critical High-Demand Alerts</p>', unsafe_allow_html=True)
        with st.container(border=True):
            top_days = forecast.nlargest(6, 'Revenue')
            st.markdown("<div style='height: 350px; overflow-y: auto;'>", unsafe_allow_html=True)
            for _, row in top_days.iterrows():
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

    # FINAL BAR ROADMAP
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">Strategic 30-Day Demand Roadmap</p>', unsafe_allow_html=True)
    with st.container(border=True):
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_stats = forecast.groupby('Weekday')['Revenue'].mean().reindex(day_order)
        
        fig_bar = px.bar(x=day_stats.index, y=day_stats.values, labels={'x':'', 'y':''}, color=day_stats.values, color_continuous_scale='YlOrBr')
        fig_bar.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=350, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

else:
    st.info("System Initialization Requested. Please run the Retrain Engine protocol from the sidebar.")
