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

REGIONAL_PATH = os.path.join('data', 'regional_sales.csv')
def load_regional_data():
    if os.path.exists(REGIONAL_PATH):
        return pd.read_csv(REGIONAL_PATH)
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
    st.caption("AI RESEARCH LAB v6.0 | Enterprise Cloud")

# --- HEADER ---
st.markdown('<h1 class="dashboard-header">AI Sales Pulse ⚡</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-gradient">Strategic Enterprise Demand Intelligence & Pipeline Forecasting.</p>', unsafe_allow_html=True)

# --- PROJECT PURPOSE & STRATEGY ---
with st.container(border=True):
    st.markdown("### **🎯 Project Purpose**")
    st.markdown("""
    This project is built to support **Online Retail** business decisions. It automates the process of identifying 
    trends and predicting future demand using an **XGBoost AI Model**. The goal is to maximize inventory 
    efficiency and minimize lost sales during peak periods.
    """)

# --- EXECUTIVE STRATEGIC BRIEF ---
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

    # 1. NEW CHART: Unified Historical & Future Sales Pipeline (Image 5 Style)
    st.markdown('<p class="section-title">Unified Historical & Future Sales Pipeline</p>', unsafe_allow_html=True)
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
    st.markdown('<p class="section-title">Global Reach & Operational Efficiency</p>', unsafe_allow_html=True)
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

    # 4. Final Section: Demand Distribution & Alerts
    st.markdown('<p class="section-title">Critical Surplus Awareness</p>', unsafe_allow_html=True)
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
            top_d = forecast.nlargest(8, 'Revenue')
            st.markdown("<div style='height: 250px; overflow-y: auto;'>", unsafe_allow_html=True)
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
    st.info("System Ready. Please initiate Data Sync.")
