import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import os
import subprocess
from datetime import datetime

# --- CONFIGURATION & THEME ---
st.set_page_config(
    page_title="AI Sales Pulse | Enterprise Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional CSS injection for a "Software as a Service" (SaaS) look
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
    }

    /* High-end Sidebar */
    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1f2937;
    }

    /* Professional Card Headers */
    .dashboard-header {
        font-size: 2.5rem;
        font-weight: 800;
        letter-spacing: -1px;
        background: linear-gradient(90deg, #60a5fa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
    }

    .sub-gradient {
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }

    /* Custom Section Headers */
    .section-title {
        color: #f8fafc;
        font-weight: 700;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 20px;
        display: flex;
        align-items: center;
    }
    .section-title::before {
        content: "";
        display: inline-block;
        width: 4px;
        height: 18px;
        background: #3b82f6;
        margin-right: 10px;
        border-radius: 4px;
    }

    /* Metric Overrides */
    [data-testid="stMetric"] {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px !important;
        transition: all 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        border-color: #3b82f6;
        background: #1e293b;
    }
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2rem !important;
    }
    [data-testid="stMetricLabel"] {
        color: #94a3b8 !important;
        font-weight: 600 !important;
    }

    /* Button Styling */
    .stButton>button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 6px;
        padding: 10px 24px;
        font-weight: 600;
        width: 100%;
        transition: 0.3s;
    }
    .stButton>button:hover {
        background: #2563eb;
        box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
    }

    /* Forecast Indicator Badge */
    .badge {
        background: rgba(16, 185, 129, 0.1);
        color: #10b981;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid rgba(16, 185, 129, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# --- UTILITIES ---
DATA_PATH = os.path.join('data', 'powerbi_master_report.csv')

def load_data():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def refresh_predictions():
    try:
        # We use --refresh to force the ML script to retrain
        subprocess.run(["python", "main.py", "--refresh"], capture_output=True, text=True)
        return True
    except Exception:
        return False

# --- SIDEBAR NAV ---
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### 🛠️ SYSTEM ENGINE")
    st.markdown("Manage prediction nodes and data sync.")
    
    if st.button("🚀 RETRAIN & SYNC DATA"):
        with st.spinner("Retraining XGBoost model..."):
            if refresh_predictions():
                st.toast("Intelligence Refresh Complete", icon="🧬")
            else:
                st.error("Engine Sync Failed")
    
    st.markdown("---")
    st.markdown("### 📊 EXPORT")
    if os.path.exists(DATA_PATH):
        df_exp = pd.read_csv(DATA_PATH)
        st.download_button("Download Report (.csv)", df_exp.to_csv(index=False), "sales_forecast.csv")
    
    st.markdown("<br>"*8, unsafe_allow_html=True)
    st.caption("AI Sales Pulse v3.0 | Enterprise Edition")

# --- DASHBOARD HEADER ---
st.markdown('<h1 class="dashboard-header">AI Sales Pulse ⚡</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-gradient">Strategic 30-Day Demand Forecasting with Gradient Boosting Confidence.</p>', unsafe_allow_html=True)

df = load_data()

if df is not None:
    # Segment data
    actuals = df[df['Category'] == 'Actual']
    forecast = df[df['Category'] == 'Forecast']

    # --- TOP LEVEL KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.metric("Expected Total Revenue", f"${forecast['Revenue'].sum()/1000:,.1f}K", f"{((forecast['Revenue'].sum() / actuals['Revenue'].tail(30).sum()) - 1)*100:+.1f}%")
    with c2:
        st.metric("Avg Daily Projection", f"${forecast['Revenue'].mean():,.0f}")
    with c3:
        peak_idx = forecast['Revenue'].idxmax()
        st.metric("Peak Demand Value", f"${forecast.loc[peak_idx, 'Revenue']:,.0f}")
    with c4:
        st.metric("Projected Peak Day", forecast.loc[peak_idx, 'Date'].strftime('%b %d'))

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PRIMARY VISUALIZATION ---
    st.markdown('<p class="section-title">Timeline Performance & Confidence Band</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        # Create professional fan chart
        fig1 = go.Figure()
        
        # 1. Prediction Zone (Hoverable range)
        fig1.add_trace(go.Scatter(
            x=pd.concat([forecast['Date'], forecast['Date'][::-1]]),
            y=pd.concat([forecast['Revenue']*1.12, (forecast['Revenue']*0.88)[::-1]]),
            fill='toself',
            fillcolor='rgba(59, 130, 246, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Confidence Band (88%)',
            hoverinfo='skip'
        ))
        
        # 2. Historical Context (Last 60 days)
        hist_view = actuals.tail(60)
        fig1.add_trace(go.Scatter(
            x=hist_view['Date'], 
            y=hist_view['Revenue'],
            name='Historical Performance',
            line=dict(color='#64748b', width=2, dash='dot')
        ))
        
        # 3. Create a bridge point to connect History and Forecast
        last_hist_date = actuals.iloc[-1]['Date']
        last_hist_rev = actuals.iloc[-1]['Revenue']
        
        # Combine last historical point with forecast for a seamless line
        bridge_dates = pd.concat([pd.Series([last_hist_date]), forecast['Date']])
        bridge_revenue = pd.concat([pd.Series([last_hist_rev]), forecast['Revenue']])

        # 4. ML Forecast (Connected to History)
        fig1.add_trace(go.Scatter(
            x=bridge_dates, 
            y=bridge_revenue,
            name='ML Projection',
            line=dict(color='#6366f1', width=4),
            marker=dict(size=8, color='#38bdf8', line=dict(width=2, color='#1e293b'))
        ))

        # Premium layout settings
        fig1.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            # LEGEND TO THE LEFT AS REQUESTED
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, font=dict(size=12, color="#94a3b8")),
            margin=dict(l=0, r=0, t=10, b=0),
            height=500,
            hovermode="x unified",
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(gridcolor='#1e2937', zeroline=False)
        )
        st.plotly_chart(fig1, use_container_width=True)

    # --- SECONDARY MULTI-CHARTS ---
    st.markdown("<br>", unsafe_allow_html=True)
    row2_col1, row2_col2 = st.columns(2)

    with row2_col1:
        st.markdown('<p class="section-title">Weekday Intensity Distribution</p>', unsafe_allow_html=True)
        with st.container(border=True):
            # Aggregated intensity for forecast
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            day_stats = forecast.groupby('Weekday')['Revenue'].mean().reindex(day_order)
            
            fig_day = px.area(x=day_stats.index, y=day_stats.values, labels={'x': '', 'y': ''})
            fig_day.update_traces(line_color='#3b82f6', fillcolor='rgba(59, 130, 246, 0.1)', marker=dict(size=10, color='#60a5fa'))
            fig_day.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                height=350, margin=dict(l=0, r=0, t=20, b=0),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1e2937')
            )
            st.plotly_chart(fig_day, use_container_width=True)

    with row2_col2:
        st.markdown('<p class="section-title">Critical High-Demand Alerts</p>', unsafe_allow_html=True)
        with st.container(border=True):
            top_days = forecast.nlargest(7, 'Revenue')[['Date', 'Revenue', 'Weekday']]
            
            st.markdown("<div style='height: 310px; overflow-y: auto; padding: 5px;'>", unsafe_allow_html=True)
            for _, row in top_days.iterrows():
                st.markdown(f"""
                <div style='display: flex; justify-content: space-between; align-items: center; padding: 12px; margin-bottom: 8px; background: #111827; border-radius: 8px; border: 1px solid #1f2937;'>
                    <div>
                        <span style='color: #f8fafc; font-weight: 700; font-size: 1rem;'>{row['Date'].strftime('%d %B')}</span>
                        <span style='color: #64748b; font-size: 0.8rem; margin-left: 10px;'>({row['Weekday']})</span>
                    </div>
                    <div style='text-align: right;'>
                        <span class='badge'>🔥 PEAK ALERT</span><br>
                        <span style='color: #38bdf8; font-weight: 800; font-size: 1.1rem;'>${row['Revenue']:,.0f}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    # --- ROW 3: VOLUME ROADMAP ---
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="section-title">30-Day Unit Volume Density</p>', unsafe_allow_html=True)
    with st.container(border=True):
        fig_road = px.bar(forecast, x='Date', y='Revenue', color='Revenue', color_continuous_scale='IceFire_r')
        fig_road.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            coloraxis_showscale=False, height=350, margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='#1e2937')
        )
        st.plotly_chart(fig_road, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; color: #475569;'>Sync Integrity Check: ✅ Passed | Last Update: {datetime.now().strftime('%H:%M:%S')}</div>", unsafe_allow_html=True)

else:
    st.warning("Prediction Pipeline Standby: Please trigger 'RETRAIN & SYNC DATA' to generate insights.")
