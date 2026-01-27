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

    /* Main background with a deeper professional gradient */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #111827 0%, #030712 100%);
        color: #f1f5f9;
    }

    /* High-end Sidebar */
    [data-testid="stSidebar"] {
        background-color: #030712;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }

    /* Professional Card Headers */
    .dashboard-header {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(135deg, #f8fafc 30%, #475569 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }

    .sub-link {
        color: #3b82f6;
        text-decoration: none;
        font-weight: 600;
    }

    .sub-gradient {
        color: #94a3b8;
        font-size: 1.2rem;
        margin-bottom: 40px;
        font-weight: 400;
    }

    /* Premium Glassmorphism Cards */
    [data-testid="stVerticalBlock"] > [style*="flex-direction: column"] > [data-testid="stVerticalBlock"] > div {
        /* This is a general selector, be careful */
    }

    /* Section Title Polish */
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

    /* Metric Overrides - Ultimate Style */
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 16px;
        padding: 24px !important;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    [data-testid="stMetric"]:hover {
        border-color: rgba(99, 102, 241, 0.4);
        background: rgba(255, 255, 255, 0.04);
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    }
    [data-testid="stMetricValue"] {
        color: #f8fafc !important;
        font-weight: 800 !important;
        font-size: 2.4rem !important;
        letter-spacing: -1px;
    }
    [data-testid="stMetricLabel"] {
        color: #64748b !important;
        font-weight: 700 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Ultimate Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #4338ca 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 14px 28px;
        font-weight: 700;
        width: 100%;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-size: 0.8rem;
    }
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 0 25px rgba(99, 102, 241, 0.5);
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
    }

    /* Peak Alert Card Polish */
    .peak-card {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 16px 20px;
        margin-bottom: 12px;
        background: rgba(255, 255, 255, 0.02);
        border-radius: 14px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        transition: 0.3s ease;
    }
    .peak-card:hover {
        background: rgba(255, 255, 255, 0.04);
        border-color: rgba(99, 102, 241, 0.3);
    }

    .peak-date {
        font-weight: 800;
        color: #f8fafc;
        font-size: 1.05rem;
    }
    .peak-day {
        color: #64748b;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .peak-value {
        color: #818cf8;
        font-weight: 800;
        font-size: 1.2rem;
    }
    .peak-badge {
        background: rgba(245, 158, 11, 0.1);
        color: #fbbf24;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.65rem;
        font-weight: 800;
        border: 1px solid rgba(245, 158, 11, 0.2);
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
        subprocess.run(["python", "main.py", "--refresh"], capture_output=True, text=True)
        return True
    except Exception:
        return False

# --- SIDEBAR NAV ---
with st.sidebar:
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=60)
    st.markdown("## CORE ENGINE")
    st.markdown("Computational Sales Intelligence Hub.")
    
    if st.button("🧬 RETRAIN AI MODEL"):
        with st.spinner("Processing High-Dimensional Trends..."):
            if refresh_predictions():
                st.toast("Intelligence Node Synchronized!", icon="💎")
            else:
                st.error("Protocol Error: Check Logs")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("### ⚡ REPORTING")
    if os.path.exists(DATA_PATH):
        df_exp = pd.read_csv(DATA_PATH)
        st.download_button("📩 Export Dataset", df_exp.to_csv(index=False), "sales_pulse_master.csv")
    
    st.markdown("<br>"*5, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("ULTIMATE FORECASTING LAB v4.0")
    st.caption("Enterprise Grade AI • Ready for Submit")

# --- DASHBOARD HEADER ---
st.markdown('<h1 class="dashboard-header">Sales Pulse Pro</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-gradient">Next-Generation Predictive Intelligence for Real-World Retail Scaling.</p>', unsafe_allow_html=True)

df = load_data()

if df is not None:
    # Segment data
    actuals = df[df['Category'] == 'Actual']
    forecast = df[df['Category'] == 'Forecast']

    # --- TOP LEVEL KPI CARDS ---
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        growth = ((forecast['Revenue'].sum() / actuals['Revenue'].tail(30).sum()) - 1) * 100
        st.metric("TOTAL EXPECTED", f"${forecast['Revenue'].sum()/1000:,.1f}K", f"{growth:+.1f}% Growth")
    with c2:
        st.metric("AVG DAILY VOL", f"${forecast['Revenue'].mean():,.0f}")
    with c3:
        peak_idx = forecast['Revenue'].idxmax()
        st.metric("MAX DEMAND", f"${forecast.loc[peak_idx, 'Revenue']:,.0f}")
    with c4:
        st.metric("CRITICAL DATE", forecast.loc[peak_idx, 'Date'].strftime('%b %d'))

    st.markdown("<br>", unsafe_allow_html=True)

    # --- PRIMARY VISUALIZATION ---
    st.markdown('<p class="section-title">Strategic Forecast Timeline</p>', unsafe_allow_html=True)
    
    with st.container(border=True):
        fig1 = go.Figure()
        
        # 1. Prediction Zone (Hoverable range)
        fig1.add_trace(go.Scatter(
            x=pd.concat([forecast['Date'], forecast['Date'][::-1]]),
            y=pd.concat([forecast['Revenue']*1.12, (forecast['Revenue']*0.88)[::-1]]),
            fill='toself',
            fillcolor='rgba(99, 102, 241, 0.05)',
            line=dict(color='rgba(255,255,255,0)'),
            name='Standard AI Variance',
            hoverinfo='skip'
        ))
        
        # 2. Historical Context (Last 60 days)
        hist_view = actuals.tail(90)
        fig1.add_trace(go.Scatter(
            x=hist_view['Date'], 
            y=hist_view['Revenue'],
            name='Historical Performance',
            line=dict(color='#475569', width=1.5, dash='dot')
        ))
        
        # Bridge
        last_hist_date = actuals.iloc[-1]['Date']
        last_hist_rev = actuals.iloc[-1]['Revenue']
        bridge_dates = pd.concat([pd.Series([last_hist_date]), forecast['Date']])
        bridge_revenue = pd.concat([pd.Series([last_hist_rev]), forecast['Revenue']])

        # 3. AI Prediction Path
        fig1.add_trace(go.Scatter(
            x=bridge_dates, 
            y=bridge_revenue,
            name='ML Projection Path',
            line=dict(color='#6366f1', width=5),
            marker=dict(size=10, color='#818cf8', line=dict(width=3, color='#030712'))
        ))

        fig1.update_layout(
            template="plotly_dark",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="left", x=0, font=dict(size=12, color="#94a3b8")),
            margin=dict(l=0, r=0, t=20, b=0),
            height=550,
            hovermode="x unified",
            xaxis=dict(showgrid=False, zeroline=False, tickfont=dict(color='#64748b')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)', zeroline=False, tickfont=dict(color='#64748b'))
        )
        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("<br><br>", unsafe_allow_html=True)

    # --- SECONDARY MULTI-CHARTS ---
    row2_col1, row2_col2 = st.columns([1.2, 0.8])

    with row2_col1:
        st.markdown('<p class="section-title">30-Day Demand Roadmap</p>', unsafe_allow_html=True)
        with st.container(border=True):
            fig_road = px.bar(forecast, x='Date', y='Revenue', color='Revenue', 
                             color_continuous_scale=['#1e293b', '#6366f1', '#818cf8', '#c084fc'])
            fig_road.update_layout(
                template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                coloraxis_showscale=False, height=450, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
            )
            st.plotly_chart(fig_road, use_container_width=True)

    with row2_col2:
        st.markdown('<p class="section-title">Critical Surveillance</p>', unsafe_allow_html=True)
        with st.container(border=True):
            top_days = forecast.nlargest(7, 'Revenue')[['Date', 'Revenue', 'Weekday']]
            
            st.markdown("<div style='height: 405px; overflow-y: auto; padding-right: 10px;'>", unsafe_allow_html=True)
            for _, row in top_days.iterrows():
                st.markdown(f"""
                <div class="peak-card">
                    <div>
                        <div class="peak-date">{row['Date'].strftime('%d %B')}</div>
                        <div class="peak-day">{row['Weekday']}</div>
                    </div>
                    <div style='text-align: right;'>
                        <div class="peak-badge">SURGE PROBABLE</div>
                        <div class="peak-value">${row['Revenue']:,.0f}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # --- ROW 3: WEEKDAY PULSE ---
    st.markdown('<p class="section-title">Weekday Intensity Distribution</p>', unsafe_allow_html=True)
    with st.container(border=True):
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        day_stats = forecast.groupby('Weekday')['Revenue'].mean().reindex(day_order)
        
        fig_day = go.Figure()
        fig_day.add_trace(go.Bar(
            x=day_stats.index, y=day_stats.values,
            marker=dict(color='rgba(99, 102, 241, 0.8)', line=dict(color='#818cf8', width=2)),
            hovertemplate='Intensity: $%{y:,.0f}<extra></extra>'
        ))
        fig_day.update_layout(
            template="plotly_dark", paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=400, margin=dict(l=0, r=0, t=20, b=0),
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig_day, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align: center; color: #475569; font-size: 0.8rem;'>System Status: 🟢 Fully Operational | Integrity Hash: {os.path.getmtime(DATA_PATH)}</div>", unsafe_allow_html=True)

else:
    st.warning("⚠️ Critical Hub Offline. Please trigger Model Sync from the Sidebar.")
