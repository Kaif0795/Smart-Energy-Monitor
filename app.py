"""
app.py — Smart Energy Monitoring & Management System
=====================================================
Run with:
    streamlit run app.py

Tabs
----
     Dashboard      — KPI cards, trend line, appliance bars, pie chart
     Data Input     — Upload CSV or enter rows manually
     Predictions    — ML forecast + accuracy metrics
     Optimisation   — Tips, peak-shift analysis, efficiency grades
     Alerts         — Anomaly detection and warnings
     Reports        — Download CSV / PDF
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import os
import sys

# ── Path setup ────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(__file__)
sys.path.insert(0, BASE_DIR)

from modules.data_processor  import DataProcessor, load_and_validate
from modules.predictor       import EnergyPredictor
from modules.optimizer       import EnergyOptimizer
from modules.report_generator import export_csv, export_summary_pdf

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Energy Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── General typography ── */
html, body, [class*="css"] { font-family: 'Segoe UI', sans-serif; }

/* ── Metric cards ── */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    border: 1px solid #0f3460;
    border-radius: 12px;
    padding: 14px 18px;
    color: white;
    box-shadow: 0 4px 15px rgba(0,0,0,0.3);
}
div[data-testid="metric-container"] label {
    color: #a0c4ff !important;
    font-size: 0.78rem !important;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.6rem !important;
    font-weight: 700;
}

/* ── Section headings ── */
h2 { color: #4fc3f7 !important; }
h3 { color: #81d4fa !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2a 0%, #1b2838 100%);
}
[data-testid="stSidebar"] * { color: #cfe2f3 !important; }

/* ── Tab bar ── */
div[data-baseweb="tab-list"] button {
    font-size: 0.9rem;
    font-weight: 600;
}

/* ── Tip cards ── */
.tip-card {
    background: #0d2137;
    border-left: 4px solid #4fc3f7;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #dde8f0;
}

/* ── Alert cards ── */
.alert-card {
    background: #2a1a0a;
    border-left: 4px solid #ff9800;
    border-radius: 8px;
    padding: 10px 14px;
    margin: 6px 0;
    font-size: 0.9rem;
    color: #ffe0a0;
}

/* ── Score circle ── */
.score-box {
    text-align: center;
    padding: 20px;
    background: radial-gradient(circle, #0a2744, #0d1b2a);
    border-radius: 50%;
    width: 130px;
    height: 130px;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    margin: auto;
    border: 3px solid #4fc3f7;
}
</style>
""", unsafe_allow_html=True)

# ── Session-state defaults ────────────────────────────────────────────────────
if "df_raw"  not in st.session_state: st.session_state["df_raw"]  = None
if "proc"    not in st.session_state: st.session_state["proc"]    = None
if "pred"    not in st.session_state: st.session_state["pred"]    = None
if "optim"   not in st.session_state: st.session_state["optim"]   = None
if "rate"    not in st.session_state: st.session_state["rate"]    = 8.0


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚡ Smart Energy Monitor")
    st.markdown("---")

    # ── Data source ───────────────────────────────────────────────────────────
    st.markdown("### 📂 Data Source")
    data_source = st.radio(
        "Choose input",
        ["📁 Load Sample Dataset", "⬆️ Upload CSV"],
        label_visibility="collapsed"
    )

    if data_source == "⬆️ Upload CSV":
        uploaded = st.file_uploader(
            "Upload energy_data.csv",
            type=["csv"],
            help="Required columns: date, time, appliance, watts, duration_hours",
        )
        if uploaded:
            try:
                df_raw = load_and_validate(uploaded)
                st.session_state["df_raw"] = df_raw
                st.success(f"Loaded {len(df_raw):,} rows")
            except ValueError as e:
                st.error(str(e))
    else:
        sample_path = os.path.join(BASE_DIR, "sample_data", "energy_data.csv")
        if os.path.exists(sample_path):
            if st.button("📊 Load Sample Data", use_container_width=True):
                df_raw = pd.read_csv(sample_path)
                st.session_state["df_raw"] = df_raw
                st.success(f"Sample loaded ({len(df_raw):,} rows)")
        else:
            st.warning("Run `generate_sample_data.py` first.")

    # ── Settings ──────────────────────────────────────────────────────────────
    if st.session_state["df_raw"] is not None:
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        rate = st.number_input(
            "Electricity rate (₹/kWh)",
            min_value=1.0, max_value=50.0,
            value=st.session_state["rate"], step=0.5,
        )
        st.session_state["rate"] = rate

        forecast_days = st.slider("Forecast horizon (days)", 3, 30, 7)
        st.session_state["forecast_days"] = forecast_days

    # ── Quick stats ───────────────────────────────────────────────────────────
    if st.session_state["proc"] is not None:
        stats = st.session_state["proc"].summary_stats()
        st.markdown("---")
        st.markdown("### 📌 Quick Stats")
        st.metric("Total kWh",   f"{stats['total_kwh']:,.1f}")
        st.metric("Avg Daily",   f"{stats['avg_daily_kwh']:.2f} kWh")
        st.metric("Est. Cost",   f"₹ {stats['cost_estimate']:,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# BUILD PROCESSOR / OPTIMIZER / PREDICTOR (cached in session state)
# ══════════════════════════════════════════════════════════════════════════════
df_raw = st.session_state.get("df_raw")

if df_raw is not None:
    if st.session_state["proc"] is None or True:   # rebuild on each load
        proc  = DataProcessor(df_raw)
        optim = EnergyOptimizer(proc)
        pred  = EnergyPredictor()
        pred.train(proc.daily_consumption())
        st.session_state["proc"]  = proc
        st.session_state["optim"] = optim
        st.session_state["pred"]  = pred

proc  = st.session_state.get("proc")
optim = st.session_state.get("optim")
pred  = st.session_state.get("pred")


# ── No data loaded yet ────────────────────────────────────────────────────────
if proc is None:
    st.markdown("""
    <div style="text-align:center; padding:80px 20px;">
        <h1 style="font-size:4rem;">⚡</h1>
        <h2 style="color:#4fc3f7;">Smart Energy Monitoring System</h2>
        <p style="color:#aaa; font-size:1.1rem;">
            Load the sample dataset or upload your own CSV from the sidebar to begin.
        </p>
        <p style="color:#666; font-size:0.9rem;">
            Required CSV columns: <code>date, time, appliance, watts, duration_hours</code>
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Shared data ───────────────────────────────────────────────────────────────
stats        = proc.summary_stats()
daily_df     = proc.daily_consumption()
appliance_df = proc.appliance_consumption()
hourly_df    = proc.hourly_pattern()
dow_df       = proc.day_of_week_pattern()
forecast_days = st.session_state.get("forecast_days", 7)

# Colour palette (consistent across charts)
PALETTE = px.colors.qualitative.Bold


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_dash, tab_input, tab_pred, tab_opt, tab_alerts, tab_report = st.tabs([
    "Dashboard",
    "Data Input",
    "Predictions",
    "Optimisation",
    "Alerts",
    "Reports",
])


# ──────────────────────────────────────────────────────────────────────────────
# TAB 1 — DASHBOARD
# ──────────────────────────────────────────────────────────────────────────────
with tab_dash:
    st.header("Energy Dashboard")

    # ── KPI cards ─────────────────────────────────────────────────────────────
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: st.metric("⚡ Total kWh",      f"{stats['total_kwh']:,.2f}")
    with c2: st.metric("Avg Daily kWh",  f"{stats['avg_daily_kwh']:.2f}")
    with c3: st.metric("Peak-Hour kWh",  f"{stats['peak_kwh']:,.2f}")
    with c4: st.metric("Est. Cost",       f"₹ {stats['cost_estimate']:,.0f}")
    with c5: st.metric("Top Appliance",   stats["top_appliance"])

    st.divider()

    # ── Row 1: Line chart + Hourly heatmap ────────────────────────────────────
    col_line, col_dow = st.columns([3, 2])

    with col_line:
        st.subheader("Daily Energy Consumption (kWh)")
        # Add 7-day rolling avg
        daily_plot = daily_df.copy()
        daily_plot["7-day avg"] = daily_plot["kwh"].rolling(7, min_periods=1).mean()

        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=daily_plot["date"], y=daily_plot["kwh"],
            mode="lines", name="Daily kWh",
            line=dict(color="#4fc3f7", width=1.5),
            fill="tozeroy", fillcolor="rgba(79,195,247,0.08)",
        ))
        fig_line.add_trace(go.Scatter(
            x=daily_plot["date"], y=daily_plot["7-day avg"],
            mode="lines", name="7-day avg",
            line=dict(color="#ff9800", width=2.5, dash="dot"),
        ))
        # Peak threshold line
        threshold = daily_plot["kwh"].quantile(0.75)
        fig_line.add_hline(
            y=threshold, line_dash="dash",
            line_color="rgba(255,82,82,0.6)",
            annotation_text=f"75th pct ({threshold:.1f} kWh)",
        )
        fig_line.update_layout(
            template="plotly_dark", height=320,
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Date", yaxis_title="kWh",
        )
        st.plotly_chart(fig_line, use_container_width=True)

    with col_dow:
        st.subheader("Avg. kWh by Day of Week")
        fig_dow = px.bar(
            dow_df, x="day_of_week", y="avg_kwh",
            color="avg_kwh",
            color_continuous_scale="Blues",
            labels={"day_of_week": "", "avg_kwh": "Avg kWh"},
        )
        fig_dow.update_layout(
            template="plotly_dark", height=320,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_dow, use_container_width=True)

    st.divider()

    # ── Row 2: Bar chart + Pie chart ──────────────────────────────────────────
    col_bar, col_pie = st.columns(2)

    with col_bar:
        st.subheader("Appliance-wise Consumption (kWh)")
        fig_bar = px.bar(
            appliance_df.sort_values("kwh"),
            x="kwh", y="appliance",
            orientation="h",
            color="kwh",
            color_continuous_scale="Viridis",
            text="kwh",
            labels={"kwh": "kWh", "appliance": ""},
        )
        fig_bar.update_traces(texttemplate="%{text:.1f}", textposition="outside")
        fig_bar.update_layout(
            template="plotly_dark", height=360,
            coloraxis_showscale=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_pie:
        st.subheader("Energy Distribution (%)")
        fig_pie = px.pie(
            appliance_df,
            values="kwh", names="appliance",
            hole=0.45,
            color_discrete_sequence=PALETTE,
        )
        fig_pie.update_traces(
            textposition="inside", textinfo="percent+label",
            hovertemplate="<b>%{label}</b><br>%{value:.1f} kWh  (%{percent})<extra></extra>",
        )
        fig_pie.update_layout(
            template="plotly_dark", height=360,
            showlegend=False,
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    st.divider()

    # ── Row 3: Hourly heatmap ─────────────────────────────────────────────────
    st.subheader("Average Energy Use by Hour of Day")
    fig_hour = px.bar(
        hourly_df, x="hour", y="avg_kwh",
        color="avg_kwh", color_continuous_scale="Reds",
        labels={"hour": "Hour of Day", "avg_kwh": "Avg kWh"},
        text="avg_kwh",
    )
    # Shade peak hours
    for h in range(17, 22):
        fig_hour.add_vrect(x0=h - 0.5, x1=h + 0.5,
                           fillcolor="rgba(255,100,100,0.12)", line_width=0)
    fig_hour.update_traces(texttemplate="%{text:.3f}", textposition="outside")
    fig_hour.update_layout(
        template="plotly_dark", height=280,
        coloraxis_showscale=False,
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(tickmode="linear", dtick=1),
    )
    st.plotly_chart(fig_hour, use_container_width=True)
    st.caption("Shaded bands = peak hours (5 PM – 9 PM) — highest tariff window.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 2 — DATA INPUT
# ──────────────────────────────────────────────────────────────────────────────
with tab_input:
    st.header("Data Input & Management")

    st.subheader("Raw Data Preview")
    search_app = st.selectbox(
        "Filter by appliance",
        ["All"] + sorted(df_raw["appliance"].unique().tolist()),
        key="raw_filter"
    )
    filtered = df_raw if search_app == "All" else df_raw[df_raw["appliance"] == search_app]
    st.dataframe(filtered.head(200), use_container_width=True, height=280)
    st.caption(f"Showing {min(200, len(filtered))} of {len(filtered):,} rows "
               f"({df_raw['date'].nunique()} days, {df_raw['appliance'].nunique()} appliances)")

    st.divider()

    # ── Manual entry ──────────────────────────────────────────────────────────
    st.subheader("Add a Manual Record")
    with st.form("manual_entry", clear_on_submit=True):
        mc1, mc2, mc3 = st.columns(3)
        with mc1:
            m_date = st.date_input("Date", value=datetime.today())
            m_time = st.time_input("Time", value=datetime.now().time())
        with mc2:
            known_appliances = sorted(df_raw["appliance"].unique().tolist())
            m_app   = st.selectbox("Appliance", known_appliances + ["Other…"])
            if m_app == "Other…":
                m_app = st.text_input("Custom appliance name")
        with mc3:
            m_watts    = st.number_input("Power (Watts)", min_value=1.0, max_value=10000.0, value=100.0)
            m_duration = st.number_input("Duration (hours)", min_value=0.05, max_value=24.0, value=1.0, step=0.05)

        submitted = st.form_submit_button("➕ Add Record", use_container_width=True)
        if submitted:
            new_row = pd.DataFrame([{
                "date":           str(m_date),
                "time":           m_time.strftime("%H:%M"),
                "appliance":      m_app,
                "watts":          m_watts,
                "duration_hours": m_duration,
            }])
            st.session_state["df_raw"] = pd.concat(
                [st.session_state["df_raw"], new_row], ignore_index=True
            )
            # Force rebuild
            st.session_state["proc"] = None
            st.success(f"Added: {m_app} | {m_watts}W × {m_duration}h "
                       f"= {m_watts * m_duration / 1000:.3f} kWh")
            st.rerun()

    st.divider()

    # ── kWh calculator ────────────────────────────────────────────────────────
    st.subheader("Quick kWh Calculator")
    kc1, kc2, kc3, kc4 = st.columns(4)
    with kc1: calc_watts = st.number_input("Watts", min_value=0.0, value=1000.0, key="calc_w")
    with kc2: calc_hrs   = st.number_input("Hours/day", min_value=0.0, value=4.0, step=0.5, key="calc_h")
    with kc3: calc_days  = st.number_input("Days", min_value=1, value=30, key="calc_d")
    with kc4: calc_rate  = st.number_input("Rate (₹/kWh)", min_value=0.1, value=st.session_state["rate"], key="calc_r")
    kwh_total = calc_watts * calc_hrs * calc_days / 1000
    cost_total = kwh_total * calc_rate
    st.info(f"⚡ **{kwh_total:.2f} kWh** over {calc_days} days  ≈  **₹ {cost_total:,.2f}**")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 3 — PREDICTIONS
# ──────────────────────────────────────────────────────────────────────────────
with tab_pred:
    st.header("ML Energy Forecast")

    # ── Model metrics ─────────────────────────────────────────────────────────
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    metrics = pred.metrics
    with col_m1: st.metric("MAE",  f"{metrics.get('MAE', 0):.3f} kWh")
    with col_m2: st.metric("RMSE", f"{metrics.get('RMSE', 0):.3f} kWh")
    with col_m3: st.metric("R²",   f"{metrics.get('R²', 0):.3f}")
    with col_m4: st.metric("MAPE", f"{metrics.get('MAPE', 0):.1f}%")

    r2 = metrics.get("R²", 0)
    quality = "Excellent" if r2 >= 0.85 else ("Good" if r2 >= 0.65 else "Fair")
    st.info(f"**Model quality:** {quality}  |  **Trend:** {pred.get_trend()}")

    st.divider()

    # ── Forecast chart ────────────────────────────────────────────────────────
    st.subheader(f"📈 {forecast_days}-Day Energy Forecast")

    # In-sample fitted values
    fitted = pred.train(daily_df)   # re-get fitted values

    # Future dates
    last_date    = pd.to_datetime(daily_df["date"].max())
    future_dates = [last_date + timedelta(days=i + 1) for i in range(forecast_days)]
    future_kwh   = pred.predict(forecast_days)

    fig_pred = go.Figure()

    # Actual values
    fig_pred.add_trace(go.Scatter(
        x=daily_df["date"], y=daily_df["kwh"],
        mode="lines", name="Actual",
        line=dict(color="#4fc3f7", width=1.5),
    ))
    # Fitted
    fig_pred.add_trace(go.Scatter(
        x=daily_df["date"], y=fitted,
        mode="lines", name="Model fit",
        line=dict(color="#ff9800", width=2, dash="dot"),
    ))
    # Forecast
    fig_pred.add_trace(go.Scatter(
        x=future_dates, y=future_kwh,
        mode="lines+markers", name="Forecast",
        line=dict(color="#a5d6a7", width=2.5, dash="dash"),
        marker=dict(size=8, symbol="diamond"),
    ))
    # Confidence band (±10%)
    fig_pred.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=list(future_kwh * 1.10) + list(future_kwh * 0.90)[::-1],
        fill="toself", fillcolor="rgba(165,214,167,0.12)",
        line=dict(width=0), name="±10% band", showlegend=True,
    ))
    # Divider line
    fig_pred.add_shape(
        type="line",
        x0=last_date,
        x1=last_date,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",   # full height
        line=dict(
            color="rgba(255,255,255,0.3)",
            width=2,
            dash="longdash",
        ),
    )

# Add annotation separately (safe)
    fig_pred.add_annotation(
        x=last_date,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast →",
        showarrow=False,
        yshift=10,
        font=dict(color="white"),
    )
    fig_pred.update_layout(
        template="plotly_dark", height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Date", yaxis_title="kWh",
    )
    st.plotly_chart(fig_pred, use_container_width=True)

    # ── Forecast table ────────────────────────────────────────────────────────
    st.subheader("Forecast Detail")
    forecast_table = pd.DataFrame({
        "Date":     [d.strftime("%a, %d %b %Y") for d in future_dates],
        "Predicted kWh": np.round(future_kwh, 3),
        "Est. Cost (₹)": np.round(future_kwh * st.session_state["rate"], 2),
    })
    st.dataframe(forecast_table, use_container_width=True, hide_index=True)
    st.caption(f"Avg forecast: **{future_kwh.mean():.2f} kWh/day** "
               f"| Total over {forecast_days} days: **{future_kwh.sum():.2f} kWh**"
               f" ≈ **₹ {(future_kwh.sum() * st.session_state['rate']):,.0f}**")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 4 — OPTIMISATION
# ──────────────────────────────────────────────────────────────────────────────
with tab_opt:
    st.header("Optimisation Suggestions")

    # ── Efficiency score ──────────────────────────────────────────────────────
    score = optim.overall_score()
    sc1, sc2 = st.columns([1, 4])
    with sc1:
        colour = "#4caf50" if score >= 70 else ("#ff9800" if score >= 45 else "#f44336")
        st.markdown(
            f"""<div style="text-align:center;padding:18px;
                background:radial-gradient(circle,#0a2744,#0d1b2a);
                border-radius:50%;width:140px;height:140px;
                display:flex;flex-direction:column;align-items:center;
                justify-content:center;border:3px solid {colour};margin:auto;">
                <span style="font-size:2.6rem;font-weight:700;color:{colour};">{score}</span>
                <span style="font-size:0.7rem;color:#aaa;letter-spacing:0.05em;">EFFICIENCY</span>
            </div>""",
            unsafe_allow_html=True
        )
    with sc2:
        label = "Excellent" if score >= 70 else ("⚙️ Average" if score >= 45 else "🔴 Poor — Action Needed")
        st.subheader(f"Overall Efficiency Score: {score}/100  —  {label}")
        st.markdown(
            "This score compares your appliances against **industry benchmarks**. "
            "A score ≥ 70 means your usage is at or below reference levels. "
            "Below 45 indicates significant over-consumption in one or more areas."
        )

    st.divider()

    # ── Efficiency grades table ───────────────────────────────────────────────
    st.subheader("Appliance Efficiency Grades")
    grades = optim.get_efficiency_ratings()

    def colour_grade(g):
        colours = {"A+": "#4caf50", "A": "#8bc34a", "B": "#ff9800",
                   "C": "#ff5722", "D": "#f44336"}
        c = colours.get(g, "#aaa")
        return f"background-color:{c}22; color:{c}; font-weight:700; text-align:center"

    styled = grades.style.map(colour_grade, subset=["Grade"])
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.divider()

    # ── Peak-shift savings ────────────────────────────────────────────────────
    st.subheader("Peak-Hour Shift Potential")
    peak_info = optim.get_peak_shift_savings()
    ps1, ps2, ps3, ps4 = st.columns(4)
    with ps1: st.metric("Peak kWh",        f"{peak_info['total_peak_kwh']:,.2f}")
    with ps2: st.metric("Peak Share",      f"{peak_info['peak_ratio_pct']:.1f}%")
    with ps3: st.metric("Shiftable kWh",   f"{peak_info['shiftable_kwh']:,.2f}")
    with ps4: st.metric("Potential Saving", f"₹ {peak_info['estimated_saving']:,.0f}/month")

    peak_bd = pd.DataFrame(peak_info["breakdown"]).sort_values("peak_kwh", ascending=False)
    fig_peak = px.bar(
        peak_bd, x="appliance", y=["peak_kwh", "shiftable_kwh"],
        barmode="group",
        color_discrete_sequence=["#ef5350", "#66bb6a"],
        labels={"value": "kWh", "variable": "Category"},
    )
    fig_peak.update_layout(
        template="plotly_dark", height=280,
        legend_title_text="",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig_peak, use_container_width=True)

    st.divider()

    # ── Per-appliance tips ────────────────────────────────────────────────────
    st.subheader("🛠️ Appliance-Specific Tips")
    suggestions = optim.get_appliance_suggestions()

    for s in suggestions:
        pct_badge_colour = "#f44336" if s["pct"] > 20 else ("#ff9800" if s["pct"] > 10 else "#4caf50")
        with st.expander(
            f"{s['priority']}  **{s['appliance']}**  —  "
            f"{s['kwh']:.1f} kWh ({s['pct']:.1f}%)  "
            f"| Potential save: ~{s['saving_pct']}% ≈ {s['estimated_saving_kwh']:.2f} kWh"
        ):
            for tip in s["tips"]:
                st.markdown(
                    f'<div class="tip-card">{tip}</div>',
                    unsafe_allow_html=True
                )


# ──────────────────────────────────────────────────────────────────────────────
# TAB 5 — ALERTS
# ──────────────────────────────────────────────────────────────────────────────
with tab_alerts:
    st.header("Alerts & Anomalies")

    alerts = optim.get_alert_messages()
    if alerts:
        st.warning(f"**{len(alerts)} alert(s) detected.** Review and act below.")
        for alert in alerts:
            st.markdown(
                f'<div class="alert-card">{alert}</div>',
                unsafe_allow_html=True
            )
    else:
        st.success("No critical alerts! Your usage looks normal.")

    st.divider()

    # ── High-usage days ───────────────────────────────────────────────────────
    st.subheader("High-Usage Days (Top 25%)")
    high = proc.detect_high_usage(75)
    if not high.empty:
        avg = daily_df["kwh"].mean()
        high["vs_avg_%"] = ((high["kwh"] - avg) / avg * 100).round(1)
        high["date"] = high["date"].dt.strftime("%a, %d %b %Y")
        st.dataframe(
            high.rename(columns={"kwh": "Daily kWh", "vs_avg_%": "vs Avg (%)"}),
            use_container_width=True, hide_index=True
        )

    st.divider()

    # ── Live simulation indicator ─────────────────────────────────────────────
    st.subheader("Simulated Real-Time Monitor")
    st.info("This simulates a live reading — values update each time you interact.")
    sim_c1, sim_c2, sim_c3 = st.columns(3)
    curr_hour = datetime.now().hour
    is_peak   = 17 <= curr_hour < 22
    avg_hour  = hourly_df[hourly_df["hour"] == curr_hour]["avg_kwh"].values
    current_draw = float(avg_hour[0]) if len(avg_hour) else 0.0
    with sim_c1:
        st.metric("Current Hour",     f"{curr_hour:02d}:00")
    with sim_c2:
        st.metric("Avg Draw (this hr)", f"{current_draw:.4f} kWh")
    with sim_c3:
        st.metric("Peak Status", "PEAK HOURS" if is_peak else "Off-peak")

    if is_peak:
        st.error("You are currently in peak hours (5 PM – 9 PM). "
                 "Consider deferring heavy loads like washing machine, water heater, iron.")
    else:
        st.success("Currently off-peak — ideal time to run high-wattage appliances.")


# ──────────────────────────────────────────────────────────────────────────────
# TAB 6 — REPORTS
# ──────────────────────────────────────────────────────────────────────────────
with tab_report:
    st.header("Export Reports")

    col_csv, col_pdf = st.columns(2)

    # ── CSV export ─────────────────────────────────────────────────────────────
    with col_csv:
        st.subheader("CSV Export")
        csv_choice = st.selectbox(
            "Which dataset?",
            ["Raw data", "Daily totals", "Appliance summary", "Forecast"]
        )
        if csv_choice == "Raw data":
            csv_data = export_csv(df_raw)
            fname = "energy_raw_data.csv"
        elif csv_choice == "Daily totals":
            csv_data = export_csv(daily_df)
            fname = "energy_daily.csv"
        elif csv_choice == "Appliance summary":
            csv_data = export_csv(appliance_df)
            fname = "energy_appliances.csv"
        else:
            future_dates_str = [d.strftime("%Y-%m-%d") for d in
                                 [pd.to_datetime(daily_df["date"].max()) + timedelta(days=i+1)
                                  for i in range(forecast_days)]]
            fc_df    = pd.DataFrame({"date": future_dates_str,
                                     "predicted_kwh": np.round(pred.predict(forecast_days), 3)})
            csv_data = export_csv(fc_df)
            fname = "energy_forecast.csv"

        st.download_button(
            label="⬇️ Download CSV",
            data=csv_data,
            file_name=fname,
            mime="text/csv",
            use_container_width=True,
        )

    # ── PDF export ─────────────────────────────────────────────────────────────
    with col_pdf:
        st.subheader("PDF Summary Report")
        st.markdown("Generates a formatted PDF containing KPIs, appliance table, tips and alerts.")
        if st.button("Generate PDF Report", use_container_width=True):
            with st.spinner("Building PDF…"):
                pdf_bytes = export_summary_pdf(
                    stats        = stats,
                    appliance_df = appliance_df,
                    suggestions  = optim.get_appliance_suggestions(),
                    alerts       = optim.get_alert_messages(),
                )
            st.download_button(
                label="Download PDF",
                data=pdf_bytes,
                file_name="energy_report.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
            st.success("PDF ready!")

    st.divider()

    # ── Shareable summary ─────────────────────────────────────────────────────
    st.subheader("On-Screen Summary")
    summary_md = f"""
| Metric | Value |
|---|---|
| Period | {daily_df['date'].min().strftime('%d %b %Y')} – {daily_df['date'].max().strftime('%d %b %Y')} |
| Total Consumption | **{stats['total_kwh']:,.2f} kWh** |
| Average Daily | **{stats['avg_daily_kwh']:.2f} kWh** |
| Peak Day | **{stats['max_daily_kwh']:.2f} kWh** |
| Top Appliance | **{stats['top_appliance']}** |
| Efficiency Score | **{optim.overall_score()} / 100** |
| Estimated Cost | **₹ {stats['cost_estimate']:,.0f}** |
| {forecast_days}-day Forecast | **{pred.predict(forecast_days).sum():.2f} kWh** |
"""
    st.markdown(summary_md)
    st.download_button(
        "⬇️ Download Summary (Markdown)",
        data=summary_md.encode(),
        file_name="energy_summary.md",
        mime="text/markdown",
    )
