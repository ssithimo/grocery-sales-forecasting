import streamlit as st
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings("ignore")

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stater Bros | Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand palette ─────────────────────────────────────────────────────────────
RED    = "#C8102E"
DARK   = "#1A1A2E"
MID    = "#16213E"
CARD   = "#0F3460"
ACCENT = "#E94560"
GOLD   = "#F5A623"
GREEN  = "#27AE60"
LIGHT  = "#ECF0F1"
MUTED  = "#95A5A6"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* import oranienbaum font from google fonts for headers */
/* import inter font for everything else, with various weights for flexibility */

    @import url('https://fonts.googleapis.com/css2?family=Oranienbaum&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');

    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}
    .dashboard-title {{
        font-family: 'Oranienbaum', serif !important;
    }}

/* Global colors and backgrounds */
            
  .stApp {{ background: {DARK}; color: {LIGHT}; }}

  /* Sidebar styling */

  section[data-testid="stSidebar"] {{ background: {MID}; border-right: 1px solid #2A2A4A; }}
  section[data-testid="stSidebar"] * {{ color: {LIGHT} !important; }}

  /* Metric cards */

  .metric-card {{
    background: {CARD}; border-radius: 12px; padding: 20px 24px;
    border-left: 4px solid {RED}; margin-bottom: 8px;
  }}
  .metric-card.green {{ border-left-color: {GREEN}; }}
  .metric-card.gold  {{ border-left-color: {GOLD};  }}
  .metric-card.accent{{ border-left-color: {ACCENT};}}
  .metric-label {{ font-size: 0.75rem; font-weight: 600; letter-spacing: 0.08em;
                   text-transform: uppercase; color: {MUTED}; margin-bottom: 4px; }}
  .metric-value {{ font-size: 1.9rem; font-weight: 800; color: {LIGHT}; line-height: 1; }}
  .metric-sub   {{ font-size: 0.8rem; color: {MUTED}; margin-top: 4px; }}

  /* Section headers */

  .section-header {{
    font-size: 1.2rem; font-weight: 700; color: {LIGHT};
    border-bottom: 2px solid {RED}; padding-bottom: 6px; margin: 24px 0 16px;
  }}

  /* Insight boxes */

  .insight-box {{
    background: {MID}; border: 1px solid #2A2A4A; border-radius: 10px;
    padding: 16px 20px; margin: 8px 0; font-size: 0.88rem; color: {LIGHT};
  }}
  .insight-box strong {{ color: {GOLD}; }}
  div[data-testid="stTabs"] button {{ color: {LIGHT} !important; }}
  div[data-testid="stTabs"] button[aria-selected="true"] {{
    color: {RED} !important; border-bottom: 2px solid {RED} !important;
  }}

  /* Form elements */

  .stSlider label {{ color: {LIGHT} !important; }}
  .stSelectbox label {{ color: {LIGHT} !important; }}
  .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}
</style>
""", unsafe_allow_html=True)

# ── Load & prep data ──────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/staterbros.csv")
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.sort_values("Date").reset_index(drop=True)
    df["Month"]     = df["Date"].dt.month
    df["Year"]      = df["Date"].dt.year
    df["MonthName"] = df["Date"].dt.strftime("%b")
    df["SalesM"]    = df["Sales"] / 1_000_000
    df["YearMonth"] = df["Date"].dt.to_period("M")
    return df

df = load_data()

# ── Helper: plot theme ────────────────────────────────────────────────────────
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color=LIGHT, size=12),
    xaxis=dict(showgrid=False, color=MUTED, linecolor="#2A2A4A"),
    yaxis=dict(showgrid=True, gridcolor="#2A2A4A", color=MUTED, linecolor="#2A2A4A"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=LIGHT)),
)

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
# max 6 month forecast horizon, beyond that is unreliable; default to 3
# toggle for confidence interval display, default on
# store count input for per-store calculations, default to 169 (current store count)

with st.sidebar:
    st.markdown(f"""
                <div 
                class = "dashboard-title" 
                style = 'font-size:1.8rem;font-weight:800;color:{RED};margin-bottom:4px;'>
                🛒 Stater Bros
                </div>
                """, unsafe_allow_html = True)
    st.markdown(f"""
                <div 
                style='font-size:0.8rem;color:{MUTED};margin-bottom:24px;'>
                Sales Forecasting Dashboard
                </div>
                """, unsafe_allow_html = True)

    st.markdown("### Forecast Settings")
    horizon = st.slider(
        label = "Forecast horizon (months)",
        min_value = 1,
        max_value = 6,
        value = 3,
        help = "How many months ahead to forecast"
    )
    show_ci = st.toggle("Show confidence interval", value=True)

    stores = st.number_input("Store count", min_value = 1, max_value = 500, value = 169, step = 1, help = "Used for per-store calculations")

    st.markdown("---")
    st.markdown(f"<div style='font-size:0.7rem;color:{MUTED};'>Data: Jan 2022 – Nov 2025 | 47 months<br>Model: Holt-Winters Triple Exponential Smoothing</div>", unsafe_allow_html=True)

# ── FIT MODEL ─────────────────────────────────────────────────────────────────

# create a model fitting fct that takes the sales series, horizon, and store count as inputs and 
# returns fitted values, forecast, confidence intervals, and error metrics. Use @st.cache_data to cache 
# results for faster performance on repeated runs with the same parameters.
# our model was optimized with MNM in Rstudio, so we will do mulitplicative error, no trend, and multiplicative seasonality
# python doesn't have mulitplicative error

@st.cache_data
def fit_hw(sales_series, horizon, stores):
    model = ExponentialSmoothing(
        sales_series,
        trend = None,
        seasonal = "multiplicative",
        seasonal_periods = 12,
        initialization_method = "estimated",
    ).fit(optimized=True)

    fitted   = model.fittedvalues
    forecast = model.forecast(horizon)

    # prediction confidence interval via simulation
    from statsmodels.tsa.holtwinters import ExponentialSmoothing as ES
    sim = model.simulate(horizon, repetitions=500, random_errors="bootstrap")
    ci_lower = sim.quantile(0.10, axis=1)
    ci_upper = sim.quantile(0.90, axis=1)

    return fitted, forecast, ci_lower, ci_upper

# Prepare the sales series for modeling
_s = df.set_index("Date")["SalesM"].copy()
_s.index = pd.DatetimeIndex(_s.index).to_period("M").to_timestamp()
_s.index.freq = "MS"
sales_series = _s

fitted_vals, forecast_vals, ci_lo, ci_hi = fit_hw(sales_series, horizon, stores)

mape = 20.39

# forecast dates needed for plotting and tables
# gen future dates based on last date + 1 month, with monthly frequency, for the number of periods in the horizon
# freq "ME" means month end frequency, so it will generate dates like 2025-12-31, 2026-01-31, etc. which is appropriate for monthly sales data
last_date = df["Date"].max()
fc_dates = pd.date_range(
    start = last_date + pd.DateOffset(months = 1), 
    periods = horizon, 
    freq = "ME")

dec_forecast = None
for d, v in zip(fc_dates, forecast_vals):
    if d.month == 12:
        dec_forecast = v
        break

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown(f"""
            <h1 class = "dashboard-title" 
            style = 'font-size:3rem;font-weight:800;color:{LIGHT};margin-bottom:0;'>
            Stater Bros Sales Forecasting
            </h1>
            """, unsafe_allow_html=True)
st.markdown(f"""
            <p style = 'color:{MUTED};margin-top:4px;margin-bottom:24px;'>
            Holt-Winters Triple Exponential Smoothing · {len(df)} months training · {horizon}-month horizon
            </p>
            """, unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📈 Forecast", "📊 Historical Analysis", "🔍 Model Summary"])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 – FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab1:

    # === KPI row ===
    c1, c2, c3, c4 = st.columns(4)
    next_month_fc   = forecast_vals.iloc[0]
    next_month_name = fc_dates[0].strftime("%b %Y")
    dec_str = f"${dec_forecast:,.1f}M"
    dec_per = f"${(dec_forecast / stores):,.2f}M"
    fc_avg = forecast_vals.mean()
    fc_total = forecast_vals.sum()

    # decide display format since we could be in millions or billions depending on the horizon
    # fc_total is already in millions so has to be >= 1k to be in billions
    # divide by 1k to show in B otherwise leave as is and show with M

    if fc_total >= 1_000:
        fc_display = fc_total / 1_000
        fc_label = f"${fc_display:,.2f}B"
    else:
        fc_display = fc_total
        fc_label = f"${fc_display:,.0f}M"
    
    #metrics list to loop through (label, value, subtext, color)
    metrics_tab1 = [
        ("Dec Forecast", dec_per, "Per Store","red"),
        ("Dec Forecast", dec_str, "Chain wide", "green"),
        (f"{horizon}-Month Average", f"${fc_avg:,.0f}M", "Monthly avg forecast", "gold"),
        (f"{horizon}-Month Total", fc_label, "Cumulative forecast", "accent")
    ]
    cols = [c1, c2, c3, c4]

    for col, (label, val, sub, color) in zip(cols, metrics_tab1):
        
         with col:
            st.markdown(f"""
                <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html = True)

    # === Main forecast chart ===
    st.markdown(f"<div class='section-header'>Sales Forecast — {horizon}-Month Horizon</div>", unsafe_allow_html=True)

    fig = go.Figure()

    # Historical
    fig.add_trace(go.Scatter(
        x=df["Date"], y=df["SalesM"],
        mode="lines+markers",
        name="Actual Sales",
        line=dict(color=LIGHT, width=2),
        marker=dict(size=4),
    ))

    # Fitted
    fig.add_trace(go.Scatter(
        x=df["Date"], y=fitted_vals.values,
        mode="lines",
        name="Model Fit",
        line=dict(color=GOLD, width=1.5, dash="dot"),
    ))

    # CI
    if show_ci:
        fig.add_trace(go.Scatter(
            x=list(fc_dates) + list(fc_dates[::-1]),
            y=list(ci_hi) + list(ci_lo[::-1]),
            fill = "toself",
            mode = "none",
            fillcolor = "rgba(211, 175, 55, 0.5)",
            name = "80% CI",
            showlegend = True,
        ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=fc_dates, y=forecast_vals,
        mode="lines+markers",
        name="Forecast",
        line=dict(color = GOLD, width = 1.5),
        marker=dict(size = 5, color = GOLD),
    ))

    # Divider
    fig.add_vline(x=last_date, line_dash="dash", line_color=MUTED, line_width=1.5)
    fig.add_annotation(x=last_date, y=df["SalesM"].max()*1.05,
                       text="Forecast start", font=dict(color=MUTED, size=10),
                       showarrow=False, xshift=40)

    fig.update_layout(**PLOT_LAYOUT, height=400,
                      yaxis_title="Sales ($M)",
                      hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

    # ===Forecast Detail Table===
    st.markdown(f"<div class='section-header'>Forecast Detail</div>", unsafe_allow_html=True)
    fc_df = pd.DataFrame({
        "Month":         [d.strftime("%b %Y") for d in fc_dates],
        "Forecast ($M)": forecast_vals.round(1).values,
        "Low ($M)":      ci_lo.round(1).values if show_ci else ["—"]*horizon,
        "High ($M)":     ci_hi.round(1).values if show_ci else ["—"]*horizon,
        "Per Store ($)": [(v * 1e6 / stores) for v in forecast_vals.round(0).values],
    })
    fc_df["Per Store ($)"] = fc_df["Per Store ($)"].apply(lambda x: f"${x:,.0f}")
    st.dataframe(
        fc_df.style
            .format({"Forecast ($M)": "{:.1f}", "Low ($M)": "{:.1f}", "High ($M)": "{:.1f}"})
            .background_gradient(subset=["Forecast ($M)"], cmap="Greens"),
        use_container_width=True,
        hide_index=True,
    )

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 – HISTORICAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:

    # === KPIs ===
    c1, c2, c3, c4 = st.columns(4)
    avg_monthly  = df["SalesM"].mean()
    peak_row     = df.loc[df["Sales"].idxmax()]
    trough_row   = df.loc[df["Sales"].idxmin()]
    total_sales  = df["SalesM"].sum() / 1_000 #since SalesM is in millions, dividing by 1,000 gives us billions for the cumulative sales figure
    
    #metrics list to loop through (label, value, subtext, color)
    metrics_tab2 = [
            ("Avg Monthly Sales", f"${avg_monthly:,.0f}M", f"${(avg_monthly*1e6/stores):,.0f} per store", "red"),
            ("Peak Month", f"${peak_row['SalesM']:,.0f}M", peak_row['Date'].strftime("%b %Y"), "green"),
            ("Lowest Month", f"${trough_row['SalesM']:,.0f}M", trough_row['Date'].strftime("%b %Y"), "gold"),
            ("Cumulative Sales", f"${total_sales:,.0f}B", f"{len(df)} months", "accent")
    ]

    cols = [c1, c2, c3, c4]

    for col, (label, val, sub, color) in zip(cols, metrics_tab2):
        
         with col:
            st.markdown(f"""
                <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # === Monthly Sales by Year and Average Sales by Month ===

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"<div class='section-header'>Monthly Sales by Year</div>", unsafe_allow_html=True)
        year_colors = {2022: LIGHT, 2023: GOLD, 2024: GREEN, 2025: RED}
        fig2 = go.Figure()
        for yr in sorted(df["Year"].unique()):
            yd = df[df["Year"] == yr]
            fig2.add_trace(go.Scatter(
                x=yd["Month"], y=yd["SalesM"],
                mode="lines+markers",
                name=str(yr),
                line=dict(color=year_colors.get(yr, ACCENT), width=2),
                marker=dict(size=5),
            ))
        fig2.update_layout(**PLOT_LAYOUT, height = 320, yaxis_title = "Sales ($M)")
        fig2.update_xaxes(tickmode = "array", tickvals = list(range(1,13)), 
                          ticktext = MONTH_LABELS)

        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown(f"<div class='section-header'>Average Sales by Month (Seasonality)</div>", unsafe_allow_html=True)
        seasonal = df.groupby("Month")["SalesM"].mean().reset_index()
        seasonal["MonthName"] = pd.to_datetime(seasonal["Month"], format="%m").dt.strftime("%b")
        fig3 = go.Figure(go.Bar(
            x=seasonal["MonthName"], y=seasonal["SalesM"],
            marker_color=[RED if v == seasonal["SalesM"].max() else CARD for v in seasonal["SalesM"]],
            text=seasonal["SalesM"].round(0).astype(int),
            texttemplate="$%{text}M",
            textposition="outside",
            textfont=dict(color=LIGHT, size=10),
        ))
        fig3.update_layout(**PLOT_LAYOUT, height=320, yaxis_title="Avg Sales ($M)",
                           showlegend=False)
        st.plotly_chart(fig3, use_container_width=True)

    # === Year-over-Year Annual Sales ===

    st.markdown(f"<div class='section-header'>Year-over-Year Annual Sales</div>", unsafe_allow_html=True)

    annual = df.groupby("Year")["SalesM"].sum().reset_index()
    annual["YoY_pct"] = annual["SalesM"].pct_change() * 100

    fig4 = make_subplots(specs=[[{"secondary_y": True}]])
    fig4.add_trace(go.Bar(
        x=annual["Year"].astype(str), y=annual["SalesM"],
        name="Total Sales ($M)", marker_color=RED,
        text=annual["SalesM"].round(0).astype(int),
        texttemplate="$%{text}M", textposition="outside",
        textfont=dict(color=LIGHT, size=10),
    ), secondary_y=False)
    fig4.update_layout(**PLOT_LAYOUT, height=320, showlegend=True)
    fig4.update_yaxes(title_text="Annual Sales ($M)", secondary_y=False,
                      showgrid=True, gridcolor="#2A2A4A", color=MUTED)
    st.plotly_chart(fig4, use_container_width=True)

    # === Key insights ===

    st.markdown(f"<div class='section-header'>Key Observations</div>", unsafe_allow_html=True)

    peak_month_name = pd.to_datetime(str(seasonal.loc[seasonal["SalesM"].idxmax(), "Month"]), format="%m").strftime("%B")
    low_month_name  = pd.to_datetime(str(seasonal.loc[seasonal["SalesM"].idxmin(), "Month"]), format="%m").strftime("%B")

    insights = [
        f"<strong>Seasonal Peak:</strong> {peak_month_name} consistently drives the highest average monthly sales — likely driven by holiday shopping and end-of-year grocery demand.",
        f"<strong>Seasonal Trough:</strong> {low_month_name} is the weakest month on average, suggesting post-holiday pullback in consumer spending.",
        f"<strong>High Volatility:</strong> Monthly sales range from ${trough_row['SalesM']:.0f}M to ${peak_row['SalesM']:.0f}M — a {(peak_row['Sales']/trough_row['Sales']-1)*100:.0f}% swing, underscoring the importance of seasonal inventory planning.",
        f"<strong>2025 Partial Year:</strong> 2025 data runs through November only; full-year totals will exceed prior years once December is recorded.",
    ]

    for ins in insights:
        st.markdown(f"<div class='insight-box'>{ins}</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 – MODEL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    # === KPIs ===

    st.markdown(f"<div class='section-header'>Model Metrics</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    mape = 20.39
    rmse_val = 127_910_148/1_000_000
    mae = 99_277_645/1_000_000

    #metrics list to loop through (label, value, subtext, color)
    metrics_tab3 = [
            ("MAPE (out-of-sample)", f"{mape:.2f}%", "Mean Absolute Percentage Error", "red"),
            ("RMSE", f"${rmse_val:.1f}M", "Root Mean Squared Error", "green"),
            ("MAE", f"${mae:.1f}M", "Mean Absolute Error", "gold")
            ]
    
    cols = [c1, c2, c3]

    for col, (label, val, sub, color) in zip(cols, metrics_tab3):

        with col:
            st.markdown(f"""
                <div class="metric-card {color}">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-sub">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    # === Model performance charts ===
    st.markdown(f"<div class='section-header'>Model Performance</div>", unsafe_allow_html=True)

    chart1, chart2 = st.columns(2)

    with chart1:
        # Actual vs Fitted
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=df["Date"], y=df["SalesM"],
            mode="lines", name="Actual", line=dict(color=LIGHT, width=2),
        ))
        fig5.add_trace(go.Scatter(
            x=df["Date"], y=fitted_vals.values,
            mode="lines", name="Fitted", line=dict(color=RED, width=2, dash="dot"),
        ))
        fig5.update_layout(**PLOT_LAYOUT, height=280, yaxis_title="Sales ($M)",
                           title=dict(text="Actual vs Fitted", font=dict(color=LIGHT, size=13)))
        st.plotly_chart(fig5, use_container_width=True)

    with chart2:
        # Residuals
        residuals = df["SalesM"].values - fitted_vals.values
        fig6 = go.Figure(go.Bar(
            x=df["Date"], y=residuals,
            marker_color=[GREEN if v >= 0 else RED for v in residuals],
            name="Residual",
        ))
        fig6.add_hline(y=0, line_dash="dash", line_color=MUTED)
        fig6.update_layout(**PLOT_LAYOUT, height=230, yaxis_title="Residual ($M)",
                           title=dict(text="Residuals", font=dict(color=LIGHT, size=13)))
        st.plotly_chart(fig6, use_container_width=True)

    # === Methodology summary ===

    st.markdown(f"<div class='section-header'>Methodology</div>", unsafe_allow_html=True)
    
    methodology = [
        "<strong>Model:</strong> Holt-Winters Triple Exponential Smoothing (no trend + multiplicative seasonality)",
        "<strong>Seasonal Period:</strong> 12 months — captures annual grocery seasonality (holidays, summer, etc.)",
        "<strong>Parameters:</strong> Optimized via MLE — α (level), β (trend), γ (seasonal) all estimated from data",
        "<strong>Confidence Interval:</strong> 80% PI via 500-iteration bootstrap simulation of forecast residuals",
        "<strong>Training Data:</strong> Jan 2022 – Nov 2025 (47 months; adjustable via sidebar date filter)",
        "<strong>Per-Store Calc:</strong> Assumes 169 stores (adjustable in sidebar) — evenly distributed allocation",
    ]
    
    # 2 columns × 3 rows
    for i in range(0, len(methodology), 2):
        c1, c2 = st.columns(2)

        with c1:
            st.markdown(f"<div class='insight-box'>{methodology[i]}</div>", unsafe_allow_html=True)

        # only render second column if it exists
        if i + 1 < len(methodology):
            with c2:
                st.markdown(f"<div class='insight-box'>{methodology[i+1]}</div>", unsafe_allow_html=True)
