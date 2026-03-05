"""
============================================================
PHASE 6: STREAMLIT DASHBOARD — Space Collision Intelligence
============================================================
Satellite Collision Risk Intelligence Platform
------------------------------------------------------------
Run: streamlit run app.py
Open: http://localhost:8501
============================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# ─────────────────────────────────────────────
# PAGE CONFIG — Must be first Streamlit command
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="🛰️ Space Collision Intelligence",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS — Dark Space Theme
# ─────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0D1117; }
    .main .block-container { padding-top: 1rem; max-width: 100%; }

    /* Sidebar */
    section[data-testid="stSidebar"] { background-color: #161B22; }
    section[data-testid="stSidebar"] * { color: #C9D1D9 !important; }

    /* KPI Cards */
    .kpi-card {
        background: linear-gradient(135deg, #161B22 0%, #21262D 100%);
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .kpi-value {
        font-size: 2.4rem;
        font-weight: 800;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #8B949E;
        margin-top: 4px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    /* Risk tier badges */
    .badge-critical { background: #FF4136; color: white; padding: 2px 8px;
                      border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .badge-high     { background: #FF851B; color: white; padding: 2px 8px;
                      border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .badge-medium   { background: #FFDC00; color: #0D1117; padding: 2px 8px;
                      border-radius: 4px; font-weight: bold; font-size: 0.8rem; }
    .badge-low      { background: #2ECC40; color: white; padding: 2px 8px;
                      border-radius: 4px; font-weight: bold; font-size: 0.8rem; }

    /* Section headers */
    .section-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #58A6FF;
        border-bottom: 2px solid #21262D;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    /* Dataframe */
    .dataframe { font-size: 0.85rem !important; }
    [data-testid="stDataFrame"] { border-radius: 8px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

RISK_COLORS = {
    "CRITICAL": "#FF4136",
    "HIGH":     "#FF851B",
    "MEDIUM":   "#FFDC00",
    "LOW":      "#2ECC40",
}

PLOTLY_TEMPLATE = dict(
    layout=go.Layout(
        paper_bgcolor="#0D1117",
        plot_bgcolor="#161B22",
        font=dict(color="#C9D1D9", family="Inter, Arial, sans-serif"),
        xaxis=dict(gridcolor="#21262D", linecolor="#30363D"),
        yaxis=dict(gridcolor="#21262D", linecolor="#30363D"),
        colorway=["#58A6FF", "#3FB950", "#FF851B", "#FF4136", "#BC8CFF", "#E3B341"],
        legend=dict(bgcolor="rgba(22,27,34,0.8)", bordercolor="#30363D"),
        margin=dict(l=40, r=20, t=50, b=40),
    )
)


# --------------------------------------------------
# DATA LOADING
# --------------------------------------------------

@st.cache_data(ttl=3600)
def load_data():
    """Load risk summary. Falls back to demo data if not yet generated."""

    primary_path = "dashboard/data/final_satellite_risk_scores.csv"

    if os.path.exists(primary_path):

        df = pd.read_csv(primary_path)

        # Normalize column names from pipeline
        rename_map = {
            "risk_score": "collision_risk_score",
            "risk_category": "risk_tier"
        }

        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})

        # Add rank column if missing
        if "rank" not in df.columns:
            df = df.sort_values("collision_risk_score", ascending=False)
            df["rank"] = range(1, len(df) + 1)

        return df, False

    # ── DEMO DATA — runs before pipeline is complete ──
    st.warning("⚠️  Pipeline not yet run. Showing DEMO DATA. Run the 5 Python scripts first.")
    np.random.seed(42)
    n = 1000

    altitudes = np.concatenate([
        np.random.normal(550,  80,  int(n * 0.35)),   # Starlink zone
        np.random.normal(780,  50,  int(n * 0.15)),   # Walker constellation zone
        np.random.normal(1200, 100, int(n * 0.10)),
        np.random.uniform(2000, 35500, int(n * 0.20)),
        np.random.normal(35786, 50, int(n * 0.20)),   # GEO
    ])[:n]
    altitudes = np.abs(altitudes) + 200

    risk_scores = np.clip(
        1 / (1 + altitudes / 500) * 0.7 + np.random.uniform(0, 0.4, n), 0, 1
    )

    def tier(s):
        if s >= 0.70: return "CRITICAL"
        elif s >= 0.50: return "HIGH"
        elif s >= 0.30: return "MEDIUM"
        return "LOW"

    demo_df = pd.DataFrame({
        "rank": range(1, n + 1),
        "sat_id": np.random.randint(10000, 99999, n),
        "name": [f"DEMO-SAT-{i:04d}" for i in range(n)],
        "collision_risk_score": risk_scores,
        "risk_tier": [tier(s) for s in risk_scores],
        "altitude_km": altitudes,
        "inclination": np.random.uniform(0, 120, n),
        "velocity_kms": np.sqrt(398600 / (6371 + altitudes)),
        "orbital_density": np.random.randint(10, 2000, n),
        "debris_density": np.random.randint(0, 500, n),
        "collision_probability": risk_scores * 0.9 + np.random.uniform(0, 0.1, n),
    }).sort_values("collision_risk_score", ascending=False).reset_index(drop=True)
    demo_df["rank"] = range(1, n + 1)

    return demo_df, True


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

def render_sidebar(df):
    st.sidebar.markdown("## 🛰️ Control Panel")
    st.sidebar.markdown("---")

    st.sidebar.markdown("### Risk Tier Filter")
    selected_tiers = st.sidebar.multiselect(
        "Show risk tiers:",
        options=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
        default=["CRITICAL", "HIGH", "MEDIUM", "LOW"],
    )

    st.sidebar.markdown("### Altitude Range (km)")
    alt_min = float(df["altitude_km"].min()) if "altitude_km" in df.columns else 100
    alt_max = float(df["altitude_km"].max()) if "altitude_km" in df.columns else 50000
    alt_range = st.sidebar.slider(
        "Altitude range:",
        min_value=int(alt_min),
        max_value=min(int(alt_max), 100000),
        value=(int(alt_min), min(int(alt_max), 40000)),
        step=100,
    )

    st.sidebar.markdown("### Risk Score Filter")
    min_score = st.sidebar.slider("Minimum risk score:", 0.0, 1.0, 0.0, 0.01)

    st.sidebar.markdown("### Top N Satellites")
    top_n = st.sidebar.slider("Show top N in table:", 10, 200, 50, 10)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.markdown("""
    **Data Source:** CelesTrak API  
    **Model:** XGBoost Classifier  
    **Updated:** Live from pipeline  
    
    [📁 GitHub Repo](#) | [📊 Full Report](#)
    """)

    return selected_tiers, alt_range, min_score, top_n


# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────

def filter_data(df, selected_tiers, alt_range, min_score):
    filtered = df.copy()

    if "risk_tier" in filtered.columns and selected_tiers:
        filtered = filtered[filtered["risk_tier"].isin(selected_tiers)]

    if "altitude_km" in filtered.columns:
        filtered = filtered[
            (filtered["altitude_km"] >= alt_range[0]) &
            (filtered["altitude_km"] <= alt_range[1])
        ]

    if "collision_risk_score" in filtered.columns:
        filtered = filtered[filtered["collision_risk_score"] >= min_score]

    return filtered


# ─────────────────────────────────────────────
# KPI CARDS
# ─────────────────────────────────────────────

def render_kpis(df, filtered_df):
    total_sats   = len(df)
    critical     = len(df[df["risk_tier"] == "CRITICAL"]) if "risk_tier" in df.columns else 0
    high_risk    = len(df[df["risk_tier"] == "HIGH"])     if "risk_tier" in df.columns else 0
    leo_count    = len(df[df["altitude_km"] < 1000])      if "altitude_km" in df.columns else 0
    avg_risk     = df["collision_risk_score"].mean()       if "collision_risk_score" in df.columns else 0
    showing      = len(filtered_df)

    cols = st.columns(6)
    kpi_data = [
        (f"{total_sats:,}",      "Total Satellites",      "#58A6FF"),
        (f"{leo_count:,}",       "In LEO (<1000km)",      "#E3B341"),
        (f"{critical:,}",        "CRITICAL Risk",         "#FF4136"),
        (f"{high_risk:,}",       "HIGH Risk",             "#FF851B"),
        (f"{avg_risk:.3f}",      "Avg Risk Score",        "#BC8CFF"),
        (f"{showing:,}",         "Satellites Shown",      "#3FB950"),
    ]

    for col, (value, label, color) in zip(cols, kpi_data):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-value" style="color: {color};">{value}</div>
                <div class="kpi-label">{label}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# CHART: Orbital Scatter Map
# ─────────────────────────────────────────────

def chart_orbital_scatter(filtered_df):
    st.markdown('<div class="section-header">🌐 Orbital Map — Altitude vs Inclination</div>',
                unsafe_allow_html=True)

    if "altitude_km" not in filtered_df.columns:
        st.warning("Altitude data not available")
        return

    plot_df = filtered_df.copy()
    plot_df["altitude_km"] = plot_df["altitude_km"].clip(upper=50000)

    fig = px.scatter(
        plot_df,
        x="altitude_km",
        y="inclination" if "inclination" in plot_df.columns else None,
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        size="collision_risk_score" if "collision_risk_score" in plot_df.columns else None,
        size_max=12,
        opacity=0.65,
        hover_name="name" if "name" in plot_df.columns else None,
        hover_data={
            "altitude_km": ":.0f",
            "collision_risk_score": ":.3f",
            "risk_tier": True,
        },
        labels={
            "altitude_km": "Orbital Altitude (km)",
            "inclination": "Inclination (degrees)",
            "risk_tier": "Risk Tier",
        },
        title="",
        template=None,
    )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=420,
        showlegend=True,
    )

    # Zone annotations
    for alt, label, color in [(600, "LEO-Low Boundary", "#FF4136"), (1000, "LEO-Mid Boundary", "#FF851B")]:
        fig.add_vline(x=alt, line_dash="dash", line_color=color, line_width=1.5, opacity=0.7,
                      annotation_text=f"{label} ({alt}km)", annotation_font_color=color,
                      annotation_font_size=10)

    st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------
# CHART: 3D Orbital Map
# --------------------------------------------------

def chart_orbital_3d(filtered_df):

    st.markdown(
        '<div class="section-header">🌍 3D Orbital Map</div>',
        unsafe_allow_html=True
    )

    if "altitude_km" not in filtered_df.columns:
        st.warning("Altitude data not available")
        return

    # Limit dataset for performance
    plot_df = filtered_df.sample(
        min(len(filtered_df), 5000),
        random_state=42
    )

    fig = px.scatter_3d(
        plot_df,
        x="altitude_km",
        y="inclination",
        z="collision_risk_score",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        hover_name="name",
        size="collision_risk_score",
	size_max=18,
        opacity=0.7,
    )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=600,
        scene=dict(
            xaxis_title="Altitude (km)",
            yaxis_title="Inclination (°)",
            zaxis_title="Risk Score"
        )
    )

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# CHART: Risk Distribution Pie
# ─────────────────────────────────────────────

def chart_risk_pie(filtered_df):
    st.markdown('<div class="section-header">📊 Risk Distribution</div>',
                unsafe_allow_html=True)

    if "risk_tier" not in filtered_df.columns:
        st.warning("Risk tier data not available")
        return

    tier_counts = filtered_df["risk_tier"].value_counts().reset_index()
    tier_counts.columns = ["Risk Tier", "Count"]
    tier_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    tier_counts["Risk Tier"] = pd.Categorical(tier_counts["Risk Tier"], categories=tier_order, ordered=True)
    tier_counts = tier_counts.sort_values("Risk Tier")

    fig = go.Figure(data=[go.Pie(
        labels=tier_counts["Risk Tier"],
        values=tier_counts["Count"],
        marker=dict(
            colors=[RISK_COLORS.get(t, "#58A6FF") for t in tier_counts["Risk Tier"]],
            line=dict(color="#0D1117", width=2)
        ),
        hole=0.45,
        textinfo="label+percent",
        textfont=dict(color="#C9D1D9", size=12),
        pull=[0.05 if t in ["CRITICAL", "HIGH"] else 0 for t in tier_counts["Risk Tier"]],
    )])

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=420,
        showlegend=True,
        annotations=[dict(text="Risk<br>Tiers", x=0.5, y=0.5, font_size=14,
                          font_color="#C9D1D9", showarrow=False)],
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# CHART: Altitude Congestion Histogram
# ─────────────────────────────────────────────

def chart_altitude_histogram(filtered_df):
    st.markdown('<div class="section-header">🔥 Orbital Congestion by Altitude</div>',
                unsafe_allow_html=True)

    if "altitude_km" not in filtered_df.columns:
        st.warning("Altitude data not available")
        return

    plot_df = filtered_df[filtered_df["altitude_km"] < 50000].copy()

    fig = px.histogram(
        plot_df,
        x="altitude_km",
        color="risk_tier",
        color_discrete_map=RISK_COLORS,
        nbins=100,
        opacity=0.8,
        labels={"altitude_km": "Orbital Altitude (km)", "count": "Satellite Count"},
        barmode="stack",
        template=None,
    )

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=350,
    )

    for alt, color in [(600, "#FF4136"), (1000, "#FF851B")]:
        fig.add_vline(x=alt, line_dash="dash", line_color=color, opacity=0.8,
                      annotation_text=f"{alt}km", annotation_font_color=color, annotation_font_size=9)

    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# CHART: Risk Score by Altitude Band
# ─────────────────────────────────────────────

def chart_band_risk(filtered_df):
    st.markdown('<div class="section-header">📡 Average Risk by Orbital Band</div>',
                unsafe_allow_html=True)

    if "altitude_km" not in filtered_df.columns or "collision_risk_score" not in filtered_df.columns:
        st.warning("Required data not available")
        return

    def get_band(alt):
        if alt < 600:   return "LEO-Low"
        elif alt < 1000: return "LEO-Mid"
        elif alt < 2000: return "LEO-High"
        elif alt < 35786: return "MEO"
        elif alt < 36000: return "GEO"
        return "HEO"

    df_band = filtered_df.copy()
    df_band["band"] = df_band["altitude_km"].apply(get_band)
    band_stats = df_band.groupby("band")["collision_risk_score"].agg(["mean", "count"]).reset_index()
    band_stats.columns = ["Band", "Avg Risk", "Count"]
    band_order = ["LEO-Low", "LEO-Mid", "LEO-High", "MEO", "GEO", "HEO"]
    band_stats = band_stats.set_index("Band").reindex(band_order).reset_index().dropna()

    fig = go.Figure(go.Bar(
        x=band_stats["Band"],
        y=band_stats["Avg Risk"],
        marker_color=[
            "#FF4136" if r >= 0.5 else "#FF851B" if r >= 0.35 else "#FFDC00" if r >= 0.25 else "#2ECC40"
            for r in band_stats["Avg Risk"]
        ],
        text=[f"{r:.3f}<br>({int(c)} sats)" for r, c in zip(band_stats["Avg Risk"], band_stats["Count"])],
        textposition="outside",
        textfont=dict(color="#C9D1D9", size=10),
    ))

    fig.update_layout(
        **PLOTLY_TEMPLATE["layout"].to_plotly_json(),
        height=350,
        xaxis_title="Orbital Band",
        yaxis_title="Avg Collision Risk Score",
        yaxis_range=[0, min(1.2, band_stats["Avg Risk"].max() * 1.3)],
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────
# RISK RANKING TABLE
# ─────────────────────────────────────────────

def render_risk_table(filtered_df, top_n):
    st.markdown('<div class="section-header">🚨 Satellite Risk Ranking</div>',
                unsafe_allow_html=True)

    display_cols = {
        "rank": "Rank",
        "name": "Satellite",
        "collision_risk_score": "Risk Score",
        "risk_tier": "Risk Tier",
        "altitude_km": "Altitude (km)",
        "inclination": "Inclination (°)",
        "velocity_kms": "Velocity (km/s)",
        "orbital_density": "Orbital Density",
        "collision_probability": "ML Probability",
    }

    available_cols = {k: v for k, v in display_cols.items() if k in filtered_df.columns}
    table_df = filtered_df.sort_values(
        "collision_risk_score", ascending=False
    ).head(top_n)[list(available_cols.keys())].copy()

    # Round numeric columns
    for col in ["collision_risk_score", "collision_probability", "velocity_kms"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(4)
    for col in ["altitude_km", "inclination"]:
        if col in table_df.columns:
            table_df[col] = table_df[col].round(1)

    table_df = table_df.rename(columns=available_cols)

    # Streamlit dataframe with column config
    col_config = {}
    if "Risk Score" in table_df.columns:
        col_config["Risk Score"] = st.column_config.ProgressColumn(
            "Risk Score", min_value=0, max_value=1, format="%.3f"
        )
    if "ML Probability" in table_df.columns:
        col_config["ML Probability"] = st.column_config.ProgressColumn(
            "ML Probability", min_value=0, max_value=1, format="%.3f"
        )

    st.dataframe(
        table_df,
        use_container_width=True,
        height=500,
        column_config=col_config,
        hide_index=True,
    )

    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="⬇️  Download Full Risk Report (CSV)",
        data=csv,
        file_name="satellite_risk_summary.csv",
        mime="text/csv",
    )

# --------------------------------------------------
# SATELLITE SEARCH
# --------------------------------------------------

def render_satellite_search(df):

    st.markdown(
        '<div class="section-header">🔎 Satellite Search</div>',
        unsafe_allow_html=True
    )

    search_term = st.text_input(
        "Enter satellite name (example: STARLINK, ISS, COSMOS)"
    )

    if search_term:

        results = df[df["name"].str.contains(search_term.upper(), na=False)]

        if len(results) == 0:
            st.warning("No satellites found.")
            return

        results = results.sort_values(
            "collision_risk_score",
            ascending=False
        ).head(20)

        st.dataframe(
            results[[
                "name",
                "collision_risk_score",
                "risk_tier",
                "altitude_km",
                "inclination",
                "velocity_kms",
                "collision_probability"
            ]],
            use_container_width=True
        )


# ─────────────────────────────────────────────
# MAIN DASHBOARD
# ─────────────────────────────────────────────

def main():
    # ── Header ──
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 16px; margin-bottom: 8px;">
        <div style="font-size: 2.5rem;">🛰️</div>
        <div>
            <h1 style="margin: 0; color: #F0F6FC; font-size: 1.8rem; font-weight: 800;">
                Satellite Collision Risk Intelligence Platform
            </h1>
            <p style="margin: 0; color: #8B949E; font-size: 0.9rem;">
                Real-time orbital risk monitoring · CelesTrak data · XGBoost model
            </p>
        </div>
    </div>
    <hr style="border-color: #21262D; margin-bottom: 1.5rem;">
    """, unsafe_allow_html=True)

    # ── Load Data ──
    with st.spinner("Loading satellite data..."):
        df, is_demo = load_data()

    if is_demo:
        st.info("📡 **Demo Mode** — Run the 5 pipeline scripts to load real CelesTrak data")

    # ── Sidebar ──
    selected_tiers, alt_range, min_score, top_n = render_sidebar(df)
    filtered_df = filter_data(df, selected_tiers, alt_range, min_score)

    # ── KPI Row ──
    render_kpis(df, filtered_df)

    # ── Main Charts Row ──
    col1, col2 = st.columns([2, 1])
    with col1:
        chart_orbital_scatter(filtered_df)
    with col2:
        chart_risk_pie(filtered_df)

    # 🌍 3D Orbital Map
    chart_orbital_3d(filtered_df)	

    # ── Second Charts Row ──
    col3, col4 = st.columns([1, 1])
    with col3:
        chart_altitude_histogram(filtered_df)
    with col4:
        chart_band_risk(filtered_df)

    # ── Risk Table ──
    render_risk_table(filtered_df, top_n)

    # 🔎 Satellite Search
    render_satellite_search(df)

    # ── Footer ──
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6E7681; font-size: 0.8rem; padding: 8px;">
        🛰️ Satellite Collision Risk Intelligence Platform &nbsp;|&nbsp;
        Data: CelesTrak (celestrak.org) &nbsp;|&nbsp;
        Model: XGBoost Classifier &nbsp;|&nbsp;
        Built with Python · Pandas · Scikit-learn · Streamlit
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
