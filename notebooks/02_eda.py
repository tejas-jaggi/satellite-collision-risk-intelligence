"""
============================================================
PHASE 2: EXPLORATORY DATA ANALYSIS
============================================================
Satellite Collision Risk Intelligence Platform
------------------------------------------------------------
What this script does:
  1. Loads satellite + debris data from processed CSVs
  2. Generates 6 key insight charts
  3. Prints business-level insights and statistics
  4. Saves all plots to outputs/plots/

Run: python notebooks/02_eda.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Dark space-themed style
plt.rcParams.update({
    "figure.facecolor":  "#0D1117",
    "axes.facecolor":    "#161B22",
    "axes.edgecolor":    "#30363D",
    "axes.labelcolor":   "#C9D1D9",
    "text.color":        "#C9D1D9",
    "xtick.color":       "#8B949E",
    "ytick.color":       "#8B949E",
    "grid.color":        "#21262D",
    "grid.linewidth":    0.8,
    "figure.dpi":        120,
    "font.family":       "DejaVu Sans",
})

RISK_COLORS = {
    "CRITICAL": "#FF4136",
    "HIGH":     "#FF851B",
    "MEDIUM":   "#FFDC00",
    "LOW":      "#2ECC40",
}

ACCENT   = "#58A6FF"
GREEN    = "#3FB950"
ORANGE   = "#F78166"
PURPLE   = "#BC8CFF"
YELLOW   = "#E3B341"


# ─────────────────────────────────────────────
# ORBITAL BAND CLASSIFICATION
# ─────────────────────────────────────────────

ORBITAL_BANDS = [
    ("LEO-Low",  160,    600,   "#FF4136"),
    ("LEO-Mid",  600,    1000,  "#FF851B"),
    ("LEO-High", 1000,   2000,  "#FFDC00"),
    ("MEO",      2000,   35786, "#58A6FF"),
    ("GEO",      35786,  36000, "#3FB950"),
    ("HEO",      36000,  200000,"#BC8CFF"),
]


def classify_orbital_band(altitude_km: float) -> str:
    for band, low, high, _ in ORBITAL_BANDS:
        if low <= altitude_km < high:
            return band
    return "Unknown"


def classify_risk_zone(altitude_km: float) -> str:
    if altitude_km < 600:
        return "CRITICAL"
    elif altitude_km < 1000:
        return "HIGH"
    elif altitude_km < 2000:
        return "MEDIUM"
    else:
        return "LOW"


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

def load_data():
    print("Loading data...")
    sat_df = pd.read_csv("data/processed/satellites_clean.csv")
    debris_df = pd.read_csv("data/processed/debris_clean.csv")

    # Filter to realistic altitude range
    sat_df = sat_df[
        (sat_df["altitude_km"] > 100) &
        (sat_df["altitude_km"] < 200000)
    ].copy()
    debris_df = debris_df[
        (debris_df["altitude_km"] > 100) &
        (debris_df["altitude_km"] < 200000)
    ].copy()

    # Add classifications
    sat_df["orbital_band"] = sat_df["altitude_km"].apply(classify_orbital_band)
    sat_df["risk_zone"]    = sat_df["altitude_km"].apply(classify_risk_zone)
    debris_df["orbital_band"] = debris_df["altitude_km"].apply(classify_orbital_band)

    print(f"✓ Loaded {len(sat_df):,} satellites, {len(debris_df):,} debris objects\n")
    return sat_df, debris_df


# ─────────────────────────────────────────────
# CHART 1: Orbital Altitude Histogram
# ─────────────────────────────────────────────

def chart_altitude_histogram(sat_df, debris_df):
    fig, ax = plt.subplots(figsize=(14, 6))

    # Cap at 50,000 km for readability (remove GEO outliers)
    sat_leo    = sat_df[sat_df["altitude_km"] < 50000]["altitude_km"]
    debris_leo = debris_df[debris_df["altitude_km"] < 50000]["altitude_km"]

    ax.hist(debris_leo, bins=120, color="#FF4136", alpha=0.5, label=f"Debris ({len(debris_df):,})", linewidth=0)
    ax.hist(sat_leo,    bins=120, color="#58A6FF", alpha=0.7, label=f"Active Satellites ({len(sat_df):,})", linewidth=0)

    # Shade orbital zones
    zone_info = [
        (160,   600,   "#FF4136", "LEO-Low\n(CRITICAL)"),
        (600,   1000,  "#FF851B", "LEO-Mid\n(HIGH)"),
        (1000,  2000,  "#FFDC00", "LEO-High\n(MEDIUM)"),
    ]
    for low, high, color, label in zone_info:
        ax.axvspan(low, high, alpha=0.08, color=color)
        ax.text((low + high) / 2, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 10,
                label, ha="center", fontsize=8, color=color, alpha=0.9)

    ax.set_xlabel("Orbital Altitude (km)", fontsize=12)
    ax.set_ylabel("Object Count", fontsize=12)
    ax.set_title("🛰️  Satellite & Debris Distribution by Orbital Altitude", fontsize=14, color="#F0F6FC", pad=15)
    ax.legend(loc="upper right", framealpha=0.3)
    ax.set_xlim(0, 50000)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/01_altitude_histogram.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Chart 1: Altitude histogram saved")


# ─────────────────────────────────────────────
# CHART 2: Top Countries by Satellite Count
# ─────────────────────────────────────────────

def chart_countries(sat_df):
    # Use country from data or extract from satellite name pattern
    # CelesTrak doesn't always provide country — we simulate from name patterns
    def infer_country(row):
        name = str(row.get("name", "")).upper()
        if "STARLINK" in name or "USA" in name or "GOES" in name or "INTELSAT" in name:
            return "USA"
        elif "COSMOS" in name or "RESURS" in name or "METEOR" in name:
            return "Russia"
        elif "TIANGONG" in name or "CZ-" in name or "FENGYUN" in name or "YAOGAN" in name:
            return "China"
        elif "ASTRA" in name or "SENTINEL" in name or "METEOSAT" in name:
            return "Europe"
        elif "IRNSS" in name or "CARTOSAT" in name or "INSAT" in name:
            return "India"
        elif "HIMAWARI" in name or "MICHIBIKI" in name or "DAICHI" in name:
            return "Japan"
        elif "ONEWEB" in name:
            return "UK"
        else:
            return sat_df["country"].iloc[0] if "country" in sat_df.columns else "Other"

    sat_df["inferred_country"] = sat_df.apply(infer_country, axis=1)
    country_counts = sat_df["inferred_country"].value_counts().head(15)

    fig, ax = plt.subplots(figsize=(12, 7))
    colors = [ACCENT if i == 0 else "#21262D" for i in range(len(country_counts))]
    bars = ax.barh(country_counts.index[::-1], country_counts.values[::-1], color=colors[::-1], edgecolor="#30363D")

    for bar, val in zip(bars, country_counts.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", color="#C9D1D9", fontsize=10)

    ax.set_xlabel("Number of Satellites", fontsize=12)
    ax.set_title("🌍  Satellites by Country of Operation (Top 15)", fontsize=14, color="#F0F6FC", pad=15)
    ax.set_xlim(0, country_counts.max() * 1.15)
    ax.grid(axis="x", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/02_satellites_by_country.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Chart 2: Country breakdown saved")
    return sat_df


# ─────────────────────────────────────────────
# CHART 3: Orbital Band Heatmap (Altitude × Inclination)
# ─────────────────────────────────────────────

def chart_orbital_heatmap(sat_df, debris_df):
    combined = pd.concat([
        sat_df[["altitude_km", "inclination"]].assign(type="Satellite"),
        debris_df[["altitude_km", "inclination"]].assign(type="Debris")
    ]).dropna()
    combined = combined[combined["altitude_km"] < 10000]

    alt_bins  = np.linspace(100, 10000, 50)
    incl_bins = np.linspace(0, 180, 37)

    heatmap, xedges, yedges = np.histogram2d(
        combined["altitude_km"], combined["inclination"],
        bins=[alt_bins, incl_bins]
    )

    fig, ax = plt.subplots(figsize=(14, 7))
    im = ax.imshow(
        heatmap.T, origin="lower", aspect="auto",
        extent=[alt_bins[0], alt_bins[-1], incl_bins[0], incl_bins[-1]],
        cmap="inferno", interpolation="gaussian"
    )
    plt.colorbar(im, ax=ax, label="Object Count")

    ax.set_xlabel("Orbital Altitude (km)", fontsize=12)
    ax.set_ylabel("Orbital Inclination (degrees)", fontsize=12)
    ax.set_title("🔥  Orbital Congestion Heatmap — Altitude × Inclination", fontsize=14, color="#F0F6FC", pad=15)

    # Mark key zones
    ax.axvline(600,   color="#FF851B", lw=1.5, alpha=0.8, linestyle="--", label="600 km (LEO-Low boundary)")
    ax.axvline(1000,  color="#FFDC00", lw=1.5, alpha=0.8, linestyle="--", label="1000 km (LEO-Mid boundary)")
    ax.axhline(97,    color="#58A6FF", lw=1.0, alpha=0.6, linestyle=":", label="~97° Sun-Sync inclination")
    ax.legend(loc="upper right", framealpha=0.4, fontsize=9)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/03_orbital_heatmap.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Chart 3: Orbital heatmap saved")


# ─────────────────────────────────────────────
# CHART 4: Debris Density by Altitude (KDE)
# ─────────────────────────────────────────────

def chart_debris_density(debris_df):
    fig, ax = plt.subplots(figsize=(14, 5))

    leo_debris = debris_df[debris_df["altitude_km"] < 3000]["altitude_km"].dropna()

    sns.kdeplot(leo_debris, ax=ax, color="#FF4136", linewidth=2.5, fill=True, alpha=0.3, label="Debris Density")

    ax.set_xlabel("Orbital Altitude (km)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("💥  Space Debris Density Distribution (LEO Focus: 0–3000 km)", fontsize=14, color="#F0F6FC", pad=15)
    ax.set_xlim(100, 3000)

    ax.axvspan(160, 600, alpha=0.1, color="#FF4136", label="CRITICAL Zone")
    ax.axvspan(600, 1000, alpha=0.1, color="#FF851B", label="HIGH Zone")
    ax.legend(framealpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/04_debris_density_kde.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Chart 4: Debris density KDE saved")


# ─────────────────────────────────────────────
# CHART 5: Inclination Distribution
# ─────────────────────────────────────────────

def chart_inclination(sat_df):
    fig, ax = plt.subplots(figsize=(12, 5))

    incl = sat_df["inclination"].dropna()
    ax.hist(incl, bins=90, color=PURPLE, alpha=0.8, edgecolor="#30363D", linewidth=0.3)

    # Mark key inclinations
    annotations = [
        (0,   "#58A6FF", "Equatorial\n(GEO)"),
        (28,  "#3FB950", "28.5°\n(Cape Canaveral)"),
        (51.6,"#FFDC00", "51.6°\n(ISS)"),
        (97,  "#FF851B", "~97°\n(Sun-Sync)"),
        (98,  "#FF851B", ""),
    ]
    for angle, color, label in annotations:
        ax.axvline(angle, color=color, lw=1.5, alpha=0.9, linestyle="--")
        if label:
            ax.text(angle + 1, ax.get_ylim()[1] * 0.85 if ax.get_ylim()[1] > 0 else 5,
                    label, color=color, fontsize=8)

    ax.set_xlabel("Orbital Inclination (degrees)", fontsize=12)
    ax.set_ylabel("Satellite Count", fontsize=12)
    ax.set_title("📐  Orbital Inclination Distribution", fontsize=14, color="#F0F6FC", pad=15)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/05_inclination_distribution.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Chart 5: Inclination distribution saved")


# ─────────────────────────────────────────────
# CHART 6: Satellites vs Debris by Orbital Band
# ─────────────────────────────────────────────

def chart_band_comparison(sat_df, debris_df):
    bands_order = ["LEO-Low", "LEO-Mid", "LEO-High", "MEO", "GEO", "HEO"]

    sat_counts    = sat_df["orbital_band"].value_counts().reindex(bands_order, fill_value=0)
    debris_counts = debris_df["orbital_band"].value_counts().reindex(bands_order, fill_value=0)

    x = np.arange(len(bands_order))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, sat_counts.values,    width, label="Active Satellites", color=ACCENT, alpha=0.85, edgecolor="#30363D")
    bars2 = ax.bar(x + width/2, debris_counts.values, width, label="Debris Objects",    color="#FF4136", alpha=0.75, edgecolor="#30363D")

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 50, f"{int(h):,}",
                        ha="center", va="bottom", fontsize=8, color="#C9D1D9")

    ax.set_xlabel("Orbital Band", fontsize=12)
    ax.set_ylabel("Object Count", fontsize=12)
    ax.set_title("🌐  Active Satellites vs Debris by Orbital Band", fontsize=14, color="#F0F6FC", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(bands_order)
    ax.legend(framealpha=0.3)
    ax.grid(axis="y", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/06_band_comparison.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Chart 6: Band comparison saved")


# ─────────────────────────────────────────────
# INSIGHT REPORT
# ─────────────────────────────────────────────

def print_insights(sat_df, debris_df):

    total_objects = len(sat_df) + len(debris_df)

    leo_sats = len(sat_df[sat_df["altitude_km"] < 1000])
    leo_debris = len(debris_df[debris_df["altitude_km"] < 1000])

    critical_sats = len(sat_df[sat_df["altitude_km"] < 600])

    geo_sats = len(
        sat_df[
            (sat_df["altitude_km"] >= 35500)
            & (sat_df["altitude_km"] <= 36000)
        ]
    )

    print("\n" + "=" * 60)
    print("  📊 EDA KEY INSIGHTS")
    print("=" * 60)

    print(f"  Total tracked objects:          {total_objects:>10,}")
    print(f"  Active satellites:              {len(sat_df):>10,}")
    print(f"  Space debris objects:           {len(debris_df):>10,}")

    # Satellite LEO percentage
    sat_pct = 100 * leo_sats / len(sat_df) if len(sat_df) > 0 else 0

    print(f"  Satellites below 1000km (LEO):  {leo_sats:>10,} ({sat_pct:.1f}%)")

    # Debris safe handling
    if len(debris_df) > 0:
        debris_pct = 100 * leo_debris / len(debris_df)
        print(f"  Debris below 1000km:            {leo_debris:>10,} ({debris_pct:.1f}%)")
    else:
        print("  Debris below 1000km:            0 (no debris data)")

    print(f"  Satellites in CRITICAL zone:    {critical_sats:>10,} (<600km)")
    print(f"  GEO satellites (35.5–36k km):   {geo_sats:>10,}")

    print("\n  Orbital Band Breakdown — Satellites:")

    band_counts = sat_df["orbital_band"].value_counts()

    for band, count in band_counts.items():

        pct = 100 * count / len(sat_df)

        bar = "█" * int(pct / 2)

        print(f"    {band:<12} {bar:<25} {count:>6,} ({pct:.1f}%)")

    print("\n  Top Insight for LinkedIn / Resume:")

    if total_objects > 0:
        leo_pct = 100 * (leo_sats + leo_debris) / total_objects
    else:
        leo_pct = 0

    print(f"  → {leo_pct:.0f}% of all tracked space objects cluster below 1,000 km")
    print(f"  → creating a critical congestion zone where collision risk is highest")

    print("=" * 60)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n🔬 " * 20)
    print("  PHASE 2: Exploratory Data Analysis")
    print("🔬 " * 20 + "\n")

    sat_df, debris_df = load_data()

    print("Generating charts...")
    chart_altitude_histogram(sat_df, debris_df)
    sat_df = chart_countries(sat_df)
    chart_orbital_heatmap(sat_df, debris_df)
    chart_debris_density(debris_df)
    chart_inclination(sat_df)
    chart_band_comparison(sat_df, debris_df)

    print_insights(sat_df, debris_df)

    # Save enriched satellite df for next phase
    sat_df.to_csv("data/processed/satellites_eda.csv", index=False)
    print(f"\n  ✓ Enriched satellite data saved to data/processed/satellites_eda.csv")
    print("\n✅ Phase 2 Complete! Charts saved to outputs/plots/")
    print("   Next step: Run 03_feature_engineering.py\n")


if __name__ == "__main__":
    main()
