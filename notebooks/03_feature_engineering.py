"""
============================================================
PHASE 3: ORBITAL FEATURE ENGINEERING
============================================================
Satellite Collision Risk Intelligence Platform
------------------------------------------------------------
What this script does:
  1. Loads cleaned satellite + debris data
  2. Engineers 10+ collision-risk features
  3. Validates feature quality (nulls, distributions)
  4. Exports model-ready dataframe

Key features engineered:
  - orbital_density    (objects in same altitude band)
  - debris_density     (debris in proximity window)
  - proximity_score    (normalized closeness to debris)
  - velocity_risk      (relative to safe threshold)
  - altitude_band      (categorical zone classification)

Run: python notebooks/03_feature_engineering.py
============================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import MinMaxScaler

PLOTS_DIR = "outputs/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "#0D1117",
    "axes.facecolor": "#161B22",
    "axes.edgecolor": "#30363D",
    "axes.labelcolor": "#C9D1D9",
    "text.color": "#C9D1D9",
    "xtick.color": "#8B949E",
    "ytick.color": "#8B949E",
    "grid.color": "#21262D",
})


# ─────────────────────────────────────────────
# LOAD DATA
# ─────────────────────────────────────────────

def load_data():
    sat_path = "data/processed/satellites_eda.csv"
    if not os.path.exists(sat_path):
        sat_path = "data/processed/satellites_clean.csv"

    sat_df    = pd.read_csv(sat_path)
    debris_df = pd.read_csv("data/processed/debris_clean.csv")

    # Clean filtering
    sat_df = sat_df[
        (sat_df["altitude_km"] > 100) &
        (sat_df["altitude_km"] < 200000) &
        (sat_df["inclination"].notna()) &
        (sat_df["velocity_kms"].notna())
    ].copy().reset_index(drop=True)

    debris_df = debris_df[
        (debris_df["altitude_km"] > 100) &
        (debris_df["altitude_km"] < 200000)
    ].copy().reset_index(drop=True)

    print(f"✓ Loaded {len(sat_df):,} satellites, {len(debris_df):,} debris")
    return sat_df, debris_df


# ─────────────────────────────────────────────
# FEATURE 1: Orbital Band (Categorical)
# ─────────────────────────────────────────────

def feature_orbital_band(df):
    """Classify each satellite into orbital band."""
    bands = [
        (160,   600,   "LEO-Low"),
        (600,   1000,  "LEO-Mid"),
        (1000,  2000,  "LEO-High"),
        (2000,  35786, "MEO"),
        (35786, 36000, "GEO"),
        (36000, 999999,"HEO"),
    ]

    def classify(alt):
        for low, high, name in bands:
            if low <= alt < high:
                return name
        return "Unknown"

    df["altitude_band"] = df["altitude_km"].apply(classify)
    df["is_leo"] = (df["altitude_km"] < 2000).astype(int)
    print("  ✓ Feature: altitude_band, is_leo")
    return df


# ─────────────────────────────────────────────
# FEATURE 2: High Inclination Flag
# ─────────────────────────────────────────────

def feature_inclination_flags(df):
    """
    Polar orbits (inclination 80-110°) cross all other orbital planes
    repeatedly, creating many potential conjunction events.
    """
    df["high_inclination"] = (
        (df["inclination"] > 80) & (df["inclination"] < 110)
    ).astype(int)

    df["sun_sync_orbit"] = (
        (df["inclination"] > 95) & (df["inclination"] < 100)
    ).astype(int)

    print("  ✓ Feature: high_inclination, sun_sync_orbit")
    return df


# ─────────────────────────────────────────────
# FEATURE 3: Orbital Density (Optimized)
# ─────────────────────────────────────────────

def feature_orbital_density(sat_df, debris_df, window_km=50):
    """
    Count of satellites in same altitude window (±window_km).
    Uses pandas cut() binning for speed — much faster than row-by-row.
    """
    print(f"  Computing orbital density (±{window_km}km window)...")

    # Create altitude bins
    alt_min = max(100, sat_df["altitude_km"].min() - window_km)
    alt_max = sat_df["altitude_km"].max() + window_km
    bins = np.arange(alt_min, alt_max, window_km)

    # Bin satellites
    sat_df["alt_bin"] = pd.cut(sat_df["altitude_km"], bins=bins, labels=False)
    debris_df["alt_bin"] = pd.cut(debris_df["altitude_km"], bins=bins, labels=False)

    # Count objects per bin
    sat_bin_counts    = sat_df.groupby("alt_bin").size().rename("orbital_density")
    debris_bin_counts = debris_df.groupby("alt_bin").size().rename("debris_density")

    sat_df = sat_df.join(sat_bin_counts, on="alt_bin")
    sat_df = sat_df.join(debris_bin_counts, on="alt_bin")
    sat_df["orbital_density"] = sat_df["orbital_density"].fillna(0).astype(int)
    sat_df["debris_density"]  = sat_df["debris_density"].fillna(0).astype(int)

    # Adjust: subtract self from satellite count
    sat_df["orbital_density"] = (sat_df["orbital_density"] - 1).clip(lower=0)

    print(f"  ✓ Feature: orbital_density (max={sat_df['orbital_density'].max():,})")
    print(f"  ✓ Feature: debris_density (max={sat_df['debris_density'].max():,})")
    return sat_df


# ─────────────────────────────────────────────
# FEATURE 4: Proximity Score
# ─────────────────────────────────────────────

def feature_proximity_score(sat_df, debris_df):
    """
    For each satellite, find the altitude of the nearest debris cluster centroid.
    Proximity score = 1 / (1 + distance_to_nearest_debris_km)
    Normalized to 0–1 range.
    """
    print("  Computing proximity scores...")

    # Get debris altitude percentiles as cluster centroids
    debris_alts = debris_df["altitude_km"].dropna().values
    debris_alts = np.sort(debris_alts)

    # For each satellite altitude, find minimum distance to any debris
    sat_alts = sat_df["altitude_km"].values

    # Vectorized: use searchsorted for speed
    def min_debris_distance(alt):
        idx = np.searchsorted(debris_alts, alt)
        candidates = []
        if idx > 0:
            candidates.append(abs(alt - debris_alts[idx - 1]))
        if idx < len(debris_alts):
            candidates.append(abs(alt - debris_alts[idx]))
        return min(candidates) if candidates else 999999

    distances = np.array([min_debris_distance(a) for a in sat_alts])
    sat_df["min_debris_distance_km"] = distances
    sat_df["proximity_score"] = 1 / (1 + distances / 100)   # Scale: 100km = score 0.5

    print(f"  ✓ Feature: proximity_score (mean={sat_df['proximity_score'].mean():.3f})")
    return sat_df


# ─────────────────────────────────────────────
# FEATURE 5: Velocity Risk
# ─────────────────────────────────────────────

def feature_velocity_risk(df):
    """
    Higher orbital velocity = more destructive collisions.
    Normalize to 0–1 based on LEO velocity range (7.5–8.0 km/s).
    """
    v_min = 1.0     # km/s (very low — HEO apogee)
    v_max = 8.0     # km/s (LEO max)

    df["velocity_risk"] = (df["velocity_kms"] - v_min) / (v_max - v_min)
    df["velocity_risk"] = df["velocity_risk"].clip(0, 1)

    print(f"  ✓ Feature: velocity_risk (mean={df['velocity_risk'].mean():.3f})")
    return df


# ─────────────────────────────────────────────
# FEATURE 6: Eccentricity Risk
# ─────────────────────────────────────────────

def feature_eccentricity_risk(df):
    """
    High eccentricity = elliptical orbit = crosses multiple altitude shells.
    More potential conjunction events at different altitudes.
    """
    df["eccentricity_risk"] = df["eccentricity"].clip(0, 1)
    df["high_eccentricity"]  = (df["eccentricity"] > 0.1).astype(int)

    print(f"  ✓ Feature: eccentricity_risk, high_eccentricity")
    return df


# ─────────────────────────────────────────────
# FEATURE 7: Altitude Risk Score
# ─────────────────────────────────────────────

def feature_altitude_risk(df):
    """
    Objects below 600km are in the highest debris density zone.
    Altitude risk is non-linear — peaks in 400-600km range.
    """
    def alt_risk(alt):
        if 300 <= alt <= 600:
            return 1.0                         # Maximum risk zone
        elif 600 < alt <= 1000:
            return 0.8 - 0.3 * (alt - 600) / 400
        elif 1000 < alt <= 2000:
            return 0.5 - 0.2 * (alt - 1000) / 1000
        elif alt < 300:
            return 0.6                         # Decays quickly, but still risky
        else:
            return max(0.1, 0.3 - 0.2 * (alt - 2000) / 33000)

    df["altitude_risk"] = df["altitude_km"].apply(alt_risk)
    print(f"  ✓ Feature: altitude_risk (mean={df['altitude_risk'].mean():.3f})")
    return df


# ─────────────────────────────────────────────
# NORMALIZE ALL DENSITY FEATURES
# ─────────────────────────────────────────────

def normalize_features(df):
    """Normalize continuous features to 0–1 range."""
    scaler = MinMaxScaler()
    cols_to_scale = ["orbital_density", "debris_density", "proximity_score"]

    for col in cols_to_scale:
        if col in df.columns:
            df[f"{col}_norm"] = scaler.fit_transform(df[[col]])

    print("  ✓ Normalized: orbital_density, debris_density, proximity_score")
    return df


# ─────────────────────────────────────────────
# FEATURE VALIDATION
# ─────────────────────────────────────────────

def validate_features(df):
    print("\n" + "=" * 60)
    print("  FEATURE VALIDATION REPORT")
    print("=" * 60)

    feature_cols = [
        "altitude_km", "inclination", "eccentricity", "velocity_kms",
        "orbital_density", "debris_density", "proximity_score",
        "velocity_risk", "altitude_risk", "eccentricity_risk",
        "is_leo", "high_inclination", "sun_sync_orbit"
    ]

    print(f"\n  Dataset shape: {df.shape}")
    print(f"\n  {'Feature':<28} {'Min':>8} {'Max':>10} {'Mean':>10} {'Nulls':>8}")
    print("  " + "-" * 68)

    for col in feature_cols:
        if col in df.columns:
            nulls = df[col].isnull().sum()
            null_flag = " ⚠️" if nulls > 0 else ""
            print(f"  {col:<28} {df[col].min():>8.2f} {df[col].max():>10.2f} "
                  f"{df[col].mean():>10.2f} {nulls:>8}{null_flag}")

    print("=" * 60)


# ─────────────────────────────────────────────
# CORRELATION HEATMAP
# ─────────────────────────────────────────────

def plot_correlation_heatmap(df):
    numeric_cols = [
        "altitude_km", "inclination", "eccentricity", "velocity_kms",
        "orbital_density", "debris_density", "proximity_score",
        "velocity_risk", "altitude_risk"
    ]
    existing_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[existing_cols].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, ax=ax, annot=True, fmt=".2f",
        cmap="coolwarm", center=0, square=True,
        linewidths=0.5, linecolor="#30363D",
        cbar_kws={"shrink": 0.8},
        annot_kws={"size": 9}
    )
    ax.set_title("🔬  Feature Correlation Matrix", fontsize=14, color="#F0F6FC", pad=15)
    plt.tight_layout()
    plt.savefig(f"{PLOTS_DIR}/07_feature_correlation.png", bbox_inches="tight")
    plt.close()
    print("  ✓ Correlation heatmap saved")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n⚙️  " * 20)
    print("  PHASE 3: Feature Engineering")
    print("⚙️  " * 20 + "\n")

    sat_df, debris_df = load_data()

    print("\nEngineering features...")
    sat_df = feature_orbital_band(sat_df)
    sat_df = feature_inclination_flags(sat_df)
    sat_df = feature_orbital_density(sat_df, debris_df)
    sat_df = feature_proximity_score(sat_df, debris_df)
    sat_df = feature_velocity_risk(sat_df)
    sat_df = feature_eccentricity_risk(sat_df)
    sat_df = feature_altitude_risk(sat_df)
    sat_df = normalize_features(sat_df)

    # Fill any remaining nulls
    numeric_cols = sat_df.select_dtypes(include=[np.number]).columns
    sat_df[numeric_cols] = sat_df[numeric_cols].fillna(0)

    validate_features(sat_df)
    plot_correlation_heatmap(sat_df)

    # Save features
    output_path = "data/processed/features_df.csv"
    sat_df.to_csv(output_path, index=False)
    print(f"\n  ✓ Feature dataset saved: {output_path}")
    print(f"  Shape: {sat_df.shape[0]:,} rows × {sat_df.shape[1]} columns")

    print("\n✅ Phase 3 Complete!")
    print("   Next step: Run 04_collision_model.py\n")


if __name__ == "__main__":
    main()
