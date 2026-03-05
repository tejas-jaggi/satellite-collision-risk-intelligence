"""
============================================================
PHASE 5: COLLISION RISK SCORING ENGINE
============================================================

Converts ML collision probability into operational risk scores
for satellite monitoring.

Run:
python notebooks/05_risk_scoring.py
"""

import pandas as pd
import numpy as np
import os

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

INPUT_PATH = "data/processed/model_output.csv"
OUTPUT_PATH = "data/processed/final_satellite_risk_scores.csv"

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\n🛰 PHASE 5: Collision Risk Scoring\n")

df = pd.read_csv(INPUT_PATH)

print(f"Loaded {len(df):,} satellite records")

# ---------------------------------------------------------
# CREATE RISK SCORE (0-100)
# ---------------------------------------------------------

# Base probability from ML model
df["base_risk"] = df["collision_probability"]

# Congestion factor
density_norm = df["orbital_density"] / df["orbital_density"].max()

# Velocity factor
velocity_norm = df["velocity_kms"] / df["velocity_kms"].max()

# Risk score formula
df["risk_score"] = (
    0.6 * df["base_risk"] +
    0.25 * density_norm +
    0.15 * velocity_norm
)

# Scale to 0–100
df["risk_score"] = (df["risk_score"] * 100).round(1)

# ---------------------------------------------------------
# RISK CATEGORIES
# ---------------------------------------------------------

def risk_category(score):

    if score >= 75:
        return "CRITICAL"

    elif score >= 50:
        return "HIGH"

    elif score >= 25:
        return "MEDIUM"

    else:
        return "LOW"

df["risk_category"] = df["risk_score"].apply(risk_category)

# ---------------------------------------------------------
# SORT BY RISK
# ---------------------------------------------------------

df = df.sort_values("risk_score", ascending=False)

# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

df.to_csv(OUTPUT_PATH, index=False)

print("\nTop 10 Highest Risk Satellites:\n")

print(
    df[[
        "sat_id",
        "name",
        "risk_score",
        "risk_category",
        "collision_probability"
    ]].head(10)
)

print(f"\nSaved results → {OUTPUT_PATH}")