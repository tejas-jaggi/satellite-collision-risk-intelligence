"""
============================================================
PHASE 4: COLLISION RISK MODEL
============================================================
Satellite Collision Risk Intelligence Platform

This version fixes:
• Label leakage
• Class imbalance
• Unrealistic metrics
• Adds feature importance plot

Run:
python notebooks/04_collision_model.py
============================================================
"""

import pandas as pd
import numpy as np
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------

DATA_PATH = "data/processed/features_df.csv"
MODEL_PATH = "models/collision_model.pkl"
OUTPUT_PATH = "data/processed/model_output.csv"
PLOT_PATH = "outputs/plots/08_feature_importance.png"

os.makedirs("models", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

print("\n🚀 PHASE 4: Collision Risk Model\n")

df = pd.read_csv(DATA_PATH)

print(f"Loaded dataset: {df.shape[0]:,} rows × {df.shape[1]} columns")

# ---------------------------------------------------------
# CREATE LABEL (collision risk)
# ---------------------------------------------------------

df["collision_risk"] = (
    (df["altitude_km"] < 1200) &
    (df["inclination"] > 50)
).astype(int)

print("\nLabel distribution:")
print(df["collision_risk"].value_counts())

# ---------------------------------------------------------
# REMOVE LEAKAGE FEATURES
# ---------------------------------------------------------

# Only use raw physical parameters

FEATURES = [
    "eccentricity",
    "velocity_kms",
    "orbital_density",
    "is_leo",
    "high_inclination",
    "sun_sync_orbit"
]

X = df[FEATURES]
y = df["collision_risk"]

print("\nUsing features:")
print(FEATURES)

# ---------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain set:", X_train.shape[0])
print("Test set:", X_test.shape[0])

# ---------------------------------------------------------
# HANDLE CLASS IMBALANCE
# ---------------------------------------------------------

neg = sum(y_train == 0)
pos = sum(y_train == 1)

# Safety guard
if pos == 0:
    raise ValueError(
        "No positive samples generated for collision_risk. Adjust the label rule."
    )

scale_pos_weight = neg / pos

print(f"\nClass imbalance ratio: {scale_pos_weight:.2f}")

# ---------------------------------------------------------
# MODEL
# ---------------------------------------------------------

model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric="logloss"
)

print("\nTraining XGBoost model...")

model.fit(X_train, y_train)

# ---------------------------------------------------------
# PREDICTIONS
# ---------------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# ---------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)

print("\nMODEL EVALUATION")
print("-------------------------")

print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC:  {roc:.4f}")

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# ---------------------------------------------------------
# FEATURE IMPORTANCE
# ---------------------------------------------------------

import pandas as pd

importance = pd.DataFrame({
    "feature": FEATURES,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\nFeature Importance:")
print(importance)

import matplotlib.pyplot as plt

plt.figure(figsize=(8,5))

plt.barh(
    importance["feature"],
    importance["importance"]
)

plt.xlabel("Importance Score")
plt.title("Collision Risk Feature Importance")

plt.gca().invert_yaxis()

plt.tight_layout()

plt.savefig("outputs/plots/08_feature_importance.png")

print("\nFeature importance chart saved → outputs/plots/08_feature_importance.png")

# ---------------------------------------------------------
# SAVE MODEL
# ---------------------------------------------------------

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print(f"✓ Model saved: {MODEL_PATH}")

# ---------------------------------------------------------
# GENERATE RISK PROBABILITY
# ---------------------------------------------------------

df["collision_probability"] = model.predict_proba(X)[:,1]

df.to_csv(OUTPUT_PATH, index=False)

print(f"✓ Model output saved: {OUTPUT_PATH}")