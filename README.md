# 🛰️ Satellite Collision Risk Intelligence Platform

> *Real orbital data. Real ML model. Real collision risk scoring.*

[![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python)](https://python.org)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red?logo=streamlit)](https://streamlit.io)
[![Data](https://img.shields.io/badge/Data-CelesTrak-green)](https://celestrak.org)

---

## 🎯 Project Overview

An end-to-end intelligence platform that ingests live satellite orbital data from **CelesTrak**, engineers collision-risk features using orbital mechanics, trains an **XGBoost classifier**, and serves real-time risk scores through an **interactive Streamlit dashboard**.

This mimics real-world **Space Traffic Management (STM)** systems used by NASA, ESA, and commercial operators.

**Core Question:**
> *"Which satellites have the highest probability of collision in the next orbit cycle?"*

---

## 🚀 Live Demo

[🌐 Open Dashboard](https://your-app.streamlit.app) ← *Deploy to Streamlit Cloud*

---

## 📊 Key Findings

*(Fill these in after running the pipeline)*

- **XX%** of tracked objects cluster in the high-risk LEO zone below 1,000km
- **Top risk country:** [fill]
- **Model ROC-AUC:** [fill after training]
- **CRITICAL tier satellites:** [fill]

---

## 🏗️ Architecture

```
CelesTrak API (~35,000 objects)
        ↓
01_data_ingestion.py  ← ETL pipeline → SQLite
        ↓
02_eda.py             ← Exploratory analysis + charts
        ↓
03_feature_engineering.py  ← 10+ orbital risk features
        ↓
04_collision_model.py ← XGBoost binary classifier
        ↓
05_risk_scoring.py    ← Composite risk score + tiers
        ↓
app.py                ← Streamlit monitoring dashboard
```

---

## 📁 Project Structure

```
space-collision-intelligence/
├── data/
│   ├── raw/                    ← CelesTrak API JSON responses
│   ├── processed/              ← Cleaned + feature-engineered CSVs
│   └── space_data.db           ← SQLite database
├── notebooks/
│   ├── 01_data_ingestion.py
│   ├── 02_eda.py
│   ├── 03_feature_engineering.py
│   ├── 04_collision_model.py
│   └── 05_risk_scoring.py
├── models/
│   └── collision_model.pkl     ← Trained XGBoost model
├── outputs/
│   ├── satellite_risk_summary.csv
│   └── plots/                  ← EDA + model charts
├── app.py                      ← Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## ⚙️ Setup & Run

### 1. Clone & Install

```bash
git clone https://github.com/YOUR_USERNAME/space-collision-intelligence.git
cd space-collision-intelligence

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

### 2. Run Pipeline (in order)

```bash
python notebooks/01_data_ingestion.py    # ~5 min (API fetch)
python notebooks/02_eda.py               # ~2 min
python notebooks/03_feature_engineering.py  # ~3 min
python notebooks/04_collision_model.py   # ~5 min
python notebooks/05_risk_scoring.py      # ~1 min
```

### 3. Launch Dashboard

```bash
streamlit run app.py
# Opens at http://localhost:8501
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Data Pipeline | Python · Requests · SQLite |
| Data Processing | Pandas · NumPy |
| Visualization | Matplotlib · Seaborn · Plotly |
| Machine Learning | Scikit-learn · XGBoost |
| Dashboard | Streamlit |
| Deployment | GitHub · Streamlit Cloud |

---

## 📡 Data Sources

| Source | Description | Size |
|--------|-------------|------|
| [CelesTrak Active](https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json) | Live orbital elements for active satellites | ~9,000 objects |
| [CelesTrak Debris](https://celestrak.org/NORAD/elements/gp.php?GROUP=debris&FORMAT=json) | Space debris tracking data | ~25,000 objects |

---

## 🔬 Features Engineered

| Feature | Description |
|---------|-------------|
| `altitude_km` | Orbital altitude derived from TLE mean motion |
| `orbital_density` | Objects in ±50km altitude window |
| `debris_density` | Debris objects in proximity band |
| `proximity_score` | Normalized distance to nearest debris cluster |
| `velocity_risk` | Risk-weighted orbital velocity |
| `altitude_risk` | Zone-based altitude risk (peaks at 400-600km) |
| `eccentricity_risk` | Orbit crossing risk from eccentricity |
| `is_leo` | Binary: altitude < 2000km |
| `high_inclination` | Binary: inclination 80-110° (polar crossings) |
| `sun_sync_orbit` | Binary: inclination 95-100° (congested zone) |

---

## 🏆 Risk Scoring Formula

```
collision_risk = 0.40 × model_probability
               + 0.25 × proximity_score
               + 0.20 × orbital_density_score
               + 0.15 × velocity_risk
```

| Tier | Score Range | Action |
|------|-------------|--------|
| 🔴 CRITICAL | ≥ 0.70 | Immediate monitoring required |
| 🟠 HIGH | 0.50 – 0.70 | Close monitoring required |
| 🟡 MEDIUM | 0.30 – 0.50 | Standard monitoring |
| 🟢 LOW | < 0.30 | Routine tracking |

---

## 💼 Portfolio Skills Demonstrated

- **Data Engineering:** API ingestion, ETL, SQLite schema design
- **Data Analysis:** EDA, insight discovery, orbital mechanics
- **Data Science:** Feature engineering, XGBoost modeling, ROC-AUC evaluation
- **Risk Analytics:** Composite scoring, tier classification (analogous to credit/supplier risk)
- **Deployment:** GitHub, Streamlit Cloud, interactive dashboard

---

*Built as a portfolio project demonstrating full-stack data science capabilities in a real-world aerospace analytics context.*
