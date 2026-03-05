🛰️ Satellite Collision Risk Intelligence Platform

An end-to-end data science and machine learning platform that analyzes 58,000+ satellites to estimate orbital congestion and potential collision risk.

The system ingests real satellite orbital data, engineers physics-based features, trains a machine learning model, and presents results through an interactive **Streamlit dashboard**.

---

🌍 Live Dashboard

**Interactive App:**
https://satellite-collision-risk.streamlit.app/

Explore:

* Satellite collision risk levels
* Orbital congestion zones
* Satellite search and filtering
* Interactive orbital visualizations

---

🚀 Project Overview

Space around Earth is becoming increasingly crowded.

With thousands of satellites and debris objects in orbit, the probability of collisions is rising. A single collision can generate thousands of fragments, increasing the danger of further collisions.

This project builds a **collision risk intelligence system** that:

1. Collects satellite orbital data
2. Analyzes orbital congestion
3. Uses machine learning to estimate risk
4. Displays results through an interactive dashboard

---

🧠 Key Features

✔ Real satellite data ingestion from **CelesTrak**
✔ Orbital mechanics feature engineering
✔ Machine learning collision risk model (XGBoost)
✔ Interactive orbital visualization
✔ Satellite search and filtering
✔ Risk tier classification (Low → Critical)
✔ Deployable Streamlit dashboard

---

⚙️ Data Pipeline Architecture

```
CelesTrak API
      ↓
Data Ingestion
      ↓
Orbital Feature Engineering
      ↓
Machine Learning Model
      ↓
Risk Scoring Engine
      ↓
Streamlit Dashboard
```

---

🔬 Machine Learning Model

The system uses **XGBoost**, a gradient boosting algorithm well suited for structured data.

### Input Features

Examples include:

```
altitude_km
inclination
velocity_kms
orbital_density
eccentricity
is_leo
high_inclination
sun_sync_orbit
```

### Output

The model predicts: collision_probability

Which is converted into four risk categories:

```
LOW
MEDIUM
HIGH
CRITICAL
```

---

📈 Feature Engineering

Physics-inspired features were created to represent orbital risk:

| Feature           | Meaning                          |
| ----------------- | -------------------------------- |
| orbital_density   | number of nearby satellites      |
| proximity_score   | closeness of neighboring objects |
| velocity_risk     | collision impact severity        |
| altitude_risk     | crowded orbital bands            |
| eccentricity_risk | unstable orbits                  |

---

📊 Model Performance

```
Accuracy: 97.4%
ROC-AUC: 0.9976
```

These results show the model effectively identifies high-risk orbital environments.

---

🖥 Technologies Used

**Programming**

Python

**Data Processing**

* pandas
* numpy

**Machine Learning**

* scikit-learn
* XGBoost

**Visualization**

* Plotly
* Matplotlib
* Seaborn

**Dashboard**

* Streamlit

**Database**

SQLite

---

🔎 Dashboard Capabilities

The dashboard allows users to:

✔ Search satellites (STARLINK, ISS, COSMOS, etc.)
✔ Filter by altitude
✔ Filter by risk score
✔ Filter by risk tier
✔ View orbital congestion zones
✔ Download satellite risk reports

---

📚 What This Project Demonstrates

This project showcases skills in:

* Data engineering pipelines
* Orbital mechanics feature engineering
* Machine learning modeling
* Data visualization
* Interactive dashboards
* Cloud deployment
* Git version control

---

🚀 Future Improvements

Possible extensions include:

* Real-time satellite tracking
* Collision probability simulation
* Debris tracking integration
* Space traffic management tools
* AI-driven orbital congestion prediction

---

# 👨‍💻 Author

**Tejas Jaggi**

MS in Information Systems
University of Illinois Urbana-Champaign

GitHub
https://github.com/tejas-jaggi

---
