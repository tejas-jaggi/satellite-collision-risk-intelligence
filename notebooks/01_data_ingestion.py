"""
============================================================
PHASE 1: DATA ENGINEERING — Satellite Data Ingestion Pipeline
============================================================
Satellite Collision Risk Intelligence Platform
------------------------------------------------------------
What this script does:
  1. Fetches live satellite + debris data from CelesTrak API
  2. Parses orbital mechanics fields (TLE elements)
  3. Computes altitude and velocity from mean_motion
  4. Loads everything into a normalized SQLite database

Run: python notebooks/01_data_ingestion.py
============================================================
"""

import requests
import sqlite3
import json
import os
import time
import numpy as np
import pandas as pd
from datetime import datetime

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DB_PATH = "data/space_data.db"
RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

CELESTRAK_URLS = {
    "active":  "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json",
    "debris":  "https://celestrak.org/NORAD/elements/gp.php?GROUP=active-debris&FORMAT=json",
}

# Earth orbital mechanics constants
MU = 398600.4418      # km³/s² — Earth gravitational parameter
EARTH_RADIUS = 6371   # km

# ─────────────────────────────────────────────
# PHYSICS HELPERS
# ─────────────────────────────────────────────

def mean_motion_to_altitude(mean_motion_revday: float) -> float:
    """
    Convert TLE mean motion (revolutions/day) to orbital altitude (km).
    Uses Kepler's 3rd law: T = 2π√(a³/μ)
    """
    if mean_motion_revday <= 0:
        return None
    n = mean_motion_revday * 2 * np.pi / 86400   # Convert to rad/s
    a = (MU / (n ** 2)) ** (1 / 3)               # Semi-major axis in km
    altitude = a - EARTH_RADIUS
    return round(max(altitude, 0), 2)


def altitude_to_velocity(altitude_km: float) -> float:
    """
    Compute circular orbital velocity (km/s) at a given altitude.
    v = √(μ/r)
    """
    if altitude_km is None or altitude_km < 0:
        return None
    r = EARTH_RADIUS + altitude_km
    v = np.sqrt(MU / r)
    return round(v, 4)


def mean_motion_to_period(mean_motion_revday: float) -> float:
    """Convert mean motion to orbital period in minutes."""
    if mean_motion_revday <= 0:
        return None
    return round(1440 / mean_motion_revday, 2)   # 1440 min/day


# ─────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────

def create_database(conn: sqlite3.Connection):
    """Create the SQLite schema — 3 tables."""
    cursor = conn.cursor()

    cursor.executescript("""
        CREATE TABLE IF NOT EXISTS satellites (
            sat_id      INTEGER PRIMARY KEY,
            name        TEXT,
            country     TEXT DEFAULT 'Unknown',
            launch_year INTEGER,
            orbit_type  TEXT DEFAULT 'Unknown',
            purpose     TEXT DEFAULT 'Unknown'
        );

        CREATE TABLE IF NOT EXISTS orbits (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            sat_id          INTEGER,
            altitude_km     REAL,
            inclination     REAL,
            eccentricity    REAL,
            mean_motion     REAL,
            period_min      REAL,
            velocity_kms    REAL,
            ra_asc_node     REAL,
            arg_perigee     REAL,
            bstar           REAL,
            epoch           TEXT,
            timestamp       TEXT,
            FOREIGN KEY(sat_id) REFERENCES satellites(sat_id)
        );

        CREATE TABLE IF NOT EXISTS debris (
            debris_id       INTEGER PRIMARY KEY,
            name            TEXT,
            altitude_km     REAL,
            inclination     REAL,
            eccentricity    REAL,
            mean_motion     REAL,
            period_min      REAL,
            velocity_kms    REAL,
            epoch           TEXT
        );
    """)
    conn.commit()
    print("✓ Database schema created: satellites, orbits, debris")


# ─────────────────────────────────────────────
# DATA FETCHING
# ─────────────────────────────────────────────

def fetch_celestrak(group: str, url: str) -> list:
    """
    Fetch satellite data from CelesTrak API.
    Returns list of dicts. Saves raw JSON to data/raw/.
    """
    raw_path = os.path.join(RAW_DIR, f"{group}_raw.json")

    # Use cached file if exists and less than 24h old
    if os.path.exists(raw_path):
        age_hours = (time.time() - os.path.getmtime(raw_path)) / 3600
        if age_hours < 24:
            print(f"  Using cached {group} data (age: {age_hours:.1f}h)")
            with open(raw_path, "r") as f:
                return json.load(f)

    print(f"  Fetching {group} data from CelesTrak...")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
        }

        response = requests.get(url, headers=headers, timeout=90)
        response.raise_for_status()
        data = response.json()

        # Save raw JSON
        with open(raw_path, "w") as f:
            json.dump(data, f)

        print(f"  ✓ Fetched {len(data):,} {group} objects")
        return data

    except requests.exceptions.Timeout:
        print(f"  ✗ Timeout fetching {group}. Try again or check internet.")
        return []

    except requests.exceptions.HTTPError as e:
        print(f"  ✗ HTTP error {e.response.status_code}: {e}")
        return []

    except Exception as e:
        print(f"  ✗ Error: {e}")
        return []

# ─────────────────────────────────────────────
# DATA LOADING — SATELLITES
# ─────────────────────────────────────────────

def load_satellites(conn: sqlite3.Connection, data: list):
    """Parse active satellite JSON and insert into satellites + orbits tables."""
    cursor = conn.cursor()
    timestamp = datetime.utcnow().isoformat()

    sat_rows = []
    orbit_rows = []

    for item in data:
        sat_id = item.get("NORAD_CAT_ID")
        if not sat_id:
            continue

        name = item.get("OBJECT_NAME", "UNKNOWN")
        mean_motion = item.get("MEAN_MOTION", 0)
        epoch = item.get("EPOCH", "")

        altitude = mean_motion_to_altitude(mean_motion)
        velocity = altitude_to_velocity(altitude)
        period = mean_motion_to_period(mean_motion)

        sat_rows.append((
            sat_id, name, "Unknown", None, "Unknown", "Unknown"
        ))
        orbit_rows.append((
            sat_id,
            altitude,
            item.get("INCLINATION"),
            item.get("ECCENTRICITY"),
            mean_motion,
            period,
            velocity,
            item.get("RA_OF_ASC_NODE"),
            item.get("ARG_OF_PERICENTER"),
            item.get("BSTAR"),
            epoch,
            timestamp
        ))

    # Insert with conflict handling
    cursor.executemany("""
        INSERT OR IGNORE INTO satellites (sat_id, name, country, launch_year, orbit_type, purpose)
        VALUES (?, ?, ?, ?, ?, ?)
    """, sat_rows)

    cursor.executemany("""
        INSERT INTO orbits 
        (sat_id, altitude_km, inclination, eccentricity, mean_motion, period_min,
         velocity_kms, ra_asc_node, arg_perigee, bstar, epoch, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, orbit_rows)

    conn.commit()
    print(f"  ✓ Loaded {len(sat_rows):,} satellites into database")


# ─────────────────────────────────────────────
# DATA LOADING — DEBRIS
# ─────────────────────────────────────────────

def load_debris(conn: sqlite3.Connection, data: list):
    """Parse debris JSON and insert into debris table."""
    cursor = conn.cursor()
    debris_rows = []

    for item in data:
        debris_id = item.get("NORAD_CAT_ID")
        if not debris_id:
            continue

        mean_motion = item.get("MEAN_MOTION", 0)
        altitude = mean_motion_to_altitude(mean_motion)
        velocity = altitude_to_velocity(altitude)
        period = mean_motion_to_period(mean_motion)

        debris_rows.append((
            debris_id,
            item.get("OBJECT_NAME", "DEBRIS"),
            altitude,
            item.get("INCLINATION"),
            item.get("ECCENTRICITY"),
            mean_motion,
            period,
            velocity,
            item.get("EPOCH", "")
        ))

    cursor.executemany("""
        INSERT OR IGNORE INTO debris 
        (debris_id, name, altitude_km, inclination, eccentricity, mean_motion, period_min, velocity_kms, epoch)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, debris_rows)

    conn.commit()
    print(f"  ✓ Loaded {len(debris_rows):,} debris objects into database")


# ─────────────────────────────────────────────
# VALIDATION
# ─────────────────────────────────────────────

def validate_database(conn: sqlite3.Connection):
    """Print summary statistics to confirm data loaded correctly."""
    print("\n" + "="*55)
    print("  DATABASE VALIDATION")
    print("="*55)

    queries = {
        "Total satellites": "SELECT COUNT(*) FROM satellites",
        "Total orbit records": "SELECT COUNT(*) FROM orbits",
        "Total debris objects": "SELECT COUNT(*) FROM debris",
        "Avg satellite altitude (km)": "SELECT ROUND(AVG(altitude_km), 1) FROM orbits WHERE altitude_km > 0",
        "Min altitude (km)": "SELECT ROUND(MIN(altitude_km), 1) FROM orbits WHERE altitude_km > 100",
        "Max altitude (km)": "SELECT ROUND(MAX(altitude_km), 1) FROM orbits WHERE altitude_km < 200000",
        "Objects below 1000km": "SELECT COUNT(*) FROM orbits WHERE altitude_km < 1000 AND altitude_km > 100",
        "Objects in GEO belt (35k-36k km)": "SELECT COUNT(*) FROM orbits WHERE altitude_km BETWEEN 35500 AND 36000",
    }

    for label, query in queries.items():
        result = conn.execute(query).fetchone()[0]
        print(f"  {label:<38} {result:>12,}" if isinstance(result, int) 
              else f"  {label:<38} {result}")

    print("="*55)


def export_combined_csv(conn: sqlite3.Connection):
    """Export joined satellite + orbit data to CSV for downstream phases."""
    query = """
        SELECT 
            s.sat_id,
            s.name,
            s.country,
            s.launch_year,
            s.orbit_type,
            o.altitude_km,
            o.inclination,
            o.eccentricity,
            o.mean_motion,
            o.period_min,
            o.velocity_kms,
            o.ra_asc_node,
            o.arg_perigee,
            o.bstar,
            o.epoch
        FROM satellites s
        JOIN orbits o ON s.sat_id = o.sat_id
        WHERE o.altitude_km > 100
          AND o.altitude_km < 200000
        ORDER BY o.altitude_km
    """
    df = pd.read_sql(query, conn)
    out_path = "data/processed/satellites_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  ✓ Exported {len(df):,} rows to {out_path}")

    debris_df = pd.read_sql(
        "SELECT * FROM debris WHERE altitude_km > 100 AND altitude_km < 200000", conn
    )
    debris_df.to_csv("data/processed/debris_clean.csv", index=False)
    print(f"  ✓ Exported {len(debris_df):,} rows to data/processed/debris_clean.csv")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "🛰️ " * 20)
    print("  SATELLITE COLLISION RISK INTELLIGENCE PLATFORM")
    print("  Phase 1: Data Engineering — Ingestion Pipeline")
    print("🛰️ " * 20 + "\n")

    # Connect to database
    conn = sqlite3.connect(DB_PATH)
    create_database(conn)

    # Fetch + load active satellites
    print("\n[1/2] Fetching Active Satellites...")
    sat_data = fetch_celestrak("active", CELESTRAK_URLS["active"])
    if sat_data:
        load_satellites(conn, sat_data)
    else:
        print("  ✗ No satellite data. Check internet connection.")

    time.sleep(2)  # Be polite to the API

    # Fetch + load debris
    print("\n[2/2] Fetching Space Debris...")
    debris_data = fetch_celestrak("debris", CELESTRAK_URLS["debris"])
    if debris_data:
        load_debris(conn, debris_data)
    else:
        print("  ✗ No debris data. Will retry next run.")

    # Validate and export
    validate_database(conn)
    export_combined_csv(conn)

    conn.close()
    print("\n✅ Phase 1 Complete! Database ready at:", DB_PATH)
    print("   Next step: Run 02_eda.py\n")


if __name__ == "__main__":
    main()
