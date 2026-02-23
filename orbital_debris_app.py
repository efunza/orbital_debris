# orbital_debris_app.py
# Optical Tracking of Orbital Debris – Software-Based System
# Streamlit Cloud Ready

import streamlit as st
import numpy as np
import pandas as pd
import requests
from datetime import datetime, timedelta
from sgp4.api import Satrec, jday
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="Optical Tracking of Orbital Debris",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ Optical Tracking of Orbital Debris")
st.caption("Software-based orbital simulation and collision prediction using public TLE data.")

# -----------------------------
# SETTINGS
# -----------------------------
st.sidebar.header("Simulation Settings")

num_objects = st.sidebar.slider("Number of objects to simulate", 50, 500, 200)
days_to_simulate = st.sidebar.slider("Simulation window (days)", 1, 30, 7)
distance_threshold_km = st.sidebar.slider("Collision threshold (km)", 0.1, 5.0, 1.0)

# -----------------------------
# LOAD TLE DATA
# -----------------------------
@st.cache_data
def load_tle():
    url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
    r = requests.get(url, timeout=30)
    lines = r.text.strip().split("\n")

    sats = []
    for i in range(0, len(lines), 3):
        try:
            name = lines[i].strip()
            line1 = lines[i+1].strip()
            line2 = lines[i+2].strip()
            sats.append((name, line1, line2))
        except:
            continue
    return sats

st.write("Loading TLE data...")
tle_data = load_tle()

if len(tle_data) == 0:
    st.error("Failed to load TLE data.")
    st.stop()

tle_data = tle_data[:num_objects]

st.success(f"Loaded {len(tle_data)} orbital objects.")

# -----------------------------
# ORBIT PROPAGATION
# -----------------------------
def propagate_satellite(sat, days):
    positions = []
    now = datetime.utcnow()

    for day in range(days):
        t = now + timedelta(days=day)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        e, r, v = sat.sgp4(jd, fr)
        if e == 0:
            positions.append(r)
    return np.array(positions)

sat_positions = {}

for name, l1, l2 in tle_data:
    sat = Satrec.twoline2rv(l1, l2)
    pos = propagate_satellite(sat, days_to_simulate)
    if len(pos) > 0:
        sat_positions[name] = pos

st.write(f"Propagated {len(sat_positions)} objects.")

# -----------------------------
# PROXIMITY DETECTION
# -----------------------------
st.subheader("🔍 Conjunction Detection")

close_events = []

names = list(sat_positions.keys())

for i in range(len(names)):
    for j in range(i+1, len(names)):
        sat1 = sat_positions[names[i]]
        sat2 = sat_positions[names[j]]

        min_len = min(len(sat1), len(sat2))
        if min_len == 0:
            continue

        distances = np.linalg.norm(sat1[:min_len] - sat2[:min_len], axis=1)
        min_distance = np.min(distances)

        if min_distance < distance_threshold_km:
            close_events.append((names[i], names[j], min_distance))

st.write(f"High-risk conjunctions detected: {len(close_events)}")

if close_events:
    df_events = pd.DataFrame(close_events, columns=["Object 1", "Object 2", "Min Distance (km)"])
    st.dataframe(df_events)

# -----------------------------
# ORBIT VISUALIZATION
# -----------------------------
st.subheader("🌍 Orbital Heatmap Projection (X-Y Plane)")

fig, ax = plt.subplots()

for name in list(sat_positions.keys())[:50]:
    pos = sat_positions[name]
    ax.scatter(pos[:, 0], pos[:, 1], s=1)

ax.set_xlabel("X Position (km)")
ax.set_ylabel("Y Position (km)")
ax.set_title("Orbital Object Distribution")
st.pyplot(fig)

# -----------------------------
# DENSITY MODEL
# -----------------------------
st.subheader("📊 Debris Density Distribution")

all_positions = np.vstack(list(sat_positions.values()))

radius = np.linalg.norm(all_positions, axis=1)

fig2, ax2 = plt.subplots()
ax2.hist(radius, bins=50)
ax2.set_xlabel("Orbital Radius (km)")
ax2.set_ylabel("Frequency")
ax2.set_title("Orbital Radius Density")
st.pyplot(fig2)

# -----------------------------
# SUMMARY
# -----------------------------
st.subheader("📘 Study Summary")

st.markdown("""
This software-based system:

- Loads public TLE orbital data
- Propagates satellite motion using SGP4 orbital mechanics
- Simulates up to 30 days of orbital movement
- Detects close conjunction events below user-defined threshold
- Visualizes debris density and orbital distribution

This demonstrates how computational astrodynamics tools can support
space situational awareness without optical hardware.

Educational Concepts:
- Keplerian orbital motion
- SGP4 perturbation model
- Vector distance calculations
- Collision risk modeling
- Space sustainability analysis
""")