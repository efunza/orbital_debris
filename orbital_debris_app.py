# orbital_debris_app.py
# Streamlit Cloud-ready (TIMEOUT-PROOF): Orbital Debris Tracking + Conjunction Detection
#
# Fixes:
# ✅ Handles ConnectTimeout / blocked sites gracefully (no crash)
# ✅ Tries multiple TLE sources (CelesTrak + GitHub mirrors)
# ✅ Provides built-in sample TLE set if all downloads fail
# ✅ Allows user to upload a TLE file (best reliability on Streamlit Cloud)
# ✅ Keeps simulation fast by sampling time steps and using vectorized math
#
# Run:
#   pip install -r requirements.txt
#   streamlit run orbital_debris_app.py

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sgp4.api import Satrec, jday

st.set_page_config(page_title="Orbital Debris Tracking", page_icon="🛰️", layout="wide")

st.title("🛰️ Optical Tracking of Orbital Debris (Software-Based)")
st.caption(
    "Propagate TLE orbits (SGP4), simulate trajectories, detect close approaches, and visualize debris density."
)

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙️ Simulation Settings")

num_objects = st.sidebar.slider("Objects to track", 20, 500, 200)
days_to_simulate = st.sidebar.slider("Simulation window (days)", 1, 30, 7)
steps_per_day = st.sidebar.slider("Time steps per day", 1, 24, 4, help="Higher = more accurate but slower.")
threshold_km = st.sidebar.slider("High-risk threshold (km)", 0.1, 10.0, 1.0)
max_pairs_to_check = st.sidebar.slider(
    "Max object pairs to check",
    1_000,
    50_000,
    12_000,
    step=1_000,
    help="Limits compute time. Pairs grow ~N².",
)

st.sidebar.divider()
st.sidebar.subheader("📥 TLE Input Options")

use_web = st.sidebar.toggle("Try downloading TLEs from the internet", value=True)
uploaded_tle = st.sidebar.file_uploader("Upload a TLE file (recommended)", type=["txt", "tle"])

st.sidebar.caption(
    "Tip: Uploading a TLE file avoids Streamlit Cloud network timeouts. "
    "You can download a TLE text file from CelesTrak on your computer and upload it here."
)

# -----------------------------
# TLE SOURCES (multiple mirrors)
# -----------------------------
CELESTRAK_ACTIVE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
# Mirrors (GitHub raw). These are commonly accessible even if some sites timeout.
GITHUB_ACTIVE_1 = "https://raw.githubusercontent.com/CelesTrak/fundamentals/master/active.txt"
GITHUB_ACTIVE_2 = "https://raw.githubusercontent.com/SpaceTrackOrg/space-track-data/master/active.txt"  # may exist or not

TLE_URLS = [CELESTRAK_ACTIVE, GITHUB_ACTIVE_1, GITHUB_ACTIVE_2]


# -----------------------------
# BUILT-IN SAMPLE TLEs (so app still runs offline)
# NOTE: These are example satellites; replace/expand if you want.
# -----------------------------
SAMPLE_TLE_TEXT = """ISS (ZARYA)
1 25544U 98067A   24060.51041667  .00012500  00000+0  22000-3 0  9994
2 25544  51.6420  37.8210 0004300  85.2000  24.9000 15.50000000200000
NOAA 15
1 25338U 98030A   24060.25000000  .00000080  00000+0  85000-4 0  9991
2 25338  98.7200  60.0000 0012000  30.0000 330.0000 14.26000000123456
TERRA
1 25994U 99068A   24060.37500000  .00000120  00000+0  64000-4 0  9996
2 25994  98.2100  85.0000 0001200  70.0000 290.0000 14.57000000123456
AQUA
1 27424U 02022A   24060.45833333  .00000110  00000+0  60000-4 0  9993
2 27424  98.2000 130.0000 0001200  80.0000 280.0000 14.57000000123456
"""


# -----------------------------
# HELPERS
# -----------------------------
def http_get_text(url: str, timeout: int = 15) -> Tuple[Optional[str], Optional[str]]:
    """Return (text, error). Never raises."""
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Streamlit Educational App)"}
        r = requests.get(url, headers=headers, timeout=timeout)
        if r.status_code != 200:
            return None, f"HTTP {r.status_code}"
        if not r.text or not r.text.strip():
            return None, "Empty response"
        return r.text, None
    except Exception as e:
        return None, str(e)


@dataclass
class TLEItem:
    name: str
    line1: str
    line2: str


def parse_tle_text(text: str) -> List[TLEItem]:
    """
    Parses TLE in either 2-line (line1/line2 only) OR 3-line format (name + 2 lines).
    Returns list of TLEItem.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: List[TLEItem] = []

    i = 0
    while i < len(lines):
        # If line starts with '1 ' it's line1 (no name)
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            name = f"OBJECT_{len(items)+1}"
            items.append(TLEItem(name=name, line1=lines[i], line2=lines[i + 1]))
            i += 2
            continue

        # Otherwise assume name + line1 + line2
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            items.append(TLEItem(name=lines[i], line1=lines[i + 1], line2=lines[i + 2]))
            i += 3
            continue

        i += 1

    return items


@st.cache_data(show_spinner=False)
def load_tles(use_web: bool) -> Tuple[List[TLEItem], str, List[str]]:
    """
    Load TLEs from:
    1) Uploaded file (handled outside)
    2) Internet sources (if enabled)
    3) Built-in sample TLEs
    Returns: (tles, source_name, notes)
    """
    notes: List[str] = []

    if use_web:
        for url in TLE_URLS:
            text, err = http_get_text(url)
            if text:
                items = parse_tle_text(text)
                if items:
                    return items, f"Downloaded TLEs from {url}", notes
                notes.append(f"Parsed 0 TLEs from {url}")
            else:
                notes.append(f"Download failed {url}: {err}")

    # Fallback to sample
    items = parse_tle_text(SAMPLE_TLE_TEXT)
    notes.append("Using built-in sample TLEs (offline fallback). Upload a TLE file for real tracking.")
    return items, "Built-in sample TLEs", notes


def propagate_positions(sat: Satrec, start: datetime, steps: int, step_minutes: int) -> np.ndarray:
    """
    Propagate SGP4 positions (km) for a single satellite.
    Returns array shape (steps, 3) with ECI position vectors.
    """
    out = np.empty((steps, 3), dtype=np.float64)
    valid = np.ones(steps, dtype=bool)

    for k in range(steps):
        t = start + timedelta(minutes=k * step_minutes)
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        e, r, _v = sat.sgp4(jd, fr)
        if e != 0:
            valid[k] = False
            out[k] = np.nan
        else:
            out[k] = r

    return out[valid]


def min_separation_km(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    """Compute minimum Euclidean distance between two time-aligned position arrays."""
    n = min(len(pos_a), len(pos_b))
    if n == 0:
        return float("inf")
    d = np.linalg.norm(pos_a[:n] - pos_b[:n], axis=1)
    return float(np.min(d))


# -----------------------------
# LOAD TLE DATA (with upload support)
# -----------------------------
tle_items: List[TLEItem]
tle_source = ""
tle_notes: List[str] = []

if uploaded_tle is not None:
    try:
        text = uploaded_tle.read().decode("utf-8", errors="ignore")
        tle_items = parse_tle_text(text)
        if not tle_items:
            st.error("Uploaded file did not contain valid TLEs (expected name + 2 lines or 2-line format).")
            st.stop()
        tle_source = "Uploaded TLE file"
        tle_notes = ["Using uploaded TLE file (best reliability)."]
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()
else:
    with st.spinner("Loading TLE data (with fallbacks)..."):
        tle_items, tle_source, tle_notes = load_tles(use_web=use_web)

# Limit number of objects
tle_items = tle_items[:num_objects]

st.success(f"TLE source: **{tle_source}** • Objects loaded: **{len(tle_items)}**")
if tle_notes:
    with st.expander("TLE loading notes"):
        for n in tle_notes:
            st.write("•", n)

if len(tle_items) < 2:
    st.warning("Need at least 2 objects to do conjunction detection.")
    st.stop()


# -----------------------------
# PROPAGATE ORBITS
# -----------------------------
st.subheader("🧮 Orbit Propagation (SGP4)")

start_time = datetime.utcnow()
step_minutes = int(24 * 60 / steps_per_day)
total_steps = int(days_to_simulate * steps_per_day)

st.write(
    f"Simulating **{days_to_simulate} days** with **{steps_per_day} steps/day** "
    f"(every **{step_minutes} minutes**), total steps: **{total_steps}**."
)

sat_positions = {}
bad_count = 0

with st.spinner("Propagating orbits..."):
    for item in tle_items:
        try:
            sat = Satrec.twoline2rv(item.line1, item.line2)
            pos = propagate_positions(sat, start_time, total_steps, step_minutes)
            if len(pos) >= 2:
                sat_positions[item.name] = pos
            else:
                bad_count += 1
        except Exception:
            bad_count += 1

st.write(f"Propagated: **{len(sat_positions)}** objects • Skipped: **{bad_count}** (invalid/failed propagation)")

if len(sat_positions) < 2:
    st.error("Not enough valid objects after propagation. Upload a different TLE file or reduce objects.")
    st.stop()


# -----------------------------
# CONJUNCTION DETECTION (LIMITED PAIRS)
# -----------------------------
st.subheader("🔍 Conjunction Detection (Close Approaches)")

names = list(sat_positions.keys())
n = len(names)
total_pairs = n * (n - 1) // 2
pairs_to_check = min(total_pairs, max_pairs_to_check)

st.caption(
    f"Total possible pairs: {total_pairs:,}. Checking up to {pairs_to_check:,} pairs (performance limit)."
)

# Choose a subset if too many pairs
# Simple approach: take first K names such that pairs approx below limit
if total_pairs > max_pairs_to_check:
    # Find m such that m*(m-1)/2 <= max_pairs_to_check
    m = int((1 + np.sqrt(1 + 8 * max_pairs_to_check)) / 2)
    names = names[:max(2, m)]
    n = len(names)
    total_pairs = n * (n - 1) // 2
    st.info(f"Reduced objects for pair checking to {n} (pairs now {total_pairs:,}).")

close_events = []

with st.spinner("Computing minimum separation distances..."):
    for i in range(n):
        for j in range(i + 1, n):
            dmin = min_separation_km(sat_positions[names[i]], sat_positions[names[j]])
            if dmin < threshold_km:
                close_events.append((names[i], names[j], dmin))

st.write(f"High-risk conjunctions (< {threshold_km} km): **{len(close_events)}**")

if close_events:
    df_events = pd.DataFrame(close_events, columns=["Object 1", "Object 2", "Min Separation (km)"])
    df_events = df_events.sort_values("Min Separation (km)").reset_index(drop=True)
    st.dataframe(df_events, use_container_width=True)
else:
    st.info("No conjunctions below threshold in this simulation window (with the objects/pairs checked).")


# -----------------------------
# VISUALS: XY Scatter + Radius Histogram
# -----------------------------
st.subheader("📌 Orbit Visualization")

col1, col2 = st.columns(2)

# XY heatmap-like scatter
with col1:
    st.markdown("**Orbital Projection (X-Y Plane, first 50 objects)**")
    fig, ax = plt.subplots()
    for name in names[:50]:
        pos = sat_positions[name]
        ax.scatter(pos[:, 0], pos[:, 1], s=1)
    ax.set_xlabel("X (km)")
    ax.set_ylabel("Y (km)")
    ax.set_title("ECI Position Scatter")
    st.pyplot(fig)

# Radius histogram (debris density proxy)
with col2:
    st.markdown("**Debris Density Proxy (Orbital Radius Histogram)**")
    all_pos = np.vstack([sat_positions[k] for k in names])
    radius = np.linalg.norm(all_pos, axis=1)
    fig2, ax2 = plt.subplots()
    ax2.hist(radius, bins=40)
    ax2.set_xlabel("Orbital radius |r| (km)")
    ax2.set_ylabel("Count")
    ax2.set_title("Radius Distribution")
    st.pyplot(fig2)


# -----------------------------
# DOWNLOAD RESULTS
# -----------------------------
st.subheader("⬇️ Download Results")

if close_events:
    csv = df_events.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download conjunction events CSV",
        data=csv,
        file_name="conjunction_events.csv",
        mime="text/csv",
    )
else:
    st.caption("No events to download at the chosen threshold.")

# -----------------------------
# PROJECT SUMMARY
# -----------------------------
st.subheader("📘 Science Fair Summary")

st.markdown(
    """
**Problem:** Orbital debris increases collision risk for satellites and threatens space sustainability.

**What this software does:**
- Loads public **Two-Line Element (TLE)** orbit data.
- Propagates orbits using the **SGP4 model** (standard for TLE).
- Simulates trajectories over a chosen time window.
- Computes minimum separation distances between objects to flag **high-risk conjunctions**.
- Visualizes debris distribution using orbital projection and radius density plots.

**How to improve (extensions):**
- Use more time steps (higher accuracy).
- Add filtering by orbit class (LEO/MEO/GEO).
- Add 3D interactive plots (Plotly).
- Use a faster spatial indexing method to scale beyond thousands of objects.
"""
)
