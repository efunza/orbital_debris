# orbital_debris_app.py
# UPGRADED + FIXED: Orbital Debris Tracking + Orbit Class Filter + 3D Plotly + Coarse→Refined Conjunction Detection
# FIXES INCLUDED:
# - Keep time alignment across satellites (no dropping invalid SGP4 points)
# - Skip NaN/invalid points in binning and distance checks (prevents bin_key crash)
# - Refine step uses the same time grid logic safely

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sgp4.api import Satrec, jday

import plotly.graph_objects as go

# -----------------------------
# CONFIG
# -----------------------------
EARTH_RADIUS_KM = 6371.0

CELESTRAK_ACTIVE = "https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=tle"
GITHUB_ACTIVE_1 = "https://raw.githubusercontent.com/CelesTrak/fundamentals/master/active.txt"

TLE_URLS = [CELESTRAK_ACTIVE, GITHUB_ACTIVE_1]

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
# STREAMLIT PAGE
# -----------------------------
st.set_page_config(page_title="Orbital Debris Tracker", page_icon="🛰️", layout="wide")
st.title("🛰️ Optical Tracking of Orbital Debris (Software-Based)")
st.caption("SGP4 orbit propagation + LEO/MEO/GEO filtering + 3D visualization + coarse→refined conjunction detection.")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("📥 TLE Input")
try_web = st.sidebar.toggle("Try downloading TLEs (may time out on Streamlit Cloud)", value=True)
uploaded_tle = st.sidebar.file_uploader("Upload a TLE file (recommended)", type=["txt", "tle"])

st.sidebar.divider()
st.sidebar.header("⚙️ Simulation Settings")

num_objects = st.sidebar.slider("Max objects to load", 20, 600, 250, step=10)
days_to_simulate = st.sidebar.slider("Simulation window (days)", 1, 30, 7)

# Coarse sampling
coarse_steps_per_day = st.sidebar.slider("Coarse steps/day", 1, 24, 4)
# Refinement
refine_minutes = st.sidebar.slider("Refined step (minutes)", 1, 60, 10)
refine_window_hours = st.sidebar.slider("Refine window (± hours around closest coarse step)", 1, 72, 24)

st.sidebar.divider()
st.sidebar.header("🔎 Detection Thresholds")

final_threshold_km = st.sidebar.slider("High-risk threshold (km)", 0.1, 10.0, 1.0)
coarse_candidate_km = st.sidebar.slider(
    "Coarse candidate threshold (km)",
    1.0,
    200.0,
    25.0,
    help="Pairs closer than this in coarse scan get refined."
)

st.sidebar.divider()
st.sidebar.header("🧭 Orbit Class Filter")
orbit_class = st.sidebar.selectbox("Analyze orbit class", ["All", "LEO", "MEO", "GEO"])

# Spatial indexing (grid binning)
st.sidebar.divider()
st.sidebar.header("⚡ Speed Controls")
bin_size_km = st.sidebar.slider("3D bin size for coarse scan (km)", 50, 1000, 250, step=50)
max_pairs_per_step = st.sidebar.slider("Max pairs compared per time step", 500, 20000, 6000, step=500)
max_total_candidates = st.sidebar.slider("Max candidate pairs to refine", 100, 20000, 4000, step=100)

# 3D plot controls
st.sidebar.divider()
st.sidebar.header("🧊 3D Plot")
plot_tracks = st.sidebar.slider("How many object tracks to plot (3D)", 5, 80, 25)
plot_points_cap = st.sidebar.slider("Max points per track (3D)", 30, 500, 150, step=10)

# -----------------------------
# TLE PARSING + LOADING
# -----------------------------
@dataclass
class TLEItem:
    name: str
    line1: str
    line2: str


def http_get_text(url: str, timeout: int = 12) -> Tuple[Optional[str], Optional[str]]:
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


def parse_tle_text(text: str) -> List[TLEItem]:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    items: List[TLEItem] = []
    i = 0
    while i < len(lines):
        # 2-line format (no name)
        if lines[i].startswith("1 ") and i + 1 < len(lines) and lines[i + 1].startswith("2 "):
            items.append(TLEItem(name=f"OBJECT_{len(items)+1}", line1=lines[i], line2=lines[i + 1]))
            i += 2
            continue
        # 3-line format (name + 2 lines)
        if i + 2 < len(lines) and lines[i + 1].startswith("1 ") and lines[i + 2].startswith("2 "):
            items.append(TLEItem(name=lines[i], line1=lines[i + 1], line2=lines[i + 2]))
            i += 3
            continue
        i += 1
    return items


@st.cache_data(show_spinner=False)
def load_tles_from_web() -> Tuple[List[TLEItem], str, List[str]]:
    notes: List[str] = []
    for url in TLE_URLS:
        text, err = http_get_text(url)
        if text:
            items = parse_tle_text(text)
            if items:
                return items, f"Downloaded from {url}", notes
            notes.append(f"Parsed 0 TLEs from {url}")
        else:
            notes.append(f"Download failed {url}: {err}")
    items = parse_tle_text(SAMPLE_TLE_TEXT)
    notes.append("Using built-in sample TLEs (web download failed). Upload a TLE file for real tracking.")
    return items, "Built-in sample", notes


# Load TLE items
if uploaded_tle is not None:
    raw = uploaded_tle.read().decode("utf-8", errors="ignore")
    tle_items = parse_tle_text(raw)
    tle_source = "Uploaded TLE file"
    tle_notes = ["Using uploaded file (best reliability on Streamlit Cloud)."]
else:
    if try_web:
        tle_items, tle_source, tle_notes = load_tles_from_web()
    else:
        tle_items = parse_tle_text(SAMPLE_TLE_TEXT)
        tle_source = "Built-in sample"
        tle_notes = ["Web download disabled; using built-in sample. Upload a TLE file for real tracking."]

tle_items = tle_items[:num_objects]

st.success(f"TLE source: **{tle_source}** • Loaded: **{len(tle_items)} objects**")
if tle_notes:
    with st.expander("TLE loading notes"):
        for n in tle_notes:
            st.write("•", n)

if len(tle_items) < 2:
    st.error("Need at least 2 TLE objects to run conjunction detection.")
    st.stop()

# -----------------------------
# ORBIT PROPAGATION
# -----------------------------
def propagate_times(start: datetime, steps: int, step_minutes: int) -> List[datetime]:
    return [start + timedelta(minutes=k * step_minutes) for k in range(steps)]


def sgp4_positions(sat: Satrec, times: List[datetime]) -> np.ndarray:
    """
    Return ECI position vectors (km) with time alignment preserved.
    Invalid points are filled with NaNs (instead of being dropped).
    Shape is always (len(times), 3).
    """
    out = np.empty((len(times), 3), dtype=np.float64)
    for i, t in enumerate(times):
        jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second)
        e, r, _v = sat.sgp4(jd, fr)
        out[i] = r if e == 0 else (np.nan, np.nan, np.nan)
    return out


def classify_orbit_by_alt_km(alt_km: float) -> str:
    # Simple bins (good enough for science fair)
    if alt_km < 2000:
        return "LEO"
    if alt_km < 35786:
        return "MEO"
    # GEO band (loose; real GEO has inclination/ecc constraints)
    if 33000 <= alt_km <= 39000:
        return "GEO"
    return "HighEarth/Other"


# Coarse time grid
coarse_step_minutes = int(24 * 60 / coarse_steps_per_day)
coarse_steps = int(days_to_simulate * coarse_steps_per_day)
t0 = datetime.utcnow()
coarse_times = propagate_times(t0, coarse_steps, coarse_step_minutes)

st.subheader("🧮 Propagation + Orbit Class Filtering")

with st.spinner("Propagating coarse trajectories (SGP4)..."):
    sat_pos_coarse: Dict[str, np.ndarray] = {}
    sat_class: Dict[str, str] = {}
    sat_alt0: Dict[str, float] = {}
    bad = 0

    for item in tle_items:
        try:
            sat = Satrec.twoline2rv(item.line1, item.line2)
            pos = sgp4_positions(sat, coarse_times)

            # Find first valid point for classification
            finite_mask = np.isfinite(pos).all(axis=1)
            if not np.any(finite_mask):
                bad += 1
                continue
            first_idx = int(np.argmax(finite_mask))
            r0 = float(np.linalg.norm(pos[first_idx]))
            alt0 = r0 - EARTH_RADIUS_KM
            cls = classify_orbit_by_alt_km(alt0)

            sat_pos_coarse[item.name] = pos
            sat_class[item.name] = cls
            sat_alt0[item.name] = alt0
        except Exception:
            bad += 1

names_all = list(sat_pos_coarse.keys())

if orbit_class != "All":
    names = [n for n in names_all if sat_class.get(n) == orbit_class]
else:
    names = names_all

st.write(
    f"Valid propagated: **{len(names_all)}** • Skipped: **{bad}** • "
    f"After filter ({orbit_class}): **{len(names)}**"
)

if len(names) < 2:
    st.warning("Not enough objects after orbit-class filtering. Choose 'All' or upload a larger TLE set.")
    st.stop()

# Show class counts
counts = pd.Series([sat_class[n] for n in names_all]).value_counts()
with st.expander("Orbit class counts (based on first valid-step altitude)"):
    st.dataframe(counts.rename("count").to_frame(), use_container_width=True)

# -----------------------------
# COARSE CONJUNCTION DETECTION WITH GRID BINNING
# -----------------------------
def bin_key(vec: np.ndarray, bin_km: int) -> Optional[Tuple[int, int, int]]:
    """
    Map a 3D ECI point to a grid bin. Returns None if vec is invalid (NaN/inf/wrong shape).
    """
    if vec is None:
        return None
    vec = np.asarray(vec)
    if vec.shape != (3,):
        return None
    if not np.isfinite(vec).all():
        return None
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return (int(math.floor(x / bin_km)), int(math.floor(y / bin_km)), int(math.floor(z / bin_km)))


def neighbor_bins(k: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    x, y, z = k
    out: List[Tuple[int, int, int]] = []
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dz in (-1, 0, 1):
                out.append((x + dx, y + dy, z + dz))
    return out


st.subheader("🔍 Coarse→Refined Conjunction Detection")

st.caption(
    "Step 1 (coarse): grid-binning reduces comparisons. "
    "Step 2 (refined): for candidate pairs, re-simulate with finer time steps around closest coarse moment."
)

# Build a 3D array [time, obj, 3] with aligned time axis
min_T = len(coarse_times)
names = names[:]  # copy
pos_cube = np.stack([sat_pos_coarse[n][:min_T] for n in names], axis=1)  # (T, N, 3)

candidate_pairs: Dict[Tuple[int, int], Tuple[float, int]] = {}  # (i,j)->(best_coarse_dist, best_t_index)

with st.spinner("Running coarse scan (grid-binning)..."):
    for t_idx in range(min_T):
        pts = pos_cube[t_idx]  # (N,3)

        # Bin objects (skip invalid points)
        bins: Dict[Tuple[int, int, int], List[int]] = {}
        for i in range(len(names)):
            k = bin_key(pts[i], bin_size_km)
            if k is None:
                continue
            bins.setdefault(k, []).append(i)

        compared = 0
        for bk, _idxs in bins.items():
            neighbor_idxs: List[int] = []
            for nb in neighbor_bins(bk):
                if nb in bins:
                    neighbor_idxs.extend(bins[nb])

            if len(neighbor_idxs) < 2:
                continue
            neighbor_idxs = sorted(set(neighbor_idxs))

            for a_pos in range(len(neighbor_idxs)):
                ia = neighbor_idxs[a_pos]
                pa = pts[ia]
                if not np.isfinite(pa).all():
                    continue

                for b_pos in range(a_pos + 1, len(neighbor_idxs)):
                    ib = neighbor_idxs[b_pos]
                    pb = pts[ib]
                    if not np.isfinite(pb).all():
                        continue

                    compared += 1
                    if compared > max_pairs_per_step:
                        break

                    d = float(np.linalg.norm(pa - pb))
                    if d <= coarse_candidate_km:
                        key = (ia, ib)
                        prev = candidate_pairs.get(key)
                        if (prev is None) or (d < prev[0]):
                            candidate_pairs[key] = (d, t_idx)

                if compared > max_pairs_per_step:
                    break
            if compared > max_pairs_per_step:
                break

# Limit candidate pairs to refine
cand_items = sorted(candidate_pairs.items(), key=lambda kv: kv[1][0])[:max_total_candidates]

st.write(f"Candidate pairs found (coarse ≤ {coarse_candidate_km} km): **{len(candidate_pairs)}**")
st.write(f"Candidate pairs refined (cap {max_total_candidates}): **{len(cand_items)}**")

# -----------------------------
# REFINED CHECK FOR CANDIDATES
# -----------------------------
def refine_pair(
    tle_a: TLEItem,
    tle_b: TLEItem,
    center_time: datetime,
    refine_step_min: int,
    window_hours: int,
) -> float:
    """
    Re-simulate A and B with finer steps around center_time.
    Return min distance (km) in that refined window.
    """
    satA = Satrec.twoline2rv(tle_a.line1, tle_a.line2)
    satB = Satrec.twoline2rv(tle_b.line1, tle_b.line2)

    start = center_time - timedelta(hours=window_hours)
    end = center_time + timedelta(hours=window_hours)

    total_minutes = int((end - start).total_seconds() / 60)
    steps = max(2, total_minutes // refine_step_min)
    times = [start + timedelta(minutes=k * refine_step_min) for k in range(steps + 1)]

    pa = sgp4_positions(satA, times)
    pb = sgp4_positions(satB, times)

    valid = np.isfinite(pa).all(axis=1) & np.isfinite(pb).all(axis=1)
    if not np.any(valid):
        return float("inf")

    d = np.linalg.norm(pa[valid] - pb[valid], axis=1)
    return float(np.min(d))


# Map name->TLEItem for refinement
tle_by_name = {it.name: it for it in tle_items}

refined_events: List[Tuple[str, str, float, float]] = []  # (A,B, refined_min_km, coarse_min_km)

with st.spinner("Refining candidate pairs (finer time steps)..."):
    for (i, j), (coarse_d, t_best) in cand_items:
        A = names[i]
        B = names[j]

        if A not in tle_by_name or B not in tle_by_name:
            continue

        center_time = t0 + timedelta(minutes=t_best * coarse_step_minutes)

        d_ref = refine_pair(
            tle_by_name[A],
            tle_by_name[B],
            center_time=center_time,
            refine_step_min=refine_minutes,
            window_hours=refine_window_hours,
        )

        if d_ref <= final_threshold_km:
            refined_events.append((A, B, d_ref, coarse_d))

# Results table
st.subheader("✅ High-Risk Conjunction Events")

if refined_events:
    df_events = pd.DataFrame(
        refined_events,
        columns=["Object 1", "Object 2", "Refined Min Sep (km)", "Coarse Min Sep (km)"],
    ).sort_values("Refined Min Sep (km)").reset_index(drop=True)

    st.dataframe(df_events, use_container_width=True)

    st.download_button(
        "Download conjunction events CSV",
        data=df_events.to_csv(index=False).encode("utf-8"),
        file_name="conjunction_events_refined.csv",
        mime="text/csv",
    )
else:
    st.info("No refined conjunctions below the chosen high-risk threshold. Try:")
    st.markdown("- Increase objects\n- Increase days\n- Increase candidate threshold\n- Use 'All' orbit class\n- Upload a larger TLE set")

# -----------------------------
# 3D PLOTLY VISUALIZATION
# -----------------------------
st.subheader("🧊 3D Interactive Orbit Viewer (Plotly)")

plot_names = names[:min(plot_tracks, len(names))]

T_plot = min(min_T, plot_points_cap)
idxs = np.linspace(0, min_T - 1, T_plot).astype(int)

fig = go.Figure()

highlight_pair: Set[str] = set()
if refined_events:
    highlight_pair = {refined_events[0][0], refined_events[0][1]}

for nm in plot_names:
    track = sat_pos_coarse[nm][:min_T][idxs]
    valid = np.isfinite(track).all(axis=1)
    track = track[valid]
    if len(track) < 2:
        continue

    is_highlight = nm in highlight_pair
    fig.add_trace(
        go.Scatter3d(
            x=track[:, 0],
            y=track[:, 1],
            z=track[:, 2],
            mode="lines",
            name=nm if is_highlight else None,
            showlegend=is_highlight,
            line=dict(width=5 if is_highlight else 2),
            opacity=1.0 if is_highlight else 0.5,
        )
    )

# Add Earth sphere
u = np.linspace(0, 2 * np.pi, 50)
v = np.linspace(0, np.pi, 25)
xs = EARTH_RADIUS_KM * np.outer(np.cos(u), np.sin(v))
ys = EARTH_RADIUS_KM * np.outer(np.sin(u), np.sin(v))
zs = EARTH_RADIUS_KM * np.outer(np.ones_like(u), np.cos(v))

fig.add_trace(
    go.Surface(
        x=xs, y=ys, z=zs,
        showscale=False,
        opacity=0.2,
        name="Earth",
        hoverinfo="skip"
    )
)

fig.update_layout(
    scene=dict(
        xaxis_title="X (km)",
        yaxis_title="Y (km)",
        zaxis_title="Z (km)",
        aspectmode="data",
    ),
    margin=dict(l=0, r=0, t=30, b=0),
)

st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# SUMMARY + EXTENSIONS (for your poster)
# -----------------------------
st.subheader("📘 What you upgraded (for your project report)")

st.markdown(
    """
**1) More time steps (accuracy):**  
You used a **coarse scan** (fast) and then a **refined scan** (accurate) only for likely-close pairs.

**2) Orbit class filtering (LEO/MEO/GEO):**  
You classify objects by altitude estimate and can focus analysis on one orbit region.

**3) 3D interactive Plotly:**  
A 3D viewer makes orbit paths and crowding easy to understand for judges.

**4) Faster scaling:**  
Grid-binning spatial indexing cuts comparisons drastically compared to checking every pair (N²).
"""
)
