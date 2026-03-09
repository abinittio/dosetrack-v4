"""
DoseTrack V4 — Minimal cumulative dose tracker with named accounts.

Same PK/PD engine as V3. Interface: name → date/time/mg → curve.
Data persisted in SQLite (doses.db) keyed by username.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import sqlite3
import streamlit as st
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, date as date_type, time as time_type

from dosetrack import simulate, Dose

# ── Database ──────────────────────────────────────────────────────────────

DB_PATH = os.path.join(os.path.dirname(__file__), "doses.db")


def db_init():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS doses (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT    NOT NULL,
            dt       TEXT    NOT NULL,
            mg       REAL    NOT NULL
        )
    """)
    con.commit()
    con.close()


def db_load(username: str) -> list[dict]:
    con = sqlite3.connect(DB_PATH)
    rows = con.execute(
        "SELECT id, dt, mg FROM doses WHERE username=? ORDER BY dt",
        (username,)
    ).fetchall()
    con.close()
    return [{"id": r[0], "dt": datetime.fromisoformat(r[1]), "mg": r[2]} for r in rows]


def db_insert(username: str, dt: datetime, mg: float) -> int:
    con = sqlite3.connect(DB_PATH)
    cur = con.execute(
        "INSERT INTO doses (username, dt, mg) VALUES (?, ?, ?)",
        (username, dt.isoformat(), mg)
    )
    row_id = cur.lastrowid
    con.commit()
    con.close()
    return row_id


def db_delete(row_id: int):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM doses WHERE id=?", (row_id,))
    con.commit()
    con.close()


def db_clear(username: str):
    con = sqlite3.connect(DB_PATH)
    con.execute("DELETE FROM doses WHERE username=?", (username,))
    con.commit()
    con.close()


db_init()

# ── Page config ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="DoseTrack",
    page_icon="💊",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, .stApp {
    background: #08111a !important;
    color: #e2e8f0;
    font-family: 'Inter', -apple-system, sans-serif;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 2.5rem; padding-bottom: 2rem; max-width: 860px; }

h1 {
    color: #f0fdfa !important;
    font-size: 1.6rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0 !important;
}

.subtitle {
    font-size: 0.78rem;
    color: #334155;
    font-weight: 400;
    margin-bottom: 1.6rem;
}

.stNumberInput input, .stTextInput input, .stDateInput input, .stTimeInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    color: #e2e8f0 !important;
    border-radius: 10px !important;
    font-size: 0.95rem !important;
}
.stNumberInput input:focus, .stDateInput input:focus,
.stTimeInput input:focus, .stTextInput input:focus {
    border-color: rgba(20,184,166,0.4) !important;
    box-shadow: 0 0 0 2px rgba(20,184,166,0.08) !important;
}

label { color: #64748b !important; font-size: 0.82rem !important; }
.stMarkdown p { color: #94a3b8 !important; }

.stButton > button {
    background: rgba(15,118,110,0.85) !important;
    color: #f0fdfa !important;
    border: 1px solid rgba(20,184,166,0.25) !important;
    border-radius: 10px;
    font-weight: 700;
    font-size: 1rem;
    width: 100%;
    padding: 6px 0;
    transition: all 0.18s ease;
}
.stButton > button:hover {
    background: rgba(20,184,166,0.9) !important;
    box-shadow: 0 0 18px rgba(20,184,166,0.15);
}

.account-box {
    background: rgba(20,184,166,0.06);
    border: 1px solid rgba(20,184,166,0.12);
    border-radius: 14px;
    padding: 20px 24px;
    margin-bottom: 24px;
}
.account-name {
    font-size: 1rem;
    font-weight: 700;
    color: #5eead4;
}
.account-sub {
    font-size: 0.78rem;
    color: #334155;
}

.dose-row {
    display: flex;
    align-items: center;
    padding: 10px 16px;
    background: rgba(255,255,255,0.025);
    border-radius: 10px;
    margin-bottom: 6px;
    border: 1px solid rgba(255,255,255,0.05);
    font-size: 0.9rem;
    color: #cbd5e1;
}
.dose-date { color: #64748b; font-size: 0.8rem; min-width: 100px; }
.dose-time { color: #94a3b8; margin: 0 12px; }
.dose-mg   { font-weight: 700; color: #5eead4; margin-left: auto; margin-right: 8px; }

hr { border-color: rgba(255,255,255,0.05) !important; }

div[data-testid="stMetricValue"] {
    color: #5eead4 !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────

if "username" not in st.session_state:
    st.session_state.username = ""


# ── Header ────────────────────────────────────────────────────────────────

st.markdown("# DoseTrack")
st.markdown('<div class="subtitle">Log doses. See the curve.</div>', unsafe_allow_html=True)


# ── Account login ─────────────────────────────────────────────────────────

if not st.session_state.username:
    st.markdown("**Enter your name to get started:**")
    col_name, col_go = st.columns([4, 1])
    name_input = col_name.text_input("Name", placeholder="e.g. Nabil", label_visibility="collapsed")
    with col_go:
        st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
        if st.button("Go →", key="login_btn"):
            if name_input.strip():
                st.session_state.username = name_input.strip().lower()
                st.rerun()
            else:
                st.warning("Enter a name first.")
    st.stop()


# ── Logged in ─────────────────────────────────────────────────────────────

username = st.session_state.username

col_acct, col_logout = st.columns([5, 1])
with col_acct:
    st.markdown(f"""
    <div class="account-box">
        <div class="account-name">{username.title()}</div>
        <div class="account-sub">Your doses are saved automatically.</div>
    </div>
    """, unsafe_allow_html=True)
with col_logout:
    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    if st.button("Switch", key="logout_btn"):
        st.session_state.username = ""
        st.rerun()


# ── Load doses from DB ────────────────────────────────────────────────────

doses = db_load(username)


# ── Input row ─────────────────────────────────────────────────────────────

c1, c2, c3, c4 = st.columns([2.2, 1.8, 1.8, 1])

with c1:
    input_date = st.date_input("Date", value=date_type.today(), label_visibility="visible")
with c2:
    input_time = st.time_input("Time", value=time_type(8, 0), label_visibility="visible")
with c3:
    input_mg = st.number_input("Dose (mg)", min_value=5.0, max_value=120.0,
                                value=50.0, step=5.0, label_visibility="visible")
with c4:
    st.markdown("<div style='height:26px'></div>", unsafe_allow_html=True)
    if st.button("＋", key="add_dose"):
        dt = datetime.combine(input_date, input_time)
        db_insert(username, dt, float(input_mg))
        st.rerun()


# ── No doses state ────────────────────────────────────────────────────────

if not doses:
    st.markdown("""
    <div style="text-align:center; padding:70px 0 50px 0; color:#1e3a34;">
        <div style="font-size:2.5rem; margin-bottom:10px;">💊</div>
        <div style="font-size:0.88rem; font-weight:500; color:#334155;">
            Log a dose above to see the simulation curve.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()


# ── Simulation ────────────────────────────────────────────────────────────

t0 = doses[0]["dt"]

dose_objects = [
    Dose(time_h=max((d["dt"] - t0).total_seconds() / 3600.0, 0.0), amount_mg=d["mg"])
    for d in doses
]

last_h = max(d.time_h for d in dose_objects)
sim_end = max(last_h + 28.0, 30.0)

try:
    result = simulate(
        doses=dose_objects,
        weight_kg=70.0,
        model="prodrug",
        t_span=(0.0, sim_end),
    )
except Exception as e:
    st.error(f"Simulation error: {e}")
    st.stop()


# ── Plot ──────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(9, 3.8))
fig.patch.set_facecolor("#08111a")
ax.set_facecolor("#08111a")

# Zone bands
ax.axhspan(85, 100, alpha=0.06, color="#ef4444", zorder=1)
ax.axhspan(65,  85, alpha=0.07, color="#059669", zorder=1)
ax.axhspan(40,  65, alpha=0.07, color="#10b981", zorder=1)
ax.axhspan(15,  40, alpha=0.05, color="#f59e0b", zorder=1)

# Threshold
ax.axhline(40, color="#1e3a34", linewidth=1.0, linestyle="--", zorder=2)

# Dose markers
for d in doses:
    h = (d["dt"] - t0).total_seconds() / 3600.0
    ax.axvline(h, color="#334155", linewidth=1, linestyle=":", zorder=2)
    ax.text(h + 0.15, 97, f"{d['mg']:.0f}mg", color="#475569",
            fontsize=7.5, va="top", ha="left", zorder=4)

# Effect curve
ax.fill_between(result.t, result.effect_pct, alpha=0.12, color="#14b8a6", zorder=3)
ax.plot(result.t, result.effect_pct, color="#5eead4", linewidth=2.2, zorder=4)

# Zone labels
for y, label, col in [
    (92, "SUPRA", "#ef4444"),
    (75, "PEAK", "#059669"),
    (52, "THERAPEUTIC", "#14b8a6"),
    (27, "SUB", "#f59e0b"),
]:
    ax.text(sim_end * 0.995, y, label, color=col, fontsize=6.5,
            ha="right", va="center", alpha=0.5, zorder=3)

# X axis
n_ticks = min(10, int(sim_end / 4) + 1)
tick_h = np.linspace(0, sim_end, n_ticks)
tick_labels = [(t0 + timedelta(hours=float(h))).strftime("%a %H:%M") for h in tick_h]
ax.set_xticks(tick_h)
ax.set_xticklabels(tick_labels, rotation=30, ha="right", fontsize=7.5, color="#334155")

ax.set_ylim(0, 100)
ax.set_yticks([0, 40, 65, 85, 100])
ax.set_yticklabels(["0%", "40%", "65%", "85%", "100%"], fontsize=7.5, color="#334155")
ax.set_xlim(0, sim_end)

for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(colors="#334155", length=0)
ax.grid(axis="y", color="#111f2e", linewidth=0.8, zorder=0)

fig.tight_layout(pad=1.2)
st.pyplot(fig, use_container_width=True)
plt.close(fig)


# ── Stats row ─────────────────────────────────────────────────────────────

now_h = (datetime.now() - t0).total_seconds() / 3600.0
col_a, col_b, col_c = st.columns(3)

if 0 <= now_h <= sim_end:
    idx = int(np.argmin(np.abs(result.t - now_h)))
    col_a.metric("Effect now", f"{float(result.effect_pct[idx]):.0f}%")
    col_b.metric("DA stores", f"{float(result.da_stores[idx]) * 100:.0f}%")
else:
    col_a.metric("Effect now", "—")
    col_b.metric("DA stores", "—")

col_c.metric("Peak effect", f"{result.peak_effect:.0f}%")


# ── Dose log ──────────────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<div style='font-size:0.72rem; color:#334155; font-weight:600; "
    "text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>"
    "Logged Doses</div>",
    unsafe_allow_html=True,
)

for d in doses:
    col_label, col_btn = st.columns([7, 1])
    col_label.markdown(
        f"<div class='dose-row'>"
        f"<span class='dose-date'>{d['dt'].strftime('%a %d %b')}</span>"
        f"<span class='dose-time'>{d['dt'].strftime('%H:%M')}</span>"
        f"<span class='dose-mg'>{d['mg']:.0f} mg</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    if col_btn.button("✕", key=f"rm_{d['id']}"):
        db_delete(d["id"])
        st.rerun()

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
if st.button("Clear all doses", key="clear_all"):
    db_clear(username)
    st.rerun()
