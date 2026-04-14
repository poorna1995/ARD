"""
dashboard.py — Task Complexity Estimator · Full Analysis Dashboard
==================================================================
Run:
    streamlit run dashboard.py

Dataset paths are resolved relative to this file OR upload .parquet files directly.
"""

from __future__ import annotations
import sys, os, math, io, time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
for candidate in [_HERE / "src", _HERE / "complexity_tce" / "src", _HERE]:
    if (candidate / "complexity").exists():
        sys.path.insert(0, str(candidate))
        break

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(
    page_title="TCE · Benchmark Analysis",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Design tokens
# ─────────────────────────────────────────────────────────────────────────────
BG, BG2, CARD, BORDER = "#0b0d14", "#11131c", "#171a26", "#222638"
ACCENT, ACCENT2       = "#5b8dee", "#8b5cf6"
GREEN, YELLOW, RED    = "#10d9a0", "#f59e0b", "#f43f5e"
TXT, TXT2, GRID       = "#dde3f0", "#68748a", "#1c2035"

DATASET_COLORS = {
    "musique":   "#5b8dee",
    "mmlu_pro":  "#10d9a0",
    "aime":      "#f59e0b",
    "gaia":      "#8b5cf6",
    "swe_bench": "#f43f5e",
}
DEFAULT_DS_COLOR = "#aabbcc"

DIM_COLORS = {"S": "#5b8dee", "R": "#8b5cf6", "T": "#10d9a0", "D": "#f59e0b", "TT": "#f43f5e"}
DIM_LABELS = {"S": "Surface", "R": "Reasoning", "T": "Tool", "D": "Domain", "TT": "Task Type"}
DIMS       = ["S", "R", "T", "D", "TT"]

DEFAULT_W  = {"S": 0.15, "R": 0.35, "T": 0.20, "D": 0.15, "TT": 0.15}

EXPECTED_RANK = {"musique": 1, "mmlu_pro": 2, "aime": 3, "gaia": 4, "swe_bench": 5}

QUERY_COL_MAP = {
    "gaia": "query", "aime": "query", "mmlu_pro": "query",
    "musique": "query", "swe_bench": "query",
}

PL = dict(
    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="'JetBrains Mono', 'Fira Code', monospace", color=TXT, size=11),
    margin=dict(l=12, r=12, t=38, b=12),
    xaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
    yaxis=dict(gridcolor=GRID, zerolinecolor=GRID),
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');
html, body, [class*="css"] {{
    background-color:{BG}; color:{TXT};
    font-family:'JetBrains Mono',monospace;
}}
section[data-testid="stSidebar"] {{
    background:{BG2}; border-right:1px solid {BORDER};
}}
.stTabs [data-baseweb="tab-list"] {{
    gap:3px; background:{CARD}; border-radius:10px;
    padding:4px; border:1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    background:transparent; color:{TXT2}; border-radius:7px;
    font-size:0.75rem; letter-spacing:0.06em; padding:6px 14px; border:none;
    font-family:'JetBrains Mono',monospace;
}}
.stTabs [aria-selected="true"] {{background:{ACCENT}!important; color:white!important;}}
.kcard {{
    background:{CARD}; border:1px solid {BORDER}; border-radius:12px;
    padding:18px 20px; margin-bottom:10px;
}}
.kcard-hi  {{ border-left:3px solid {ACCENT}; }}
.kcard-red {{ border-left:3px solid {RED}; }}
.kcard-grn {{ border-left:3px solid {GREEN}; }}
.ds-badge {{
    display:inline-block; padding:2px 10px; border-radius:20px;
    font-size:0.68rem; font-weight:600; letter-spacing:0.1em;
    text-transform:uppercase; margin:2px;
}}
.pg-title {{
    font-family:'Space Grotesk',sans-serif; font-size:2rem;
    font-weight:700; letter-spacing:-0.5px; margin-bottom:2px;
}}
.pg-sub {{
    font-size:0.72rem; color:{TXT2}; letter-spacing:0.1em;
    text-transform:uppercase; margin-bottom:22px;
}}
.sec {{
    font-size:0.63rem; letter-spacing:0.14em; text-transform:uppercase;
    color:{TXT2}; margin:14px 0 7px;
}}
div[data-testid="stMetricValue"] {{
    font-family:'Space Grotesk',sans-serif; font-size:1.7rem; color:{ACCENT};
}}
div.stButton>button {{
    background:{ACCENT}; color:white; border:none; border-radius:7px;
    font-family:'JetBrains Mono',monospace; font-size:0.78rem; padding:7px 18px;
}}
div.stButton>button:hover {{ background:#4a7cd8; }}
.sbar-wrap {{ margin:3px 0; }}
.sbar-row  {{ display:flex; align-items:center; gap:10px; margin-bottom:5px; }}
.sbar-lbl  {{ font-size:0.7rem; color:{TXT2}; width:80px; flex-shrink:0; }}
.sbar-track {{ flex:1; height:7px; background:{BG2}; border-radius:4px; overflow:hidden; }}
.sbar-fill  {{ height:100%; border-radius:4px; }}
.sbar-val   {{ font-size:0.7rem; color:{TXT}; width:38px; text-align:right; flex-shrink:0; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────
BAND_COLOR = {"LOW": GREEN, "MEDIUM": ACCENT, "HIGH": YELLOW, "VERY HIGH": RED}


def assign_band_vec(c_arr, t_low: float, t_med: float, t_high: float) -> np.ndarray:
    c = np.asarray(c_arr, dtype=float)
    return np.where(c < t_low,  "LOW",
           np.where(c < t_med,  "MEDIUM",
           np.where(c < t_high, "HIGH", "VERY HIGH")))


def ds_color(name: str) -> str:
    return DATASET_COLORS.get(name, DEFAULT_DS_COLOR)


def hex_rgba(hex_col: str, alpha: float = 0.25) -> str:
    r, g, b = int(hex_col[1:3], 16), int(hex_col[3:5], 16), int(hex_col[5:7], 16)
    return f"rgba({r},{g},{b},{alpha})"


def score_bar_html(dims: dict) -> str:
    html = '<div class="sbar-wrap">'
    for dim, (val, color) in dims.items():
        pct = int(np.clip(val, 0, 1) * 100)
        html += (f'<div class="sbar-row">'
                 f'<span class="sbar-lbl">{dim}</span>'
                 f'<div class="sbar-track">'
                 f'<div class="sbar-fill" style="width:{pct}%;background:{color}"></div>'
                 f'</div>'
                 f'<span class="sbar-val">{val:.3f}</span>'
                 f'</div>')
    html += "</div>"
    return html


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — Expensive: run estimator once, cache sub-dim scores + timing
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False, ttl=7200)
def load_and_score_subdims(
    base_path_str: str,
    uploaded_bytes: dict | None,
) -> tuple[dict, dict]:
    """
    Scores S, R, T, D, TT for every query in every dataset.
    Also records wall-clock timing at the load and score stage.

    Returns
    -------
    results : dict[name -> DataFrame | None]
        Sub-dimension scores merged with the original dataframe.

    timing_store : dict with keys
        "total_sec"      float   — wall time for the whole batch
        "total_queries"  int     — queries processed across all datasets
        "overall_qps"    float   — queries per second (scoring only)
        "datasets"       dict[name -> {
                             "n_queries"    : int,
                             "load_sec"     : float,
                             "score_sec"    : float,
                             "total_sec"    : float,
                             "qps"          : float,
                             "ms_per_query" : float,
                             "error"        : str | None,
                         }]
    """
    try:
        from src.complexity import TaskComplexityEstimator
    except ImportError as e:
        return {"__error__": str(e)}, {}

    est = TaskComplexityEstimator(weights=DEFAULT_W)

    results:      dict = {}
    timing_store: dict = {
        "datasets":       {},
        "total_sec":      0.0,
        "total_queries":  0,
        "overall_qps":    0.0,
    }

    # ── Build source map ──────────────────────────────────────────────────────
    sources: dict = {}
    if uploaded_bytes:
        sources = {name: ("upload", raw) for name, raw in uploaded_bytes.items()}
    else:
        base = Path(base_path_str)
        for name in QUERY_COL_MAP:
            p = base / f"{name}.parquet"
            if p.exists():
                sources[name] = ("disk", str(p))

    batch_start = time.perf_counter()

    # ── Per-dataset loop ──────────────────────────────────────────────────────
    for name, (src_type, src) in sources.items():
        ds_timing: dict = {"error": None}
        try:
            # Load -----------------------------------------------------------
            t0 = time.perf_counter()
            df = pd.read_parquet(
                io.BytesIO(src) if src_type == "upload" else src
            )
            ds_timing["load_sec"] = round(time.perf_counter() - t0, 4)

            # Detect query column --------------------------------------------
            qcol = next(
                (c for c in df.columns
                 if any(k in c.lower()
                        for k in ["query", "question", "text", "prompt"])),
                None,
            )
            if qcol is None:
                raise ValueError("No query/question/text/prompt column found")

            queries              = df[qcol].fillna("").astype(str).tolist()
            n_q                  = len(queries)
            ds_timing["n_queries"] = n_q

            # Score ----------------------------------------------------------
            t0     = time.perf_counter()
            scored = est.score_batch(queries)
            ds_timing["score_sec"] = round(time.perf_counter() - t0, 4)

            keep   = [c for c in ["S","R","T","D","TT","task_type","bloom_level",
                                   "tool_categories","domains"]
                      if c in scored.columns]
            merged = pd.concat(
                [df.reset_index(drop=True), scored[keep].reset_index(drop=True)],
                axis=1,
            )
            merged["_dataset"] = name
            merged["_qcol"]    = qcol
            results[name]      = merged

        except Exception as exc:
            results[name]            = None
            ds_timing["error"]       = str(exc)
            ds_timing.setdefault("n_queries", 0)
            ds_timing.setdefault("load_sec",  0.0)
            ds_timing.setdefault("score_sec", 0.0)

        # Derived per-dataset stats ------------------------------------------
        total_ds = ds_timing.get("load_sec", 0) + ds_timing.get("score_sec", 0)
        n_q      = ds_timing.get("n_queries", 0)
        sc       = ds_timing.get("score_sec", 0)
        ds_timing["total_sec"]    = round(total_ds, 4)
        ds_timing["qps"]          = round(n_q / sc, 1)          if sc > 0 else 0.0
        ds_timing["ms_per_query"] = round(sc / n_q * 1000, 3)   if n_q > 0 else 0.0
        timing_store["datasets"][name] = ds_timing

    # ── Batch totals ──────────────────────────────────────────────────────────
    total_sec  = round(time.perf_counter() - batch_start, 4)
    total_q    = sum(v.get("n_queries", 0) for v in timing_store["datasets"].values())
    total_sc   = sum(v.get("score_sec",  0) for v in timing_store["datasets"].values())

    timing_store["total_sec"]     = total_sec
    timing_store["total_queries"] = total_q
    timing_store["overall_qps"]   = round(total_q / total_sc, 1) if total_sc > 0 else 0.0

    return results, timing_store


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — Instant: recompute C from cached sub-dim scores + new weights
# ─────────────────────────────────────────────────────────────────────────────
def apply_weights(
    raw_loaded: dict,
    norm_w: dict,
    t_low: float,
    t_med: float,
    t_high: float,
) -> dict:
    out = {}
    for name, df in raw_loaded.items():
        d         = df.copy()
        d["C"]    = sum(norm_w[dim] * d[dim] for dim in DIMS)
        d["band"] = assign_band_vec(d["C"].values, t_low, t_med, t_high)
        out[name] = d
    return out


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="pg-title" style="font-size:1.4rem">⬡ TCE</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="pg-sub">Task Complexity Estimator</div>',
                unsafe_allow_html=True)

    # ── Data source ───────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Data Source</div>', unsafe_allow_html=True)
    source_mode = st.radio("Source", ["📁 Disk path", "⬆ Upload .parquet"],
                           horizontal=True, label_visibility="collapsed")

    uploaded_bytes = None
    base_path      = ""

    if source_mode == "⬆ Upload .parquet":
        up_files = st.file_uploader(
            "Drop one or more .parquet files",
            type=["parquet"], accept_multiple_files=True,
        )
        if up_files:
            uploaded_bytes = {f.name.replace(".parquet", ""): f.read() for f in up_files}
            st.success(f"✓ {len(uploaded_bytes)} file(s) ready")
    else:
        candidates = [
            _HERE / "datasets" / "processed",
            _HERE.parent / "datasets" / "processed",
            _HERE / "data" / "processed",
            Path.home() / "Downloads" / "research" / "datasets" / "processed",
        ]
        default_path = next((str(p) for p in candidates if p.exists()),
                            str(_HERE.parent / "datasets" / "processed"))
        base_path    = st.text_input("Path", value=default_path,
                                     label_visibility="collapsed")
        bpath        = Path(base_path)
        found_n      = sum((bpath / f"{n}.parquet").exists() for n in QUERY_COL_MAP)
        st.caption(f"{found_n} / {len(QUERY_COL_MAP)} parquet files found at path")

    # ── Dimension weights ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">Dimension Weights</div>', unsafe_allow_html=True)
    custom_w = {}
    for dim, label in DIM_LABELS.items():
        custom_w[dim] = st.slider(f"{dim} · {label}", 0.0, 1.0,
                                  DEFAULT_W[dim], 0.01, key=f"w_{dim}")

    total_w = sum(custom_w.values()) or 1.0
    norm_w  = {k: v / total_w for k, v in custom_w.items()}
    if abs(total_w - 1.0) > 0.01:
        st.caption(f"↑ auto-normalised (sum={total_w:.2f})")

    if st.button("↺  Reset weights"):
        for dim in DIM_LABELS:
            st.session_state[f"w_{dim}"] = DEFAULT_W[dim]
        st.rerun()

    # ── Band thresholds ───────────────────────────────────────────────────────
    st.markdown('<div class="sec">Band Thresholds</div>', unsafe_allow_html=True)

    low_range  = st.slider("🟢 LOW",    0.00, 1.00, (0.00, 0.20), 0.01, key="b_low")
    t_low      = low_range[1]

    med_range  = st.slider("🔵 MEDIUM", 0.00, 1.00,
                           (t_low, min(max(round(t_low + 0.25, 2), 0.45), 0.99)),
                           0.01, key="b_med")
    t_med      = max(med_range[1], t_low + 0.01)

    hi_range   = st.slider("🟡 HIGH",   0.00, 1.00,
                           (t_med, min(max(round(t_med + 0.15, 2), 0.60), 0.99)),
                           0.01, key="b_high")
    t_high     = max(hi_range[1], t_med + 0.01)

    st.markdown(
        f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0 10px">'
        f'<span style="font-size:0.72rem;color:{RED}">🔴 VERY HIGH</span>'
        f'<span style="font-size:0.68rem;color:{TXT2}">{t_high:.2f} → 1.00 (auto)</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    w_low = t_low * 100
    w_med = (t_med  - t_low)  * 100
    w_hi  = (t_high - t_med)  * 100
    w_vhi = (1.0    - t_high) * 100
    st.markdown(
        f'<div style="border-radius:6px;overflow:hidden;height:14px;display:flex;'
        f'margin-bottom:4px;border:1px solid {BORDER}">'
        f'<div style="width:{w_low:.1f}%;background:{GREEN};display:flex;align-items:center;'
        f'justify-content:center;font-size:0.55rem;color:#000;font-weight:700;overflow:hidden">'
        f'{"L" if w_low > 6 else ""}</div>'
        f'<div style="width:{w_med:.1f}%;background:{ACCENT};display:flex;align-items:center;'
        f'justify-content:center;font-size:0.55rem;color:#fff;font-weight:700;overflow:hidden">'
        f'{"M" if w_med > 6 else ""}</div>'
        f'<div style="width:{w_hi:.1f}%;background:{YELLOW};display:flex;align-items:center;'
        f'justify-content:center;font-size:0.55rem;color:#000;font-weight:700;overflow:hidden">'
        f'{"H" if w_hi > 6 else ""}</div>'
        f'<div style="width:{w_vhi:.1f}%;background:{RED};display:flex;align-items:center;'
        f'justify-content:center;font-size:0.55rem;color:#fff;font-weight:700;overflow:hidden">'
        f'{"VH" if w_vhi > 6 else ""}</div>'
        f'</div>'
        f'<div style="display:flex;justify-content:space-between;font-size:0.62rem;'
        f'color:{TXT2};margin-bottom:8px">'
        f'<span>0.00</span><span>{t_low:.2f}</span>'
        f'<span>{t_med:.2f}</span><span>{t_high:.2f}</span><span>1.00</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if st.button("↺  Reset bands"):
        for key, val in [("b_low", (0.00, 0.20)), ("b_med", (0.20, 0.45)),
                         ("b_high", (0.45, 0.60))]:
            st.session_state[key] = val
        st.rerun()

    # ── Weight radar ──────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Weight Allocation</div>', unsafe_allow_html=True)
    vals_r = list(norm_w.values())
    fig_r  = go.Figure(go.Scatterpolar(
        r=vals_r + [vals_r[0]],
        theta=list(DIM_LABELS.values()) + [list(DIM_LABELS.values())[0]],
        fill="toself", fillcolor=hex_rgba(ACCENT, 0.2),
        line=dict(color=ACCENT, width=2), marker=dict(size=5, color=ACCENT),
    ))
    fig_r.update_layout(**{**PL, "height": 200,
        "polar": dict(
            bgcolor=BG2,
            radialaxis=dict(range=[0, max(vals_r) * 1.35 + 0.02],
                            gridcolor=BORDER, tickfont=dict(size=7, color=TXT2),
                            linecolor=BORDER),
            angularaxis=dict(gridcolor=BORDER, linecolor=BORDER,
                             tickfont=dict(size=9, color=TXT)),
        ),
        "margin": dict(l=10, r=10, t=10, b=10),
    })
    st.plotly_chart(fig_r, use_container_width=True, config={"displayModeBar": False})


# ─────────────────────────────────────────────────────────────────────────────
# Load data (expensive once, timing captured inside the cached function)
# ─────────────────────────────────────────────────────────────────────────────
if not uploaded_bytes and not base_path:
    st.warning("Set a dataset path or upload .parquet files in the sidebar.")
    st.stop()

with st.spinner("Scoring sub-dimensions… cached after first run ⚡"):
    raw_data, timing_store = load_and_score_subdims(base_path, uploaded_bytes)

if "__error__" in raw_data:
    st.error(f"Import error: {raw_data['__error__']}")
    st.stop()

raw_loaded = {k: v for k, v in raw_data.items() if v is not None}
missing    = [k for k, v in raw_data.items() if v is None]

if not raw_loaded:
    st.error("No datasets loaded. Check path or upload .parquet files.")
    st.stop()

loaded     = apply_weights(raw_loaded, norm_w, t_low, t_med, t_high)
def_loaded = apply_weights(raw_loaded, DEFAULT_W, t_low, t_med, t_high)
all_df     = pd.concat(loaded.values(), ignore_index=True)

if missing:
    st.warning(f"Could not load: {', '.join(missing)}")

# ─────────────────────────────────────────────────────────────────────────────
# Page header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="pg-title">Benchmark Complexity Analysis</div>',
            unsafe_allow_html=True)
badges = " ".join(
    f'<span class="ds-badge" style="background:{hex_rgba(ds_color(n),0.13)};'
    f'color:{ds_color(n)};border:1px solid {hex_rgba(ds_color(n),0.4)}">{n}</span>'
    for n in loaded
)
st.markdown(
    f'<div class="pg-sub">{badges} &nbsp;·&nbsp; {len(all_df):,} queries scored'
    f' &nbsp;·&nbsp; scored in <b>{timing_store.get("total_sec", 0):.2f}s</b></div>',
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
(t_overview, t_dist, t_dims_tab,
 t_bands, t_corr, t_explorer, t_timing) = st.tabs([
    "◈ Overview",
    "◈ Distributions",
    "◈ Dimensions",
    "◈ Band Analysis",
    "◈ Correlations",
    "◈ Query Explorer",
    "◈ Timing",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with t_overview:

    kpi_cols = st.columns(len(loaded))
    for col, (name, df) in zip(kpi_cols, loaded.items()):
        color  = ds_color(name)
        mean_c = df["C"].mean()
        band   = assign_band_vec([mean_c], t_low, t_med, t_high)[0]
        with col:
            st.markdown(
                f'<div class="kcard" style="border-top:3px solid {color}">'
                f'<div style="font-size:0.63rem;color:{TXT2};letter-spacing:0.1em;'
                f'text-transform:uppercase">{name}</div>'
                f'<div style="font-family:\'Space Grotesk\';font-size:2rem;font-weight:700;'
                f'color:{color};line-height:1.1">{mean_c:.3f}</div>'
                f'<div style="font-size:0.68rem;color:{BAND_COLOR.get(band,ACCENT)};'
                f'margin-top:2px">{band} · {len(df):,} queries</div>'
                f'{score_bar_html({d: (df[d].mean(), DIM_COLORS[d]) for d in DIMS})}'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown('<div class="sec">Current Weight Configuration</div>',
                unsafe_allow_html=True)
    wcols = st.columns(5)
    for col, dim in zip(wcols, DIMS):
        with col:
            st.markdown(
                f'<div class="kcard" style="text-align:center;padding:12px 8px">'
                f'<div style="font-size:0.65rem;color:{TXT2}">{DIM_LABELS[dim]}</div>'
                f'<div style="font-size:1.8rem;font-weight:700;color:{DIM_COLORS[dim]}">'
                f'{norm_w[dim]:.2f}</div>'
                f'<div style="font-size:0.65rem;color:{TXT2}">({dim})</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div class="kcard kcard-hi" style="font-size:0.78rem">'
        f'Band thresholds — '
        f'<span style="color:{GREEN}">LOW</span> &lt; <b>{t_low:.2f}</b> ≤ '
        f'<span style="color:{ACCENT}">MEDIUM</span> &lt; <b>{t_med:.2f}</b> ≤ '
        f'<span style="color:{YELLOW}">HIGH</span> &lt; <b>{t_high:.2f}</b> ≤ '
        f'<span style="color:{RED}">VERY HIGH</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Weighted Contribution per Dimension × Dataset</div>',
                unsafe_allow_html=True)
    fig_contrib = go.Figure()
    for dim in DIMS:
        contribs = [round(norm_w[dim] * loaded[n][dim].mean(), 4) for n in loaded]
        fig_contrib.add_trace(go.Bar(
            name=DIM_LABELS[dim], x=list(loaded.keys()), y=contribs,
            marker_color=DIM_COLORS[dim],
        ))
    fig_contrib.update_layout(**{**PL, "height": 300, "barmode": "stack",
        "legend": dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        "yaxis": dict(gridcolor=GRID, title="w · mean_score contribution"),
        "xaxis": dict(gridcolor="rgba(0,0,0,0)"),
    })
    st.plotly_chart(fig_contrib, use_container_width=True, config={"displayModeBar": False})

    c1, c2 = st.columns(2)
    with c1:
        st.markdown('<div class="sec">Summary Table</div>', unsafe_allow_html=True)
        rows = []
        for name, df in loaded.items():
            rows.append({
                "Dataset": name, "N": len(df),
                "Mean C": round(df["C"].mean(), 3),
                "Std C":  round(df["C"].std(),  3),
                "Min C":  round(df["C"].min(),  3),
                "Max C":  round(df["C"].max(),  3),
                "Exp Rank": EXPECTED_RANK.get(name, "–"),
            })
        s_df = pd.DataFrame(rows).sort_values("Exp Rank")
        st.dataframe(
            s_df.style
                .background_gradient(subset=["Mean C"], cmap="Blues")
                .format({"Mean C": "{:.3f}", "Std C": "{:.3f}",
                         "Min C": "{:.3f}", "Max C": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

    with c2:
        st.markdown('<div class="sec">Rank Stability — Current vs Default</div>',
                    unsafe_allow_html=True)
        cur_rank = {n: i + 1 for i, n in
                    enumerate(sorted(loaded, key=lambda n: loaded[n]["C"].mean()))}
        def_rank = {n: i + 1 for i, n in
                    enumerate(sorted(def_loaded, key=lambda n: def_loaded[n]["C"].mean()))}
        stab_rows = []
        for name in loaded:
            cr    = cur_rank[name]
            dr    = def_rank.get(name, "–")
            moved = (cr - dr) if isinstance(dr, int) else 0
            stab_rows.append({
                "Dataset": name, "Def Rank": dr, "Cur Rank": cr,
                "Exp Rank": EXPECTED_RANK.get(name, "–"), "Δ": moved,
                "C cur": round(loaded[name]["C"].mean(), 3),
                "C def": round(def_loaded[name]["C"].mean(), 3),
            })

        def _col_delta(v):
            if isinstance(v, (int, float)):
                if v < 0: return f"color:{GREEN}"
                if v > 0: return f"color:{RED}"
            return f"color:{TXT2}"

        st.dataframe(
            pd.DataFrame(stab_rows).style
              .applymap(_col_delta, subset=["Δ"])
              .format({"C cur": "{:.3f}", "C def": "{:.3f}"}),
            use_container_width=True, hide_index=True,
        )

    st.markdown('<div class="sec">Weight Sensitivity Sweep</div>', unsafe_allow_html=True)
    st.caption("Each line sweeps one weight 0→1 (others re-normalise). Impact on mean C across all datasets.")
    sweep    = np.linspace(0.01, 1.0, 40)
    fig_line = go.Figure()
    for dim in DIMS:
        c_means = []
        for wval in sweep:
            trial = norm_w.copy()
            trial[dim] = wval
            tw    = sum(trial.values())
            trial = {k: v / tw for k, v in trial.items()}
            col_c = sum(trial[d] * all_df[d] for d in DIMS)
            c_means.append(round(float(col_c.mean()), 4))
        fig_line.add_trace(go.Scatter(
            x=sweep, y=c_means, mode="lines", name=DIM_LABELS[dim],
            line=dict(color=DIM_COLORS[dim], width=2.5),
            hovertemplate=f"<b>{DIM_LABELS[dim]}</b><br>w=%{{x:.2f}}<br>μC=%{{y:.3f}}<extra></extra>",
        ))
    for dim in DIMS:
        cur_mean = float((sum(norm_w[d] * all_df[d] for d in DIMS)).mean())
        fig_line.add_trace(go.Scatter(
            x=[norm_w[dim]], y=[cur_mean], mode="markers",
            marker=dict(color=DIM_COLORS[dim], size=10, symbol="diamond"),
            showlegend=False,
            hovertemplate=f"Current {DIM_LABELS[dim]} w={norm_w[dim]:.2f}<extra></extra>",
        ))
    fig_line.update_layout(**{**PL, "height": 340,
        "legend": dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        "xaxis": dict(title=dict(text="Weight value (swept)", font=dict(size=9, color=TXT2)),
                      range=[0, 1], gridcolor=GRID),
        "yaxis": dict(title=dict(text="Mean C (all datasets)", font=dict(size=9, color=TXT2)),
                      range=[0, 1], gridcolor=GRID),
    })
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — DISTRIBUTIONS
# ══════════════════════════════════════════════════════════════════════════════
with t_dist:

    st.markdown('<div class="sec">C Score Distribution — Violin per Dataset</div>',
                unsafe_allow_html=True)
    fig_vio = go.Figure()
    for name, df in loaded.items():
        fig_vio.add_trace(go.Violin(
            y=df["C"], name=name,
            fillcolor=hex_rgba(ds_color(name), 0.25),
            line_color=ds_color(name),
            box_visible=True, meanline_visible=True, points=False,
            hovertemplate=f"<b>{name}</b><br>C: %{{y:.3f}}<extra></extra>",
        ))
    for tval, tlbl, tcol in [(t_low, "LOW", GREEN), (t_med, "MED", ACCENT), (t_high, "HIGH", YELLOW)]:
        fig_vio.add_hline(y=tval, line_dash="dot", line_color=tcol, line_width=1,
                          annotation_text=f"  {tlbl} ({tval:.2f})",
                          annotation_font=dict(color=tcol, size=9))
    fig_vio.update_layout(**{**PL, "height": 380, "showlegend": True,
        "legend": dict(orientation="h", y=1.08, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        "yaxis": dict(title="C score", range=[0, 1.05], gridcolor=GRID),
    })
    st.plotly_chart(fig_vio, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">C Score Histograms — All Datasets Overlaid</div>',
                unsafe_allow_html=True)
    fig_hist = go.Figure()
    for name, df in loaded.items():
        fig_hist.add_trace(go.Histogram(
            x=df["C"], name=name, nbinsx=40, opacity=0.55,
            marker_color=ds_color(name),
        ))
    for tval, tlbl, tcol in [(t_low, "LOW", GREEN), (t_med, "MED", ACCENT), (t_high, "HIGH", YELLOW)]:
        fig_hist.add_vline(x=tval, line_dash="dot", line_color=tcol, line_width=1.5,
                           annotation_text=tlbl, annotation_font=dict(color=tcol, size=9))
    fig_hist.update_layout(**{**PL, "height": 320, "barmode": "overlay",
        "legend": dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        "xaxis": dict(title="C score", range=[0, 1], gridcolor=GRID),
        "yaxis": dict(title="Count", gridcolor=GRID),
    })
    st.plotly_chart(fig_hist, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">Box Plots with Mean ± Std</div>', unsafe_allow_html=True)
    fig_box = go.Figure()
    for name, df in loaded.items():
        fig_box.add_trace(go.Box(
            y=df["C"], name=name, boxmean="sd",
            marker_color=ds_color(name), line_color=ds_color(name),
            fillcolor=hex_rgba(ds_color(name), 0.2),
        ))
    fig_box.update_layout(**{**PL, "height": 300, "showlegend": False,
        "yaxis": dict(range=[0, 1.05], gridcolor=GRID),
    })
    st.plotly_chart(fig_box, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">Empirical CDF — Cumulative C Score Distribution</div>',
                unsafe_allow_html=True)
    fig_ecdf = go.Figure()
    for name, df in loaded.items():
        sc   = np.sort(df["C"].values)
        ecdf = np.arange(1, len(sc) + 1) / len(sc)
        fig_ecdf.add_trace(go.Scatter(
            x=sc, y=ecdf, mode="lines", name=name,
            line=dict(color=ds_color(name), width=2.5),
            hovertemplate=f"<b>{name}</b><br>C≤%{{x:.3f}}: %{{y:.1%}}<extra></extra>",
        ))
    for tval, tcol in [(t_low, GREEN), (t_med, ACCENT), (t_high, YELLOW)]:
        fig_ecdf.add_vline(x=tval, line_dash="dot", line_color=tcol, line_width=1.2)
    fig_ecdf.update_layout(**{**PL, "height": 320,
        "legend": dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        "xaxis": dict(title="C score", range=[0, 1], gridcolor=GRID),
        "yaxis": dict(title="Cumulative fraction", range=[0, 1.02], gridcolor=GRID,
                      tickformat=".0%"),
    })
    st.plotly_chart(fig_ecdf, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — DIMENSIONS
# ══════════════════════════════════════════════════════════════════════════════
with t_dims_tab:

    st.markdown('<div class="sec">Sub-Dimension Mean Scores — Heatmap</div>',
                unsafe_allow_html=True)
    heat_z = np.array([[loaded[n][d].mean() for d in DIMS] for n in loaded])
    fig_heat = go.Figure(go.Heatmap(
        z=heat_z,
        x=[DIM_LABELS[d] for d in DIMS],
        y=list(loaded.keys()),
        colorscale="Blues",
        text=np.round(heat_z, 3), texttemplate="%{text}",
        textfont=dict(size=12, color=TXT),
        hovertemplate="<b>%{y}</b> · %{x}: %{z:.3f}<extra></extra>",
    ))
    fig_heat.update_layout(**{**PL,
        "height": max(250, 65 * len(loaded)),
        "margin": dict(l=90, r=12, t=38, b=60),
    })
    st.plotly_chart(fig_heat, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">Sub-Dimension Radar — Per Dataset</div>',
                unsafe_allow_html=True)
    n_ds   = len(loaded)
    cols_r = min(n_ds, 3)
    rows_r = math.ceil(n_ds / cols_r)
    fig_rad = make_subplots(
        rows=rows_r, cols=cols_r,
        specs=[[{"type": "polar"}] * cols_r for _ in range(rows_r)],
        subplot_titles=list(loaded.keys()),
    )
    theta_labels = [DIM_LABELS[d] for d in DIMS]
    for idx, (name, df) in enumerate(loaded.items()):
        row, col_i = divmod(idx, cols_r)
        vals  = [df[d].mean() for d in DIMS]
        color = ds_color(name)
        fig_rad.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=theta_labels + [theta_labels[0]],
            fill="toself", fillcolor=hex_rgba(color, 0.25),
            line=dict(color=color, width=2), name=name, showlegend=False,
        ), row=row + 1, col=col_i + 1)
    fig_rad.update_layout(**{**PL, "height": 300 * rows_r,
                             "margin": dict(l=20, r=20, t=40, b=20)})
    st.plotly_chart(fig_rad, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">Weighted Contribution Breakdown — Per Dataset</div>',
                unsafe_allow_html=True)
    pie_cols = st.columns(min(len(loaded), 4))
    for col, (name, df) in zip(pie_cols, loaded.items()):
        contribs = [norm_w[d] * df[d].mean() for d in DIMS]
        fig_pie  = go.Figure(go.Pie(
            labels=[DIM_LABELS[d] for d in DIMS], values=contribs,
            marker_colors=[DIM_COLORS[d] for d in DIMS],
            hole=0.45, textfont=dict(size=10),
        ))
        fig_pie.update_layout(**{**PL, "height": 220,
            "margin": dict(l=0, r=0, t=28, b=0),
            "title": dict(text=name, font=dict(size=11, color=ds_color(name))),
            "showlegend": False,
        })
        with col:
            st.plotly_chart(fig_pie, use_container_width=True,
                            config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — BAND ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with t_bands:
    st.markdown(
        f'<div class="kcard kcard-hi" style="font-size:0.8rem;margin-bottom:12px">'
        f'<b>Live band thresholds</b> — adjust sliders in the sidebar; all charts update instantly.<br>'
        f'<span style="color:{GREEN}">■ LOW</span> &lt; {t_low:.2f} · '
        f'<span style="color:{ACCENT}">■ MEDIUM</span> &lt; {t_med:.2f} · '
        f'<span style="color:{YELLOW}">■ HIGH</span> &lt; {t_high:.2f} · '
        f'<span style="color:{RED}">■ VERY HIGH</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown('<div class="sec">Band Distribution — Stacked % per Dataset</div>',
                unsafe_allow_html=True)
    fig_bnd = go.Figure()
    for band in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]:
        vals = [round((df["band"] == band).sum() / len(df) * 100, 1)
                for df in loaded.values()]
        fig_bnd.add_trace(go.Bar(
            name=band, x=list(loaded.keys()), y=vals,
            marker_color=BAND_COLOR[band],
            text=[f"{v}%" for v in vals], textposition="inside",
            textfont=dict(size=9, color="white"),
        ))
    fig_bnd.update_layout(**{**PL, "height": 320, "barmode": "stack",
        "legend": dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        "yaxis": dict(range=[0, 100], ticksuffix="%", gridcolor=GRID),
        "xaxis": dict(gridcolor="rgba(0,0,0,0)"),
    })
    st.plotly_chart(fig_bnd, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">Band Counts per Dataset</div>', unsafe_allow_html=True)
    band_rows = []
    for name, df in loaded.items():
        row = {"Dataset": name, "Total": len(df)}
        for band in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]:
            cnt = (df["band"] == band).sum()
            row[band]       = cnt
            row[f"{band}%"] = f"{cnt / len(df) * 100:.1f}%"
        band_rows.append(row)
    st.dataframe(pd.DataFrame(band_rows), use_container_width=True, hide_index=True)

    st.markdown('<div class="sec">Sub-Dimension Profiles by Band (All Datasets Combined)</div>',
                unsafe_allow_html=True)
    fig_bp = go.Figure()
    for band in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]:
        subset = all_df[all_df["band"] == band]
        if len(subset) == 0:
            continue
        vals = [subset[d].mean() for d in DIMS]
        fig_bp.add_trace(go.Bar(
            name=band, x=[DIM_LABELS[d] for d in DIMS], y=vals,
            marker_color=BAND_COLOR[band],
            text=[f"{v:.3f}" for v in vals],
            textposition="outside", textfont=dict(size=9, color=TXT),
        ))
    fig_bp.update_layout(**{**PL, "height": 320, "barmode": "group",
        "legend": dict(orientation="h", y=1.1, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        "yaxis": dict(range=[0, 1.1], gridcolor=GRID),
        "xaxis": dict(gridcolor="rgba(0,0,0,0)"),
    })
    st.plotly_chart(fig_bp, use_container_width=True, config={"displayModeBar": False})

    st.markdown('<div class="sec">Dataset → Band Sunburst</div>', unsafe_allow_html=True)
    sb_ids, sb_labels, sb_parents, sb_values, sb_colors = [], [], [], [], []
    for name, df in loaded.items():
        sb_ids.append(name); sb_labels.append(name)
        sb_parents.append(""); sb_values.append(len(df))
        sb_colors.append(ds_color(name))
        for band in ["LOW", "MEDIUM", "HIGH", "VERY HIGH"]:
            cnt = (df["band"] == band).sum()
            if cnt > 0:
                uid = f"{name}_{band}"
                sb_ids.append(uid); sb_labels.append(band)
                sb_parents.append(name); sb_values.append(cnt)
                sb_colors.append(BAND_COLOR[band])
    fig_sun = go.Figure(go.Sunburst(
        ids=sb_ids, labels=sb_labels, parents=sb_parents,
        values=sb_values, marker=dict(colors=sb_colors),
        branchvalues="total",
        hovertemplate="<b>%{label}</b><br>%{value} queries<extra></extra>",
    ))
    fig_sun.update_layout(**{**PL, "height": 420,
                             "margin": dict(l=0, r=0, t=30, b=0)})
    st.plotly_chart(fig_sun, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — CORRELATIONS
# ══════════════════════════════════════════════════════════════════════════════
with t_corr:

    st.markdown('<div class="sec">Correlation Matrix — C and Sub-Dimensions</div>',
                unsafe_allow_html=True)
    corr_cols = ["C"] + DIMS
    corr_mat  = all_df[corr_cols].corr().round(3)
    fig_corr  = go.Figure(go.Heatmap(
        z=corr_mat.values,
        x=corr_mat.columns, y=corr_mat.index,
        colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
        text=corr_mat.values.round(2), texttemplate="%{text}",
        textfont=dict(size=11),
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    fig_corr.update_layout(**{**PL, "height": 400,
                              "margin": dict(l=70, r=12, t=38, b=70)})
    st.plotly_chart(fig_corr, use_container_width=True, config={"displayModeBar": False})


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — QUERY EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
with t_explorer:
    st.markdown('<div class="sec">Interactive Query Explorer</div>', unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1:
        sel_ds    = st.multiselect("Datasets", list(loaded.keys()),
                                   default=list(loaded.keys()))
    with f2:
        c_range   = st.slider("C score range", 0.0, 1.0, (0.0, 1.0), 0.01)
    with f3:
        sel_bands = st.multiselect("Bands",
                                   ["LOW", "MEDIUM", "HIGH", "VERY HIGH"],
                                   default=["LOW", "MEDIUM", "HIGH", "VERY HIGH"])

    mask = (
        all_df["_dataset"].isin(sel_ds) &
        all_df["C"].between(*c_range) &
        all_df["band"].isin(sel_bands)
    )
    explore_df = all_df[mask].copy()
    st.caption(f"{len(explore_df):,} queries match current filters")

    fig_filt = go.Figure()
    for name in sel_ds:
        sub = explore_df[explore_df["_dataset"] == name]
        if len(sub) == 0:
            continue
        fig_filt.add_trace(go.Histogram(
            x=sub["C"], name=name, nbinsx=30, opacity=0.6,
            marker_color=ds_color(name),
        ))
    for tval, tcol in [(t_low, GREEN), (t_med, ACCENT), (t_high, YELLOW)]:
        fig_filt.add_vline(x=tval, line_dash="dot", line_color=tcol, line_width=1.2)
    fig_filt.update_layout(**{**PL, "height": 220, "barmode": "overlay",
        "xaxis": dict(title="C score", range=[0, 1], gridcolor=GRID),
        "yaxis": dict(gridcolor=GRID),
        "legend": dict(orientation="h", y=1.1, font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
    })
    st.plotly_chart(fig_filt, use_container_width=True, config={"displayModeBar": False})

    display_cols = ["_dataset", "C", "band"] + DIMS
    for ec in ["task_type", "bloom_level"]:
        if ec in explore_df.columns:
            display_cols.append(ec)
    if "_qcol" in explore_df.columns and len(explore_df) > 0:
        qcol_name = explore_df["_qcol"].iloc[0]
        if qcol_name and qcol_name in explore_df.columns:
            display_cols = [qcol_name] + display_cols

    st.markdown('<div class="sec">Sample Rows</div>', unsafe_allow_html=True)
    n_show  = st.slider("Rows to display", 5, 200, 25, 5)
    show_df = explore_df[display_cols].head(n_show)
    fmt     = {d: "{:.3f}" for d in DIMS + ["C"] if d in show_df.columns}
    st.dataframe(
        show_df.style
               .background_gradient(subset=["C"] if "C" in show_df.columns else [],
                                    cmap="Blues")
               .format(fmt),
        use_container_width=True, hide_index=True,
    )

    csv = explore_df.to_csv(index=False).encode()
    st.download_button("⬇  Download filtered data as CSV",
                       csv, "tce_filtered.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
# TAB 7 — TIMING
# ══════════════════════════════════════════════════════════════════════════════
with t_timing:

    ds_times  = timing_store.get("datasets", {})
    total_sec = timing_store.get("total_sec", 0.0)
    total_q   = timing_store.get("total_queries", 0)
    overall_q = timing_store.get("overall_qps", 0.0)

    # ── Note: timing is from the cached run — unchanged by weight slider ──────
    st.markdown(
        f'<div class="kcard kcard-hi" style="font-size:0.8rem;margin-bottom:14px">'
        f'Timing is captured once during scoring and <b>cached</b>. '
        f'Adjusting weights or band thresholds does <b>not</b> re-score — '
        f'these numbers reflect the actual estimator run time.'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── KPI strip ─────────────────────────────────────────────────────────────
    st.markdown('<div class="sec">Processing Summary</div>', unsafe_allow_html=True)
    k1, k2, k3, k4 = st.columns(4)
    kpi_style = (
        f'background:{CARD};border:1px solid {BORDER};border-radius:12px;'
        f'padding:14px 18px;text-align:center'
    )
    for col, label, value, unit, color in [
        (k1, "Total Wall Time",    f"{total_sec:.3f}",    "seconds",       ACCENT),
        (k2, "Queries Scored",     f"{total_q:,}",         "queries",       GREEN),
        (k3, "Overall Throughput", f"{overall_q:,.0f}",    "queries / sec", YELLOW),
        (k4, "Datasets Loaded",    str(len(ds_times)),     "datasets",      RED),
    ]:
        with col:
            st.markdown(
                f'<div style="{kpi_style}">'
                f'<div style="font-size:0.62rem;color:{TXT2};letter-spacing:0.08em">'
                f'{label}</div>'
                f'<div style="font-family:\'Space Grotesk\',sans-serif;font-size:1.9rem;'
                f'font-weight:700;color:{color};line-height:1.15">{value}</div>'
                f'<div style="font-size:0.64rem;color:{TXT2}">{unit}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ── Per-dataset table ─────────────────────────────────────────────────────
    st.markdown('<div class="sec">Per-Dataset Breakdown</div>', unsafe_allow_html=True)
    t_rows = []
    for name, t in ds_times.items():
        t_rows.append({
            "Dataset":          name,
            "Queries":          t.get("n_queries",    0),
            "Load (s)":         t.get("load_sec",     0.0),
            "Score (s)":        t.get("score_sec",    0.0),
            "Total (s)":        t.get("total_sec",    0.0),
            "Throughput (q/s)": t.get("qps",          0.0),
            "ms / query":       t.get("ms_per_query", 0.0),
            "Error":            t.get("error") or "—",
        })
    t_df = pd.DataFrame(t_rows)
    st.dataframe(
        t_df.style
            .background_gradient(subset=["Total (s)"],        cmap="YlOrRd")
            .background_gradient(subset=["Throughput (q/s)"], cmap="Greens")
            .format({
                "Load (s)":           "{:.4f}",
                "Score (s)":          "{:.4f}",
                "Total (s)":          "{:.4f}",
                "Throughput (q/s)":   "{:,.1f}",
                "ms / query":         "{:.3f}",
            }),
        use_container_width=True,
        hide_index=True,
    )

    # ── Load vs Score grouped bar ─────────────────────────────────────────────
    if t_rows:
        names  = [r["Dataset"]    for r in t_rows]
        loads  = [r["Load (s)"]   for r in t_rows]
        scores = [r["Score (s)"]  for r in t_rows]

        st.markdown('<div class="sec">Load Time vs Score Time per Dataset</div>',
                    unsafe_allow_html=True)
        fig_lt = go.Figure()
        fig_lt.add_trace(go.Bar(
            name="Load (disk/memory)", x=names, y=loads,
            marker_color=ACCENT,
            text=[f"{v:.3f}s" for v in loads],
            textposition="outside", textfont=dict(size=9, color=TXT),
        ))
        fig_lt.add_trace(go.Bar(
            name="Scoring (estimator)", x=names, y=scores,
            marker_color=GREEN,
            text=[f"{v:.3f}s" for v in scores],
            textposition="outside", textfont=dict(size=9, color=TXT),
        ))
        fig_lt.update_layout(**{**PL, "height": 320, "barmode": "group",
            "legend": dict(orientation="h", y=1.12, font=dict(size=10),
                           bgcolor="rgba(0,0,0,0)"),
            "yaxis": dict(title="seconds", gridcolor=GRID),
            "xaxis": dict(gridcolor="rgba(0,0,0,0)"),
        })
        st.plotly_chart(fig_lt, use_container_width=True,
                        config={"displayModeBar": False})

        # ── Stacked total time per dataset ────────────────────────────────────
        st.markdown('<div class="sec">Stacked Time per Dataset (Load + Score)</div>',
                    unsafe_allow_html=True)
        fig_stk = go.Figure()
        fig_stk.add_trace(go.Bar(
            name="Load", x=names, y=loads, marker_color=ACCENT,
        ))
        fig_stk.add_trace(go.Bar(
            name="Score", x=names, y=scores, marker_color=GREEN,
            text=[f"{l+s:.3f}s total" for l, s in zip(loads, scores)],
            textposition="outside", textfont=dict(size=9, color=TXT),
        ))
        fig_stk.update_layout(**{**PL, "height": 300, "barmode": "stack",
            "legend": dict(orientation="h", y=1.12, font=dict(size=10),
                           bgcolor="rgba(0,0,0,0)"),
            "yaxis": dict(title="seconds", gridcolor=GRID),
            "xaxis": dict(gridcolor="rgba(0,0,0,0)"),
        })
        st.plotly_chart(fig_stk, use_container_width=True,
                        config={"displayModeBar": False})

        # ── Throughput bar ────────────────────────────────────────────────────
        st.markdown('<div class="sec">Throughput — Queries Scored per Second</div>',
                    unsafe_allow_html=True)
        qps_vals  = [r["Throughput (q/s)"] for r in t_rows]
        max_qps   = max(qps_vals) if qps_vals else 1
        fig_qps   = go.Figure(go.Bar(
            x=names, y=qps_vals,
            marker_color=[GREEN if v >= max_qps * 0.7 else
                          YELLOW if v >= max_qps * 0.4 else RED
                          for v in qps_vals],
            text=[f"{v:,.0f} q/s" for v in qps_vals],
            textposition="outside", textfont=dict(size=10, color=TXT),
        ))
        fig_qps.add_hline(
            y=overall_q, line_dash="dot", line_color=ACCENT, line_width=1.5,
            annotation_text=f"  overall: {overall_q:,.0f} q/s",
            annotation_font=dict(color=ACCENT, size=9),
        )
        fig_qps.update_layout(**{**PL, "height": 280, "showlegend": False,
            "yaxis": dict(title="queries / sec", gridcolor=GRID),
            "xaxis": dict(gridcolor="rgba(0,0,0,0)"),
        })
        st.plotly_chart(fig_qps, use_container_width=True,
                        config={"displayModeBar": False})

        # ── ms / query bar ────────────────────────────────────────────────────
        st.markdown('<div class="sec">Latency — Milliseconds per Query</div>',
                    unsafe_allow_html=True)
        ms_vals = [r["ms / query"] for r in t_rows]
        max_ms  = max(ms_vals) if ms_vals else 1
        fig_ms  = go.Figure(go.Bar(
            x=names, y=ms_vals,
            marker_color=[RED if v >= max_ms * 0.7 else
                          YELLOW if v >= max_ms * 0.4 else GREEN
                          for v in ms_vals],
            text=[f"{v:.2f} ms" for v in ms_vals],
            textposition="outside", textfont=dict(size=10, color=TXT),
        ))
        fig_ms.update_layout(**{**PL, "height": 280, "showlegend": False,
            "yaxis": dict(title="ms / query", gridcolor=GRID),
            "xaxis": dict(gridcolor="rgba(0,0,0,0)"),
        })
        st.plotly_chart(fig_ms, use_container_width=True,
                        config={"displayModeBar": False})

        # ── Pie: share of total scoring time ──────────────────────────────────
        c_pie, c_txt = st.columns([1, 1])
        with c_pie:
            st.markdown('<div class="sec">Share of Total Scoring Time</div>',
                        unsafe_allow_html=True)
            pie_colors = [ds_color(n) for n in names]
            fig_tpie   = go.Figure(go.Pie(
                labels=names, values=scores,
                marker_colors=pie_colors,
                hole=0.5,
                textinfo="label+percent",
                hovertemplate="<b>%{label}</b><br>Score time: %{value:.4f}s"
                              "<br>%{percent}<extra></extra>",
            ))
            fig_tpie.update_layout(**{**PL, "height": 280,
                                       "margin": dict(l=0, r=0, t=28, b=0)})
            st.plotly_chart(fig_tpie, use_container_width=True,
                            config={"displayModeBar": False})

        with c_txt:
            st.markdown('<div class="sec">Query Count vs Time Scatter</div>',
                        unsafe_allow_html=True)
            n_qs = [r["Queries"] for r in t_rows]
            fig_sc = go.Figure()
            for i, (name, n, s) in enumerate(zip(names, n_qs, scores)):
                fig_sc.add_trace(go.Scatter(
                    x=[n], y=[s], mode="markers+text",
                    name=name,
                    text=[name], textposition="top center",
                    textfont=dict(size=9, color=ds_color(name)),
                    marker=dict(size=14, color=ds_color(name),
                                line=dict(color="white", width=1)),
                    showlegend=False,
                    hovertemplate=f"<b>{name}</b><br>Queries: {n:,}"
                                  f"<br>Score time: {s:.4f}s<extra></extra>",
                ))
            # Linear trend if > 1 dataset
            if len(n_qs) > 1:
                xs = np.array(n_qs, dtype=float)
                ys = np.array(scores, dtype=float)
                m, b = np.polyfit(xs, ys, 1)
                x_line = np.linspace(min(xs), max(xs), 50)
                fig_sc.add_trace(go.Scatter(
                    x=x_line, y=np.clip(m * x_line + b, 0, None),
                    mode="lines", line=dict(color=TXT2, width=1, dash="dot"),
                    showlegend=False, hoverinfo="skip",
                ))
            fig_sc.update_layout(**{**PL, "height": 280, "showlegend": False,
                "xaxis": dict(title="Query count", gridcolor=GRID),
                "yaxis": dict(title="Score time (s)", gridcolor=GRID),
            })
            st.plotly_chart(fig_sc, use_container_width=True,
                            config={"displayModeBar": False})

        # ── Error summary ─────────────────────────────────────────────────────
        errors = {n: t.get("error") for n, t in ds_times.items() if t.get("error")}
        if errors:
            st.markdown('<div class="sec">Errors</div>', unsafe_allow_html=True)
            for name, err in errors.items():
                st.markdown(
                    f'<div class="kcard kcard-red" style="font-size:0.78rem">'
                    f'<b style="color:{RED}">{name}</b> — {err}</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.markdown(
                f'<div class="kcard kcard-grn" style="font-size:0.78rem">'
                f'✓ All datasets loaded and scored without errors.</div>',
                unsafe_allow_html=True,
            )