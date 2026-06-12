"""Feature helpers for D-AAR 2.0 (trace-augmented Gate 1b)."""

from __future__ import annotations

import pandas as pd

from config.local.constants import DIMS
from config.local.constants.agents import ROUTER_AGENTS
from daar.trace_skeleton import TRACE_COLUMNS

CVEC_COLS: tuple[str, ...] = tuple(DIMS.main)

# Per-query trace summary from react agent (primary tool-using strategy).
TRACE_PROXY_AGENT = "react"
TRACE_QUERY_COLS: tuple[str, ...] = (
    "trace_react_depth",
    "trace_react_tool_loops",
    "trace_react_planner_nodes",
    "trace_react_planner_depth",
    "trace_max_depth",
    "trace_max_tool_loops",
)


def trace_skeleton_feature_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in TRACE_QUERY_COLS if c in df.columns]


def gate1b_trace_feature_cols(df: pd.DataFrame) -> list[str]:
    """cvec5 + react-proxy trace features for trajectory-aware Gate 1b."""
    cvec = [c for c in CVEC_COLS if c in df.columns]
    trace = trace_skeleton_feature_cols(df)
    if len(cvec) != len(CVEC_COLS):
        raise ValueError(f"missing cvec columns — have {cvec}")
    if not trace:
        raise ValueError("missing trace_* columns — run attach_query_trace_features")
    return cvec + trace


def attach_query_trace_features(
    query: pd.DataFrame,
    traces: pd.DataFrame,
) -> pd.DataFrame:
    """Attach query-level trace summaries from ``daar_trace_skeleton``."""
    out = query.copy()
    out["training_id"] = out["training_id"].astype(str)
    tr = traces.copy()
    tr["training_id"] = tr["training_id"].astype(str)

    numeric_trace = [
        c
        for c in TRACE_COLUMNS
        if c not in ("training_id", "agent", "trace_type")
    ]

    react = tr[tr["agent"] == TRACE_PROXY_AGENT][
        ["training_id", *numeric_trace]
    ].rename(
        columns={
            "estimated_depth": "trace_react_depth",
            "estimated_tool_loops": "trace_react_tool_loops",
            "planner_nodes": "trace_react_planner_nodes",
            "planner_depth": "trace_react_planner_depth",
        }
    )
    agg = (
        tr.groupby("training_id", sort=False)[["estimated_depth", "estimated_tool_loops"]]
        .max()
        .rename(
            columns={
                "estimated_depth": "trace_max_depth",
                "estimated_tool_loops": "trace_max_tool_loops",
            }
        )
        .reset_index()
    )
    out = out.merge(react, on="training_id", how="left")
    out = out.merge(agg, on="training_id", how="left")
    for col in TRACE_QUERY_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0.0)
    return out


def residual_feature_cols(df: pd.DataFrame, *, agent: str) -> list[str]:
    """φ(q) + per-agent trace numeric cols for Δ̂ training."""
    cvec = [c for c in CVEC_COLS if c in df.columns]
    prefix = f"trace_{agent}_"
    trace = [c for c in df.columns if c.startswith(prefix)]
    return cvec + trace


def attach_per_agent_trace_features(long_df: pd.DataFrame, traces: pd.DataFrame) -> pd.DataFrame:
    """Long (q, agent) frame with ``trace_{agent}_*`` columns."""
    out = long_df.copy()
    out["training_id"] = out["training_id"].astype(str)
    tr = traces.copy()
    tr["training_id"] = tr["training_id"].astype(str)
    numeric = [
        c
        for c in TRACE_COLUMNS
        if c not in ("training_id", "agent", "trace_type")
    ]
    wide_parts: list[pd.DataFrame] = []
    for agent in ROUTER_AGENTS:
        sub = tr[tr["agent"] == agent][["training_id", *numeric]].copy()
        sub.columns = ["training_id"] + [f"trace_{agent}_{c}" for c in numeric]
        wide_parts.append(sub)
    wide = wide_parts[0]
    for part in wide_parts[1:]:
        wide = wide.merge(part, on="training_id", how="outer")
    return out.merge(wide, on="training_id", how="left")
