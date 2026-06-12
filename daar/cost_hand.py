"""D3 — cost simulation: T(q,a) → ĉ(q,a) (Gate 2 phase 2).

Handwritten cost simulator calibrated on train oracle; not a learned regressor.
Combined with D2 (``daar.trace_skeleton``), forms Gate 2 for utility routing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calibrate_scales(train_oracle: pd.DataFrame, traces: pd.DataFrame) -> dict[str, float]:
    """Operator calibration on train split only — not a learned regressor."""
    m = train_oracle.merge(traces, on=["training_id", "agent"])
    m["X2_dynamic"] = m["X2"] - m["C_static"]

    cot = m[m["agent"] == "cot"]
    react = m[m["agent"] == "react"]
    multi = m[m["agent"] == "multiagent"]

    react_d = (react["estimated_depth"] - 1).clip(lower=1)
    multi_d = (multi["estimated_depth"] - 1).clip(lower=1)

    return {
        "cot_per_node": float(cot["X2_dynamic"].median() / cot["planner_nodes"].median()),
        "react_per_depth": float((react["X2_dynamic"] / react_d).median()),
        "react_per_tool": float(
            (react["X2_dynamic"] / react["estimated_tool_loops"].replace(0, np.nan)).median()
        ),
        "multi_per_depth": float((multi["X2_dynamic"] / multi_d).median()),
        "multi_per_tool": float(
            (multi["X2_dynamic"] / multi["estimated_tool_loops"].replace(0, np.nan)).median()
        ),
    }


def simulate_cost_hand(
    traces: pd.DataFrame,
    c_static: pd.Series,
    scales: dict[str, float],
    *,
    w1: float = 1.0,
    w2: float = 1.0,
) -> pd.DataFrame:
    df = traces.copy()
    df["training_id"] = df["training_id"].astype(str)
    df = df.merge(
        c_static.rename("C_static"),
        left_on="training_id",
        right_index=True,
        how="left",
    )

    agent = df["agent"]
    d = df["estimated_depth"].astype(float)
    nodes = df["planner_nodes"].astype(float)
    tools = df["estimated_tool_loops"].astype(float)

    df["d_sim"] = d
    df["X1_hat"] = np.where(agent.isin(["react", "multiagent"]), tools, 0.0)

    x2 = np.zeros(len(df), dtype=float)
    x2[agent.eq("cot")] = scales["cot_per_node"] * nodes[agent.eq("cot")]

    react_m = agent.eq("react")
    x2[react_m] = scales["react_per_depth"] * np.maximum(d[react_m] - 1, 0) + scales[
        "react_per_tool"
    ] * df.loc[react_m, "X1_hat"]

    multi_m = agent.eq("multiagent")
    x2[multi_m] = scales["multi_per_depth"] * np.maximum(d[multi_m] - 1, 0) + scales[
        "multi_per_tool"
    ] * df.loc[multi_m, "X1_hat"]

    df["X2_dynamic_hat"] = x2
    df["c_hat"] = w1 * df["X1_hat"] + w2 * (df["C_static"] + df["X2_dynamic_hat"])
    return df[
        ["training_id", "agent", "X1_hat", "X2_dynamic_hat", "d_sim", "c_hat"]
    ].copy()
