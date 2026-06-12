"""D2 — trajectory expansion: G(q) → T(q,a).

Maps QCE procedure graph G(q) and agent execution rules to a structural trace
skeleton per (query, agent). Consumed by D3 (``daar.cost_hand``) for ĉ(q,a).
"""

from __future__ import annotations

from typing import Any

from config.local.constants.agents import ROUTER_AGENTS
from qce.graph import build_task_dag

REACT_MAX_STEPS = 8
MULTI_MAX_DEPTH = 20

TRACE_COLUMNS = (
    "training_id",
    "agent",
    "trace_type",
    "estimated_depth",
    "estimated_tool_loops",
    "estimated_workers",
    "has_verification",
    "recursion_flag",
    "planner_nodes",
    "planner_depth",
)


def _planner_metrics(plan: dict[str, Any]) -> dict[str, Any]:
    subtasks = list(plan.get("subtasks") or [])
    feats = build_task_dag(plan).features
    max_depth = int(feats.get("max_depth", 0) or 0)
    width = int(feats.get("width", 0) or 0)
    return {
        "planner_nodes": int(feats.get("n_nodes", 0) or 0),
        "planner_depth": max_depth,
        "n_tool_nodes": int(feats.get("n_tool_nodes", 0) or 0),
        "critical_path_tool_steps": int(feats.get("critical_path_tool_steps", 0) or 0),
        "n_subtasks": len(subtasks),
        "n_tool_subtasks": sum(1 for st in subtasks if st.get("needs_tool")),
        "has_verification": any(
            str(st.get("step_type", "")).strip().lower() == "verify" for st in subtasks
        ),
        "recursion_flag": max_depth >= 3 and width >= 2,
    }


def _trace(metrics: dict[str, Any], **fields: Any) -> dict[str, Any]:
    return {
        "has_verification": metrics["has_verification"],
        "recursion_flag": metrics["recursion_flag"],
        "planner_nodes": metrics["planner_nodes"],
        "planner_depth": metrics["planner_depth"],
        **fields,
    }


def build_trace_row(plan: dict[str, Any], agent: str) -> dict[str, Any]:
    m = _planner_metrics(plan)
    if agent in ("raw", "cot"):
        trace = _trace(
            m,
            trace_type="linear",
            estimated_depth=1,
            estimated_tool_loops=0,
            estimated_workers=0,
        )
    elif agent == "react":
        loops = min(
            max(m["critical_path_tool_steps"], m["n_tool_subtasks"], 1 if m["n_tool_nodes"] else 0),
            REACT_MAX_STEPS,
        )
        trace = _trace(
            m,
            trace_type="react_loop",
            estimated_depth=min(max(1 + loops * 2, 1), REACT_MAX_STEPS),
            estimated_tool_loops=loops,
            estimated_workers=0,
        )
    elif agent == "multiagent":
        subtasks = list(plan.get("subtasks") or [])
        workers = max(m["planner_nodes"], m["n_subtasks"], 1)
        worker_llm = sum(
            1 + (6 if st.get("needs_tool") else 0) for st in subtasks
        ) if subtasks else workers
        trace = _trace(
            m,
            trace_type="multi_round",
            estimated_depth=min(max(1 + worker_llm + 1, 1), MULTI_MAX_DEPTH),
            estimated_tool_loops=m["n_tool_subtasks"],
            estimated_workers=workers,
        )
    else:
        raise ValueError(f"unknown agent {agent!r}")
    return {"training_id": str(plan.get("training_id", "")), "agent": agent, **trace}


def build_trace_skeleton(plans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        build_trace_row(plan, agent)
        for plan in plans
        for agent in ROUTER_AGENTS
    ]
