"""Synthetic procedure graph Ĝ(q) from distilled complexity φ̂(q)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from config.local.constants import DIMS

CVEC_COLS = tuple(DIMS.main)
_STEP_CYCLE = ("retrieve", "compute", "select", "reason", "verify")


def synthetic_plan_from_phi(
    *,
    training_id: str,
    query: str,
    phi: dict[str, float],
    dataset: str = "",
) -> dict[str, Any]:
    """Build a chain plan from φ̂ for Gate 2 trace expansion when QCE is skipped."""
    structural = float(phi.get("dim_structural", 0.5))
    reasoning = float(phi.get("dim_reasoning", 0.5))
    tool = float(phi.get("dim_tool", 0.3))
    n_steps = int(max(2, min(6, round(2 + structural * 3 + reasoning * 2))))
    n_tool = int(max(0, min(n_steps - 1, round(tool * n_steps))))
    subtasks: list[dict[str, Any]] = []
    for i in range(n_steps):
        needs_tool = i < n_tool
        stype = _STEP_CYCLE[i % len(_STEP_CYCLE)]
        if needs_tool:
            stype = "retrieve"
        subtasks.append(
            {
                "id": f"t{i + 1}",
                "goal": f"subtask {i + 1}",
                "relation_type": "entity_lookup",
                "answer_granularity": "",
                "step_type": stype,
                "needs_tool": needs_tool,
                "tool": "web_search" if needs_tool else "none",
                "search_query": "",
                "depends_on": [f"t{i}"] if i > 0 else [],
            }
        )
    return {
        "training_id": str(training_id),
        "dataset": dataset,
        "query": query,
        "final_constraint": "",
        "subtasks": subtasks,
        "plan_status": "synthetic_from_phi",
        "plan_warns": [],
    }


def phi_vector_from_row(row: pd.Series) -> dict[str, float]:
    return {c: float(pd.to_numeric(row[c], errors="coerce") or 0.0) for c in CVEC_COLS}
