"""Heuristic feature column names for router training."""

from __future__ import annotations

from config.local.constants.tools import TOOLS

_HEUR_DIM_KEYS = (
    "task_length",
    "reasoning_depth",
    "evidence",
    "tool_dependency",
    "task_type",
    "coordination_uncertainty",
)
_HEUR_ATOMIC_COLS = (
    "heur_tl_words",
    "heur_tl_chars",
    "heur_tl_mattr",
    "heur_tl_entity_density",
    "heur_rd_multihop",
    "heur_rd_hop_prior",
    "heur_rd_negation",
    "heur_rd_cognitive",
    "heur_ev_bridge",
    "heur_ev_temporal",
    "heur_ev_comparison",
    "heur_ev_nested_wh",
    "heur_ev_retrieve_cue",
    "heur_tt_form_score",
    "heur_tt_wh",
    "heur_tt_mcq",
    "heur_unc_multi_tool",
    "heur_unc_tool_conflict",
    "heur_unc_lexical",
)

HEUR_DIMENSION_KEYS = _HEUR_DIM_KEYS
HEUR_ROUTER_DIM_COLS = tuple(f"heur_{d}" for d in _HEUR_DIM_KEYS)
HEUR_ROUTER_ATOMIC_COLS = _HEUR_ATOMIC_COLS
PLANNER_TOOL_NAMES = TOOLS.planner_names
HEUR_ROUTER_TD_COLS = tuple(f"heur_td_{t}" for t in PLANNER_TOOL_NAMES)
HEUR_ROUTER_COLS = (*HEUR_ROUTER_DIM_COLS, *HEUR_ROUTER_ATOMIC_COLS, *HEUR_ROUTER_TD_COLS)

__all__ = [
    "HEUR_DIMENSION_KEYS",
    "HEUR_ROUTER_ATOMIC_COLS",
    "HEUR_ROUTER_COLS",
    "HEUR_ROUTER_DIM_COLS",
    "HEUR_ROUTER_TD_COLS",
    "PLANNER_TOOL_NAMES",
]
