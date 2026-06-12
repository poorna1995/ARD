"""Heuristic feature column order (from legacy difficulty/feature_measure; logic unchanged)."""

from __future__ import annotations

HEUR_DIMENSION_KEYS: tuple[str, ...] = (
    "task_length",
    "reasoning_depth",
    "evidence",
    "tool_dependency",
    "task_type",
    "coordination_uncertainty",
)

HEUR_ROUTER_DIM_COLS: tuple[str, ...] = tuple(f"heur_{d}" for d in HEUR_DIMENSION_KEYS)

HEUR_ROUTER_ATOMIC_COLS: tuple[str, ...] = (
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

PLANNER_TOOL_NAMES: tuple[str, ...] = (
    "web_search",
    "wikipedia",
    "math_tool",
    "python_exec",
    "read_file",
    "arxiv_search",
    "github_search",
    "pdb_parse",
    "retrieve_tool",
)

HEUR_ROUTER_TD_COLS: tuple[str, ...] = tuple(f"heur_td_{t}" for t in PLANNER_TOOL_NAMES)

HEUR_ROUTER_COLS: tuple[str, ...] = (
    *HEUR_ROUTER_DIM_COLS,
    *HEUR_ROUTER_ATOMIC_COLS,
    *HEUR_ROUTER_TD_COLS,
)
