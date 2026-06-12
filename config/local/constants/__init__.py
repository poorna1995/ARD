"""
Project constants — single source of truth.

Domain modules::

    config.local.constants.datasets   — DS, META_KEYS
    config.local.constants.tools      — TOOLS
    config.local.constants.agents     — ROUTER_AGENTS, ALL_AGENTS
    config.local.constants.policy     — POLICY
    config.local.constants.runtime    — BAD_ANSWERS, RUNTIME_KWARGS
    config.local.constants.qce        — GRAPH, PLAN, DIMS, SCORE, STATUS

Import (unchanged)::

    from config.local.constants import DS, TOOLS, PLAN, DIMS, GRAPH, STATUS, SCORE
    from config.local.constants import POLICY, BAD_ANSWERS, AGENTS, ALL_AGENTS
"""

from __future__ import annotations

from config.local.constants.agents import AGENTS, ALL_AGENTS, ROUTER_AGENTS
from config.local.constants.datasets import (
    DATASET_DISPLAY,
    DATASET_DISPLAY_NAMES,
    DISCUSSION_DATASET_ORDER,
    DS,
    META_KEYS,
    Datasets,
    disk_dir_name,
    eval_parquet_stem,
    resolve_dataset_name,
)
from config.local.constants.heuristics import (
    HEUR_DIMENSION_KEYS,
    HEUR_ROUTER_ATOMIC_COLS,
    HEUR_ROUTER_COLS,
    HEUR_ROUTER_DIM_COLS,
    HEUR_ROUTER_TD_COLS,
    PLANNER_TOOL_NAMES,
)
from config.local.constants.models import (
    DEFAULT_AGENT_MODEL,
    DEFAULT_DECOMPOSE_MODEL,
    DEFAULT_LLM_MODEL,
)
from config.local.constants.policy import POLICY, Policy
from config.local.constants.qce import (
    DIMS,
    GRAPH,
    MAX_SEARCH_LEN,
    PLAN,
    PLAN_OK,
    PLAN_PARSE_FAIL,
    SCORE,
    STATUS,
    TRUST_COLS,
    Dims,
    Graph,
    Plan,
    Score,
    Status,
)
from config.local.constants.runtime import BAD_ANSWERS, HOP_KEYS, RUNTIME_KWARGS
from config.local.constants.tools import TOOLS, Tools
from config.local.constants.types import EvidenceMode

__all__ = [
    "ALL_AGENTS",
    "AGENTS",
    "BAD_ANSWERS",
    "DATASET_DISPLAY",
    "DATASET_DISPLAY_NAMES",
    "DEFAULT_AGENT_MODEL",
    "DEFAULT_DECOMPOSE_MODEL",
    "DEFAULT_LLM_MODEL",
    "DISCUSSION_DATASET_ORDER",
    "disk_dir_name",
    "eval_parquet_stem",
    "DIMS",
    "DS",
    "EvidenceMode",
    "HEUR_DIMENSION_KEYS",
    "HEUR_ROUTER_ATOMIC_COLS",
    "HEUR_ROUTER_COLS",
    "HEUR_ROUTER_DIM_COLS",
    "HEUR_ROUTER_TD_COLS",
    "PLANNER_TOOL_NAMES",
    "resolve_dataset_name",
    "GRAPH",
    "HOP_KEYS",
    "MAX_SEARCH_LEN",
    "META_KEYS",
    "PLAN",
    "PLAN_OK",
    "PLAN_PARSE_FAIL",
    "POLICY",
    "ROUTER_AGENTS",
    "RUNTIME_KWARGS",
    "SCORE",
    "STATUS",
    "TOOLS",
    "TRUST_COLS",
    "Datasets",
    "Graph",
    "Plan",
    "Policy",
    "Score",
    "Status",
    "Tools",
    "Dims",
]
