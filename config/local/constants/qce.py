"""QCE graph, plan decomposition, complexity dimensions, and scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Final

_STEP_TYPES = frozenset({"retrieve", "reason", "compute", "verify", "select"})
_REL_ALL = frozenset(
    {
        "temporal",
        "location",
        "country",
        "comparison",
        "numeric",
        "causal",
        "entity_lookup",
        "compositional",
        "verification",
    }
)
_REL_SEM = frozenset(
    {"compositional", "temporal", "location", "comparison", "country", "causal"}
)
_REL_TEMP = frozenset({"temporal"})
_REL_COMP = frozenset({"comparison"})
_GRANS = frozenset(
    {
        "",
        "year",
        "date",
        "span",
        "number",
        "letter",
        "country",
        "city",
        "name",
        "phrase",
        "boolean",
        "unit",
        "other",
    }
)

_DIM_MAIN = (
    "dim_structural",
    "dim_reasoning",
    "dim_evidence",
    "dim_tool",
    "dim_coordination_uncertainty",
)
_DIM_LEGACY_SUFFIX = ("structural", "compositional", "retrieval", "verification", "uncertainty")

_BAD_QUERY_PAT: Final = (
    r"(?i)\b(from t\d+|entity from|country from|city from|person from|step t\d+)\b"
)
_TASK_ID_PAT: Final = r"t(\d+)$"
MAX_SEARCH_LEN: Final = 120


@dataclass(frozen=True)
class Graph:
    start: str = "__start__"
    end: str = "__end__"


@dataclass(frozen=True)
class Status:
    ok: str = "ok"
    repair: str = "repaired"
    empty: str = "empty"
    parse_fail: str = "parse_fail"


@dataclass(frozen=True)
class Dims:
    main: tuple[str, ...] = _DIM_MAIN
    legacy: tuple[str, ...] = tuple(f"dim_legacy_{s}" for s in _DIM_LEGACY_SUFFIX)
    ver_main: str = "c-vector-v3.0-main"


TRUST_COLS = (
    "plan_trust",
    "verify_fraction",
    "terminal_sink_ok",
    "sink_intermediate_risk",
)


@dataclass(frozen=True)
class Score:
    repair: dict[str, float] = field(
        default_factory=lambda: {
            "chain_fallback": 0.45,
            "infer_linear_chain": 0.25,
            "infer_dep": 0.04,
            "drop_cycle_edge": 0.12,
            "skip_back_edge": 0.10,
            "drop_non_forward_dep": 0.06,
            "unknown_parent": 0.08,
            "self_loop": 0.05,
        }
    )
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "max_depth": 0.22,
            "n_tool_nodes": 0.18,
            "tool_fraction": 0.16,
            "critical_path_tool_steps": 0.18,
            "log_nodes": 0.14,
            "verify_fraction": 0.12,
        }
    )
    prefixes: tuple[str, ...] = (
        "chain_fallback:",
        "infer_linear_chain:",
        "drop_cycle_edge:",
    )


@dataclass(frozen=True)
class Plan:
    steps: frozenset[str] = _STEP_TYPES
    relations: frozenset[str] = _REL_ALL
    grans: frozenset[str] = _GRANS
    bands: dict[str, tuple[int, int]] = field(
        default_factory=lambda: dict(
            hotpot=(2, 2),
            musique=(2, 5),
            math=(2, 4),
            mmlu=(3, 3),
            gaia=(2, 4),
        )
    )
    sem_rels: frozenset[str] = _REL_SEM
    temp_rels: frozenset[str] = _REL_TEMP
    comp_rels: frozenset[str] = _REL_COMP
    step_wts: dict[str, float] = field(
        default_factory=lambda: {"retrieve": 1.3, "verify": 1.15, "select": 1.05}
    )
    bad_query: re.Pattern[str] = field(default_factory=lambda: re.compile(_BAD_QUERY_PAT))
    task_id: re.Pattern[str] = field(
        default_factory=lambda: re.compile(_TASK_ID_PAT, re.IGNORECASE)
    )
    default_relation: str = "entity_lookup"


GRAPH = Graph()
STATUS = Status()
DIMS = Dims()
SCORE = Score()
PLAN = Plan()

# Backward-compatible aliases
PLAN_OK = STATUS.ok
PLAN_PARSE_FAIL = STATUS.parse_fail

assert _REL_SEM <= _REL_ALL
assert _REL_TEMP <= _REL_ALL
assert _REL_COMP <= _REL_ALL

__all__ = [
    "DIMS",
    "GRAPH",
    "Graph",
    "MAX_SEARCH_LEN",
    "PLAN",
    "PLAN_OK",
    "PLAN_PARSE_FAIL",
    "SCORE",
    "STATUS",
    "Score",
    "Status",
    "Dims",
    "Plan",
    "TRUST_COLS",
]
