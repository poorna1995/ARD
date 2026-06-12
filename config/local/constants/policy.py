"""Per-dataset agent policies: allowed tools, step limits, evidence mode."""

from __future__ import annotations

from dataclasses import dataclass

from config.local.constants.tools import TOOLS, Tools
from config.local.constants.types import EvidenceMode

_POLICY_SPECS: tuple[tuple[str, int, EvidenceMode], ...] = (
    ("gaia", 14, "open_web"),
    ("hotpot", 7, "provided_context"),
    ("musique", 8, "provided_context"),
    ("math", 8, "internal"),
    ("mmlu", 6, "internal"),
)


def _gaia_allowed(t: Tools) -> frozenset[str]:
    return (
        t.open_web
        | t.file
        | (t.code & frozenset({"python_exec", "math_tool"}))
        | frozenset({"arxiv_search"})
    )


@dataclass(frozen=True)
class Policy:
    allowed: dict[str, frozenset[str]]
    max_steps: dict[str, int]
    evidence: dict[str, EvidenceMode]


POLICY = Policy(
    allowed={
        "math": frozenset({"math_tool"}),
        "hotpot": TOOLS.retrieve,
        "musique": TOOLS.retrieve,
        "gaia": _gaia_allowed(TOOLS),
        "mmlu": frozenset(),
    },
    max_steps={ds: n for ds, n, _ in _POLICY_SPECS},
    evidence={ds: ev for ds, _, ev in _POLICY_SPECS},
)

__all__ = ["POLICY", "Policy"]
