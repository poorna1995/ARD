"""Agent identifiers for routing and evaluation."""

from __future__ import annotations

_ROUTER_AGENTS = ("react", "cot", "raw", "multiagent")
_EXTRA_AGENTS = ("self_consistency", "debate")

ROUTER_AGENTS = _ROUTER_AGENTS
ALL_AGENTS: tuple[str, ...] = ("raw", "cot", "react", "multiagent", *_EXTRA_AGENTS)

# Backward-compatible alias
AGENTS = ROUTER_AGENTS

__all__ = ["AGENTS", "ALL_AGENTS", "ROUTER_AGENTS"]
