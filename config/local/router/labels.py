"""Router training labels and agent columns."""

from __future__ import annotations

from config.local.constants.agents import ROUTER_AGENTS

TARGET = "oracle_agent"
SOFT_DOMINANT = "soft_dominant_agent"
AGENTS: tuple[str, ...] = ROUTER_AGENTS
PROBA_COLS: list[str] = [f"p_{a}" for a in AGENTS]

__all__ = ["AGENTS", "PROBA_COLS", "SOFT_DOMINANT", "TARGET"]
