"""Default LLM model ids shared across agents, router, and QCE."""

from __future__ import annotations

DEFAULT_LLM_MODEL = "gpt-4o-mini"

# Backward-compatible aliases
DEFAULT_AGENT_MODEL = DEFAULT_LLM_MODEL
DEFAULT_DECOMPOSE_MODEL = DEFAULT_LLM_MODEL

__all__ = ["DEFAULT_AGENT_MODEL", "DEFAULT_DECOMPOSE_MODEL", "DEFAULT_LLM_MODEL"]
