# agent/registry.py
"""
Central registry of six agent strategies.
Import this in the training runner and evaluation pipeline.
"""
from __future__ import annotations
from typing import Callable, Any
from agent.base import AgentResponse

from agent import raw as _raw
from agent import cot as _cot
from agent import self_consistency as _sc
from agent import react as _react
from agent import debate as _debate
from agent import multiagent as _multiagent
from agent.dataset_profile import agent_kwargs_from_row

STRATEGY_REGISTRY: dict[str, dict[str, Any]] = {
    "raw": {
        "id": 1,
        "agent_id": "raw_001",
        "name": "Raw Direct Completion",
        "run": _raw.run,
        "expected_cost_tier": "lowest",   # 100–400 tokens
        "uses_tools": False,
        "num_llm_calls_typical": 1,
    },
    "cot": {
        "id": 2,
        "agent_id": "cot_002",
        "name": "Chain of Thought",
        "run": _cot.run,
        "expected_cost_tier": "low",      # 300–1200 tokens
        "uses_tools": False,
        "num_llm_calls_typical": 1,
    },
    "self_consistency": {
        "id": 3,
        "agent_id": "self_consistency_003",
        "name": "Self-Consistency",
        "run": _sc.run,
        "expected_cost_tier": "medium-low",  # ~3× CoT
        "uses_tools": False,
        "num_llm_calls_typical": 3,
    },
    "react": {
        "id": 4,
        "agent_id": "react_004",
        "name": "ReAct",
        "run": _react.run,
        "expected_cost_tier": "medium",   # 600–4000 tokens
        "uses_tools": True,
        "num_llm_calls_typical": 5,
    },
    "debate": {
        "id": 6,
        "agent_id": "debate_006",
        "name": "Multi-Agent Debate",
        "run": _debate.run,
        "expected_cost_tier": "medium",   # 600–2500 tokens
        "uses_tools": False,
        "num_llm_calls_typical": 2,  # 3 if disagreement
    },
    "multiagent": {
        "id": 7,
        "agent_id": "multiagent_007",
        "name": "Plan+Execute+Verify",
        "run": _multiagent.run,
        "expected_cost_tier": "highest",  # 2000–12000 tokens
        "uses_tools": True,
        "num_llm_calls_typical": 5,
    },
}

STRATEGY_IDS = list(STRATEGY_REGISTRY.keys())


def run_strategy(
    strategy_name: str,
    query: str,
    model: str,
    dataset: str,
    **kwargs: Any,
) -> AgentResponse:
    """Single entry point to run any strategy by name."""
    if strategy_name not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{strategy_name}'. "
            f"Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[strategy_name]["run"](
        query=query, model=model, dataset=dataset, **kwargs
    )


__all__ = [
    "STRATEGY_REGISTRY",
    "STRATEGY_IDS",
    "run_strategy",
    "agent_kwargs_from_row",
]
