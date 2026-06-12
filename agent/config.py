from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


COMMON_CONFIG_KEYS: set[str] = {
    "model",
    "dataset",
    "temperature",
    "max_tokens",
    "seed",
}


MULTIAGENT_CONFIG_KEYS: set[str] = {
    "max_workers",
    "max_subtasks",
    "max_retries",
    "max_replan",
    "enable_tools_when_needed",
    "tool_max_steps",
    "planner_model",
    "worker_model",
    "judge_model",
    "worker_retry_threshold",
    "worker_result_forward_chars",
    "worker_evidence_forward_chars",
}

# Optional params for Self-Consistency (Wang et al., 2023) and related agents.
SELF_CONSISTENCY_CONFIG_KEYS: set[str] = {
    "num_paths",
    "sample_temperature",
    "log_sc_paths",
    # canonical | literal_ci | auto — see agent.self_consistency._vote_key_for_path
    "vote_key_strategy",
}

REACT_CONFIG_KEYS: set[str] = {
    "max_steps",
    "max_format_retries",
}

AGENT_PARAM_KEYS: set[str] = (
    MULTIAGENT_CONFIG_KEYS | SELF_CONSISTENCY_CONFIG_KEYS | REACT_CONFIG_KEYS
)

CANONICAL_CONFIG_KEYS: set[str] = COMMON_CONFIG_KEYS | AGENT_PARAM_KEYS

# Per-strategy allowlist — kwargs for other strategies are ignored (not unknown).
STRATEGY_PARAM_KEYS: dict[str, frozenset[str]] = {
    "raw": frozenset(),
    "cot": frozenset(),
    "debate": frozenset(),
    "react": frozenset(REACT_CONFIG_KEYS),
    "self_consistency": frozenset(SELF_CONSISTENCY_CONFIG_KEYS),
    "multiagent": frozenset(MULTIAGENT_CONFIG_KEYS),
}

# Passed through run() but not part of model/agent configuration.
from config.local.constants import RUNTIME_KWARGS


CONFIG_ALIASES: dict[str, str] = {
    # Tools/retries
    "max_tool_steps": "tool_max_steps",
    "tool_worker_max_steps": "tool_max_steps",
    "tool_worker_retries": "max_retries",
    # Model roles
    "cheap_model": "worker_model",
    "verifier_model": "judge_model",
}


@dataclass
class AgentConfig:
    model: str
    dataset: str
    temperature: float = 0.1
    max_tokens: int = 1024
    seed: int  = 42
    agent_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class NormalizedConfigResult:
    config: AgentConfig
    used_aliases: dict[str, str] = field(default_factory=dict)
    unknown_keys: list[str] = field(default_factory=list)
    ignored_keys: list[str] = field(default_factory=list)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y"}:
            return True
        if lowered in {"false", "0", "no", "n"}:
            return False
    return bool(value)


def _coerce_common_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    if "temperature" in normalized and normalized["temperature"] is not None:
        normalized["temperature"] = float(normalized["temperature"])
    if "max_tokens" in normalized and normalized["max_tokens"] is not None:
        normalized["max_tokens"] = int(normalized["max_tokens"])
    if "seed" in normalized and normalized["seed"] is not None:
        normalized["seed"] = int(normalized["seed"])
    return normalized


def _coerce_agent_param_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    int_keys = {
        "max_workers",
        "max_subtasks",
        "max_retries",
        "max_replan",
        "tool_max_steps",
        "worker_result_forward_chars",
        "worker_evidence_forward_chars",
        "num_paths",
        "max_steps",
    }
    float_keys = {"worker_retry_threshold", "sample_temperature"}
    bool_keys = {"enable_tools_when_needed", "log_sc_paths"}

    for key in int_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = int(normalized[key])
    for key in float_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = float(normalized[key])
    for key in bool_keys:
        if key in normalized and normalized[key] is not None:
            normalized[key] = _coerce_bool(normalized[key])
    if "vote_key_strategy" in normalized and normalized["vote_key_strategy"] is not None:
        normalized["vote_key_strategy"] = str(normalized["vote_key_strategy"]).strip().lower()
    return normalized


def normalize_agent_config(
    *,
    model: str,
    dataset: str,
    kwargs: Mapping[str, Any] | None = None,
    strategy: str | None = None,
) -> NormalizedConfigResult:
    """
    Normalize raw kwargs into a canonical agent configuration.

    When ``strategy`` is set (e.g. ``"cot"``, ``"react"``), only that agent's
    param keys are placed in ``agent_params``; other agent-specific keys are
    listed in ``ignored_keys`` rather than polluting config or ``unknown_keys``.

    Notes:
    - This function is backward-compatible with legacy key names via CONFIG_ALIASES.
    - Runtime keys (e.g. ``expected_answer``) are excluded from ``unknown_keys``.
    """
    raw = dict(kwargs or {})
    alias_hits: dict[str, str] = {}
    normalized_items: dict[str, Any] = {}

    for key, value in raw.items():
        canonical = CONFIG_ALIASES.get(key, key)
        if canonical != key:
            alias_hits[key] = canonical
        normalized_items[canonical] = value

    normalized_items = _coerce_common_values(normalized_items)
    normalized_items = _coerce_agent_param_values(normalized_items)

    cfg = AgentConfig(
        model=str(normalized_items.get("model", model)),
        dataset=str(normalized_items.get("dataset", dataset)),
        temperature=float(normalized_items.get("temperature", 0.1)),
        max_tokens=int(normalized_items.get("max_tokens", 1024)),
        seed=normalized_items.get("seed"),
        agent_params={},
    )

    allowed: frozenset[str] | None = None
    if strategy is not None:
        allowed = STRATEGY_PARAM_KEYS.get(strategy)
        if allowed is None:
            raise ValueError(
                f"Unknown strategy {strategy!r}; "
                f"expected one of {sorted(STRATEGY_PARAM_KEYS)}"
            )

    ignored_keys: list[str] = []
    for key, value in normalized_items.items():
        if key in COMMON_CONFIG_KEYS or key in RUNTIME_KWARGS:
            continue
        if key not in AGENT_PARAM_KEYS:
            continue
        if allowed is not None:
            if key in allowed:
                cfg.agent_params[key] = value
            else:
                ignored_keys.append(key)
        else:
            cfg.agent_params[key] = value

    unknown_keys = sorted(
        k
        for k in normalized_items.keys()
        if k not in CANONICAL_CONFIG_KEYS and k not in RUNTIME_KWARGS
    )

    return NormalizedConfigResult(
        config=cfg,
        used_aliases=alias_hits,
        unknown_keys=unknown_keys,
        ignored_keys=sorted(ignored_keys),
    )
