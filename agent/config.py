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

AGENT_PARAM_KEYS: set[str] = MULTIAGENT_CONFIG_KEYS | SELF_CONSISTENCY_CONFIG_KEYS

CANONICAL_CONFIG_KEYS: set[str] = COMMON_CONFIG_KEYS | AGENT_PARAM_KEYS


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


def _coerce_multiagent_values(values: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(values)
    int_keys = {
        "max_workers",
        "max_subtasks",
        "max_retries",
        "max_replan",
        "tool_max_steps",
        "worker_result_forward_chars",
        "worker_evidence_forward_chars",
    }
    float_keys = {"worker_retry_threshold"}
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
    if "num_paths" in normalized and normalized["num_paths"] is not None:
        normalized["num_paths"] = int(normalized["num_paths"])
    if "sample_temperature" in normalized and normalized["sample_temperature"] is not None:
        normalized["sample_temperature"] = float(normalized["sample_temperature"])
    if "vote_key_strategy" in normalized and normalized["vote_key_strategy"] is not None:
        normalized["vote_key_strategy"] = str(normalized["vote_key_strategy"]).strip().lower()
    return normalized


def normalize_agent_config(
    *,
    model: str,
    dataset: str,
    kwargs: Mapping[str, Any] | None = None,
) -> NormalizedConfigResult:
    """
    Normalize raw kwargs into a canonical agent configuration.

    Notes:
    - This function is backward-compatible with legacy key names via CONFIG_ALIASES.
    - Non-config runtime keys (e.g., expected_answer) remain unknown by design and
      should be handled separately by caller code.
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
    normalized_items = _coerce_multiagent_values(normalized_items)

    cfg = AgentConfig(
        model=str(normalized_items.get("model", model)),
        dataset=str(normalized_items.get("dataset", dataset)),
        temperature=float(normalized_items.get("temperature", 0.1)),
        max_tokens=int(normalized_items.get("max_tokens", 1024)),
        seed=normalized_items.get("seed"),
        agent_params={},
    )

    for key, value in normalized_items.items():
        if key not in COMMON_CONFIG_KEYS and key in AGENT_PARAM_KEYS:
            cfg.agent_params[key] = value

    unknown_keys = sorted(k for k in normalized_items.keys() if k not in CANONICAL_CONFIG_KEYS)

    return NormalizedConfigResult(
        config=cfg,
        used_aliases=alias_hits,
        unknown_keys=unknown_keys,
    )
