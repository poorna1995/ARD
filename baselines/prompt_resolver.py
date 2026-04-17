from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from baselines.constants import (
    validate_dataset,
    validate_modality,
    validate_model,
)
from baselines.prompt_matrix import build_prompt_matrix_v1
from baselines.prompt_templates import SYSTEM_PROMPTS_V1, USER_PROMPTS_V1


@dataclass(frozen=True)
class ResolvedPrompt:
    system_prompt: str
    user_prompt: str
    system_prompt_version: str
    user_prompt_version: str
    prompt_hash: str


def _require_variables(
    required_variables: tuple[str, ...],
    values: dict[str, Any],
    label: str,
) -> None:
    missing = [name for name in required_variables if name not in values]
    if missing:
        raise ValueError(f"Missing required {label} variables: {missing}")


def resolve_prompts(
    *,
    modality: str,
    dataset: str,
    model_name: str,
    system_values: dict[str, Any],
    user_values: dict[str, Any],
) -> ResolvedPrompt:
    validate_modality(modality)
    validate_dataset(dataset)
    validate_model(model_name)

    key = (modality, dataset)
    matrix = build_prompt_matrix_v1()

    if key not in SYSTEM_PROMPTS_V1 or key not in USER_PROMPTS_V1:
        raise ValueError(f"No prompt templates found for key={key}")
    if key not in matrix.system_templates or key not in matrix.user_templates:
        raise ValueError(f"No prompt specs found for key={key}")

    system_spec = matrix.system_templates[key]
    user_spec = matrix.user_templates[key]

    _require_variables(system_spec.required_variables, system_values, "system")
    _require_variables(user_spec.required_variables, user_values, "user")

    system_prompt = SYSTEM_PROMPTS_V1[key].format(**system_values)
    user_prompt = USER_PROMPTS_V1[key].format(**user_values)

    prompt_hash = hashlib.sha256(
        f"{modality}|{dataset}|{system_prompt}\n---\n{user_prompt}".encode("utf-8")
    ).hexdigest()

    return ResolvedPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        system_prompt_version=system_spec.version,
        user_prompt_version=user_spec.version,
        prompt_hash=prompt_hash,
    )
