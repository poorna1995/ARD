# from __future__ import annotations

# import hashlib
# from dataclasses import dataclass
# from typing import Any

# from baselines.constants import (
#     validate_dataset,
#     validate_modality,
#     validate_model,
# )
# from baselines.prompt_matrix import build_prompt_matrix_v1
# from baselines.prompt_templates import SYSTEM_PROMPTS_V1, USER_PROMPTS_V1


# @dataclass(frozen=True)
# class ResolvedPrompt:
#     system_prompt: str
#     user_prompt: str
#     system_prompt_version: str
#     user_prompt_version: str
#     prompt_hash: str


# def _require_variables(
#     required_variables: tuple[str, ...],
#     values: dict[str, Any],
#     label: str,
# ) -> None:
#     missing = [name for name in required_variables if name not in values]
#     if missing:
#         raise ValueError(f"Missing required {label} variables: {missing}")


# def resolve_prompts(
#     *,
#     modality: str,
#     dataset: str,
#     model_name: str,
#     system_values: dict[str, Any],
#     user_values: dict[str, Any],
# ) -> ResolvedPrompt:
#     validate_modality(modality)
#     validate_dataset(dataset)
#     validate_model(model_name)

#     key = (modality, dataset)
#     matrix = build_prompt_matrix_v1()

#     if key not in SYSTEM_PROMPTS_V1 or key not in USER_PROMPTS_V1:
#         raise ValueError(f"No prompt templates found for key={key}")
#     if key not in matrix.system_templates or key not in matrix.user_templates:
#         raise ValueError(f"No prompt specs found for key={key}")

#     system_spec = matrix.system_templates[key]
#     user_spec = matrix.user_templates[key]

#     _require_variables(system_spec.required_variables, system_values, "system")
#     _require_variables(user_spec.required_variables, user_values, "user")

#     system_prompt = SYSTEM_PROMPTS_V1[key].format(**system_values)
#     user_prompt = USER_PROMPTS_V1[key].format(**user_values)

#     prompt_hash = hashlib.sha256(
#         f"{modality}|{dataset}|{system_prompt}\n---\n{user_prompt}".encode("utf-8")
#     ).hexdigest()

#     return ResolvedPrompt(
#         system_prompt=system_prompt,
#         user_prompt=user_prompt,
#         system_prompt_version=system_spec.version,
#         user_prompt_version=user_spec.version,
#         prompt_hash=prompt_hash,
#     )



from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from baselines.constants import (
    validate_dataset,
    validate_modality,
    validate_model,
)
from baselines.prompt_matrix import build_prompt_matrix_v1
from baselines.prompt_templates import SYSTEM_PROMPTS_V1, USER_PROMPTS_V1


# ---------------------------------------------------------------------------
# Build the matrix once at import time — not on every resolve_prompts() call.
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def _get_matrix():
    return build_prompt_matrix_v1()


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ResolvedPrompt:
    system_prompt: str
    user_prompt: str
    system_prompt_version: str
    user_prompt_version: str
    prompt_hash: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _require_variables(
    required_variables: tuple[str, ...],
    values: dict[str, Any],
    label: str,
) -> None:
    missing = [name for name in required_variables if name not in values]
    if missing:
        raise ValueError(f"Missing required {label} variables: {missing}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def resolve_prompts(
    *,
    modality: str,
    dataset: str,
    model_name: str,
    system_values: dict[str, Any],
    user_values: dict[str, Any],
) -> ResolvedPrompt:
    """Resolve, validate, and format prompts for a given (modality, dataset) key.

    ``model_name`` is automatically injected into ``system_values`` so callers
    do not have to pass it twice.  An explicit value in ``system_values`` takes
    precedence if you need to override it.
    """
    # 1. Validate inputs
    validate_modality(modality)
    validate_dataset(dataset)
    validate_model(model_name)

    key = (modality, dataset)

    # 2. Single source-of-truth check — the matrix is built from the same
    #    templates, so checking only one dict is sufficient.
    if key not in SYSTEM_PROMPTS_V1:
        raise KeyError(f"No system prompt template found for key={key}")
    if key not in USER_PROMPTS_V1:
        raise KeyError(f"No user prompt template found for key={key}")

    # 3. Fetch specs (versions + required_variables) from the cached matrix
    matrix = _get_matrix()
    system_spec = matrix.system_templates[key]
    user_spec = matrix.user_templates[key]

    # 4. Auto-inject model_name so callers don't have to put it in system_values
    #    manually (system templates always reference {model_name}).
    system_values = {**system_values, "model_name": model_name}

    # 5. Validate all required variables are present before formatting
    _require_variables(system_spec.required_variables, system_values, "system")
    _require_variables(user_spec.required_variables, user_values, "user")

    # 6. Format
    system_prompt = SYSTEM_PROMPTS_V1[key].format(**system_values)
    user_prompt = USER_PROMPTS_V1[key].format(**user_values)

    # 7. Stable hash — includes model_name so different models produce
    #    different hashes even with identical prompt text.
    prompt_hash = hashlib.sha256(
        f"{modality}|{dataset}|{model_name}|{system_prompt}\n---\n{user_prompt}"
        .encode("utf-8")
    ).hexdigest()

    return ResolvedPrompt(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        system_prompt_version=system_spec.version,
        user_prompt_version=user_spec.version,
        prompt_hash=prompt_hash,
    )