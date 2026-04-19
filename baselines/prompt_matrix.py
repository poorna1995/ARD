# from __future__ import annotations

# from dataclasses import dataclass, field

# from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES

# DATASETS = ALLOWED_DATASETS
# MODALITIES = ALLOWED_MODALITIES


# @dataclass(frozen=True)
# class PromptTemplateSpec:
#     template_id: str
#     version: str
#     modality: str
#     dataset: str
#     required_variables: tuple[str, ...]
#     notes: str = ""


# @dataclass(frozen=True)
# class PromptMatrix:
#     system_templates: dict[tuple[str, str], PromptTemplateSpec] = field(default_factory=dict)
#     user_templates: dict[tuple[str, str], PromptTemplateSpec] = field(default_factory=dict)


# def build_prompt_matrix_v1() -> PromptMatrix:
#     """
#     Step-3 contract only:
#     - Defines template names and required variables.
#     - Does not include final prompt text yet.
#     """
#     system_templates: dict[tuple[str, str], PromptTemplateSpec] = {}
#     user_templates: dict[tuple[str, str], PromptTemplateSpec] = {}

#     for modality in MODALITIES:
#         for dataset in DATASETS:
#             key = (modality, dataset)

#             system_required = ["model_name", "output_contract"]
#             if modality in ("react", "multiagent"):
#                 system_required.extend(["max_steps", "tool_budget"])

#             system_templates[key] = PromptTemplateSpec(
#                 template_id=f"{modality}_{dataset}_system",
#                 version="v1",
#                 modality=modality,
#                 dataset=dataset,
#                 required_variables=tuple(system_required),
#                 notes="System-level behavior and output rules.",
#             )

#             user_required = ["query", "dataset"]
#             if dataset == "mmlu_pro":
#                 user_required.append("options")
#             if dataset in ("gaia", "math_hard", "swe_bench_verified"):
#                 user_required.append("ground_truth_format_hint")
#             if modality in ("react", "multiagent"):
#                 user_required.append("tools_available")

#             user_templates[key] = PromptTemplateSpec(
#                 template_id=f"{modality}_{dataset}_user",
#                 version="v1",
#                 modality=modality,
#                 dataset=dataset,
#                 required_variables=tuple(user_required),
#                 notes="Dataset-specific query packaging for the modality.",
#             )

#     return PromptMatrix(
#         system_templates=system_templates,
#         user_templates=user_templates,
#     )



from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES

DATASETS = ALLOWED_DATASETS
MODALITIES = ALLOWED_MODALITIES


@dataclass(frozen=True)
class PromptTemplateSpec:
    template_id: str
    version: str
    modality: str
    dataset: str
    required_variables: tuple[str, ...]
    notes: str = ""


@dataclass(frozen=True)
class PromptMatrix:
    """
    Truly immutable after construction.

    MappingProxyType wraps the dicts so callers cannot mutate the registry
    after build_prompt_matrix_v1() returns — consistent with frozen=True.
    """
    system_templates: Mapping[tuple[str, str], PromptTemplateSpec]
    user_templates: Mapping[tuple[str, str], PromptTemplateSpec]


def build_prompt_matrix_v1() -> PromptMatrix:
    """
    Build the prompt-spec registry for v1 templates.

    Required-variable sets mirror exactly what each (modality, dataset)
    template actually uses, so resolve_prompts() can catch missing values
    before .format() is called.
    """
    system_templates: dict[tuple[str, str], PromptTemplateSpec] = {}
    user_templates: dict[tuple[str, str], PromptTemplateSpec] = {}

    for modality in MODALITIES:
        for dataset in DATASETS:
            key = (modality, dataset)

            # ── system required variables ─────────────────────────────────
            # model_name is auto-injected by resolve_prompts(), but we still
            # declare it here so the spec is self-documenting.
            system_required = ["model_name", "output_contract"]
            if modality in ("react", "multiagent"):
                system_required.extend(["max_steps", "tool_budget"])

            system_templates[key] = PromptTemplateSpec(
                template_id=f"{modality}_{dataset}_system",
                version="v1",
                modality=modality,
                dataset=dataset,
                required_variables=tuple(system_required),
                notes="System-level behaviour and output rules.",
            )

            # ── user required variables ───────────────────────────────────
            user_required = ["query", "dataset"]

            # mmlu_pro always needs the option list
            if dataset == "mmlu_pro":
                user_required.append("options")

            # BUG FIX: ground_truth_format_hint is only used in *vanilla*
            # user prompts — zero_shot_cot / few_shot_cot / react / multiagent don't
            # reference it, so requiring it for those modalities would force
            # callers to supply an unused variable (and fail validation if
            # they don't).
            if modality in ("vanilla", "routellm") and dataset in (
                "gaia",
                "math_hard",
                "swe_bench_verified",
            ):
                user_required.append("ground_truth_format_hint")

            # react and multiagent templates always reference {tools_available}
            if modality in ("react", "multiagent"):
                user_required.append("tools_available")

            user_templates[key] = PromptTemplateSpec(
                template_id=f"{modality}_{dataset}_user",
                version="v1",
                modality=modality,
                dataset=dataset,
                required_variables=tuple(user_required),
                notes="Dataset-specific query packaging for the modality.",
            )

    # Wrap in MappingProxyType so the registry is read-only at runtime,
    # matching the frozen=True intent of PromptMatrix.
    return PromptMatrix(
        system_templates=MappingProxyType(system_templates),
        user_templates=MappingProxyType(user_templates),
    )