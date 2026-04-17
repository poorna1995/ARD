from __future__ import annotations

from dataclasses import dataclass, field

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
    system_templates: dict[tuple[str, str], PromptTemplateSpec] = field(default_factory=dict)
    user_templates: dict[tuple[str, str], PromptTemplateSpec] = field(default_factory=dict)


def build_prompt_matrix_v1() -> PromptMatrix:
    """
    Step-3 contract only:
    - Defines template names and required variables.
    - Does not include final prompt text yet.
    """
    system_templates: dict[tuple[str, str], PromptTemplateSpec] = {}
    user_templates: dict[tuple[str, str], PromptTemplateSpec] = {}

    for modality in MODALITIES:
        for dataset in DATASETS:
            key = (modality, dataset)

            system_required = ["model_name", "output_contract"]
            if modality in ("react", "multiagent"):
                system_required.extend(["max_steps", "tool_budget"])

            system_templates[key] = PromptTemplateSpec(
                template_id=f"{modality}_{dataset}_system",
                version="v1",
                modality=modality,
                dataset=dataset,
                required_variables=tuple(system_required),
                notes="System-level behavior and output rules.",
            )

            user_required = ["query", "dataset"]
            if dataset == "mmlu_pro":
                user_required.append("options")
            if dataset in ("gaia", "math_hard", "swe_bench_verified"):
                user_required.append("ground_truth_format_hint")
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

    return PromptMatrix(
        system_templates=system_templates,
        user_templates=user_templates,
    )
