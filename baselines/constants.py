from __future__ import annotations


ALLOWED_DATASETS: tuple[str, ...] = (
    "gaia",
    "mmlu_pro",
    "math_hard",
    "swe_bench_verified",
)

ALLOWED_MODALITIES: tuple[str, ...] = (
    "vanilla",
    "zero_shot_cot",
    "few_shot_cot",
    "react",
    "multiagent",
    "routellm",
)

ALLOWED_MODELS: tuple[str, ...] = (
    "gpt-4o-mini",
    "gpt-4o",
    "routellm",
)


def validate_dataset(dataset: str) -> None:
    if dataset not in ALLOWED_DATASETS:
        raise ValueError(
            f"Invalid dataset '{dataset}'. Allowed values: {ALLOWED_DATASETS}"
        )


def validate_modality(modality: str) -> None:
    if modality not in ALLOWED_MODALITIES:
        raise ValueError(
            f"Invalid modality '{modality}'. Allowed values: {ALLOWED_MODALITIES}"
        )


def validate_model(model: str) -> None:
    if model not in ALLOWED_MODELS:
        raise ValueError(
            f"Invalid model '{model}'. Allowed values: {ALLOWED_MODELS}"
        )
