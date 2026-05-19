"""
Dataset-aware answer matching (grading only — parsing is evaluator.parse).

All dataset-specific normalizers live in ``evaluator.dataset_normalize``.
"""

from __future__ import annotations

from evaluator.dataset_normalize import (
    canonicalise_for_dataset,
    canonicalise_gaia,
    canonicalise_hotpot,
    canonicalise_mmlu,
    canonicalise_musique,
    canonicalise_qa,
    grade,
    grade_gaia_nem,
    grade_hotpot_nem,
    grade_math,
    grade_mmlu_nem,
    grade_musique_nem,
    normalize_for_dataset,
    normalize_gaia_string,
    normalize_hotpot_answer,
    normalize_math_answer,
    normalize_mmlu_answer,
    normalize_musique_answer,
)

__all__ = [
    "canonicalise_for_dataset",
    "canonicalise_gaia",
    "canonicalise_hotpot",
    "canonicalise_mmlu",
    "canonicalise_musique",
    "canonicalise_qa",
    "grade",
    "grade_gaia_nem",
    "grade_hotpot_nem",
    "grade_math",
    "grade_mmlu_nem",
    "grade_musique_nem",
    "normalize_for_dataset",
    "normalize_gaia_string",
    "normalize_hotpot_answer",
    "normalize_math_answer",
    "normalize_mmlu_answer",
    "normalize_musique_answer",
]
