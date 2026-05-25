"""
Evaluator public API.

- ``parse`` — extract structured fields from LLM completions
- ``grade`` — dataset-aware normalization and correctness checks
"""

from __future__ import annotations

from evaluator.grade import (
    apply_gaia_token_overrides,
    canonicalise_answer,
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
    is_correct,
    is_equiv,
    normalize_for_dataset,
    normalize_gaia_string,
    normalize_hotpot_answer,
    normalize_math_answer,
    normalize_mmlu_answer,
    normalize_musique_answer,
    strip_string,
)
from evaluator.parse import (
    ParsedLLMOutput,
    extract_last_json_dict,
    extract_reasoning_steps,
    optional_float,
    parse_llm_output,
    parse_llm_output_detailed,
)

# Legacy names
normalise = parse_llm_output
normalise_answer = parse_llm_output
parse_agent_output = parse_llm_output

__all__ = [
    "ParsedLLMOutput",
    "apply_gaia_token_overrides",
    "canonicalise_answer",
    "canonicalise_for_dataset",
    "canonicalise_gaia",
    "canonicalise_hotpot",
    "canonicalise_mmlu",
    "canonicalise_musique",
    "canonicalise_qa",
    "extract_last_json_dict",
    "extract_reasoning_steps",
    "grade",
    "grade_gaia_nem",
    "grade_hotpot_nem",
    "grade_math",
    "grade_mmlu_nem",
    "grade_musique_nem",
    "is_correct",
    "is_equiv",
    "normalise",
    "normalise_answer",
    "normalize_for_dataset",
    "normalize_gaia_string",
    "normalize_hotpot_answer",
    "normalize_math_answer",
    "normalize_mmlu_answer",
    "normalize_musique_answer",
    "optional_float",
    "parse_agent_output",
    "parse_llm_output",
    "parse_llm_output_detailed",
    "strip_string",
]
