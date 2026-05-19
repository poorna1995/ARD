# Grading and answer matching.
# Parsing: evaluator.parse | Dataset matchers: evaluator.match
from __future__ import annotations

from typing import Optional, Tuple

from evaluator.match import canonicalise_for_dataset, grade
from evaluator.parse import optional_float, parse_llm_output

# Backward-compatible alias: default QA canonicalisation when dataset omitted.
canonicalise_answer = canonicalise_for_dataset


def normalise(text: str) -> Tuple[str, Optional[float], Optional[float]]:
    """Parse a completion fragment (alias of :func:`parse_llm_output`)."""
    return parse_llm_output(text)


def normalise_answer(raw: str | None) -> Tuple[str, Optional[float], Optional[float]]:
    """Extract answer string plus optional confidence/complexity from LLM output."""
    return parse_llm_output(raw)


def is_correct(
    predicted_answer: str,
    expected: str,
    dataset: str | None = None,
) -> bool:
    """Compare model prediction to gold using dataset-appropriate rules."""
    return grade(predicted_answer, expected, dataset=dataset)


parse_agent_output = parse_llm_output

__all__ = [
    "canonicalise_answer",
    "canonicalise_for_dataset",
    "grade",
    "is_correct",
    "normalise",
    "normalise_answer",
    "optional_float",
    "parse_agent_output",
    "parse_llm_output",
]
