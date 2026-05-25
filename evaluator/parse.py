"""
Extract structured answers from raw LLM completions.

Contract (prompts.json_footer examples):
  {"answer": "...", "confidence": 0.0-1.0, "complexity": 0.0-1.0}

Used by all agents before grading (evaluator.is_correct).
"""

from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass
from typing import Any, Optional

# Full-string markdown fence
_FENCE_RE = re.compile(
    r"^```(?:json)?\s*\n?(.*?)\n?```\s*$",
    re.DOTALL | re.IGNORECASE,
)

_DEFAULT_ANSWER_KEYS: tuple[str, ...] = ("answer",)
_PLACEHOLDER_ANSWERS = frozenset(
    {
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "...",
        "<value>",
        "<short value>",
        "<answer>",
        "value",
        "answer",
        "your answer",
        "your answer here",
        "insert answer",
        "tbd",
    }
)


def optional_float(v: object) -> Optional[float]:
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def _strip_markdown_fences(text: str) -> str:
    t = text.strip()
    m = _FENCE_RE.match(t)
    if m:
        return m.group(1).strip()
    if t.startswith("```"):
        t = re.sub(r"^```(?:json)?\s*", "", t, count=1, flags=re.IGNORECASE)
        t = re.sub(r"\s*```\s*$", "", t)
    return t.strip()


def _balanced_object_spans(text: str) -> list[tuple[int, int]]:
    """Return (start, end_exclusive) spans of top-level `{...}` objects."""
    spans: list[tuple[int, int]] = []
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_string = False
        escape = False
        quote = ""
        j = i
        while j < n:
            ch = text[j]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == quote:
                    in_string = False
            else:
                if ch in ('"', "'"):
                    in_string = True
                    quote = ch
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        spans.append((i, j + 1))
                        j += 1
                        break
            j += 1
        i = j if j > i else i + 1
    return spans


def _loads_dict(blob: str) -> dict[str, Any] | None:
    blob = blob.strip()
    if not blob.startswith("{"):
        return None
    try:
        obj = json.loads(blob)
        if isinstance(obj, dict):
            return obj
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    try:
        obj = ast.literal_eval(blob)
        if isinstance(obj, dict):
            return obj
    except (ValueError, SyntaxError, TypeError):
        pass
    return None


def extract_last_json_dict(text: str | None) -> dict[str, Any] | None:
    """Last brace-balanced JSON/Python dict in ``text``, or None."""
    if text is None:
        return None
    t = _strip_markdown_fences(str(text).strip())
    if not t:
        return None

    whole = _loads_dict(t)
    if whole is not None:
        return whole

    spans = _balanced_object_spans(t)
    for start, end in reversed(spans):
        obj = _loads_dict(t[start:end])
        if obj is not None:
            return obj
    return None


def _fields_from_dict(
    obj: dict[str, Any],
    answer_keys: tuple[str, ...],
) -> tuple[str, Optional[float], Optional[float]] | None:
    for key in answer_keys:
        if key not in obj:
            continue
        raw_ans = obj[key]
        if raw_ans is None:
            continue
        answer = str(raw_ans).strip()
        if not answer:
            # Key present but empty — structured completion with no value.
            return (
                "",
                optional_float(obj.get("confidence")),
                optional_float(obj.get("complexity")),
            )
        if answer.lower() in _PLACEHOLDER_ANSWERS:
            continue
        return (
            answer,
            optional_float(obj.get("confidence")),
            optional_float(obj.get("complexity")),
        )
    return None


def _try_parse_structured(
    text: str,
    answer_keys: tuple[str, ...],
) -> tuple[str, Optional[float], Optional[float]] | None:
    whole = _loads_dict(text)
    if whole is not None:
        hit = _fields_from_dict(whole, answer_keys)
        if hit is not None:
            return hit

    for start, end in reversed(_balanced_object_spans(text)):
        obj = _loads_dict(text[start:end])
        if obj is None:
            continue
        hit = _fields_from_dict(obj, answer_keys)
        if hit is not None:
            return hit
    return None


@dataclass(frozen=True)
class ParsedLLMOutput:
    predicted_answer: str
    confidence: Optional[float]
    complexity: Optional[float]
    structured: bool  # True when extracted from JSON with an answer key


def parse_llm_output_detailed(
    raw: str | None,
    *,
    answer_keys: tuple[str, ...] = _DEFAULT_ANSWER_KEYS,
) -> ParsedLLMOutput:
    """
    Extract fields from an LLM completion.

    ``structured`` is True only when an ``answer_keys`` field was found in JSON.
    Otherwise ``predicted_answer`` is the stripped raw text (plain-text fallback).
    """
    if raw is None:
        return ParsedLLMOutput("", None, None, False)

    text = _strip_markdown_fences(str(raw).strip())
    if not text:
        return ParsedLLMOutput("", None, None, False)

    hit = _try_parse_structured(text, answer_keys)
    if hit is not None:
        ans, conf, comp = hit
        return ParsedLLMOutput(ans, conf, comp, True)

    return ParsedLLMOutput(text, None, None, False)


def final_json_missing_answer_field(raw: str | None) -> bool:
    """True when text contains a JSON object but no ``answer`` key."""
    obj = extract_last_json_dict(raw)
    if obj is None:
        return False
    return "answer" not in obj


def final_json_placeholder_answer(raw: str | None) -> bool:
    """True when JSON has an ``answer`` key whose value is empty or a placeholder."""
    obj = extract_last_json_dict(raw)
    if obj is None or "answer" not in obj:
        return False
    raw_ans = obj["answer"]
    if raw_ans is None:
        return True
    answer = str(raw_ans).strip()
    if not answer:
        return True
    return answer.lower() in _PLACEHOLDER_ANSWERS


def parse_llm_output(
    raw: str | None,
    *,
    answer_keys: tuple[str, ...] = _DEFAULT_ANSWER_KEYS,
) -> tuple[str, Optional[float], Optional[float]]:
    """Extract (predicted_answer, confidence, complexity) from an LLM completion."""
    p = parse_llm_output_detailed(raw, answer_keys=answer_keys)
    return p.predicted_answer, p.confidence, p.complexity


def parse_react_final_json(
    body: str | None,
    *,
    answer_keys: tuple[str, ...] = _DEFAULT_ANSWER_KEYS,
) -> ParsedLLMOutput | None:
    """
    ReAct Final Answer body: one JSON object only.

    No plain-text fallback. Returns None if the body is not exclusively
    a single structured object with a valid answer field.
    """
    if body is None:
        return None
    text = _strip_markdown_fences(str(body).strip())
    if not text.startswith("{"):
        return None

    spans = _balanced_object_spans(text)
    if len(spans) != 1:
        return None
    start, end = spans[0]
    if text[:start].strip() or text[end:].strip():
        return None

    obj = _loads_dict(text[start:end])
    if obj is None:
        return None
    hit = _fields_from_dict(obj, answer_keys)
    if hit is None:
        return None
    ans, conf, comp = hit
    return ParsedLLMOutput(ans, conf, comp, True)


def extract_reasoning_steps(raw_answer: str) -> list[str]:
    """CoT/SC: lines before the first line that looks like answer JSON."""
    steps: list[str] = []
    for line in raw_answer.strip().split("\n"):
        stripped = line.strip()
        if stripped.startswith("{") and "answer" in stripped.lower():
            break
        if stripped:
            steps.append(stripped)
    return steps
