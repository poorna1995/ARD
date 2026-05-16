"""
LaTeX answer normalization for Hendrycks MATH (ported from hendrycks/math).

Standalone module (no heavy src.utils imports) for loaders, scripts, and eval.
"""

from __future__ import annotations

import re
from typing import Optional

_MATH_DATASETS = frozenset({"math", "math_hard"})


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        for substr in substrs[1:]:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                if len(substr) < 2:
                    return string
                a, b = substr[0], substr[1]
                if b != "{":
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}{" + b + "}" + post
                else:
                    post = substr[2:] if len(substr) > 2 else ""
                    new_str += "{" + a + "}" + b + post
    return new_str


def _fix_a_slash_b(string: str) -> str:
    parts = string.split("/")
    if len(parts) != 2:
        return string
    a, b = parts[0], parts[1]
    try:
        ai, bi = int(a), int(b)
        if string == f"{ai}/{bi}":
            return f"\\frac{{{ai}}}{{{bi}}}"
    except ValueError:
        pass
    return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        if len(splits) == 2:
            return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split and split[0] != "{":
            new_string += "\\sqrt{" + split[0] + "}" + split[1:]
        else:
            new_string += "\\sqrt" + split
    return new_string


def strip_math_string(string: Optional[str]) -> str:
    """Canonical LaTeX string form used by the MATH benchmark for exact match."""
    if string is None:
        return ""
    string = str(string).strip()
    if not string:
        return ""

    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(r"\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if not string:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)

    if string == "0.5":
        string = "\\frac{1}{2}"

    return _fix_a_slash_b(string)


_BOXED_RE = re.compile(r"\\boxed\s*\{")


def extract_last_boxed(text: str) -> Optional[str]:
    """Return inner content of the last \\boxed{...} (handles nested braces)."""
    last: Optional[str] = None
    for m in _BOXED_RE.finditer(text):
        i, depth = m.end(), 1
        while i < len(text) and depth:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1].strip()
    return last


def normalize_math_answer(raw: Optional[str]) -> str:
    """Strip \\boxed{} if present, then apply MATH strip_string normalization."""
    if raw is None:
        return ""
    text = str(raw).strip()
    if not text:
        return ""
    inner = extract_last_boxed(text)
    if inner is not None:
        text = inner
    return strip_math_string(text)


def is_math_equiv(pred: Optional[str], expected: Optional[str]) -> bool:
    """Exact match after MATH LaTeX normalization."""
    try:
        return normalize_math_answer(pred) == normalize_math_answer(expected)
    except Exception:
        return str(pred or "").strip() == str(expected or "").strip()


def is_math_dataset(dataset: Optional[str]) -> bool:
    return (dataset or "") in _MATH_DATASETS
