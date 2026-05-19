"""
LaTeX answer normalization for Hendrycks MATH (ported from hendrycks/math).

Standalone module (no heavy src.utils imports) for loaders, scripts, and eval.
"""

from __future__ import annotations

import re
from typing import Optional

_MATH_DATASETS = frozenset({"math", "math_hard"})

# MATH / sympy-style base suffix: ``1_6`` = digit string ``1`` in base 6 (value 1), not 16.
_BASE_INT_RE = re.compile(r"^([0-9]+)_(\d+)$")


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


def parse_base_suffixed_int(text: Optional[str]) -> Optional[tuple[int, int]]:
    """
    Parse ``digits_base`` answers (e.g. ``1_6`` → value ``1``, base ``6``).

    This is MATH competition notation for integers in a given base, not Python's
    numeric underscore grouping (and not ``sympy.sympify``, which reads ``1_6`` as 16).
    """
    s = str(text or "").strip().replace(" ", "")
    if not s:
        return None
    m = _BASE_INT_RE.match(s)
    if not m:
        return None
    digits, base_s = m.group(1), m.group(2)
    base = int(base_s)
    if base < 2:
        return None
    try:
        return int(digits, base), base
    except ValueError:
        return None


_ASCII_SQRT_RE = re.compile(r"sqrt\s*\(", re.IGNORECASE)


def _ascii_sqrt_to_latex(s: str) -> str:
    """Turn ``sqrt(10)`` into ``\\sqrt{10}`` for MATH ``strip_string`` / ``is_equiv``."""
    out: list[str] = []
    i = 0
    n = len(s)
    while i < n:
        m = _ASCII_SQRT_RE.match(s, i)
        if not m:
            out.append(s[i])
            i += 1
            continue
        depth = 1
        j = m.end()
        while j < n and depth:
            if s[j] == "(":
                depth += 1
            elif s[j] == ")":
                depth -= 1
            j += 1
        if depth != 0:
            out.append(s[i])
            i += 1
            continue
        inner = s[m.end() : j - 1].strip()
        out.append("\\sqrt{" + inner + "}")
        i = j
    return "".join(out)


_THOUSAND_COMMA_RE = re.compile(r"(?<=\d),(?=\d)")


def preprocess_math_notation(raw: Optional[str]) -> str:
    """
    Surface normalizations before ``is_equiv`` (commas, ascii sqrt, ``*``, ``pi``).

    Keeps upstream ``is_equiv`` / ``strip_string`` as the final authority; this only
    aligns common model/gold formatting differences (see project grading notes).
    """
    if raw is None:
        return ""
    s = str(raw).strip()
    if not s:
        return ""
    s = _THOUSAND_COMMA_RE.sub("", s)
    s = _ascii_sqrt_to_latex(s)
    # ``28*\\sqrt{3}`` or ``100*\\pi`` → drop ``*`` between digit and ``\\``
    s = re.sub(r"(\d)\s*\*\s*\\", r"\1\\", s)
    # ``100*pi`` / ``100 * pi`` → ``100\\pi``
    s = re.sub(r"(\d)\s*\*\s*pi\b", r"\1\\pi", s, flags=re.IGNORECASE)
    # bare ``pi`` token (not ``\\pi``, not inside a longer word)
    s = re.sub(r"(?<!\\)\bpi\b", r"\\pi", s, flags=re.IGNORECASE)
    return s


def _base_int_equiv(pred: Optional[str], expected: Optional[str]) -> Optional[bool]:
    """Compare base-suffixed integers; bare digits use the suffixed side's base."""
    p = str(pred or "").strip()
    e = str(expected or "").strip()
    if not p or not e:
        return None

    pe, ee = parse_base_suffixed_int(p), parse_base_suffixed_int(e)
    if pe is not None and ee is not None:
        return pe[0] == ee[0]
    if ee is not None and pe is None and p.isdigit():
        val_e, base = ee
        try:
            return int(p, base) == val_e
        except ValueError:
            return False
    if pe is not None and ee is None and e.isdigit():
        val_p, base = pe
        try:
            return int(e, base) == val_p
        except ValueError:
            return False
    return None


def is_math_equiv(pred: Optional[str], expected: Optional[str]) -> bool:
    """
    MATH grading: surface normalize, then hendrycks ``is_equiv``, then base-suffix ints.

    Base suffix (e.g. gold ``1_6``, pred ``1``) is MATH dataset notation, not in the
    upstream repo but required for this project's gold answers.
    """
    from evaluator.math_equivalence import is_equiv

    p = preprocess_math_notation(pred) if pred is not None else ""
    e = preprocess_math_notation(expected) if expected is not None else ""
    if is_equiv(p or None, e or None):
        return True
    base_result = _base_int_equiv(pred, expected)
    if base_result is not None:
        return base_result
    return False


def is_math_dataset(dataset: Optional[str]) -> bool:
    return (dataset or "") in _MATH_DATASETS
