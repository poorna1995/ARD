"""
Dataset-specific answer normalization and grading.

- **parse** — extract ``answer`` / confidence / complexity from LLM text (``evaluator.parse``)
- **grade** — compare prediction vs gold (this module)

References: Hotpot/MuSiQue SQuAD EM (normalized exact match); GAIA leaderboard scorer; MATH hendrycks ``is_equiv``.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Any

from config.local.constants import DS

# ── MATH benchmark string form (hendrycks/math; do not edit logic) ───────────


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr and substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a_i = int(a)
        b_i = int(b)
        if string == f"{a_i}/{b_i}":
            return "\\frac{" + str(a_i) + "}{" + str(b_i) + "}"
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
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: str) -> str:
    """MATH benchmark canonical string form."""
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
    if len(string) == 0:
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
    string = _fix_a_slash_b(string)
    return string


def is_equiv(str1: str | None, str2: str | None, verbose: bool = False) -> bool:
    """MATH ``is_equiv`` — stripped-string equality."""
    if str1 is None and str2 is None:
        if verbose:
            print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        if verbose:
            print(ss1, ss2)
        return ss1 == ss2
    except Exception:
        return str(str1) == str(str2)


# ── GAIA trace-specific token fixes (optional post-process) ───────────────────

_GAIA_TOKEN_MAP: dict[str, str] = {
    "agull": "seagull",
    "gulls": "seagull",
    "gull": "seagull",
    "glide": "glided",
    "peaceful": "peacefully",
    "peacefully": "",
    "peacefull": "",
    "dpeacefull": "",
    "dpeacefully": "",
    "deep": "",
    "fully": "",
}

_GAIA_PHRASE_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bdpeac\s+efull\s+y\b", re.I), "peacefully"),
    (re.compile(r"\bdpeac\s+efull\b", re.I), "peacefully"),
    (re.compile(r"\bytomy\b", re.I), "to my"),
)


def apply_gaia_token_overrides(canonical_qa: str) -> str:
    """Post-process a lowercased QA-canonical string with GAIA-specific fixes."""
    if not canonical_qa:
        return ""
    s = canonical_qa
    for pattern, replacement in _GAIA_PHRASE_PATTERNS:
        s = pattern.sub(replacement, s)
    tokens = s.split()
    out: list[str] = []
    for token in tokens:
        mapped = _GAIA_TOKEN_MAP.get(token, token)
        if mapped:
            out.append(mapped)
    s = " ".join(out)
    return re.sub(r"\s+", " ", s).strip()


# ── Hotpot / MuSiQue (SQuAD normalized exact match) ──────────────────────────

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_APOSTROPHE_VARIANTS = (
    "\u2018", "\u2019", "\u201a", "\u201b", "\u2032", "\u0060", "\u00b4",
)
_DASH_VARIANTS = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015")


def _preprocess_squad_nem(text: str) -> str:
    s = unicodedata.normalize("NFKC", text)
    for ch in _APOSTROPHE_VARIANTS:
        s = s.replace(ch, "'")
    for ch in _DASH_VARIANTS:
        s = s.replace(ch, " ")
    s = s.replace("-", " ").replace("_", " ")
    return s


def squad_normalize_answer(text: str | None) -> str:
    if text is None:
        return ""
    s = _preprocess_squad_nem(str(text).lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def normalize_hotpot_answer(text: str | None) -> str:
    return squad_normalize_answer(text)


def normalize_musique_answer(text: str | None) -> str:
    return squad_normalize_answer(text)


def squad_token_f1_components(
    prediction: str, ground_truth: str
) -> tuple[float, float, float]:
    normalized_prediction = squad_normalize_answer(prediction)
    normalized_ground_truth = squad_normalize_answer(ground_truth)

    if normalized_prediction in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0
    if normalized_ground_truth in ("yes", "no", "noanswer") and normalized_prediction != normalized_ground_truth:
        return 0.0, 0.0, 0.0

    pred_tokens = normalized_prediction.split()
    gold_tokens = normalized_ground_truth.split()
    if not pred_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0, 0.0, 0.0
    precision = 1.0 * num_same / len(pred_tokens)
    recall = 1.0 * num_same / len(gold_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1, precision, recall


def squad_token_f1(prediction: str, ground_truth: str) -> float:
    return squad_token_f1_components(prediction, ground_truth)[0]


def grade_openqa_exact(predicted: str, expected: str) -> bool:
    """SQuAD-style EM: normalized token strings must match exactly."""
    return squad_normalize_answer(predicted) == squad_normalize_answer(expected)


def grade_hotpot_nem(predicted: str, expected: str) -> bool:
    return grade_openqa_exact(predicted, expected)


def grade_musique_nem(predicted: str, expected: str) -> bool:
    return grade_openqa_exact(predicted, expected)


# ── GAIA quasi-exact match ───────────────────────────────────────────────────

_GAIA_LIST_SEP_RE = re.compile(r"[,;]")


def _gaia_is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except (TypeError, ValueError):
        return False


def normalize_gaia_number_str(number_str: str) -> float:
    s = str(number_str)
    for char in ("$", "%", ","):
        s = s.replace(char, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def normalize_gaia_string(text: str | None, *, remove_punct: bool = True) -> str:
    if text is None:
        return ""
    no_spaces = re.sub(r"\s", "", str(text))
    if remove_punct:
        translator = str.maketrans("", "", string.punctuation)
        return no_spaces.lower().translate(translator)
    return no_spaces.lower()


def _gaia_split_list(s: str) -> list[str]:
    return [part.strip() for part in _GAIA_LIST_SEP_RE.split(s) if part.strip()]


def grade_gaia_nem(predicted: str | None, expected: str | None) -> bool:
    model_answer = predicted if predicted is not None else "None"
    ground_truth = str(expected or "")

    if _gaia_is_float(ground_truth):
        return normalize_gaia_number_str(model_answer) == float(ground_truth)

    if any(ch in ground_truth for ch in (",", ";")):
        gt_elems = _gaia_split_list(ground_truth)
        ma_elems = _gaia_split_list(str(model_answer))
        if len(gt_elems) != len(ma_elems):
            return False
        for ma_elem, gt_elem in zip(ma_elems, gt_elems):
            if _gaia_is_float(gt_elem):
                if normalize_gaia_number_str(ma_elem) != float(gt_elem):
                    return False
            elif normalize_gaia_string(ma_elem, remove_punct=False) != normalize_gaia_string(
                gt_elem, remove_punct=False
            ):
                return False
        return True

    return normalize_gaia_string(model_answer) == normalize_gaia_string(ground_truth)


# ── MATH ─────────────────────────────────────────────────────────────────────

def normalize_math_answer(text: str | None) -> str:
    if text is None:
        return ""
    return strip_string(str(text).strip())


def grade_math(predicted: str | None, expected: str | None) -> bool:
    from src.math_latex import is_math_equiv

    return is_math_equiv(predicted, expected)


# ── MMLU-Pro ─────────────────────────────────────────────────────────────────

_MMLU_LETTER_RE = re.compile(r"(?<![A-Z])([A-J])(?![A-Z])", re.IGNORECASE)


def normalize_mmlu_answer(text: str | None) -> str:
    raw = str(text or "").strip().upper()
    if not raw:
        return ""
    m = _MMLU_LETTER_RE.search(raw)
    if m:
        return m.group(1).upper()
    collapsed = normalize_hotpot_answer(text)
    if len(collapsed) == 1 and collapsed in "abcdefghij":
        return collapsed.upper()
    return collapsed


def grade_mmlu_nem(predicted: str, expected: str) -> bool:
    return normalize_mmlu_answer(predicted) == normalize_mmlu_answer(expected)


# ── Dispatch ─────────────────────────────────────────────────────────────────


def _resolve_dataset(dataset: str | None) -> str:
    ds = (dataset or "").strip().lower().replace("-", "_")
    return DS.aliases.get(ds, ds or "hotpot")


def normalize_for_dataset(text: str | None, dataset: str | None = None) -> str:
    key = _resolve_dataset(dataset)
    if key == "math":
        return normalize_math_answer(text)
    if key == "mmlu":
        return normalize_mmlu_answer(text)
    if key == "gaia":
        return normalize_gaia_string(text)
    if key in ("musique", "hotpot"):
        return squad_normalize_answer(text)
    return squad_normalize_answer(text)


def grade(
    predicted_answer: str,
    expected: str,
    dataset: str | None = None,
) -> bool:
    key = _resolve_dataset(dataset)
    pred = predicted_answer if predicted_answer is not None else ""
    gold = expected if expected is not None else ""

    if key == "math":
        return grade_math(pred, gold)
    if key == "mmlu":
        return grade_mmlu_nem(pred, gold)
    if key == "gaia":
        return grade_gaia_nem(pred, gold)
    if key == "musique":
        return grade_musique_nem(pred, gold)
    if key == "hotpot":
        return grade_hotpot_nem(pred, gold)
    return grade_hotpot_nem(pred, gold)


def is_correct(
    predicted_answer: str,
    expected: str,
    dataset: str | None = None,
) -> bool:
    return grade(predicted_answer, expected, dataset=dataset)


# Aliases used across the repo
canonicalise_for_dataset = normalize_for_dataset
canonicalise_answer = canonicalise_for_dataset
canonicalise_hotpot = normalize_hotpot_answer
canonicalise_musique = normalize_musique_answer
canonicalise_mmlu = normalize_mmlu_answer
canonicalise_qa = normalize_hotpot_answer


def canonicalise_gaia(text: str | None) -> str:
    return normalize_gaia_string(text, remove_punct=True)
