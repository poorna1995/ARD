"""
Dataset-specific answer normalization and grading (single source of truth).

Normalization references
--------------------------
- **hotpot** — SQuAD-style token **F1** (HotpotQA / MuSiQue official ``f1_score`` logic);
  default threshold **0.5**; when normalized gold contains `` and `` (multi-entity lists),
  threshold **0.62** so partial single-name answers (e.g. one of two people) stay wrong.
- **musique** — same token F1 as Hotpot (``metrics/answer.py`` in MuSiQue repo).
- **gaia** — GAIA leaderboard ``scorer.question_scorer`` / ``normalize_str``
- **math** — surface normalize (commas, ascii ``sqrt``, ``pi``), Hendrycks ``is_equiv``,
  then competition base-suffix ints (``src.math_latex``)
- **mmlu_pro** — single letter A–J extraction, then compare
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Any

from evaluator.math_equivalence import strip_string as math_strip_string

# Hotpot / MuSiQue: official token F1 (SQuAD Counter), not substring heuristics.
_OPENQA_F1_THRESHOLD = 0.5
# When gold lists two entities with `` and `` (e.g. "A and B"), require higher F1 so
# "A" alone (train_0586) does not pass at ~0.57 while typical short answers still can.
_OPENQA_F1_THRESHOLD_CONJUNCTION = 0.62

# ── HotpotQA / MuSiQue (SQuAD-style normalized exact match) ─────────────────

_ARTICLE_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)

# Curly/typographic quotes → ASCII apostrophe (not in ``string.punctuation``).
_APOSTROPHE_VARIANTS = (
    "\u2018",
    "\u2019",
    "\u201a",
    "\u201b",
    "\u2032",
    "\u0060",
    "\u00b4",
)
# Hyphens/dashes → word break before punct strip (avoids ``sickle-cell`` → ``sicklecell``).
_DASH_VARIANTS = ("\u2010", "\u2011", "\u2012", "\u2013", "\u2014", "\u2015")


def _preprocess_squad_nem(text: str) -> str:
    """
    Extensions on top of Hotpot/MuSiQue NEM for benchmark gold vs model surface forms.

    - NFKC + unify apostrophe variants, then strip (``d'Aujourd'hui`` ≡ curly gold)
    - Hyphens/underscores/dashes → space (``Sickle-cell`` ≡ ``Sickle cell``)
    """
    s = unicodedata.normalize("NFKC", text)
    for ch in _APOSTROPHE_VARIANTS:
        s = s.replace(ch, "'")
    for ch in _DASH_VARIANTS:
        s = s.replace(ch, " ")
    s = s.replace("-", " ").replace("_", " ")
    return s


def squad_normalize_answer(text: str | None) -> str:
    """
    SQuAD / HotpotQA ``normalize_answer`` token string: lower, strip punct, drop articles.

    Applies ``_preprocess_squad_nem`` first (hyphen / apostrophe parity with gold).
    """
    if text is None:
        return ""
    s = _preprocess_squad_nem(str(text).lower())
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = _ARTICLE_RE.sub(" ", s)
    return " ".join(s.split())


def normalize_hotpot_answer(text: str | None) -> str:
    """Alias of ``squad_normalize_answer`` (used for keys / docs)."""
    return squad_normalize_answer(text)


def normalize_musique_answer(text: str | None) -> str:
    """MuSiQue uses the same normalization as Hotpot/SQuAD."""
    return squad_normalize_answer(text)


def squad_token_f1_components(
    prediction: str, ground_truth: str
) -> tuple[float, float, float]:
    """Return (f1, precision, recall) after SQuAD normalization."""
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
    """HotpotQA ``f1_score`` (token multiset overlap). Returns 0.0–1.0."""
    return squad_token_f1_components(prediction, ground_truth)[0]


def grade_openqa_squad_f1(predicted: str, expected: str) -> bool:
    """
    Hotpot / MuSiQue: SQuAD token F1 with precision/recall floors.

    F1 alone can be ≥ 0.5 when recall is high but the prediction adds extra wrong
    tokens (e.g. gold ``biophysicist``, pred ``physiologist and biophysicist``).
    Require **precision ≥ 0.5 and recall ≥ 0.5** in addition to F1 ≥ threshold.
    Normalized exact match always passes.
    """
    if squad_normalize_answer(predicted) == squad_normalize_answer(expected):
        return True

    gold_n = squad_normalize_answer(expected)
    thr = (
        _OPENQA_F1_THRESHOLD_CONJUNCTION
        if " and " in gold_n
        else _OPENQA_F1_THRESHOLD
    )
    f1, precision, recall = squad_token_f1_components(predicted, expected)
    return (
        f1 >= thr
        and precision >= _OPENQA_F1_THRESHOLD
        and recall >= _OPENQA_F1_THRESHOLD
    )


def grade_hotpot_nem(predicted: str, expected: str) -> bool:
    """HotpotQA: SQuAD token F1 (see module docstring; name kept for imports)."""
    return grade_openqa_squad_f1(predicted, expected)


def grade_musique_nem(predicted: str, expected: str) -> bool:
    """MuSiQue: same SQuAD token F1 as Hotpot."""
    return grade_openqa_squad_f1(predicted, expected)


# ── GAIA (quasi-exact match; type inferred from gold) ─────────────────────────

_GAIA_LIST_SEP_RE = re.compile(r"[,;]")


def _gaia_is_float(element: Any) -> bool:
    try:
        float(element)
        return True
    except (TypeError, ValueError):
        return False


def normalize_gaia_number_str(number_str: str) -> float:
    """GAIA leaderboard: strip $, %, commas then parse float."""
    s = str(number_str)
    for char in ("$", "%", ","):
        s = s.replace(char, "")
    try:
        return float(s)
    except ValueError:
        return float("inf")


def normalize_gaia_string(text: str | None, *, remove_punct: bool = True) -> str:
    """
    GAIA ``normalize_str``: remove all whitespace, optional punct strip, lowercase.
    """
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
    """
    GAIA ``question_scorer``: numeric, list, or string quasi-EM from gold format.
    """
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


# ── MATH (hendrycks ``is_equiv`` + base-suffix ints) ──────────────────────────

def normalize_math_answer(text: str | None) -> str:
    """Canonical stripped LaTeX string (for vote keys / display), via MATH benchmark."""
    if text is None:
        return ""
    return math_strip_string(str(text).strip())


def grade_math(predicted: str | None, expected: str | None) -> bool:
    """
    MATH equivalence: official ``is_equiv``, then ``1`` vs ``1_6``-style base suffix.

    Implemented in ``src.math_latex.is_math_equiv`` (shared with loaders / agents).
    """
    from src.math_latex import is_math_equiv

    return is_math_equiv(predicted, expected)


# ── MMLU-Pro (letter EM) ─────────────────────────────────────────────────────

_MMLU_LETTER_RE = re.compile(r"(?<![A-Z])([A-J])(?![A-Z])", re.IGNORECASE)


def normalize_mmlu_answer(text: str | None) -> str:
    """Extract option letter A–J when present; else Hotpot-style normalize."""
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

_DATASET_ALIASES: dict[str, str] = {
    "math": "math",
    "math_hard": "math",
    "hotpot": "hotpot",
    "musique": "musique",
    "gaia": "gaia",
    "mmlu_pro": "mmlu_pro",
    "mmlu": "mmlu_pro",
}


def _resolve_dataset(dataset: str | None) -> str:
    ds = (dataset or "").strip().lower()
    return _DATASET_ALIASES.get(ds, ds or "hotpot")


def normalize_for_dataset(text: str | None, dataset: str | None = None) -> str:
    """Return the canonical compared form for ``text`` under ``dataset`` rules."""
    key = _resolve_dataset(dataset)
    if key == "math":
        return normalize_math_answer(text)
    if key == "mmlu_pro":
        return normalize_mmlu_answer(text)
    if key == "gaia":
        # String form only; list/number gold uses ``grade_gaia_nem`` branches.
        return normalize_gaia_string(text)
    if key == "musique":
        return squad_normalize_answer(text)
    if key == "hotpot":
        return squad_normalize_answer(text)
    return squad_normalize_answer(text)


def grade(
    predicted_answer: str,
    expected: str,
    dataset: str | None = None,
) -> bool:
    """Compare prediction to gold using dataset-appropriate normalization."""
    key = _resolve_dataset(dataset)
    pred = predicted_answer if predicted_answer is not None else ""
    gold = expected if expected is not None else ""

    if key == "math":
        return grade_math(pred, gold)
    if key == "mmlu_pro":
        return grade_mmlu_nem(pred, gold)
    if key == "gaia":
        return grade_gaia_nem(pred, gold)
    if key == "musique":
        return grade_musique_nem(pred, gold)
    if key == "hotpot":
        return grade_hotpot_nem(pred, gold)
    return grade_hotpot_nem(pred, gold)


# Backward-compatible names used elsewhere in the repo
canonicalise_for_dataset = normalize_for_dataset
canonicalise_hotpot = normalize_hotpot_answer
canonicalise_musique = normalize_musique_answer

def canonicalise_gaia(text: str | None) -> str:
    return normalize_gaia_string(text, remove_punct=True)

canonicalise_mmlu = normalize_mmlu_answer
canonicalise_qa = normalize_hotpot_answer
