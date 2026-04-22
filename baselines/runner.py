from __future__ import annotations

import json
import os
import re
import time
import unicodedata
import inspect
from dataclasses import dataclass
from typing import Any, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

from agents.react_operator import react_operator
from agents.vanilla_operator import vanilla_operator          # BUG FIX 1 (see § 8)
from baselines.constants import (
    ALLOWED_DATASETS,
    ALLOWED_MODALITIES,
    validate_dataset,
    validate_modality,
    validate_model,
)
from baselines.prompt_resolver import resolve_prompts
from baselines.schema import UnifiedExperimentRecord

try:
    from sympy import N, simplify, sympify
    from sympy.parsing.latex import parse_latex
    SYMPY_AVAILABLE = True
except Exception:
    SYMPY_AVAILABLE = False


# ══════════════════════════════════════════════════════════════════════════════
# § 1  CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o":      {"input": 0.0025,  "output": 0.01},
}

_REACT_MAX_TOKENS_PER_STEP = 512

GAIA_ALIASES: dict[str, list[str]] = {
    "united states":           ["us", "usa", "u.s.", "u.s.a.", "united states of america"],
    "united kingdom":          ["uk", "u.k.", "britain", "great britain"],
    "world war ii":            ["wwii", "ww2", "world war 2", "second world war"],
    "artificial intelligence": ["ai"],
    "united nations":          ["un", "u.n."],
}

_RE_WHITESPACE    = re.compile(r"\s+")
# Whole-line capture (used where only one marker per line is expected).
_RE_THE_ANSWER    = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
_RE_FINAL_ANSWER  = re.compile(r"[Ff]inal answer is[:\s]+([^\n]+)")
# Stops before the next marker so several "The answer is …" on one line are split.
_RE_THE_ANSWER_SEG = re.compile(
    r"(?is)\bThe answer is[:\s]+(.+?)(?=\s*\bThe answer is\b|\s*\bFinal answer is\b|\Z)"
)
_RE_FINAL_ANSWER_SEG = re.compile(
    r"(?is)\bFinal answer is[:\s]+(.+?)(?=\s*\bThe answer is\b|\s*\bFinal answer is\b|\Z)"
)
_RE_PLACEHOLDER_CAPTURE = re.compile(r"^<[^>]+>$", re.IGNORECASE)
_RE_BOXED         = re.compile(r"\\boxed\{([^}]+)\}")
_RE_EQ_END        = re.compile(r"=\s*([^\n=]+)\s*$", re.MULTILINE)
_RE_IMPLICIT_MUL  = re.compile(r"(\d)(pi\b|[a-df-wyzA-Z])")
_RE_LETTER_MMLU   = re.compile(r"\b([A-Ja-j])\b")
_GAIA_PREAMBLE    = re.compile(
    r"^(To\s|In\s+order|We\s+need|Let's\s|Let\s+us\s|Here's\s|Here\s+is\s|"
    r"The\s+question|However,|I\s+need|Based\s+on|The\s+first\s+step|"
    r"I\s+will|I'll\s|Note\s+that)",
    re.IGNORECASE,
)
_RE_STEP_MARKERS  = re.compile(
    r"(?m)(?:^\s*\d+[\.)]\s+\S|^\s*[-*•]\s+\S|\bstep\s+\d+)",
    re.IGNORECASE,
)
_RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# `\text{...}` (balanced `{`…`}`) — unwrap for math_hard eval (case-insensitive ``\text``).
_RE_TEXT_CMD_OPEN = re.compile(r"\\text\{", re.IGNORECASE)
_RE_MATHISH_TAIL = re.compile(
    r"\\(?:frac|sqrt|sum|int|prod|lim|log|ln|sin|cos|tan|cdot|times|infty|cup|cap|in|leq|geq|neq|pm)"
)
_MATH_TRANS = str.maketrans({
    "−": "-", "∞": r"\infty", "^": "**", "×": "*", "÷": "/", "π": "pi",
})


# ══════════════════════════════════════════════════════════════════════════════
# § 2  ALIAS MAP
# ══════════════════════════════════════════════════════════════════════════════

def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.strip().lower())
    filtered = (c for c in text if not unicodedata.combining(c))
    text = "".join(filtered).rstrip(".,:;!")
    return _RE_WHITESPACE.sub(" ", text)


def _build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in GAIA_ALIASES.items():
        canon = _norm(canonical)
        for alias in aliases:
            alias_map[_norm(alias)] = canon
    return alias_map


_GAIA_ALIAS_MAP: dict[str, str] = _build_alias_map()


def _is_placeholder_captured_value(t: str) -> bool:
    """True if text after a marker (or a bare token) is an instruction placeholder."""
    s = (t or "").strip().rstrip(".,;:!` \t")
    if not s:
        return True
    if _RE_PLACEHOLDER_CAPTURE.match(s):
        return True
    if re.fullmatch(r"<[a-z][a-z0-9_-]*>", s, re.IGNORECASE):
        return True
    sl = s.lower()
    if sl in {"<final_answer>", "<value>", "<letter>", "<patch>", "final_answer"}:
        return True
    return False


def _is_placeholder_answer_fragment(s: str) -> bool:
    """True if the model echoed a template token or a full 'The answer is <…>' meta line."""
    t = (s or "").strip().rstrip(".,;:!` \t")
    if not t:
        return True
    m = re.match(r"(?is)^the\s+answer\s+is\s*:?\s*(.+)$", t)
    if m and _is_placeholder_captured_value(m.group(1).strip().rstrip(".,;:!`")):
        return True
    return _is_placeholder_captured_value(t)


def _normalize_extracted_answer(s: str) -> str:
    """Strip wrappers models often add around GAIA-style short answers."""
    t = (s or "").strip()
    t = t.rstrip(".,;:!`")
    # ASCII / straight quotes
    while len(t) >= 2 and t[0] == t[-1] and t[0] in "\"'":
        t = t[1:-1].strip().rstrip(".,;:!`")
    # Unicode curly quotes
    if len(t) >= 2 and t[0] in "\u201c\u2018" and t[-1] in "\u201d\u2019":
        t = t[1:-1].strip().rstrip(".,;:!`")
    # Markdown bold
    if len(t) >= 4 and t.startswith("**") and t.endswith("**"):
        t = t[2:-2].strip().rstrip(".,;:!`")
    return t.strip()


def _line_has_answer_marker(line: str) -> bool:
    return bool(
        re.search(r"(?i)\b(?:The answer is|Final answer is)\b", line)
    )


def _last_marker_capture_on_line(line: str) -> Optional[str]:
    """Last substantive capture on a single line (handles several markers on one line)."""
    caps: list[str] = []
    for m in _RE_THE_ANSWER_SEG.finditer(line):
        caps.append(m.group(1).strip())
    for m in _RE_FINAL_ANSWER_SEG.finditer(line):
        caps.append(m.group(1).strip())
    for cand in reversed(caps):
        # Do not strip closing " or ' here — that breaks balanced-quote normalization.
        cleaned = cand.strip().rstrip(".,;:!`")
        if not _is_placeholder_captured_value(cleaned):
            return _normalize_extracted_answer(cleaned)
    return None


# Several GAIA completions are sometimes pasted into one string. The next
# blocks often start with these openings right after a newline.
_GAIA_CONJOIN_SPLIT = re.compile(
    r"(?m)(?=\n(?:To determine\b|In Unlambda\b|The minimum perigee\b))"
)


def _gaia_first_response_scope(full: str) -> str:
    """
    If multiple Q&A blocks are concatenated, keep only the first response for
    extraction (matches one ``QueryInput`` / one API completion in normal use).
    """
    m = _GAIA_CONJOIN_SPLIT.search(full)
    if m is None or m.start() < 80:
        return full.strip()
    return full[: m.start()].strip()


def _gaia_style_extract_markers(full: str) -> Optional[str]:
    """
    Prefer the last *line* that contains an answer marker.

    Whole-text DOTALL segmenting merges the first short ``The answer is …`` with
    long following paragraphs until the next marker; scanning lines bottom-up
    matches how models usually emit the final answer on its own line.
    """
    lines = [ln.rstrip() for ln in full.splitlines()]
    for line in reversed(lines):
        if not line.strip():
            continue
        if not _line_has_answer_marker(line):
            continue
        cap = _last_marker_capture_on_line(line)
        if cap:
            return cap
    return None


def _last_answer_after_marker(full: str) -> Optional[str]:
    """
    Prefer the *last* substantive 'The answer is …' / 'Final answer is …' span.

    Models often emit several markers during reasoning; the first may be wrong
    (e.g. a partial word) or a literal ``<final_answer>`` placeholder.
    Uses segment regexes so multiple markers on one line are not swallowed by
    the first ``[^\n]+`` match.
    """
    spans: list[tuple[int, str]] = []
    for m in _RE_THE_ANSWER_SEG.finditer(full):
        spans.append((m.start(), m.group(1).strip()))
    for m in _RE_FINAL_ANSWER_SEG.finditer(full):
        spans.append((m.start(), m.group(1).strip()))
    spans.sort(key=lambda x: x[0])
    for _, cap in reversed(spans):
        cleaned = cap.strip().rstrip(".,;:!`")
        if not _is_placeholder_captured_value(cleaned):
            return _normalize_extracted_answer(cleaned)
    return None


# ══════════════════════════════════════════════════════════════════════════════
# § 3  DATA CONTRACTS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class RunSelection:
    datasets:   tuple[str, ...]
    modalities: tuple[str, ...]
    model:       str   = "gpt-4o-mini"
    max_tokens:  int   = 2048
    temperature: float = 0.0
    seed: Optional[int] = 42
    max_steps:   int   = 6
    tool_budget: int   = 4
    routellm_router:       str   = "mf"
    routellm_threshold:    float = 0.11593
    routellm_strong_model: str   = "gpt-4o"
    routellm_weak_model:   str   = "gpt-4o-mini"

    def __post_init__(self) -> None:
        validate_model(self.model)
        for d in self.datasets:   validate_dataset(d)
        for m in self.modalities: validate_modality(m)
        if "routellm" in self.modalities:
            if not (0.0 <= self.routellm_threshold <= 1.0):
                raise ValueError("routellm_threshold must be in [0, 1].")

    def routellm_config(self) -> dict[str, Any]:
        cfg_dict = {
            "router": self.routellm_router.strip(),
            "threshold": float(self.routellm_threshold),
            "strong_model": self.routellm_strong_model.strip(),
            "weak_model": self.routellm_weak_model.strip(),
        }
        try:
            from baselines.route_llm import RoutellmConfig
            return RoutellmConfig(**cfg_dict)
        except Exception:
            return cfg_dict


@dataclass(frozen=True)
class QueryInput:
    query_id:     str
    dataset:      str
    query:        str
    ground_truth: str
    options:  Optional[str]            = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        validate_dataset(self.dataset)


# ══════════════════════════════════════════════════════════════════════════════
# § 4  COST HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return round(
        (prompt_tokens     / 1000.0) * rates["input"] +
        (completion_tokens / 1000.0) * rates["output"], 6,
    )


def _estimate_cost_routellm(
    selection: RunSelection,
    resolved_model: Optional[str],
    prompt_tokens: int,
    completion_tokens: int,
) -> float:
    r = (resolved_model or "").lower()
    for key in (selection.routellm_strong_model, selection.routellm_weak_model,
                "gpt-4o", "gpt-4o-mini"):
        if key and key.lower() in r and key in COST_PER_1K_TOKENS:
            return _estimate_cost(key, prompt_tokens, completion_tokens)
    fallback = (selection.routellm_weak_model
                if selection.routellm_weak_model in COST_PER_1K_TOKENS
                else "gpt-4o-mini")
    return _estimate_cost(fallback, prompt_tokens, completion_tokens)


# ══════════════════════════════════════════════════════════════════════════════
# § 5  EVALUATION HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _try_float(text: str) -> Optional[float]:
    try:
        t = (
            (text or "")
            .replace(r"\$", "")
            .replace("$", "")
            .replace(",", "")
            .replace("%", "")
            .strip()
        )
        return float(t)
    except ValueError:
        return None


def _eval_gaia(predicted: str, ground_truth: str) -> bool:
    p_norm, g_norm = _norm(predicted), _norm(ground_truth)
    if p_norm == g_norm:
        return True
    if _GAIA_ALIAS_MAP.get(p_norm, p_norm) == _GAIA_ALIAS_MAP.get(g_norm, g_norm):
        return True
    pn, gn = _try_float(predicted), _try_float(ground_truth)
    return pn is not None and gn is not None and abs(pn - gn) < 1e-6


def _extract_letter_mmlu(text: str) -> str:
    text = text.strip()
    m = _RE_LETTER_MMLU.search(text)
    if m:
        return m.group(1).upper()
    return text[0].upper() if text and text[0].upper() in "ABCDEFGHIJ" else text.upper()


def _eval_mmlu(predicted: str, ground_truth: str) -> bool:
    return _extract_letter_mmlu(predicted) == _extract_letter_mmlu(ground_truth)


def _count_reasoning_steps(raw: str, predicted: str) -> int:
    text = (raw or "").strip()
    pred = (predicted or "").strip()
    if not text or text == pred:
        return 0
    markers = _RE_STEP_MARKERS.findall(text)
    if markers:
        return len(markers)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 1
    last = lines[-1]
    if _RE_THE_ANSWER.search(last) or _RE_FINAL_ANSWER.search(last):
        lines = lines[:-1]
    elif pred and (last.rstrip(".,;:!") == pred or last.endswith(pred)):
        lines = lines[:-1]
    return max(1, len(lines))


def _unwrap_latex_text_once(expr: str) -> str:
    """Replace each ``\\text{...}`` (balanced braces) with its inner text."""
    s = expr or ""
    out: list[str] = []
    pos = 0
    while True:
        m = _RE_TEXT_CMD_OPEN.search(s, pos)
        if not m:
            out.append(s[pos:])
            break
        out.append(s[pos : m.start()])
        open_brace = m.end() - 1
        if open_brace >= len(s) or s[open_brace] != "{":
            out.append(s[m.start() :])
            break
        depth = 0
        k = open_brace
        while k < len(s):
            ch = s[k]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    inner = s[open_brace + 1 : k]
                    out.append(inner)
                    pos = k + 1
                    break
            k += 1
        else:
            out.append(s[m.start() :])
            break
    return "".join(out)


def _unwrap_latex_text(expr: str) -> str:
    """Repeatedly unwrap ``\\text`` until fixed point (nested ``\\text``)."""
    prev: Optional[str] = None
    cur = expr or ""
    while cur != prev:
        prev = cur
        cur = _unwrap_latex_text_once(cur)
    return cur


_RE_TEXT_SIMPLE = re.compile(
    # Drop ``\\text{...}`` when the body is "label-like" (contains a letter) and
    # has no ``{``, ``\\``, or ``$`` — avoids stripping ``\\text{\\$1,157}`` / ``\\text{4.5}``.
    r"(?i)\\text\s*\{[^{}\\$]*[A-Za-z][^{}\\$]*\}",
)


def _normalize_latex_text_commands(s: str) -> str:
    """
    Remove label-like ``\\text{...}`` spans, then balanced-unwrap remaining ``\\text``.
    Repeats until stable so models can mix broken and valid ``\\text`` usage.
    """
    prev = None
    cur = s or ""
    while cur != prev:
        prev = cur
        cur = _RE_TEXT_SIMPLE.sub(" ", cur)
        cur = _unwrap_latex_text_once(cur)
    return _RE_WHITESPACE.sub(" ", cur).strip()


def _trim_trailing_unit_prose(s: str) -> str:
    """
    If the string is ``<number> <letters...>`` with no LaTeX math commands,
    keep only the leading numeric token (e.g. ``15 \\text{seconds}`` → ``15`` after unwrap).
    """
    s = (s or "").strip()
    if not s or _RE_MATHISH_TAIL.search(s):
        return s
    m = re.match(r"^(-?\d+(?:\.\d+)?)(\s+[A-Za-z].*)$", s)
    if m:
        return m.group(1)
    return s


def _canonicalize_math_for_eval(expr: str) -> str:
    """Normalize LaTeX ``\\text{...}``, display ``\\\\``, and light prose for math_hard."""
    s = _normalize_latex_text_commands((expr or "").strip())
    s = _trim_trailing_unit_prose(s)
    s = _strip_math_wrappers(s)
    s = re.sub(r"\\\\+\s*", " ", s)
    return _RE_WHITESPACE.sub(" ", s).strip()


def _strip_math_wrappers(expr: str) -> str:
    expr = (expr or "").strip()
    if expr.startswith(r"\(") and expr.endswith(r"\)"):
        expr = expr[2:-2].strip()
    if expr.startswith(r"\[") and expr.endswith(r"\]"):
        expr = expr[2:-2].strip()
    while len(expr) >= 2 and expr[0] == "$" and expr[-1] == "$":
        expr = expr[1:-1].strip()
    return expr.replace(r"\left", "").replace(r"\right", "").strip()


def _extract_math_final(text: str) -> str:
    text = (text or "").strip().strip("`").strip('"').strip("'").strip()
    text = _strip_math_wrappers(text)
    for pattern in (_RE_THE_ANSWER_SEG, _RE_FINAL_ANSWER_SEG):
        caps = [m.group(1) for m in pattern.finditer(text)]
        for cap in reversed(caps):
            frag = cap.strip().rstrip(".")
            if _is_placeholder_captured_value(frag):
                continue
            return _canonicalize_math_for_eval(_strip_math_wrappers(frag))
    for pattern in (_RE_BOXED, _RE_EQ_END):
        m = pattern.search(text)
        if m:
            return _canonicalize_math_for_eval(
                _strip_math_wrappers(m.group(1).strip().rstrip("."))
            )
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for line in reversed(lines):
        if _is_placeholder_answer_fragment(line):
            continue
        return _canonicalize_math_for_eval(_strip_math_wrappers(line.rstrip(".")))
    return ""


def _preprocess_math(expr: str) -> str:
    expr = _canonicalize_math_for_eval(expr)
    out = (expr.replace("π","pi").replace("\u03c0","pi").replace("\u2212","-")
               .replace("\u00d7","*").replace("\u00f7","/")
               .replace("²","**2").replace("³","**3")
               .translate(_MATH_TRANS).strip())
    out = _RE_IMPLICIT_MUL.sub(r"\1*\2", out)
    return _RE_WHITESPACE.sub("", out)


def _norm_math(text: str) -> str:
    t = _canonicalize_math_for_eval(text).lower()
    return _RE_WHITESPACE.sub("", t.translate(_MATH_TRANS)).rstrip(".")


def _sympy_equal_math(predicted: str, ground_truth: str) -> Optional[bool]:
    if not SYMPY_AVAILABLE:
        return None
    a1, a2 = _preprocess_math(predicted), _preprocess_math(ground_truth)
    for fn in (sympify, parse_latex):
        try:
            diff = simplify(fn(a1) - fn(a2))
            if diff == 0 or abs(complex(N(diff))) < 1e-2:
                return True
        except Exception:
            continue
    return None


def _eval_math(predicted: str, ground_truth: str) -> bool:
    p = _canonicalize_math_for_eval(_extract_math_final(predicted))
    g = _canonicalize_math_for_eval((ground_truth or "").strip())
    r = _sympy_equal_math(p, g)
    if r is not None:
        return r
    if _norm_math(p) == _norm_math(g):
        return True
    pn, gn = _try_float(p), _try_float(g)
    return pn is not None and gn is not None and abs(pn - gn) < 1e-6


def _compute_correctness(dataset: str, predicted: str, ground_truth: str) -> bool:
    if not predicted or not ground_truth:
        return False
    if dataset == "gaia":      return _eval_gaia(predicted, ground_truth)
    if dataset == "mmlu_pro":  return _eval_mmlu(predicted, ground_truth)
    if dataset == "math_hard": return _eval_math(predicted, ground_truth)
    return predicted.strip().lower() == ground_truth.strip().lower()


# ══════════════════════════════════════════════════════════════════════════════
# § 6  PROMPT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _ground_truth_hint(dataset: str) -> str:
    if dataset == "mmlu_pro":           return "gold is one letter A–J; output that letter only"
    if dataset == "math_hard":          return "gold is LaTeX/math final; match that form"
    if dataset == "swe_bench_verified": return "gold is patch/diff text; output minimal patch"
    return "gold is short factual (word, number, name, date); match exactly"


def _output_contract_for(dataset: str, modality: str) -> str:
    if modality in ("vanilla", "routellm"):
        if dataset == "gaia":               return "Output one line: final answer only."
        if dataset == "mmlu_pro":           return "Output one character: A–J only."
        if dataset == "math_hard":          return "Output final math only; no sentences."
        if dataset == "swe_bench_verified": return "Output patch only; no prose."
    if modality in ("zero_shot_cot", "few_shot_cot"):
        if dataset == "gaia":               return "Brief reasoning then: The answer is <value>."
        if dataset == "mmlu_pro":           return "Brief reasoning then: The answer is <A–J>."
        if dataset == "math_hard":          return "Brief reasoning then: The answer is <TeX/math>."
        if dataset == "swe_bench_verified": return "Brief reasoning then: The answer is <patch>."
    if dataset == "mmlu_pro":           return "End with one letter A–J."
    if dataset == "math_hard":          return "End with final math expression only."
    if dataset == "swe_bench_verified": return "End with patch or minimal artifact."
    return "End with one-line final answer."


def _build_system_values(selection: RunSelection, dataset: str, modality: str) -> dict[str, Any]:
    if modality == "routellm":
        model_name = (
            f"RouteLLM router={selection.routellm_router} thr={selection.routellm_threshold:g} "
            f"strong={selection.routellm_strong_model} weak={selection.routellm_weak_model}"
        )
    else:
        model_name = selection.model
    return {
        "model_name":      model_name,
        "output_contract": _output_contract_for(dataset, modality),
        "max_steps":       selection.max_steps,
        "tool_budget":     selection.tool_budget,
    }


def _build_user_values(query: QueryInput, modality: str) -> dict[str, Any]:
    values: dict[str, Any] = {"dataset": query.dataset, "query": query.query}

    if query.dataset == "mmlu_pro":
        values["options"] = query.options or ""

    # ground_truth_format_hint is only referenced by vanilla (and routellm,
    # which reuses the vanilla template).  Other modalities don't use it, but
    # passing an extra key to str.format() is harmless.
    if query.dataset in ("gaia", "math_hard", "swe_bench_verified"):
        values["ground_truth_format_hint"] = _ground_truth_hint(query.dataset)

    # BUG FIX 2: react and multiagent user-prompt templates require
    # {tools_available}.  The original never added it, so resolve_prompts()
    # would always raise ValueError("Missing required user variables:
    # ['tools_available']") for every multiagent query.
    # react builds its own prompts internally, but multiagent goes through
    # resolve_prompts, so it must be present here.
    if modality in ("react", "multiagent"):
        values["tools_available"] = "web_search, python_repl, file_reader"

    return values


def _extract_short_answer(raw: str, dataset: str) -> tuple[str, str]:
    full = (raw or "").strip()
    if not full:
        return "", ""
    if dataset in ("gaia", "swe_bench_verified"):
        scoped = _gaia_first_response_scope(full) if dataset == "gaia" else full
        line_ok = _gaia_style_extract_markers(scoped)
        if line_ok is not None:
            return line_ok, full
        last_ok = _last_answer_after_marker(scoped)
        if last_ok is not None:
            return last_ok, full
        lines = [ln.strip() for ln in scoped.splitlines() if ln.strip()]
        for line in reversed(lines):
            if len(line) > 220 or _GAIA_PREAMBLE.match(line):
                continue
            if _line_has_answer_marker(line):
                cap = _last_marker_capture_on_line(line)
                if cap:
                    return cap, full
                continue
            if _is_placeholder_answer_fragment(line):
                continue
            return line.rstrip(".,;:!"), full
        for frag in reversed(_RE_SENTENCE_SPLIT.split(scoped)):
            frag = frag.strip()
            if not frag or len(frag) > 220 or _GAIA_PREAMBLE.match(frag):
                continue
            if _line_has_answer_marker(frag):
                cap = _last_marker_capture_on_line(frag)
                if cap:
                    return cap, full
                continue
            if _is_placeholder_answer_fragment(frag):
                continue
            return frag.rstrip(".,;:!"), full
        return "", full
    if dataset == "mmlu_pro":
        line_ok = _gaia_style_extract_markers(full)
        if line_ok is not None:
            frag = line_ok.rstrip(".")
            lm = re.match(r"\(?([A-Ja-j])\)?[\.\):\s]?", frag)
            return (lm.group(1).upper() if lm else frag), full
        last_ok = _last_answer_after_marker(full)
        if last_ok is not None:
            frag = last_ok.rstrip(".")
            lm = re.match(r"\(?([A-Ja-j])\)?[\.\):\s]?", frag)
            return (lm.group(1).upper() if lm else frag), full
        lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
        while lines and (
            _is_placeholder_answer_fragment(lines[-1])
            or (
                _line_has_answer_marker(lines[-1])
                and _last_marker_capture_on_line(lines[-1]) is None
            )
        ):
            lines.pop()
        letters = _RE_LETTER_MMLU.findall(full)
        if letters:
            return letters[-1].upper(), full
        return (lines[-1].rstrip(".") if lines else ""), full
    if dataset == "math_hard":
        return _extract_math_final(full), full
    return full, full


# ══════════════════════════════════════════════════════════════════════════════
# § 7  REACT ANSWER EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

def _extract_react_answer(
    react_trace: list[dict[str, Any]],
    raw_output: str,
    dataset: str,
) -> tuple[str, int]:
    finish_steps    = [s for s in react_trace if s.get("action") == "finish"]
    non_finish      = [s for s in react_trace if s.get("action") != "finish"]
    reasoning_steps = len(non_finish)

    if finish_steps:
        answer = finish_steps[-1].get("action_input", "").strip()
        if answer:
            return answer, reasoning_steps

    for step in reversed(non_finish):
        if not step.get("error") and step.get("observation"):
            obs = step["observation"].strip()
            if obs and len(obs) < 400:
                return obs, reasoning_steps

    predicted, _ = _extract_short_answer(raw_output, dataset)
    return predicted, reasoning_steps


# ══════════════════════════════════════════════════════════════════════════════
# § 8  MAIN RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def _print_verbose_run_trace(
    *,
    query_id: str,
    dataset: str,
    modality: str,
    query: str,
    system_prompt: str,
    user_prompt: str,
    raw_response: str,
    predicted_answer: str,
    ground_truth: str,
) -> None:
    """Echo query, resolved prompts, raw model text, and scored fields to stdout."""
    bar = "=" * 80
    print(f"\n{bar}", flush=True)
    print(
        f"RUN TRACE  query_id={query_id!r}  dataset={dataset!r}  modality={modality!r}",
        flush=True,
    )
    print(bar, flush=True)
    print("--- input query (task text) ---", flush=True)
    print(query, flush=True)
    print("\n--- system prompt ---", flush=True)
    print(system_prompt, flush=True)
    print("\n--- user prompt ---", flush=True)
    print(user_prompt, flush=True)
    print("\n--- raw model response (verbatim / concatenated turns) ---", flush=True)
    print(raw_response, flush=True)
    print("\n--- predicted answer (after extraction) ---", flush=True)
    print(predicted_answer, flush=True)
    print("\n--- ground truth ---", flush=True)
    print(ground_truth, flush=True)
    print(f"{bar}\n", flush=True)


_VANILLA_ACCEPTS_SEED = "seed" in inspect.signature(vanilla_operator).parameters


def _load_routellm_helpers() -> tuple[bool, str | None, Any, Any]:
    """
    Lazy-load RouteLLM helpers so non-RouteLLM runs do not require
    `baselines.route_llm` to exist at import time.
    """
    try:
        from baselines.route_llm import build_controller, routellm_chat_completion
        return True, None, build_controller, routellm_chat_completion
    except Exception as exc:  # pragma: no cover - runtime import path
        return False, str(exc), None, None


def run_selected(
    *,
    queries: list[QueryInput],
    selection: RunSelection,
    client: Optional[OpenAI] = None,
    verbose_prompts: bool = False,
) -> list[UnifiedExperimentRecord]:
    llm = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    groq_llm: Optional[OpenAI] = None

    def _client_for_model(model_name: str) -> OpenAI:
        nonlocal groq_llm
        if "llama" not in model_name.lower():
            return llm
        if groq_llm is None:
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise RuntimeError(
                    "GROQ_API_KEY is required for llama models but is not set."
                )
            groq_llm = OpenAI(
                api_key=groq_api_key,
                base_url="https://api.groq.com/openai/v1",
            )
        return groq_llm

    results: list[UnifiedExperimentRecord] = []

    wanted_datasets   = set(selection.datasets)
    wanted_modalities = set(selection.modalities)

    routellm_client: Optional[Any] = None
    routellm_chat_fn: Optional[Any] = None
    if "routellm" in wanted_modalities:
        ok, import_error, build_controller_fn, routellm_chat_fn = _load_routellm_helpers()
        if not ok:
            msg = "Modality 'routellm' needs the RouteLLM PyPI package."
            if import_error:
                msg += f"\n\nImport diagnostic:\n  {import_error}"
            raise ImportError(msg)
        assert build_controller_fn is not None
        routellm_client = build_controller_fn(selection.routellm_config())

    system_values_cache: dict[tuple[str, str], dict[str, Any]] = {
        (ds, mod): _build_system_values(selection, ds, mod)
        for ds in wanted_datasets
        for mod in wanted_modalities
        if mod != "react"
    }

    for query in queries:
        if query.dataset not in wanted_datasets:
            continue

        for modality in wanted_modalities:
            is_routellm = modality == "routellm"
            is_react    = modality == "react"

            # BUG FIX 3: the original used `locals().get("resolved")` at
            # record-building time to extract version/hash fields.  That is
            # fragile — locals() is an implementation detail and the lookup
            # silently returns None whenever `resolved` is out of scope (e.g.
            # the react branch).  Use an explicit sentinel instead.
            resolved = None

            # ── REACT ────────────────────────────────────────────────────
            if is_react:
                call_client = _client_for_model(selection.model)
                (
                    raw_output,
                    usage,
                    error,
                    elapsed,
                    react_trace,
                    react_system_prompt,
                    react_user_prompt,
                ) = react_operator(
                    client=call_client,
                    model_name=selection.model,
                    question=query.query,
                    query_input=query,
                    dataset=query.dataset,
                    max_steps=selection.max_steps,
                    max_tokens=_REACT_MAX_TOKENS_PER_STEP,
                    temperature=0.0,
                    seed=selection.seed,
                )
                predicted, reasoning_steps = _extract_react_answer(
                    react_trace, raw_output, query.dataset
                )
                tool_trace_json = json.dumps(react_trace, ensure_ascii=False)
                resolved_model  = None

            # ── ROUTELLM ─────────────────────────────────────────────────
            elif is_routellm:
                system_values = system_values_cache[(query.dataset, modality)]
                user_values   = _build_user_values(query, modality)
                resolved = resolve_prompts(
                    modality=modality, dataset=query.dataset,
                    model_name=selection.model,
                    system_values=system_values, user_values=user_values,
                )
                assert routellm_client is not None
                assert routellm_chat_fn is not None
                raw_output, usage, error, elapsed, resolved_model = routellm_chat_fn(
                    routellm_client, cfg=selection.routellm_config(),
                    system_prompt=resolved.system_prompt,
                    user_prompt=resolved.user_prompt,
                    max_tokens=selection.max_tokens,
                    temperature=selection.temperature,
                    seed=selection.seed,
                )
                predicted, _    = _extract_short_answer(raw_output, query.dataset)
                reasoning_steps = _count_reasoning_steps(raw_output, predicted)
                tool_trace_json = None

            # ── VANILLA / COT / MULTIAGENT ───────────────────────────────
            else:
                call_client = _client_for_model(selection.model)
                system_values = system_values_cache[(query.dataset, modality)]
                user_values   = _build_user_values(query, modality)
                resolved = resolve_prompts(
                    modality=modality, dataset=query.dataset,
                    model_name=selection.model,
                    system_values=system_values, user_values=user_values,
                )
                # BUG FIX 1: original had a typo `vanallia_operator` which
                # caused NameError on every vanilla / cot / multiagent call.
                vanilla_kwargs: dict[str, Any] = {
                    "client": call_client,
                    "model_name": selection.model,
                    "question": resolved.user_prompt,
                    "system_prompt": resolved.system_prompt,
                    "max_tokens": selection.max_tokens,
                    "temperature": selection.temperature,
                }
                if _VANILLA_ACCEPTS_SEED:
                    vanilla_kwargs["seed"] = selection.seed
                raw_output, usage, error, elapsed = vanilla_operator(**vanilla_kwargs)
                predicted, _    = _extract_short_answer(raw_output, query.dataset)
                reasoning_steps = _count_reasoning_steps(raw_output, predicted)
                tool_trace_json = None
                resolved_model  = None

            # ── Build record ──────────────────────────────────────────────
            raw_stripped = raw_output.strip()
            row_meta: dict[str, Any] = dict(query.metadata) if query.metadata else {}

            if is_routellm:
                row_meta.update({
                    "routellm_router":       selection.routellm_router,
                    "routellm_threshold":    selection.routellm_threshold,
                    "routellm_strong_model": selection.routellm_strong_model,
                    "routellm_weak_model":   selection.routellm_weak_model,
                })
                if resolved_model:
                    row_meta["routellm_resolved_model"] = resolved_model

            if is_react:
                row_meta["react_steps_taken"] = reasoning_steps

            is_correct = _compute_correctness(
                dataset=query.dataset,
                predicted=predicted,
                ground_truth=query.ground_truth,
            )

            if verbose_prompts:
                if is_react:
                    _print_verbose_run_trace(
                        query_id=query.query_id,
                        dataset=query.dataset,
                        modality=modality,
                        query=query.query,
                        system_prompt=react_system_prompt,
                        user_prompt=react_user_prompt,
                        raw_response=raw_output,
                        predicted_answer=predicted,
                        ground_truth=query.ground_truth,
                    )
                elif resolved is not None:
                    _print_verbose_run_trace(
                        query_id=query.query_id,
                        dataset=query.dataset,
                        modality=modality,
                        query=query.query,
                        system_prompt=resolved.system_prompt,
                        user_prompt=resolved.user_prompt,
                        raw_response=raw_output,
                        predicted_answer=predicted,
                        ground_truth=query.ground_truth,
                    )

            reasoning_trace: Optional[str] = None
            if modality in (
                "zero_shot_cot",
                "few_shot_cot",
                "react",
                "multiagent",
            ):
                reasoning_trace = raw_output
            elif raw_stripped != predicted.strip():
                reasoning_trace = raw_output

            prompt_tokens     = usage.get("prompt_tokens",     0)
            completion_tokens = usage.get("completion_tokens", 0)
            cost_usd = (
                _estimate_cost_routellm(selection, resolved_model,
                                        prompt_tokens, completion_tokens)
                if is_routellm
                else _estimate_cost(selection.model, prompt_tokens, completion_tokens)
            )

            # BUG FIX 3 (cont.): use the explicit `resolved` sentinel rather
            # than locals().get("resolved"), which is fragile and reads stale
            # values across loop iterations when resolved is not reassigned.
            sys_ver = resolved.system_prompt_version if resolved else "react-v1"
            usr_ver = resolved.user_prompt_version   if resolved else "react-v1"
            p_hash  = resolved.prompt_hash           if resolved else ""

            results.append(UnifiedExperimentRecord(
                query_id=query.query_id,
                dataset=query.dataset,
                modality=modality,
                model="routellm" if is_routellm else selection.model,
                query=query.query,
                ground_truth=query.ground_truth,
                predicted_answer=predicted,
                is_correct=is_correct,
                raw_model_output=raw_output,
                reasoning_trace=reasoning_trace,
                tool_trace=tool_trace_json,
                reasoning_steps=reasoning_steps,
                react_trace_length=len(react_trace) if is_react else None,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=usage.get("total_tokens", 0),
                cost_usd=cost_usd,
                latency_s=round(elapsed, 3),
                error_type="runtime_error" if error else None,
                error_message=error,
                system_prompt_version=sys_ver,
                user_prompt_version=usr_ver,
                prompt_hash=p_hash,
                metadata=row_meta,
            ))

    return results


def default_selection() -> RunSelection:
    modalities = tuple(m for m in ALLOWED_MODALITIES if m != "routellm")
    return RunSelection(datasets=ALLOWED_DATASETS, modalities=modalities)
