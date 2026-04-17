from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional
import unicodedata
from dotenv import load_dotenv
from openai import OpenAI
import os
load_dotenv()

from agents.vanallia_operator import vanallia_operator
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


COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

GAIA_ALIASES: dict[str, list[str]] = {
    "united states": ["us", "usa", "u.s.", "u.s.a.", "united states of america"],
    "united kingdom": ["uk", "u.k.", "britain", "great britain"],
    "world war ii": ["wwii", "ww2", "world war 2", "second world war"],
    "artificial intelligence": ["ai"],
    "united nations": ["un", "u.n."],
}

_RE_WHITESPACE = re.compile(r"\s+")
_RE_THE_ANSWER = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
_RE_FINAL_ANSWER = re.compile(r"[Ff]inal answer is[:\s]+([^\n]+)")
_RE_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
_RE_EQ_END = re.compile(r"=\s*([^\n=]+)\s*$", re.MULTILINE)
_RE_IMPLICIT_MUL = re.compile(r"(\d)(pi\b|[a-df-wyzA-Z])")


@dataclass(frozen=True)
class RunSelection:
    datasets: tuple[str, ...]
    modalities: tuple[str, ...]
    model: str = "gpt-4o-mini"
    max_tokens: int = 1024
    temperature: float = 0.1
    max_steps: int = 6
    tool_budget: int = 4

    def __post_init__(self) -> None:
        validate_model(self.model)
        for dataset in self.datasets:
            validate_dataset(dataset)
        for modality in self.modalities:
            validate_modality(modality)


@dataclass(frozen=True)
class QueryInput:
    query_id: str
    dataset: str
    query: str
    ground_truth: str
    options: Optional[str] = None
    metadata: Optional[dict[str, Any]] = None

    def __post_init__(self) -> None:
        validate_dataset(self.dataset)


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
    return round(
        (prompt_tokens / 1000.0) * rates["input"]
        + (completion_tokens / 1000.0) * rates["output"],
        6,
    )


def _norm(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.rstrip(".,:;!")
    return _RE_WHITESPACE.sub(" ", text)


def _build_alias_map() -> dict[str, str]:
    alias_map: dict[str, str] = {}
    for canonical, aliases in GAIA_ALIASES.items():
        canon = _norm(canonical)
        for alias in aliases:
            alias_map[_norm(alias)] = canon
    return alias_map


_GAIA_ALIAS_MAP = _build_alias_map()


def _resolve_alias(text: str) -> str:
    n = _norm(text)
    return _GAIA_ALIAS_MAP.get(n, n)


def _try_float(text: str) -> Optional[float]:
    try:
        return float(text.replace(",", "").replace("$", "").replace("%", "").strip())
    except ValueError:
        return None


def _eval_gaia(predicted: str, ground_truth: str) -> bool:
    p, g = _norm(predicted), _norm(ground_truth)
    if p == g:
        return True
    if _resolve_alias(p) == _resolve_alias(g):
        return True

    pn, gn = _try_float(predicted), _try_float(ground_truth)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6

    if g and (g in p or p in g):
        return True
    return False


def _eval_mmlu(predicted: str, ground_truth: str) -> bool:
    def _letter(text: str) -> str:
        text = text.strip()
        m = re.search(r"\b([A-Ja-j])\b", text)
        if m:
            return m.group(1).upper()
        return text[0].upper() if text and text[0].upper() in "ABCDEFGHIJ" else text.upper()

    return _letter(predicted) == _letter(ground_truth)


def _extract_math_final(text: str) -> str:
    text = (text or "").strip()
    # Remove common markdown/code wrappers first.
    text = text.strip("`").strip('"').strip("'").strip()
    text = _strip_math_wrappers(text)
    m = _RE_THE_ANSWER.search(text)
    if m:
        return _strip_math_wrappers(m.group(1).strip().rstrip("."))
    m = _RE_FINAL_ANSWER.search(text)
    if m:
        return _strip_math_wrappers(m.group(1).strip().rstrip("."))
    m = _RE_BOXED.search(text)
    if m:
        return _strip_math_wrappers(m.group(1).strip())
    m = _RE_EQ_END.search(text)
    if m:
        return _strip_math_wrappers(m.group(1).strip())
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return _strip_math_wrappers(lines[-1].rstrip(".")) if lines else text


def _strip_math_wrappers(expr: str) -> str:
    expr = (expr or "").strip()
    # Remove escaped inline/display wrappers if present at boundaries.
    if expr.startswith(r"\(") and expr.endswith(r"\)"):
        expr = expr[2:-2].strip()
    if expr.startswith(r"\[") and expr.endswith(r"\]"):
        expr = expr[2:-2].strip()
    # Remove markdown/math delimiters.
    while len(expr) >= 2 and expr.startswith("$") and expr.endswith("$"):
        expr = expr[1:-1].strip()
    # Remove \left/\right artifacts.
    expr = expr.replace(r"\left", "").replace(r"\right", "")
    return expr.strip()


def _preprocess_math(expr: str) -> str:
    expr = _strip_math_wrappers(expr)

    out = (
        expr.replace("π", "pi")
        .replace("\u03c0", "pi")
        .replace("\u2212", "-")
        .replace("\u00d7", "*")
        .replace("\u00f7", "/")
        .replace("²", "**2")
        .replace("³", "**3")
        .replace("^", "**")
        .strip()
    )
    out = _RE_IMPLICIT_MUL.sub(r"\1*\2", out)
    return _RE_WHITESPACE.sub("", out)


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


def _norm_math(text: str) -> str:
    t = _strip_math_wrappers(text)

    t = t.lower()
    t = (
        t.replace("−", "-")
        .replace("∞", r"\infty")
    )

    return (
        _RE_WHITESPACE.sub("", t)
        .replace("^", "**")
        .replace("×", "*")
        .replace("÷", "/")
        .replace("π", "pi")
        .rstrip(".")
    )


def _eval_math(predicted: str, ground_truth: str) -> bool:
    # Many model outputs include rationale; pull likely final answer first.
    p = _extract_math_final(predicted)
    g = (ground_truth or "").strip()

    r = _sympy_equal_math(p, g)
    if r is not None:
        return r
    if _norm_math(p) == _norm_math(g):
        return True
    pn, gn = _try_float(p), _try_float(g)
    if pn is not None and gn is not None:
        return abs(pn - gn) < 1e-6
    return False


def _compute_correctness(dataset: str, predicted: str, ground_truth: str) -> bool:
    if not predicted or not ground_truth:
        return False
    if dataset == "gaia":
        return _eval_gaia(predicted, ground_truth)
    if dataset == "mmlu_pro":
        return _eval_mmlu(predicted, ground_truth)
    if dataset == "math_hard":
        return _eval_math(predicted, ground_truth)
    return predicted.strip().lower() == ground_truth.strip().lower()


def _ground_truth_hint(dataset: str) -> str:
    if dataset == "mmlu_pro":
        return "single option letter"
    if dataset == "math_hard":
        return "single numeric/symbolic final result"
    if dataset == "swe_bench_verified":
        return "short task-completion output"
    return "short exact answer"


def _tools_available(modality: str) -> str:
    if modality in ("react", "multiagent"):
        return "none"
    return "not_applicable"


def _build_system_values(selection: RunSelection) -> dict[str, Any]:
    return {
        "model_name": selection.model,
        "output_contract": "Return accurate final answer only.",
        "max_steps": selection.max_steps,
        "tool_budget": selection.tool_budget,
    }


def _build_user_values(query: QueryInput, modality: str) -> dict[str, Any]:
    values: dict[str, Any] = {
        "dataset": query.dataset,
        "query": query.query,
    }
    if query.dataset == "mmlu_pro":
        values["options"] = query.options or ""
    if query.dataset in ("gaia", "math_hard", "swe_bench_verified"):
        values["ground_truth_format_hint"] = _ground_truth_hint(query.dataset)
    if modality in ("react", "multiagent"):
        values["tools_available"] = _tools_available(modality)
    return values


def run_selected(
    *,
    queries: list[QueryInput],
    selection: RunSelection,
    client: Optional[OpenAI] = None,
) -> list[UnifiedExperimentRecord]:
    """
    Minimal scalable runner:
    - Runs only requested datasets/modalities
    - Single-call execution path for all modalities (prompt-driven baseline)
    """
    llm = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    results: list[UnifiedExperimentRecord] = []

    wanted_datasets = set(selection.datasets)
    wanted_modalities = set(selection.modalities)

    for query in queries:
        if query.dataset not in wanted_datasets:
            continue

        for modality in wanted_modalities:
            system_values = _build_system_values(selection)
            user_values = _build_user_values(query, modality)

            resolved = resolve_prompts(
                modality=modality,
                dataset=query.dataset,
                model_name=selection.model,
                system_values=system_values,
                user_values=user_values,
            )

            predicted, usage, error, elapsed = vanallia_operator(
                client=llm,
                model_name=selection.model,
                question=resolved.user_prompt,
                system_prompt=resolved.system_prompt,
                max_tokens=selection.max_tokens,
                temperature=selection.temperature,
            )

            is_correct = _compute_correctness(
                dataset=query.dataset,
                predicted=predicted,
                ground_truth=query.ground_truth,
            )
            error_type = "runtime_error" if error else None

            results.append(
                UnifiedExperimentRecord(
                    query_id=query.query_id,
                    dataset=query.dataset,
                    modality=modality,
                    model=selection.model,
                    query=query.query,
                    ground_truth=query.ground_truth,
                    predicted_answer=predicted,
                    is_correct=is_correct,
                    reasoning_trace=predicted if modality in ("cot", "react", "multiagent") else None,
                    tool_trace=None,
                    prompt_tokens=usage.get("prompt_tokens", 0),
                    completion_tokens=usage.get("completion_tokens", 0),
                    total_tokens=usage.get("total_tokens", 0),
                    cost_usd=_estimate_cost(
                        selection.model,
                        usage.get("prompt_tokens", 0),
                        usage.get("completion_tokens", 0),
                    ),
                    latency_s=round(elapsed, 3),
                    error_type=error_type,
                    error_message=error,
                    system_prompt_version=resolved.system_prompt_version,
                    user_prompt_version=resolved.user_prompt_version,
                    prompt_hash=resolved.prompt_hash,
                    metadata=query.metadata or {},
                )
            )

    return results


def default_selection() -> RunSelection:
    return RunSelection(
        datasets=ALLOWED_DATASETS,
        modalities=ALLOWED_MODALITIES,
    )
