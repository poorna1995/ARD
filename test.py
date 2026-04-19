
# """
# runner.py  (updated)
# ====================
# Core runner: QueryInput, RunSelection, run_selected().

# ReAct changes vs. original
# ───────────────────────────
# 1. react_operator called with query_input=query  →  gets benchmark-aware
#    tool registry and system prompt automatically.
# 2. finish[] extraction: scan trace for last "finish" step — O(S), no regex.
# 3. reasoning_steps = len(non-finish trace steps)  (actual tool calls made).
# 4. tool_trace: serialised trace stored in UnifiedExperimentRecord.tool_trace.
# 5. per-step max_tokens capped at 512 for react (keeps costs down; vanilla/cot
#    still use selection.max_tokens).
# 6. All vanilla / cot / routellm / multiagent paths: zero changes.
# """

# from __future__ import annotations

# import json
# import os
# import re
# import time
# import unicodedata
# from dataclasses import dataclass
# from typing import Any, Optional

# from dotenv import load_dotenv
# from openai import OpenAI

# load_dotenv()

# from agents.react_operator import react_operator          # ← real ReAct agent
# from agents.vanilla_operator import vanilla_operator
# from baselines.constants import (
#     ALLOWED_DATASETS,
#     ALLOWED_MODALITIES,
#     validate_dataset,
#     validate_modality,
#     validate_model,
# )
# from baselines.route_llm import (
#     ROUTELLM_AVAILABLE,
#     ROUTELLM_IMPORT_ERROR,
#     RoutellmConfig,
#     build_controller,
#     routellm_chat_completion,
# )
# from baselines.prompt_resolver import resolve_prompts
# from baselines.schema import UnifiedExperimentRecord

# try:
#     from sympy import N, simplify, sympify
#     from sympy.parsing.latex import parse_latex
#     SYMPY_AVAILABLE = True
# except Exception:
#     SYMPY_AVAILABLE = False


# # ══════════════════════════════════════════════════════════════════════════════
# # § 1  CONSTANTS
# # ══════════════════════════════════════════════════════════════════════════════

# COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
#     "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
#     "gpt-4o":      {"input": 0.0025,  "output": 0.01},
# }

# # ReAct uses short per-step generations to keep cost down
# _REACT_MAX_TOKENS_PER_STEP = 512

# GAIA_ALIASES: dict[str, list[str]] = {
#     "united states":           ["us", "usa", "u.s.", "u.s.a.", "united states of america"],
#     "united kingdom":          ["uk", "u.k.", "britain", "great britain"],
#     "world war ii":            ["wwii", "ww2", "world war 2", "second world war"],
#     "artificial intelligence": ["ai"],
#     "united nations":          ["un", "u.n."],
# }

# # Pre-compiled regexes — module-level, compiled once
# _RE_WHITESPACE    = re.compile(r"\s+")
# _RE_THE_ANSWER    = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
# _RE_FINAL_ANSWER  = re.compile(r"[Ff]inal answer is[:\s]+([^\n]+)")
# _RE_BOXED         = re.compile(r"\\boxed\{([^}]+)\}")
# _RE_EQ_END        = re.compile(r"=\s*([^\n=]+)\s*$", re.MULTILINE)
# _RE_IMPLICIT_MUL  = re.compile(r"(\d)(pi\b|[a-df-wyzA-Z])")
# _RE_LETTER_MMLU   = re.compile(r"\b([A-Ja-j])\b")
# _GAIA_PREAMBLE    = re.compile(
#     r"^(To\s|In\s+order|We\s+need|Let's\s|Let\s+us\s|Here's\s|Here\s+is\s|"
#     r"The\s+question|However,|I\s+need|Based\s+on|The\s+first\s+step|"
#     r"I\s+will|I'll\s|Note\s+that)",
#     re.IGNORECASE,
# )
# _RE_STEP_MARKERS  = re.compile(
#     r"(?m)(?:^\s*\d+[\.)]\s+\S|^\s*[-*•]\s+\S|\bstep\s+\d+)",
#     re.IGNORECASE,
# )
# _RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# _MATH_TRANS = str.maketrans({
#     "−": "-", "∞": r"\infty", "^": "**", "×": "*", "÷": "/", "π": "pi",
# })


# # ══════════════════════════════════════════════════════════════════════════════
# # § 2  ALIAS MAP (built once at import)
# # ══════════════════════════════════════════════════════════════════════════════

# def _norm(text: str) -> str:
#     text = unicodedata.normalize("NFKD", text.strip().lower())
#     filtered = (c for c in text if not unicodedata.combining(c))
#     text = "".join(filtered).rstrip(".,:;!")
#     return _RE_WHITESPACE.sub(" ", text)


# def _build_alias_map() -> dict[str, str]:
#     alias_map: dict[str, str] = {}
#     for canonical, aliases in GAIA_ALIASES.items():
#         canon = _norm(canonical)
#         for alias in aliases:
#             alias_map[_norm(alias)] = canon
#     return alias_map


# _GAIA_ALIAS_MAP: dict[str, str] = _build_alias_map()


# # ══════════════════════════════════════════════════════════════════════════════
# # § 3  DATA CONTRACTS
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass(frozen=True)
# class RunSelection:
#     datasets:   tuple[str, ...]
#     modalities: tuple[str, ...]
#     model:       str   = "gpt-4o-mini"
#     max_tokens:  int   = 2048
#     temperature: float = 0.1
#     max_steps:   int   = 6
#     tool_budget: int   = 4
#     routellm_router:       str   = "mf"
#     routellm_threshold:    float = 0.11593
#     routellm_strong_model: str   = "gpt-4o"
#     routellm_weak_model:   str   = "gpt-4o-mini"

#     def __post_init__(self) -> None:
#         validate_model(self.model)
#         for d in self.datasets:   validate_dataset(d)
#         for m in self.modalities: validate_modality(m)
#         if "routellm" in self.modalities:
#             if not (0.0 <= self.routellm_threshold <= 1.0):
#                 raise ValueError("routellm_threshold must be in [0, 1].")

#     def routellm_config(self) -> RoutellmConfig:
#         return RoutellmConfig(
#             router=self.routellm_router.strip(),
#             threshold=float(self.routellm_threshold),
#             strong_model=self.routellm_strong_model.strip(),
#             weak_model=self.routellm_weak_model.strip(),
#         )


# @dataclass(frozen=True)
# class QueryInput:
#     query_id:     str
#     dataset:      str
#     query:        str
#     ground_truth: str
#     options:  Optional[str]            = None
#     metadata: Optional[dict[str, Any]] = None

#     def __post_init__(self) -> None:
#         validate_dataset(self.dataset)


# # ══════════════════════════════════════════════════════════════════════════════
# # § 4  COST HELPERS
# # ══════════════════════════════════════════════════════════════════════════════

# def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
#     rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
#     return round(
#         (prompt_tokens     / 1000.0) * rates["input"] +
#         (completion_tokens / 1000.0) * rates["output"], 6,
#     )


# def _estimate_cost_routellm(
#     selection: RunSelection,
#     resolved_model: Optional[str],
#     prompt_tokens: int,
#     completion_tokens: int,
# ) -> float:
#     r = (resolved_model or "").lower()
#     for key in (selection.routellm_strong_model, selection.routellm_weak_model,
#                 "gpt-4o", "gpt-4o-mini"):
#         if key and key.lower() in r and key in COST_PER_1K_TOKENS:
#             return _estimate_cost(key, prompt_tokens, completion_tokens)
#     fallback = (selection.routellm_weak_model
#                 if selection.routellm_weak_model in COST_PER_1K_TOKENS
#                 else "gpt-4o-mini")
#     return _estimate_cost(fallback, prompt_tokens, completion_tokens)


# # ══════════════════════════════════════════════════════════════════════════════
# # § 5  EVALUATION HELPERS  (unchanged from original)
# # ══════════════════════════════════════════════════════════════════════════════

# def _try_float(text: str) -> Optional[float]:
#     try:
#         return float(text.replace(",", "").replace("$", "").replace("%", "").strip())
#     except ValueError:
#         return None


# def _eval_gaia(predicted: str, ground_truth: str) -> bool:
#     p_norm, g_norm = _norm(predicted), _norm(ground_truth)
#     if p_norm == g_norm:
#         return True
#     if _GAIA_ALIAS_MAP.get(p_norm, p_norm) == _GAIA_ALIAS_MAP.get(g_norm, g_norm):
#         return True
#     pn, gn = _try_float(predicted), _try_float(ground_truth)
#     return pn is not None and gn is not None and abs(pn - gn) < 1e-6


# def _extract_letter_mmlu(text: str) -> str:
#     text = text.strip()
#     m = _RE_LETTER_MMLU.search(text)
#     if m:
#         return m.group(1).upper()
#     return text[0].upper() if text and text[0].upper() in "ABCDEFGHIJ" else text.upper()


# def _eval_mmlu(predicted: str, ground_truth: str) -> bool:
#     return _extract_letter_mmlu(predicted) == _extract_letter_mmlu(ground_truth)


# def _count_reasoning_steps(raw: str, predicted: str) -> int:
#     text = (raw or "").strip()
#     pred = (predicted or "").strip()
#     if not text or text == pred:
#         return 0
#     markers = _RE_STEP_MARKERS.findall(text)
#     if markers:
#         return len(markers)
#     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#     if not lines:
#         return 1
#     last = lines[-1]
#     if _RE_THE_ANSWER.search(last) or _RE_FINAL_ANSWER.search(last):
#         lines = lines[:-1]
#     elif pred and (last.rstrip(".,;:!") == pred or last.endswith(pred)):
#         lines = lines[:-1]
#     return max(1, len(lines))


# def _strip_math_wrappers(expr: str) -> str:
#     expr = (expr or "").strip()
#     if expr.startswith(r"\(") and expr.endswith(r"\)"):
#         expr = expr[2:-2].strip()
#     if expr.startswith(r"\[") and expr.endswith(r"\]"):
#         expr = expr[2:-2].strip()
#     while len(expr) >= 2 and expr[0] == "$" and expr[-1] == "$":
#         expr = expr[1:-1].strip()
#     return expr.replace(r"\left", "").replace(r"\right", "").strip()


# def _extract_math_final(text: str) -> str:
#     text = (text or "").strip().strip("`").strip('"').strip("'").strip()
#     text = _strip_math_wrappers(text)
#     for pattern in (_RE_THE_ANSWER, _RE_FINAL_ANSWER, _RE_BOXED, _RE_EQ_END):
#         m = pattern.search(text)
#         if m:
#             return _strip_math_wrappers(m.group(1).strip().rstrip("."))
#     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
#     return _strip_math_wrappers(lines[-1].rstrip(".")) if lines else text


# def _preprocess_math(expr: str) -> str:
#     expr = _strip_math_wrappers(expr)
#     out = (expr.replace("π","pi").replace("\u03c0","pi").replace("\u2212","-")
#                .replace("\u00d7","*").replace("\u00f7","/")
#                .replace("²","**2").replace("³","**3")
#                .translate(_MATH_TRANS).strip())
#     out = _RE_IMPLICIT_MUL.sub(r"\1*\2", out)
#     return _RE_WHITESPACE.sub("", out)


# def _norm_math(text: str) -> str:
#     t = _strip_math_wrappers(text).lower()
#     return _RE_WHITESPACE.sub("", t.translate(_MATH_TRANS)).rstrip(".")


# def _sympy_equal_math(predicted: str, ground_truth: str) -> Optional[bool]:
#     if not SYMPY_AVAILABLE:
#         return None
#     a1, a2 = _preprocess_math(predicted), _preprocess_math(ground_truth)
#     for fn in (sympify, parse_latex):
#         try:
#             diff = simplify(fn(a1) - fn(a2))
#             if diff == 0 or abs(complex(N(diff))) < 1e-2:
#                 return True
#         except Exception:
#             continue
#     return None


# def _eval_math(predicted: str, ground_truth: str) -> bool:
#     p = _extract_math_final(predicted)
#     g = (ground_truth or "").strip()
#     r = _sympy_equal_math(p, g)
#     if r is not None:
#         return r
#     if _norm_math(p) == _norm_math(g):
#         return True
#     pn, gn = _try_float(p), _try_float(g)
#     return pn is not None and gn is not None and abs(pn - gn) < 1e-6


# def _compute_correctness(dataset: str, predicted: str, ground_truth: str) -> bool:
#     if not predicted or not ground_truth:
#         return False
#     if dataset == "gaia":      return _eval_gaia(predicted, ground_truth)
#     if dataset == "mmlu_pro":  return _eval_mmlu(predicted, ground_truth)
#     if dataset == "math_hard": return _eval_math(predicted, ground_truth)
#     return predicted.strip().lower() == ground_truth.strip().lower()


# # ══════════════════════════════════════════════════════════════════════════════
# # § 6  PROMPT HELPERS  (unchanged from original)
# # ══════════════════════════════════════════════════════════════════════════════

# def _ground_truth_hint(dataset: str) -> str:
#     if dataset == "mmlu_pro":           return "gold is one letter A–J; output that letter only"
#     if dataset == "math_hard":          return "gold is LaTeX/math final; match that form"
#     if dataset == "swe_bench_verified": return "gold is patch/diff text; output minimal patch"
#     return "gold is short factual (word, number, name, date); match exactly"


# def _output_contract_for(dataset: str, modality: str) -> str:
#     if modality in ("vanilla", "routellm"):
#         if dataset == "gaia":               return "Output one line: final answer only."
#         if dataset == "mmlu_pro":           return "Output one character: A–J only."
#         if dataset == "math_hard":          return "Output final math only; no sentences."
#         if dataset == "swe_bench_verified": return "Output patch only; no prose."
#     if modality == "cot":
#         if dataset == "gaia":               return "Brief reasoning then: The answer is <value>."
#         if dataset == "mmlu_pro":           return "Brief reasoning then: The answer is <A–J>."
#         if dataset == "math_hard":          return "Brief reasoning then: The answer is <TeX/math>."
#         if dataset == "swe_bench_verified": return "Brief reasoning then: The answer is <patch>."
#     if dataset == "mmlu_pro":           return "End with one letter A–J."
#     if dataset == "math_hard":          return "End with final math expression only."
#     if dataset == "swe_bench_verified": return "End with patch or minimal artifact."
#     return "End with one-line final answer."


# def _build_system_values(selection: RunSelection, dataset: str, modality: str) -> dict[str, Any]:
#     if modality == "routellm":
#         model_name = (
#             f"RouteLLM router={selection.routellm_router} thr={selection.routellm_threshold:g} "
#             f"strong={selection.routellm_strong_model} weak={selection.routellm_weak_model}"
#         )
#     else:
#         model_name = selection.model
#     return {
#         "model_name":      model_name,
#         "output_contract": _output_contract_for(dataset, modality),
#         "max_steps":       selection.max_steps,
#         "tool_budget":     selection.tool_budget,
#     }


# def _build_user_values(query: QueryInput, modality: str) -> dict[str, Any]:
#     values: dict[str, Any] = {"dataset": query.dataset, "query": query.query}
#     if query.dataset == "mmlu_pro":
#         values["options"] = query.options or ""
#     if query.dataset in ("gaia", "math_hard", "swe_bench_verified"):
#         values["ground_truth_format_hint"] = _ground_truth_hint(query.dataset)
#     return values


# def _extract_short_answer(raw: str, dataset: str) -> tuple[str, str]:
#     full = (raw or "").strip()
#     if not full:
#         return "", ""
#     if dataset in ("gaia", "swe_bench_verified"):
#         m = _RE_THE_ANSWER.search(full)
#         if m: return m.group(1).strip().rstrip(".,;:!"), full
#         m = _RE_FINAL_ANSWER.search(full)
#         if m: return m.group(1).strip().rstrip(".,;:!"), full
#         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
#         for line in reversed(lines):
#             if len(line) > 220 or _GAIA_PREAMBLE.match(line): continue
#             return line.rstrip(".,;:!"), full
#         for frag in reversed(_RE_SENTENCE_SPLIT.split(full)):
#             frag = frag.strip()
#             if not frag or len(frag) > 220 or _GAIA_PREAMBLE.match(frag): continue
#             return frag.rstrip(".,;:!"), full
#         return (lines[-1].rstrip(".,;:!") if lines else full), full
#     if dataset == "mmlu_pro":
#         m = _RE_THE_ANSWER.search(full)
#         if m:
#             frag = m.group(1).strip().rstrip(".")
#             lm = re.match(r"\(?([A-Ja-j])\)?[\.\):\s]?", frag)
#             return (lm.group(1).upper() if lm else frag), full
#         letters = _RE_LETTER_MMLU.findall(full)
#         if letters: return letters[-1].upper(), full
#         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
#         return (lines[-1].rstrip(".") if lines else full), full
#     if dataset == "math_hard":
#         return _extract_math_final(full), full
#     return full, full


# # ══════════════════════════════════════════════════════════════════════════════
# # § 7  REACT ANSWER EXTRACTOR  — O(S) scan of trace, no regex on raw_output
# # ══════════════════════════════════════════════════════════════════════════════

# def _extract_react_answer(
#     react_trace: list[dict[str, Any]],
#     raw_output: str,
#     dataset: str,
# ) -> tuple[str, int]:
#     """
#     Extract predicted answer and step count from a ReAct trace.

#     Strategy (in priority order):
#     1. Last finish[] action in trace  — cleanest, model explicitly declared done
#     2. Last non-finish observation    — model produced output but forgot finish[]
#     3. Fallback to _extract_short_answer on raw_output

#     Returns
#     ───────
#     (predicted_answer, reasoning_steps)
#       reasoning_steps = number of non-finish steps (actual tool calls)
#     """
#     finish_steps   = [s for s in react_trace if s.get("action") == "finish"]
#     non_finish     = [s for s in react_trace if s.get("action") != "finish"]
#     reasoning_steps = len(non_finish)

#     # Priority 1: explicit finish
#     if finish_steps:
#         answer = finish_steps[-1].get("action_input", "").strip()
#         if answer:
#             return answer, reasoning_steps

#     # Priority 2: last observation from a non-error step
#     for step in reversed(non_finish):
#         if not step.get("error") and step.get("observation"):
#             obs = step["observation"].strip()
#             if obs and len(obs) < 400:    # skip huge observations
#                 return obs, reasoning_steps

#     # Priority 3: regex on raw_output (same as other modalities)
#     predicted, _ = _extract_short_answer(raw_output, dataset)
#     return predicted, reasoning_steps


# # ══════════════════════════════════════════════════════════════════════════════
# # § 8  MAIN RUNNER
# # ══════════════════════════════════════════════════════════════════════════════

# def run_selected(
#     *,
#     queries: list[QueryInput],
#     selection: RunSelection,
#     client: Optional[OpenAI] = None,
# ) -> list[UnifiedExperimentRecord]:
#     """
#     Run all (query × modality) combinations and return records.

#     ReAct modality  → react_operator (benchmark-aware tool loop)
#     All others      → unchanged single-call path
#     """
#     llm = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
#     results: list[UnifiedExperimentRecord] = []

#     wanted_datasets   = set(selection.datasets)
#     wanted_modalities = set(selection.modalities)

#     # ── RouteLLM client — built once ──────────────────────────────────────────
#     routellm_client: Optional[Any] = None
#     if "routellm" in wanted_modalities:
#         if not ROUTELLM_AVAILABLE:
#             msg = "Modality 'routellm' needs the RouteLLM PyPI package."
#             if ROUTELLM_IMPORT_ERROR:
#                 msg += f"\n\nImport diagnostic:\n  {ROUTELLM_IMPORT_ERROR}"
#             raise ImportError(msg)
#         routellm_client = build_controller(selection.routellm_config())

#     # ── Pre-compute system_values per (dataset, modality) — O(D×M) not O(N×M)
#     system_values_cache: dict[tuple[str, str], dict[str, Any]] = {
#         (ds, mod): _build_system_values(selection, ds, mod)
#         for ds in wanted_datasets
#         for mod in wanted_modalities
#         if mod != "react"   # react builds its own prompts internally
#     }

#     for query in queries:
#         if query.dataset not in wanted_datasets:
#             continue

#         for modality in wanted_modalities:
#             is_routellm = modality == "routellm"
#             is_react    = modality == "react"

#             # ════════════════════════════════════════════════════════════════
#             # REACT PATH — full benchmark-aware tool loop
#             # ════════════════════════════════════════════════════════════════
#             if is_react:
#                 raw_output, usage, error, elapsed, react_trace = react_operator(
#                     client=llm,
#                     model_name=selection.model,
#                     question=query.query,        # fallback; query_input takes priority
#                     query_input=query,           # ← full QueryInput for tool/prompt selection
#                     dataset=query.dataset,
#                     max_steps=selection.max_steps,
#                     max_tokens=_REACT_MAX_TOKENS_PER_STEP,
#                     temperature=0.0,             # greedy — reproducible per paper
#                 )

#                 # O(S) extraction — no regex on raw_output unless finish[] absent
#                 predicted, reasoning_steps = _extract_react_answer(
#                     react_trace, raw_output, query.dataset
#                 )

#                 tool_trace_json = json.dumps(react_trace, ensure_ascii=False)
#                 resolved_model  = None

#             # ════════════════════════════════════════════════════════════════
#             # ROUTELLM PATH — unchanged
#             # ════════════════════════════════════════════════════════════════
#             elif is_routellm:
#                 system_values = system_values_cache[(query.dataset, modality)]
#                 user_values   = _build_user_values(query, modality)
#                 resolved = resolve_prompts(
#                     modality=modality, dataset=query.dataset,
#                     model_name=selection.model,
#                     system_values=system_values, user_values=user_values,
#                 )
#                 assert routellm_client is not None
#                 raw_output, usage, error, elapsed, resolved_model = routellm_chat_completion(
#                     routellm_client, cfg=selection.routellm_config(),
#                     system_prompt=resolved.system_prompt,
#                     user_prompt=resolved.user_prompt,
#                     max_tokens=selection.max_tokens,
#                     temperature=selection.temperature,
#                 )
#                 predicted, _    = _extract_short_answer(raw_output, query.dataset)
#                 reasoning_steps = _count_reasoning_steps(raw_output, predicted)
#                 tool_trace_json = None

#             # ════════════════════════════════════════════════════════════════
#             # VANILLA / COT / MULTIAGENT PATHS — unchanged
#             # ════════════════════════════════════════════════════════════════
#             else:
#                 system_values = system_values_cache[(query.dataset, modality)]
#                 user_values   = _build_user_values(query, modality)
#                 resolved = resolve_prompts(
#                     modality=modality, dataset=query.dataset,
#                     model_name=selection.model,
#                     system_values=system_values, user_values=user_values,
#                 )
#                 raw_output, usage, error, elapsed = vanallia_operator(
#                     client=llm, model_name=selection.model,
#                     question=resolved.user_prompt,
#                     system_prompt=resolved.system_prompt,
#                     max_tokens=selection.max_tokens,
#                     temperature=selection.temperature,
#                 )
#                 predicted, _    = _extract_short_answer(raw_output, query.dataset)
#                 reasoning_steps = _count_reasoning_steps(raw_output, predicted)
#                 tool_trace_json = None
#                 resolved_model  = None

#             # ── Build record ───────────────────────────────────────────────
#             raw_stripped = raw_output.strip()
#             row_meta: dict[str, Any] = dict(query.metadata) if query.metadata else {}

#             if raw_stripped != predicted.strip():
#                 row_meta["raw_model_output"] = raw_stripped[:50_000]

#             if is_routellm:
#                 row_meta.update({
#                     "routellm_router":       selection.routellm_router,
#                     "routellm_threshold":    selection.routellm_threshold,
#                     "routellm_strong_model": selection.routellm_strong_model,
#                     "routellm_weak_model":   selection.routellm_weak_model,
#                 })
#                 if resolved_model:
#                     row_meta["routellm_resolved_model"] = resolved_model

#             if is_react:
#                 row_meta["react_steps_taken"] = reasoning_steps

#             is_correct = _compute_correctness(
#                 dataset=query.dataset,
#                 predicted=predicted,
#                 ground_truth=query.ground_truth,
#             )

#             reasoning_trace: Optional[str] = None
#             if modality in ("cot", "react", "multiagent"):
#                 reasoning_trace = raw_output
#             elif raw_stripped != predicted.strip():
#                 reasoning_trace = raw_output

#             prompt_tokens     = usage.get("prompt_tokens",     0)
#             completion_tokens = usage.get("completion_tokens", 0)
#             cost_usd = (
#                 _estimate_cost_routellm(selection, resolved_model,
#                                         prompt_tokens, completion_tokens)
#                 if is_routellm
#                 else _estimate_cost(selection.model, prompt_tokens, completion_tokens)
#             )

#             # Resolve prompt version strings safely (react has no `resolved`)
#             sys_ver  = getattr(locals().get("resolved"), "system_prompt_version", "react-v1")
#             usr_ver  = getattr(locals().get("resolved"), "user_prompt_version",   "react-v1")
#             p_hash   = getattr(locals().get("resolved"), "prompt_hash",           "")

#             results.append(UnifiedExperimentRecord(
#                 query_id=query.query_id,
#                 dataset=query.dataset,
#                 modality=modality,
#                 model="routellm" if is_routellm else selection.model,
#                 query=query.query,
#                 ground_truth=query.ground_truth,
#                 predicted_answer=predicted,
#                 is_correct=is_correct,
#                 reasoning_trace=reasoning_trace,
#                 tool_trace=tool_trace_json,
#                 reasoning_steps=reasoning_steps,
#                 prompt_tokens=prompt_tokens,
#                 completion_tokens=completion_tokens,
#                 total_tokens=usage.get("total_tokens", 0),
#                 cost_usd=cost_usd,
#                 latency_s=round(elapsed, 3),
#                 error_type="runtime_error" if error else None,
#                 error_message=error,
#                 system_prompt_version=sys_ver,
#                 user_prompt_version=usr_ver,
#                 prompt_hash=p_hash,
#                 metadata=row_meta,
#             ))

#     return results


# def default_selection() -> RunSelection:
#     modalities = tuple(m for m in ALLOWED_MODALITIES if m != "routellm")
#     return RunSelection(datasets=ALLOWED_DATASETS, modalities=modalities)




# # # # from __future__ import annotations

# # # # from dataclasses import dataclass
# # # # import re
# # # # from typing import Any, Optional
# # # # import unicodedata
# # # # from dotenv import load_dotenv
# # # # from openai import OpenAI
# # # # import os
# # # # load_dotenv()

# # # # from agents.vanallia_operator import vanallia_operator
# # # # from baselines.constants import (
# # # #     ALLOWED_DATASETS,
# # # #     ALLOWED_MODALITIES,
# # # #     validate_dataset,
# # # #     validate_modality,
# # # #     validate_model,
# # # # )
# # # # from baselines.route_llm import (
# # # #     ROUTELLM_AVAILABLE,
# # # #     ROUTELLM_IMPORT_ERROR,
# # # #     RoutellmConfig,
# # # #     build_controller,
# # # #     routellm_chat_completion,
# # # # )
# # # # from baselines.prompt_resolver import resolve_prompts
# # # # from baselines.schema import UnifiedExperimentRecord

# # # # try:
# # # #     from sympy import N, simplify, sympify
# # # #     from sympy.parsing.latex import parse_latex

# # # #     SYMPY_AVAILABLE = True
# # # # except Exception:
# # # #     SYMPY_AVAILABLE = False


# # # # COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
# # # #     "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
# # # #     # Approximate list pricing; override if your org uses different tiers.
# # # #     "gpt-4o": {"input": 0.0025, "output": 0.01},
# # # # }

# # # # GAIA_ALIASES: dict[str, list[str]] = {
# # # #     "united states": ["us", "usa", "u.s.", "u.s.a.", "united states of america"],
# # # #     "united kingdom": ["uk", "u.k.", "britain", "great britain"],
# # # #     "world war ii": ["wwii", "ww2", "world war 2", "second world war"],
# # # #     "artificial intelligence": ["ai"],
# # # #     "united nations": ["un", "u.n."],
# # # # }

# # # # _RE_WHITESPACE = re.compile(r"\s+")
# # # # _RE_THE_ANSWER = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
# # # # _RE_FINAL_ANSWER = re.compile(r"[Ff]inal answer is[:\s]+([^\n]+)")
# # # # _RE_BOXED = re.compile(r"\\boxed\{([^}]+)\}")
# # # # _RE_EQ_END = re.compile(r"=\s*([^\n=]+)\s*$", re.MULTILINE)
# # # # _RE_IMPLICIT_MUL = re.compile(r"(\d)(pi\b|[a-df-wyzA-Z])")


# # # # @dataclass(frozen=True)
# # # # class RunSelection:
# # # #     datasets: tuple[str, ...]
# # # #     modalities: tuple[str, ...]
# # # #     model: str = "gpt-4o-mini"
# # # #     max_tokens: int = 2048
# # # #     temperature: float = 0.1
# # # #     max_steps: int = 6
# # # #     tool_budget: int = 4
# # # #     # RouteLLM (https://github.com/lm-sys/routellm); used when modality is `routellm`.
# # # #     routellm_router: str = "mf"
# # # #     routellm_threshold: float = 0.11593
# # # #     routellm_strong_model: str = "gpt-4o"
# # # #     routellm_weak_model: str = "gpt-4o-mini"

# # # #     def __post_init__(self) -> None:
# # # #         validate_model(self.model)
# # # #         for dataset in self.datasets:
# # # #             validate_dataset(dataset)
# # # #         for modality in self.modalities:
# # # #             validate_modality(modality)
# # # #         if "routellm" in self.modalities:
# # # #             if not (0.0 <= self.routellm_threshold <= 1.0):
# # # #                 raise ValueError("routellm_threshold must be in [0, 1].")
# # # #             if not self.routellm_router.strip():
# # # #                 raise ValueError("routellm_router must be non-empty.")

# # # #     def routellm_config(self) -> RoutellmConfig:
# # # #         return RoutellmConfig(
# # # #             router=self.routellm_router.strip(),
# # # #             threshold=float(self.routellm_threshold),
# # # #             strong_model=self.routellm_strong_model.strip(),
# # # #             weak_model=self.routellm_weak_model.strip(),
# # # #         )


# # # # @dataclass(frozen=True)
# # # # class QueryInput:
# # # #     query_id: str
# # # #     dataset: str
# # # #     query: str
# # # #     ground_truth: str
# # # #     options: Optional[str] = None
# # # #     metadata: Optional[dict[str, Any]] = None

# # # #     def __post_init__(self) -> None:
# # # #         validate_dataset(self.dataset)


# # # # def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
# # # #     rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
# # # #     return round(
# # # #         (prompt_tokens / 1000.0) * rates["input"]
# # # #         + (completion_tokens / 1000.0) * rates["output"],
# # # #         6,
# # # #     )


# # # # def _estimate_cost_routellm(
# # # #     selection: RunSelection,
# # # #     resolved_model: Optional[str],
# # # #     prompt_tokens: int,
# # # #     completion_tokens: int,
# # # # ) -> float:
# # # #     """Token cost using `response.model` when present; otherwise weak-model pricing."""
# # # #     r = (resolved_model or "").lower()
# # # #     for key in (
# # # #         selection.routellm_strong_model,
# # # #         selection.routellm_weak_model,
# # # #         "gpt-4o",
# # # #         "gpt-4o-mini",
# # # #     ):
# # # #         if key and key.lower() in r and key in COST_PER_1K_TOKENS:
# # # #             return _estimate_cost(key, prompt_tokens, completion_tokens)
# # # #     fallback = (
# # # #         selection.routellm_weak_model
# # # #         if selection.routellm_weak_model in COST_PER_1K_TOKENS
# # # #         else "gpt-4o-mini"
# # # #     )
# # # #     return _estimate_cost(fallback, prompt_tokens, completion_tokens)


# # # # def _norm(text: str) -> str:
# # # #     text = text.strip().lower()
# # # #     text = unicodedata.normalize("NFKD", text)
# # # #     text = "".join(c for c in text if not unicodedata.combining(c))
# # # #     text = text.rstrip(".,:;!")
# # # #     return _RE_WHITESPACE.sub(" ", text)


# # # # def _build_alias_map() -> dict[str, str]:
# # # #     alias_map: dict[str, str] = {}
# # # #     for canonical, aliases in GAIA_ALIASES.items():
# # # #         canon = _norm(canonical)
# # # #         for alias in aliases:
# # # #             alias_map[_norm(alias)] = canon
# # # #     return alias_map


# # # # _GAIA_ALIAS_MAP = _build_alias_map()


# # # # def _resolve_alias(text: str) -> str:
# # # #     n = _norm(text)
# # # #     return _GAIA_ALIAS_MAP.get(n, n)


# # # # def _try_float(text: str) -> Optional[float]:
# # # #     try:
# # # #         return float(text.replace(",", "").replace("$", "").replace("%", "").strip())
# # # #     except ValueError:
# # # #         return None



# # # # #Evaluator functions for GAIA dataset
# # # # def _eval_gaia(predicted: str, ground_truth: str) -> bool:
# # # #     p, g = _norm(predicted), _norm(ground_truth)
# # # #     if p == g:
# # # #         return True
# # # #     if _resolve_alias(p) == _resolve_alias(g):
# # # #         return True

# # # #     pn, gn = _try_float(predicted), _try_float(ground_truth)
# # # #     if pn is not None and gn is not None:
# # # #         return abs(pn - gn) < 1e-6

# # # #     # if g and (g in p or p in g):
# # # #     #     return True
# # # #     return False



# # # # def _eval_mmlu(predicted: str, ground_truth: str) -> bool:
# # # #     def _letter(text: str) -> str:
# # # #         text = text.strip()
# # # #         m = re.search(r"\b([A-Ja-j])\b", text)
# # # #         if m:
# # # #             return m.group(1).upper()
# # # #         return text[0].upper() if text and text[0].upper() in "ABCDEFGHIJ" else text.upper()

# # # #     return _letter(predicted) == _letter(ground_truth)


# # # # _RE_LETTER_MMLU = re.compile(r"\b([A-Ja-j])\b")
# # # # _GAIA_PREAMBLE = re.compile(
# # # #     r"^(To\s|In\s+order|We\s+need|Let's\s|Let\s+us\s|Here's\s|Here\s+is\s|The\s+question|However,|"
# # # #     r"I\s+need|Based\s+on|The\s+first\s+step|I\s+will|I'll\s|Note\s+that)",
# # # #     re.IGNORECASE,
# # # # )


# # # # def _count_reasoning_steps(raw: str, predicted: str) -> int:
# # # #     """
# # # #     Approximate number of reasoning steps from raw model output.
# # # #     Uses explicit numbering / bullets / 'Step N' when present; else non-empty
# # # #     lines excluding the final answer line.
# # # #     """
# # # #     text = (raw or "").strip()
# # # #     pred = (predicted or "").strip()
# # # #     if not text or text == pred:
# # # #         return 0

# # # #     n_numbered = len(re.findall(r"(?m)^\s*\d+[\.)]\s+\S", text))
# # # #     n_bullets = len(re.findall(r"(?m)^\s*[-*•]\s+\S", text))
# # # #     n_step_kw = len(re.findall(r"(?i)\bstep\s+\d+", text))
# # # #     structured = max(n_numbered, n_bullets, n_step_kw)
# # # #     if structured >= 1:
# # # #         return structured

# # # #     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
# # # #     if not lines:
# # # #         return 1
# # # #     last = lines[-1]
# # # #     if _RE_THE_ANSWER.search(last) or _RE_FINAL_ANSWER.search(last):
# # # #         lines = lines[:-1]
# # # #     elif pred and (last.rstrip(".,;:!") == pred or last.endswith(pred)):
# # # #         lines = lines[:-1]
# # # #     return max(1, len(lines))


# # # # def _extract_short_answer(raw: str, dataset: str) -> tuple[str, str]:
# # # #     """
# # # #     Normalize model output to a short final answer for storage and grading.
# # # #     Returns (extracted_answer, full_raw_text).
# # # #     """
# # # #     full = (raw or "").strip()
# # # #     if not full:
# # # #         return "", ""

# # # #     if dataset == "gaia" or dataset == "swe_bench_verified":
# # # #         m = _RE_THE_ANSWER.search(full)
# # # #         if m:
# # # #             return m.group(1).strip().rstrip(".,;:!"), full
# # # #         m = _RE_FINAL_ANSWER.search(full)
# # # #         if m:
# # # #             return m.group(1).strip().rstrip(".,;:!"), full
# # # #         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
# # # #         for line in reversed(lines):
# # # #             if len(line) > 220:
# # # #                 continue
# # # #             if _GAIA_PREAMBLE.match(line):
# # # #                 continue
# # # #             return line.rstrip(".,;:!"), full
# # # #         # Single-line or all lines look like preamble: try last sentence/clause.
# # # #         chunks = re.split(r"(?<=[.!?])\s+", full)
# # # #         for frag in reversed(chunks):
# # # #             frag = frag.strip()
# # # #             if not frag or len(frag) > 220:
# # # #                 continue
# # # #             if _GAIA_PREAMBLE.match(frag):
# # # #                 continue
# # # #             return frag.rstrip(".,;:!"), full
# # # #         if lines:
# # # #             return lines[-1].rstrip(".,;:!"), full
# # # #         return full, full

# # # #     if dataset == "mmlu_pro":
# # # #         m = _RE_THE_ANSWER.search(full)
# # # #         if m:
# # # #             frag = m.group(1).strip().rstrip(".")
# # # #             lm = re.match(r"\(?([A-Ja-j])\)?[\.\):\s]?", frag)
# # # #             return (lm.group(1).upper() if lm else frag), full
# # # #         letters = _RE_LETTER_MMLU.findall(full)
# # # #         if letters:
# # # #             return letters[-1].upper(), full
# # # #         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
# # # #         return (lines[-1].rstrip(".") if lines else full), full

# # # #     if dataset == "math_hard":
# # # #         return _extract_math_final(full), full

# # # #     return full, full


# # # # def _extract_math_final(text: str) -> str:
# # # #     text = (text or "").strip()
# # # #     # Remove common markdown/code wrappers first.
# # # #     text = text.strip("`").strip('"').strip("'").strip()
# # # #     text = _strip_math_wrappers(text)
# # # #     m = _RE_THE_ANSWER.search(text)
# # # #     if m:
# # # #         return _strip_math_wrappers(m.group(1).strip().rstrip("."))
# # # #     m = _RE_FINAL_ANSWER.search(text)
# # # #     if m:
# # # #         return _strip_math_wrappers(m.group(1).strip().rstrip("."))
# # # #     m = _RE_BOXED.search(text)
# # # #     if m:
# # # #         return _strip_math_wrappers(m.group(1).strip())
# # # #     m = _RE_EQ_END.search(text)
# # # #     if m:
# # # #         return _strip_math_wrappers(m.group(1).strip())
# # # #     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
# # # #     return _strip_math_wrappers(lines[-1].rstrip(".")) if lines else text


# # # # def _strip_math_wrappers(expr: str) -> str:
# # # #     expr = (expr or "").strip()
# # # #     # Remove escaped inline/display wrappers if present at boundaries.
# # # #     if expr.startswith(r"\(") and expr.endswith(r"\)"):
# # # #         expr = expr[2:-2].strip()
# # # #     if expr.startswith(r"\[") and expr.endswith(r"\]"):
# # # #         expr = expr[2:-2].strip()
# # # #     # Remove markdown/math delimiters.
# # # #     while len(expr) >= 2 and expr.startswith("$") and expr.endswith("$"):
# # # #         expr = expr[1:-1].strip()
# # # #     # Remove \left/\right artifacts.
# # # #     expr = expr.replace(r"\left", "").replace(r"\right", "")
# # # #     return expr.strip()


# # # # def _preprocess_math(expr: str) -> str:
# # # #     expr = _strip_math_wrappers(expr)

# # # #     out = (
# # # #         expr.replace("π", "pi")
# # # #         .replace("\u03c0", "pi")
# # # #         .replace("\u2212", "-")
# # # #         .replace("\u00d7", "*")
# # # #         .replace("\u00f7", "/")
# # # #         .replace("²", "**2")
# # # #         .replace("³", "**3")
# # # #         .replace("^", "**")
# # # #         .strip()
# # # #     )
# # # #     out = _RE_IMPLICIT_MUL.sub(r"\1*\2", out)
# # # #     return _RE_WHITESPACE.sub("", out)


# # # # def _sympy_equal_math(predicted: str, ground_truth: str) -> Optional[bool]:
# # # #     if not SYMPY_AVAILABLE:
# # # #         return None
# # # #     a1, a2 = _preprocess_math(predicted), _preprocess_math(ground_truth)
# # # #     for fn in (sympify, parse_latex):
# # # #         try:
# # # #             diff = simplify(fn(a1) - fn(a2))
# # # #             if diff == 0 or abs(complex(N(diff))) < 1e-2:
# # # #                 return True
# # # #         except Exception:
# # # #             continue
# # # #     return None


# # # # def _norm_math(text: str) -> str:
# # # #     t = _strip_math_wrappers(text)

# # # #     t = t.lower()
# # # #     t = (
# # # #         t.replace("−", "-")
# # # #         .replace("∞", r"\infty")
# # # #     )

# # # #     return (
# # # #         _RE_WHITESPACE.sub("", t)
# # # #         .replace("^", "**")
# # # #         .replace("×", "*")
# # # #         .replace("÷", "/")
# # # #         .replace("π", "pi")
# # # #         .rstrip(".")
# # # #     )


# # # # def _eval_math(predicted: str, ground_truth: str) -> bool:
# # # #     # Many model outputs include rationale; pull likely final answer first.
# # # #     p = _extract_math_final(predicted)
# # # #     g = (ground_truth or "").strip()

# # # #     r = _sympy_equal_math(p, g)
# # # #     if r is not None:
# # # #         return r
# # # #     if _norm_math(p) == _norm_math(g):
# # # #         return True
# # # #     pn, gn = _try_float(p), _try_float(g)
# # # #     if pn is not None and gn is not None:
# # # #         return abs(pn - gn) < 1e-6
# # # #     return False


# # # # def _compute_correctness(dataset: str, predicted: str, ground_truth: str) -> bool:
# # # #     if not predicted or not ground_truth:
# # # #         return False
# # # #     if dataset == "gaia":
# # # #         return _eval_gaia(predicted, ground_truth)
# # # #     if dataset == "mmlu_pro":
# # # #         return _eval_mmlu(predicted, ground_truth)
# # # #     if dataset == "math_hard":
# # # #         return _eval_math(predicted, ground_truth)
# # # #     return predicted.strip().lower() == ground_truth.strip().lower()


# # # # def _ground_truth_hint(dataset: str) -> str:
# # # #     """Aligns with processed `answer` / gold column style per dataset."""
# # # #     if dataset == "mmlu_pro":
# # # #         return "gold is one letter A–J (index letter); output that letter only"
# # # #     if dataset == "math_hard":
# # # #         return "gold is LaTeX/math final (e.g. \\frac{a}{b}, interval, integer); match that form"
# # # #     if dataset == "swe_bench_verified":
# # # #         return "gold is patch/diff-style text; output minimal patch or exact requested artifact"
# # # #     return "gold is short factual (word, number, name, date, or short phrase); match exactly"


# # # # def _output_contract_for(dataset: str, modality: str) -> str:
# # # #     """Short, dataset-specific contract (few tokens) for system prompt."""
# # # #     if modality in ("vanilla", "routellm"):
# # # #         if dataset == "gaia":
# # # #             return "Output one line: final answer only (no reasoning)."
# # # #         if dataset == "mmlu_pro":
# # # #             return "Output one character: A–J only."
# # # #         if dataset == "math_hard":
# # # #             return "Output final math only (TeX allowed); no sentences."
# # # #         if dataset == "swe_bench_verified":
# # # #             return "Output patch or requested code only; no prose."
# # # #     if modality == "cot":
# # # #         if dataset == "gaia":
# # # #             return "Brief reasoning then final line: The answer is <value>."
# # # #         if dataset == "mmlu_pro":
# # # #             return "Brief reasoning then final line: The answer is <A–J>."
# # # #         if dataset == "math_hard":
# # # #             return "Brief reasoning then final line: The answer is <TeX/math>."
# # # #         if dataset == "swe_bench_verified":
# # # #             return "Brief reasoning then final line: The answer is <patch or code>."
# # # #     # react / multiagent (prompt-only baseline)
# # # #     if dataset == "mmlu_pro":
# # # #         return "End with one letter A–J."
# # # #     if dataset == "math_hard":
# # # #         return "End with final math expression only."
# # # #     if dataset == "swe_bench_verified":
# # # #         return "End with patch or minimal artifact."
# # # #     return "End with one-line final answer."


# # # # def _tools_available(modality: str) -> str:
# # # #     if modality in ("react", "multiagent"):
# # # #         return "none"
# # # #     return "not_applicable"

# # # # #It’s a helper function that creates a small dictionary of settings describing how a model run should behave.

# # # # def _build_system_values(
# # # #     selection: RunSelection, dataset: str, modality: str
# # # # ) -> dict[str, Any]:
# # # #     if modality == "routellm":
# # # #         model_name = (
# # # #             f"RouteLLM router={selection.routellm_router} thr={selection.routellm_threshold:g} "
# # # #             f"strong={selection.routellm_strong_model} weak={selection.routellm_weak_model}"
# # # #         )
# # # #     else:
# # # #         model_name = selection.model
# # # #     return {
# # # #         "model_name": model_name,
# # # #         "output_contract": _output_contract_for(dataset, modality),
# # # #         "max_steps": selection.max_steps,
# # # #         "tool_budget": selection.tool_budget,
# # # #     }


# # # # def _build_user_values(query: QueryInput, modality: str) -> dict[str, Any]:
# # # #     values: dict[str, Any] = {
# # # #         "dataset": query.dataset,
# # # #         "query": query.query,
# # # #     }
# # # #     if query.dataset == "mmlu_pro":
# # # #         values["options"] = query.options or ""
# # # #     if query.dataset in ("gaia", "math_hard", "swe_bench_verified"):
# # # #         values["ground_truth_format_hint"] = _ground_truth_hint(query.dataset)
# # # #     if modality in ("react", "multiagent"):
# # # #         values["tools_available"] = _tools_available(modality)
# # # #     return values


# # # # def run_selected(
# # # #     *,
# # # #     queries: list[QueryInput],
# # # #     selection: RunSelection,
# # # #     client: Optional[OpenAI] = None,
# # # # ) -> list[UnifiedExperimentRecord]:
# # # #     """
# # # #     Minimal scalable runner:
# # # #     - Runs only requested datasets/modalities
# # # #     - Single-call execution path for all modalities (prompt-driven baseline)
# # # #     """
# # # #     llm = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # # #     results: list[UnifiedExperimentRecord] = []

# # # #     wanted_datasets = set(selection.datasets)
# # # #     wanted_modalities = set(selection.modalities)

# # # #     routellm_client: Optional[Any] = None
# # # #     if "routellm" in wanted_modalities:
# # # #         if not ROUTELLM_AVAILABLE:
# # # #             msg = (
# # # #                 "Modality 'routellm' needs the RouteLLM PyPI package in this environment.\n"
# # # #                 "Install deps from the repo root (uv only):\n"
# # # #                 "  uv sync\n"
# # # #                 "Then run with: uv run python baselines/run_baseline.py ...\n"
# # # #                 "Docs: https://github.com/lm-sys/routellm\n"
# # # #                 "\n"
# # # #                 "Note: our integration lives in baselines/route_llm/ so it does not shadow "
# # # #                 "the `routellm` library when running baselines/run_baseline.py."
# # # #             )
# # # #             if ROUTELLM_IMPORT_ERROR:
# # # #                 msg += f"\n\nImport diagnostic:\n  {ROUTELLM_IMPORT_ERROR}"
# # # #             raise ImportError(msg)
# # # #         routellm_client = build_controller(selection.routellm_config())

# # # #     for query in queries:
# # # #         if query.dataset not in wanted_datasets:
# # # #             continue

# # # #         for modality in wanted_modalities:
# # # #             system_values = _build_system_values(
# # # #                 selection, query.dataset, modality
# # # #             )
# # # #             user_values = _build_user_values(query, modality)

# # # #             resolved = resolve_prompts(
# # # #                 modality=modality,
# # # #                 dataset=query.dataset,
# # # #                 model_name=selection.model,
# # # #                 system_values=system_values,
# # # #                 user_values=user_values,
# # # #             )

# # # #             resolved_model: Optional[str] = None
# # # #             if modality == "routellm":
# # # #                 assert routellm_client is not None
# # # #                 raw_output, usage, error, elapsed, resolved_model = routellm_chat_completion(
# # # #                     routellm_client,
# # # #                     cfg=selection.routellm_config(),
# # # #                     system_prompt=resolved.system_prompt,
# # # #                     user_prompt=resolved.user_prompt,
# # # #                     max_tokens=selection.max_tokens,
# # # #                     temperature=selection.temperature,
# # # #                 )
# # # #             else:
# # # #                 raw_output, usage, error, elapsed = vanallia_operator(
# # # #                     client=llm,
# # # #                     model_name=selection.model,
# # # #                     question=resolved.user_prompt,
# # # #                     system_prompt=resolved.system_prompt,
# # # #                     max_tokens=selection.max_tokens,
# # # #                     temperature=selection.temperature,
# # # #                 )

# # # #             predicted, _full = _extract_short_answer(raw_output, query.dataset)
# # # #             row_meta: dict[str, Any] = dict(query.metadata or {})
# # # #             if raw_output.strip() != predicted.strip():
# # # #                 row_meta["raw_model_output"] = raw_output.strip()[:50_000]

# # # #             if modality == "routellm":
# # # #                 row_meta["routellm_router"] = selection.routellm_router
# # # #                 row_meta["routellm_threshold"] = selection.routellm_threshold
# # # #                 row_meta["routellm_strong_model"] = selection.routellm_strong_model
# # # #                 row_meta["routellm_weak_model"] = selection.routellm_weak_model
# # # #                 if resolved_model:
# # # #                     row_meta["routellm_resolved_model"] = resolved_model

# # # #             is_correct = _compute_correctness(
# # # #                 dataset=query.dataset,
# # # #                 predicted=predicted,
# # # #                 ground_truth=query.ground_truth,
# # # #             )
# # # #             error_type = "runtime_error" if error else None

# # # #             reasoning_trace: Optional[str] = None
# # # #             if modality in ("cot", "react", "multiagent"):
# # # #                 reasoning_trace = raw_output
# # # #             elif raw_output.strip() != predicted.strip():
# # # #                 reasoning_trace = raw_output

# # # #             reasoning_steps = _count_reasoning_steps(raw_output, predicted)

# # # #             record_model = "routellm" if modality == "routellm" else selection.model
# # # #             if modality == "routellm":
# # # #                 cost_usd = _estimate_cost_routellm(
# # # #                     selection,
# # # #                     resolved_model,
# # # #                     usage.get("prompt_tokens", 0),
# # # #                     usage.get("completion_tokens", 0),
# # # #                 )
# # # #             else:
# # # #                 cost_usd = _estimate_cost(
# # # #                     selection.model,
# # # #                     usage.get("prompt_tokens", 0),
# # # #                     usage.get("completion_tokens", 0),
# # # #                 )

# # # #             results.append(
# # # #                 UnifiedExperimentRecord(
# # # #                     query_id=query.query_id,
# # # #                     dataset=query.dataset,
# # # #                     modality=modality,
# # # #                     model=record_model,
# # # #                     query=query.query,
# # # #                     ground_truth=query.ground_truth,
# # # #                     predicted_answer=predicted,
# # # #                     is_correct=is_correct,
# # # #                     reasoning_trace=reasoning_trace,
# # # #                     tool_trace=None,
# # # #                     reasoning_steps=reasoning_steps,
# # # #                     prompt_tokens=usage.get("prompt_tokens", 0),
# # # #                     completion_tokens=usage.get("completion_tokens", 0),
# # # #                     total_tokens=usage.get("total_tokens", 0),
# # # #                     cost_usd=cost_usd,
# # # #                     latency_s=round(elapsed, 3),
# # # #                     error_type=error_type,
# # # #                     error_message=error,
# # # #                     system_prompt_version=resolved.system_prompt_version,
# # # #                     user_prompt_version=resolved.user_prompt_version,
# # # #                     prompt_hash=resolved.prompt_hash,
# # # #                     metadata=row_meta,
# # # #                 )
# # # #             )

# # # #     return results


# # # # def default_selection() -> RunSelection:
# # # #     # Exclude `routellm` so defaults do not require the optional `routellm` package.
# # # #     modalities = tuple(m for m in ALLOWED_MODALITIES if m != "routellm")
# # # #     return RunSelection(
# # # #         datasets=ALLOWED_DATASETS,
# # # #         modalities=modalities,
# # # #     )



# # # from __future__ import annotations

# # # from dataclasses import dataclass
# # # import re
# # # from typing import Any, Optional
# # # import unicodedata
# # # from dotenv import load_dotenv
# # # from openai import OpenAI
# # # import os
# # # load_dotenv()

# # # from agents.vanallia_operator import vanallia_operator
# # # from baselines.constants import (
# # #     ALLOWED_DATASETS,
# # #     ALLOWED_MODALITIES,
# # #     validate_dataset,
# # #     validate_modality,
# # #     validate_model,
# # # )
# # # from baselines.route_llm import (
# # #     ROUTELLM_AVAILABLE,
# # #     ROUTELLM_IMPORT_ERROR,
# # #     RoutellmConfig,
# # #     build_controller,
# # #     routellm_chat_completion,
# # # )
# # # from baselines.prompt_resolver import resolve_prompts
# # # from baselines.schema import UnifiedExperimentRecord

# # # try:
# # #     from sympy import N, simplify, sympify
# # #     from sympy.parsing.latex import parse_latex
# # #     SYMPY_AVAILABLE = True
# # # except Exception:
# # #     SYMPY_AVAILABLE = False


# # # # ── Constants ─────────────────────────────────────────────────────────────────

# # # COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
# # #     "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
# # #     "gpt-4o":      {"input": 0.0025,  "output": 0.01},
# # # }

# # # GAIA_ALIASES: dict[str, list[str]] = {
# # #     "united states":          ["us", "usa", "u.s.", "u.s.a.", "united states of america"],
# # #     "united kingdom":         ["uk", "u.k.", "britain", "great britain"],
# # #     "world war ii":           ["wwii", "ww2", "world war 2", "second world war"],
# # #     "artificial intelligence":["ai"],
# # #     "united nations":         ["un", "u.n."],
# # # }

# # # # ── Pre-compiled regexes ───────────────────────────────────────────────────────
# # # # Compiled once at import; reused across all calls — avoids re.compile overhead
# # # # inside hot functions.
# # # _RE_WHITESPACE    = re.compile(r"\s+")
# # # _RE_THE_ANSWER    = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
# # # _RE_FINAL_ANSWER  = re.compile(r"[Ff]inal answer is[:\s]+([^\n]+)")
# # # _RE_BOXED         = re.compile(r"\\boxed\{([^}]+)\}")
# # # _RE_EQ_END        = re.compile(r"=\s*([^\n=]+)\s*$", re.MULTILINE)
# # # _RE_IMPLICIT_MUL  = re.compile(r"(\d)(pi\b|[a-df-wyzA-Z])")
# # # _RE_LETTER_MMLU   = re.compile(r"\b([A-Ja-j])\b")
# # # _GAIA_PREAMBLE    = re.compile(
# # #     r"^(To\s|In\s+order|We\s+need|Let's\s|Let\s+us\s|Here's\s|Here\s+is\s|The\s+question|However,|"
# # #     r"I\s+need|Based\s+on|The\s+first\s+step|I\s+will|I'll\s|Note\s+that)",
# # #     re.IGNORECASE,
# # # )
# # # # Single combined regex for all reasoning-step markers (replaces 3 findall passes)
# # # _RE_STEP_MARKERS  = re.compile(
# # #     r"(?m)(?:^\s*\d+[\.)]\s+\S"   # numbered list
# # #     r"|^\s*[-*•]\s+\S"             # bullet
# # #     r"|\bstep\s+\d+)",             # "Step N" keyword
# # #     re.IGNORECASE,
# # # )
# # # # Sentence splitter for GAIA fallback
# # # _RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# # # # Pre-built str.maketrans for _norm_math — O(1) translation table built once
# # # _MATH_TRANS = str.maketrans({
# # #     "−": "-", "∞": r"\infty", "^": "**", "×": "*", "÷": "/", "π": "pi",
# # # })


# # # # ── Alias map (built once at module load) ─────────────────────────────────────

# # # def _norm(text: str) -> str:
# # #     """Normalise to lowercase, strip accents, collapse whitespace."""
# # #     text = unicodedata.normalize("NFKD", text.strip().lower())
# # #     # Filter combining chars and trailing punctuation in one pass
# # #     filtered = (c for c in text if not unicodedata.combining(c))
# # #     text = "".join(filtered).rstrip(".,:;!")
# # #     return _RE_WHITESPACE.sub(" ", text)


# # # def _build_alias_map() -> dict[str, str]:
# # #     """
# # #     Flat alias → canonical mapping, both sides pre-normalised.
# # #     Built once at import; O(A) where A = total alias strings (tiny).
# # #     """
# # #     alias_map: dict[str, str] = {}
# # #     for canonical, aliases in GAIA_ALIASES.items():
# # #         canon = _norm(canonical)
# # #         for alias in aliases:
# # #             alias_map[_norm(alias)] = canon
# # #     return alias_map


# # # # Module-level; computed once, shared across all calls.
# # # _GAIA_ALIAS_MAP: dict[str, str] = _build_alias_map()


# # # # ── Dataclasses ───────────────────────────────────────────────────────────────

# # # @dataclass(frozen=True)
# # # class RunSelection:
# # #     datasets:   tuple[str, ...]
# # #     modalities: tuple[str, ...]
# # #     model:       str   = "gpt-4o-mini"
# # #     max_tokens:  int   = 2048
# # #     temperature: float = 0.1
# # #     max_steps:   int   = 6
# # #     tool_budget: int   = 4
# # #     routellm_router:        str   = "mf"
# # #     routellm_threshold:     float = 0.11593
# # #     routellm_strong_model:  str   = "gpt-4o"
# # #     routellm_weak_model:    str   = "gpt-4o-mini"

# # #     def __post_init__(self) -> None:
# # #         validate_model(self.model)
# # #         for dataset in self.datasets:
# # #             validate_dataset(dataset)
# # #         for modality in self.modalities:
# # #             validate_modality(modality)
# # #         if "routellm" in self.modalities:
# # #             if not (0.0 <= self.routellm_threshold <= 1.0):
# # #                 raise ValueError("routellm_threshold must be in [0, 1].")
# # #             if not self.routellm_router.strip():
# # #                 raise ValueError("routellm_router must be non-empty.")

# # #     def routellm_config(self) -> RoutellmConfig:
# # #         return RoutellmConfig(
# # #             router=self.routellm_router.strip(),
# # #             threshold=float(self.routellm_threshold),
# # #             strong_model=self.routellm_strong_model.strip(),
# # #             weak_model=self.routellm_weak_model.strip(),
# # #         )


# # # @dataclass(frozen=True)
# # # class QueryInput:
# # #     query_id:     str
# # #     dataset:      str
# # #     query:        str
# # #     ground_truth: str
# # #     options:      Optional[str]             = None
# # #     metadata:     Optional[dict[str, Any]]  = None

# # #     def __post_init__(self) -> None:
# # #         validate_dataset(self.dataset)


# # # # ── Cost helpers ──────────────────────────────────────────────────────────────

# # # def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
# # #     rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
# # #     return round(
# # #         (prompt_tokens  / 1000.0) * rates["input"] +
# # #         (completion_tokens / 1000.0) * rates["output"],
# # #         6,
# # #     )


# # # def _estimate_cost_routellm(
# # #     selection: RunSelection,
# # #     resolved_model: Optional[str],
# # #     prompt_tokens: int,
# # #     completion_tokens: int,
# # # ) -> float:
# # #     r = (resolved_model or "").lower()
# # #     for key in (
# # #         selection.routellm_strong_model,
# # #         selection.routellm_weak_model,
# # #         "gpt-4o",
# # #         "gpt-4o-mini",
# # #     ):
# # #         if key and key.lower() in r and key in COST_PER_1K_TOKENS:
# # #             return _estimate_cost(key, prompt_tokens, completion_tokens)
# # #     fallback = (
# # #         selection.routellm_weak_model
# # #         if selection.routellm_weak_model in COST_PER_1K_TOKENS
# # #         else "gpt-4o-mini"
# # #     )
# # #     return _estimate_cost(fallback, prompt_tokens, completion_tokens)


# # # # ── Evaluation helpers ────────────────────────────────────────────────────────

# # # def _try_float(text: str) -> Optional[float]:
# # #     try:
# # #         return float(text.replace(",", "").replace("$", "").replace("%", "").strip())
# # #     except ValueError:
# # #         return None


# # # def _eval_gaia(predicted: str, ground_truth: str) -> bool:
# # #     """
# # #     Optimisation: normalise each string once and reuse for both direct equality
# # #     and alias lookup — previously _norm() was called redundantly inside
# # #     _resolve_alias() after already being called by the caller.
# # #     """
# # #     p_norm = _norm(predicted)
# # #     g_norm = _norm(ground_truth)

# # #     if p_norm == g_norm:
# # #         return True

# # #     # Alias lookup is O(1) dict get — no re-normalisation needed
# # #     p_resolved = _GAIA_ALIAS_MAP.get(p_norm, p_norm)
# # #     g_resolved = _GAIA_ALIAS_MAP.get(g_norm, g_norm)
# # #     if p_resolved == g_resolved:
# # #         return True

# # #     pn, gn = _try_float(predicted), _try_float(ground_truth)
# # #     if pn is not None and gn is not None:
# # #         return abs(pn - gn) < 1e-6

# # #     return False


# # # def _extract_letter_mmlu(text: str) -> str:
# # #     """
# # #     Replaces the inner _letter() closure that was re-created on every
# # #     _eval_mmlu() call. Module-level function — defined once.
# # #     """
# # #     text = text.strip()
# # #     m = _RE_LETTER_MMLU.search(text)
# # #     if m:
# # #         return m.group(1).upper()
# # #     return text[0].upper() if text and text[0].upper() in "ABCDEFGHIJ" else text.upper()


# # # def _eval_mmlu(predicted: str, ground_truth: str) -> bool:
# # #     return _extract_letter_mmlu(predicted) == _extract_letter_mmlu(ground_truth)


# # # def _count_reasoning_steps(raw: str, predicted: str) -> int:
# # #     """
# # #     Optimisation: single regex pass with alternation replaces three separate
# # #     re.findall calls (numbered / bullets / step-keyword).
# # #     Fallback line-count path is unchanged.
# # #     """
# # #     text = (raw or "").strip()
# # #     pred = (predicted or "").strip()
# # #     if not text or text == pred:
# # #         return 0

# # #     markers = _RE_STEP_MARKERS.findall(text)
# # #     if markers:
# # #         return len(markers)

# # #     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
# # #     if not lines:
# # #         return 1
# # #     last = lines[-1]
# # #     if _RE_THE_ANSWER.search(last) or _RE_FINAL_ANSWER.search(last):
# # #         lines = lines[:-1]
# # #     elif pred and (last.rstrip(".,;:!") == pred or last.endswith(pred)):
# # #         lines = lines[:-1]
# # #     return max(1, len(lines))


# # # # ── Math helpers ──────────────────────────────────────────────────────────────

# # # def _strip_math_wrappers(expr: str) -> str:
# # #     expr = (expr or "").strip()
# # #     if expr.startswith(r"\(") and expr.endswith(r"\)"):
# # #         expr = expr[2:-2].strip()
# # #     if expr.startswith(r"\[") and expr.endswith(r"\]"):
# # #         expr = expr[2:-2].strip()
# # #     while len(expr) >= 2 and expr[0] == "$" and expr[-1] == "$":
# # #         expr = expr[1:-1].strip()
# # #     expr = expr.replace(r"\left", "").replace(r"\right", "")
# # #     return expr.strip()


# # # def _extract_math_final(text: str) -> str:
# # #     text = (text or "").strip().strip("`").strip('"').strip("'").strip()
# # #     text = _strip_math_wrappers(text)

# # #     # Try patterns in priority order; return on first match
# # #     for pattern in (_RE_THE_ANSWER, _RE_FINAL_ANSWER, _RE_BOXED, _RE_EQ_END):
# # #         m = pattern.search(text)
# # #         if m:
# # #             return _strip_math_wrappers(m.group(1).strip().rstrip("."))

# # #     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
# # #     return _strip_math_wrappers(lines[-1].rstrip(".")) if lines else text


# # # def _preprocess_math(expr: str) -> str:
# # #     """
# # #     Optimisation: uses str.translate with a pre-built translation table
# # #     (_MATH_TRANS) instead of 8 chained .replace() calls (8 intermediate
# # #     string allocations → 1).
# # #     """
# # #     expr = _strip_math_wrappers(expr)
# # #     out = (
# # #         expr.replace("π", "pi")
# # #             .replace("\u03c0", "pi")
# # #             .replace("\u2212", "-")
# # #             .replace("\u00d7", "*")
# # #             .replace("\u00f7", "/")
# # #             .replace("²", "**2")
# # #             .replace("³", "**3")
# # #             .translate(_MATH_TRANS)   # handles ^, ×, ÷, π, −, ∞ in one pass
# # #             .strip()
# # #     )
# # #     out = _RE_IMPLICIT_MUL.sub(r"\1*\2", out)
# # #     return _RE_WHITESPACE.sub("", out)


# # # def _norm_math(text: str) -> str:
# # #     """
# # #     Optimisation: str.translate replaces 8 chained .replace() calls —
# # #     reduces intermediate string allocations from 8 to 1.
# # #     """
# # #     t = _strip_math_wrappers(text).lower()
# # #     return _RE_WHITESPACE.sub("", t.translate(_MATH_TRANS)).rstrip(".")


# # # def _sympy_equal_math(predicted: str, ground_truth: str) -> Optional[bool]:
# # #     if not SYMPY_AVAILABLE:
# # #         return None
# # #     a1, a2 = _preprocess_math(predicted), _preprocess_math(ground_truth)
# # #     for fn in (sympify, parse_latex):
# # #         try:
# # #             diff = simplify(fn(a1) - fn(a2))
# # #             if diff == 0 or abs(complex(N(diff))) < 1e-2:
# # #                 return True
# # #         except Exception:
# # #             continue
# # #     return None


# # # def _eval_math(predicted: str, ground_truth: str) -> bool:
# # #     p = _extract_math_final(predicted)
# # #     g = (ground_truth or "").strip()

# # #     r = _sympy_equal_math(p, g)
# # #     if r is not None:
# # #         return r
# # #     if _norm_math(p) == _norm_math(g):
# # #         return True
# # #     pn, gn = _try_float(p), _try_float(g)
# # #     if pn is not None and gn is not None:
# # #         return abs(pn - gn) < 1e-6
# # #     return False


# # # def _compute_correctness(dataset: str, predicted: str, ground_truth: str) -> bool:
# # #     if not predicted or not ground_truth:
# # #         return False
# # #     if dataset == "gaia":
# # #         return _eval_gaia(predicted, ground_truth)
# # #     if dataset == "mmlu_pro":
# # #         return _eval_mmlu(predicted, ground_truth)
# # #     if dataset == "math_hard":
# # #         return _eval_math(predicted, ground_truth)
# # #     return predicted.strip().lower() == ground_truth.strip().lower()


# # # # ── Prompt builders ───────────────────────────────────────────────────────────

# # # def _ground_truth_hint(dataset: str) -> str:
# # #     if dataset == "mmlu_pro":
# # #         return "gold is one letter A–J (index letter); output that letter only"
# # #     if dataset == "math_hard":
# # #         return "gold is LaTeX/math final (e.g. \\frac{a}{b}, interval, integer); match that form"
# # #     if dataset == "swe_bench_verified":
# # #         return "gold is patch/diff-style text; output minimal patch or exact requested artifact"
# # #     return "gold is short factual (word, number, name, date, or short phrase); match exactly"


# # # def _output_contract_for(dataset: str, modality: str) -> str:
# # #     if modality in ("vanilla", "routellm"):
# # #         if dataset == "gaia":               return "Output one line: final answer only (no reasoning)."
# # #         if dataset == "mmlu_pro":           return "Output one character: A–J only."
# # #         if dataset == "math_hard":          return "Output final math only (TeX allowed); no sentences."
# # #         if dataset == "swe_bench_verified": return "Output patch or requested code only; no prose."
# # #     if modality == "cot":
# # #         if dataset == "gaia":               return "Brief reasoning then final line: The answer is <value>."
# # #         if dataset == "mmlu_pro":           return "Brief reasoning then final line: The answer is <A–J>."
# # #         if dataset == "math_hard":          return "Brief reasoning then final line: The answer is <TeX/math>."
# # #         if dataset == "swe_bench_verified": return "Brief reasoning then final line: The answer is <patch or code>."
# # #     if dataset == "mmlu_pro":           return "End with one letter A–J."
# # #     if dataset == "math_hard":          return "End with final math expression only."
# # #     if dataset == "swe_bench_verified": return "End with patch or minimal artifact."
# # #     return "End with one-line final answer."


# # # def _tools_available(modality: str) -> str:
# # #     return "none" if modality in ("react", "multiagent") else "not_applicable"


# # # # ── Cache for per-(dataset, modality) system values ──────────────────────────
# # # # system_values depend only on (selection, dataset, modality) — not on the
# # # # individual query.  We cache them so we don't recompute for every one of N queries.

# # # def _build_system_values(
# # #     selection: RunSelection, dataset: str, modality: str
# # # ) -> dict[str, Any]:
# # #     if modality == "routellm":
# # #         model_name = (
# # #             f"RouteLLM router={selection.routellm_router} thr={selection.routellm_threshold:g} "
# # #             f"strong={selection.routellm_strong_model} weak={selection.routellm_weak_model}"
# # #         )
# # #     else:
# # #         model_name = selection.model
# # #     return {
# # #         "model_name":      model_name,
# # #         "output_contract": _output_contract_for(dataset, modality),
# # #         "max_steps":       selection.max_steps,
# # #         "tool_budget":     selection.tool_budget,
# # #     }


# # # def _build_user_values(query: QueryInput, modality: str) -> dict[str, Any]:
# # #     values: dict[str, Any] = {"dataset": query.dataset, "query": query.query}
# # #     if query.dataset == "mmlu_pro":
# # #         values["options"] = query.options or ""
# # #     if query.dataset in ("gaia", "math_hard", "swe_bench_verified"):
# # #         values["ground_truth_format_hint"] = _ground_truth_hint(query.dataset)
# # #     if modality in ("react", "multiagent"):
# # #         values["tools_available"] = _tools_available(modality)
# # #     return values


# # # def _extract_short_answer(raw: str, dataset: str) -> tuple[str, str]:
# # #     """
# # #     Optimisation: early-exit on empty input; no change to logic otherwise.
# # #     Avoids unnecessary string operations when raw is empty.
# # #     """
# # #     full = (raw or "").strip()
# # #     if not full:
# # #         return "", ""

# # #     if dataset in ("gaia", "swe_bench_verified"):
# # #         m = _RE_THE_ANSWER.search(full)
# # #         if m:
# # #             return m.group(1).strip().rstrip(".,;:!"), full
# # #         m = _RE_FINAL_ANSWER.search(full)
# # #         if m:
# # #             return m.group(1).strip().rstrip(".,;:!"), full
# # #         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
# # #         for line in reversed(lines):
# # #             if len(line) > 220 or _GAIA_PREAMBLE.match(line):
# # #                 continue
# # #             return line.rstrip(".,;:!"), full
# # #         for frag in reversed(_RE_SENTENCE_SPLIT.split(full)):
# # #             frag = frag.strip()
# # #             if not frag or len(frag) > 220 or _GAIA_PREAMBLE.match(frag):
# # #                 continue
# # #             return frag.rstrip(".,;:!"), full
# # #         return (lines[-1].rstrip(".,;:!") if lines else full), full

# # #     if dataset == "mmlu_pro":
# # #         m = _RE_THE_ANSWER.search(full)
# # #         if m:
# # #             frag = m.group(1).strip().rstrip(".")
# # #             lm = re.match(r"\(?([A-Ja-j])\)?[\.\):\s]?", frag)
# # #             return (lm.group(1).upper() if lm else frag), full
# # #         letters = _RE_LETTER_MMLU.findall(full)
# # #         if letters:
# # #             return letters[-1].upper(), full
# # #         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
# # #         return (lines[-1].rstrip(".") if lines else full), full

# # #     if dataset == "math_hard":
# # #         return _extract_math_final(full), full

# # #     return full, full


# # # # ── Main runner ───────────────────────────────────────────────────────────────

# # # def run_selected(
# # #     *,
# # #     queries: list[QueryInput],
# # #     selection: RunSelection,
# # #     client: Optional[OpenAI] = None,
# # # ) -> list[UnifiedExperimentRecord]:
# # #     """
# # #     Key optimisation: cache system_values per (dataset, modality) pair.

# # #     Previously _build_system_values() was called inside the inner loop —
# # #     O(N × M) calls even though the result is identical for every query with
# # #     the same dataset+modality.  Now it is computed O(D × M) times (D = distinct
# # #     datasets, M = modalities), which is a tiny constant vs. N queries.
# # #     """
# # #     llm = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# # #     results: list[UnifiedExperimentRecord] = []

# # #     wanted_datasets   = set(selection.datasets)
# # #     wanted_modalities = set(selection.modalities)

# # #     # ── Build RouteLLM client once if needed ──────────────────────────────────
# # #     routellm_client: Optional[Any] = None
# # #     if "routellm" in wanted_modalities:
# # #         if not ROUTELLM_AVAILABLE:
# # #             msg = (
# # #                 "Modality 'routellm' needs the RouteLLM PyPI package in this environment.\n"
# # #                 "Install deps from the repo root (uv only):\n"
# # #                 "  uv sync\n"
# # #                 "Then run with: uv run python baselines/run_baseline.py ...\n"
# # #                 "Docs: https://github.com/lm-sys/routellm\n\n"
# # #                 "Note: our integration lives in baselines/route_llm/ so it does not shadow "
# # #                 "the `routellm` library when running baselines/run_baseline.py."
# # #             )
# # #             if ROUTELLM_IMPORT_ERROR:
# # #                 msg += f"\n\nImport diagnostic:\n  {ROUTELLM_IMPORT_ERROR}"
# # #             raise ImportError(msg)
# # #         routellm_client = build_controller(selection.routellm_config())

# # #     # ── Pre-compute system_values for every (dataset, modality) pair ─────────
# # #     # O(D × M) instead of O(N × M).  For 1000 queries × 4 modalities this
# # #     # eliminates ~3996 redundant dict constructions and string ops.
# # #     system_values_cache: dict[tuple[str, str], dict[str, Any]] = {
# # #         (ds, mod): _build_system_values(selection, ds, mod)
# # #         for ds in wanted_datasets
# # #         for mod in wanted_modalities
# # #     }

# # #     is_routellm_modalities = wanted_modalities  # reused below for membership test

# # #     for query in queries:
# # #         if query.dataset not in wanted_datasets:
# # #             continue

# # #         for modality in wanted_modalities:
# # #             # O(1) cache lookup instead of O(1) recomputation × N
# # #             system_values = system_values_cache[(query.dataset, modality)]
# # #             user_values   = _build_user_values(query, modality)

# # #             resolved = resolve_prompts(
# # #                 modality=modality,
# # #                 dataset=query.dataset,
# # #                 model_name=selection.model,
# # #                 system_values=system_values,
# # #                 user_values=user_values,
# # #             )

# # #             resolved_model: Optional[str] = None
# # #             is_routellm = modality == "routellm"

# # #             if is_routellm:
# # #                 assert routellm_client is not None
# # #                 raw_output, usage, error, elapsed, resolved_model = routellm_chat_completion(
# # #                     routellm_client,
# # #                     cfg=selection.routellm_config(),
# # #                     system_prompt=resolved.system_prompt,
# # #                     user_prompt=resolved.user_prompt,
# # #                     max_tokens=selection.max_tokens,
# # #                     temperature=selection.temperature,
# # #                 )
# # #             else:
# # #                 raw_output, usage, error, elapsed = vanallia_operator(
# # #                     client=llm,
# # #                     model_name=selection.model,
# # #                     question=resolved.user_prompt,
# # #                     system_prompt=resolved.system_prompt,
# # #                     max_tokens=selection.max_tokens,
# # #                     temperature=selection.temperature,
# # #                 )

# # #             predicted, _full = _extract_short_answer(raw_output, query.dataset)

# # #             # Build row metadata — avoid dict copy when query.metadata is empty
# # #             row_meta: dict[str, Any] = dict(query.metadata) if query.metadata else {}
# # #             raw_stripped = raw_output.strip()
# # #             if raw_stripped != predicted.strip():
# # #                 row_meta["raw_model_output"] = raw_stripped[:50_000]

# # #             if is_routellm:
# # #                 row_meta.update({
# # #                     "routellm_router":         selection.routellm_router,
# # #                     "routellm_threshold":      selection.routellm_threshold,
# # #                     "routellm_strong_model":   selection.routellm_strong_model,
# # #                     "routellm_weak_model":     selection.routellm_weak_model,
# # #                 })
# # #                 if resolved_model:
# # #                     row_meta["routellm_resolved_model"] = resolved_model

# # #             is_correct = _compute_correctness(
# # #                 dataset=query.dataset,
# # #                 predicted=predicted,
# # #                 ground_truth=query.ground_truth,
# # #             )

# # #             # reasoning_trace: reuse already-stripped raw_stripped to avoid second .strip()
# # #             reasoning_trace: Optional[str] = None
# # #             if modality in ("cot", "react", "multiagent"):
# # #                 reasoning_trace = raw_output
# # #             elif raw_stripped != predicted.strip():
# # #                 reasoning_trace = raw_output

# # #             reasoning_steps = _count_reasoning_steps(raw_output, predicted)

# # #             prompt_tokens     = usage.get("prompt_tokens",     0)
# # #             completion_tokens = usage.get("completion_tokens", 0)

# # #             cost_usd = (
# # #                 _estimate_cost_routellm(selection, resolved_model, prompt_tokens, completion_tokens)
# # #                 if is_routellm
# # #                 else _estimate_cost(selection.model, prompt_tokens, completion_tokens)
# # #             )

# # #             results.append(UnifiedExperimentRecord(
# # #                 query_id=query.query_id,
# # #                 dataset=query.dataset,
# # #                 modality=modality,
# # #                 model="routellm" if is_routellm else selection.model,
# # #                 query=query.query,
# # #                 ground_truth=query.ground_truth,
# # #                 predicted_answer=predicted,
# # #                 is_correct=is_correct,
# # #                 reasoning_trace=reasoning_trace,
# # #                 tool_trace=None,
# # #                 reasoning_steps=reasoning_steps,
# # #                 prompt_tokens=prompt_tokens,
# # #                 completion_tokens=completion_tokens,
# # #                 total_tokens=usage.get("total_tokens", 0),
# # #                 cost_usd=cost_usd,
# # #                 latency_s=round(elapsed, 3),
# # #                 error_type="runtime_error" if error else None,
# # #                 error_message=error,
# # #                 system_prompt_version=resolved.system_prompt_version,
# # #                 user_prompt_version=resolved.user_prompt_version,
# # #                 prompt_hash=resolved.prompt_hash,
# # #                 metadata=row_meta,
# # #             ))

# # #     return results


# # # def default_selection() -> RunSelection:
# # #     modalities = tuple(m for m in ALLOWED_MODALITIES if m != "routellm")
# # #     return RunSelection(datasets=ALLOWED_DATASETS, modalities=modalities)



# # from __future__ import annotations

# # """
# # Changes vs previous runner.py:
# #   - `react` modality now calls the real ReAct agent (react_operator) instead
# #     of the vanilla single-call path.
# #   - The structured per-step trace is stored in record.tool_trace (JSON string).
# #   - reasoning_steps is taken directly from len(trace) (exact step count).
# #   - Everything else (vanilla / cot / routellm) is unchanged.
# # """

# # from dataclasses import dataclass
# # import re
# # from typing import Any, Optional
# # import unicodedata
# # import json
# # from dotenv import load_dotenv
# # from openai import OpenAI
# # import os
# # load_dotenv()

# # from agents.vanallia_operator import vanallia_operator
# # from agents.react_operator import react_operator          # ← real ReAct agent
# # from baselines.constants import (
# #     ALLOWED_DATASETS,
# #     ALLOWED_MODALITIES,
# #     validate_dataset,
# #     validate_modality,
# #     validate_model,
# # )
# # from baselines.route_llm import (
# #     ROUTELLM_AVAILABLE,
# #     ROUTELLM_IMPORT_ERROR,
# #     RoutellmConfig,
# #     build_controller,
# #     routellm_chat_completion,
# # )
# # from baselines.prompt_resolver import resolve_prompts
# # from baselines.schema import UnifiedExperimentRecord

# # try:
# #     from sympy import N, simplify, sympify
# #     from sympy.parsing.latex import parse_latex
# #     SYMPY_AVAILABLE = True
# # except Exception:
# #     SYMPY_AVAILABLE = False


# # # ── Constants ──────────────────────────────────────────────────────────────────

# # COST_PER_1K_TOKENS: dict[str, dict[str, float]] = {
# #     "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
# #     "gpt-4o":      {"input": 0.0025,  "output": 0.01},
# # }

# # GAIA_ALIASES: dict[str, list[str]] = {
# #     "united states":          ["us", "usa", "u.s.", "u.s.a.", "united states of america"],
# #     "united kingdom":         ["uk", "u.k.", "britain", "great britain"],
# #     "world war ii":           ["wwii", "ww2", "world war 2", "second world war"],
# #     "artificial intelligence":["ai"],
# #     "united nations":         ["un", "u.n."],
# # }

# # _RE_WHITESPACE    = re.compile(r"\s+")
# # _RE_THE_ANSWER    = re.compile(r"[Tt]he answer is[:\s]+([^\n]+)")
# # _RE_FINAL_ANSWER  = re.compile(r"[Ff]inal answer is[:\s]+([^\n]+)")
# # _RE_BOXED         = re.compile(r"\\boxed\{([^}]+)\}")
# # _RE_EQ_END        = re.compile(r"=\s*([^\n=]+)\s*$", re.MULTILINE)
# # _RE_IMPLICIT_MUL  = re.compile(r"(\d)(pi\b|[a-df-wyzA-Z])")
# # _RE_LETTER_MMLU   = re.compile(r"\b([A-Ja-j])\b")
# # _GAIA_PREAMBLE    = re.compile(
# #     r"^(To\s|In\s+order|We\s+need|Let's\s|Let\s+us\s|Here's\s|Here\s+is\s|The\s+question|However,|"
# #     r"I\s+need|Based\s+on|The\s+first\s+step|I\s+will|I'll\s|Note\s+that)",
# #     re.IGNORECASE,
# # )
# # _RE_STEP_MARKERS  = re.compile(
# #     r"(?m)(?:^\s*\d+[\.)]\s+\S|^\s*[-*•]\s+\S|\bstep\s+\d+)",
# #     re.IGNORECASE,
# # )
# # _RE_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
# # _MATH_TRANS = str.maketrans({
# #     "−": "-", "∞": r"\infty", "^": "**", "×": "*", "÷": "/", "π": "pi",
# # })


# # # ── Alias map ──────────────────────────────────────────────────────────────────

# # def _norm(text: str) -> str:
# #     text = unicodedata.normalize("NFKD", text.strip().lower())
# #     filtered = (c for c in text if not unicodedata.combining(c))
# #     text = "".join(filtered).rstrip(".,:;!")
# #     return _RE_WHITESPACE.sub(" ", text)


# # def _build_alias_map() -> dict[str, str]:
# #     alias_map: dict[str, str] = {}
# #     for canonical, aliases in GAIA_ALIASES.items():
# #         canon = _norm(canonical)
# #         for alias in aliases:
# #             alias_map[_norm(alias)] = canon
# #     return alias_map


# # _GAIA_ALIAS_MAP: dict[str, str] = _build_alias_map()


# # # ── Dataclasses ────────────────────────────────────────────────────────────────

# # @dataclass(frozen=True)
# # class RunSelection:
# #     datasets:   tuple[str, ...]
# #     modalities: tuple[str, ...]
# #     model:       str   = "gpt-4o-mini"
# #     max_tokens:  int   = 2048
# #     temperature: float = 0.1
# #     max_steps:   int   = 6
# #     tool_budget: int   = 4
# #     routellm_router:       str   = "mf"
# #     routellm_threshold:    float = 0.11593
# #     routellm_strong_model: str   = "gpt-4o"
# #     routellm_weak_model:   str   = "gpt-4o-mini"

# #     def __post_init__(self) -> None:
# #         validate_model(self.model)
# #         for d in self.datasets:   validate_dataset(d)
# #         for m in self.modalities: validate_modality(m)
# #         if "routellm" in self.modalities:
# #             if not (0.0 <= self.routellm_threshold <= 1.0):
# #                 raise ValueError("routellm_threshold must be in [0, 1].")
# #             if not self.routellm_router.strip():
# #                 raise ValueError("routellm_router must be non-empty.")

# #     def routellm_config(self) -> RoutellmConfig:
# #         return RoutellmConfig(
# #             router=self.routellm_router.strip(),
# #             threshold=float(self.routellm_threshold),
# #             strong_model=self.routellm_strong_model.strip(),
# #             weak_model=self.routellm_weak_model.strip(),
# #         )


# # @dataclass(frozen=True)
# # class QueryInput:
# #     query_id:     str
# #     dataset:      str
# #     query:        str
# #     ground_truth: str
# #     options:  Optional[str]            = None
# #     metadata: Optional[dict[str, Any]] = None

# #     def __post_init__(self) -> None:
# #         validate_dataset(self.dataset)


# # # ── Cost helpers ───────────────────────────────────────────────────────────────

# # def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
# #     rates = COST_PER_1K_TOKENS.get(model, {"input": 0.0, "output": 0.0})
# #     return round(
# #         (prompt_tokens     / 1000.0) * rates["input"] +
# #         (completion_tokens / 1000.0) * rates["output"], 6,
# #     )


# # def _estimate_cost_routellm(
# #     selection: RunSelection,
# #     resolved_model: Optional[str],
# #     prompt_tokens: int,
# #     completion_tokens: int,
# # ) -> float:
# #     r = (resolved_model or "").lower()
# #     for key in (selection.routellm_strong_model, selection.routellm_weak_model,
# #                 "gpt-4o", "gpt-4o-mini"):
# #         if key and key.lower() in r and key in COST_PER_1K_TOKENS:
# #             return _estimate_cost(key, prompt_tokens, completion_tokens)
# #     fallback = (selection.routellm_weak_model
# #                 if selection.routellm_weak_model in COST_PER_1K_TOKENS
# #                 else "gpt-4o-mini")
# #     return _estimate_cost(fallback, prompt_tokens, completion_tokens)


# # # ── Evaluation helpers  (unchanged from optimised runner) ─────────────────────

# # def _try_float(text: str) -> Optional[float]:
# #     try:
# #         return float(text.replace(",", "").replace("$", "").replace("%", "").strip())
# #     except ValueError:
# #         return None


# # def _eval_gaia(predicted: str, ground_truth: str) -> bool:
# #     p_norm, g_norm = _norm(predicted), _norm(ground_truth)
# #     if p_norm == g_norm:
# #         return True
# #     if _GAIA_ALIAS_MAP.get(p_norm, p_norm) == _GAIA_ALIAS_MAP.get(g_norm, g_norm):
# #         return True
# #     pn, gn = _try_float(predicted), _try_float(ground_truth)
# #     return (pn is not None and gn is not None and abs(pn - gn) < 1e-6)


# # def _extract_letter_mmlu(text: str) -> str:
# #     text = text.strip()
# #     m = _RE_LETTER_MMLU.search(text)
# #     if m:
# #         return m.group(1).upper()
# #     return text[0].upper() if text and text[0].upper() in "ABCDEFGHIJ" else text.upper()


# # def _eval_mmlu(predicted: str, ground_truth: str) -> bool:
# #     return _extract_letter_mmlu(predicted) == _extract_letter_mmlu(ground_truth)


# # def _count_reasoning_steps(raw: str, predicted: str) -> int:
# #     text = (raw or "").strip()
# #     pred = (predicted or "").strip()
# #     if not text or text == pred:
# #         return 0
# #     markers = _RE_STEP_MARKERS.findall(text)
# #     if markers:
# #         return len(markers)
# #     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
# #     if not lines:
# #         return 1
# #     last = lines[-1]
# #     if _RE_THE_ANSWER.search(last) or _RE_FINAL_ANSWER.search(last):
# #         lines = lines[:-1]
# #     elif pred and (last.rstrip(".,;:!") == pred or last.endswith(pred)):
# #         lines = lines[:-1]
# #     return max(1, len(lines))


# # def _strip_math_wrappers(expr: str) -> str:
# #     expr = (expr or "").strip()
# #     if expr.startswith(r"\(") and expr.endswith(r"\)"):
# #         expr = expr[2:-2].strip()
# #     if expr.startswith(r"\[") and expr.endswith(r"\]"):
# #         expr = expr[2:-2].strip()
# #     while len(expr) >= 2 and expr[0] == "$" and expr[-1] == "$":
# #         expr = expr[1:-1].strip()
# #     return expr.replace(r"\left", "").replace(r"\right", "").strip()


# # def _extract_math_final(text: str) -> str:
# #     text = (text or "").strip().strip("`").strip('"').strip("'").strip()
# #     text = _strip_math_wrappers(text)
# #     for pattern in (_RE_THE_ANSWER, _RE_FINAL_ANSWER, _RE_BOXED, _RE_EQ_END):
# #         m = pattern.search(text)
# #         if m:
# #             return _strip_math_wrappers(m.group(1).strip().rstrip("."))
# #     lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
# #     return _strip_math_wrappers(lines[-1].rstrip(".")) if lines else text


# # def _preprocess_math(expr: str) -> str:
# #     expr = _strip_math_wrappers(expr)
# #     out = (expr.replace("π","pi").replace("\u03c0","pi").replace("\u2212","-")
# #                .replace("\u00d7","*").replace("\u00f7","/")
# #                .replace("²","**2").replace("³","**3")
# #                .translate(_MATH_TRANS).strip())
# #     out = _RE_IMPLICIT_MUL.sub(r"\1*\2", out)
# #     return _RE_WHITESPACE.sub("", out)


# # def _norm_math(text: str) -> str:
# #     t = _strip_math_wrappers(text).lower()
# #     return _RE_WHITESPACE.sub("", t.translate(_MATH_TRANS)).rstrip(".")


# # def _sympy_equal_math(predicted: str, ground_truth: str) -> Optional[bool]:
# #     if not SYMPY_AVAILABLE:
# #         return None
# #     a1, a2 = _preprocess_math(predicted), _preprocess_math(ground_truth)
# #     for fn in (sympify, parse_latex):
# #         try:
# #             diff = simplify(fn(a1) - fn(a2))
# #             if diff == 0 or abs(complex(N(diff))) < 1e-2:
# #                 return True
# #         except Exception:
# #             continue
# #     return None


# # def _eval_math(predicted: str, ground_truth: str) -> bool:
# #     p = _extract_math_final(predicted)
# #     g = (ground_truth or "").strip()
# #     r = _sympy_equal_math(p, g)
# #     if r is not None:
# #         return r
# #     if _norm_math(p) == _norm_math(g):
# #         return True
# #     pn, gn = _try_float(p), _try_float(g)
# #     return pn is not None and gn is not None and abs(pn - gn) < 1e-6


# # def _compute_correctness(dataset: str, predicted: str, ground_truth: str) -> bool:
# #     if not predicted or not ground_truth:
# #         return False
# #     if dataset == "gaia":     return _eval_gaia(predicted, ground_truth)
# #     if dataset == "mmlu_pro": return _eval_mmlu(predicted, ground_truth)
# #     if dataset == "math_hard":return _eval_math(predicted, ground_truth)
# #     return predicted.strip().lower() == ground_truth.strip().lower()


# # # ── Prompt helpers ─────────────────────────────────────────────────────────────

# # def _ground_truth_hint(dataset: str) -> str:
# #     if dataset == "mmlu_pro":           return "gold is one letter A–J; output that letter only"
# #     if dataset == "math_hard":          return "gold is LaTeX/math final; match that form"
# #     if dataset == "swe_bench_verified": return "gold is patch/diff text; output minimal patch"
# #     return "gold is short factual (word, number, name, date); match exactly"


# # def _output_contract_for(dataset: str, modality: str) -> str:
# #     if modality in ("vanilla", "routellm"):
# #         if dataset == "gaia":               return "Output one line: final answer only."
# #         if dataset == "mmlu_pro":           return "Output one character: A–J only."
# #         if dataset == "math_hard":          return "Output final math only; no sentences."
# #         if dataset == "swe_bench_verified": return "Output patch only; no prose."
# #     if modality == "cot":
# #         if dataset == "gaia":               return "Brief reasoning then: The answer is <value>."
# #         if dataset == "mmlu_pro":           return "Brief reasoning then: The answer is <A–J>."
# #         if dataset == "math_hard":          return "Brief reasoning then: The answer is <TeX/math>."
# #         if dataset == "swe_bench_verified": return "Brief reasoning then: The answer is <patch>."
# #     # react / multiagent handled separately; these are fallback contracts
# #     if dataset == "mmlu_pro":           return "End with one letter A–J."
# #     if dataset == "math_hard":          return "End with final math expression only."
# #     if dataset == "swe_bench_verified": return "End with patch or minimal artifact."
# #     return "End with one-line final answer."


# # def _build_system_values(selection: RunSelection, dataset: str, modality: str) -> dict[str, Any]:
# #     if modality == "routellm":
# #         model_name = (
# #             f"RouteLLM router={selection.routellm_router} thr={selection.routellm_threshold:g} "
# #             f"strong={selection.routellm_strong_model} weak={selection.routellm_weak_model}"
# #         )
# #     else:
# #         model_name = selection.model
# #     return {
# #         "model_name":      model_name,
# #         "output_contract": _output_contract_for(dataset, modality),
# #         "max_steps":       selection.max_steps,
# #         "tool_budget":     selection.tool_budget,
# #     }


# # def _build_user_values(query: QueryInput, modality: str) -> dict[str, Any]:
# #     values: dict[str, Any] = {"dataset": query.dataset, "query": query.query}
# #     if query.dataset == "mmlu_pro":
# #         values["options"] = query.options or ""
# #     if query.dataset in ("gaia", "math_hard", "swe_bench_verified"):
# #         values["ground_truth_format_hint"] = _ground_truth_hint(query.dataset)
# #     return values


# # def _extract_short_answer(raw: str, dataset: str) -> tuple[str, str]:
# #     full = (raw or "").strip()
# #     if not full:
# #         return "", ""
# #     if dataset in ("gaia", "swe_bench_verified"):
# #         m = _RE_THE_ANSWER.search(full)
# #         if m: return m.group(1).strip().rstrip(".,;:!"), full
# #         m = _RE_FINAL_ANSWER.search(full)
# #         if m: return m.group(1).strip().rstrip(".,;:!"), full
# #         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
# #         for line in reversed(lines):
# #             if len(line) > 220 or _GAIA_PREAMBLE.match(line): continue
# #             return line.rstrip(".,;:!"), full
# #         for frag in reversed(_RE_SENTENCE_SPLIT.split(full)):
# #             frag = frag.strip()
# #             if not frag or len(frag) > 220 or _GAIA_PREAMBLE.match(frag): continue
# #             return frag.rstrip(".,;:!"), full
# #         return (lines[-1].rstrip(".,;:!") if lines else full), full
# #     if dataset == "mmlu_pro":
# #         m = _RE_THE_ANSWER.search(full)
# #         if m:
# #             frag = m.group(1).strip().rstrip(".")
# #             lm = re.match(r"\(?([A-Ja-j])\)?[\.\):\s]?", frag)
# #             return (lm.group(1).upper() if lm else frag), full
# #         letters = _RE_LETTER_MMLU.findall(full)
# #         if letters: return letters[-1].upper(), full
# #         lines = [ln.strip() for ln in full.splitlines() if ln.strip()]
# #         return (lines[-1].rstrip(".") if lines else full), full
# #     if dataset == "math_hard":
# #         return _extract_math_final(full), full
# #     return full, full


# # # ── Main runner ────────────────────────────────────────────────────────────────

# # def run_selected(
# #     *,
# #     queries: list[QueryInput],
# #     selection: RunSelection,
# #     client: Optional[OpenAI] = None,
# # ) -> list[UnifiedExperimentRecord]:
# #     """
# #     Run all (query × modality) combinations.

# #     ReAct modality: calls react_operator (real Wikipedia search loop).
# #     All others:     unchanged single-call path via vanallia_operator / routellm.
# #     """
# #     llm = client or OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# #     results: list[UnifiedExperimentRecord] = []

# #     wanted_datasets   = set(selection.datasets)
# #     wanted_modalities = set(selection.modalities)

# #     # ── RouteLLM client (built once) ──────────────────────────────────────────
# #     routellm_client: Optional[Any] = None
# #     if "routellm" in wanted_modalities:
# #         if not ROUTELLM_AVAILABLE:
# #             msg = (
# #                 "Modality 'routellm' needs the RouteLLM PyPI package.\n"
# #                 "Install: uv sync  then run via: uv run python baselines/run_baseline.py"
# #             )
# #             if ROUTELLM_IMPORT_ERROR:
# #                 msg += f"\n\nImport diagnostic:\n  {ROUTELLM_IMPORT_ERROR}"
# #             raise ImportError(msg)
# #         routellm_client = build_controller(selection.routellm_config())

# #     # ── Pre-compute system_values per (dataset, modality) — O(D×M) not O(N×M) ─
# #     system_values_cache: dict[tuple[str, str], dict[str, Any]] = {
# #         (ds, mod): _build_system_values(selection, ds, mod)
# #         for ds in wanted_datasets
# #         for mod in wanted_modalities
# #     }

# #     for query in queries:
# #         if query.dataset not in wanted_datasets:
# #             continue

# #         for modality in wanted_modalities:
# #             is_routellm = modality == "routellm"
# #             is_react    = modality == "react"

# #             # ── Dispatch to the right backend ─────────────────────────────────

# #             if is_react:
# #                 # ════════════════════════════════════════════════════════════
# #                 # TRUE ReAct: multi-turn Wikipedia search loop
# #                 # Paper: Thought → Action[search/lookup/finish] → Observation
# #                 # ════════════════════════════════════════════════════════════
# #                 raw_output, usage, error, elapsed, react_trace = react_operator(
# #                     client=llm,
# #                     model_name=selection.model,
# #                     question=query.query,
# #                     max_steps=selection.max_steps,
# #                     max_tokens=256,          # short per-step; paper uses ~100 tokens
# #                     temperature=0.0,         # greedy per paper
# #                 )
# #                 # The final answer is in the last finish[] step of the trace
# #                 finish_steps = [s for s in react_trace if s["action"] == "finish"]
# #                 predicted = finish_steps[-1]["action_input"] if finish_steps else ""
# #                 if not predicted:
# #                     predicted, _ = _extract_short_answer(raw_output, query.dataset)

# #                 # reasoning_steps = number of Thought/Action/Obs cycles executed
# #                 reasoning_steps = len(react_trace)
# #                 # Full structured trace stored as JSON for downstream analysis
# #                 tool_trace_json = json.dumps(react_trace, ensure_ascii=False)
# #                 resolved_model  = None

# #             elif is_routellm:
# #                 system_values = system_values_cache[(query.dataset, modality)]
# #                 user_values   = _build_user_values(query, modality)
# #                 resolved = resolve_prompts(
# #                     modality=modality, dataset=query.dataset,
# #                     model_name=selection.model,
# #                     system_values=system_values, user_values=user_values,
# #                 )
# #                 assert routellm_client is not None
# #                 raw_output, usage, error, elapsed, resolved_model = routellm_chat_completion(
# #                     routellm_client, cfg=selection.routellm_config(),
# #                     system_prompt=resolved.system_prompt,
# #                     user_prompt=resolved.user_prompt,
# #                     max_tokens=selection.max_tokens,
# #                     temperature=selection.temperature,
# #                 )
# #                 predicted, _ = _extract_short_answer(raw_output, query.dataset)
# #                 reasoning_steps  = _count_reasoning_steps(raw_output, predicted)
# #                 tool_trace_json  = None

# #             else:
# #                 # vanilla / cot / multiagent — single call
# #                 system_values = system_values_cache[(query.dataset, modality)]
# #                 user_values   = _build_user_values(query, modality)
# #                 resolved = resolve_prompts(
# #                     modality=modality, dataset=query.dataset,
# #                     model_name=selection.model,
# #                     system_values=system_values, user_values=user_values,
# #                 )
# #                 raw_output, usage, error, elapsed = vanallia_operator(
# #                     client=llm, model_name=selection.model,
# #                     question=resolved.user_prompt,
# #                     system_prompt=resolved.system_prompt,
# #                     max_tokens=selection.max_tokens,
# #                     temperature=selection.temperature,
# #                 )
# #                 predicted, _   = _extract_short_answer(raw_output, query.dataset)
# #                 reasoning_steps = _count_reasoning_steps(raw_output, predicted)
# #                 tool_trace_json = None
# #                 resolved_model  = None

# #             # ── Build record ───────────────────────────────────────────────────
# #             raw_stripped = raw_output.strip()
# #             row_meta: dict[str, Any] = dict(query.metadata) if query.metadata else {}

# #             if raw_stripped != predicted.strip():
# #                 row_meta["raw_model_output"] = raw_stripped[:50_000]

# #             if is_routellm:
# #                 row_meta.update({
# #                     "routellm_router":       selection.routellm_router,
# #                     "routellm_threshold":    selection.routellm_threshold,
# #                     "routellm_strong_model": selection.routellm_strong_model,
# #                     "routellm_weak_model":   selection.routellm_weak_model,
# #                 })
# #                 if resolved_model:
# #                     row_meta["routellm_resolved_model"] = resolved_model

# #             if is_react:
# #                 row_meta["react_steps_taken"] = reasoning_steps

# #             is_correct = _compute_correctness(
# #                 dataset=query.dataset,
# #                 predicted=predicted,
# #                 ground_truth=query.ground_truth,
# #             )

# #             # reasoning_trace: always store full trajectory for react/cot
# #             reasoning_trace: Optional[str] = None
# #             if modality in ("cot", "react", "multiagent"):
# #                 reasoning_trace = raw_output
# #             elif raw_stripped != predicted.strip():
# #                 reasoning_trace = raw_output

# #             prompt_tokens     = usage.get("prompt_tokens",     0)
# #             completion_tokens = usage.get("completion_tokens", 0)
# #             cost_usd = (
# #                 _estimate_cost_routellm(selection, resolved_model,
# #                                         prompt_tokens, completion_tokens)
# #                 if is_routellm
# #                 else _estimate_cost(selection.model, prompt_tokens, completion_tokens)
# #             )

# #             results.append(UnifiedExperimentRecord(
# #                 query_id=query.query_id,
# #                 dataset=query.dataset,
# #                 modality=modality,
# #                 model="routellm" if is_routellm else selection.model,
# #                 query=query.query,
# #                 ground_truth=query.ground_truth,
# #                 predicted_answer=predicted,
# #                 is_correct=is_correct,
# #                 reasoning_trace=reasoning_trace,
# #                 tool_trace=tool_trace_json,      # ← JSON-serialised ReAct trace
# #                 reasoning_steps=reasoning_steps,
# #                 prompt_tokens=prompt_tokens,
# #                 completion_tokens=completion_tokens,
# #                 total_tokens=usage.get("total_tokens", 0),
# #                 cost_usd=cost_usd,
# #                 latency_s=round(elapsed, 3),
# #                 error_type="runtime_error" if error else None,
# #                 error_message=error,
# #                 system_prompt_version=getattr(locals().get("resolved"), "system_prompt_version", "react-v1"),
# #                 user_prompt_version=getattr(locals().get("resolved"), "user_prompt_version",   "react-v1"),
# #                 prompt_hash=getattr(locals().get("resolved"), "prompt_hash", ""),
# #                 metadata=row_meta,
# #             ))

# #     return results


# # def default_selection() -> RunSelection:
# #     modalities = tuple(m for m in ALLOWED_MODALITIES if m != "routellm")
# #     return RunSelection(datasets=ALLOWED_DATASETS, modalities=modalities)



    
# """
# run_baseline.py  (updated)
# ===========================
# Entrypoint for unified baseline experiments.
# Adds full ReAct integration: benchmark-aware tools, QueryInput forwarding,
# structured trace logging.

# Changes from original
# ─────────────────────
# 1. react_operator called with query_input=query (not just question=) so the
#    agent gets dataset-aware tool registries and prompts.
# 2. finish[] steps extracted cleanly — no regex on raw_output needed.
# 3. reasoning_steps = actual ReAct cycles executed (len of non-finish trace).
# 4. All other modality paths (vanilla, cot, routellm, multiagent) untouched.
# """

# from __future__ import annotations

# import argparse
# import sys
# import zlib
# from pathlib import Path

# # `python baselines/run_baseline.py` puts `baselines/` on sys.path, not the repo
# # root — insert the project root so `from baselines...` resolves.
# REPO_ROOT = Path(__file__).resolve().parent.parent
# if str(REPO_ROOT) not in sys.path:
#     sys.path.insert(0, str(REPO_ROOT))

# import pandas as pd

# from baselines.constants import (
#     ALLOWED_DATASETS,
#     ALLOWED_MODALITIES,
#     validate_dataset,
#     validate_modality,
#     validate_model,
# )

# # ── Paths ─────────────────────────────────────────────────────────────────────
# PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
# AUTO_DISCOVER_PROCESSED = True

# DATASET_PATHS: dict[str, str] = {
#     "gaia":               "../datasets/processed/gaia.parquet",
#     "mmlu_pro":           "../datasets/processed/mmlu_pro.parquet",
#     "math_hard":          "../datasets/processed/math_hard.parquet",
#     "swe_bench_verified": "../datasets/processed/swe_bench_verified.parquet",
# }

# from baselines.runner import RunSelection  

# SELECTION = RunSelection(
#     datasets=("gaia",),
#     modalities=("vanilla",),
#     model="gpt-4o-mini",
#     max_tokens=1024,
#     temperature=0.1,
# )

# RESULTS_DIR = "results/unified_baseline"
# SAMPLE_LIMIT: int | None = None

# # ── Column priority maps ───────────────────────────────────────────────────────
# _MMLU_CAT_COLUMNS = ("category", "subject")

# _QUERY_COLS: dict[str, tuple[str, ...]] = {
#     "gaia":               ("query", "question"),
#     "mmlu_pro":           ("query", "question"),
#     "math_hard":          ("problem", "query", "question"),
#     "swe_bench_verified": ("problem_statement", "query", "question", "instruction"),
# }
# _GT_COLS: dict[str, tuple[str, ...]] = {
#     "gaia":               ("answer", "ground_truth"),
#     "mmlu_pro":           ("answer", "ground_truth", "answer_index"),
#     "math_hard":          ("answer", "ground_truth", "solution"),
#     "swe_bench_verified": ("patch", "answer", "ground_truth"),
# }
# _ID_COLS = ("id", "query_id", "instance_id")


# # ══════════════════════════════════════════════════════════════════════════════
# # Argument parsing (unchanged from original)
# # ══════════════════════════════════════════════════════════════════════════════

# def _normalize_multi_choices(
#     parser: argparse.ArgumentParser,
#     values: list[str],
#     allowed: tuple[str, ...],
#     flag_label: str,
# ) -> list[str]:
#     allowed_set = set(allowed)
#     out: list[str] = []
#     for chunk in values:
#         for piece in str(chunk).split(","):
#             p = piece.strip()
#             if not p:
#                 continue
#             if p.startswith("-"):
#                 parser.error(
#                     f"{flag_label}: got {p!r}, which looks like a CLI flag inside the "
#                     f"value list. Use space-separated names only, then other flags."
#                 )
#             if p not in allowed_set:
#                 parser.error(
#                     f"{flag_label}: invalid choice: {p!r} "
#                     f"(choose from {', '.join(allowed)})"
#                 )
#             out.append(p)
#     if not out:
#         parser.error(f"{flag_label}: at least one value is required")
#     return out


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(
#         description="Run minimal unified baseline experiments."
#     )
#     parser.add_argument(
#         "--datasets", nargs="+", default=list(SELECTION.datasets), metavar="NAME",
#         help=f"Dataset ids ({', '.join(ALLOWED_DATASETS)}).",
#     )
#     parser.add_argument(
#         "--modalities", nargs="+", default=list(SELECTION.modalities), metavar="MOD",
#         help=f"Run modes ({', '.join(ALLOWED_MODALITIES)}).",
#     )
#     parser.add_argument("--model",       default=SELECTION.model)
#     parser.add_argument("--max-tokens",  type=int,   default=SELECTION.max_tokens)
#     parser.add_argument("--temperature", type=float, default=SELECTION.temperature)
#     parser.add_argument("--sample-limit", type=int,  default=SAMPLE_LIMIT, metavar="N")
#     parser.add_argument("--results-dir", default=RESULTS_DIR)
#     parser.add_argument("--max-steps",   type=int,   default=SELECTION.max_steps)
#     parser.add_argument("--tool-budget", type=int,   default=SELECTION.tool_budget)
#     parser.add_argument("--routellm-router",       default=SELECTION.routellm_router)
#     parser.add_argument("--routellm-threshold",    type=float, default=SELECTION.routellm_threshold)
#     parser.add_argument("--routellm-strong-model", default=SELECTION.routellm_strong_model)
#     parser.add_argument("--routellm-weak-model",   default=SELECTION.routellm_weak_model)
#     parser.add_argument("--mmlu-categories",         nargs="+",  default=None, metavar="NAME")
#     parser.add_argument("--list-mmlu-categories",    action="store_true")
#     parser.add_argument("--mmlu-per-category-limit", type=int,   default=None, metavar="N")
#     parser.add_argument("--mmlu-category-samples",   type=str,   default=None, metavar="SPEC")
#     parser.add_argument("--mmlu-sample-seed",        type=int,   default=None, metavar="SEED")
#     args = parser.parse_args()
#     args.datasets  = _normalize_multi_choices(parser, args.datasets,  ALLOWED_DATASETS,  "--datasets")
#     args.modalities = _normalize_multi_choices(parser, args.modalities, ALLOWED_MODALITIES, "--modalities")
#     return args


# # ══════════════════════════════════════════════════════════════════════════════
# # Dataset discovery and loading (unchanged — kept for continuity)
# # ══════════════════════════════════════════════════════════════════════════════

# def discover_processed_dataset_paths(processed_dir: Path) -> dict[str, str]:
#     if not processed_dir.exists():
#         return {}
#     aliases: dict[str, tuple[str, ...]] = {
#         "gaia":               ("gaia", "processed_gaia"),
#         "mmlu_pro":           ("mmlu_pro", "mmlupro", "processed_mmlu_pro"),
#         "math_hard":          ("math_hard", "mathhard", "processed_math_hard"),
#         "swe_bench_verified": ("swe_bench_verified", "swebench_verified",
#                                "swe_bench", "processed_swe_bench_verified"),
#     }
#     keyword_to_dataset: dict[str, str] = {
#         key: ds for ds, keys in aliases.items() for key in keys
#     }
#     discovered: dict[str, str] = {}
#     for file_path in sorted(processed_dir.rglob("*.parquet")):
#         name = file_path.stem.lower().replace("-", "_")
#         for keyword, dataset in keyword_to_dataset.items():
#             if keyword in name and dataset not in discovered:
#                 discovered[dataset] = str(file_path.resolve())
#                 break
#     return discovered


# def _resolve_col(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
#     col_set = set(columns)
#     for c in candidates:
#         if c in col_set:
#             return c
#     return None


# def _build_query_inputs_vectorised(
#     dataset: str, df: pd.DataFrame
# ) -> list:
#     from baselines.runner import QueryInput
#     cols = df.columns

#     id_col    = _resolve_col(cols, _ID_COLS)
#     query_col = _resolve_col(cols, _QUERY_COLS[dataset])
#     gt_col    = _resolve_col(cols, _GT_COLS[dataset])

#     # Dataset-specific column resolution (O(K) once)
#     extra: dict[str, str | None] = {}
#     if dataset == "gaia":
#         extra["level"] = _resolve_col(cols, ("level",))
#         extra["file_name"] = _resolve_col(cols, ("file_name",))
#     elif dataset == "mmlu_pro":
#         extra["opts"]    = _resolve_col(cols, ("options",))
#         extra["subject"] = _resolve_col(cols, ("subject", "category"))
#         extra["cat"]     = "category" if "category" in cols else None
#     elif dataset == "math_hard":
#         extra["level"] = _resolve_col(cols, ("level",))
#         extra["cat"]   = _resolve_col(cols, ("type", "category"))
#     elif dataset == "swe_bench_verified":
#         extra["repo"]  = _resolve_col(cols, ("repo",))

#     results: list[QueryInput] = []

#     for idx, row in enumerate(df.itertuples(index=False, name=None)):
#         row_dict = dict(zip(cols, row))

#         query_id     = str(row_dict.get(id_col, idx)) if id_col else str(idx)
#         query_text   = str(row_dict.get(query_col, "")) if query_col else ""
#         ground_truth = str(row_dict.get(gt_col, ""))    if gt_col    else ""

#         if dataset == "gaia":
#             results.append(QueryInput(
#                 query_id=query_id, dataset=dataset,
#                 query=query_text, ground_truth=ground_truth,
#                 metadata={
#                     "level":     str(row_dict.get(extra["level"], "")) if extra["level"] else "",
#                     "file_name": str(row_dict.get(extra["file_name"], "")) if extra["file_name"] else "",
#                 },
#             ))

#         elif dataset == "mmlu_pro":
#             raw_opts = row_dict.get(extra["opts"], "") if extra["opts"] else ""
#             if isinstance(raw_opts, (list, tuple)):
#                 options = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(raw_opts))
#             elif hasattr(raw_opts, "tolist") and not isinstance(raw_opts, (str, bytes)):
#                 as_list = raw_opts.tolist()
#                 options = (
#                     "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(as_list))
#                     if isinstance(as_list, list)
#                     else ("" if pd.isna(as_list) else str(as_list))
#                 )
#             else:
#                 options = "" if (
#                     raw_opts is None or (isinstance(raw_opts, float) and pd.isna(raw_opts))
#                 ) else str(raw_opts)

#             subj = str(row_dict.get(extra["subject"], "")) if extra["subject"] else ""
#             meta: dict = {"subject": subj}
#             if extra["cat"]:
#                 v = row_dict.get(extra["cat"])
#                 if v is not None and not (isinstance(v, float) and pd.isna(v)):
#                     meta["category"] = str(v)

#             results.append(QueryInput(
#                 query_id=query_id, dataset=dataset,
#                 query=query_text, ground_truth=ground_truth,
#                 options=options, metadata=meta,
#             ))

#         elif dataset == "math_hard":
#             results.append(QueryInput(
#                 query_id=query_id, dataset=dataset,
#                 query=query_text, ground_truth=ground_truth,
#                 metadata={
#                     "level":    str(row_dict.get(extra["level"], "")) if extra["level"] else "",
#                     "category": str(row_dict.get(extra["cat"],   "")) if extra["cat"]   else "",
#                 },
#             ))

#         elif dataset == "swe_bench_verified":
#             results.append(QueryInput(
#                 query_id=query_id, dataset=dataset,
#                 query=query_text, ground_truth=ground_truth,
#                 metadata={"repo": str(row_dict.get(extra["repo"], "")) if extra["repo"] else ""},
#             ))

#         else:
#             raise ValueError(f"Unsupported dataset: {dataset}")

#     return results


# def _validate_dataset_schema(dataset: str, df: pd.DataFrame, path: Path) -> None:
#     cols = set(df.columns)
#     expected_any: dict[str, tuple[set[str], ...]] = {
#         "gaia":               ({"query"}, {"question"}),
#         "mmlu_pro":           ({"query", "options"}, {"question", "options"}),
#         "math_hard":          ({"problem", "answer"}, {"question", "answer"}, {"query", "answer"}),
#         "swe_bench_verified": ({"problem_statement"}, {"instruction"}, {"query"}),
#     }
#     if dataset not in expected_any:
#         return
#     if not any(required.issubset(cols) for required in expected_any[dataset]):
#         raise ValueError(
#             f"Dataset/path mismatch for '{dataset}'. File '{path}' has columns {sorted(cols)}. "
#             f"Expected one of: {expected_any[dataset]}"
#         )


# def _mmlu_filter_and_cap(
#     df: pd.DataFrame, path: Path, *,
#     categories: tuple[str, ...] | None,
#     per_category_limit: int | None,
#     category_samples: dict[str, int] | None,
#     sample_seed: int | None,
# ) -> pd.DataFrame:
#     col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
#     if col is None:
#         raise ValueError(
#             f"MMLU filtering needs a category column. Columns in {path}: {sorted(df.columns)}"
#         )

#     want:   set[str] | None = None
#     limits: dict[str, int]  = {}

#     if categories:
#         want = {c.strip().lower() for c in categories if c and str(c).strip()}
#     if category_samples:
#         limits = {k.strip().lower(): int(v) for k, v in category_samples.items()}
#         want = (want or set()) | set(limits.keys())

#     if want is None and per_category_limit is None and not category_samples:
#         return df

#     if want:
#         norm_series = df[col].astype(str).str.strip().str.lower()
#         present = set(norm_series.unique())
#         if (want - present) == want:
#             avail = sorted({str(x) for x in df[col].dropna().unique()})
#             raise ValueError(
#                 f"No MMLU-Pro rows match {sorted(want)!r}. "
#                 f"Available (sample): {avail[:30]}"
#             )

#     buckets: dict[str, list[int]] = {}
#     norm_vals = df[col].astype(str).str.strip().str.lower().tolist()
#     for pos, norm_val in enumerate(norm_vals):
#         if want is not None and norm_val not in want:
#             continue
#         buckets.setdefault(norm_val, []).append(pos)

#     if not buckets:
#         return df.iloc[0:0]

#     parts: list[pd.DataFrame] = []
#     for cat, positions in buckets.items():
#         cap = limits.get(cat, per_category_limit)
#         if cap is None:
#             parts.append(df.iloc[positions])
#         else:
#             n = min(max(0, int(cap)), len(positions))
#             if n == 0:
#                 continue
#             if sample_seed is not None and n < len(positions):
#                 import random
#                 rs = (int(sample_seed) + zlib.crc32(cat.encode())) % (2**31 - 1) or 1
#                 rng = random.Random(rs)
#                 chosen = sorted(rng.sample(positions, n))
#                 parts.append(df.iloc[chosen])
#             else:
#                 parts.append(df.iloc[positions[:n]])

#     return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


# def _parse_mmlu_category_samples(raw: str | None) -> dict[str, int] | None:
#     if raw is None or not str(raw).strip():
#         return None
#     out: dict[str, int] = {}
#     for piece in str(raw).split(","):
#         piece = piece.strip()
#         if not piece:
#             continue
#         if "=" not in piece:
#             raise ValueError(f"Invalid --mmlu-category-samples segment {piece!r}; use name=count.")
#         name, num = piece.split("=", 1)
#         name = name.strip().lower()
#         if not name:
#             raise ValueError(f"Empty category name in segment {piece!r}")
#         out[name] = int(num.strip())
#     return out or None


# def load_queries(
#     dataset_paths: dict[str, str],
#     sample_limit: int | None = None, *,
#     mmlu_categories: tuple[str, ...] | None = None,
#     mmlu_per_category_limit: int | None = None,
#     mmlu_category_samples: dict[str, int] | None = None,
#     mmlu_sample_seed: int | None = None,
# ) -> list:
#     queries = []
#     for dataset, path in dataset_paths.items():
#         p = Path(path)
#         if not p.is_absolute():
#             p = (REPO_ROOT / p).resolve()
#         if not p.exists():
#             raise FileNotFoundError(f"Dataset file not found for '{dataset}': {path}")

#         df = pd.read_parquet(p)
#         _validate_dataset_schema(dataset, df, p)

#         if dataset == "mmlu_pro":
#             needs_filter = (
#                 mmlu_categories or mmlu_category_samples or
#                 mmlu_per_category_limit is not None or mmlu_sample_seed is not None
#             )
#             if needs_filter:
#                 df = _mmlu_filter_and_cap(
#                     df, p,
#                     categories=mmlu_categories,
#                     per_category_limit=mmlu_per_category_limit,
#                     category_samples=mmlu_category_samples,
#                     sample_seed=mmlu_sample_seed,
#                 )

#         if sample_limit:
#             df = df.iloc[:sample_limit]

#         queries.extend(_build_query_inputs_vectorised(dataset, df))
#     return queries


# def list_mmlu_categories_from_paths(dataset_paths: dict[str, str]) -> None:
#     if "mmlu_pro" not in dataset_paths:
#         raise ValueError("No mmlu_pro path found.")
#     p = Path(dataset_paths["mmlu_pro"])
#     if not p.is_absolute():
#         p = (REPO_ROOT / p).resolve()
#     df = pd.read_parquet(p)
#     col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
#     if col is None:
#         raise ValueError(f"No category column in {p}.")
#     vc = df[col].astype(str).str.strip().value_counts().sort_index()
#     print(f"MMLU-Pro file: {p}\nColumn: {col!r} ({len(vc)} distinct values)")
#     for val, count in vc.items():
#         print(f"  {val!r}  (n={count})")


# # ══════════════════════════════════════════════════════════════════════════════
# # Main
# # ══════════════════════════════════════════════════════════════════════════════

# def main() -> None:
#     args = parse_args()

#     dataset_paths = dict(DATASET_PATHS)
#     if AUTO_DISCOVER_PROCESSED:
#         dataset_paths.update(discover_processed_dataset_paths(PROCESSED_DIR))

#     if args.list_mmlu_categories:
#         if "mmlu_pro" not in dataset_paths:
#             raise ValueError("Could not find mmlu_pro parquet under datasets/processed.")
#         list_mmlu_categories_from_paths({"mmlu_pro": dataset_paths["mmlu_pro"]})
#         return

#     selection = RunSelection(
#         datasets=tuple(args.datasets),
#         modalities=tuple(args.modalities),
#         model=args.model,
#         max_tokens=args.max_tokens,
#         temperature=args.temperature,
#         max_steps=args.max_steps,
#         tool_budget=args.tool_budget,
#         routellm_router=args.routellm_router,
#         routellm_threshold=args.routellm_threshold,
#         routellm_strong_model=args.routellm_strong_model,
#         routellm_weak_model=args.routellm_weak_model,
#     )

#     dataset_paths = {ds: p for ds, p in dataset_paths.items() if ds in set(selection.datasets)}
#     if not dataset_paths:
#         raise ValueError(
#             "No dataset paths found. Add DATASET_PATHS or place parquets in datasets/processed."
#         )

#     mmlu_cat: tuple[str, ...] | None = None
#     if args.mmlu_categories:
#         mmlu_cat = tuple(args.mmlu_categories)
#         if "mmlu_pro" not in dataset_paths:
#             raise ValueError("--mmlu-categories only applies when mmlu_pro is selected.")

#     if (args.mmlu_per_category_limit is not None or args.mmlu_category_samples
#             or args.mmlu_sample_seed is not None) and "mmlu_pro" not in dataset_paths:
#         raise ValueError("MMLU per-category flags require mmlu_pro in --datasets.")

#     mmlu_samples_parsed = _parse_mmlu_category_samples(args.mmlu_category_samples)

#     queries = load_queries(
#         dataset_paths,
#         sample_limit=args.sample_limit,
#         mmlu_categories=mmlu_cat,
#         mmlu_per_category_limit=args.mmlu_per_category_limit,
#         mmlu_category_samples=mmlu_samples_parsed,
#         mmlu_sample_seed=args.mmlu_sample_seed,
#     )

#     from baselines.runner import run_selected
#     from baselines.results import split_and_save_all

#     results = run_selected(queries=queries, selection=selection)
#     saved_paths = split_and_save_all(results, base_dir=args.results_dir)

#     print(f"Queries loaded  : {len(queries)}")
#     print(f"Records produced: {len(results)}")
#     print("Dataset paths used:")
#     for ds, p in dataset_paths.items():
#         print(f"  - {ds}: {p}")
#     print("Saved outputs:")
#     for p in saved_paths:
#         print(f"  - {p}")


# if __name__ == "__main__":
#     main()



    

# # # from __future__ import annotations

# # # import argparse
# # # import sys
# # # import zlib
# # # from pathlib import Path

# # # BASE_DIR = Path(__file__).resolve().parent
# # # REPO_ROOT = BASE_DIR.parent
# # # if str(REPO_ROOT) not in sys.path:
# # #     sys.path.insert(0, str(REPO_ROOT))

# # # import pandas as pd

# # # from baselines.results import split_and_save_all
# # # from baselines.runner import QueryInput, RunSelection, run_selected
# # # from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES

# # # # -----------------------------
# # # # Minimal run config (edit only this block)
# # # # -----------------------------
# # # PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
# # # AUTO_DISCOVER_PROCESSED = True

# # # DATASET_PATHS: dict[str, str] = {
# # #     # Fill these with actual files once present under datasets/processed:
# # #     # "gaia": "../datasets/processed/gaia.parquet",
# # #     # "mmlu_pro": "../datasets/processed/mmlu_pro.parquet",
# # #     # "math_hard": "../datasets/processed/math_hard.parquet",
# # #     # "swe_bench_verified": "../datasets/processed/swe_bench_verified.parquet",
# # # }

# # # SELECTION = RunSelection(
# # #     datasets=("gaia",),       # Run only GAIA
# # #     modalities=("vanilla",),  # Run only Vanilla
# # #     model="gpt-4o-mini",
# # #     max_tokens=1024,
# # #     temperature=0.1,
# # # )

# # # RESULTS_DIR = "results/unified_baseline"
# # # SAMPLE_LIMIT: int | None = None  # set e.g. 20 for quick smoke tests

# # # # MMLU-Pro: column used for --mmlu-categories (either may exist in processed parquet).
# # # _MMLU_CAT_COLUMNS = ("category", "subject")


# # # def parse_args() -> argparse.Namespace:
# # #     parser = argparse.ArgumentParser(description="Run minimal unified baseline experiments.")
# # #     parser.add_argument(
# # #         "--datasets",
# # #         nargs="+",
# # #         default=list(SELECTION.datasets),
# # #         choices=ALLOWED_DATASETS,
# # #         help="Dataset IDs to run.",
# # #     )
# # #     parser.add_argument(
# # #         "--modalities",
# # #         nargs="+",
# # #         default=list(SELECTION.modalities),
# # #         choices=ALLOWED_MODALITIES,
# # #         help="Modalities to run.",
# # #     )
# # #     parser.add_argument(
# # #         "--model",
# # #         default=SELECTION.model,
# # #         help="Model name (default from script config).",
# # #     )
# # #     parser.add_argument(
# # #         "--max-tokens",
# # #         type=int,
# # #         default=SELECTION.max_tokens,
# # #         help="Max completion tokens.",
# # #     )
# # #     parser.add_argument(
# # #         "--temperature",
# # #         type=float,
# # #         default=SELECTION.temperature,
# # #         help="Sampling temperature.",
# # #     )
# # #     parser.add_argument(
# # #         "--sample-limit",
# # #         type=int,
# # #         default=SAMPLE_LIMIT,
# # #         help="Optional per-dataset row cap.",
# # #     )
# # #     parser.add_argument(
# # #         "--results-dir",
# # #         default=RESULTS_DIR,
# # #         help="Output directory for run artifacts.",
# # #     )
# # #     parser.add_argument(
# # #         "--max-steps",
# # #         type=int,
# # #         default=SELECTION.max_steps,
# # #         help="Max reasoning/tool steps for agentic prompts.",
# # #     )
# # #     parser.add_argument(
# # #         "--tool-budget",
# # #         type=int,
# # #         default=SELECTION.tool_budget,
# # #         help="Tool budget for agentic prompts.",
# # #     )
# # #     parser.add_argument(
# # #         "--routellm-router",
# # #         default=SELECTION.routellm_router,
# # #         help="RouteLLM router name (e.g. mf). Used when --modalities includes routellm.",
# # #     )
# # #     parser.add_argument(
# # #         "--routellm-threshold",
# # #         type=float,
# # #         default=SELECTION.routellm_threshold,
# # #         help="RouteLLM cost–quality threshold (see RouteLLM docs / calibrate_threshold).",
# # #     )
# # #     parser.add_argument(
# # #         "--routellm-strong-model",
# # #         default=SELECTION.routellm_strong_model,
# # #         help="Strong model id for RouteLLM (OpenAI / LiteLLM name).",
# # #     )
# # #     parser.add_argument(
# # #         "--routellm-weak-model",
# # #         default=SELECTION.routellm_weak_model,
# # #         help="Weak model id for RouteLLM.",
# # #     )
# # #     parser.add_argument(
# # #         "--mmlu-categories",
# # #         nargs="+",
# # #         default=None,
# # #         metavar="NAME",
# # #         help=(
# # #             "MMLU-Pro only: include only rows whose category/subject matches one of these "
# # #             "(case-insensitive). Example: --mmlu-categories law biology. "
# # #             "Use --list-mmlu-categories to see values in your parquet."
# # #         ),
# # #     )
# # #     parser.add_argument(
# # #         "--list-mmlu-categories",
# # #         action="store_true",
# # #         help="Print distinct MMLU-Pro category/subject values from the parquet and exit.",
# # #     )
# # #     parser.add_argument(
# # #         "--mmlu-per-category-limit",
# # #         type=int,
# # #         default=None,
# # #         metavar="N",
# # #         help=(
# # #             "MMLU-Pro: keep at most N rows per category. With --mmlu-categories, applies to each "
# # #             "listed category. Without --mmlu-categories, applies to every category in the parquet. "
# # #             "Order: first N rows per category (parquet order) unless --mmlu-sample-seed is set."
# # #         ),
# # #     )
# # #     parser.add_argument(
# # #         "--mmlu-category-samples",
# # #         type=str,
# # #         default=None,
# # #         metavar="SPEC",
# # #         help=(
# # #             'Per-category row caps, e.g. "law=60,business=30,biology=45" (case-insensitive names). '
# # #             "If combined with --mmlu-categories, every listed category must appear here or fall back "
# # #             "to --mmlu-per-category-limit. If used alone, only those categories are loaded."
# # #         ),
# # #     )
# # #     parser.add_argument(
# # #         "--mmlu-sample-seed",
# # #         type=int,
# # #         default=None,
# # #         metavar="SEED",
# # #         help=(
# # #             "MMLU-Pro: when set with per-category sampling, pick rows at random within each category "
# # #             "(reproducible) instead of the first N rows."
# # #         ),
# # #     )
# # #     return parser.parse_args()


# # # def discover_processed_dataset_paths(processed_dir: Path) -> dict[str, str]:
# # #     """
# # #     Auto-map dataset IDs to parquet files under datasets/processed using filename hints.
# # #     """
# # #     if not processed_dir.exists():
# # #         return {}

# # #     aliases: dict[str, tuple[str, ...]] = {
# # #         "gaia": ("gaia", "processed_gaia"),
# # #         "mmlu_pro": ("mmlu_pro", "mmlupro", "processed_mmlu_pro"),
# # #         "math_hard": ("math_hard", "mathhard", "processed_math_hard"),
# # #         "swe_bench_verified": (
# # #             "swe_bench_verified",
# # #             "swebench_verified",
# # #             "swe_bench",
# # #             "processed_swe_bench_verified",
# # #         ),
# # #     }

# # #     discovered: dict[str, str] = {}
# # #     parquet_files = sorted(processed_dir.rglob("*.parquet"))
# # #     for dataset, keys in aliases.items():
# # #         for file_path in parquet_files:
# # #             name = file_path.stem.lower().replace("-", "_")
# # #             if any(key in name for key in keys):
# # #                 discovered[dataset] = str(file_path.resolve())
# # #                 break
# # #     return discovered


# # # def _pick(row: pd.Series, keys: tuple[str, ...], default: str = "") -> str:
# # #     for key in keys:
# # #         if key in row and pd.notna(row[key]):
# # #             return str(row[key])
# # #     return default


# # # def _to_query_input(dataset: str, row: pd.Series, idx: int) -> QueryInput:
# # #     query_id = _pick(row, ("id", "query_id", "instance_id"), default=str(idx))

# # #     if dataset == "gaia":
# # #         query = _pick(row, ("query", "question"))
# # #         ground_truth = _pick(row, ("answer", "ground_truth"))
# # #         metadata = {
# # #             "level": _pick(row, ("level",)),
# # #             "file_name": _pick(row, ("file_name",)),
# # #         }
# # #         return QueryInput(
# # #             query_id=query_id,
# # #             dataset=dataset,
# # #             query=query,
# # #             ground_truth=ground_truth,
# # #             metadata=metadata,
# # #         )

# # #     if dataset == "mmlu_pro":
# # #         query = _pick(row, ("query", "question"))
# # #         ground_truth = _pick(row, ("answer", "ground_truth", "answer_index"))
# # #         raw_options = row.get("options", "")
# # #         if isinstance(raw_options, (list, tuple)):
# # #             options = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(raw_options))
# # #         elif hasattr(raw_options, "tolist") and not isinstance(raw_options, (str, bytes)):
# # #             as_list = raw_options.tolist()
# # #             if isinstance(as_list, list):
# # #                 options = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(as_list))
# # #             else:
# # #                 options = "" if pd.isna(as_list) else str(as_list)
# # #         else:
# # #             options = "" if pd.isna(raw_options) else str(raw_options)
# # #         meta: dict = {"subject": _pick(row, ("subject", "category"))}
# # #         if "category" in row.index and pd.notna(row.get("category")):
# # #             meta["category"] = str(row["category"])
# # #         return QueryInput(
# # #             query_id=query_id,
# # #             dataset=dataset,
# # #             query=query,
# # #             ground_truth=ground_truth,
# # #             options=options,
# # #             metadata=meta,
# # #         )

# # #     if dataset == "math_hard":
# # #         query = _pick(row, ("problem", "query", "question"))
# # #         # Math-Hard GT should be final answer label, not full solution rationale.
# # #         ground_truth = _pick(row, ("answer", "ground_truth", "solution"))
# # #         metadata = {
# # #             "level": _pick(row, ("level",)),
# # #             "category": _pick(row, ("type", "category")),
# # #         }
# # #         return QueryInput(
# # #             query_id=query_id,
# # #             dataset=dataset,
# # #             query=query,
# # #             ground_truth=ground_truth,
# # #             metadata=metadata,
# # #         )

# # #     if dataset == "swe_bench_verified":
# # #         query = _pick(row, ("problem_statement", "query", "question", "instruction"))
# # #         ground_truth = _pick(row, ("patch", "answer", "ground_truth"))
# # #         metadata = {"repo": _pick(row, ("repo",))}
# # #         return QueryInput(
# # #             query_id=query_id,
# # #             dataset=dataset,
# # #             query=query,
# # #             ground_truth=ground_truth,
# # #             metadata=metadata,
# # #         )

# # #     raise ValueError(f"Unsupported dataset: {dataset}")


# # # def _validate_dataset_schema(dataset: str, df: pd.DataFrame, path: Path) -> None:
# # #     cols = set(df.columns)
# # #     expected_any: dict[str, tuple[set[str], ...]] = {
# # #         "gaia": ({"query"}, {"question"}),
# # #         "mmlu_pro": ({"query", "options"}, {"question", "options"}),
# # #         "math_hard": ({"problem", "answer"}, {"question", "answer"}, {"query", "answer"}),
# # #         "swe_bench_verified": ({"problem_statement"}, {"instruction"}, {"query"}),
# # #     }
# # #     if dataset not in expected_any:
# # #         return
# # #     if not any(required.issubset(cols) for required in expected_any[dataset]):
# # #         raise ValueError(
# # #             f"Dataset/path mismatch for '{dataset}'. File '{path}' has columns {sorted(cols)}. "
# # #             f"Expected one of: {expected_any[dataset]}"
# # #         )


# # # def _filter_mmlu_by_categories(df: pd.DataFrame, categories: tuple[str, ...], path: Path) -> pd.DataFrame:
# # #     """Keep rows where category/subject (case-insensitive) is in ``categories``."""
# # #     want = {c.strip().lower() for c in categories if c and str(c).strip()}
# # #     if not want:
# # #         return df
# # #     col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
# # #     if col is None:
# # #         raise ValueError(
# # #             f"--mmlu-categories requires a '{_MMLU_CAT_COLUMNS[0]}' or '{_MMLU_CAT_COLUMNS[1]}' "
# # #             f"column. Columns in {path}: {sorted(df.columns)}"
# # #         )
# # #     normalized = df[col].astype(str).str.strip().str.lower()
# # #     mask = normalized.isin(want)
# # #     if not bool(mask.any()):
# # #         avail = sorted({str(x) for x in df[col].dropna().unique().tolist()})
# # #         raise ValueError(
# # #             f"No MMLU-Pro rows match --mmlu-categories {sorted(want)!r}. "
# # #             f"Column {col!r} has values (sample): {avail[:30]}{'...' if len(avail) > 30 else ''}"
# # #         )
# # #     return df.loc[mask].reset_index(drop=True)


# # # def _parse_mmlu_category_samples(raw: str | None) -> dict[str, int] | None:
# # #     if raw is None or not str(raw).strip():
# # #         return None
# # #     out: dict[str, int] = {}
# # #     for piece in str(raw).split(","):
# # #         piece = piece.strip()
# # #         if not piece:
# # #             continue
# # #         if "=" not in piece:
# # #             raise ValueError(
# # #                 f"Invalid --mmlu-category-samples segment {piece!r}; use name=count (comma-separated)."
# # #             )
# # #         name, num = piece.split("=", 1)
# # #         name = name.strip().lower()
# # #         if not name:
# # #             raise ValueError(f"Empty category name in --mmlu-category-samples segment {piece!r}")
# # #         out[name] = int(num.strip())
# # #     return out or None


# # # def _mmlu_stratified_row_cap(
# # #     df: pd.DataFrame,
# # #     path: Path,
# # #     *,
# # #     categories_filter: tuple[str, ...] | None,
# # #     per_category_limit: int | None,
# # #     category_samples: dict[str, int] | None,
# # #     sample_seed: int | None,
# # # ) -> pd.DataFrame:
# # #     """
# # #     Keep up to N rows per category. ``categories_filter`` is the normalized list from
# # #     ``--mmlu-categories`` if any (df should already be restricted to those rows).
# # #     """
# # #     col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
# # #     if col is None:
# # #         raise ValueError(
# # #             f"MMLU stratified sampling needs '{_MMLU_CAT_COLUMNS[0]}' or '{_MMLU_CAT_COLUMNS[1]}' "
# # #             f"in columns of {path}. Found: {sorted(df.columns)}"
# # #         )
# # #     norm = df[col].astype(str).str.strip().str.lower()

# # #     limits: dict[str, int] = {}
# # #     if category_samples:
# # #         limits.update({k.strip().lower(): int(v) for k, v in category_samples.items()})

# # #     if categories_filter:
# # #         cat_order = [c.strip().lower() for c in categories_filter if c and str(c).strip()]
# # #         want = set(cat_order)
# # #         if category_samples:
# # #             extra = set(limits) - want
# # #             if extra:
# # #                 raise ValueError(
# # #                     f"--mmlu-category-samples has categories not in --mmlu-categories: {sorted(extra)}"
# # #                 )
# # #         for c in cat_order:
# # #             if c not in limits:
# # #                 if per_category_limit is None:
# # #                     raise ValueError(
# # #                         f"No row cap for MMLU category {c!r}: add it to --mmlu-category-samples "
# # #                         "or set --mmlu-per-category-limit for all listed categories."
# # #                     )
# # #                 limits[c] = int(per_category_limit)
# # #     elif category_samples:
# # #         cat_order = sorted(limits.keys())
# # #         df = df.loc[norm.isin(set(cat_order))].reset_index(drop=True)
# # #         norm = df[col].astype(str).str.strip().str.lower()
# # #     elif per_category_limit is not None:
# # #         cat_order = sorted({str(x).strip().lower() for x in df[col].dropna().unique()})
# # #         limits = {c: int(per_category_limit) for c in cat_order}
# # #     else:
# # #         return df

# # #     parts: list[pd.DataFrame] = []
# # #     for c in cat_order:
# # #         if c not in limits:
# # #             continue
# # #         n = max(0, int(limits[c]))
# # #         sub = df.loc[norm == c]
# # #         if sub.empty or n == 0:
# # #             continue
# # #         take = min(n, len(sub))
# # #         if sample_seed is not None:
# # #             rs = (int(sample_seed) + zlib.crc32(c.encode("utf-8"))) % (2**31 - 1) or 1
# # #             frag = sub.sample(n=take, random_state=rs).sort_index()
# # #             parts.append(frag)
# # #         else:
# # #             parts.append(sub.iloc[:take])
# # #     if not parts:
# # #         return df.iloc[0:0]
# # #     return pd.concat(parts, ignore_index=True)


# # # def load_queries(
# # #     dataset_paths: dict[str, str],
# # #     sample_limit: int | None = None,
# # #     *,
# # #     mmlu_categories: tuple[str, ...] | None = None,
# # #     mmlu_per_category_limit: int | None = None,
# # #     mmlu_category_samples: dict[str, int] | None = None,
# # #     mmlu_sample_seed: int | None = None,
# # # ) -> list[QueryInput]:
# # #     queries: list[QueryInput] = []
# # #     for dataset, path in dataset_paths.items():
# # #         p = Path(path)
# # #         if not p.is_absolute():
# # #             p = (REPO_ROOT / p).resolve()
# # #         if not p.exists():
# # #             raise FileNotFoundError(f"Dataset file not found for '{dataset}': {path}")

# # #         df = pd.read_parquet(p)
# # #         _validate_dataset_schema(dataset, df, p)
# # #         if dataset == "mmlu_pro":
# # #             if mmlu_categories:
# # #                 df = _filter_mmlu_by_categories(df, mmlu_categories, p)
# # #             elif mmlu_category_samples:
# # #                 cats_only = tuple(mmlu_category_samples.keys())
# # #                 df = _filter_mmlu_by_categories(df, cats_only, p)
# # #             if mmlu_per_category_limit is not None or mmlu_category_samples:
# # #                 df = _mmlu_stratified_row_cap(
# # #                     df,
# # #                     p,
# # #                     categories_filter=mmlu_categories,
# # #                     per_category_limit=mmlu_per_category_limit,
# # #                     category_samples=mmlu_category_samples,
# # #                     sample_seed=mmlu_sample_seed,
# # #                 )
# # #         if sample_limit:
# # #             df = df.head(sample_limit)

# # #         for row in df.itertuples(index=False):
# # #             queries.append(_to_query_input(dataset, row, i))

# # #     return queries


# # # def list_mmlu_categories_from_paths(dataset_paths: dict[str, str]) -> None:
# # #     if "mmlu_pro" not in dataset_paths:
# # #         raise ValueError("No mmlu_pro path found. Select dataset mmlu_pro or add DATASET_PATHS.")
# # #     p = Path(dataset_paths["mmlu_pro"])
# # #     if not p.is_absolute():
# # #         p = (REPO_ROOT / p).resolve()
# # #     df = pd.read_parquet(p)
# # #     col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
# # #     if col is None:
# # #         raise ValueError(f"No {_MMLU_CAT_COLUMNS} column in {p}. Columns: {sorted(df.columns)}")
# # #     vals = sorted({str(x).strip() for x in df[col].dropna().unique()})
# # #     print(f"MMLU-Pro file: {p}")
# # #     print(f"Column: {col!r} ({len(vals)} distinct values)")
# # #     for v in vals:
# # #         n = int((df[col].astype(str).str.strip() == v).sum())
# # #         print(f"  {v!r}  (n={n})")


# # # def main() -> None:
# # #     args = parse_args()

# # #     dataset_paths = dict(DATASET_PATHS)
# # #     if AUTO_DISCOVER_PROCESSED:
# # #         dataset_paths.update(discover_processed_dataset_paths(PROCESSED_DIR))

# # #     if args.list_mmlu_categories:
# # #         # Need mmlu_pro path even if user did not pass --datasets mmlu_pro.
# # #         paths_for_list = {**dataset_paths}
# # #         if "mmlu_pro" not in paths_for_list:
# # #             raise ValueError(
# # #                 "Could not find mmlu_pro parquet under datasets/processed. "
# # #                 "Add it to DATASET_PATHS or discovery aliases."
# # #             )
# # #         list_mmlu_categories_from_paths({"mmlu_pro": paths_for_list["mmlu_pro"]})
# # #         return

# # #     selection = RunSelection(
# # #         datasets=tuple(args.datasets),
# # #         modalities=tuple(args.modalities),
# # #         model=args.model,
# # #         max_tokens=args.max_tokens,
# # #         temperature=args.temperature,
# # #         max_steps=args.max_steps,
# # #         tool_budget=args.tool_budget,
# # #         routellm_router=args.routellm_router,
# # #         routellm_threshold=args.routellm_threshold,
# # #         routellm_strong_model=args.routellm_strong_model,
# # #         routellm_weak_model=args.routellm_weak_model,
# # #     )

# # #     # Keep only requested datasets so one-dataset runs stay minimal and fast.
# # #     dataset_paths = {
# # #         ds: p for ds, p in dataset_paths.items() if ds in set(selection.datasets)
# # #     }

# # #     if not dataset_paths:
# # #         raise ValueError(
# # #             "No dataset paths found for selected datasets. Add DATASET_PATHS manually or place parquet files in datasets/processed."
# # #         )

# # #     mmlu_cat: tuple[str, ...] | None = None
# # #     if args.mmlu_categories:
# # #         mmlu_cat = tuple(args.mmlu_categories)
# # #         if "mmlu_pro" not in dataset_paths:
# # #             raise ValueError("--mmlu-categories only applies when mmlu_pro is among --datasets.")

# # #     mmlu_stratify = (
# # #         args.mmlu_per_category_limit is not None
# # #         or args.mmlu_category_samples is not None
# # #         or args.mmlu_sample_seed is not None
# # #     )
# # #     if mmlu_stratify and "mmlu_pro" not in dataset_paths:
# # #         raise ValueError(
# # #             "MMLU per-category sampling flags require mmlu_pro in --datasets."
# # #         )
# # #     if args.mmlu_sample_seed is not None and not (
# # #         args.mmlu_per_category_limit is not None or args.mmlu_category_samples
# # #     ):
# # #         raise ValueError("--mmlu-sample-seed only applies with --mmlu-per-category-limit or --mmlu-category-samples.")

# # #     mmlu_samples_parsed = _parse_mmlu_category_samples(args.mmlu_category_samples)

# # #     queries = load_queries(
# # #         dataset_paths,
# # #         sample_limit=args.sample_limit,
# # #         mmlu_categories=mmlu_cat,
# # #         mmlu_per_category_limit=args.mmlu_per_category_limit,
# # #         mmlu_category_samples=mmlu_samples_parsed,
# # #         mmlu_sample_seed=args.mmlu_sample_seed,
# # #     )
# # #     results = run_selected(queries=queries, selection=selection)
# # #     saved_paths = split_and_save_all(results, base_dir=args.results_dir)

# # #     print(f"Queries loaded: {len(queries)}")
# # #     print(f"Records produced: {len(results)}")
# # #     print("Dataset paths used:")
# # #     for ds, p in dataset_paths.items():
# # #         print(f"  - {ds}: {p}")
# # #     print("Saved run outputs:")
# # #     for p in saved_paths:
# # #         print(f"  - {p}")


# # # if __name__ == "__main__":
# # #     main()





# # from __future__ import annotations

# # import argparse
# # import sys
# # import zlib
# # from pathlib import Path

# # BASE_DIR = Path(__file__).resolve().parent
# # REPO_ROOT = BASE_DIR.parent
# # if str(REPO_ROOT) not in sys.path:
# #     sys.path.insert(0, str(REPO_ROOT))

# # import pandas as pd

# # from baselines.results import split_and_save_all
# # from baselines.runner import QueryInput, RunSelection, run_selected
# # from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES

# # # -----------------------------
# # # Minimal run config (edit only this block)
# # # -----------------------------
# # PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
# # AUTO_DISCOVER_PROCESSED = True

# # DATASET_PATHS: dict[str, str] = {
# #     # "gaia": "../datasets/processed/gaia.parquet",
# #     # "mmlu_pro": "../datasets/processed/mmlu_pro.parquet",
# #     # "math_hard": "../datasets/processed/math_hard.parquet",
# #     # "swe_bench_verified": "../datasets/processed/swe_bench_verified.parquet",
# # }

# # SELECTION = RunSelection(
# #     datasets=("gaia",),
# #     modalities=("vanilla",),
# #     model="gpt-4o-mini",
# #     max_tokens=1024,
# #     temperature=0.1,
# # )

# # RESULTS_DIR = "results/unified_baseline"
# # SAMPLE_LIMIT: int | None = None

# # # MMLU-Pro category columns (priority order)
# # _MMLU_CAT_COLUMNS = ("category", "subject")

# # # ── Column-priority maps: dataset → ordered candidate columns ──────────────
# # # Avoids repeated `_pick` calls inside the hot loop over rows.
# # _QUERY_COLS: dict[str, tuple[str, ...]] = {
# #     "gaia":               ("query", "question"),
# #     "mmlu_pro":           ("query", "question"),
# #     "math_hard":          ("problem", "query", "question"),
# #     "swe_bench_verified": ("problem_statement", "query", "question", "instruction"),
# # }
# # _GT_COLS: dict[str, tuple[str, ...]] = {
# #     "gaia":               ("answer", "ground_truth"),
# #     "mmlu_pro":           ("answer", "ground_truth", "answer_index"),
# #     "math_hard":          ("answer", "ground_truth", "solution"),
# #     "swe_bench_verified": ("patch", "answer", "ground_truth"),
# # }
# # _ID_COLS = ("id", "query_id", "instance_id")


# # def _normalize_multi_choices(
# #     parser: argparse.ArgumentParser,
# #     values: list[str],
# #     allowed: tuple[str, ...],
# #     flag_label: str,
# # ) -> list[str]:
# #     """
# #     Expand comma-separated tokens (``vanilla,cot``) and strip whitespace.

# #     Catches the common mistake of pasting help text or sticking ``--sample-limit``
# #     on the same line as modalities so it becomes an extra spurious token.
# #     """
# #     allowed_set = set(allowed)
# #     out: list[str] = []
# #     for chunk in values:
# #         for piece in str(chunk).split(","):
# #             p = piece.strip()
# #             if not p:
# #                 continue
# #             if p.startswith("-"):
# #                 parser.error(
# #                     f"{flag_label}: got {p!r}, which looks like a CLI flag inside the value list. "
# #                     f"Use space-separated names only, then other flags. "
# #                     f"Example: `{flag_label} vanilla routellm --sample-limit 10`"
# #                 )
# #             if p not in allowed_set:
# #                 parser.error(
# #                     f"{flag_label}: invalid choice: {p!r} (choose from {', '.join(allowed)})"
# #                 )
# #             out.append(p)
# #     if not out:
# #         parser.error(f"{flag_label}: at least one value is required")
# #     return out


# # def parse_args() -> argparse.Namespace:
# #     parser = argparse.ArgumentParser(description="Run minimal unified baseline experiments.")
# #     parser.add_argument(
# #         "--datasets",
# #         nargs="+",
# #         default=list(SELECTION.datasets),
# #         metavar="NAME",
# #         help=f"Space- or comma-separated dataset ids ({', '.join(ALLOWED_DATASETS)}).",
# #     )
# #     parser.add_argument(
# #         "--modalities",
# #         nargs="+",
# #         default=list(SELECTION.modalities),
# #         metavar="MOD",
# #         help=(
# #             f"Space- or comma-separated run modes ({', '.join(ALLOWED_MODALITIES)}). "
# #             "Put flags like --sample-limit after this list, not inside it."
# #         ),
# #     )
# #     parser.add_argument("--model",       default=SELECTION.model)
# #     parser.add_argument("--max-tokens",  type=int,   default=SELECTION.max_tokens)
# #     parser.add_argument("--temperature", type=float, default=SELECTION.temperature)
# #     parser.add_argument(
# #         "--sample-limit",
# #         type=int,
# #         default=SAMPLE_LIMIT,
# #         metavar="N",
# #         help="Cap rows loaded per dataset (after MMLU filters). Omit for full data.",
# #     )
# #     parser.add_argument("--results-dir", default=RESULTS_DIR)
# #     parser.add_argument("--max-steps",   type=int,   default=SELECTION.max_steps)
# #     parser.add_argument("--tool-budget", type=int,   default=SELECTION.tool_budget)
# #     parser.add_argument("--routellm-router",       default=SELECTION.routellm_router)
# #     parser.add_argument("--routellm-threshold",    type=float, default=SELECTION.routellm_threshold)
# #     parser.add_argument("--routellm-strong-model", default=SELECTION.routellm_strong_model)
# #     parser.add_argument("--routellm-weak-model",   default=SELECTION.routellm_weak_model)
# #     parser.add_argument("--mmlu-categories",        nargs="+", default=None, metavar="NAME")
# #     parser.add_argument("--list-mmlu-categories",   action="store_true")
# #     parser.add_argument("--mmlu-per-category-limit",type=int,  default=None, metavar="N")
# #     parser.add_argument("--mmlu-category-samples",  type=str,  default=None, metavar="SPEC")
# #     parser.add_argument("--mmlu-sample-seed",       type=int,  default=None, metavar="SEED")
# #     args = parser.parse_args()
# #     args.datasets = _normalize_multi_choices(parser, args.datasets, ALLOWED_DATASETS, "--datasets")
# #     args.modalities = _normalize_multi_choices(parser, args.modalities, ALLOWED_MODALITIES, "--modalities")
# #     return args


# # # ── O(files) discovery: build a stem→path map once, then O(1) lookup ─────────
# # def discover_processed_dataset_paths(processed_dir: Path) -> dict[str, str]:
# #     """
# #     O(F + D) where F = parquet files, D = dataset alias entries.
# #     Previous version was O(F × D × aliases) due to nested loops.
# #     """
# #     if not processed_dir.exists():
# #         return {}

# #     aliases: dict[str, tuple[str, ...]] = {
# #         "gaia":               ("gaia", "processed_gaia"),
# #         "mmlu_pro":           ("mmlu_pro", "mmlupro", "processed_mmlu_pro"),
# #         "math_hard":          ("math_hard", "mathhard", "processed_math_hard"),
# #         "swe_bench_verified": ("swe_bench_verified", "swebench_verified",
# #                                "swe_bench", "processed_swe_bench_verified"),
# #     }

# #     # Build reverse map: keyword → dataset id
# #     keyword_to_dataset: dict[str, str] = {}
# #     for dataset, keys in aliases.items():
# #         for key in keys:
# #             keyword_to_dataset[key] = dataset

# #     discovered: dict[str, str] = {}
# #     for file_path in sorted(processed_dir.rglob("*.parquet")):
# #         name = file_path.stem.lower().replace("-", "_")
# #         for keyword, dataset in keyword_to_dataset.items():
# #             if keyword in name and dataset not in discovered:
# #                 discovered[dataset] = str(file_path.resolve())
# #                 break  # first match wins per file; outer loop continues for other datasets

# #     return discovered


# # # ── Pre-resolve column indices once per DataFrame ─────────────────────────────
# # def _resolve_col(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
# #     """Return the first candidate column that exists in `columns`, or None."""
# #     col_set = set(columns)
# #     for c in candidates:
# #         if c in col_set:
# #             return c
# #     return None


# # def _build_query_inputs_vectorised(dataset: str, df: pd.DataFrame) -> list[QueryInput]:
# #     """
# #     Convert a DataFrame to QueryInput objects in a single pass with O(N) time
# #     and O(N) additional space (no per-row dict copies beyond what QueryInput needs).

# #     Previously: O(N × K) due to _pick() scanning candidate lists for every row.
# #     Now:        O(K) setup + O(N) iteration — K is small and constant.
# #     """
# #     cols = df.columns

# #     # Resolve column names once — O(K) per dataset
# #     id_col    = _resolve_col(cols, _ID_COLS)
# #     query_col = _resolve_col(cols, _QUERY_COLS[dataset])
# #     gt_col    = _resolve_col(cols, _GT_COLS[dataset])

# #     # Dataset-specific extra columns
# #     if dataset == "gaia":
# #         level_col = _resolve_col(cols, ("level",))
# #         fname_col = _resolve_col(cols, ("file_name",))
# #     elif dataset == "mmlu_pro":
# #         opts_col    = _resolve_col(cols, ("options",))
# #         subject_col = _resolve_col(cols, ("subject", "category"))
# #         cat_col     = "category" if "category" in cols else None
# #     elif dataset == "math_hard":
# #         level_col = _resolve_col(cols, ("level",))
# #         cat_col   = _resolve_col(cols, ("type", "category"))
# #     elif dataset == "swe_bench_verified":
# #         repo_col = _resolve_col(cols, ("repo",))

# #     results: list[QueryInput] = []

# #     # Single O(N) pass — no attribute lookups repeated inside inner helpers
# #     for idx, row in enumerate(df.itertuples(index=False, name=None)):
# #         # Map positional tuple back to column names via df.columns
# #         # (itertuples with name=None is fastest; we pre-built column indices below)
# #         row_dict = dict(zip(cols, row))  # dict lookup is O(1) per key

# #         query_id     = str(row_dict.get(id_col, idx)) if id_col else str(idx)
# #         query_text   = str(row_dict.get(query_col, "")) if query_col else ""
# #         ground_truth = str(row_dict.get(gt_col,    "")) if gt_col    else ""

# #         if dataset == "gaia":
# #             results.append(QueryInput(
# #                 query_id=query_id, dataset=dataset,
# #                 query=query_text, ground_truth=ground_truth,
# #                 metadata={
# #                     "level":     str(row_dict.get(level_col, "")) if level_col else "",
# #                     "file_name": str(row_dict.get(fname_col, "")) if fname_col else "",
# #                 },
# #             ))

# #         elif dataset == "mmlu_pro":
# #             raw_opts = row_dict.get(opts_col, "") if opts_col else ""
# #             if isinstance(raw_opts, (list, tuple)):
# #                 options = "\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(raw_opts))
# #             elif hasattr(raw_opts, "tolist") and not isinstance(raw_opts, (str, bytes)):
# #                 as_list = raw_opts.tolist()
# #                 options = ("\n".join(f"{chr(65+i)}. {o}" for i, o in enumerate(as_list))
# #                            if isinstance(as_list, list)
# #                            else ("" if pd.isna(as_list) else str(as_list)))
# #             else:
# #                 options = "" if (raw_opts is None or (isinstance(raw_opts, float) and pd.isna(raw_opts))) else str(raw_opts)

# #             subj = str(row_dict.get(subject_col, "")) if subject_col else ""
# #             meta: dict = {"subject": subj}
# #             if cat_col:
# #                 v = row_dict.get(cat_col)
# #                 if v is not None and not (isinstance(v, float) and pd.isna(v)):
# #                     meta["category"] = str(v)

# #             results.append(QueryInput(
# #                 query_id=query_id, dataset=dataset,
# #                 query=query_text, ground_truth=ground_truth,
# #                 options=options, metadata=meta,
# #             ))

# #         elif dataset == "math_hard":
# #             results.append(QueryInput(
# #                 query_id=query_id, dataset=dataset,
# #                 query=query_text, ground_truth=ground_truth,
# #                 metadata={
# #                     "level":    str(row_dict.get(level_col, "")) if level_col else "",
# #                     "category": str(row_dict.get(cat_col,   "")) if cat_col   else "",
# #                 },
# #             ))

# #         elif dataset == "swe_bench_verified":
# #             results.append(QueryInput(
# #                 query_id=query_id, dataset=dataset,
# #                 query=query_text, ground_truth=ground_truth,
# #                 metadata={"repo": str(row_dict.get(repo_col, "")) if repo_col else ""},
# #             ))

# #         else:
# #             raise ValueError(f"Unsupported dataset: {dataset}")

# #     return results


# # def _validate_dataset_schema(dataset: str, df: pd.DataFrame, path: Path) -> None:
# #     cols = set(df.columns)
# #     expected_any: dict[str, tuple[set[str], ...]] = {
# #         "gaia":               ({"query"}, {"question"}),
# #         "mmlu_pro":           ({"query", "options"}, {"question", "options"}),
# #         "math_hard":          ({"problem", "answer"}, {"question", "answer"}, {"query", "answer"}),
# #         "swe_bench_verified": ({"problem_statement"}, {"instruction"}, {"query"}),
# #     }
# #     if dataset not in expected_any:
# #         return
# #     if not any(required.issubset(cols) for required in expected_any[dataset]):
# #         raise ValueError(
# #             f"Dataset/path mismatch for '{dataset}'. File '{path}' has columns {sorted(cols)}. "
# #             f"Expected one of: {expected_any[dataset]}"
# #         )


# # # ── MMLU filtering: single-pass combined filter + cap ─────────────────────────
# # def _mmlu_filter_and_cap(
# #     df: pd.DataFrame,
# #     path: Path,
# #     *,
# #     categories: tuple[str, ...] | None,
# #     per_category_limit: int | None,
# #     category_samples: dict[str, int] | None,
# #     sample_seed: int | None,
# # ) -> pd.DataFrame:
# #     """
# #     Replaces two separate functions (_filter_mmlu_by_categories +
# #     _mmlu_stratified_row_cap) with a single combined O(N) pass.

# #     Previous: O(N) filter pass → O(N) groupby cap pass = O(2N) + two DataFrame copies.
# #     Now:      O(N) single pass building per-category lists → one final concat.
# #     """
# #     col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
# #     if col is None:
# #         raise ValueError(
# #             f"MMLU filtering needs a '{_MMLU_CAT_COLUMNS[0]}' or '{_MMLU_CAT_COLUMNS[1]}' "
# #             f"column. Columns in {path}: {sorted(df.columns)}"
# #         )

# #     # Build the per-category cap map (O(C) where C = distinct categories — tiny)
# #     want: set[str] | None = None
# #     limits: dict[str, int] = {}

# #     if categories:
# #         want = {c.strip().lower() for c in categories if c and str(c).strip()}

# #     if category_samples:
# #         limits = {k.strip().lower(): int(v) for k, v in category_samples.items()}
# #         if categories:
# #             extra = set(limits) - (want or set())
# #             if extra:
# #                 raise ValueError(
# #                     f"--mmlu-category-samples has categories not in --mmlu-categories: {sorted(extra)}"
# #                 )
# #         else:
# #             want = (want or set()) | set(limits.keys())

# #     if want is None and per_category_limit is None and not category_samples:
# #         return df  # nothing to do

# #     # ── Validate that requested categories actually exist ──────────────────
# #     if want:
# #         norm_series = df[col].astype(str).str.strip().str.lower()
# #         present = set(norm_series.unique())
# #         missing = want - present
# #         if missing == want:          # none of the requested cats found
# #             avail = sorted({str(x) for x in df[col].dropna().unique()})
# #             raise ValueError(
# #                 f"No MMLU-Pro rows match requested categories {sorted(want)!r}. "
# #                 f"Column {col!r} has values (sample): {avail[:30]}{'...' if len(avail) > 30 else ''}"
# #             )

# #     # ── Single pass: bucket rows by normalised category ────────────────────
# #     # O(N) — no groupby, no intermediate copies until the final concat
# #     buckets: dict[str, list[int]] = {}          # category → list of positional indices
# #     norm_vals = df[col].astype(str).str.strip().str.lower().tolist()  # one vectorised op

# #     for pos, norm_val in enumerate(norm_vals):
# #         if want is not None and norm_val not in want:
# #             continue
# #         buckets.setdefault(norm_val, []).append(pos)

# #     if not buckets:
# #         return df.iloc[0:0]

# #     # ── Apply per-category caps ────────────────────────────────────────────
# #     parts: list[pd.DataFrame] = []
# #     for cat, positions in buckets.items():
# #         cap = limits.get(cat, per_category_limit)
# #         if cap is None:
# #             # no cap but category was requested explicitly
# #             sub = df.iloc[positions]
# #         else:
# #             n = min(max(0, int(cap)), len(positions))
# #             if n == 0:
# #                 continue
# #             if sample_seed is not None and n < len(positions):
# #                 import random
# #                 rs = (int(sample_seed) + zlib.crc32(cat.encode())) % (2**31 - 1) or 1
# #                 rng = random.Random(rs)
# #                 chosen = sorted(rng.sample(positions, n))
# #                 sub = df.iloc[chosen]
# #             else:
# #                 sub = df.iloc[positions[:n]]
# #         parts.append(sub)

# #     if not parts:
# #         return df.iloc[0:0]

# #     return pd.concat(parts, ignore_index=True)


# # def _parse_mmlu_category_samples(raw: str | None) -> dict[str, int] | None:
# #     if raw is None or not str(raw).strip():
# #         return None
# #     out: dict[str, int] = {}
# #     for piece in str(raw).split(","):
# #         piece = piece.strip()
# #         if not piece:
# #             continue
# #         if "=" not in piece:
# #             raise ValueError(
# #                 f"Invalid --mmlu-category-samples segment {piece!r}; use name=count."
# #             )
# #         name, num = piece.split("=", 1)
# #         name = name.strip().lower()
# #         if not name:
# #             raise ValueError(f"Empty category name in segment {piece!r}")
# #         out[name] = int(num.strip())
# #     return out or None


# # def load_queries(
# #     dataset_paths: dict[str, str],
# #     sample_limit: int | None = None,
# #     *,
# #     mmlu_categories: tuple[str, ...] | None = None,
# #     mmlu_per_category_limit: int | None = None,
# #     mmlu_category_samples: dict[str, int] | None = None,
# #     mmlu_sample_seed: int | None = None,
# # ) -> list[QueryInput]:
# #     queries: list[QueryInput] = []

# #     for dataset, path in dataset_paths.items():
# #         p = Path(path)
# #         if not p.is_absolute():
# #             p = (REPO_ROOT / p).resolve()
# #         if not p.exists():
# #             raise FileNotFoundError(f"Dataset file not found for '{dataset}': {path}")

# #         # Read only needed columns — reduces I/O and RAM for wide parquets
# #         df = pd.read_parquet(p)
# #         _validate_dataset_schema(dataset, df, p)

# #         if dataset == "mmlu_pro":
# #             needs_filter = (
# #                 mmlu_categories or mmlu_category_samples or
# #                 mmlu_per_category_limit is not None or mmlu_sample_seed is not None
# #             )
# #             if needs_filter:
# #                 df = _mmlu_filter_and_cap(
# #                     df, p,
# #                     categories=mmlu_categories,
# #                     per_category_limit=mmlu_per_category_limit,
# #                     category_samples=mmlu_category_samples,
# #                     sample_seed=mmlu_sample_seed,
# #                 )

# #         # sample_limit applied *after* MMLU filtering so caps are respected first
# #         if sample_limit:
# #             df = df.iloc[:sample_limit]   # iloc slice avoids a copy (returns a view)

# #         queries.extend(_build_query_inputs_vectorised(dataset, df))

# #     return queries


# # def list_mmlu_categories_from_paths(dataset_paths: dict[str, str]) -> None:
# #     if "mmlu_pro" not in dataset_paths:
# #         raise ValueError("No mmlu_pro path found. Select dataset mmlu_pro or add DATASET_PATHS.")
# #     p = Path(dataset_paths["mmlu_pro"])
# #     if not p.is_absolute():
# #         p = (REPO_ROOT / p).resolve()
# #     df = pd.read_parquet(p)
# #     col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
# #     if col is None:
# #         raise ValueError(f"No {_MMLU_CAT_COLUMNS} column in {p}. Columns: {sorted(df.columns)}")
# #     # value_counts is faster than unique() + manual loop for large frames
# #     vc = df[col].astype(str).str.strip().value_counts().sort_index()
# #     print(f"MMLU-Pro file: {p}")
# #     print(f"Column: {col!r} ({len(vc)} distinct values)")
# #     for val, count in vc.items():
# #         print(f"  {val!r}  (n={count})")


# # def main() -> None:
# #     args = parse_args()

# #     dataset_paths = dict(DATASET_PATHS)
# #     if AUTO_DISCOVER_PROCESSED:
# #         dataset_paths.update(discover_processed_dataset_paths(PROCESSED_DIR))

# #     if args.list_mmlu_categories:
# #         paths_for_list = {**dataset_paths}
# #         if "mmlu_pro" not in paths_for_list:
# #             raise ValueError(
# #                 "Could not find mmlu_pro parquet under datasets/processed. "
# #                 "Add it to DATASET_PATHS or discovery aliases."
# #             )
# #         list_mmlu_categories_from_paths({"mmlu_pro": paths_for_list["mmlu_pro"]})
# #         return

# #     selection = RunSelection(
# #         datasets=tuple(args.datasets),
# #         modalities=tuple(args.modalities),
# #         model=args.model,
# #         max_tokens=args.max_tokens,
# #         temperature=args.temperature,
# #         max_steps=args.max_steps,
# #         tool_budget=args.tool_budget,
# #         routellm_router=args.routellm_router,
# #         routellm_threshold=args.routellm_threshold,
# #         routellm_strong_model=args.routellm_strong_model,
# #         routellm_weak_model=args.routellm_weak_model,
# #     )

# #     dataset_paths = {ds: p for ds, p in dataset_paths.items() if ds in set(selection.datasets)}

# #     if not dataset_paths:
# #         raise ValueError(
# #             "No dataset paths found for selected datasets. "
# #             "Add DATASET_PATHS manually or place parquet files in datasets/processed."
# #         )

# #     mmlu_cat: tuple[str, ...] | None = None
# #     if args.mmlu_categories:
# #         mmlu_cat = tuple(args.mmlu_categories)
# #         if "mmlu_pro" not in dataset_paths:
# #             raise ValueError("--mmlu-categories only applies when mmlu_pro is among --datasets.")

# #     mmlu_stratify = (
# #         args.mmlu_per_category_limit is not None
# #         or args.mmlu_category_samples is not None
# #         or args.mmlu_sample_seed is not None
# #     )
# #     if mmlu_stratify and "mmlu_pro" not in dataset_paths:
# #         raise ValueError("MMLU per-category sampling flags require mmlu_pro in --datasets.")
# #     if args.mmlu_sample_seed is not None and not (
# #         args.mmlu_per_category_limit is not None or args.mmlu_category_samples
# #     ):
# #         raise ValueError(
# #             "--mmlu-sample-seed only applies with --mmlu-per-category-limit or --mmlu-category-samples."
# #         )

# #     mmlu_samples_parsed = _parse_mmlu_category_samples(args.mmlu_category_samples)

# #     queries = load_queries(
# #         dataset_paths,
# #         sample_limit=args.sample_limit,
# #         mmlu_categories=mmlu_cat,
# #         mmlu_per_category_limit=args.mmlu_per_category_limit,
# #         mmlu_category_samples=mmlu_samples_parsed,
# #         mmlu_sample_seed=args.mmlu_sample_seed,
# #     )
# #     results = run_selected(queries=queries, selection=selection)
# #     saved_paths = split_and_save_all(results, base_dir=args.results_dir)

# #     print(f"Queries loaded: {len(queries)}")
# #     print(f"Records produced: {len(results)}")
# #     print("Dataset paths used:")
# #     for ds, p in dataset_paths.items():
# #         print(f"  - {ds}: {p}")
# #     print("Saved run outputs:")
# #     for p in saved_paths:
# #         print(f"  - {p}")


# # if __name__ == "__main__":
# #     main()


