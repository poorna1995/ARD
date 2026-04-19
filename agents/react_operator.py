"""
react_operator.py
=================
Production-quality ReAct agent for four benchmarks:
  gaia | mmlu_pro | math_hard | swe_bench_verified

Design constraints
──────────────────
* O(S) time and space where S = max_steps  (no per-step copies of history)
* Tool registry built ONCE per (dataset, tool_budget) — not per query
* Single re-exported function: react_operator(...)
* Zero third-party dependencies beyond openai + sympy (already in your env)
* Temperature=0, greedy — matches paper & reproducibility requirement
* Full structured trace returned for UnifiedExperimentRecord.tool_trace

Thought → Action[tool_name: input] → Observation loop
Final answer extracted from finish[...] action or last non-empty line.
"""

from __future__ import annotations

import ast
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional
from baselines.schema import ReActStep

# ── Optional heavy deps — graceful degradation ────────────────────────────────
try:
    from sympy import N, simplify, sympify
    from sympy.parsing.latex import parse_latex
    _SYMPY_OK = True
except Exception:
    _SYMPY_OK = False

try:
    import urllib.request, urllib.parse, json as _json
    _NET_OK = True
except Exception:
    _NET_OK = False


# ══════════════════════════════════════════════════════════════════════════════
# § 1  TRACE SCHEMA
# ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class ReActStep:
#     """
#     One Thought/Action/Observation cycle.
#     Stored as a plain dataclass — cheaper than dict for large traces.
#     Serialised to dict only once at the end for JSON storage.
#     """
#     step:         int
#     thought:      str
#     action:       str        # tool name  (e.g. "calculator")
#     action_input: str        # raw string passed to the tool
#     observation:  str        # tool return value
#     error:        bool = False

#     def to_dict(self) -> dict[str, Any]:
#         return {
#             "step":         self.step,
#             "thought":      self.thought,
#             "action":       self.action,
#             "action_input": self.action_input,
#             "observation":  self.observation,
#             "error":        self.error,
#         }


# ══════════════════════════════════════════════════════════════════════════════
# § 2  TOOL IMPLEMENTATIONS  (pure functions, no closures, no globals mutated)
# ══════════════════════════════════════════════════════════════════════════════

# ── 2a  Safe Python / SymPy evaluator ─────────────────────────────────────────

_SAFE_NAMES: dict[str, Any] = {
    "abs": abs, "round": round, "min": min, "max": max,
    "sum": sum, "len": len, "range": range, "int": int, "float": float,
    "pow": pow, "divmod": divmod,
    "math": math,
    # common constants
    "pi": math.pi, "e": math.e, "inf": math.inf,
}

# Reject obviously dangerous patterns before eval
_UNSAFE_RE = re.compile(
    r"\b(import|exec|eval|open|os\.|sys\.|subprocess|__)"
)


def _tool_python(code: str) -> str:
    """
    Execute arithmetic/math Python in a restricted namespace.
    Falls back to SymPy sympify for pure expressions if exec fails.
    O(1) space — no copies of the namespace between calls.
    """
    code = code.strip()
    if _UNSAFE_RE.search(code):
        return "Error: unsafe code rejected"
    ns: dict[str, Any] = dict(_SAFE_NAMES)  # shallow copy — O(K) constant
    if _SYMPY_OK:
        try:
            from sympy import symbols, sqrt, log, exp, sin, cos, tan, pi as sym_pi
            ns.update({
                "sqrt": sqrt, "log": log, "exp": exp,
                "sin": sin, "cos": cos, "tan": tan,
                "symbols": symbols, "sympify": sympify,
                "simplify": simplify, "N": N,
            })
        except Exception:
            pass
    try:
        # Single-expression: try eval first (faster path)
        result = eval(code, {"__builtins__": {}}, ns)  # noqa: S307
        return str(result)
    except SyntaxError:
        pass
    except Exception as exc:
        # Not a syntax error — could be a multi-line block
        pass
    try:
        exec(code, {"__builtins__": {}}, ns)  # noqa: S102
        # Capture last assigned variable named 'result' or 'answer' or 'ans'
        for varname in ("result", "answer", "ans", "output"):
            if varname in ns:
                return str(ns[varname])
        # Last resort: repr of all newly-defined names
        new_keys = [k for k in ns if k not in _SAFE_NAMES and not k.startswith("_")]
        if new_keys:
            return str(ns[new_keys[-1]])
        return "executed (no output variable found)"
    except Exception as exc:
        # SymPy fallback for pure math expressions
        if _SYMPY_OK:
            try:
                expr = sympify(code)
                return str(N(expr, 6))
            except Exception:
                pass
        return f"Error: {exc}"


def _tool_sympy(expr: str) -> str:
    """Symbolic simplification and numeric evaluation."""
    if not _SYMPY_OK:
        return "Error: sympy not installed"
    expr = expr.strip()
    try:
        parsed = sympify(expr)
        simplified = simplify(parsed)
        return str(simplified)
    except Exception:
        pass
    try:
        parsed = parse_latex(expr)
        return str(simplify(parsed))
    except Exception as exc:
        return f"Error: {exc}"


# ── 2b  Wikipedia search (GAIA / general QA) ──────────────────────────────────

_WIKI_API = "https://en.wikipedia.org/w/api.php"
_WIKI_CACHE: dict[str, str] = {}   # process-level cache — avoids duplicate API calls


def _tool_wikipedia(query: str) -> str:
    """
    Wikipedia search → first paragraph of top result.
    O(1) after first call for the same query (cache hit).
    """
    if not _NET_OK:
        return "Error: network unavailable"
    q = query.strip()
    if q in _WIKI_CACHE:
        return _WIKI_CACHE[q]
    try:
        params = urllib.parse.urlencode({
            "action": "query", "list": "search",
            "srsearch": q, "format": "json",
            "srlimit": 1, "utf8": 1,
        })
        with urllib.request.urlopen(f"{_WIKI_API}?{params}", timeout=6) as r:
            data = _json.loads(r.read())
        hits = data.get("query", {}).get("search", [])
        if not hits:
            return "No Wikipedia results found."
        title = hits[0]["title"]
        # Fetch extract
        params2 = urllib.parse.urlencode({
            "action": "query", "prop": "extracts",
            "exintro": True, "explaintext": True,
            "titles": title, "format": "json", "utf8": 1,
        })
        with urllib.request.urlopen(f"{_WIKI_API}?{params2}", timeout=6) as r:
            data2 = _json.loads(r.read())
        pages = data2.get("query", {}).get("pages", {})
        extract = next(iter(pages.values()), {}).get("extract", "")
        # Return first 800 chars — enough context, avoids bloating the prompt
        result = extract[:800].strip() or f"Found article: {title}"
        _WIKI_CACHE[q] = result
        return result
    except Exception as exc:
        return f"Error: {exc}"


def _tool_wikipedia_lookup(entity: str) -> str:
    """Direct page lookup (ReAct paper's lookup[] action)."""
    return _tool_wikipedia(entity)


# ── 2c  Web search (Tavily / DuckDuckGo fallback) ─────────────────────────────

def _tool_search(query: str) -> str:
    """
    Web search with Tavily if API key present, else Wikipedia fallback.
    Falls back gracefully — no hard crash.
    """
    import os
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if tavily_key and _NET_OK:
        try:
            payload = _json.dumps({
                "api_key": tavily_key,
                "query": query.strip(),
                "search_depth": "basic",
                "max_results": 3,
            }).encode()
            req = urllib.request.Request(
                "https://api.tavily.com/search",
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=8) as r:
                data = _json.loads(r.read())
            results = data.get("results", [])
            if results:
                snippets = [
                    f"[{i+1}] {r.get('title','')}: {r.get('content','')[:300]}"
                    for i, r in enumerate(results[:3])
                ]
                return "\n".join(snippets)
        except Exception:
            pass
    # Fallback: Wikipedia
    return _tool_wikipedia(query)


# ── 2d  Bash stub (SWE-Bench — real execution happens in Docker harness) ───────

_BASH_HISTORY: list[str] = []   # module-level log (cleared per query via reset)

def _make_bash_tool(sandbox_exec: Optional[Callable[[str], str]] = None):
    """
    Returns a bash tool function.
    If sandbox_exec is provided (Docker), uses it.
    Otherwise returns a safe stub that describes what would happen.
    This keeps the ReAct loop functional even without Docker.
    """
    def _tool_bash(command: str) -> str:
        command = command.strip()
        if sandbox_exec is not None:
            try:
                return sandbox_exec(command)
            except Exception as exc:
                return f"Error: {exc}"
        # Stub mode — used when Docker is not available
        # Still useful: the LM can reason over the described output
        _BASH_HISTORY.append(command)
        return (
            f"[STUB] Would execute: {command}\n"
            "Note: Connect a Docker sandbox via sandbox_exec for real execution."
        )
    return _tool_bash


def _tool_finish(answer: str) -> str:
    """Sentinel tool — signals loop termination. Returns the answer unchanged."""
    return answer.strip()


# ══════════════════════════════════════════════════════════════════════════════
# § 3  TOOL REGISTRY  — built once per dataset, O(1) lookup
# ══════════════════════════════════════════════════════════════════════════════

# Tool = (callable, one-line description for the system prompt)
@dataclass(frozen=True)
class _ToolSpec:
    fn:          Callable[[str], str]
    description: str


# Master registry — superset of all tools
_ALL_TOOLS: dict[str, _ToolSpec] = {
    "search":     _ToolSpec(_tool_search,            "Web/Wikipedia search. Input: query string."),
    "lookup":     _ToolSpec(_tool_wikipedia_lookup,  "Wikipedia direct lookup. Input: entity name."),
    "calculator": _ToolSpec(_tool_python,            "Safe arithmetic eval. Input: Python math expression."),
    "python":     _ToolSpec(_tool_python,            "Execute Python math/logic. Input: code block."),
    "sympy":      _ToolSpec(_tool_sympy,             "Symbolic math simplification. Input: expression or LaTeX."),
    "bash":       _ToolSpec(_make_bash_tool(),       "Run shell command (sandboxed). Input: bash command."),
    "finish":     _ToolSpec(_tool_finish,            "Emit final answer and stop. Input: final answer string."),
}

# Dataset → tool names (ordered by expected frequency of use)
_DATASET_TOOLS: dict[str, tuple[str, ...]] = {
    "gaia":               ("search", "lookup", "calculator", "python", "finish"),
    "mmlu_pro":           ("calculator", "python", "sympy", "finish"),
    "math_hard":          ("python", "sympy", "calculator", "finish"),
    "swe_bench_verified": ("bash", "python", "search", "finish"),
}


def _build_tool_registry(dataset: str) -> dict[str, _ToolSpec]:
    """
    O(T) where T = tools for this dataset (≤ 5).
    Returns a frozen-equivalent dict — callers must not mutate it.
    """
    names = _DATASET_TOOLS.get(dataset, ("calculator", "search", "finish"))
    return {name: _ALL_TOOLS[name] for name in names if name in _ALL_TOOLS}


def _tool_list_str(registry: dict[str, _ToolSpec]) -> str:
    """One-line tool menu for the system prompt. O(T)."""
    return "\n".join(
        f"  {name}: {spec.description}"
        for name, spec in registry.items()
        if name != "finish"   # finish is implicit — keeps prompt shorter
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 4  PROMPT TEMPLATES  — benchmark-aware, tightly constrained
# ══════════════════════════════════════════════════════════════════════════════

# Action regex — compiled once at import time
_ACTION_RE = re.compile(
    r"Action\s*:\s*(\w+)\s*[\[\(]([^\]\)]*?)[\]\)]",
    re.DOTALL | re.IGNORECASE,
)
_THOUGHT_RE = re.compile(r"Thought\s*:\s*(.+?)(?=Action\s*:|$)", re.DOTALL | re.IGNORECASE)
_FINISH_RE  = re.compile(r"finish\s*[\[\(](.+?)[\]\)]", re.DOTALL | re.IGNORECASE)

# Maximum chars of observation fed back into context
# Keeps token count bounded — O(1) per step regardless of tool output size
_OBS_MAX_CHARS: dict[str, int] = {
    "gaia":               600,
    "mmlu_pro":           300,
    "math_hard":          400,
    "swe_bench_verified": 1200,
}
_OBS_MAX_DEFAULT = 500


def _make_system_prompt(dataset: str, tool_registry: dict[str, _ToolSpec],
                         max_steps: int, tool_budget: int) -> str:
    tool_str = _tool_list_str(tool_registry)

    # Dataset-specific answer format instruction
    answer_fmt = {
        "gaia":               "one word, number, name, date, or short phrase",
        "mmlu_pro":           "exactly one uppercase letter A–J",
        "math_hard":          "exact mathematical expression (LaTeX preferred, e.g. \\\\frac{1}{2})",
        "swe_bench_verified": "minimal unified diff patch",
    }.get(dataset, "short factual answer")

    return (
        f"You are a ReAct agent solving {dataset} benchmark tasks.\n\n"
        "## Format (STRICT — parser is regex-based)\n"
        "Each turn output EXACTLY:\n"
        "Thought: <your reasoning>\n"
        "Action: tool_name[tool input]\n\n"
        "When you have the final answer:\n"
        "Thought: I now know the final answer.\n"
        "Action: finish[<your answer>]\n\n"
        f"## Constraints\n"
        f"- Max steps: {max_steps}  |  Tool calls: {tool_budget}\n"
        f"- Final answer format: {answer_fmt}\n\n"
        f"## Available tools\n{tool_str}\n\n"
        "## Rules\n"
        "- Never output 'Observation:' — that is injected by the environment.\n"
        "- Use finish[] as soon as you are confident. Do not over-reason.\n"
        "- If a tool errors, adapt your approach rather than repeating.\n"
    )


def _make_user_prompt(query: QueryInput) -> str:  # type: ignore[name-defined]
    base = f"Task: {query.query}"
    if query.dataset == "mmlu_pro" and query.options:
        base += f"\n\nOptions:\n{query.options}"
    meta = query.metadata or {}
    if query.dataset == "gaia" and meta.get("file_name"):
        base += f"\n\n[Attached file: {meta['file_name']}]"
    if query.dataset == "math_hard" and meta.get("level"):
        base += f"\n[Difficulty: {meta['level']}]"
    return base


# ══════════════════════════════════════════════════════════════════════════════
# § 5  CORE REACT LOOP  — O(S) time, O(S) space
# ══════════════════════════════════════════════════════════════════════════════

def _parse_action(text: str) -> tuple[str, str]:
    """
    Extract (tool_name, tool_input) from model output.
    Returns ("finish", extracted_answer) as fallback if action not found.
    O(len(text)) — single regex scan.
    """
    # Check for finish inline first (common pattern)
    m = _FINISH_RE.search(text)
    if m:
        return "finish", m.group(1).strip()

    m = _ACTION_RE.search(text)
    if m:
        return m.group(1).strip().lower(), m.group(2).strip()

    # No action found — extract last non-empty line as implicit finish
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return "finish", (lines[-1] if lines else "")


def _parse_thought(text: str) -> str:
    """Extract Thought content. O(len(text))."""
    m = _THOUGHT_RE.search(text)
    if m:
        thought = m.group(1).strip()
        # Truncate at Action: boundary if regex was greedy
        if "\nAction" in thought:
            thought = thought[:thought.index("\nAction")].strip()
        return thought
    return ""


def _truncate_observation(obs: str, dataset: str) -> str:
    """
    Hard-cap observation length to prevent context blowout.
    O(1) after the slice.
    """
    limit = _OBS_MAX_CHARS.get(dataset, _OBS_MAX_DEFAULT)
    if len(obs) <= limit:
        return obs
    return obs[:limit] + f"… [truncated to {limit} chars]"


# ── QueryInput forward reference (imported at call site to avoid circular) ───

def react_operator(
    *,
    client: Any,                  # openai.OpenAI instance
    model_name: str,
    question: str,                # raw question string  (legacy compat)
    max_steps: int        = 6,
    tool_budget: int      = 4,
    max_tokens: int       = 512,
    temperature: float    = 0.0,
    seed: Optional[int]   = None,
    # Extended API — used when called from run_selected with full QueryInput
    query_input: Any      = None,  # Optional[QueryInput]
    dataset: str          = "gaia",
    sandbox_exec: Optional[Callable[[str], str]] = None,
) -> tuple[str, dict[str, int], Optional[str], float, list[dict[str, Any]], str, str]:
    """
    Execute ReAct loop for a single query.

    Parameters
    ──────────
    client       : OpenAI client
    model_name   : e.g. "gpt-4o-mini"
    question     : raw question text (used if query_input is None)
    max_steps    : hard loop ceiling  (default 6 — matches RunSelection default)
    max_tokens   : per-step generation budget
    temperature  : 0.0 = greedy (reproducible)
    query_input  : full QueryInput — preferred; enables dataset-aware tools/prompts
    dataset      : fallback dataset name when query_input is None
    sandbox_exec : optional Docker exec callable for SWE-Bench

    Returns
    ───────
    (raw_output, usage_dict, error_str, elapsed_s, trace_list, system_prompt, user_prompt)
      raw_output     : full concatenated model turns (for reasoning_trace)
      usage_dict     : {"prompt_tokens":…, "completion_tokens":…, "total_tokens":…}
      error_str      : None on success, message on exception
      elapsed_s      : wall-clock seconds
      trace_list     : list of ReActStep.to_dict() — for tool_trace JSON
      system_prompt  : initial system message sent to the model
      user_prompt    : initial user message (benchmark-specific task text)
    """
    t0 = time.perf_counter()

    # ── Resolve dataset and query ─────────────────────────────────────────────
    if query_input is not None:
        ds   = query_input.dataset
        q_text = _make_user_prompt(query_input)
    else:
        ds     = dataset
        q_text = question

    # ── Build tool registry (O(T) constant) ───────────────────────────────────
    registry = _build_tool_registry(ds)

    # Swap in real bash tool if sandbox provided
    if sandbox_exec is not None and "bash" in registry:
        registry = dict(registry)
        registry["bash"] = _ToolSpec(_make_bash_tool(sandbox_exec), registry["bash"].description)

    # Effective tool budget cannot exceed loop steps or available tools.
    effective_budget = max(1, min(tool_budget, max_steps, len(registry)))

    # ── Build prompts ─────────────────────────────────────────────────────────
    system_prompt = _make_system_prompt(ds, registry, max_steps, effective_budget)

    # ── Message history — list of dicts, appended in-place ───────────────────
    # Pre-allocate estimate: 2*max_steps+1 messages
    messages: list[dict[str, str]] = [
        {"role": "system",  "content": system_prompt},
        {"role": "user",    "content": q_text},
    ]

    # ── Accumulator state ─────────────────────────────────────────────────────
    trace:              list[ReActStep] = []
    raw_turns:          list[str]       = []
    total_prompt       = 0
    total_completion   = 0
    tool_calls_used    = 0
    final_answer       = ""
    error_message: Optional[str] = None

    # ── Main loop ─────────────────────────────────────────────────────────────
    for step in range(max_steps):
        try:
            create_kw: dict[str, Any] = {
                "model": model_name,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if seed is not None:
                create_kw["seed"] = seed
            resp = client.chat.completions.create(**create_kw)
        except Exception as exc:
            error_message = f"LLM call failed at step {step}: {exc}"
            break

        # Token accounting
        usage = resp.usage
        if usage:
            total_prompt     += usage.prompt_tokens
            total_completion += usage.completion_tokens

        model_raw = resp.choices[0].message.content or ""
        model_text = model_raw.strip()
        raw_turns.append(model_raw)
        messages.append({"role": "assistant", "content": model_text})

        # ── Parse action ──────────────────────────────────────────────────────
        thought    = _parse_thought(model_text)
        tool_name, tool_input = _parse_action(model_text)

        # ── Dispatch ──────────────────────────────────────────────────────────
        is_finish   = (tool_name == "finish")
        is_error    = False

        if is_finish:
            final_answer = tool_input
            trace.append(ReActStep(
                step=step, thought=thought,
                action="finish", action_input=tool_input,
                observation=tool_input, error=False,
            ))
            break

        # Check budget before calling
        if tool_calls_used >= effective_budget:
            # Force finish — budget exhausted
            final_answer = tool_input or _extract_fallback(raw_turns, ds)
            trace.append(ReActStep(
                step=step, thought="Tool budget exhausted — returning best answer.",
                action="finish", action_input=final_answer,
                observation=final_answer, error=False,
            ))
            break

        # Look up and call tool
        if tool_name in registry:
            try:
                observation = registry[tool_name].fn(tool_input)
                tool_calls_used += 1
            except Exception as exc:
                observation = f"Tool error: {exc}"
                is_error    = True
        else:
            # Unknown tool — tell the model which tools exist
            observation = (
                f"Unknown tool '{tool_name}'. "
                f"Available: {', '.join(registry.keys())}"
            )
            is_error = True

        obs_trimmed = _truncate_observation(observation, ds)

        trace.append(ReActStep(
            step=step, thought=thought,
            action=tool_name, action_input=tool_input,
            observation=obs_trimmed, error=is_error,
        ))

        # Inject observation — single string append, no copy of history
        obs_msg = f"Observation: {obs_trimmed}"
        messages.append({"role": "user", "content": obs_msg})

    else:
        # Loop exhausted without finish — extract best available answer
        if not final_answer:
            final_answer = _extract_fallback(raw_turns, ds)
            error_message = error_message or "MAX_STEPS_EXCEEDED"

    elapsed = time.perf_counter() - t0

    usage_dict = {
        "prompt_tokens":     total_prompt,
        "completion_tokens": total_completion,
        "total_tokens":      total_prompt + total_completion,
    }

    raw_output = "\n\n".join(raw_turns)

    # Serialise trace once — O(S)
    trace_list = [s.to_dict() for s in trace]

    return (
        raw_output,
        usage_dict,
        error_message,
        round(elapsed, 3),
        trace_list,
        system_prompt,
        q_text,
    )


# ══════════════════════════════════════════════════════════════════════════════
# § 6  FALLBACK ANSWER EXTRACTOR
# ══════════════════════════════════════════════════════════════════════════════

_ANSWER_PATTERNS = (
    re.compile(r"[Ff]inal [Aa]nswer\s*:?\s*(.+?)$",   re.MULTILINE),
    re.compile(r"[Tt]he answer is\s*:?\s*(.+?)$",      re.MULTILINE),
    re.compile(r"\\boxed\{([^}]+)\}"),
    re.compile(r"[Aa]nswer\s*:\s*([A-Ja-j])\b"),       # MMLU letter
)


def _extract_fallback(raw_turns: list[str], dataset: str) -> str:
    """
    Best-effort answer extraction when finish[] was never called.
    Scans turns in reverse — most recent output is most reliable.
    O(S * len(last_turn)).
    """
    for turn in reversed(raw_turns):
        for pat in _ANSWER_PATTERNS:
            m = pat.search(turn)
            if m:
                return m.group(1).strip().rstrip(".,;:!")
        # Last non-empty line fallback
        lines = [ln.strip() for ln in turn.splitlines() if ln.strip()]
        if lines:
            candidate = lines[-1].rstrip(".,;:!")
            # Reject meta-commentary lines
            if not any(w in candidate.lower() for w in
                       ("let me", "i need to", "i'll", "i will", "step")):
                return candidate
    return ""