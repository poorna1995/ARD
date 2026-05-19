
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Callable, Optional

from dotenv import load_dotenv

from agent.base import BaseAgent, AgentResponse
from agent.dataset_profile import apply_react_profile, extract_hop_metadata
from agent.tools import (
    arxiv_search,
    github_search,
    math_tool,
    pdb_parse,
    web_fetch,
    web_search,
    wikipedia_search,
)
from agent.tools.decorator import tool
from evaluator.parse import parse_llm_output
from prompts.prompts import SYSTEM_PROMPT, USER_PROMPT

AGENT_ID = "react_004"

load_dotenv()


# ── Constants ──────────────────────────────────────────────────────────────────

# Maximum number of consecutive invalid-finish-payload retries before the
# step budget is no longer spent on fixing the payload.
DEFAULT_FINISH_RETRY_LIMIT: int = 3

# Retry attempts for transient LLM API errors (429, 500, network hiccups).
LLM_RETRY_ATTEMPTS: int = 3
LLM_RETRY_BASE_DELAY: float = 1.0   # seconds; doubles on each attempt


# ── Finish tool ────────────────────────────────────────────────────────────────

@tool(
    "finish",
    'Submit the final answer. Input: JSON string '
    '{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}.',
)
def finish(answer: str) -> str:
    return answer


# ── Tool aliases ───────────────────────────────────────────────────────────────
# FIX Warn 3: aliases live here only — _call_tool applies them at dispatch time.
# The manual registry["wikipedia"] / registry["wiki"] entries in the old code
# have been removed to eliminate duplicate logic.

TOOL_ALIASES: dict[str, str] = {
    "wikipedia":        "wikipedia_search",
    "wiki":             "wikipedia_search",
    "wikipedia_search": "wikipedia_search",
}


# ── Tool registry ──────────────────────────────────────────────────────────────

def _build_tools_registry() -> dict[str, Callable]:
    """Return a normalised (lowercase-keyed) registry of all available tools."""
    fns: list[Callable] = [
        web_search, web_fetch, wikipedia_search,
        arxiv_search, github_search, pdb_parse, math_tool, finish,
    ]
    try:
        from agent.tools.readfile import read_file
        fns.insert(-1, read_file)
    except ImportError:
        pass

    # FIX Warn 3: no manual alias entries here; aliases resolved at call time.
    return {fn._tool_name: fn for fn in fns}


TOOLS: dict[str, Callable] = _build_tools_registry()

# Per-dataset allowlists — must match prompts/prompts.py tool guidance.
# Unknown datasets keep the full registry (backward compatible).
_DATASET_REACT_TOOL_ALLOWLIST: dict[str, frozenset[str]] = {
    "math": frozenset({"math_tool", "finish"}),
    "hotpot": frozenset({"wikipedia_search", "finish"}),
    "musique": frozenset({"wikipedia_search", "finish"}),
    "gaia": frozenset({
        "read_file",
        "web_search",
        "web_fetch",
        "arxiv_search",
        "github_search",
        "pdb_parse",
        "finish",
        
    }),
    # mmlu react prompts use pseudo reason[] steps; only finish is a real tool.
    "mmlu_pro": frozenset({"finish"}),
}


def tools_for_dataset(
    dataset: str,
    *,
    registry: dict[str, Callable] | None = None,
) -> dict[str, Callable]:
    """Return the ReAct tool registry permitted for *dataset*."""
    pool = registry or TOOLS
    key = (dataset or "").strip().lower()
    allow = _DATASET_REACT_TOOL_ALLOWLIST.get(key)
    if allow is None:
        return {k.lower(): v for k, v in pool.items()}
    return {
        k.lower(): v
        for k, v in pool.items()
        if k.lower() in allow
    }


def react_tool_names_for_dataset(dataset: str) -> list[str]:
    """Sorted tool names for *dataset*, excluding ``finish`` (planner metadata)."""
    return sorted(
        n for n in tools_for_dataset(dataset).keys() if n != "finish"
    )


# ── ReActStep ──────────────────────────────────────────────────────────────────

class ReActStep:
    """Immutable record of one Thought → Action → Observation cycle."""

    __slots__ = ("step_num", "thought", "action", "action_input", "observation")

    def __init__(
        self,
        step_num: int,
        thought: str,
        action: str,
        action_input: str,
        observation: str = "",
    ) -> None:
        self.step_num     = step_num
        self.thought      = thought
        self.action       = action
        self.action_input = action_input
        self.observation  = observation

    def __repr__(self) -> str:
        return (
            f"[Step {self.step_num}]\n"
            f"  Thought     : {self.thought}\n"
            f"  Action      : {self.action}[{self.action_input}]\n"
            f"  Observation : {self.observation}\n"
        )


# ── ReActParser ────────────────────────────────────────────────────────────────

class ReActParser:
    """Parse LLM output into (thought, action_name, action_input).

    Canonical format::

        Thought: <text>
        Action: tool_name[input]

    Falls back gracefully to "finish" when no action is detected.
    """

    THOUGHT_RE = re.compile(
        r"Thought\s*:\s*(.+?)(?=\nAction\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    ACTION_BRACKET_RE = re.compile(
        r"Action\s*:\s*([A-Za-z_]\w*)\s*\[([^\]]*)\]",
        re.DOTALL | re.IGNORECASE,
    )
    # FIX Warn 4: capped at 500 chars to prevent greedy over-consumption of
    # multi-line inputs that would swallow subsequent Thought/Observation blocks.
    ACTION_SPLIT_RE = re.compile(
        r"Action\s*:\s*([A-Za-z_]\w*)\s*\n+\s*(?:Action\s+)?Input\s*:\s*"
        r"(.{1,500}?)(?=\nThought|\nObservation|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    FINISH_RE = re.compile(r"\bFinish\s*\[([^\]]*)\]", re.DOTALL | re.IGNORECASE)

    @classmethod
    def parse(cls, text: str) -> tuple[str, str, str]:
        """Return ``(thought, action_name, action_input)``.

        ``action_name`` is always lowercase.
        """
        thought = ""
        m = cls.THOUGHT_RE.search(text)
        if m:
            thought = m.group(1).strip()

        for pattern in (cls.ACTION_BRACKET_RE, cls.ACTION_SPLIT_RE):
            m = pattern.search(text)
            if m:
                return thought, m.group(1).strip().lower(), m.group(2).strip()

        m = cls.FINISH_RE.search(text)
        if m:
            return thought, "finish", m.group(1).strip()

        return thought, "finish", text.strip()

    @classmethod
    def extract_final_answer(cls, raw: str) -> tuple[str, Optional[float], Optional[float]]:
        return parse_llm_output(raw)


# ── Fallback-answer guards ─────────────────────────────────────────────────────

def _looks_like_raw_tool_output(s: str) -> bool:
    t   = (s or "").strip()
    low = t.lower()
    return bool(
        len(t) >= 4 and (
            "search failed" in low[:160]
            or low.startswith("no results found")
            or "\nurl     :" in t
            or "\nsnippet :" in t
            or (t.startswith("[") and "]" in t[:160])
            or (t.startswith("{") and any(k in t[:1200] for k in ('"highlights"', '"error"', '"ok"')))
            or "error: unknown tool" in low[:120]
            or "invalid finish payload" in low[:200]
        )
    )


def _reject_fallback_answer(fa: str, raw_llm: str) -> bool:
    """Return True when ``fa`` should be discarded as a fallback answer.

    FIX Bug 3: scratchpad markers (Thought / Action) now cause rejection
    regardless of string length, closing the <600-char loophole in the
    original check.
    """
    ft  = (fa or "").strip()
    rl  = (raw_llm or "").strip()
    rl_lower = rl.lower()

    if not ft:
        return True
    if bool(rl) and ft == rl:
        return True
    if _looks_like_raw_tool_output(fa):
        return True
    # Reject whenever scratchpad structure is present — length no longer matters.
    if "thought:" in rl_lower and "action:" in rl_lower:
        return True
    return False


# ── ReactAgent ─────────────────────────────────────────────────────────────────

class ReactAgent(BaseAgent):
    """ReAct agent that interleaves Thought / Action / Observation.

    Design notes
    ────────────
    • Scratchpad is sent as alternating assistant / user messages (correct
      multi-turn structure).
    • Tool names are normalised to lowercase at registration and dispatch time.
    • Parser enforces one canonical format: ``Action: name[input]``.
    • System prompt is the first ``messages`` entry with ``role: system``.
    • Runtime inputs are shared via **kwargs (e.g. ``expected_answer``).

    Contracts from BaseAgent (must be implemented there)
    ─────────────────────────────────────────────────────
    • ``_format_user_prompt(query, **kwargs) -> str``
    • ``_expected_answer(kwargs) -> Any``
    • ``_core_response(...) -> AgentResponse``
    • ``_get_client() -> openai.OpenAI``  (or compatible)
    """

    def __init__(
        self,
        model:               str,
        dataset:             str,
        tools:               dict[str, Callable] | None = None,
        max_steps:           int | None = None,
        finish_retry_limit:  int = DEFAULT_FINISH_RETRY_LIMIT,
        **kwargs:            Any,
    ) -> None:
        cfg = self._normalize_config(
            model=model, dataset=dataset, kwargs=kwargs, strategy="react",
        ).config

        n_hops, _hop_name = extract_hop_metadata(kwargs)
        profiled = apply_react_profile(
            cfg.dataset, cfg.agent_params, n_hops=n_hops,
        )

        self.dataset = cfg.dataset
        if max_steps is not None:
            self.max_steps = int(max_steps)
        elif "max_steps" in cfg.agent_params:
            self.max_steps = int(cfg.agent_params["max_steps"])
        else:
            self.max_steps = int(profiled.get("max_steps", 12))
        self.finish_retry_limit   = finish_retry_limit
        pool = tools if tools is not None else tools_for_dataset(self.dataset)
        self.tools                = {k.lower(): v for k, v in pool.items()}
        self.parser               = ReActParser()

        react_system = SYSTEM_PROMPT[self.dataset]["react"]
        if "{tools_block}" in react_system:
            react_system = react_system.replace("{tools_block}", self._build_tools_block())

        super().__init__(
            model         = cfg.model,
            system_prompt = react_system,
            user_prompt   = USER_PROMPT[self.dataset]["react"],
            temperature   = cfg.temperature,
            max_tokens    = cfg.max_tokens,
            seed          = cfg.seed,
        )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _build_tools_block(self) -> str:
        return "\n".join(
            f"  {name:15s}: {getattr(fn, '_tool_description', 'No description.')}"
            for name, fn in self.tools.items()
        )

    def _call_tool(self, action: str, action_input: str) -> tuple[str, float]:
        """Dispatch ``action`` to the registered tool and return ``(observation, elapsed)``."""
        start = time.perf_counter()
        # FIX Warn 3: aliases resolved here; no duplicate registry entries.
        key = TOOL_ALIASES.get(action.strip().lower(), action.strip().lower())
        fn  = self.tools.get(key)

        if fn is None:
            obs = (
                f"Error: unknown tool '{action}'. "
                f"Available tools: {', '.join(sorted(self.tools))}. "
                "Fix the Action name and try again."
            )
        else:
            try:
                obs = fn(action_input)
            except Exception as exc:
                obs = f"Tool '{action}' raised an error: {exc}. Try a different approach."

        return str(obs), time.perf_counter() - start

    def _validate_finish_payload(self, raw: str) -> tuple[bool, str]:
        """Return ``(is_valid, reason)`` for a finish JSON payload."""
        text = (raw or "").strip()
        if not text:
            return False, "empty Finish payload"

        try:
            obj = json.loads(text)
        except Exception:
            return False, "Finish payload must be valid JSON"

        if not isinstance(obj, dict):
            return False, "Finish payload must be a JSON object"

        missing = [k for k in ("answer", "confidence", "complexity") if k not in obj]
        if missing:
            return False, f"missing keys: {missing}"

        if not str(obj.get("answer", "")).strip():
            return False, "answer must be non-empty"

        def _to_score(v: Any) -> float:
            return float(v) if isinstance(v, (int, float)) else float(str(v).strip())

        try:
            conf = _to_score(obj["confidence"])
            comp = _to_score(obj["complexity"])
        except Exception:
            return False, "confidence/complexity must be numeric"

        if not (0.0 <= conf <= 1.0 and 0.0 <= comp <= 1.0):
            return False, "confidence/complexity must be in [0, 1]"

        return True, ""

    # ── Message builder ────────────────────────────────────────────────────────

    def _build_messages(self, query: str, steps: list[ReActStep], **kwargs: Any) -> list[dict]:
        """Build system + user + alternating assistant/user scratchpad messages."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": self._format_user_prompt(query, **kwargs)},
        ]
        for step in steps:
            messages += [
                {
                    "role":    "assistant",
                    "content": f"Thought: {step.thought}\nAction: {step.action}[{step.action_input}]",
                },
                {
                    "role":    "user",
                    "content": f"Observation: {step.observation}",
                },
            ]
        return messages

    # ── LLM call (with retry) ──────────────────────────────────────────────────

    def _call_llm(self, query: str, steps: list[ReActStep], **kwargs: Any) -> tuple[str, float, Any]:
        """Call the LLM with exponential-backoff retries on transient errors."""
        messages = self._build_messages(query, steps, **kwargs)
        last_exc: Exception | None = None

        for attempt in range(LLM_RETRY_ATTEMPTS):
            try:
                start    = time.perf_counter()
                response = self._get_client().chat.completions.create(
                    model       = self.model,
                    messages    = messages,
                    temperature = self.temperature,
                    max_tokens  = self.max_tokens,
                    seed        = self.seed,
                )
                return response.choices[0].message.content or "", time.perf_counter() - start, response
            except Exception as exc:
                last_exc = exc
                if attempt < LLM_RETRY_ATTEMPTS - 1:
                    time.sleep(LLM_RETRY_BASE_DELAY * (2 ** attempt))

        raise last_exc  # re-raise after all attempts exhausted

    # ── Main loop ─────────────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        steps:         list[ReActStep] = []
        tools_called:  list[str]       = []
        tools_results: list[dict]      = []

        total_latency_llm       = 0.0
        total_latency_tool      = 0.0
        total_prompt_tokens     = 0
        total_completion_tokens = 0
        num_llm_calls           = 0

        final_answer:      str             = ""
        finish_confidence: Optional[float] = None
        finish_complexity: Optional[float] = None
        is_stopped_early  = False
        error:             Optional[str]   = None
        is_failed         = False
        last_llm_out      = ""

        # FIX Warn 1: track invalid-finish retries independently of step budget.
        finish_retry_count = 0

        t0       = time.perf_counter()
        expected = self._expected_answer(kwargs)

        for step_num in range(1, self.max_steps + 1):

            # ── 1. LLM call ───────────────────────────────────────────────────
            try:
                llm_out, llm_latency, response = self._call_llm(query, steps, **kwargs)
                last_llm_out = llm_out or ""
            except Exception as exc:
                error, is_failed = str(exc), True
                break

            total_latency_llm       += llm_latency
            total_prompt_tokens     += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            num_llm_calls           += 1

            # ── 2. Parse ──────────────────────────────────────────────────────
            thought, action, action_input = self.parser.parse(llm_out)

            # ── 3. Finish? ────────────────────────────────────────────────────
            if action == "finish":
                valid, reason = self._validate_finish_payload(action_input)

                if not valid:
                    # FIX Warn 1: stop retrying if the per-finish retry cap is hit.
                    finish_retry_count += 1
                    if finish_retry_count >= self.finish_retry_limit:
                        error    = f"Finish payload invalid after {finish_retry_count} retries: {reason}"
                        is_failed = True
                        break

                    observation = (
                        f"Error: invalid Finish payload. {reason}. "
                        'Return only Finish[{"answer":"...","confidence":0.0,"complexity":0.0}]'
                    )
                    tools_called.append("finish")
                    tools_results.append({
                        "step": step_num, "action": "finish",
                        "input": action_input, "observation": observation,
                    })
                    steps.append(ReActStep(step_num, thought, "finish", action_input, observation))
                    continue

                # FIX Bug 2: redundant _call_tool("finish", ...) removed.
                # finish() just echoed its input; extract_final_answer reads
                # action_input directly, so the tool call added only noise.
                tools_called.append("finish")
                tools_results.append({
                    "step": step_num, "action": "finish",
                    "input": action_input, "observation": action_input,
                })
                steps.append(ReActStep(step_num, thought, "finish", action_input, action_input))
                final_answer, finish_confidence, finish_complexity = (
                    self.parser.extract_final_answer(action_input)
                )
                break

            # ── 4. Execute tool ───────────────────────────────────────────────
            observation, tool_latency = self._call_tool(action, action_input)
            total_latency_tool += tool_latency
            tools_called.append(action)
            tools_results.append({
                "step": step_num, "action": action,
                "input": action_input, "observation": observation,
            })
            steps.append(ReActStep(step_num, thought, action, action_input, observation))

        else:
            # max_steps exhausted without a valid Finish.
            is_stopped_early = True
            if last_llm_out.strip():
                fa, fc, fm = self.parser.extract_final_answer(last_llm_out)
                # FIX Bug 3: _reject_fallback_answer now rejects scratchpad text
                # regardless of length (no >600 char loophole).
                if fa and not _reject_fallback_answer(fa, last_llm_out):
                    final_answer, finish_confidence, finish_complexity = fa, fc, fm

        return self._core_response(
            query               = query,
            agent               = "react",
            agent_id            = AGENT_ID,
            predicted_answer    = final_answer,
            latency_total       = time.perf_counter() - t0,
            latency_llm         = total_latency_llm,
            expected_answer     = expected,
            is_failed           = is_failed,
            error               = error,
            prompt_tokens       = total_prompt_tokens,
            completion_tokens   = total_completion_tokens,
            confidence          = finish_confidence,
            complexity          = finish_complexity,
            finalize            = not is_failed,
            latency_tools       = total_latency_tool,
            reasoning_steps     = [f"[{s.step_num}] {s.thought}" for s in steps],
            num_llm_calls       = num_llm_calls,
            num_steps           = len(steps),
            tools_available     = list(self.tools.keys()),
            tools_called        = tools_called,
            tools_results       = tools_results,
            num_tool_calls      = len(tools_called),
            max_steps           = self.max_steps,
            steps_taken         = len(steps),
            is_stopped_early    = is_stopped_early,
        )


# ── Public entrypoint ──────────────────────────────────────────────────────────

def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
    return ReactAgent(model=model, dataset=dataset, **kwargs).run(query=query, **kwargs)



if __name__ == "__main__":
    import pandas as pd
 
    dataset = "gaia"
    model   = "gpt-4o-mini"
    df      = pd.read_parquet(f"datasets/golden/{dataset}.parquet", columns=["query", "answer"])
 
    for _, row in df.iterrows():
        resp = run(
            query           = str(row["query"]),
            model           = model,
            dataset         = dataset,
            expected_answer = row.get("answer"),
        )
        print(resp.agent_id, resp.predicted_answer, resp.latency_total, resp.is_failed)


# from __future__ import annotations
 
# import json
# import os
# import re
# import time
# from typing import Any, Callable, Optional
 
# from dotenv import load_dotenv
 
# from agent.base import BaseAgent, AgentResponse
# from agent.tools import (
#     arxiv_search,
#     github_search,
#     math_tool,
#     pdb_parse,
#     web_fetch,
#     web_search,
#     wikipedia_search,
# )
# from agent.tools.decorator import tool
# from evaluator.eval import normalise_answer
# from prompts.prompts import SYSTEM_PROMPT, USER_PROMPT

# AGENT_ID = "react_004"

# load_dotenv()
 
 
# # ── Built-in finish tool ───────────────────────────────────────────────────────
 
# @tool(
#     "finish",
#     'Submit the final answer. Input: JSON string '
#     '{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}.',
# )
# def finish(answer: str) -> str:
#     return answer

# # Extra aliases resolved before registry lookup (both names are also registered).
# TOOL_ALIASES: dict[str, str] = {
#     "wikipedia": "wikipedia_search",
#     "wiki": "wikipedia_search",
#     "wikipedia_search": "wikipedia_search",
# }

# def _optional_read_file() -> Callable | None:
#     try:
#         from agent.tools.readfile import read_file

#         return read_file
#     except ImportError:
#         return None


# def _build_tools_registry() -> dict[str, Callable]:
#     """Canonical tool names plus legacy aliases (prompts may use either)."""
#     fns: list[Callable] = [
#         web_search,
#         web_fetch,
#         wikipedia_search,
#         arxiv_search,
#         github_search,
#         pdb_parse,
#         math_tool,
#         finish,
#     ]
#     rf = _optional_read_file()
#     if rf is not None:
#         fns.insert(-1, rf)
#     registry: dict[str, Callable] = {fn._tool_name: fn for fn in fns}
#     # LLM often emits wikipedia[...] while @tool name is wikipedia_search.
#     registry["wikipedia"] = wikipedia_search
#     registry["wiki"] = wikipedia_search
#     return registry


# TOOLS: dict[str, Callable] = _build_tools_registry()
 
# class ReActStep:
#     __slots__ = ("step_num", "thought", "action", "action_input", "observation")
 
#     def __init__(
#         self,
#         step_num: int,
#         thought: str,
#         action: str,
#         action_input: str,
#         observation: str = "",
#     ) -> None:
#         self.step_num     = step_num
#         self.thought      = thought
#         self.action       = action
#         self.action_input = action_input
#         self.observation  = observation
 
#     def __repr__(self) -> str:
#         return (
#             f"[Step {self.step_num}]\n"
#             f"  Thought     : {self.thought}\n"
#             f"  Action      : {self.action}[{self.action_input}]\n"
#             f"  Observation : {self.observation}\n"
#         )
 
 
# # ── ReActParser ────────────────────────────────────────────────────────────────
 
# class ReActParser:
#     """Parse LLM output into (thought, action_name, action_input).
 
#     Canonical format:
#         Thought: <text>
#         Action: tool_name[input]
 
#     Falls back gracefully to "finish" when no action is detected.
#     """
 
#     # Everything after "Thought:" up to the next "Action:" label
#     THOUGHT_RE = re.compile(
#         r"Thought\s*:\s*(.+?)(?=\nAction\s*:|\Z)",
#         re.DOTALL | re.IGNORECASE,
#     )
#     # Primary:  Action: tool_name[input]
#     ACTION_BRACKET_RE = re.compile(
#         r"Action\s*:\s*([A-Za-z_]\w*)\s*\[([^\]]*)\]",
#         re.DOTALL | re.IGNORECASE,
#     )
#     # Fallback: Action: tool_name\nAction Input: input
#     ACTION_SPLIT_RE = re.compile(
#         r"Action\s*:\s*([A-Za-z_]\w*)\s*\n+\s*(?:Action\s+)?Input\s*:\s*"
#         r"(.+?)(?=\nThought|\nObservation|\Z)",
#         re.DOTALL | re.IGNORECASE,
#     )
#     # Bare Finish[...] shortcut
#     FINISH_RE = re.compile(r"\bFinish\s*\[([^\]]*)\]", re.DOTALL | re.IGNORECASE)
 
#     @classmethod
#     def parse(cls, text: str) -> tuple[str, str, str]:
#         """Return (thought, action_name, action_input). action_name is lowercase."""
#         thought = ""
#         m = cls.THOUGHT_RE.search(text)
#         if m:
#             thought = m.group(1).strip()
 
#         m = cls.ACTION_BRACKET_RE.search(text)
#         if m:
#             return thought, m.group(1).strip().lower(), m.group(2).strip()
 
#         m = cls.ACTION_SPLIT_RE.search(text)
#         if m:
#             return thought, m.group(1).strip().lower(), m.group(2).strip()
 
#         m = cls.FINISH_RE.search(text)
#         if m:
#             return thought, "finish", m.group(1).strip()
 
#         # Nothing matched — treat full output as final answer
#         return thought, "finish", text.strip()
 
#     @classmethod
#     def extract_final_answer(
#         cls, raw: str
#     ) -> tuple[str, Optional[float], Optional[float]]:
#         """Delegates to evaluator.eval.normalise_answer (shared with all agents)."""
#         return normalise_answer(raw)
 
 
# # ── Fallback-answer guards ─────────────────────────────────────────────────────
 
# def _looks_like_raw_tool_output(s: str) -> bool:
#     """True when the string is obviously a tool observation, not a short QA answer."""
#     t = (s or "").strip()
#     if len(t) < 4:
#         return False
#     low = t.lower()
#     if "search failed" in low[:160] or low.startswith("no results found"):
#         return True
#     if "\nurl     :" in t or "\nsnippet :" in t:
#         return True
#     if t.startswith("[") and "]" in t[:160]:
#         return True
#     if t.startswith("{") and any(
#         k in t[:1200] for k in ('"highlights"', '"error"', '"ok"')
#     ):
#         return True
#     if "error: unknown tool" in low[:120]:
#         return True
#     if "invalid finish payload" in low[:200]:
#         return True
#     return False
 
 
# def _reject_fallback_answer(fa: str, raw_llm: str) -> bool:
#     """True if the extracted answer should NOT be used when finish never succeeded."""
#     if _looks_like_raw_tool_output(fa):
#         return True
#     if not (fa or "").strip():
#         return True
#     ft = fa.strip()
#     rl = (raw_llm or "").strip()
#     if rl and ft == rl:
#         return True
#     if len(ft) > 600 and "thought:" in rl.lower() and "action:" in rl.lower():
#         return True
#     return False
 
 
# # ── ReactAgent ─────────────────────────────────────────────────────────────────
 
# class ReactAgent(BaseAgent):
#     """ReAct agent that interleaves Thought / Action / Observation.
 
#     Design notes
#     ────────────
#     • The scratchpad is sent as alternating assistant/user messages so the LLM
#       sees correct conversational structure.
#     • Tool names are normalised to lowercase at registration *and* dispatch time,
#       preventing case-mismatch misses.
#     • The parser enforces one canonical format:  Action: name[input]
#     • Runtime inputs are shared via **kwargs (e.g., expected_answer).
#     """
 
#     def __init__(
#         self,
#         model:    str,
#         dataset:  str,
#         tools:    dict[str, Callable] | None = None,
#         max_steps: int = 12,
#         **kwargs: Any,
#     ) -> None:
#         cfg = self._normalize_config(
#             model=model, dataset=dataset, kwargs=kwargs, strategy="react",
#         ).config
#         self.dataset = cfg.dataset
#         if max_steps is not None:
#             self.max_steps = int(max_steps)
#         else:
#             self.max_steps = int(cfg.agent_params.get("max_steps", 12))
#         # Normalise all tool keys to lowercase at registration time
#         raw_tools  = tools or TOOLS
#         self.tools = {k.lower(): v for k, v in raw_tools.items()}
#         self.parser = ReActParser()

#         react_system = SYSTEM_PROMPT[self.dataset]["react"]
#         # Only substitute tools_block — do not use .format() (prompts contain JSON braces).
#         if "{tools_block}" in react_system:
#             react_system = react_system.replace(
#                 "{tools_block}", self._build_tools_block()
#             )

#         super().__init__(
#             model         = cfg.model,
#             system_prompt = react_system,
#             user_prompt   = USER_PROMPT[self.dataset]["react"],
#             temperature   = cfg.temperature,
#             max_tokens    = cfg.max_tokens,
#             seed          = cfg.seed,
#         )
 
#     # ── Helpers ────────────────────────────────────────────────────────────────
 
#     def _build_tools_block(self) -> str:
#         return "\n".join(
#             f"  {name:15s}: {getattr(fn, '_tool_description', 'No description.')}"
#             for name, fn in self.tools.items()
#         )

#     def _call_tool(self, action: str, action_input: str) -> tuple[str, float]:
#         """Dispatch to a registered tool. Returns (observation, wall_clock_seconds)."""
#         start = time.perf_counter()
#         key   = action.strip().lower()
#         key   = TOOL_ALIASES.get(key, key)
#         fn    = self.tools.get(key)

#         if fn is None:
#             obs = (
#                 f"Error: unknown tool '{action}'. "
#                 f"Available tools: {', '.join(sorted(self.tools))}. "
#                 "Fix the Action name and try again."
#             )
#         else:
#             try:
#                 obs = fn(action_input)
#             except Exception as exc:
#                 obs = f"Tool '{action}' raised an error: {exc}. Try a different approach."

#         return str(obs), time.perf_counter() - start

#     def _validate_finish_payload(self, raw: str) -> tuple[bool, str]:
#         """Validate Finish[...] payload: JSON with answer, confidence, complexity."""
#         text = (raw or "").strip()
#         if not text:
#             return False, "empty Finish payload"

#         try:
#             obj = json.loads(text)
#         except Exception:
#             return False, "Finish payload must be valid JSON"
 
#         if not isinstance(obj, dict):
#             return False, "Finish payload must be a JSON object"
 
#         missing = [k for k in ("answer", "confidence", "complexity") if k not in obj]
#         if missing:
#             return False, f"missing keys: {missing}"
 
#         if not str(obj.get("answer", "")).strip():
#             return False, "answer must be non-empty"
 
#         def _to_score(v: Any) -> float:
#             return float(v) if isinstance(v, (int, float)) else float(str(v).strip())
 
#         try:
#             conf = _to_score(obj["confidence"])
#             comp = _to_score(obj["complexity"])
#         except Exception:
#             return False, "confidence/complexity must be numeric"
 
#         if not (0.0 <= conf <= 1.0 and 0.0 <= comp <= 1.0):
#             return False, "confidence/complexity must be in [0, 1]"
 
#         return True, ""
 
#     # ── Message builder ────────────────────────────────────────────────────────
 
#     def _build_messages(
#         self, query: str, steps: list[ReActStep], **kwargs: Any,
#     ) -> list[dict]:
#         """Build an alternating message list (system → user → assistant/user pairs)."""
#         messages: list[dict] = [
#             {"role": "system", "content": self.system_prompt},
#             {"role": "user",   "content": self._format_user_prompt(query, **kwargs)},
#         ]
#         for step in steps:
#             messages.append({
#                 "role":    "assistant",
#                 "content": (
#                     f"Thought: {step.thought}\n"
#                     f"Action: {step.action}[{step.action_input}]"
#                 ),
#             })
#             messages.append({
#                 "role":    "user",
#                 "content": f"Observation: {step.observation}",
#             })
#         return messages
 
#     # ── LLM call ──────────────────────────────────────────────────────────────
 
#     def _call_llm(  # type: ignore[override]
#         self, query: str, steps: list[ReActStep], **kwargs: Any,
#     ) -> tuple[str, float, Any]:
#         messages = self._build_messages(query, steps, **kwargs)
#         start    = time.perf_counter()
#         response = self._get_client().chat.completions.create(
#             model       = self.model,
#             messages    = messages,
#             temperature = self.temperature,
#             max_tokens  = self.max_tokens,
#             seed        = self.seed,
#         )
#         return (
#             response.choices[0].message.content or "",
#             time.perf_counter() - start,
#             response,
#         )
 
#     # ── Main loop ─────────────────────────────────────────────────────────────
 
#     def run(self, query: str, **kwargs: Any) -> AgentResponse:  # type: ignore[override]
#         steps:         list[ReActStep] = []
#         tools_called:  list[str]       = []
#         tools_results: list[dict]      = []
 
#         total_latency_llm       = 0.0
#         total_latency_tool      = 0.0
#         total_prompt_tokens     = 0
#         total_completion_tokens = 0
#         num_llm_calls           = 0   # plain local — incremented inside the loop

#         final_answer:       str            = ""
#         finish_confidence:  Optional[float] = None
#         finish_complexity:  Optional[float] = None
#         is_stopped_early    = False
#         error:              Optional[str]   = None
#         is_failed           = False
#         last_llm_out        = ""
 
#         t0 = time.perf_counter()
#         expected = self._expected_answer(kwargs)
#         effective_query = query

#         for step_num in range(1, self.max_steps + 1):
 
#             # 1. LLM call ──────────────────────────────────────────────────────
#             try:
#                 llm_out, llm_latency, response = self._call_llm(
#                     effective_query, steps, **kwargs,
#                 )
#                 last_llm_out = llm_out or ""
#             except Exception as exc:
#                 error     = str(exc)
#                 is_failed = True
#                 break
 
#             total_latency_llm       += llm_latency
#             total_prompt_tokens     += response.usage.prompt_tokens
#             total_completion_tokens += response.usage.completion_tokens
#             num_llm_calls           += 1
 
#             # 2. Parse ─────────────────────────────────────────────────────────
#             thought, action, action_input = self.parser.parse(llm_out)
 
#             # 3. Finish? ───────────────────────────────────────────────────────
#             if action == "finish":
#                 valid, reason = self._validate_finish_payload(action_input)
#                 if not valid:
#                     observation = (
#                         "Error: invalid Finish payload. "
#                         f"{reason}. "
#                         'Return only Finish[{"answer":"...","confidence":0.0,"complexity":0.0}]'
#                     )
#                     tools_called.append("finish")
#                     tools_results.append({
#                         "step": step_num, "action": "finish",
#                         "input": action_input, "observation": observation,
#                     })
#                     steps.append(ReActStep(
#                         step_num=step_num, thought=thought,
#                         action="finish", action_input=action_input,
#                         observation=observation,
#                     ))
#                     continue  # give the LLM a chance to fix the payload
 
#                 observation, tool_latency = self._call_tool("finish", action_input)
#                 total_latency_tool += tool_latency
#                 tools_called.append("finish")
#                 tools_results.append({
#                     "step": step_num, "action": "finish",
#                     "input": action_input, "observation": observation,
#                 })
#                 steps.append(ReActStep(
#                     step_num=step_num, thought=thought,
#                     action="finish", action_input=action_input,
#                     observation=observation,
#                 ))
#                 final_answer, finish_confidence, finish_complexity = (
#                     self.parser.extract_final_answer(action_input)
#                 )
#                 break
 
#             # 4. Execute tool ──────────────────────────────────────────────────
#             observation, tool_latency = self._call_tool(action, action_input)
#             total_latency_tool += tool_latency

#             tools_called.append(action)
#             tools_results.append({
#                 "step": step_num, "action": action,
#                 "input": action_input, "observation": observation,
#             })
#             steps.append(ReActStep(
#                 step_num=step_num, thought=thought,
#                 action=action, action_input=action_input,
#                 observation=observation,
#             ))
 
#         else:
#             # max_steps exhausted without a valid Finish
#             is_stopped_early = True
#             final_answer, finish_confidence, finish_complexity = "", None, None
#             if last_llm_out.strip():
#                 fa, fc, fm = self.parser.extract_final_answer(last_llm_out)
#                 if fa and not _reject_fallback_answer(fa, last_llm_out):
#                     final_answer, finish_confidence, finish_complexity = fa, fc, fm
 
#         return self._core_response(
#             query=query,
#             agent="react",
#             agent_id=AGENT_ID,
#             answer=final_answer,
#             latency_total=time.perf_counter() - t0,
#             latency_llm=total_latency_llm,
#             expected_answer=expected,
#             is_failed=is_failed,
#             error=error,
#             prompt_tokens=total_prompt_tokens,
#             completion_tokens=total_completion_tokens,
#             confidence=finish_confidence,
#             complexity=finish_complexity,
#             finalize=not is_failed,
#             latency_tools=total_latency_tool,
#             reasoning_steps=[f"[{s.step_num}] {s.thought}" for s in steps],
#             num_llm_calls=num_llm_calls,
#             num_steps=len(steps),
#             tools_available=list(self.tools.keys()),
#             tools_called=tools_called,
#             tools_results=tools_results,
#             num_tool_calls=len(tools_called),
#             max_steps=self.max_steps,
#             steps_taken=len(steps),
#             is_stopped_early=is_stopped_early,
#         )


# def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
#     return ReactAgent(model=model, dataset=dataset, **kwargs).run(
#         query=query, **kwargs,
#     )


# if __name__ == "__main__":
#     import pandas as pd

#     dataset = "gaia"
#     model = "gpt-4o-mini"
#     path = f"datasets/golden/{dataset}.parquet"
#     df = pd.read_parquet(path, columns=["query", "answer"])

#     for _, row in df.iterrows():
#         resp = run(
#             query=str(row["query"]),
#             model=model,
#             dataset=dataset,
#             expected_answer=row.get("answer"),
#         )
#         print(resp.agent_id, resp.answer, resp.latency_total, resp.is_failed)
