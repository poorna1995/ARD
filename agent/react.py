from __future__ import annotations
 
import json
import os
import re
import time
from typing import Any, Callable, Optional
 
from dotenv import load_dotenv
 
from agent.base import BaseAgent, AgentResponse
from agent.tools import (
    arxiv_search,
    github_search,
    math_tool,
    pdb_parse,
    read_file,
    web_fetch,
    web_search,
    wikipedia_search,
)
from agent.tools.decorator import tool
from evaluator.eval import is_correct, normalise_answer
from prompts.prompts import SYSTEM_PROMPT, USER_PROMPT
 
load_dotenv()
 
 
# ── Built-in finish tool ───────────────────────────────────────────────────────
 
@tool("finish", 'Submit the final answer. Input: JSON string {"answer": "<value>"}.')
def finish(answer: str) -> str:
    return answer
TOOLS: dict[str, Callable] = {
    fn._tool_name: fn
    for fn in [
        web_search,
        web_fetch,
        wikipedia_search,
        arxiv_search,
        github_search,
        pdb_parse,
        read_file,
        math_tool,
        finish,
    ]
}
 
class ReActStep:
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
 
    Canonical format:
        Thought: <text>
        Action: tool_name[input]
 
    Falls back gracefully to "finish" when no action is detected.
    """
 
    # Everything after "Thought:" up to the next "Action:" label
    THOUGHT_RE = re.compile(
        r"Thought\s*:\s*(.+?)(?=\nAction\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    # Primary:  Action: tool_name[input]
    ACTION_BRACKET_RE = re.compile(
        r"Action\s*:\s*([A-Za-z_]\w*)\s*\[([^\]]*)\]",
        re.DOTALL | re.IGNORECASE,
    )
    # Fallback: Action: tool_name\nAction Input: input
    ACTION_SPLIT_RE = re.compile(
        r"Action\s*:\s*([A-Za-z_]\w*)\s*\n+\s*(?:Action\s+)?Input\s*:\s*"
        r"(.+?)(?=\nThought|\nObservation|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    # Bare Finish[...] shortcut
    FINISH_RE = re.compile(r"\bFinish\s*\[([^\]]*)\]", re.DOTALL | re.IGNORECASE)
 
    @classmethod
    def parse(cls, text: str) -> tuple[str, str, str]:
        """Return (thought, action_name, action_input). action_name is lowercase."""
        thought = ""
        m = cls.THOUGHT_RE.search(text)
        if m:
            thought = m.group(1).strip()
 
        m = cls.ACTION_BRACKET_RE.search(text)
        if m:
            return thought, m.group(1).strip().lower(), m.group(2).strip()
 
        m = cls.ACTION_SPLIT_RE.search(text)
        if m:
            return thought, m.group(1).strip().lower(), m.group(2).strip()
 
        m = cls.FINISH_RE.search(text)
        if m:
            return thought, "finish", m.group(1).strip()
 
        # Nothing matched — treat full output as final answer
        return thought, "finish", text.strip()
 
    @classmethod
    def extract_final_answer(
        cls, raw: str
    ) -> tuple[str, Optional[float], Optional[float]]:
        """Delegates to evaluator.eval.normalise_answer (shared with all agents)."""
        return normalise_answer(raw)
 
 
# ── Fallback-answer guards ─────────────────────────────────────────────────────
 
def _looks_like_raw_tool_output(s: str) -> bool:
    """True when the string is obviously a tool observation, not a short QA answer."""
    t = (s or "").strip()
    if len(t) < 4:
        return False
    low = t.lower()
    if "search failed" in low[:160] or low.startswith("no results found"):
        return True
    if "\nurl     :" in t or "\nsnippet :" in t:
        return True
    if t.startswith("[") and "]" in t[:160]:
        return True
    if t.startswith("{") and any(
        k in t[:1200] for k in ('"highlights"', '"error"', '"ok"')
    ):
        return True
    if "error: unknown tool" in low[:120]:
        return True
    if "invalid finish payload" in low[:200]:
        return True
    return False
 
 
def _reject_fallback_answer(fa: str, raw_llm: str) -> bool:
    """True if the extracted answer should NOT be used when finish never succeeded."""
    if _looks_like_raw_tool_output(fa):
        return True
    if not (fa or "").strip():
        return True
    ft = fa.strip()
    rl = (raw_llm or "").strip()
    if rl and ft == rl:
        return True
    if len(ft) > 600 and "thought:" in rl.lower() and "action:" in rl.lower():
        return True
    return False
 
 
# ── ReactAgent ─────────────────────────────────────────────────────────────────
 
class ReactAgent(BaseAgent):
    """ReAct agent that interleaves Thought / Action / Observation.
 
    Design notes
    ────────────
    • The scratchpad is sent as alternating assistant/user messages so the LLM
      sees correct conversational structure.
    • Tool names are normalised to lowercase at registration *and* dispatch time,
      preventing case-mismatch misses.
    • The parser enforces one canonical format:  Action: name[input]
    • Runtime inputs are shared via **kwargs (e.g., expected_answer).
    """
 
    def __init__(
        self,
        model:    str,
        dataset:  str,
        tools:    dict[str, Callable] | None = None,
        max_steps: int = 12,
        **kwargs: Any,
    ) -> None:
        cfg_result = self._normalize_config(
            model=model,
            dataset=dataset,
            kwargs=kwargs,
        )
        cfg = cfg_result.config
        self.dataset   = cfg.dataset
        self.max_steps = max_steps
        # Normalise all tool keys to lowercase at registration time
        raw_tools  = tools or TOOLS
        self.tools = {k.lower(): v for k, v in raw_tools.items()}
        self.parser = ReActParser()

        super().__init__(
            model         = cfg.model,
            system_prompt = SYSTEM_PROMPT[self.dataset]["react"].format(
                tools_block=self._build_tools_block()
            ),
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
        """Dispatch to a registered tool. Returns (observation, wall_clock_seconds)."""
        start = time.perf_counter()
        fn    = self.tools.get(action.strip().lower())

        if fn is None:
            obs = (
                f"Error: unknown tool '{action}'. "
                f"Available tools: {', '.join(self.tools)}. "
                "Fix the Action name and try again."
            )
        else:
            try:
                obs = fn(action_input)
            except Exception as exc:
                obs = f"Tool '{action}' raised an error: {exc}. Try a different approach."

        return str(obs), time.perf_counter() - start

    def _validate_finish_payload(self, raw: str) -> tuple[bool, str]:
        """Validate the Finish[...] payload.
 
        GAIA:     strict JSON with answer, confidence, complexity.
        Non-GAIA: just requires a non-empty payload string.
        """
        text = (raw or "").strip()
        if not text:
            return False, "empty Finish payload"
 
        if self.dataset != "gaia":
            # Permissive: any non-empty string is accepted
            return True, ""
 
        # ── GAIA: strict JSON validation ──────────────────────────────────────
        try:
            obj = json.loads(text)
        except Exception:
            return False, "Finish payload must be valid JSON for GAIA"
 
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
 
    def _build_messages(self, query: str, steps: list[ReActStep]) -> list[dict]:
        """Build an alternating message list (system → user → assistant/user pairs)."""
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self.user_prompt.format(query=query)},
        ]
        for step in steps:
            messages.append({
                "role":    "assistant",
                "content": (
                    f"Thought: {step.thought}\n"
                    f"Action: {step.action}[{step.action_input}]"
                ),
            })
            messages.append({
                "role":    "user",
                "content": f"Observation: {step.observation}",
            })
        return messages
 
    # ── LLM call ──────────────────────────────────────────────────────────────
 
    def _call_llm(  # type: ignore[override]
        self, query: str, steps: list[ReActStep]
    ) -> tuple[str, float, Any]:
        messages = self._build_messages(query, steps)
        start    = time.perf_counter()
        response = self._get_client().chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            seed        = self.seed,
        )
        return (
            response.choices[0].message.content or "",
            time.perf_counter() - start,
            response,
        )
 
    # ── Main loop ─────────────────────────────────────────────────────────────
 
    def run(self, query: str, **kwargs: Any) -> AgentResponse:  # type: ignore[override]
        steps:         list[ReActStep] = []
        tools_called:  list[str]       = []
        tools_results: list[dict]      = []
 
        total_latency_llm       = 0.0
        total_latency_tool      = 0.0
        total_prompt_tokens     = 0
        total_completion_tokens = 0
        num_llm_calls           = 0   # plain local — incremented inside the loop

        final_answer:       str            = ""
        finish_confidence:  Optional[float] = None
        finish_complexity:  Optional[float] = None
        is_stopped_early    = False
        error:              Optional[str]   = None
        is_failed           = False
        last_llm_out        = ""
 
        start_total    = time.perf_counter()
        effective_query = query
 
        for step_num in range(1, self.max_steps + 1):
 
            # 1. LLM call ──────────────────────────────────────────────────────
            try:
                llm_out, llm_latency, response = self._call_llm(effective_query, steps)
                last_llm_out = llm_out or ""
            except Exception as exc:
                error     = str(exc)
                is_failed = True
                break
 
            total_latency_llm       += llm_latency
            total_prompt_tokens     += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens
            num_llm_calls           += 1
 
            # 2. Parse ─────────────────────────────────────────────────────────
            thought, action, action_input = self.parser.parse(llm_out)
 
            # 3. Finish? ───────────────────────────────────────────────────────
            if action == "finish":
                valid, reason = self._validate_finish_payload(action_input)
                if not valid:
                    observation = (
                        "Error: invalid Finish payload. "
                        f"{reason}. "
                        'Return only Finish[{"answer":"...","confidence":0.0,"complexity":0.0}]'
                    )
                    tools_called.append("finish")
                    tools_results.append({
                        "step": step_num, "action": "finish",
                        "input": action_input, "observation": observation,
                    })
                    steps.append(ReActStep(
                        step_num=step_num, thought=thought,
                        action="finish", action_input=action_input,
                        observation=observation,
                    ))
                    continue  # give the LLM a chance to fix the payload
 
                observation, tool_latency = self._call_tool("finish", action_input)
                total_latency_tool += tool_latency
                tools_called.append("finish")
                tools_results.append({
                    "step": step_num, "action": "finish",
                    "input": action_input, "observation": observation,
                })
                steps.append(ReActStep(
                    step_num=step_num, thought=thought,
                    action="finish", action_input=action_input,
                    observation=observation,
                ))
                final_answer, finish_confidence, finish_complexity = (
                    self.parser.extract_final_answer(action_input)
                )
                break
 
            # 4. Execute tool ──────────────────────────────────────────────────
            observation, tool_latency = self._call_tool(action, action_input)
            total_latency_tool += tool_latency

            tools_called.append(action)
            tools_results.append({
                "step": step_num, "action": action,
                "input": action_input, "observation": observation,
            })
            steps.append(ReActStep(
                step_num=step_num, thought=thought,
                action=action, action_input=action_input,
                observation=observation,
            ))
 
        else:
            # max_steps exhausted without a valid Finish
            is_stopped_early = True
            final_answer, finish_confidence, finish_complexity = "", None, None
            if last_llm_out.strip():
                fa, fc, fm = self.parser.extract_final_answer(last_llm_out)
                if fa and not _reject_fallback_answer(fa, last_llm_out):
                    final_answer, finish_confidence, finish_complexity = fa, fc, fm
 
        total_latency = time.perf_counter() - start_total
        expected      = kwargs.get("expected_answer")

        response_obj = AgentResponse(
            # Core
            query   = query,
            answer  = final_answer,
            model   = self.model,
            agent   = "react",
            dataset = self.dataset,
            # Latency
            latency_total = total_latency,
            latency_llm   = total_latency_llm,
            latency_tools = total_latency_tool,
            # Tokens + cost
            prompt_tokens     = total_prompt_tokens,
            completion_tokens = total_completion_tokens,
            total_tokens      = total_prompt_tokens + total_completion_tokens,
            cost_usd          = self._compute_cost(
                total_prompt_tokens, total_completion_tokens
            ),
            # Reasoning
            reasoning_steps = [f"[{s.step_num}] {s.thought}" for s in steps],
            num_llm_calls   = num_llm_calls,
            num_steps       = len(steps),
            # Tools
            tools_available = list(self.tools.keys()),
            tools_called    = tools_called,
            tools_results   = tools_results,
            num_tool_calls  = len(tools_called),
            # ReAct-specific
            max_steps        = self.max_steps,
            steps_taken      = len(steps),
            is_stopped_early = is_stopped_early,
            # Evaluation
            expected_answer = expected,
            is_correct      = (
                is_correct(final_answer, str(expected)) if expected else None
            ),
            confidence = finish_confidence,
            complexity = finish_complexity,
            # Errors
            error     = error,
            is_failed = is_failed,
        )
        return self._finalize_response(response_obj)
# ── Module-level entry point ───────────────────────────────────────────────────
 
def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
    agent = ReactAgent(model=model, dataset=dataset, **kwargs)
    return agent.run(query=query, **kwargs)
# from __future__ import annotations

# import json
# import os
# import re
# import time
# from typing import Any, Callable, Optional

# from dotenv import load_dotenv

# from agent.attachments import prepare_query_with_attachment_hint
# from agent.base import BaseAgent, AgentResponse
# from agent.tools import (
#     arxiv_search,
#     github_search,
#     math_tool,
#     pdb_parse,
#     read_file,
#     web_fetch,
#     web_search,
#     wikipedia_search,
# )
# from agent.tools.decorator import tool
# from evaluator.eval import is_correct, normalise_answer
# from prompts.prompts import SYSTEM_PROMPT, USER_PROMPT

# load_dotenv()


# # ── Built-in finish tool ───────────────────────────────────────────────────────

# @tool("finish", 'Submit the final answer. Input: JSON string {"answer": "<value>"}.')
# def finish(answer: str) -> str:
#     return answer


# # ── Default tool registry ──────────────────────────────────────────────────────

# TOOLS: dict[str, Callable] = {
#     fn._tool_name: fn
#     for fn in [
#         web_search,
#         web_fetch,
#         wikipedia_search,
#         arxiv_search,
#         github_search,
#         pdb_parse,
#         read_file,
#         math_tool,
#         finish,
#     ]
# }


# # ── ReActStep ──────────────────────────────────────────────────────────────────

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
#     """
#     Parses LLM output into (thought, action_name, action_input).

#     Canonical format:
#         Thought: <text>
#         Action: tool_name[input]

#     Falls back gracefully to "finish" when no action is detected.
#     """

#     # Capture everything after "Thought:" up to the next "Action:" label
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
#     def extract_final_answer(cls, raw: str) -> tuple[str, Optional[float], Optional[float]]:
#         """Delegates to evaluator.eval.normalise_answer (shared with all agents)."""
#         return normalise_answer(raw)


# # ── Fallback-answer guards ─────────────────────────────────────────────────────

# def _looks_like_raw_tool_output(s: str) -> bool:
#     """True when the string is obviously a tool observation, not a GAIA short answer."""
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
#         k in t[:1200]
#         for k in ('"highlights"', '"error"', '"ok"')
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
#     """
#     ReAct agent that interleaves Thought / Action / Observation.

#     Design notes
#     ────────────
#     • The scratchpad is sent as alternating assistant/user messages so the
#       LLM sees correct conversational structure.
#     • Tool names are normalised to lowercase at registration *and* dispatch
#       time, preventing case-mismatch misses.
#     • The parser enforces one canonical format:  Action: name[input]
#     """

#     def __init__(
#         self,
#         model: str,
#         dataset: str,
#         tools: dict[str, Callable] | None = None,
#         max_steps: int = 12,
#         **kwargs,
#     ) -> None:
#         self.dataset   = dataset
#         self.max_steps = max_steps
#         # Normalise all tool keys to lowercase at registration time
#         raw_tools  = tools or TOOLS
#         self.tools = {k.lower(): v for k, v in raw_tools.items()}
#         self.parser = ReActParser()

#         super().__init__(
#             model         = model,
#             system_prompt = SYSTEM_PROMPT[dataset]["react"].format(
#                 tools_block=self._build_tools_block()
#             ),
#             user_prompt   = USER_PROMPT[dataset]["react"],
#             **kwargs,
#         )

#     # ── Helpers ────────────────────────────────────────────────────────────

#     def _build_tools_block(self) -> str:
#         return "\n".join(
#             f"  {name:15s}: {getattr(fn, '_tool_description', 'No description.')}"
#             for name, fn in self.tools.items()
#         )

#     def _call_tool(self, action: str, action_input: str) -> tuple[str, float]:
#         """Dispatch to a registered tool. Returns (observation, wall_clock_seconds)."""
#         start = time.perf_counter()
#         fn    = self.tools.get(action.strip().lower())

#         if fn is None:
#             obs = (
#                 f"Error: unknown tool '{action}'. "
#                 f"Available tools: {', '.join(self.tools)}. "
#                 "Fix the Action name and try again."
#             )
#         else:
#             try:
#                 obs = fn(action_input)
#             except Exception as exc:
#                 obs = f"Tool '{action}' raised an error: {exc}. Try a different approach."

#         return str(obs), time.perf_counter() - start

#     def _validate_finish_payload(self, raw: str) -> tuple[bool, str]:
#         """
#         Validate the Finish[...] payload.

#         For GAIA, enforce strict JSON:
#             {"answer": "...", "confidence": <0..1>, "complexity": <0..1>}
#         """
#         text = (raw or "").strip()
#         if not text:
#             return False, "empty Finish payload"

#         if self.dataset != "gaia":
#             return True, ""          # permissive for non-GAIA datasets

#         try:
#             obj = json.loads(text)
#         except Exception:
#             return False, "Finish payload must be valid JSON for GAIA"

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

#     # ── Message builder ────────────────────────────────────────────────────

#     def _build_messages(self, query: str, steps: list[ReActStep]) -> list[dict]:
#         """
#         Construct an alternating message list:

#             system     : instructions + tool descriptions
#             user       : original query
#             assistant  : Thought + Action   (step 1)
#             user       : Observation        (step 1)
#             …
#         """
#         messages: list[dict] = [
#             {"role": "system", "content": self.system_prompt},
#             {"role": "user",   "content": self.user_prompt.format(query=query)},
#         ]
#         for step in steps:
#             messages.append({
#                 "role":    "assistant",
#                 "content": f"Thought: {step.thought}\nAction: {step.action}[{step.action_input}]",
#             })
#             messages.append({
#                 "role":    "user",
#                 "content": f"Observation: {step.observation}",
#             })
#         return messages

#     # ── LLM call ──────────────────────────────────────────────────────────

#     def _call_llm(self, query: str, steps: list[ReActStep]) -> tuple[str, float, Any]:
#         messages = self._build_messages(query, steps)
#         start    = time.perf_counter()
#         response = self._get_client().chat.completions.create(
#             model       = self.model,
#             messages    = messages,
#             temperature = self.temperature,
#             max_tokens  = self.max_tokens,
#             seed        = self.seed,
#         )
#         return response.choices[0].message.content, time.perf_counter() - start, response

#     # ── Main loop ──────────────────────────────────────────────────────────

#     def run(self, query: str, dataset: str = "", **kwargs) -> AgentResponse:
#         steps:          list[ReActStep] = []
#         tools_called:   list[str]       = []
#         tools_results:  list[dict]      = []

#         total_latency_llm       = 0.0
#         total_latency_tool      = 0.0
#         total_prompt_tokens     = 0
#         total_completion_tokens = 0
#         num_llm_calls           = 0       # FIX: must be declared here, not inside AgentResponse()

#         final_answer       = ""
#         finish_confidence: Optional[float] = None
#         finish_complexity: Optional[float] = None
#         is_stopped_early   = False
#         error: Optional[str] = None
#         is_failed          = False
#         last_llm_out       = ""

#         start_total     = time.perf_counter()
#         effective_query, _ = prepare_query_with_attachment_hint(query)

#         for step_num in range(1, self.max_steps + 1):

#             # ── 1. LLM call ───────────────────────────────────────────────
#             try:
#                 llm_out, llm_latency, response = self._call_llm(effective_query, steps)
#                 last_llm_out = llm_out or ""
#             except Exception as exc:
#                 error     = str(exc)
#                 is_failed = True
#                 break

#             total_latency_llm       += llm_latency
#             total_prompt_tokens     += response.usage.prompt_tokens
#             total_completion_tokens += response.usage.completion_tokens
#             num_llm_calls           += 1   # FIX: increment here inside the loop

#             # ── 2. Parse ──────────────────────────────────────────────────
#             thought, action, action_input = self.parser.parse(llm_out)

#             # ── 3. Finish? ────────────────────────────────────────────────
#             if action == "finish":
#                 valid, reason = self._validate_finish_payload(action_input)
#                 if not valid:
#                     observation = (
#                         "Error: invalid Finish payload. "
#                         f"{reason}. "
#                         'Return only Finish[{{"answer":"...","confidence":0.0,"complexity":0.0}}]'
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
#                     continue   # give the LLM a chance to fix the payload

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

#             # ── 4. Execute tool ───────────────────────────────────────────
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
#             # max_steps exhausted without a valid finish
#             # Do NOT promote raw tool output as an answer
#             is_stopped_early = True
#             final_answer, finish_confidence, finish_complexity = "", None, None
#             if last_llm_out.strip():
#                 fa, fc, fm = self.parser.extract_final_answer(last_llm_out)
#                 if fa and not _reject_fallback_answer(fa, last_llm_out):
#                     final_answer, finish_confidence, finish_complexity = fa, fc, fm

#         total_latency = time.perf_counter() - start_total
#         expected      = kwargs.get("expected_answer")

#         return AgentResponse(
#             # Core
#             query   = query,
#             answer  = final_answer,
#             model   = self.model,
#             agent   = "react",
#             dataset = dataset or self.dataset,
#             # Latency
#             latency_total = total_latency,
#             latency_llm   = total_latency_llm,
#             latency_tools = total_latency_tool,
#             # Tokens + cost
#             prompt_tokens     = total_prompt_tokens,
#             completion_tokens = total_completion_tokens,
#             total_tokens      = total_prompt_tokens + total_completion_tokens,
#             cost_usd          = self._compute_cost(
#                 total_prompt_tokens, total_completion_tokens
#             ),
#             # Reasoning
#             reasoning_steps = [f"[{s.step_num}] {s.thought}" for s in steps],
#             num_llm_calls   = num_llm_calls,   # FIX: now a plain variable
#             num_steps       = len(steps),
#             # Tools
#             tools_available = list(self.tools.keys()),
#             tools_called    = tools_called,
#             tools_results   = tools_results,
#             num_tool_calls  = len(tools_called),
#             # ReAct-specific
#             max_steps        = self.max_steps,
#             steps_taken      = len(steps),
#             is_stopped_early = is_stopped_early,
#             # Evaluation
#             expected_answer = expected,
#             is_correct      = (
#                 is_correct(final_answer, str(expected))
#                 if expected else None
#             ),
#             confidence = finish_confidence,
#             complexity = finish_complexity,
#             # Errors
#             error     = error,
#             is_failed = is_failed,
#         )


# # ── Module-level entry point ───────────────────────────────────────────────────

# def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
#     agent = ReactAgent(model=model, dataset=dataset, **kwargs)
#     return agent.run(query=query, dataset=dataset, **kwargs)




# # agent/react.py
# from __future__ import annotations
# import os
# import json
# import re
# import time
# from typing import Any, Callable, Optional
# from agent.attachments import prepare_query_with_attachment_hint
# from agent.base import BaseAgent, AgentResponse
# from evaluator.eval import normalise_answer
# from prompts.prompts import USER_PROMPT, SYSTEM_PROMPT
# from agent.tools import (
#     arxiv_search,
#     github_search,
#     math_tool,
#     pdb_parse,
#     read_file,
#     web_fetch,
#     web_search,
#     wikipedia_search,
# )
# from agent.tools.decorator import tool
# from dotenv import load_dotenv

# load_dotenv()


# @tool("finish", "Submit the final answer. Input: JSON string {{\"answer\": \"<value>\"}}.")
# def finish(answer: str) -> str:
#     return answer


# # ── Default Tool Registry ──────────────────────────────────────────────────────

# TOOLS: dict[str, Callable] = {
#     fn._tool_name: fn
#     for fn in [
#         web_search,
#         web_fetch,
#         wikipedia_search,
#         arxiv_search,
#         github_search,
#         pdb_parse,
#         read_file,
#         math_tool,
#         finish,
#     ]
# }


# # ── ReAct Step dataclass ───────────────────────────────────────────────────────

# class ReActStep:
#     def __init__(
#         self,
#         step_num:     int,
#         thought:      str,
#         action:       str,
#         action_input: str,
#         observation:  str = "",
#     ):
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


# # ── ReAct Parser ───────────────────────────────────────────────────────────────

# class ReActParser:
#     """
#     Parses LLM output into (thought, action_name, action_input).

#     Enforces ONE canonical format:
#         Thought: <text>
#         Action: tool_name[input]

#     Falls back gracefully to "finish" if no action is detected.
#     """

#     # Capture everything after "Thought:" up to the next "Action:" label
#     THOUGHT_RE = re.compile(
#         r"Thought\s*:\s*(.+?)(?=\nAction\s*:|\Z)",
#         re.DOTALL | re.IGNORECASE,
#     )

#     # Primary format — Action: tool_name[input]
#     # Allows multi-line input inside the brackets
#     ACTION_BRACKET_RE = re.compile(
#         r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]*)\]",
#         re.DOTALL | re.IGNORECASE,
#     )

#     # Fallback format — Action: tool_name\nAction Input: input
#     ACTION_SPLIT_RE = re.compile(
#         r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n+\s*(?:Action\s+)?Input\s*:\s*(.+?)(?=\nThought|\nObservation|\Z)",
#         re.DOTALL | re.IGNORECASE,
#     )

#     # Bare Finish[...] shortcut
#     FINISH_RE = re.compile(r"\bFinish\s*\[([^\]]*)\]", re.DOTALL | re.IGNORECASE)

#     @classmethod
#     def parse(cls, text: str) -> tuple[str, str, str]:
#         """
#         Returns (thought, action_name, action_input).
#         action_name is always lowercase and stripped.
#         """
#         # ── Thought ───────────────────────────────────────────────────────
#         thought = ""
#         t_match = cls.THOUGHT_RE.search(text)
#         if t_match:
#             thought = t_match.group(1).strip()

#         # ── Primary: Action: tool[input] ──────────────────────────────────
#         a_match = cls.ACTION_BRACKET_RE.search(text)
#         if a_match:
#             action_name  = a_match.group(1).strip().lower()
#             action_input = a_match.group(2).strip()
#             return thought, action_name, action_input

#         # ── Secondary: Action: tool\nAction Input: input ──────────────────
#         s_match = cls.ACTION_SPLIT_RE.search(text)
#         if s_match:
#             action_name  = s_match.group(1).strip().lower()
#             action_input = s_match.group(2).strip()
#             return thought, action_name, action_input

#         # ── Bare Finish[...] ──────────────────────────────────────────────
#         f_match = cls.FINISH_RE.search(text)
#         if f_match:
#             return thought, "finish", f_match.group(1).strip()

#         # ── Nothing matched — treat entire output as final answer ─────────
#         return thought, "finish", text.strip()

#     @classmethod
#     def extract_final_answer(cls, raw: str) -> tuple[str, Optional[float], Optional[float]]:
#         """Delegates to evaluator.eval.normalise_answer (shared with all agents)."""
#         return normalise_answer(raw)


# def _looks_like_raw_tool_output(s: str) -> bool:
#     """
#     Heuristic: strings that are obviously tool observations, not GAIA short answers.
#     Used when max_steps exhausts without finish — we must not promote these to answer.
#     """
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
#     if t.startswith("{") and '"highlights"' in t[:1200]:
#         return True
#     if t.startswith("{") and '"error"' in low[:400] and ("fetch_failed" in low or "code" in low[:200]):
#         return True
#     if t.startswith("{") and '"ok"' in low[:80] and '"tool"' in low[:120]:
#         return True
#     if "error: unknown tool" in low[:120]:
#         return True
#     if "invalid finish payload" in low[:200]:
#         return True
#     return False


# def _reject_fallback_answer(fa: str, raw_llm: str) -> bool:
#     """True if extracted 'answer' should not be used when finish never succeeded."""
#     if _looks_like_raw_tool_output(fa):
#         return True
#     if not (fa or "").strip():
#         return True
#     rl = (raw_llm or "").strip()
#     ft = fa.strip()
#     if rl and ft == rl:
#         return True
#     if len(ft) > 600 and "thought:" in rl.lower() and "action:" in rl.lower():
#         return True
#     return False


# # ── ReAct Agent ────────────────────────────────────────────────────────────────

# class ReactAgent(BaseAgent):
#     """
#     ReAct agent that interleaves Thought / Action / Observation.

#     Key design decisions
#     ─────────────────────
#     • The scratchpad is sent as *alternating* assistant/user messages
#       so the LLM sees the correct conversational structure.
#     • Tool names are normalised to lowercase at registration and at
#       dispatch time, preventing case-mismatch misses.
#     • The parser enforces ONE canonical format:  Action: name[input]
#     """

#     def __init__(
#         self,
#         model:     str,
#         dataset:   str,
#         tools:     dict[str, Callable] | None = None,
#         max_steps: int = 12,
#         **kwargs,
#     ):
#         self.dataset   = dataset
#         self.max_steps = max_steps
#         # Normalise all tool keys to lowercase at registration time
#         raw_tools  = tools or TOOLS
#         self.tools = {k.lower(): v for k, v in raw_tools.items()}
#         self.parser = ReActParser()

#         tools_block = self._build_tools_block()

#         super().__init__(
#             model         = model,
#             system_prompt = SYSTEM_PROMPT[dataset]["react"].format(tools_block=tools_block),
#             user_prompt   = USER_PROMPT[dataset]["react"],
#             **kwargs,
#         )

#     # ── Helpers ────────────────────────────────────────────────────────────

#     def _build_tools_block(self) -> str:
#         lines = []
#         for name, fn in self.tools.items():
#             desc = getattr(fn, "_tool_description", "No description.")
#             lines.append(f"  {name:15s}: {desc}")
#         return "\n".join(lines)

#     def _call_tool(self, action: str, action_input: str) -> tuple[str, float]:
#         """
#         Dispatch to the registered tool function.

#         Returns (observation_string, wall_clock_seconds).
#         """
#         start = time.perf_counter()
#         normalised = action.strip().lower()
#         fn = self.tools.get(normalised)

#         if fn is None:
#             available = ", ".join(self.tools.keys())
#             obs = (
#                 f"Error: unknown tool '{action}'. "
#                 f"Available tools: {available}. "
#                 f"Fix the Action name and try again."
#             )
#         else:
#             try:
#                 obs = fn(action_input)
#             except Exception as e:
#                 obs = f"Tool '{action}' raised an error: {e}. Try a different approach."

#         return str(obs), time.perf_counter() - start

#     def _validate_finish_payload(self, raw: str) -> tuple[bool, str]:
#         """
#         Validate Finish[...] payload shape before accepting completion.

#         For GAIA, enforce strict JSON keys:
#           {"answer": "...", "confidence": <0..1>, "complexity": <0..1>}
#         """
#         text = (raw or "").strip()
#         if not text:
#             return False, "empty Finish payload"

#         if self.dataset != "gaia":
#             # Keep non-GAIA behavior permissive.
#             return True, ""

#         try:
#             obj = json.loads(text)
#         except Exception:
#             return False, "Finish payload must be valid JSON for GAIA"

#         if not isinstance(obj, dict):
#             return False, "Finish payload must be a JSON object"

#         required = ("answer", "confidence", "complexity")
#         missing = [k for k in required if k not in obj]
#         if missing:
#             return False, f"missing keys: {missing}"

#         answer = str(obj.get("answer", "")).strip()
#         if not answer:
#             return False, "answer must be non-empty"

#         def _to_score(v):
#             if isinstance(v, (int, float)):
#                 return float(v)
#             return float(str(v).strip())

#         try:
#             conf = _to_score(obj["confidence"])
#             comp = _to_score(obj["complexity"])
#         except Exception:
#             return False, "confidence/complexity must be numeric"

#         if not (0.0 <= conf <= 1.0 and 0.0 <= comp <= 1.0):
#             return False, "confidence/complexity must be in [0,1]"

#         return True, ""

#     # ── Conversation builder ───────────────────────────────────────────────

#     def _build_messages(self, query: str, steps: list[ReActStep]) -> list[dict]:
#         """
#         Build a proper alternating message list for the LLM:

#             system  : instructions + tool descriptions
#             user    : original query
#             assistant: Thought + Action   (step 1)
#             user    : Observation          (step 1)
#             assistant: Thought + Action   (step 2)
#             user    : Observation          (step 2)
#             …

#         This is far superior to concatenating everything into one user
#         message because the LLM understands role boundaries correctly.
#         """
#         messages: list[dict] = [
#             {"role": "system", "content": self.system_prompt},
#             {"role": "user",   "content": self.user_prompt.format(query=query)},
#         ]

#         for step in steps:
#             # What the assistant said
#             assistant_turn = (
#                 f"Thought: {step.thought}\n"
#                 f"Action: {step.action}[{step.action_input}]"
#             )
#             messages.append({"role": "assistant", "content": assistant_turn})

#             # What the environment replied
#             messages.append({
#                 "role":    "user",
#                 "content": f"Observation: {step.observation}",
#             })

#         return messages

#     # ── LLM call ──────────────────────────────────────────────────────────

#     def _call_llm(
#         self,
#         query: str,
#         steps: list[ReActStep],
#     ) -> tuple[str, float, Any]:
#         messages = self._build_messages(query, steps)
#         start    = time.perf_counter()
#         response = self._get_client().chat.completions.create(
#             model       = self.model,
#             messages    = messages,
#             temperature = self.temperature,
#             max_tokens  = self.max_tokens,
#             seed        = self.seed,
#         )
#         latency = time.perf_counter() - start
#         return response.choices[0].message.content, latency, response

#     # ── Main loop ──────────────────────────────────────────────────────────

#     def run(self, query: str, dataset: str = "", **kwargs) -> AgentResponse:
#         steps:                list[ReActStep] = []
#         tools_called:         list[str]       = []
#         tools_results:        list[dict]      = []
#         total_latency_llm       = 0.0
#         total_latency_tool      = 0.0
#         total_prompt_tokens     = 0
#         total_completion_tokens = 0
#         final_answer            = ""
#         finish_confidence       = None
#         finish_complexity       = None
#         is_stopped_early        = False
#         error                   = None
#         is_failed               = False

#         start_total = time.perf_counter()
#         last_llm_out = ""
#         effective_query, _ = prepare_query_with_attachment_hint(query)

#         for step_num in range(1, self.max_steps + 1):

#             # ── 1. Call the LLM ───────────────────────────────────────────
#             try:
#                 llm_out, llm_latency, response = self._call_llm(effective_query, steps)
#                 last_llm_out = llm_out or ""
#             except Exception as e:
#                 error     = str(e)
#                 is_failed = True
#                 break

#             total_latency_llm       += llm_latency
#             total_prompt_tokens     += response.usage.prompt_tokens
#             total_completion_tokens += response.usage.completion_tokens

#             # ── 2. Parse Thought / Action / Input ─────────────────────────
#             thought, action, action_input = self.parser.parse(llm_out)

#             # ── 3. Finish early? ──────────────────────────────────────────
#             if action == "finish":
#                 valid, reason = self._validate_finish_payload(action_input)
#                 if not valid:
#                     observation = (
#                         "Error: invalid Finish payload. "
#                         f"{reason}. "
#                         "Return only Finish[{\"answer\":\"...\",\"confidence\":0.0,\"complexity\":0.0}]"
#                     )
#                     tools_called.append("finish")
#                     tools_results.append({
#                         "step":        step_num,
#                         "action":      "finish",
#                         "input":       action_input,
#                         "observation": observation,
#                     })
#                     steps.append(ReActStep(
#                         step_num=step_num, thought=thought,
#                         action="finish", action_input=action_input,
#                         observation=observation,
#                     ))
#                     continue

#                 # Run the finish tool so the answer goes through the same path
#                 observation, tool_latency = self._call_tool("finish", action_input)
#                 total_latency_tool += tool_latency
#                 tools_called.append("finish")
#                 tools_results.append({
#                     "step":        step_num,
#                     "action":      "finish",
#                     "input":       action_input,
#                     "observation": observation,
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

#             # ── 4. Guard: don't call a tool if action is unrecognised ─────
#             #    (the tool dispatcher will return an error observation, which
#             #     the LLM will see and correct on the next step)

#             # ── 5. Execute Tool ───────────────────────────────────────────
#             observation, tool_latency = self._call_tool(action, action_input)
#             total_latency_tool += tool_latency

#             tools_called.append(action)
#             tools_results.append({
#                 "step":        step_num,
#                 "action":      action,
#                 "input":       action_input,
#                 "observation": observation,
#             })

#             steps.append(ReActStep(
#                 step_num=step_num, thought=thought,
#                 action=action, action_input=action_input,
#                 observation=observation,
#             ))

#         else:
#             # max_steps exhausted without a valid finish — do NOT use the last
#             # Observation as answer (it is usually raw web_search / web_fetch text).
#             is_stopped_early = True
#             final_answer, finish_confidence, finish_complexity = "", None, None
#             if last_llm_out.strip():
#                 fa, fc, fm = self.parser.extract_final_answer(last_llm_out)
#                 if fa and not _reject_fallback_answer(fa, last_llm_out):
#                     final_answer, finish_confidence, finish_complexity = fa, fc, fm

#         total_latency = time.perf_counter() - start_total
#         expected      = kwargs.get("expected_answer")

#         return AgentResponse(
#             # Core
#             query             = query,
#             answer            = final_answer,
#             model             = self.model,
#             agent             = "react",
#             dataset           = dataset or self.dataset,
#             # Latency
#             latency_total     = total_latency,
#             latency_llm       = total_latency_llm,
#             latency_tools     = total_latency_tool,
#             # Tokens + Cost
#             prompt_tokens     = total_prompt_tokens,
#             completion_tokens = total_completion_tokens,
#             total_tokens      = total_prompt_tokens + total_completion_tokens,
#             cost_usd          = self._compute_cost(
#                                     total_prompt_tokens,
#                                     total_completion_tokens,
#                                 ),
#             # Reasoning
#             reasoning_steps   = [f"[{s.step_num}] {s.thought}" for s in steps],
#             num_llm_calls = 0
#             # inside the loop:
#             llm_out, llm_latency, response = self._call_llm(...)
#             num_llm_calls += 1
#             num_steps         = len(steps),
#             # Tools
#             tools_available   = list(self.tools.keys()),
#             tools_called      = tools_called,
#             tools_results     = tools_results,
#             num_tool_calls    = len(tools_called),
#             # ReAct specific
#             max_steps         = self.max_steps,
#             steps_taken       = len(steps),
#             is_stopped_early  = is_stopped_early,
#             # Evaluation
#             expected_answer   = expected,
#             is_correct        = (
#                 final_answer.lower().strip() == str(expected).lower().strip()
#                 if expected else None
#             ),
#             confidence        = finish_confidence,
#             complexity        = finish_complexity,
#             # Errors
#             error             = error,
#             is_failed         = is_failed,
#         )


# # ── Module-level entry point ───────────────────────────────────────────────────

# def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
#     agent = ReactAgent(model=model, dataset=dataset, **kwargs)
#     return agent.run(query=query, dataset=dataset, **kwargs)




 # if ext in {".png", ".jpg", ".jpeg"}:
        #     from PIL import Image, ImageFilter, ImageEnhance
        #     import numpy as np
        #     from collections import Counter

        #     img = Image.open(path).convert("RGBA")
        #     arr = np.array(img)
        #     r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        #     red_mask   = (r > 150) & (g < 100) & (b < 100)
        #     green_mask = (r > 120) & (g > 150) & (b < 80) & (g > r - 60)
        #     blue_mask  = (r < 100) & (g < 100) & (b > 150)

        #     try:
        #         import pytesseract, cv2

        #         gray = np.array(img.convert("L"))

        #         # ── Step 1: Detect fraction bars via horizontal morphology ──
        #         _, thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        #                         cv2.THRESH_BINARY_INV, 15, 4)
        #         # A fraction bar is a short, thin horizontal dark segment
        #         h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1))
        #         h_lines  = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kernel, iterations=1)

        #         # AFTER (works on ALL OpenCV versions):
        #         output = cv2.findContours(h_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #         contours = output[1] if len(output) == 3 else output[0]

        #         SCALE    = 5          # upscale factor for fraction OCR
        #         DIGIT_CFG = "--psm 8 --oem 3 -c tessedit_char_whitelist=0123456789"

        #         fraction_regions = []   # [{frac, x, y, x2, y2, bar_y}]

        #         for cnt in contours:
        #             bx, by, bw, bh = cv2.boundingRect(cnt)
        #             aspect = bw / max(bh, 1)

        #             # Keep only short, thin bars (not full-width rules or text underlines)
        #             if aspect < 2.5 or bw < 6 or bw > gray.shape[1] * 0.25:
        #                 continue

        #             pad_x, pad_y = 6, 28   # generous vertical padding to capture digits
        #             rx1 = max(0, bx - pad_x)
        #             rx2 = min(gray.shape[1], bx + bw + pad_x)
        #             ry1 = max(0, by - pad_y)
        #             ry2 = min(gray.shape[0], by + bh + pad_y)

        #             region_rgb = img.convert("RGB").crop((rx1, ry1, rx2, ry2))
        #             region_big = region_rgb.resize(
        #                 (region_rgb.width * SCALE, region_rgb.height * SCALE), Image.LANCZOS
        #             )
        #             region_big = ImageEnhance.Contrast(region_big).enhance(2.5)
        #             region_big = ImageEnhance.Sharpness(region_big).enhance(2.5)

        #             bar_in_region = (by - ry1) * SCALE   # y of bar inside upscaled crop

        #             top_crop = region_big.crop((0, 0,                   region_big.width, bar_in_region))
        #             bot_crop = region_big.crop((0, bar_in_region + SCALE, region_big.width, region_big.height))

        #             num_txt = pytesseract.image_to_string(top_crop, config=DIGIT_CFG).strip()
        #             den_txt = pytesseract.image_to_string(bot_crop, config=DIGIT_CFG).strip()

        #             # Accept only if both sides are purely numeric
        #             if num_txt.isdigit() and den_txt.isdigit():
        #                 fraction_regions.append({
        #                     "frac":  f"{num_txt}/{den_txt}",
        #                     "x": rx1, "y": ry1, "x2": rx2, "y2": ry2,
        #                     "bar_y": by,
        #                 })

        #         # ── Step 2: Full-page OCR for everything else ──
        #         PAGE_SCALE = 4
        #         big = img.convert("RGB").resize(
        #             (img.width * PAGE_SCALE, img.height * PAGE_SCALE), Image.LANCZOS
        #         )
        #         big = ImageEnhance.Contrast(big).enhance(1.8)
        #         big = ImageEnhance.Sharpness(big).enhance(2.0)

        #         data = pytesseract.image_to_data(
        #             big, config="--psm 6 --oem 3",
        #             output_type=pytesseract.Output.DICT,
        #         )

        #         # ── Step 3: Collect page tokens, skip anything inside a fraction region ──
        #         def in_any_fraction(ox, oy, ow, oh):
        #             for fr in fraction_regions:
        #                 if ox < fr["x2"] and ox + ow > fr["x"] and \
        #                 oy < fr["y2"] and oy + oh > fr["y"]:
        #                     return True
        #             return False

        #         def get_color(oy1, oy2, ox1, ox2):
        #             region = arr[oy1:oy2, ox1:ox2]
        #             if region.size == 0:
        #                 return ""
        #             if   red_mask  [oy1:oy2, ox1:ox2].mean() > 0.2: return "RED"
        #             elif green_mask[oy1:oy2, ox1:ox2].mean() > 0.2: return "GREEN"
        #             elif blue_mask [oy1:oy2, ox1:ox2].mean() > 0.2: return "BLUE"
        #             return ""

        #         all_tokens = []
        #         for i, text in enumerate(data["text"]):
        #             text = text.strip()
        #             if not text or int(data["conf"][i]) < 25:
        #                 continue
        #             x, y, w, h = (data["left"][i], data["top"][i],
        #                         data["width"][i], data["height"][i])
        #             ox, oy, ow, oh = x // PAGE_SCALE, y // PAGE_SCALE, \
        #                             w // PAGE_SCALE, h // PAGE_SCALE

        #             if in_any_fraction(ox, oy, ow, oh):
        #                 continue   # already captured above

        #             color_tag = get_color(
        #                 max(0, oy), min(arr.shape[0], oy + oh),
        #                 max(0, ox), min(arr.shape[1], ox + ow),
        #             )
        #             all_tokens.append({
        #                 "text":  f"[{color_tag}]{text}" if color_tag else text,
        #                 "color": color_tag,
        #                 "x": x, "y": y, "w": w, "h": h,
        #             })

        #         # ── Step 4: Inject detected fractions as synthetic tokens ──
        #         for fr in fraction_regions:
        #             fy1 = max(0, fr["y"]); fy2 = min(arr.shape[0], fr["y2"])
        #             fx1 = max(0, fr["x"]); fx2 = min(arr.shape[1], fr["x2"])
        #             color_tag = get_color(fy1, fy2, fx1, fx2)
        #             display   = f"[{color_tag}]{fr['frac']}" if color_tag else fr["frac"]
        #             all_tokens.append({
        #                 "text":  display,
        #                 "color": color_tag,
        #                 "x": (fx1 + fx2) // 2 * PAGE_SCALE,
        #                 "y": fr["bar_y"] * PAGE_SCALE,
        #                 "w": (fx2 - fx1) * PAGE_SCALE,
        #                 "h": 10,
        #             })

        #         # ── Step 5: Group tokens into visual rows, sort left→right ──
        #         all_tokens.sort(key=lambda t: (t["y"], t["x"]))

        #         ROW_TOL = 10 * PAGE_SCALE   # vertical tolerance for same row
        #         rows: list[list[dict]] = []
        #         for tok in all_tokens:
        #             placed = False
        #             for row in rows:
        #                 if abs(tok["y"] - row[0]["y"]) < ROW_TOL:
        #                     row.append(tok)
        #                     placed = True
        #                     break
        #             if not placed:
        #                 rows.append([tok])

        #         red_numbers, green_numbers, blue_numbers = [], [], []
        #         line_texts = []

        #         for row in rows:
        #             row.sort(key=lambda t: t["x"])
        #             parts = []
        #             for tok in row:
        #                 parts.append(tok["text"])
        #                 tag = tok["color"]
        #                 raw = tok["text"].replace(f"[{tag}]", "") if tag else tok["text"]
        #                 for seg in raw.split("/"):
        #                     try:
        #                         num = float(seg.replace(",", ""))
        #                         if   tag == "RED":   red_numbers.append(num)
        #                         elif tag == "GREEN": green_numbers.append(num)
        #                         elif tag == "BLUE":  blue_numbers.append(num)
        #                     except ValueError:
        #                         pass
        #             line_texts.append(" ".join(parts))

        #         full_text = "\n".join(line_texts).strip()

        #         # ── Dominant colors ──
        #         pixels    = arr[:,:,:3].reshape(-1, 3)
        #         not_white = ~((pixels[:,0]>240)&(pixels[:,1]>240)&(pixels[:,2]>240))
        #         not_black = ~((pixels[:,0]<15) &(pixels[:,1]<15) &(pixels[:,2]<15))
        #         filtered  = pixels[not_white & not_black]
        #         top_colors    = Counter(map(tuple, (filtered // 32 * 32).tolist())).most_common(5) \
        #                         if len(filtered) else []
        #         color_summary = ", ".join(f"RGB{c}" for c, _ in top_colors)

        #         # ── Image type heuristic ──
        #         unique_colors = len(set(map(tuple, pixels.tolist())))
        #         dark_ratio    = ((r < 50) & (g < 50) & (b < 50)).sum() / (arr.shape[0] * arr.shape[1])
        #         color_variety = len(set(map(tuple, (filtered // 64 * 64).tolist()))) if len(filtered) else 0

        #         if   unique_colors > 50000:                        img_type = "photograph or complex diagram"
        #         elif dark_ratio > 0.05 and len(line_texts) > 10:  img_type = "text document or screenshot"
        #         elif len(line_texts) < 5  and color_variety > 20: img_type = "chart or graph"
        #         elif len(line_texts) > 5  and color_variety < 10: img_type = "table or structured data"
        #         else:                                              img_type = "mixed content (text + visuals)"

        #         return "\n".join([
        #             "[IMAGE STRUCTURE]",
        #             f"Type           : {img_type}",
        #             f"Size           : {img.width}x{img.height} px",
        #             f"Dominant Colors: {color_summary or 'N/A'}",
        #             "",
        #             "[TEXT CONTENT]",
        #             full_text if full_text else "(no text detected)",
        #             "",
        #             "[COLORED NUMBERS]",
        #             f"Red   : {red_numbers   if red_numbers   else 'none'}",
        #             f"Green : {green_numbers if green_numbers else 'none'}",
        #             f"Blue  : {blue_numbers  if blue_numbers  else 'none'}",
        #         ])

        #     except ImportError:
        #         return "pytesseract not installed — cannot extract image content."
       


        # if ext in {".png", ".jpg", ".jpeg"}:
        #     from PIL import Image
        #     import numpy as np
        #     from collections import Counter

        #     img = Image.open(path).convert("RGBA")
        #     arr = np.array(img)
        #     r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        #     red_mask   = (r > 150) & (g < 100) & (b < 100)
        #     green_mask = (r > 120) & (g > 150) & (b < 80) & (g > r - 60)
        #     blue_mask  = (r < 100) & (g < 100) & (b > 150)

        #     try:
        #         import pytesseract

        #         scale = 3
        #         big   = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        #         data  = pytesseract.image_to_data(
        #             big.convert("RGB"),
        #             config="--psm 6",
        #             output_type=pytesseract.Output.DICT,
        #         )

        #         # ── Collect all valid tokens with full positional info ──
        #         all_tokens = []
        #         for i, text in enumerate(data["text"]):
        #             text = text.strip()
        #             if not text or int(data["conf"][i]) < 35:
        #                 continue
        #             x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        #             oy1 = max(0, y // scale);  oy2 = min(arr.shape[0], (y + h) // scale)
        #             ox1 = max(0, x // scale);  ox2 = min(arr.shape[1], (x + w) // scale)
        #             color_tag = ""
        #             if arr[oy1:oy2, ox1:ox2].size > 0:
        #                 if   red_mask  [oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "RED"
        #                 elif green_mask[oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "GREEN"
        #                 elif blue_mask [oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "BLUE"
        #             all_tokens.append({
        #                 "text": text,
        #                 "x": x, "y": y, "w": w, "h": h,
        #                 "cx": x + w / 2,   # center x
        #                 "cy": y + h / 2,   # center y
        #                 "color": color_tag,
        #                 "line_key": (data["block_num"][i], data["par_num"][i], data["line_num"][i]),
        #                 "index": i,
        #             })

        #         # ── NEW: Detect vertically stacked fraction pairs ──
        #         def is_numeric(s):
        #             try:
        #                 float(s.replace(",", ""))
        #                 return True
        #             except ValueError:
        #                 return False

        #         fraction_map = {}   # token index -> fraction string (replaces both tokens)
        #         used_indices = set()

        #         numeric_tokens = [t for t in all_tokens if is_numeric(t["text"])]

        #         for i, ti in enumerate(numeric_tokens):
        #             if ti["index"] in used_indices:
        #                 continue
        #             best = None
        #             for j, tj in enumerate(numeric_tokens):
        #                 if i == j or tj["index"] in used_indices:
        #                     continue

        #                 # Must be horizontally aligned (centers close relative to width)
        #                 x_overlap = abs(ti["cx"] - tj["cx"]) < max(ti["w"], tj["w"]) * 0.65

        #                 # Must be vertically stacked: gap between them reasonable
        #                 vertical_gap = abs(ti["y"] - tj["y"])
        #                 avg_h = (ti["h"] + tj["h"]) / 2
        #                 y_stacked = avg_h * 0.4 < vertical_gap < avg_h * 5.0

        #                 # Must NOT be on the same line
        #                 same_line = ti["line_key"] == tj["line_key"]

        #                 if x_overlap and y_stacked and not same_line:
        #                     # Prefer the closest vertical neighbor
        #                     if best is None or vertical_gap < abs(ti["y"] - best["y"]):
        #                         best = tj

        #             if best is not None:
        #                 # Top token = numerator, bottom = denominator
        #                 if ti["y"] < best["y"]:
        #                     numerator, denominator = ti, best
        #                 else:
        #                     numerator, denominator = best, ti

        #                 frac_str = f"{numerator['text']}/{denominator['text']}"
        #                 color_tag = numerator["color"] or denominator["color"]
        #                 frac_display = f"[{color_tag}]{frac_str}" if color_tag else frac_str

        #                 # Map both token indices to the fraction string
        #                 # Place fraction at the numerator's line position
        #                 fraction_map[numerator["index"]] = frac_display
        #                 fraction_map[denominator["index"]] = None  # suppress denominator token
        #                 used_indices.add(numerator["index"])
        #                 used_indices.add(denominator["index"])

        #         # ── Group tokens by line, substituting fractions where detected ──
        #         lines = {}
        #         for token in all_tokens:
        #             idx = token["index"]
        #             key = token["line_key"]

        #             if idx in fraction_map:
        #                 val = fraction_map[idx]
        #                 if val is None:
        #                     continue  # this was the denominator, already consumed
        #                 lines.setdefault(key, []).append((val, token["color"]))
        #             else:
        #                 display = f"[{token['color']}]{token['text']}" if token["color"] else token["text"]
        #                 lines.setdefault(key, []).append((display, token["color"]))

        #         # ── Build text lines + collect colored numbers (including fractions) ──
        #         red_numbers, green_numbers, blue_numbers = [], [], []
        #         line_texts = []

        #         for tokens in lines.values():
        #             parts = []
        #             for token_display, tag in tokens:
        #                 parts.append(token_display)
        #                 # Extract numeric value for colored number tracking
        #                 raw = token_display.replace(f"[{tag}]", "") if tag else token_display
        #                 try:
        #                     num = float(raw.replace(",", ""))
        #                     if   tag == "RED":   red_numbers.append(num)
        #                     elif tag == "GREEN": green_numbers.append(num)
        #                     elif tag == "BLUE":  blue_numbers.append(num)
        #                 except ValueError:
        #                     pass
        #             line_texts.append(" ".join(parts))

        #         full_text = "\n".join(line_texts).strip()

        #         # ── Dominant colors ──
        #         pixels    = arr[:,:,:3].reshape(-1, 3)
        #         not_white = ~((pixels[:,0]>240)&(pixels[:,1]>240)&(pixels[:,2]>240))
        #         not_black = ~((pixels[:,0]<15) &(pixels[:,1]<15) &(pixels[:,2]<15))
        #         filtered  = pixels[not_white & not_black]
        #         top_colors    = Counter(map(tuple, (filtered // 32 * 32).tolist())).most_common(5) if len(filtered) else []
        #         color_summary = ", ".join(f"RGB{c}" for c, _ in top_colors)

        #         # ── Image type heuristic ──
        #         unique_colors = len(set(map(tuple, pixels.tolist())))
        #         dark_ratio    = ((r < 50) & (g < 50) & (b < 50)).sum() / (arr.shape[0] * arr.shape[1])
        #         color_variety = len(set(map(tuple, (filtered // 64 * 64).tolist()))) if len(filtered) else 0

        #         if   unique_colors > 50000:                       img_type = "photograph or complex diagram"
        #         elif dark_ratio > 0.05 and len(line_texts) > 10: img_type = "text document or screenshot"
        #         elif len(line_texts) < 5  and color_variety > 20: img_type = "chart or graph"
        #         elif len(line_texts) > 5  and color_variety < 10: img_type = "table or structured data"
        #         else:                                              img_type = "mixed content (text + visuals)"

        #         return "\n".join([
        #             "[IMAGE STRUCTURE]",
        #             f"Type           : {img_type}",
        #             f"Size           : {img.width}x{img.height} px",
        #             f"Dominant Colors: {color_summary or 'N/A'}",
        #             "",
        #             "[TEXT CONTENT]",
        #             full_text if full_text else "(no text detected)",
        #             "",
        #             "[COLORED NUMBERS]",
        #             f"Red   : {red_numbers   if red_numbers   else 'none'}",
        #             f"Green : {green_numbers if green_numbers else 'none'}",
        #             f"Blue  : {blue_numbers  if blue_numbers  else 'none'}",
        #         ])

        #     except ImportError:
        #         return "pytesseract not installed — cannot extract image content."
        # if ext in {".png", ".jpg", ".jpeg"}:
        #     from PIL import Image
        #     import numpy as np
        #     from collections import Counter

        #     img = Image.open(path).convert("RGBA")
        #     arr = np.array(img)
        #     r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

        #     red_mask   = (r > 150) & (g < 100) & (b < 100)
        #     green_mask = (r > 120) & (g > 150) & (b < 80) & (g > r - 60)
        #     blue_mask  = (r < 100) & (g < 100) & (b > 150)

        #     try:
        #         import pytesseract

        #         scale = 3
        #         big   = img.resize((img.width * scale, img.height * scale), Image.LANCZOS)
        #         data  = pytesseract.image_to_data(
        #             big.convert("RGB"),
        #             config="--psm 6",
        #             output_type=pytesseract.Output.DICT,
        #         )

        #         # ── Group tokens by line, tag each token with its color ──
        #         lines = {}
        #         for i, text in enumerate(data["text"]):
        #             text = text.strip()
        #             if not text or int(data["conf"][i]) < 35:
        #                 continue
        #             key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        #             x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        #             oy1 = max(0, y // scale);  oy2 = min(arr.shape[0], (y + h) // scale)
        #             ox1 = max(0, x // scale);  ox2 = min(arr.shape[1], (x + w) // scale)
        #             color_tag = ""
        #             if arr[oy1:oy2, ox1:ox2].size > 0:
        #                 if   red_mask  [oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "RED"
        #                 elif green_mask[oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "GREEN"
        #                 elif blue_mask [oy1:oy2, ox1:ox2].mean() > 0.2: color_tag = "BLUE"
        #             lines.setdefault(key, []).append((text, color_tag))

        #         # ── Build text lines + collect colored numbers ──
        #         red_numbers, green_numbers, blue_numbers = [], [], []
        #         line_texts = []

        #         for tokens in lines.values():
        #             parts = []
        #             for token, tag in tokens:
        #                 parts.append(f"[{tag}]{token}" if tag else token)
        #                 try:
        #                     num = float(token.replace(",", ""))
        #                     if   tag == "RED":   red_numbers.append(num)
        #                     elif tag == "GREEN": green_numbers.append(num)
        #                     elif tag == "BLUE":  blue_numbers.append(num)
        #                 except ValueError:
        #                     pass
        #             line_texts.append(" ".join(parts))

        #         full_text = "\n".join(line_texts).strip()

        #         # ── Dominant colors ──
        #         pixels    = arr[:,:,:3].reshape(-1, 3)
        #         not_white = ~((pixels[:,0]>240)&(pixels[:,1]>240)&(pixels[:,2]>240))
        #         not_black = ~((pixels[:,0]<15) &(pixels[:,1]<15) &(pixels[:,2]<15))
        #         filtered  = pixels[not_white & not_black]
        #         top_colors    = Counter(map(tuple, (filtered // 32 * 32).tolist())).most_common(5) if len(filtered) else []
        #         color_summary = ", ".join(f"RGB{c}" for c, _ in top_colors)

        #         # ── Image type heuristic ──
        #         unique_colors = len(set(map(tuple, pixels.tolist())))
        #         dark_ratio    = ((r < 50) & (g < 50) & (b < 50)).sum() / (arr.shape[0] * arr.shape[1])
        #         color_variety = len(set(map(tuple, (filtered // 64 * 64).tolist()))) if len(filtered) else 0

        #         if   unique_colors > 50000:                       img_type = "photograph or complex diagram"
        #         elif dark_ratio > 0.05 and len(line_texts) > 10: img_type = "text document or screenshot"
        #         elif len(line_texts) < 5  and color_variety > 20: img_type = "chart or graph"
        #         elif len(line_texts) > 5  and color_variety < 10: img_type = "table or structured data"
        #         else:                                              img_type = "mixed content (text + visuals)"

        #         return "\n".join([
        #             "[IMAGE STRUCTURE]",
        #             f"Type           : {img_type}",
        #             f"Size           : {img.width}x{img.height} px",
        #             f"Dominant Colors: {color_summary or 'N/A'}",
        #             "",
        #             "[TEXT CONTENT]",
        #             full_text if full_text else "(no text detected)",
        #             "",
        #             "[COLORED NUMBERS]",
        #             f"Red   : {red_numbers   if red_numbers   else 'none'}",
        #             f"Green : {green_numbers if green_numbers else 'none'}",
        #             f"Blue  : {blue_numbers  if blue_numbers  else 'none'}",
        #         ])

        #     except ImportError:
        #         return "pytesseract not installed — cannot extract image content."