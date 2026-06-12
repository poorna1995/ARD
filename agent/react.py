from __future__ import annotations

import json
import re
import time

from dataclasses import dataclass
from typing import Any, Callable

from agent.base import BaseAgent, AgentResponse
from config.local.constants import TOOLS as AGENT_TOOLS
from input.prompts.prompts_core import (
    REACT_BAD_FINAL_ANSWERS,
    react_format_retry_observation,
    react_loop_observation,
)


# ============================================================
# Tool Specification
# ============================================================

@dataclass
class ToolSpec:
    name: str
    description: str
    fn: Callable[[str, dict[str, Any]], str]


@dataclass
class ReActStep:
    step_num: int
    thought: str
    action: str
    action_input: str
    observation: str = ""


# ============================================================
# ReAct Parser
# ============================================================

class ReActParser:
    """Strict single-turn ReAct with light bare-tool recovery."""

    THOUGHT_RE = re.compile(
        r"Thought\s*:\s*(.+?)(?=\n(?:Action|Final Answer)\s*:|\Z)",
        re.IGNORECASE | re.DOTALL,
    )
    FINAL_RE = re.compile(
        r"Final Answer\s*:\s*(.+)",
        re.IGNORECASE | re.DOTALL,
    )
    CONF_RE = re.compile(r"Confidence\s*:\s*([0-9.]+)", re.IGNORECASE)
    COMP_RE = re.compile(r"Complexity\s*:\s*([0-9.]+)", re.IGNORECASE)
    BARE_TOOL_RE = re.compile(
        r"^([A-Za-z_][A-Za-z0-9_]*)\s*\[",
        re.IGNORECASE | re.MULTILINE,
    )

    @classmethod
    def _invalid(cls, reason: str, *, thought: str = "", raw: str = "") -> dict[str, Any]:
        return {"type": "invalid", "reason": reason, "thought": thought, "raw": raw}

    @classmethod
    def parse(cls, text: str | None) -> dict[str, Any]:
        if text is None or not str(text).strip():
            return cls._invalid("empty_output")

        raw = str(text).strip()

        if "Observation:" in raw:
            return cls._invalid("invented_observation", raw=raw)

        n_action = raw.count("Action:")
        n_final = raw.count("Final Answer:")

        if n_action > 1:
            return cls._invalid("multiple_actions", raw=raw)
        if n_final > 1:
            return cls._invalid("multiple_final_answers", raw=raw)
        if n_action and n_final:
            return cls._invalid("action_and_final_together", raw=raw)

        if n_final == 1:
            return cls._parse_final(raw)
        if n_action == 1:
            return cls._parse_action(raw)

        bare = cls._parse_bare_tool_call(raw)
        if bare is not None:
            return bare

        thought = ""
        m = cls.THOUGHT_RE.search(raw)
        if m:
            thought = m.group(1).strip()
        return cls._invalid("missing_action_or_final", thought=thought, raw=raw)

    @classmethod
    def _parse_final(cls, raw: str) -> dict[str, Any]:
        if "Action:" in raw:
            return cls._invalid("action_with_final", raw=raw)
        if not re.search(r"^\s*Thought\s*:", raw, re.IGNORECASE | re.MULTILINE):
            return cls._invalid("missing_thought", raw=raw)

        fm = cls.FINAL_RE.search(raw)
        if not fm:
            return cls._invalid("malformed_final", raw=raw)

        thought = ""
        tm = cls.THOUGHT_RE.search(raw)
        if tm:
            thought = tm.group(1).strip()

        conf = comp = None
        c = cls.CONF_RE.search(raw)
        if c:
            conf = float(c.group(1))
        cx = cls.COMP_RE.search(raw)
        if cx:
            comp = float(cx.group(1))

        return {
            "type": "final",
            "thought": thought,
            "answer": fm.group(1).strip(),
            "confidence": conf,
            "complexity": comp,
        }

    @classmethod
    def _extract_bracketed(cls, raw: str, *, require_action_prefix: bool) -> tuple[str, str, int] | None:
        if require_action_prefix:
            m = re.search(
                r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[",
                raw,
                re.IGNORECASE,
            )
        else:
            m = cls.BARE_TOOL_RE.search(raw)
        if not m:
            return None
        action = m.group(1).strip().lower()
        i = m.end()
        depth = 1
        while i < len(raw):
            ch = raw[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    return action, raw[m.end() : i].strip(), i + 1
            i += 1
        return None

    @classmethod
    def _parse_action(cls, raw: str) -> dict[str, Any]:
        if "Final Answer:" in raw:
            return cls._invalid("action_with_final", raw=raw)
        if not re.search(r"^\s*Thought\s*:", raw, re.IGNORECASE | re.MULTILINE):
            return cls._invalid("missing_thought", raw=raw)

        extracted = cls._extract_bracketed(raw, require_action_prefix=True)
        if extracted is None:
            return cls._invalid("malformed_action", raw=raw)

        action, action_input, end_pos = extracted
        if not action_input:
            return cls._invalid("empty_action_input", raw=raw)

        after = raw[end_pos:].strip()
        if after:
            return cls._invalid("extra_content_after_action", raw=raw)

        thought = ""
        tm = cls.THOUGHT_RE.search(raw)
        if tm:
            thought = tm.group(1).strip()

        if action == "finish":
            return {
                "type": "final",
                "thought": thought,
                "answer": action_input,
                "confidence": None,
                "complexity": None,
            }

        return {
            "type": "action",
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

    @classmethod
    def _parse_bare_tool_call(cls, raw: str) -> dict[str, Any] | None:
        """Recover shorthand like retrieve[query] without Thought: or Action: prefix."""
        stripped = raw.strip()
        if re.search(r"^\s*Thought\s*:", stripped, re.IGNORECASE | re.MULTILINE):
            return None
        if "Action:" in stripped or "Final Answer:" in stripped:
            return None

        # One-line shorthand only (optional trailing blank lines).
        lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
        if len(lines) != 1:
            return None
        stripped = lines[0]

        extracted = cls._extract_bracketed(stripped, require_action_prefix=False)
        if extracted is None:
            return None

        action, action_input, end_pos = extracted
        if not action_input or stripped[end_pos:].strip():
            return None

        return {
            "type": "action",
            "thought": "",
            "action": action,
            "action_input": action_input,
            "bare": True,
        }

    @classmethod
    def extract_final_answer(cls, raw: str) -> tuple[str, float | None, float | None]:
        from evaluator.parse import parse_llm_output

        return parse_llm_output(raw)


_TOKEN_STOP = frozenset({
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "is", "was", "were", "be", "by", "with", "from", "that", "this",
})
_SIMILAR_QUERY_JACCARD = 0.85
_EXACT_REPEAT_TOOLS = AGENT_TOOLS.react_repeat
_MATH_TOOL_NUDGE_AFTER = 3
_STAGNATION_NOTICE_AT = 2  # same tool + query + observation (2nd identical triplet)


class ReActAgent(BaseAgent):
    """
    Dataset-agnostic ReAct: strict parser, tool loop, format retries,
    repeat-query guard, clean failure reporting. Policy lives in prompts.
    """

    def __init__(
        self,
        model: str,
        tools: list[ToolSpec],
        system_prompt: str,
        user_prompt: str,
        max_steps: int = 8,
        max_format_retries: int = 3,
        dataset: str = "",
        **kwargs,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            **kwargs,
        )
        self.dataset = dataset
        self.max_steps = max_steps
        self.max_format_retries = max(0, int(max_format_retries))
        self.tools = {t.name.lower(): t for t in tools}
        self.parser = ReActParser()

    def _build_tools_block(self) -> str:
        return "\n".join(
            f"- {name}: {tool.description}" for name, tool in self.tools.items()
        )

    def _render_context(self, context: dict[str, Any]) -> str:
        if not context:
            return "None"

        if "paragraphs" in context:
            chunks = []
            for p in context["paragraphs"]:
                title = (p.get("title") or "").strip()
                text = (p.get("text") or "").strip()
                if not text:
                    continue
                chunks.append(f"{title}\n{text}" if title else text)
            return "\n\n---\n\n".join(chunks) if chunks else "None"

        if "documents" in context:
            chunks = []
            for d in context["documents"]:
                title = (d.get("title") or "").strip()
                text = (d.get("text") or "").strip()
                if not text:
                    continue
                chunks.append(f"{title}\n{text}" if title else text)
            return "\n\n---\n\n".join(chunks) if chunks else "None"

        return json.dumps(context)[:4000]

    def _build_messages(
        self,
        query: str,
        context: dict[str, Any],
        steps: list[ReActStep],
    ) -> list[dict[str, str]]:
        scratchpad: list[str] = []
        for step in steps:
            scratchpad.append(f"Thought: {step.thought}")
            scratchpad.append(f"Action: {step.action}[{step.action_input}]")
            scratchpad.append(f"Observation: {step.observation}")

        system = self.system_prompt.replace("{tools_block}", self._build_tools_block())
        user = self.user_prompt.replace("{query}", query)
        user += f"\n\nContext:\n{self._render_context(context)}"
        user += f"\n\nScratchpad:\n{chr(10).join(scratchpad)}"

        return [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]

    def _execute_tool(
        self,
        tool_name: str,
        tool_input: str,
        context: dict[str, Any],
    ) -> tuple[str, float]:
        start = time.perf_counter()
        tool = self.tools.get(tool_name)
        if tool is None:
            return (
                f"Unknown tool '{tool_name}'. Available tools: {list(self.tools.keys())}",
                0.0,
            )
        try:
            result = str(tool.fn(tool_input, context))
        except Exception as e:
            result = f"Tool execution failed: {e}"
        return result, time.perf_counter() - start

    @staticmethod
    def _final_json_missing_answer(raw: str) -> bool:
        from evaluator.parse import final_json_missing_answer_field

        return final_json_missing_answer_field(raw)

    @staticmethod
    def _final_json_placeholder(raw: str) -> bool:
        from evaluator.parse import final_json_placeholder_answer

        return final_json_placeholder_answer(raw)

    def _accept_final_body(
        self,
        body: str,
    ) -> tuple[str, float | None, float | None, str | None]:
        """
        Validate Final Answer body. Returns (answer, conf, comp, reject_reason).
        reject_reason is a format-retry key when validation fails.
        """
        from evaluator.parse import parse_llm_output, parse_react_final_json

        if self._final_json_missing_answer(body):
            return "", None, None, "missing_answer_field"
        if self._final_json_placeholder(body):
            return "", None, None, "placeholder_answer"

        strict = parse_react_final_json(body)
        if strict is not None:
            if self._is_bad_final_answer(strict.predicted_answer):
                return "", None, None, "placeholder_answer"
            return strict.predicted_answer, strict.confidence, strict.complexity, None

        if (self.dataset or "").strip().lower() == "gaia":
            if body.strip().startswith("{"):
                return "", None, None, "malformed_final"
            return "", None, None, "malformed_final"

        answer, conf, comp = parse_llm_output(body)
        if self._is_bad_final_answer(answer):
            return "", None, None, "placeholder_answer"
        return answer, conf, comp, None

    @staticmethod
    def _is_bad_final_answer(answer: str) -> bool:
        a = (answer or "").strip().lower()
        if a in REACT_BAD_FINAL_ANSWERS:
            return True
        if any(token in a for token in ("<value>", "<float", "<short value>", "<answer>")):
            return True
        if re.fullmatch(r"<[a-z][a-z0-9_ ]*>", a):
            return True
        return False

    @staticmethod
    def _observation_is_no_context(observation: str) -> bool:
        try:
            data = json.loads(observation)
            if isinstance(data, dict) and data.get("error") == "NO_CONTEXT":
                return True
        except (json.JSONDecodeError, TypeError):
            pass
        return "NO_CONTEXT" in (observation or "")

    @staticmethod
    def _python_exec_output_empty(observation: str) -> bool:
        try:
            data = json.loads(observation)
            if isinstance(data, dict) and data.get("ok") is True:
                return not str(data.get("output") or "").strip()
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    @staticmethod
    def _python_exec_output_weak(observation: str) -> bool:
        """True when python_exec ran but printed no usable result (empty or None)."""
        try:
            data = json.loads(observation)
            if isinstance(data, dict) and data.get("ok") is True:
                out = str(data.get("output") or "").strip().lower()
                return out in ("", "none", "null")
        except (json.JSONDecodeError, TypeError):
            pass
        return False

    @staticmethod
    def _append_observation_notice(observation: str, notice: str) -> str:
        body = (observation or "").strip()
        return notice if not body else f"{body}\n\n{notice}"

    @staticmethod
    def _normalize_tool_input(tool_input: str) -> str:
        return re.sub(r"\s+", " ", (tool_input or "").strip().lower())

    @classmethod
    def _tool_input_tokens(cls, text: str) -> set[str]:
        return {
            t
            for t in re.sub(r"[^\w\s]", " ", text.lower()).split()
            if len(t) > 1 and t not in _TOKEN_STOP
        }

    @classmethod
    def _similar_tool_input(cls, a: str, b: str) -> bool:
        ta, tb = cls._tool_input_tokens(a), cls._tool_input_tokens(b)
        if not ta or not tb:
            return False
        union = len(ta | tb)
        return (len(ta & tb) / union) >= _SIMILAR_QUERY_JACCARD if union else False

    @classmethod
    def _is_repeat_query(
        cls,
        history: list[tuple[str, str]],
        action: str,
        tool_input: str,
    ) -> bool:
        norm = cls._normalize_tool_input(tool_input)
        for prev_action, prev_norm in history:
            if prev_action != action:
                continue
            if prev_norm == norm:
                return True
            if action not in _EXACT_REPEAT_TOOLS and cls._similar_tool_input(prev_norm, tool_input):
                return True
        return False

    def _matching_tool_history(
        self,
        history: list[tuple[str, str, str]],
        action: str,
        tool_input: str,
    ) -> list[tuple[str, str, str]]:
        norm = self._normalize_tool_input(tool_input)
        out: list[tuple[str, str, str]] = []
        for a, n, obs in history:
            if a != action:
                continue
            if n == norm or (
                action not in _EXACT_REPEAT_TOOLS
                and self._similar_tool_input(n, tool_input)
            ):
                out.append((a, n, obs))
        return out

    def _stagnant_triplet_count(
        self,
        tool_history: list[tuple[str, str, str]],
        action: str,
        tool_input: str,
        observation: str,
    ) -> int:
        """Executions with same tool, query, and observation (including this one)."""
        prior = self._matching_tool_history(tool_history, action, tool_input)
        obs_norm = observation.strip()
        same_obs = sum(1 for _a, _n, obs in prior if obs.strip() == obs_norm)
        if same_obs != len(prior):
            return 0
        return same_obs + 1

    def _append_stagnation_notices(
        self,
        observation: str,
        *,
        triplet_count: int,
    ) -> str:
        if triplet_count >= _STAGNATION_NOTICE_AT:
            observation = self._append_observation_notice(
                observation,
                react_loop_observation("tool_stagnation"),
            )
        return observation

    def _loop_notice_for_repeat(
        self,
        tool_history: list[tuple[str, str, str]],
        action: str,
        tool_input: str,
        blocked_streak: int,
    ) -> str:
        """Blocked repeat: stagnation when tool+query+obs unchanged; else reformulate."""
        prior = self._matching_tool_history(tool_history, action, tool_input)
        if not prior:
            return react_loop_observation("repeated_search")

        ref_obs = prior[-1][2]
        triplet_count = self._stagnant_triplet_count(
            tool_history, action, tool_input, ref_obs,
        )
        # Blocked attempt counts as another repeat of the same triplet.
        triplet_count += blocked_streak

        if triplet_count >= _STAGNATION_NOTICE_AT:
            return react_loop_observation("tool_stagnation")
        return react_loop_observation("repeated_search")

    def _maybe_stagnant_loop_notice(
        self,
        tool_history: list[tuple[str, str, str]],
        action: str,
        tool_input: str,
        observation: str,
    ) -> str:
        """After execute: same tool+query+observation → stagnation / finalize nudges."""
        triplet_count = self._stagnant_triplet_count(
            tool_history, action, tool_input, observation,
        )
        return self._append_stagnation_notices(
            observation, triplet_count=triplet_count,
        )

    def _call_model(self, messages: list[dict[str, str]]):
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stop": ["\nObservation:"],
        }
        if self.seed is not None:
            kwargs["seed"] = self.seed
        client = self._get_client()
        try:
            return client.chat.completions.create(**kwargs)
        except Exception:
            kwargs.pop("stop", None)
            return client.chat.completions.create(**kwargs)

    def run(self, query: str, **kwargs) -> AgentResponse:
        context = kwargs.get("context", {})
        expected_answer = kwargs.get("expected_answer")

        steps: list[ReActStep] = []
        tools_called: list[str] = []
        tools_results: list[dict[str, Any]] = []
        tool_history: list[tuple[str, str, str]] = []
        repeat_block_streak: dict[tuple[str, str], int] = {}
        format_retry_reasons: list[str] = []

        total_latency_llm = 0.0
        total_latency_tool = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        num_llm_calls = 0
        num_format_retries = 0

        final_answer = ""
        confidence = None
        complexity = None
        is_failed = False
        error: str | None = None
        failure_type: str | None = None
        received_final = False
        no_context_streak = 0

        start_total = time.perf_counter()
        reasoning_step = 0

        def _format_retry(reason: str, *, thought: str = "Protocol violation") -> bool:
            nonlocal num_format_retries
            num_format_retries += 1
            format_retry_reasons.append(reason)
            msg = react_format_retry_observation(reason)
            if num_format_retries > self.max_format_retries:
                return False
            steps.append(
                ReActStep(
                    step_num=reasoning_step + 1,
                    thought=thought,
                    action="format_retry",
                    action_input=reason,
                    observation=msg,
                )
            )
            return True

        while reasoning_step < self.max_steps:
            if is_failed or received_final:
                break

            try:
                messages = self._build_messages(query, context, steps)
                start_llm = time.perf_counter()
                response = self._call_model(messages)
                llm_output = response.choices[0].message.content
                total_latency_llm += time.perf_counter() - start_llm
                usage = getattr(response, "usage", None)
                if usage is not None:
                    total_prompt_tokens += getattr(usage, "prompt_tokens", 0) or 0
                    total_completion_tokens += getattr(usage, "completion_tokens", 0) or 0
                num_llm_calls += 1
            except Exception as e:
                is_failed = True
                error = str(e)
                failure_type = "api_error"
                break

            if llm_output is None or not str(llm_output).strip():
                is_failed = True
                error = "LLM returned empty content"
                failure_type = "empty_llm_output"
                break

            parsed = self.parser.parse(llm_output)

            if parsed["type"] == "invalid":
                if not _format_retry(
                    parsed.get("reason", "invalid_format"),
                    thought=parsed.get("thought") or "Invalid turn",
                ):
                    is_failed = True
                    error = "Exceeded max format retries"
                    failure_type = "parser_failure"
                continue

            if parsed["type"] == "final":
                final_answer, conf_parsed, comp_parsed, reject = self._accept_final_body(
                    parsed["answer"]
                )
                confidence = conf_parsed
                complexity = comp_parsed
                if parsed.get("confidence") is not None:
                    confidence = parsed["confidence"]
                if parsed.get("complexity") is not None:
                    complexity = parsed["complexity"]

                if reject is not None:
                    if not _format_retry(reject, thought="Final answer rejected"):
                        is_failed = True
                        error = "Exceeded max format retries"
                        failure_type = "parser_failure"
                    continue

                received_final = True
                break

            action = parsed["action"]
            action_input = parsed["action_input"]
            norm_key = (action, self._normalize_tool_input(action_input))
            hist_pairs = [(a, n) for a, n, _ in tool_history]

            if self._is_repeat_query(hist_pairs, action, action_input):
                streak = repeat_block_streak.get(norm_key, 0)
                notice = self._loop_notice_for_repeat(
                    tool_history, action, action_input, streak,
                )
                repeat_block_streak[norm_key] = streak + 1
                steps.append(
                    ReActStep(
                        step_num=reasoning_step + 1,
                        thought=parsed.get("thought") or "Repeated query",
                        action=action,
                        action_input=action_input,
                        observation=notice,
                    )
                )
                reasoning_step += 1
                continue

            repeat_block_streak.pop(norm_key, None)

            observation, tool_latency = self._execute_tool(action, action_input, context)
            if action == "python_exec" and self._python_exec_output_empty(observation):
                observation = self._append_observation_notice(
                    observation,
                    "EMPTY_OUTPUT: script ran but printed nothing. Use print() and retry.",
                )
            elif action == "python_exec" and self._python_exec_output_weak(observation):
                observation = self._append_observation_notice(
                    observation,
                    "WEAK_OUTPUT: script printed None or empty. Fix logic, use fetched evidence, or try another approach.",
                )
            observation = self._maybe_stagnant_loop_notice(
                tool_history, action, action_input, observation,
            )
            if reasoning_step >= self.max_steps - 2:
                observation = self._append_observation_notice(
                    observation,
                    react_loop_observation("finalize_pressure"),
                )
            if action == "math_tool":
                n_math = sum(1 for t in tools_called if t == "math_tool") + 1
                if n_math >= _MATH_TOOL_NUDGE_AFTER:
                    observation = self._append_observation_notice(
                        observation,
                        react_loop_observation("math_finalize_nudge"),
                    )

            tool_history.append(
                (action, self._normalize_tool_input(action_input), observation)
            )
            total_latency_tool += tool_latency
            tools_called.append(action)
            tools_results.append({
                "step": reasoning_step + 1,
                "action": action,
                "input": action_input,
                "observation": observation,
            })
            steps.append(
                ReActStep(
                    step_num=reasoning_step + 1,
                    thought=parsed.get("thought") or "",
                    action=action,
                    action_input=action_input,
                    observation=observation,
                )
            )

            if self._observation_is_no_context(observation):
                no_context_streak += 1
                if no_context_streak >= 2:
                    is_failed = True
                    error = "Dataset context missing (retrieve returned NO_CONTEXT)"
                    failure_type = "missing_context"
                    break
            else:
                no_context_streak = 0

            reasoning_step += 1

        tool_steps = [s for s in steps if s.action != "format_retry"]
        steps_taken = len(tools_called)
        stopped_early = (
            not str(final_answer or "").strip()
            and reasoning_step >= self.max_steps
        )
        if not is_failed and not str(final_answer or "").strip():
            is_failed = True
            error = error or "Step budget exhausted without final answer"
            failure_type = failure_type or "step_budget_exhausted"

        return self._core_response(
            query=query,
            agent="react",
            agent_id="react_004",
            predicted_answer=final_answer,
            latency_total=time.perf_counter() - start_total,
            latency_llm=total_latency_llm,
            latency_tools=total_latency_tool,
            expected_answer=expected_answer,
            is_failed=is_failed,
            error=error,
            failure_type=failure_type,
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            confidence=confidence,
            complexity=complexity,
            reasoning_steps=[
                f"[{s.step_num}] {s.thought}" for s in tool_steps if s.thought
            ],
            format_retry_reasons=format_retry_reasons,
            tools_available=list(self.tools.keys()),
            tools_called=tools_called,
            tools_results=tools_results,
            num_tool_calls=len(tools_called),
            num_llm_calls=num_llm_calls,
            num_steps=steps_taken,
            max_steps=self.max_steps,
            steps_taken=steps_taken,
            num_format_retries=num_format_retries,
            is_stopped_early=stopped_early,
        )


# ── Policy → prompts → tools → agent ─────────────────────────────────────────

_TOOL_MODULE_NAMES: tuple[str, ...] = (
    "retrieve",
    "web_search",
    "web_fetch",
    "wikipedia_search",
    "arxiv_search",
    "github_search",
    "pdb_parse",
    "math_tool",
    "read_file",
    "python_exec",
)


def _build_tools_registry() -> dict[str, Callable[..., Any]]:
    from agent import tools as tool_mod

    registry: dict[str, Callable[..., Any]] = {}
    for attr in _TOOL_MODULE_NAMES:
        fn = getattr(tool_mod, attr)
        name = str(getattr(fn, "_tool_name", attr)).lower()
        registry[name] = fn
    return registry


def _tool_spec(fn: Callable[..., Any]) -> ToolSpec:
    import inspect

    name = str(getattr(fn, "_tool_name", fn.__name__)).lower()
    uses_context = bool(getattr(fn, "_uses_context", False))
    params = list(inspect.signature(fn).parameters.values())
    if not uses_context and len(params) >= 2:
        uses_context = any(p.name == "context" for p in params)

    def _run(tool_input: str, context: dict[str, Any]) -> str:
        if uses_context:
            return str(fn(tool_input, context))
        return str(fn(tool_input))

    return ToolSpec(
        name=name,
        description=str(getattr(fn, "_tool_description", name)),
        fn=_run,
    )


def tool_specs_from_policy(policy) -> list[ToolSpec]:
    from agent.dataset_policy import DatasetPolicy

    if not isinstance(policy, DatasetPolicy):
        raise TypeError(f"expected DatasetPolicy, got {type(policy)!r}")

    registry = _build_tools_registry()
    allowed = {n.lower() for n in policy.allowed_tools}
    return [_tool_spec(registry[name]) for name in sorted(registry) if name in allowed]


def tools_for_dataset(dataset: str, policy=None) -> dict[str, Callable[..., Any]]:
    from agent.dataset_policy import DatasetPolicy, default_dataset_policy

    pol = policy if isinstance(policy, DatasetPolicy) else default_dataset_policy(dataset)
    registry = _build_tools_registry()
    allowed = {n.lower() for n in pol.allowed_tools}
    return {k: v for k, v in registry.items() if k in allowed}


def react_tool_names_for_dataset(dataset: str) -> list[str]:
    return sorted(tools_for_dataset(dataset).keys())


def build_react_agent(
    model: str,
    dataset: str,
    *,
    policy=None,
    tools: list[ToolSpec] | None = None,
    **kwargs: Any,
) -> ReActAgent:
    from agent.config import normalize_agent_config
    from agent.dataset_policy import DatasetPolicy, default_dataset_policy
    from agent.dataset_profile import apply_react_profile, extract_hop_metadata
    from input.prompts.prompts import build_react_system, user_prompt

    cfg = normalize_agent_config(
        model=model,
        dataset=dataset,
        kwargs=kwargs,
        strategy="react",
    ).config
    pol: DatasetPolicy = (
        policy if isinstance(policy, DatasetPolicy) else default_dataset_policy(cfg.dataset)
    )
    if kwargs.get("evidence_mode") is not None:
        pol = pol.with_overrides(evidence_mode=kwargs["evidence_mode"])
    if kwargs.get("allowed_tools") is not None:
        raw = kwargs["allowed_tools"]
        pol = pol.with_overrides(allowed_tools=frozenset(str(t).lower() for t in raw))

    n_hops, _ = extract_hop_metadata(kwargs)
    params = apply_react_profile(cfg.dataset, cfg.agent_params, n_hops=n_hops)
    max_steps = int(kwargs.get("max_steps") or params.get("max_steps") or pol.max_steps or 8)
    max_format_retries = int(
        kwargs.get("max_format_retries") or params.get("max_format_retries") or 3
    )

    specs = tools if tools is not None else tool_specs_from_policy(pol)

    return ReActAgent(
        model=cfg.model,
        tools=specs,
        system_prompt=build_react_system(cfg.dataset),
        user_prompt=user_prompt(cfg.dataset),
        max_steps=max_steps,
        max_format_retries=max_format_retries,
        dataset=cfg.dataset,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        seed=cfg.seed,
    )


def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
    agent = build_react_agent(model=model, dataset=dataset, **kwargs)
    return agent.run(query, **kwargs)
