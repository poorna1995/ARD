"""
multiagent.py — Three-lane blackboard multi-agent QA pipeline
=============================================================

Architecture
------------
  Planner  →  dependency-wave Workers  →  Critic  →  Synthesizer

Per-subtask, three lanes race concurrently:

  Retrieval  : tool (web/wiki/arxiv/github) → LLM extraction
  Execution  : tool (math/file/pdb)         → raw output IS the answer
  Reasoning  : pure LLM from goal + context (always active)

Lane reconciler: execution > retrieval > reasoning (tie-broken by confidence)

Tool-calling convention (aligned with ReactAgent)
--------------------------------------------------
Every reference to a tool in prompts and in user-messages sent to worker LLMs
uses the canonical  tool_name[input]  syntax that ReactAgent established.

  planner JSON  :  {"tool_name": "wikipedia", "goal": "Apollo 11 moon landing"}
  worker prompt :  Executed: wikipedia[Apollo 11 moon landing]
  tools_block   :  "  wikipedia       : ..."   (identical format to React)

The `goal` field in the planner JSON IS the [input] string — the two are
interchangeable. This makes it trivial to read plans produced by either agent.
"""

from __future__ import annotations

import concurrent.futures
import inspect
import json
import re
import time
import warnings
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Callable

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
from prompts.prompts import format_multiagent1_coord_system

# ── tuneable constants ────────────────────────────────────────────────────────
DEFAULT_MAX_STEPS       = 10
DEFAULT_MAX_TOOL_CALLS  = 8
DEFAULT_MAX_WORKERS     = 4
DEFAULT_MAX_SUBTASKS    = 5
DEFAULT_WORKER_RETRIES  = 0

_TOOL_BLOB_MAX_CHARS    = 8_000
_EVIDENCE_MAX_CHARS     = 2_000
_MAX_SEARCH_INPUT_CHARS = 2_500

# ── lane priority (higher = preferred by reconciler) ─────────────────────────
_LANE_PRIORITY: dict[str, int] = {"execution": 3, "retrieval": 2, "reasoning": 1}

# ── tool category membership ──────────────────────────────────────────────────
_RETRIEVAL_TOOLS: set[str] = {
    "web_search", "web_fetch",
    "wikipedia_search",
    "arxiv_search",
    "github_search",
}
_EXECUTION_TOOLS: set[str] = {
    "math_tool",
    "read_file",
    "pdb_parse",
}
_URL_TOOLS:  set[str] = {"web_fetch"}
_FILE_TOOLS: set[str] = {"read_file", "pdb_parse"}

_NULL_ANSWERS: frozenset[str] = frozenset({"unknown", "error", "none", "null", ""})


# ═════════════════════════════════════════════════════════════════════════════
# Tool registry  (identical pattern to ReactAgent)
# ═════════════════════════════════════════════════════════════════════════════

@tool("finish", 'Submit the final answer. Input: JSON string {"answer": "<value>"}.')
def finish(answer: str) -> str:   # noqa: D401
    return answer


def _build_tools_list() -> list[Callable]:
    return [
        web_search,
        web_fetch,
        wikipedia_search,
        arxiv_search,
        github_search,
        pdb_parse,
        read_file,
        math_tool,
        # finish omitted from worker tool-set
    ]


TOOLS: dict[str, Callable] = {
    fn._tool_name: fn
    for fn in _build_tools_list()
}


# ═════════════════════════════════════════════════════════════════════════════
# Small utilities
# ═════════════════════════════════════════════════════════════════════════════

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _strip_preamble(text: str) -> str:
    text = (text or "").strip()
    for pattern in (
        r"(?i)^the\s+final\s+answer\s+is\s*[:\-]?\s*",
        r"(?i)^final\s+answer\s*[:\-]\s*",
        r"(?i)^the\s+answer\s+is\s*[:\-]?\s*",
        r"(?i)^answer\s*[:\-]\s*",
    ):
        text = re.sub(pattern, "", text).strip()
    return text


def _parse_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _strip_outer_quotes(s: str) -> str:
    url = (s or "").strip()
    for _ in range(3):
        if len(url) >= 2 and url[0] == url[-1] and url[0] in "\"'":
            url = url[1:-1].strip()
        else:
            break
    return url


def _first_http_url(text: str) -> str | None:
    match = re.search(r"https?://[^\s\]>\"']+", text or "")
    return match.group(0) if match else None


def _merge_goal_and_context(goal: str, context: str) -> str:
    g, c = (goal or "").strip(), (context or "").strip()
    return f"{g}\n\n--- Dependency context ---\n{c}" if c else g


def _cap_text(s: str, limit: int) -> str:
    s = (s or "").strip()
    if len(s) <= limit:
        return s
    cut = s[:limit].rsplit("\n", 1)[0].strip()
    return cut if cut else s[:limit]


def _url_for_web_fetch(goal: str, context: str = "") -> str:
    for chunk in (goal, _merge_goal_and_context(goal, context)):
        url = _strip_outer_quotes(chunk.strip())
        if url.startswith(("http://", "https://")):
            return url.split()[0].rstrip(").,;]\"")
        hit = _first_http_url(chunk)
        if hit:
            return hit.rstrip(").,;]\"")
    return ""


# ═════════════════════════════════════════════════════════════════════════════
# Tool invocation
# ═════════════════════════════════════════════════════════════════════════════

def _invoke_tool(
    tool_name: str,
    tool_fn: Callable[..., Any],
    goal: str,
    context: str,
) -> Any:
    """
    Call `tool_fn` with the correct single argument.

    Conceptual model (matching ReactAgent):
        tool_name[goal]   ←→   tool_fn(goal)

    The `goal` field from the planner JSON IS the [input] string.
    Category-specific preprocessing resolves the raw goal to the right
    value for each tool type — exactly as ReactAgent's _call_tool does
    before dispatching fn(action_input):

      - URL tools (web_fetch)  → extract bare http(s) URL from goal
      - File tools             → bare goal string (filename / path)
      - Everything else        → merged goal + dependency context (capped)
    """
    name = (tool_name or "").strip().lower()

    if name in _URL_TOOLS:
        tool_input = _url_for_web_fetch(goal, context)
        if not tool_input.startswith(("http://", "https://")):
            return (
                f"Error: {name} requires an http(s) URL as [input]. "
                "No URL found in goal or dependency context."
            )
    elif name in _FILE_TOOLS:
        tool_input = (goal or "").strip()
    else:
        tool_input = _cap_text(_merge_goal_and_context(goal, context), _MAX_SEARCH_INPUT_CHARS)

    # Bind to the first required (or first positional) parameter.
    _PRIMARY_PARAM_NAMES = (
        "query", "url", "tool_input", "user_input",
        "filename", "path", "file_path", "input", "text", "code",
        "expression", "pdb_id", "protein_id",
    )
    try:
        sig = inspect.signature(tool_fn)
    except (TypeError, ValueError):
        return tool_fn(tool_input)

    positional = [
        p for p in sig.parameters.values()
        if p.kind in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]
    required = [p for p in positional if p.default is inspect.Parameter.empty]

    if len(required) == 1:
        return tool_fn(**{required[0].name: tool_input})
    if not required and positional:
        return tool_fn(**{positional[0].name: tool_input})
    required_names = {p.name for p in required}
    for pname in _PRIMARY_PARAM_NAMES:
        if pname in required_names:
            return tool_fn(**{pname: tool_input})
    if required:
        return tool_fn(**{required[0].name: tool_input})

    raise TypeError(
        f"Cannot determine binding parameter for tool {tool_name!r}. "
        f"Parameters: {[p.name for p in positional]}"
    )


def _is_hard_tool_failure(tool_name: str, output: str) -> bool:
    t = (output or "").lower()
    if any(phrase in t for phrase in (
        "arxiv tool unavailable", "arxiv search failed",
        "error: empty arxiv query", "install with:",
    )):
        return True
    if tool_name == "web_fetch" and any(phrase in t for phrase in (
        "fetch_failed", "unknown url type", "no url found",
        "http error 429", "http error 403", "http error 404",
    )):
        return True
    if tool_name == "web_fetch" and t.lstrip().startswith("error:"):
        return True
    return False


def _tool_output_is_relevant(goal: str, tool_output: str) -> bool:
    _STOP = frozenset({
        "the", "a", "an", "of", "in", "on", "to", "for", "is", "was", "are",
        "and", "or", "with", "from", "at", "by", "as", "that", "this", "it",
        "find", "get", "what", "which", "where", "how", "identify", "extract",
        "according", "before", "after", "about", "paper", "article", "show",
        "shows", "used", "use", "uses", "type", "types", "word", "words",
    })
    tokens = [
        w.lower().strip(".,?!:;\"'()")
        for w in goal.split()
        if len(w) > 3 and w.lower().strip(".,?!") not in _STOP
    ]
    if not tokens:
        return True
    output_lower = tool_output.lower()
    hits = sum(1 for t in tokens if t in output_lower)
    return hits >= min(2, len(tokens))


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class LaneResult:
    lane:       str
    answer:     str
    confidence: float
    raw:        str | None  = None
    evidence:   list[str]   = field(default_factory=list)
    tool_name:  str | None  = None
    skipped:    bool        = False
    error:      str | None  = None

    @classmethod
    def skipped_lane(cls, lane: str) -> "LaneResult":
        return cls(lane=lane, answer="", confidence=0.0, skipped=True)

    @classmethod
    def failed_lane(cls, lane: str, tool_name: str, error: str) -> "LaneResult":
        return cls(lane=lane, answer="unknown", confidence=0.05,
                   tool_name=tool_name, error=error,
                   raw='{"answer":"unknown","confidence":0.05,"evidence":["tool failed"]}')


@dataclass
class Subtask:
    id:          str
    goal:        str          # = the [input] string for the assigned tool
    needs_tool:  bool      = False
    tool_name:   str       = "none"
    depends_on:  list[str] = field(default_factory=list)

    # Convenience: emit the canonical tool_name[goal] representation
    def tool_call_repr(self) -> str:
        if self.needs_tool and self.tool_name != "none":
            return f"{self.tool_name}[{self.goal}]"
        return f"none[{self.goal}]"

    def deps_satisfied(self, completed: set[str]) -> bool:
        return all(d in completed for d in self.depends_on)


@dataclass
class WorkerResult:
    subtask_id:   str
    answer:       str
    confidence:   float
    used_tool:    bool       = False
    tools_used:   list[str]  = field(default_factory=list)
    tool_calls:   int        = 0
    llm_calls:    int        = 1
    prompt_tok:   int        = 0
    compl_tok:    int        = 0
    latency:      float      = 0.0
    tool_latency: float      = 0.0
    cost:         float      = 0.0
    error:        str | None = None
    evidence:     list[str]  = field(default_factory=list)
    raw:          str | None = None
    winning_lane: str        = "reasoning"


@dataclass
class _Usage:
    prompt:     int   = 0
    completion: int   = 0
    cost:       float = 0.0

    def add(self, other: "_Usage") -> None:
        self.prompt     += other.prompt
        self.completion += other.completion
        self.cost       += other.cost


# ═════════════════════════════════════════════════════════════════════════════
# Blackboard  (thread-safe shared state)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class Blackboard:
    goal:    str
    records: dict[str, dict[str, Any]] = field(default_factory=dict)
    _lock:   Lock = field(default_factory=Lock, repr=False, compare=False)

    def write(
        self,
        subtask_id: str,
        answer: str,
        confidence: float,
        *,
        used_tool:  bool       = False,
        error:      str | None = None,
        evidence:   list[str] | None = None,
        raw:        str | None = None,
    ) -> None:
        record = {
            "answer":     (answer or "").strip(),
            "confidence": float(confidence),
            "used_tool":  bool(used_tool),
            "error":      error,
            "evidence":   list(evidence or []),
            "raw":        raw,
        }
        with self._lock:
            existing = self.records.get(subtask_id)
            if existing is None or record["confidence"] >= float(existing.get("confidence", -1.0)):
                self.records[subtask_id] = record

    write_subtask_result = write
    record               = write

    def dependency_context(self, dep_ids: list[str]) -> str:
        with self._lock:
            prior = {k: self.records[k] for k in dep_ids if k in self.records}
        if not prior:
            return ""
        return json.dumps({"goal": self.goal, "prior": prior}, separators=(",", ":"))

    context_for = dependency_context

    def snapshot_json(self) -> str:
        with self._lock:
            payload = {"goal": self.goal, "records": self.records}
        return json.dumps(payload, separators=(",", ":"))

    all_facts = snapshot_json

    def critic_summary_json(self) -> str:
        with self._lock:
            slim = {
                k: {"answer": v.get("answer"), "confidence": v.get("confidence")}
                for k, v in self.records.items()
            }
        return json.dumps({"goal": self.goal, "records": slim}, separators=(",", ":"))


Memory = Blackboard


# ═════════════════════════════════════════════════════════════════════════════
# MultiAgentAgent
# ═════════════════════════════════════════════════════════════════════════════

class MultiAgentAgent(BaseAgent):
    """
    Blackboard multi-agent QA with three concurrent lanes per subtask.

    Tool-calling convention
    -----------------------
    Identical to ReactAgent:

      tools_block  →  "  tool_name : description"   (one line per tool)
      planner      →  goal field  ==  the [input] passed to tool_name
      worker prompt→  "Executed: tool_name[goal]"
      dispatch     →  _invoke_tool(name, fn, goal, ctx)  ≡  fn(goal)
    """

    _ALIASES: dict[str, str] = {
        "calculator":       "math_tool",
        "math":             "math_tool",
        "calculate":        "math_tool",
        "search":           "web_search",
        "google":           "web_search",
        "web":              "web_search",
        "fetch":            "web_fetch",
        "url":              "web_fetch",
        "browse":           "web_fetch",
        "wiki":             "wikipedia_search",
        "wikipedia":        "wikipedia_search",
        "wikipedia_search": "wikipedia_search",
        "arxiv":            "arxiv_search",
        "paper":            "arxiv_search",
        "papers":           "arxiv_search",
        "github":           "github_search",
        "code_search":      "github_search",
        "file_read":        "read_file",
        "file":             "read_file",
        "pdb":              "pdb_parse",
        "protein":          "pdb_parse",
    }

    def __init__(
        self,
        model:          str,
        dataset:        str,
        *,
        tools:          dict[str, Callable] | None = None,
        max_steps:      int = DEFAULT_MAX_STEPS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_workers:    int = DEFAULT_MAX_WORKERS,
        max_subtasks:   int = DEFAULT_MAX_SUBTASKS,
        worker_retries: int = DEFAULT_WORKER_RETRIES,
        planner_model:  str = "gpt-4o-mini",
        worker_model:   str = "gpt-4o-mini",
        verifier_model: str = "gpt-4o-mini",
        **kwargs: Any,
    ) -> None:
        from prompts.prompts import (
            MULTIAGENT1_CRITIC_SYSTEM,
            MULTIAGENT1_PLANNER_SYSTEM,
            MULTIAGENT1_WORKER_EXTRACTION_SYSTEM,
            MULTIAGENT1_WORKER_REASONING_SYSTEM,
            USER_PROMPT,
        )

        self.dataset = dataset
        self.tools   = {k.lower(): v for k, v in (tools or TOOLS).items()}
        tools_block = self._build_tools_block()
        multiagent_system = format_multiagent1_coord_system(dataset, tools_block=tools_block)

        super().__init__(
            model=model,
            system_prompt=multiagent_system,
            user_prompt=USER_PROMPT[dataset]["multiagent"],
            **kwargs,
        )
        self.max_steps      = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_workers    = max_workers
        self.max_subtasks   = max_subtasks
        self.worker_retries = worker_retries
        self.planner_model  = planner_model
        self.worker_model   = worker_model
        self.verifier_model = verifier_model

        self._planner_system    = MULTIAGENT1_PLANNER_SYSTEM
        self._reasoning_system  = MULTIAGENT1_WORKER_REASONING_SYSTEM
        self._extraction_system = MULTIAGENT1_WORKER_EXTRACTION_SYSTEM
        self._critic_system     = MULTIAGENT1_CRITIC_SYSTEM
        self._coord_system      = multiagent_system
        self._user_prompt_tpl   = USER_PROMPT[dataset]["multiagent"]

    # ── Internal helpers ──────────────────────────────────────────────────────
    def _resolve_goal_with_deps(self, subtask: Subtask, board: Blackboard) -> str:
        """
        Replace dependency placeholders in the goal with actual answers.
        e.g. goal="fetch {s1.answer}" becomes "fetch https://arxiv.org/abs/2207.01510"
        """
        goal = subtask.goal
        with board._lock:
            for dep_id in subtask.depends_on:
                if dep_id in board.records:
                    dep_answer = board.records[dep_id].get("answer", "")
                    goal = goal.replace(f"{{{dep_id}}}", dep_answer)
                    goal = goal.replace(f"{{{dep_id}.answer}}", dep_answer)
        return goal
    def _build_tools_block(self) -> str:
        """
        Build the tools block injected into every prompt.

        Format is IDENTICAL to ReactAgent._build_tools_block():
            "  {name:15s}: {description}"

        No suffix is appended — the coordinator template provides all context.
        """
        return "\n".join(
            f"  {name:15s}: {getattr(fn, '_tool_description', 'No description.')}"
            for name, fn in self.tools.items()
        )

    def _resolve_tool_name(self, raw: str) -> str:
        lowered   = raw.strip().lower()
        candidate = self._ALIASES.get(lowered, lowered)
        return candidate if candidate in self.tools else "none"

    def _classify_tool_lane(self, tool_name: str) -> str:
        name = (tool_name or "").strip().lower()
        if name in _RETRIEVAL_TOOLS:
            return "retrieval"
        if name in _EXECUTION_TOOLS:
            return "execution"
        return "none"

    def _validate_subtasks(self, subtasks: list[Subtask]) -> list[Subtask]:
        out: list[Subtask] = []
        for s in subtasks:
            if s.needs_tool:
                resolved   = self._resolve_tool_name(s.tool_name)
                needs_tool = resolved != "none"
                tool_name  = resolved
            else:
                needs_tool, tool_name = False, "none"
            out.append(Subtask(
                id=s.id, goal=s.goal,
                needs_tool=needs_tool, tool_name=tool_name,
                depends_on=list(s.depends_on),
            ))
        return out

    def _parse_worker_json(self, raw: str, default_confidence: float) -> tuple[str, float]:
        parsed     = _parse_json(raw) or {}
        answer     = _strip_preamble(str(parsed.get("answer", raw) or "")).strip()
        confidence = _safe_float(parsed.get("confidence"), default=default_confidence)
        if not answer:
            return "unknown", min(confidence, 0.2)
        return answer, confidence

    # ── LLM call wrapper ──────────────────────────────────────────────────────

    def _call_with_messages(
        self,
        system:      str,
        user:        str,
        model:       str | None = None,
        temperature: float      = 0.2,
        max_tokens:  int        = 700,
    ) -> tuple[str, float, Any]:
        effective_model = model or self.model
        client          = self._get_client_for_model(effective_model)
        t0              = time.perf_counter()
        response        = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or "", time.perf_counter() - t0, response

    def _usage_from_response(self, model_name: str, response: Any) -> _Usage:
        if response is None:
            return _Usage()
        try:
            u = self._get_usage(response)
        except Exception:
            return _Usage()
        pt   = int(u.get("prompt_tokens", 0))
        ct   = int(u.get("completion_tokens", 0))
        cost = self._compute_cost_for_model(model_name, pt, ct)
        return _Usage(prompt=pt, completion=ct, cost=cost)

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 1 — Planner
    # ═════════════════════════════════════════════════════════════════════════

    def build_planner_messages(self, query: str) -> tuple[str, str]:
        tools_block = self._build_tools_block()
        planner_system = self._planner_system.replace("{tools_block}", tools_block)
        assert "{tools_block}" not in planner_system, "placeholder not replaced"
        assert "{" not in planner_system.split("Return ONLY")[0] or True  # optional deep check
        system = f"{self._coord_system}\n\n{planner_system}\n"
        user = self._user_prompt_tpl.format(query=query)
        return system, user
    def _stage_planner(self, query: str) -> tuple[list[Subtask], _Usage]:
        system, user = self.build_planner_messages(query)
        raw, _, response = self._call_with_messages(
            system, user,
            model=self.planner_model,
            max_tokens=520,
            temperature=0.1,
        )
        usage = self._usage_from_response(self.planner_model, response)

        parsed       = _parse_json(raw) or {}
        raw_subtasks = parsed.get("subtasks", [])

        subtasks: list[Subtask] = []
        for i, s in enumerate(raw_subtasks[: self.max_subtasks], start=1):
            if not isinstance(s, dict):
                continue
            goal = str(s.get("goal", "")).strip()
            if not goal:
                continue
            subtasks.append(Subtask(
                id=str(s.get("id", f"s{i}")),
                goal=goal,
                needs_tool=bool(s.get("needs_tool", False)),
                tool_name=str(s.get("tool_name", "none")).strip() or "none",
                depends_on=[str(d) for d in s.get("depends_on", []) if d],
            ))

        if not subtasks:
            subtasks = [Subtask(id="s1", goal=query, needs_tool=True, tool_name="web_search")]

        return subtasks, usage

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 2a — Three lanes
    # ═════════════════════════════════════════════════════════════════════════

    def _lane_retrieval(self, subtask: Subtask, board: Blackboard) -> LaneResult:
        """
        Retrieval lane: run tool_name[goal] then call extraction LLM.

        The worker prompt labels the tool call as:
            Executed: tool_name[goal]
        matching the  Action: tool_name[input]  format React uses.
        """
        if not subtask.needs_tool or subtask.tool_name not in self.tools:
            return LaneResult.skipped_lane("retrieval")
        if self._classify_tool_lane(subtask.tool_name) != "retrieval":
            return LaneResult.skipped_lane("retrieval")

        context = board.dependency_context(subtask.depends_on)
        tool_fn = self.tools[subtask.tool_name]

        try:
            raw_output = str(_invoke_tool(subtask.tool_name, tool_fn, subtask.goal, context))
        except Exception as exc:
            return LaneResult.failed_lane("retrieval", subtask.tool_name, str(exc))

        if _is_hard_tool_failure(subtask.tool_name, raw_output):
            return LaneResult.failed_lane(
                "retrieval", subtask.tool_name,
                f"tool_hard_failure: {raw_output[:200]}",
            )

        if not _tool_output_is_relevant(subtask.goal, raw_output):
            return LaneResult(
                lane="retrieval",
                answer="unknown",
                confidence=0.05,
                raw='{"answer":"unknown","confidence":0.05,"evidence":["tool output irrelevant to goal"]}',
                evidence=[raw_output[:_EVIDENCE_MAX_CHARS]],
                tool_name=subtask.tool_name,
                error="irrelevant_tool_output",
            )

        # ── Extraction LLM ────────────────────────────────────────────────────
        # Header uses  tool_name[goal]  — same convention as React's Action line
        tool_blob = raw_output[:_TOOL_BLOB_MAX_CHARS]
        executed_label = f"Executed: {subtask.tool_name}[{subtask.goal}]"

        if tool_blob.strip():
            user_msg = (
                f"{executed_label}\n\n"
                f"Context:\n{context}\n\n"
                f"Tool output:\n{tool_blob}"
            )
        else:
            user_msg = (
                "CRITICAL: Tool returned no output. Do not fabricate.\n\n"
                f"{executed_label}\n"
                f"Context:\n{context}"
            )

        raw, _, _ = self._call_with_messages(
            self._extraction_system, user_msg,
            model=self.worker_model, temperature=0.2,
        )
        answer, confidence = self._parse_worker_json(raw, default_confidence=0.5)

        return LaneResult(
            lane="retrieval",
            answer=answer,
            confidence=confidence,
            raw=raw,
            evidence=[raw_output[:_EVIDENCE_MAX_CHARS]],
            tool_name=subtask.tool_name,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _lane_reasoning(self, subtask: Subtask, board: Blackboard) -> LaneResult:
        """
        Reasoning lane: pure LLM inference from goal + dependency context.
        """
        context = board.dependency_context(subtask.depends_on)
        user_msg = (
            "Answer using only your reasoning and the prior context below.\n\n"
            f"Prior context:\n{context}\n\n"
            f"Subtask: {subtask.goal}"
        )
        raw, _, _ = self._call_with_messages(
            self._reasoning_system, user_msg,
            model=self.worker_model, temperature=0.2,
        )
        answer, confidence = self._parse_worker_json(raw, default_confidence=0.4)
        
        if answer and context:
            if answer.lower() in context.lower() and len(answer) < 50:
                confidence = min(confidence, 0.12)
        if subtask.needs_tool and subtask.tool_name != "none":
            confidence = min(confidence, 0.45)

        if answer and answer.lower() in subtask.goal.lower():
            confidence = min(confidence, 0.15)

        if subtask.depends_on:
            with board._lock:
                dep_confs = [
                    float(board.records[d].get("confidence", 1.0))
                    for d in subtask.depends_on
                    if d in board.records
                ]
            if dep_confs:
                max_dep_conf = max(dep_confs)
                if max_dep_conf < 0.4:
                    confidence = min(confidence, max_dep_conf + 0.1)

        return LaneResult(lane="reasoning", answer=answer, confidence=confidence, raw=raw)

    # ──────────────────────────────────────────────────────────────────────────

    def _lane_execution(self, subtask: Subtask, board: Blackboard) -> LaneResult:
        """
        Execution lane: run tool_name[goal] for deterministic tools.

        Same dispatch model as ReactAgent._call_tool — tool returns one string,
        no extraction LLM needed.
        """
        if not subtask.needs_tool or subtask.tool_name not in self.tools:
            return LaneResult.skipped_lane("execution")
        if self._classify_tool_lane(subtask.tool_name) != "execution":
            return LaneResult.skipped_lane("execution")

        context = board.dependency_context(subtask.depends_on)
        tool_fn = self.tools[subtask.tool_name]

        try:
            raw_output = str(_invoke_tool(subtask.tool_name, tool_fn, subtask.goal, context)).strip()
        except Exception as exc:
            return LaneResult.failed_lane("execution", subtask.tool_name, str(exc))

        answer     = _strip_preamble(raw_output) if raw_output else "unknown"
        confidence = 0.85 if answer and answer != "unknown" else 0.1

        return LaneResult(
            lane="execution",
            answer=answer,
            confidence=confidence,
            raw=raw_output,
            evidence=[raw_output[:_EVIDENCE_MAX_CHARS]],
            tool_name=subtask.tool_name,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Lane reconciler
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _reconcile(
        retrieval: LaneResult,
        reasoning: LaneResult,
        execution: LaneResult,
    ) -> LaneResult:
        # eligible must be defined before any use
        def eligible(lr: LaneResult) -> bool:
            return (
                not lr.skipped
                and bool(lr.answer)
                and lr.answer.lower() not in _NULL_ANSWERS
                and lr.confidence > 0.0
            )

        # Hard tool failure: don't return the failure — fall back to remaining lanes
        if not retrieval.skipped and "tool_hard_failure" in (retrieval.error or ""):
            candidates = [lr for lr in (execution, reasoning) if eligible(lr)]
            return candidates[0] if candidates else reasoning

        candidates = [lr for lr in (execution, retrieval, reasoning) if eligible(lr)]
        if not candidates:
            return reasoning

        candidates.sort(
            key=lambda lr: (_LANE_PRIORITY.get(lr.lane, 0), lr.confidence),
            reverse=True,
        )
        return candidates[0]

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 2b — Single subtask orchestration
    # ═════════════════════════════════════════════════════════════════════════

    def _run_one_subtask(self, subtask: Subtask, board: Blackboard) -> WorkerResult:
        t0 = time.perf_counter()
        resolved_goal = self._resolve_goal_with_deps(subtask, board)
        if resolved_goal != subtask.goal:
            subtask = Subtask(
                id=subtask.id,
                goal=resolved_goal,
                needs_tool=subtask.needs_tool,
                tool_name=subtask.tool_name,
                depends_on=subtask.depends_on,
            )

        exec_lr: LaneResult = LaneResult.skipped_lane("execution")
        retr_lr: LaneResult = LaneResult.skipped_lane("retrieval")
        reas_lr: LaneResult = LaneResult.skipped_lane("reasoning")
        winning: LaneResult = LaneResult.skipped_lane("reasoning")
        last_error: str | None = None
        tool_latency = 0.0

        for attempt in range(self.worker_retries + 1):
            try:
                t_tool = time.perf_counter()

                # All three lanes run concurrently — this is the whole point
                with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
                    f_exec = pool.submit(self._lane_execution, subtask, board)
                    f_retr = pool.submit(self._lane_retrieval, subtask, board)
                    f_reas = pool.submit(self._lane_reasoning, subtask, board)
                    exec_lr = f_exec.result()
                    retr_lr = f_retr.result()
                    reas_lr = f_reas.result()

                tool_latency += time.perf_counter() - t_tool
                winning = self._reconcile(retr_lr, reas_lr, exec_lr)
                last_error = None
                break

            except Exception as exc:
                last_error = str(exc)
                if attempt < self.worker_retries:
                    time.sleep(0.1 * (attempt + 1))

        if last_error is not None:
            reas_lr = self._lane_reasoning(subtask, board)
            winning = reas_lr

        all_evidence: list[str] = []
        tools_used: list[str] = []
        for lr in (exec_lr, retr_lr, reas_lr):
            if not lr.skipped:
                all_evidence.extend(lr.evidence or [])
                if lr.tool_name:
                    tools_used.append(lr.tool_name)

        answer = winning.answer or "unknown"
        confidence = winning.confidence if answer != "unknown" else 0.2
        latency = time.perf_counter() - t0
        llm_calls = 1 + (0 if retr_lr.skipped else 1)

        return WorkerResult(
            subtask_id=subtask.id,
            answer=answer,
            confidence=confidence,
            used_tool=bool(tools_used),
            tools_used=list(dict.fromkeys(tools_used)),
            tool_calls=len(tools_used),
            llm_calls=llm_calls,
            latency=latency,
            tool_latency=tool_latency,
            error=last_error,
            evidence=all_evidence,
            raw=winning.raw,
            winning_lane=winning.lane,
        )

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 2c — Dependency-wave scheduler
    # ═════════════════════════════════════════════════════════════════════════

    def _stage_parallel_workers(
        self, subtasks: list[Subtask], board: Blackboard
    ) -> list[WorkerResult]:
        completed: set[str]         = set()
        pending                     = list(subtasks)
        results: list[WorkerResult] = []

        while pending:
            wave = [s for s in pending if s.deps_satisfied(completed)]

            if not wave:
                warnings.warn(
                    f"Unsatisfiable dependencies among subtasks "
                    f"{[s.id for s in pending]!r}. "
                    "Running all remaining tasks concurrently to break deadlock.",
                    stacklevel=2,
                )
                wave    = pending[:]
                pending = []
            else:
                pending = [s for s in pending if s not in wave]

            if len(wave) == 1:
                wave_results = [self._run_one_subtask(wave[0], board)]
            else:
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=min(self.max_workers, len(wave))
                ) as pool:
                    fut_map = {
                        pool.submit(self._run_one_subtask, s, board): s
                        for s in wave
                    }
                    wave_results = []
                    for fut in concurrent.futures.as_completed(fut_map):
                        s = fut_map[fut]
                        try:
                            wave_results.append(fut.result())
                        except Exception as exc:
                            wave_results.append(WorkerResult(
                                subtask_id=s.id,
                                answer="unknown",
                                confidence=0.0,
                                error=str(exc),
                            ))

            for res in wave_results:
                board.write(
                    res.subtask_id, res.answer, res.confidence,
                    used_tool=res.used_tool, error=res.error,
                    evidence=res.evidence, raw=res.raw,
                )
                completed.add(res.subtask_id)
                results.append(res)

        return results

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 3 — Critic
    # ═════════════════════════════════════════════════════════════════════════

    def _stage_critic(
        self, query: str, results: list[WorkerResult], board: Blackboard
    ) -> tuple[str, float, _Usage]:
        valid = [
            r for r in results
            if r.answer and r.answer.lower() not in _NULL_ANSWERS
        ]

        if not valid:
            with board._lock:
                mem_vals = list(board.records.values())
            mem_valid = [r for r in mem_vals if str(r.get("answer", "")).strip()]
            if not mem_valid:
                return "unknown", 0.0, _Usage()
            best = max(mem_valid, key=lambda r: float(r.get("confidence", 0.0)))
            return str(best.get("answer", "unknown")).strip(), float(best.get("confidence", 0.0)), _Usage()

        if (
            len(valid) == 1
            and valid[0].winning_lane == "execution"
            and valid[0].confidence >= 0.8
            and re.fullmatch(r"[\d.]+", valid[0].answer.strip())
        ):
            return valid[0].answer.strip(), valid[0].confidence, _Usage()

        observations = "\n".join(
            f"[{r.subtask_id}] lane={r.winning_lane} conf={r.confidence:.2f} "
            f"tool={r.used_tool} → {r.answer}"
            for r in valid
        )
        user = (
            f"Original question: {query}\n\n"
            f"Worker observations:\n{observations}\n\n"
            f"Memory:\n{board.critic_summary_json()}"
        )
        raw, _, response = self._call_with_messages(
            self._critic_system, user,
            model=self.verifier_model,
            max_tokens=500,
            temperature=0.0,
        )
        usage = self._usage_from_response(self.verifier_model, response)

        parsed    = _parse_json(raw) or {}
        answer    = _strip_preamble(str(parsed.get("answer", "")).strip())
        consensus = _safe_float(parsed.get("consensus_score"), default=0.5)

        all_same       = len({r.answer for r in valid}) == 1
        tool_backed    = [r for r in valid if r.used_tool and r.winning_lane != "reasoning"]
        reasoning_only = [r for r in valid if not r.used_tool]
        if all_same and not tool_backed and reasoning_only:
            consensus = min(consensus, 0.3)

        if not answer or answer.lower() in _NULL_ANSWERS:
            best      = max(valid, key=lambda r: r.confidence)
            answer    = best.answer
            consensus = best.confidence

        return answer, consensus, usage

    # ═════════════════════════════════════════════════════════════════════════
    # Stage 4 — Synthesizer
    # ═════════════════════════════════════════════════════════════════════════

    @staticmethod
    def _stage_synthesizer(draft: str) -> str:
        return _strip_preamble((draft or "").strip())

    # ═════════════════════════════════════════════════════════════════════════
    # Orchestration entry point
    # ═════════════════════════════════════════════════════════════════════════

    def run(self, query: str, **kwargs: Any) -> AgentResponse:  # noqa: C901
        expected   = kwargs.get("expected_answer")
        t_start    = time.perf_counter()
        totals     = _Usage()

        total_latency_llm   = 0.0
        total_latency_tools = 0.0
        total_llm_calls     = 0
        total_tool_calls    = 0
        tools_called:        list[str]            = []
        tools_results:       list[dict[str, Any]] = []
        sub_agents_run:      list[str]            = []
        sub_agent_responses: list[dict[str, Any]] = []

        def _absorb_worker(r: WorkerResult) -> None:
            nonlocal totals, total_latency_llm, total_latency_tools
            nonlocal total_llm_calls, total_tool_calls
            totals.prompt       += r.prompt_tok
            totals.completion   += r.compl_tok
            totals.cost         += r.cost
            total_latency_llm   += max(0.0, r.latency - r.tool_latency)
            total_latency_tools += r.tool_latency
            total_llm_calls     += r.llm_calls
            total_tool_calls    += r.tool_calls
            tools_called.extend(r.tools_used)
            tools_results.append({
                "subtask_id":   r.subtask_id,
                "answer":       r.answer,
                "confidence":   r.confidence,
                "used_tool":    r.used_tool,
                "tools_used":   r.tools_used,
                "winning_lane": r.winning_lane,
                "error":        r.error,
                "evidence":     r.evidence,
                "raw":          r.raw,
            })

        try:
            subtasks, plan_usage = self._stage_planner(query)
            totals.add(plan_usage)
            total_llm_calls += 1
            sub_agents_run.append("planner")

            board    = Blackboard(goal=query)
            subtasks = self._validate_subtasks(subtasks)

            worker_results = self._stage_parallel_workers(subtasks, board)
            for res in worker_results:
                _absorb_worker(res)
                sub_agents_run.append(f"worker_{res.subtask_id}")
                sub_agent_responses.append({
                    "subtask_id":   res.subtask_id,
                    "answer":       res.answer,
                    "confidence":   res.confidence,
                    "winning_lane": res.winning_lane,
                    "used_tool":    res.used_tool,
                    "error":        res.error,
                })

            draft, consensus, critic_usage = self._stage_critic(query, worker_results, board)
            totals.add(critic_usage)
            if critic_usage.prompt > 0:
                total_llm_calls += 1
            sub_agents_run.append("critic")

            final_answer = self._stage_synthesizer(draft)
            sub_agents_run.append("synthesizer")
            sub_agent_responses.append({
                "role":      "critic+synthesizer",
                "answer":    final_answer,
                "consensus": consensus,
            })

            from evaluator.eval import is_correct

            return self._finalize_response(AgentResponse(
                query=query,
                answer=final_answer,
                model=self.model,
                agent="multiagent",
                dataset=self.dataset,
                latency_total=time.perf_counter() - t_start,
                latency_llm=total_latency_llm,
                latency_tools=total_latency_tools,
                prompt_tokens=totals.prompt,
                completion_tokens=totals.completion,
                total_tokens=totals.prompt + totals.completion,
                cost_usd=totals.cost,
                reasoning_steps=[s.tool_call_repr() for s in subtasks],
                num_llm_calls=total_llm_calls,
                num_steps=len(subtasks),
                tools_available=list(self.tools.keys()),
                tools_called=tools_called,
                tools_results=tools_results,
                num_tool_calls=total_tool_calls,
                max_steps=self.max_steps,
                steps_taken=total_llm_calls + total_tool_calls,
                is_stopped_early=False,
                orchestration_type="planner_blackboard_3lane_workers_critic_synth",
                sub_agents_run=sub_agents_run,
                selected_agent="synthesizer",
                sub_agent_responses=sub_agent_responses,
                consensus_score=consensus,
                expected_answer=expected,
                is_correct=is_correct(
                    final_answer, str(expected), dataset=self.dataset
                ) if expected else None,
                error=None,
                is_failed=False,
            ))

        except Exception as exc:
            return self._finalize_response(AgentResponse(
                query=query,
                answer="unknown",
                model=self.model,
                agent="multiagent",
                dataset=self.dataset,
                latency_total=time.perf_counter() - t_start,
                latency_llm=total_latency_llm,
                latency_tools=total_latency_tools,
                prompt_tokens=totals.prompt,
                completion_tokens=totals.completion,
                total_tokens=totals.prompt + totals.completion,
                cost_usd=totals.cost,
                num_llm_calls=total_llm_calls,
                tools_called=tools_called,
                tools_results=tools_results,
                num_tool_calls=total_tool_calls,
                orchestration_type="planner_blackboard_3lane_workers_critic_synth",
                sub_agents_run=sub_agents_run,
                sub_agent_responses=sub_agent_responses,
                expected_answer=expected,
                is_correct=None,
                error=str(exc),
                is_failed=True,
            ))


# ═════════════════════════════════════════════════════════════════════════════
# Module-level entry point
# ═════════════════════════════════════════════════════════════════════════════

def run(
    query:   str,
    model:   str,
    dataset: str,
    tools:   dict[str, Callable[..., Any]] | None = None,
    **kwargs: Any,
) -> AgentResponse:
    agent = MultiAgentAgent(model=model, dataset=dataset, tools=tools, **kwargs)
    return agent.run(query=query, **kwargs)