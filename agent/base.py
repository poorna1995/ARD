from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI
from agent.config import NormalizedConfigResult, normalize_agent_config
from evaluator import is_correct

load_dotenv()


# ── Model Registry ─────────────────────────────────────────────────────────────

GROQ_MODELS   = {"llama-3.3-70b-versatile"}
DEEPSEEK_MODELS = {"deepseek-r1"}
OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-o1"}

COST_PER_1M: dict[str, dict[str, float]] = {
    "llama-3.3-70b-versatile": {"input": 0.59,  "output": 0.79},
    "gpt-4o":                  {"input": 2.50,  "output": 10.00},
    "gpt-4o-mini":             {"input": 0.15,  "output": 0.60},
    "gpt-o1":                  {"input": 15.00, "output": 60.00},
    "deepseek-r1":             {"input": 0.55,  "output": 2.19},
}


# ── AgentResponse ──────────────────────────────────────────────────────────────

@dataclass
class AgentResponse:

    # ── Required ───────────────────────────────────────────────────────────
    query:            str
    predicted_answer: str
    model:            str
    agent:   str
    dataset: str
    agent_id: str = ""

    # ── Latency ────────────────────────────────────────────────────────────
    latency_total: float = 0.0
    latency_llm:   float = 0.0
    latency_tools: float = 0.0

    # ── Cost ───────────────────────────────────────────────────────────────
    prompt_tokens:     int   = 0
    completion_tokens: int   = 0
    total_tokens:      int   = 0
    cost_usd:          float = 0.0

    # ── Reasoning ──────────────────────────────────────────────────────────
    reasoning_steps: list[str] = field(default_factory=list)
    num_llm_calls:   int       = 1
    num_steps:       int       = 0

    # ── Tools ──────────────────────────────────────────────────────────────
    tools_available: list[str]  = field(default_factory=list)
    tools_called:    list[str]  = field(default_factory=list)
    tools_results:   list[dict] = field(default_factory=list)
    num_tool_calls:  int        = 0

    # ── ReAct specific ─────────────────────────────────────────────────────
    max_steps:          int  = 10
    steps_taken:        int  = 0
    is_stopped_early:   bool = False
    num_format_retries: int  = 0
    invalid_generations:  list[str] = field(default_factory=list)
    format_retry_reasons: list[str] = field(default_factory=list)

    # ── Multi-agent specific ───────────────────────────────────────────────
    orchestration_type:  str        = "none"
    sub_agents_run:      list[str]  = field(default_factory=list)
    selected_agent:      str | None = None
    sub_agent_responses: list[dict] = field(default_factory=list)
    consensus_score:     float | None = None

    # ── Evaluation ─────────────────────────────────────────────────────────
    expected_answer: str | None   = None
    is_correct:      bool | None  = None
    confidence:      float | None = None
    complexity:      float | None = None

    # ── Errors ─────────────────────────────────────────────────────────────
    error:        str | None = None
    failure_type: str | None = None
    is_failed:    bool       = False


# ── BaseAgent ──────────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(
        self,
        model:         str,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float = 0.1,
        max_tokens:    int   = 1024,
        **kwargs,
    ) -> None:
        self.model         = model
        self.system_prompt = system_prompt
        self.user_prompt   = user_prompt
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.seed          = kwargs.get("seed")
        self._groq_client   = None
        self._openai_client = None
        if self.seed is not None:
            self._set_random_seed(int(self.seed))

    # ── Utilities ──────────────────────────────────────────────────────────

    @staticmethod
    def _set_random_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)

    @staticmethod
    def _normalize_config(
        *,
        model: str,
        dataset: str,
        kwargs: dict[str, Any] | None = None,
        strategy: str | None = None,
    ) -> NormalizedConfigResult:
        """Shared adapter for canonical agent config (see ``normalize_agent_config``)."""
        return normalize_agent_config(
            model=model,
            dataset=dataset,
            kwargs=kwargs or {},
            strategy=strategy,
        )

    @staticmethod
    def _validate_agent_response(response: AgentResponse) -> None:
        """
        Validate core response fields shared by all agents.

        Agent-specific fields (tools, orchestration, etc.) remain optional and are
        validated by each agent's own logic.
        """
        required_str_fields = {
            "query": response.query,
            "predicted_answer": response.predicted_answer,
            "model": response.model,
            "agent": response.agent,
            "dataset": response.dataset,
        }
        for field_name, field_value in required_str_fields.items():
            if not isinstance(field_value, str):
                raise TypeError(f"AgentResponse.{field_name} must be a string.")
            if field_name != "predicted_answer" and not field_value.strip():
                raise ValueError(f"AgentResponse.{field_name} must be non-empty.")

    def _finalize_response(self, response: AgentResponse) -> AgentResponse:
        self._validate_agent_response(response)
        return response

    @staticmethod
    def _expected_answer(kwargs: dict[str, Any]) -> str | None:
        val = kwargs.get("expected_answer")
        return None if val is None else str(val)

    @staticmethod
    def _substitute_prompt(template: str, **values: str) -> str:
        """Replace {key} placeholders without interpreting JSON braces in values."""
        out = template
        for key, value in values.items():
            out = out.replace(f"{{{key}}}", value)
        return out

    def _format_user_prompt(self, query: str, **kwargs: Any) -> str:
        """Format USER_PROMPT with ``{query}`` (``kwargs`` reserved for callers; unused here)."""
        del kwargs
        return self._substitute_prompt(self.user_prompt, query=query)

    def _format_question(self, query: str, **kwargs: Any) -> str:
        """Question line for orchestration roles (planner, judge user turn)."""
        del kwargs
        return f"Question: {query}"

    def _eval_fields(
        self,
        predicted_answer: str,
        expected_answer: str | None,
        *,
        confidence: float | None = None,
        complexity: float | None = None,
    ) -> dict[str, Any]:
        return {
            "expected_answer": expected_answer,
            "is_correct": (
                is_correct(predicted_answer, expected_answer, dataset=self.dataset)
                if expected_answer is not None
                else None
            ),
            "confidence": confidence,
            "complexity": complexity,
        }

    def _token_cost_fields(
        self,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> dict[str, Any]:
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "cost_usd": self._compute_cost(prompt_tokens, completion_tokens),
        }

    @staticmethod
    def _response_shape_defaults() -> dict[str, Any]:
        """Uniform optional fields so every agent returns the same AgentResponse shape."""
        return {
            "latency_tools": 0.0,
            "reasoning_steps": [],
            "num_llm_calls": 1,
            "num_steps": 0,
            "tools_available": [],
            "tools_called": [],
            "tools_results": [],
            "num_tool_calls": 0,
            "max_steps": 1,
            "steps_taken": 0,
            "is_stopped_early": False,
            "num_format_retries": 0,
            "invalid_generations": [],
            "format_retry_reasons": [],
            "orchestration_type": "none",
            "sub_agents_run": [],
            "selected_agent": None,
            "sub_agent_responses": [],
            "consensus_score": None,
            "failure_type": None,
        }

    def _core_response(
        self,
        *,
        query: str,
        agent: str,
        agent_id: str,
        predicted_answer: str,
        latency_total: float,
        latency_llm: float,
        expected_answer: str | None,
        is_failed: bool,
        error: str | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        confidence: float | None = None,
        complexity: float | None = None,
        finalize: bool = True,
        **extra: Any,
    ) -> AgentResponse:
        """Shared success/failure AgentResponse shape for all baseline agents."""
        shape = self._response_shape_defaults()
        shape.update(extra)
        common = dict(
            query=query,
            predicted_answer=predicted_answer,
            model=self.model,
            agent=agent,
            agent_id=agent_id,
            dataset=self.dataset,
            latency_total=latency_total,
            latency_llm=latency_llm,
            error=error,
            is_failed=is_failed,
            **shape,
        )
        if is_failed:
            return AgentResponse(
                **common,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                cost_usd=self._compute_cost(prompt_tokens, completion_tokens),
                expected_answer=expected_answer,
                is_correct=None,
                confidence=None,
                complexity=None,
            )
        resp = AgentResponse(
            **common,
            **self._token_cost_fields(prompt_tokens, completion_tokens),
            **self._eval_fields(
                predicted_answer,
                expected_answer,
                confidence=confidence,
                complexity=complexity,
            ),
        )
        return self._finalize_response(resp) if finalize else resp

    def _get_client(self) -> Groq | OpenAI:
        """Return the client for self.model (convenience wrapper)."""
        return self._get_client_for_model(self.model)

    def _get_client_for_model(self, model_name: str) -> Groq | OpenAI:
        if model_name in GROQ_MODELS:
            if self._groq_client is None:
                self._groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
            return self._groq_client
 
        # Default: OpenAI-compatible
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        return self._openai_client

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Cost in USD for self.model."""
        return self._compute_cost_for_model(
            self.model, prompt_tokens, completion_tokens
        )

    def _compute_cost_for_model(
        self, model_name: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Cost in USD for an arbitrary model (used by multi-agent orchestrators)."""
        rates = COST_PER_1M.get(model_name, {"input": 0.0, "output": 0.0})
        return (
            prompt_tokens     * rates["input"] +
            completion_tokens * rates["output"]
        ) / 1_000_000

    def _get_usage(self, response: Any) -> dict:
        return {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

    # ── Default single-turn LLM call (override in subclasses as needed) ───

    def _call_llm(self, query: str, **kwargs: Any) -> tuple[str, float, Any]:
        start = time.perf_counter()
        response = self._get_client().chat.completions.create(
            model       = self.model,
            messages    = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self._format_user_prompt(query, **kwargs)},
            ],
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
            seed        = self.seed,
        )
        return (
            response.choices[0].message.content,
            time.perf_counter() - start,
            response,
        )

    # ── Entry point (must be overridden) ───────────────────────────────────

    def run(self, query: str, **kwargs) -> AgentResponse:
        raise NotImplementedError