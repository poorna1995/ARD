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

load_dotenv()


# ── Model Registry ─────────────────────────────────────────────────────────────

GROQ_MODELS   = {"llama-3.3-70b-versatile", "deepseek-r1"}
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
    query:   str
    answer:  str
    model:   str
    agent:   str
    dataset: str

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
    max_steps:        int  = 10
    steps_taken:      int  = 0
    is_stopped_early: bool = False

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
    error:     str | None = None
    is_failed: bool       = False


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
    ) -> NormalizedConfigResult:
        """
        Shared adapter for canonical agent config.

        This is a non-breaking utility hook: existing constructor and run paths
        remain unchanged until individual agents explicitly use this method.
        """
        return normalize_agent_config(
            model=model,
            dataset=dataset,
            kwargs=kwargs or {},
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
            "answer": response.answer,
            "model": response.model,
            "agent": response.agent,
            "dataset": response.dataset,
        }
        for field_name, field_value in required_str_fields.items():
            if not isinstance(field_value, str):
                raise TypeError(f"AgentResponse.{field_name} must be a string.")
            if field_name != "answer" and not field_value.strip():
                raise ValueError(f"AgentResponse.{field_name} must be non-empty.")

    def _finalize_response(self, response: AgentResponse) -> AgentResponse:
        self._validate_agent_response(response)
        return response

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

    def _call_llm(self, query: str) -> tuple[str, float, Any]:
        start = time.perf_counter()
        response = self._get_client().chat.completions.create(
            model       = self.model,
            messages    = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self.user_prompt.format(query=query)},
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




# from __future__ import annotations
    

# import os
# import random
# import time
# from dataclasses import dataclass, field   # ✅ field imported
# from typing import Any

# from dotenv import load_dotenv
# from groq import Groq
# import numpy as np
# from openai import OpenAI
# load_dotenv()



# # ── Model Registry ─────────────────────────────────────────────────────────────

# GROQ_MODELS   = {"llama-3.3-70b-versatile", "deepseek-r1"}
# OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-o1", "gpt-oss-20b", "gpt-oss-120b"}

# COST_PER_1M = {
#     "llama-3.3-70b-versatile": {"input": 0.59,  "output": 0.79},
#     "gpt-4o":                  {"input": 2.50,  "output": 10.00},
#     "gpt-4o-mini":             {"input": 0.15,  "output": 0.60},
#     "gpt-o1":                  {"input": 15.00, "output": 60.00},
#     "deepseek-r1":             {"input": 0.55,  "output": 2.19},
# }


# # ── AgentResponse ──────────────────────────────────────────────────────────────

# @dataclass
# class AgentResponse:

#     # ── Required ───────────────────────────────────────────────────────────
#     query:              str
#     answer:             str
#     model:              str
#     agent:              str
#     dataset:            str

#     # ── Latency ────────────────────────────────────────────────────────────
#     latency_total:      float = 0.0
#     latency_llm:        float = 0.0
#     latency_tools:      float = 0.0

#     # ── Cost ───────────────────────────────────────────────────────────────
#     prompt_tokens:      int   = 0
#     completion_tokens:  int   = 0
#     total_tokens:       int   = 0
#     cost_usd:           float = 0.0

#     # ── Reasoning ──────────────────────────────────────────────────────────
#     reasoning_steps:    list[str]  = field(default_factory=list)
#     num_llm_calls:      int        = 1
#     num_steps:          int        = 0

#     # ── Tools ──────────────────────────────────────────────────────────────
#     tools_available:    list[str]  = field(default_factory=list)
#     tools_called:       list[str]  = field(default_factory=list)
#     tools_results:      list[dict] = field(default_factory=list)
#     num_tool_calls:     int        = 0

#     # ── React Specific ─────────────────────────────────────────────────────
#     max_steps:          int        = 10
#     steps_taken:        int        = 0
#     is_stopped_early:   bool       = False

#     # ── Multiagent Specific ────────────────────────────────────────────────
#     orchestration_type:  str        = ""
#     sub_agents_run:      list[str]  = field(default_factory=list)
#     selected_agent:      str | None = None
#     sub_agent_responses: list[dict] = field(default_factory=list)
#     consensus_score:     float | None = None

#     # ── Evaluation ─────────────────────────────────────────────────────────
#     expected_answer:    str | None  = None
#     is_correct:         bool | None = None
#     confidence:         float | None = None
#     complexity:         float | None = None

#     # ── Errors ─────────────────────────────────────────────────────────────
#     error:              str | None  = None
#     is_failed:          bool        = False


# # ── BaseAgent ──────────────────────────────────────────────────────────────────

# class BaseAgent:
#     def __init__(
#         self,
#         model:         str,
#         system_prompt: str,
#         user_prompt:   str,
#         temperature:   float = 0.1,
#         max_tokens:    int   = 1024,
#         **kwargs,
#     ):
#         self.model         = model
#         self.system_prompt = system_prompt
#         self.user_prompt   = user_prompt
#         self.temperature   = temperature
#         self.max_tokens    = max_tokens
#         self.seed          = kwargs.get("seed")
#         self.groq_client   = None
#         self.openai_client = None
#         if self.seed is not None:
#             self._set_random_seed(int(self.seed))

#     @staticmethod
#     def _set_random_seed(seed: int) -> None:
#         random.seed(seed)
#         np.random.seed(seed)

#     def _get_client(self):                          # ✅ routes to correct client
#         return self._get_client_for_model(self.model)

#     def _get_client_for_model(self, model_name: str):
#         """Route to Groq vs OpenAI based on model id (same registry as _get_client)."""
#         if model_name in GROQ_MODELS:
#             if self.groq_client is None:
#                 self.groq_client = Groq(api_key=os.environ["GROQ_API_KEY"])
#             return self.groq_client
#         if self.openai_client is None:
#             self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
#         return self.openai_client

#     def _compute_cost_for_model(self, model_name: str, prompt_tokens: int, completion_tokens: int) -> float:
#         rates = COST_PER_1M.get(model_name, {"input": 0.0, "output": 0.0})
#         return (
#             prompt_tokens * rates["input"] + completion_tokens * rates["output"]
#         ) / 1_000_000
#     def _call_llm(self, query: str) -> tuple[str, float, Any]:
#         start = time.perf_counter()
#         response = self._get_client().chat.completions.create(
#             model       = self.model,
#             messages    = [
#                 {"role": "system", "content": self.system_prompt},
#                 {"role": "user",   "content": self.user_prompt.format(query=query)},
#             ],
#             temperature = self.temperature,
#             max_tokens  = self.max_tokens,
#             seed        = self.seed,
#         )
#         latency = time.perf_counter() - start
#         answer  = response.choices[0].message.content
#         return answer, latency, response

#     def _get_usage(self, response) -> dict:
#         return {
#             "prompt_tokens":     response.usage.prompt_tokens,
#             "completion_tokens": response.usage.completion_tokens,
#             "total_tokens":      response.usage.total_tokens,
#         }

#     def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
#         rates = COST_PER_1M.get(self.model, {"input": 0.0, "output": 0.0})
#         return (
#             prompt_tokens     * rates["input"] +
#             completion_tokens * rates["output"]
#         ) / 1_000_000

#     def run(self, query: str, **kwargs) -> AgentResponse:
#         raise NotImplementedError