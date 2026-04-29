from __future__ import annotations

import os
import time
from dataclasses import dataclass, field   # ✅ field imported
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from openai import OpenAI

load_dotenv()


# ── Model Registry ─────────────────────────────────────────────────────────────

GROQ_MODELS   = {"llama-3.3-70b-versatile", "deepseek-r1"}
OPENAI_MODELS = {"gpt-4o", "gpt-4o-mini", "gpt-o1"}

COST_PER_1M = {
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
    query:              str
    answer:             str
    model:              str
    agent:              str
    dataset:            str

    # ── Latency ────────────────────────────────────────────────────────────
    latency_total:      float = 0.0
    latency_llm:        float = 0.0
    latency_tools:      float = 0.0

    # ── Cost ───────────────────────────────────────────────────────────────
    prompt_tokens:      int   = 0
    completion_tokens:  int   = 0
    total_tokens:       int   = 0
    cost_usd:           float = 0.0

    # ── Reasoning ──────────────────────────────────────────────────────────
    reasoning_steps:    list[str]  = field(default_factory=list)
    num_llm_calls:      int        = 1
    num_steps:          int        = 0

    # ── Tools ──────────────────────────────────────────────────────────────
    tools_available:    list[str]  = field(default_factory=list)
    tools_called:       list[str]  = field(default_factory=list)
    tools_results:      list[dict] = field(default_factory=list)
    num_tool_calls:     int        = 0

    # ── React Specific ─────────────────────────────────────────────────────
    max_steps:          int        = 0
    steps_taken:        int        = 0
    is_stopped_early:   bool       = False

    # ── Multiagent Specific ────────────────────────────────────────────────
    orchestration_type:  str        = ""
    sub_agents_run:      list[str]  = field(default_factory=list)
    selected_agent:      str | None = None
    sub_agent_responses: list[dict] = field(default_factory=list)
    consensus_score:     float | None = None

    # ── Evaluation ─────────────────────────────────────────────────────────
    expected_answer:    str | None  = None
    is_correct:         bool | None = None

    # ── Errors ─────────────────────────────────────────────────────────────
    error:              str | None  = None
    is_failed:          bool        = False


# ── BaseAgent ──────────────────────────────────────────────────────────────────

class BaseAgent:
    def __init__(
        self,
        model:         str,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float = 0.0,
        max_tokens:    int   = 1024,
        **kwargs,
    ):
        self.model         = model
        self.system_prompt = system_prompt
        self.user_prompt   = user_prompt
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.groq_client   = Groq(api_key=os.environ["GROQ_API_KEY"])
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def _get_client(self):                          # ✅ routes to correct client
        if self.model in GROQ_MODELS:
            return self.groq_client
        return self.openai_client

    def _call_llm(self, query: str) -> tuple[str, float, Any]:   # ✅ 3 return values
        start    = time.perf_counter()
        response = self._get_client().chat.completions.create(    # ✅ correct client
            model       = self.model,
            messages    = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user",   "content": self.user_prompt.format(query=query)},
            ],
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
        )
        latency = time.perf_counter() - start
        answer  = response.choices[0].message.content
        return answer, latency, response             # ✅ 3 values

    def _get_usage(self, response) -> dict:
        return {
            "prompt_tokens":     response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens":      response.usage.total_tokens,
        }

    def _compute_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        rates = COST_PER_1M.get(self.model, {"input": 0.0, "output": 0.0})
        return (
            prompt_tokens     * rates["input"] +
            completion_tokens * rates["output"]
        ) / 1_000_000

    def run(self, query: str, **kwargs) -> AgentResponse:
        raise NotImplementedError