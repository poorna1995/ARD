from __future__ import annotations

import time

from agent.base import BaseAgent, AgentResponse
from evaluator.parse import parse_llm_output
from prompts.prompts import SYSTEM_PROMPT, USER_PROMPT

AGENT_ID = "raw_001"


class RawAgent(BaseAgent):
    """Single-shot baseline: one LLM call, parsed answer."""

    def __init__(self, model: str, dataset: str, **kwargs):
        cfg = self._normalize_config(
            model=model, dataset=dataset, kwargs=kwargs, strategy="raw",
        ).config
        self.dataset = cfg.dataset
        super().__init__(
            model=cfg.model,
            system_prompt=SYSTEM_PROMPT[self.dataset]["raw"],
            user_prompt=USER_PROMPT[self.dataset]["raw"],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    def run(self, query: str, **kwargs) -> AgentResponse:
        t0 = time.perf_counter()
        expected = self._expected_answer(kwargs)

        try:
            raw_text, latency_llm, api = self._call_llm(query, **kwargs)
            predicted_answer, confidence, complexity = parse_llm_output(raw_text)
            usage = api.usage
            return self._core_response(
                query=query,
                agent="raw",
                agent_id=AGENT_ID,
                predicted_answer=predicted_answer,
                latency_total=time.perf_counter() - t0,
                latency_llm=latency_llm,
                expected_answer=expected,
                is_failed=False,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                confidence=confidence,
                complexity=complexity,
                num_steps=1,
                steps_taken=1,
            )
        except Exception as exc:
            return self._core_response(
                query=query,
                agent="raw",
                agent_id=AGENT_ID,
                predicted_answer="",
                latency_total=time.perf_counter() - t0,
                latency_llm=0.0,
                expected_answer=expected,
                is_failed=True,
                error=str(exc),
                finalize=False,
            )


def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
    return RawAgent(model=model, dataset=dataset, **kwargs).run(
        query=query, **kwargs,
    )


if __name__ == "__main__":
    import pandas as pd

    dataset = "gaia"
    model = "gpt-4o-mini"
    path = f"datasets/golden/{dataset}.parquet"
    df = pd.read_parquet(path, columns=["query", "answer"])

    for _, row in df.iterrows():
        resp = run(
            query=str(row["query"]),
            model=model,
            dataset=dataset,
            expected_answer=row.get("answer"),
        )
        print(resp.agent_id, resp.predicted_answer, resp.latency_total, resp.is_failed)
