from __future__ import annotations

from agent.base import BaseAgent, AgentResponse
from evaluator.parse import extract_reasoning_steps, parse_llm_output
from input.prompts.prompts import build_cot_system, user_prompt


class CotAgent(BaseAgent):
    """Chain-of-thought style agent with canonical config/runtime contracts."""

    def __init__(self, model: str, dataset: str, **kwargs):
        cfg_result = self._normalize_config(
            model=model,
            dataset=dataset,
            kwargs=kwargs,
        )
        cfg = cfg_result.config
        self.dataset = cfg.dataset
        super().__init__(
            model=cfg.model,
            system_prompt=build_cot_system(self.dataset),
            user_prompt=user_prompt(self.dataset),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    def run(self, query: str, **kwargs) -> AgentResponse:
        """Runtime inputs are shared via **kwargs (e.g., expected_answer)."""
        raw_answer, latency, response = self._call_llm(query, **kwargs)
        steps = extract_reasoning_steps(raw_answer)
        predicted_answer, confidence, complexity = parse_llm_output(raw_answer)

        # Use parsed steps from reasoning; fall back to API attribute if present.
        msg = response.choices[0].message
        reasoning_steps = getattr(msg, "reasoning_steps", None) or steps
        expected = self._expected_answer(kwargs)

        return self._core_response(
            query=query,
            agent="cot",
            agent_id="cot_002",
            predicted_answer=predicted_answer,
            latency_total=latency,
            latency_llm=latency,
            expected_answer=expected,
            is_failed=False,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            confidence=confidence,
            complexity=complexity,
            reasoning_steps=reasoning_steps,
            num_llm_calls=1,
            num_steps=len(steps),
            max_steps=1,
            steps_taken=1,
            is_stopped_early=False,
        )


def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
    agent = CotAgent(model=model, dataset=dataset, **kwargs)
    return agent.run(query, **kwargs)



if __name__ == "__main__":
    import pandas as pd

    dataset = "gaia"
    model = "gpt-4o-mini"
    df = pd.read_parquet(f"datasets/golden/{dataset}.parquet", columns=["query", "answer"])
    agent = CotAgent(model=model, dataset=dataset)

    for _, row in df.iterrows():
        response = agent.run(
            query=str(row["query"]),
            expected_answer=row.get("answer"),
        )
        print(response)
