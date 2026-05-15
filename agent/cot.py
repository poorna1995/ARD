from __future__ import annotations
import re

from agent.base import BaseAgent, AgentResponse
from evaluator.eval import is_correct, parse_agent_output
from prompts.prompts import SYSTEM_PROMPT, USER_PROMPT


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
            system_prompt=SYSTEM_PROMPT[self.dataset]["cot"],
            user_prompt=USER_PROMPT[self.dataset]["cot"],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    def _extract_reasoning(self, raw_answer: str) -> tuple[list[str], str]:
        steps = []
        lines = raw_answer.strip().split("\n")
        for line in lines:
            if line.strip().startswith("{") and "answer" in line:
                break
            if line.strip():
                steps.append(line.strip())
        match = re.search(r'\{"answer":\s*"([^"]+)"\}', raw_answer)
        final = match.group(0) if match else raw_answer
        return steps, final

    def run(self, query: str, **kwargs) -> AgentResponse:
        """Runtime inputs are shared via **kwargs (e.g., expected_answer)."""
        raw_answer, latency, response = self._call_llm(query)
        steps, final = self._extract_reasoning(raw_answer)
        answer, confidence, complexity = parse_agent_output(final)

        # Use parsed steps from reasoning; fall back to API attribute if present.
        msg = response.choices[0].message
        reasoning_steps = getattr(msg, "reasoning_steps", None) or steps

        expected = kwargs.get("expected_answer")

        response_obj = AgentResponse(
            query             = query,
            answer            = answer,
            model             = self.model,
            agent             = "cot",
            dataset           = self.dataset,
            latency_total     = latency,
            latency_llm       = latency,
            cost_usd          = self._compute_cost(
                                    response.usage.prompt_tokens,
                                    response.usage.completion_tokens,
                                ),
            prompt_tokens     = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens      = response.usage.total_tokens,
            reasoning_steps   = reasoning_steps,
            num_llm_calls     = 1,
            num_steps         = len(steps),
            tools_available   = [],
            tools_called      = [],
            tools_results     = [],
            num_tool_calls    = 0,
            max_steps         = 1,
            steps_taken       = 1,
            is_stopped_early  = False,
            expected_answer   = expected,
            is_correct        = (
                is_correct(answer, str(expected))
                if expected else None
            ),
            confidence        = confidence,
            complexity        = complexity,
            error             = None,
            is_failed         = False,
        )
        return self._finalize_response(response_obj)


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
