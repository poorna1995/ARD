from agent.base import BaseAgent, AgentResponse
from evaluator.eval import is_correct, parse_agent_output
from prompts.prompts import USER_PROMPT, SYSTEM_PROMPT


class RawAgent(BaseAgent):
    """Single-shot baseline agent with canonical config/runtime contracts."""

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
            system_prompt=SYSTEM_PROMPT[self.dataset]["raw"],
            user_prompt=USER_PROMPT[self.dataset]["raw"],
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    def run(self, query: str, **kwargs) -> AgentResponse:
        """Runtime inputs are shared via **kwargs (e.g., expected_answer)."""
        raw_out, latency, response = self._call_llm(query)
        answer, confidence, complexity = parse_agent_output(raw_out)
        expected = kwargs.get("expected_answer")
        response_obj = AgentResponse(
            query             = query,
            answer            = answer,
            model             = self.model,
            agent             = "raw",
            dataset           = self.dataset,
            latency_total     = latency,
            latency_llm       = latency,
            expected_answer   = expected,
            cost_usd          = self._compute_cost(
                                    response.usage.prompt_tokens,
                                    response.usage.completion_tokens
                                ),
            prompt_tokens     = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens      = response.usage.total_tokens,
            is_correct        = (
                is_correct(answer, str(expected))
                if expected else None
            ),
            confidence        = confidence,
            complexity        = complexity,

        )
        return self._finalize_response(response_obj)


def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
    agent = RawAgent(model=model, dataset=dataset, **kwargs)
    return agent.run(query=query, **kwargs)

if __name__ == "__main__":
    import pandas as pd

    dataset = "gaia"
    model = "gpt-4o-mini"
    df = pd.read_parquet(f"datasets/golden/{dataset}.parquet", columns=["query", "answer"])
    agent = RawAgent(model=model, dataset=dataset)

    for _, row in df.iterrows():
        response = agent.run(
            query=str(row["query"]),
            expected_answer=row.get("answer"),
        )
        print(response)
