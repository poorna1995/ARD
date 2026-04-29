from agent.base import BaseAgent, AgentResponse
from prompts.prompts import USER_PROMPT, SYSTEM_PROMPT


class RawAgent(BaseAgent):
    def __init__(self, model: str,dataset: str, **kwargs):
        super().__init__(
            model         = model,
            system_prompt = SYSTEM_PROMPT[dataset]["raw"],   
            user_prompt   = USER_PROMPT[dataset]["raw"],  
        )

    def run(self, query: str, dataset: str = "", **kwargs) -> AgentResponse:
        answer, latency, response = self._call_llm(query)
        return AgentResponse(
            query             = query,
            answer            = answer,
            model             = self.model,
            agent             = "raw",
            dataset           = dataset,                    # ✅ was missing
            latency_total     = latency,                    # ✅ correct field name
            latency_llm       = latency,  
            expected_answer = kwargs.get("expected_answer"),                  # ✅ correct field name
            cost_usd          = self._compute_cost(
                                    response.usage.prompt_tokens,
                                    response.usage.completion_tokens
                                ),
            prompt_tokens     = response.usage.prompt_tokens,
            completion_tokens = response.usage.completion_tokens,
            total_tokens      = response.usage.total_tokens,
            is_correct      = answer == kwargs.get("expected_answer"),

        )


def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
    agent = RawAgent(model=model, dataset=dataset)
    return agent.run(query=query, dataset=dataset, **kwargs)