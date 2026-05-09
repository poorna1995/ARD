from __future__ import annotations
import re                                        # ✅ top of file

from agent.base import BaseAgent, AgentResponse
from prompts.prompts import USER_PROMPT, SYSTEM_PROMPT


class CotAgent(BaseAgent):
    def __init__(self, model: str, dataset: str, **kwargs):
        self.dataset = dataset
        super().__init__(
            model         = model,
            system_prompt = SYSTEM_PROMPT[dataset]["cot"],
            user_prompt   = USER_PROMPT[dataset]["cot"],
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
        raw_answer, latency, response = self._call_llm(query)
        steps, final = self._extract_reasoning(raw_answer)   # ✅ unpack both

        # use parsed steps from reasoning; fall back to API attribute if present
        msg = response.choices[0].message
        reasoning_steps = getattr(msg, "reasoning_steps", None) or steps  # ✅ use steps as fallback

        expected = kwargs.get("expected_answer")

        return AgentResponse(
            query             = query,
            answer            = final,                       
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
            reasoning_steps   = reasoning_steps,             # ✅ steps used
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
            is_correct        = final.strip() == str(expected).strip() if expected else None,  # ✅ compare final
            error             = None,
            is_failed         = False,
        )
        # ✅ removed unreachable return super().run()


def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
    agent = CotAgent(model=model, dataset=dataset)
    return agent.run(query, **kwargs)