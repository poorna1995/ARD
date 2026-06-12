# agent/debate.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from agent.base import BaseAgent, AgentResponse
from evaluator import canonicalise_answer
from evaluator.parse import parse_llm_output
from input.prompts.prompts import (
    DEBATE_AGENT_TEMPERATURE,
    DEBATE_SYNTHESIS_PROMPT,
    DEBATE_SYNTHESIS_SYSTEM,
    build_debate_system,
    user_prompt,
)

AGENT_ID = "debate_006"


@dataclass
class _LlmCallResult:
    text: str
    crashed: bool
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    wall_clock_time: float
    error: str | None = None


class DebateAgent(BaseAgent):
    """
    Strategy 6: Multi-Agent Debate.

    Round 1: Two independent debater calls (prompts.prompts debate entries) at
             dataset-specific temperature. If both answers agree after
             canonicalisation, return immediately.
    Round 2: If they disagree, synthesis (DEBATE_SYNTHESIS_*) at temperature=0.0.

    Total LLM calls: 2 (agreement) or 3 (disagreement).
    """

    SYNTHESIS_TEMPERATURE = 0.0
    AGENT_MAX_TOKENS = 1200
    SYNTHESIS_MAX_TOKENS = 800

    def __init__(self, model: str, dataset: str, **kwargs: Any) -> None:
        cfg = self._normalize_config(
            model=model, dataset=dataset, kwargs=kwargs, strategy="debate",
        ).config
        self.dataset = cfg.dataset
        self.agent_temperature = DEBATE_AGENT_TEMPERATURE.get(self.dataset, 0.7)
        self.synthesis_system = DEBATE_SYNTHESIS_SYSTEM
        self.synthesis_user_template = DEBATE_SYNTHESIS_PROMPT[self.dataset]
        super().__init__(
            model=cfg.model,
            system_prompt=build_debate_system(self.dataset),
            user_prompt=user_prompt(self.dataset),
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            seed=cfg.seed,
        )

    def _call(
        self,
        *,
        messages: list[dict[str, str]],
        system: str,
        temperature: float,
        max_tokens: int,
    ) -> _LlmCallResult:
        start = time.perf_counter()
        try:
            response = self._get_client().chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    *messages,
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                seed=self.seed,
            )
            wall = time.perf_counter() - start
            usage = response.usage
            text = response.choices[0].message.content or ""
            return _LlmCallResult(
                text=text,
                crashed=False,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_tokens=usage.total_tokens,
                wall_clock_time=wall,
            )
        except Exception as exc:
            return _LlmCallResult(
                text="",
                crashed=True,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                wall_clock_time=time.perf_counter() - start,
                error=str(exc),
            )

    def run(self, query: str, **kwargs: Any) -> AgentResponse:
        expected = self._expected_answer(kwargs)
        t0 = time.perf_counter()

        prompt_tokens = 0
        completion_tokens = 0
        latency_llm = 0.0
        num_llm_calls = 0
        trace: list[str] = []
        sub_agent_responses: list[dict[str, Any]] = []
        sub_agents_run: list[str] = []

        user_content = self._format_user_prompt(query, **kwargs)

        result_a = self._call(
            messages=[{"role": "user", "content": user_content}],
            system=self.system_prompt,
            temperature=self.agent_temperature,
            max_tokens=self.AGENT_MAX_TOKENS,
        )
        result_b = self._call(
            messages=[{"role": "user", "content": user_content}],
            system=self.system_prompt,
            temperature=self.agent_temperature,
            max_tokens=self.AGENT_MAX_TOKENS,
        )
        num_llm_calls += 2
        prompt_tokens += result_a.prompt_tokens + result_b.prompt_tokens
        completion_tokens += result_a.completion_tokens + result_b.completion_tokens
        latency_llm += result_a.wall_clock_time + result_b.wall_clock_time

        debate_extra = dict(
            orchestration_type="multi_agent_debate",
            sub_agents_run=sub_agents_run,
            sub_agent_responses=sub_agent_responses,
            max_steps=3,
        )

        if result_a.crashed and result_b.crashed:
            sub_agents_run.extend(["agent_a", "agent_b"])
            return self._core_response(
                query=query,
                agent="debate",
                agent_id=AGENT_ID,
                predicted_answer="",
                latency_total=time.perf_counter() - t0,
                latency_llm=latency_llm,
                expected_answer=expected,
                is_failed=True,
                error="both agents crashed",
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                num_llm_calls=num_llm_calls,
                finalize=False,
                **debate_extra,
            )

        answer_a, conf_a, comp_a = (
            parse_llm_output(result_a.text) if not result_a.crashed else ("", None, None)
        )
        answer_b, conf_b, comp_b = (
            parse_llm_output(result_b.text) if not result_b.crashed else ("", None, None)
        )

        trace.append(f"[Agent A]\n{result_a.text}")
        trace.append(f"[Agent B]\n{result_b.text}")
        sub_agents_run.extend(["agent_a", "agent_b"])
        sub_agent_responses.append(
            {"role": "agent_a", "raw": result_a.text, "answer": answer_a}
        )
        sub_agent_responses.append(
            {"role": "agent_b", "raw": result_b.text, "answer": answer_b}
        )

        norm_a = canonicalise_answer(answer_a, dataset=self.dataset)
        norm_b = canonicalise_answer(answer_b, dataset=self.dataset)

        if norm_a and norm_b and norm_a == norm_b:
            trace.append("[Agents agreed — no synthesis needed]")
            return self._core_response(
                query=query,
                agent="debate",
                agent_id=AGENT_ID,
                predicted_answer=answer_a,
                latency_total=time.perf_counter() - t0,
                latency_llm=latency_llm,
                expected_answer=expected,
                is_failed=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                confidence=conf_a,
                complexity=comp_a,
                reasoning_steps=trace,
                num_llm_calls=num_llm_calls,
                num_steps=num_llm_calls,
                steps_taken=num_llm_calls,
                selected_agent="agreement",
                consensus_score=1.0,
                **debate_extra,
            )

        synthesis_prompt = self._substitute_prompt(
            self.synthesis_user_template,
            question=query,
            answer_a=result_a.text[:2000],
            answer_b=result_b.text[:2000],
        )
        synth_result = self._call(
            messages=[{"role": "user", "content": synthesis_prompt}],
            system=self.synthesis_system,
            temperature=self.SYNTHESIS_TEMPERATURE,
            max_tokens=self.SYNTHESIS_MAX_TOKENS,
        )
        num_llm_calls += 1
        prompt_tokens += synth_result.prompt_tokens
        completion_tokens += synth_result.completion_tokens
        latency_llm += synth_result.wall_clock_time

        trace.append(f"[Synthesis]\n{synth_result.text}")
        sub_agents_run.append("synthesis")
        sub_agent_responses.append({"role": "synthesis", "raw": synth_result.text})

        if synth_result.crashed:
            final_answer = answer_a or answer_b
            selected_agent = "A" if answer_a else "B"
            confidence = conf_a if answer_a else conf_b
            complexity = comp_a if answer_a else comp_b
            error = synth_result.error
        else:
            final_answer, confidence, complexity = parse_llm_output(synth_result.text)
            selected_agent = "synthesis"
            error = None

        return self._core_response(
            query=query,
            agent="debate",
            agent_id=AGENT_ID,
            predicted_answer=final_answer,
            latency_total=time.perf_counter() - t0,
            latency_llm=latency_llm,
            expected_answer=expected,
            is_failed=False,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            confidence=confidence,
            complexity=complexity,
            error=error,
            reasoning_steps=trace,
            num_llm_calls=num_llm_calls,
            num_steps=num_llm_calls,
            steps_taken=num_llm_calls,
            selected_agent=selected_agent,
            consensus_score=0.6,
            **debate_extra,
        )


def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
    return DebateAgent(model=model, dataset=dataset, **kwargs).run(
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
