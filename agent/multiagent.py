from __future__ import annotations

import json
import re
import time
from typing import Any

from agent.base import BaseAgent, AgentResponse
from prompts.prompts import USER_PROMPT, SYSTEM_PROMPT


class MultiAgentAgent(BaseAgent):
    """Planner -> Workers -> Judge/Synthesizer orchestration with conditional tools."""

    def __init__(self, model: str, dataset: str, **kwargs):
        self.dataset = dataset
        self.max_workers = int(kwargs.get("max_workers", 3))
        self.max_subtasks = int(kwargs.get("max_subtasks", 4))
        self.enable_tools_when_needed = bool(kwargs.get("enable_tools_when_needed", True))
        super().__init__(
            model=model,
            system_prompt=SYSTEM_PROMPT[dataset]["multiagent"],
            user_prompt=USER_PROMPT[dataset]["multiagent"],
            temperature=float(kwargs.get("temperature", 0.1)),
            max_tokens=int(kwargs.get("max_tokens", 1024)),
        )

    @staticmethod
    def _extract_json_object(text: str) -> dict[str, Any] | None:
        text = (text or "").strip()
        if not text:
            return None

        # Try strict parse first.
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        # Fallback: extract first JSON object substring.
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None

    def _call_with_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> tuple[str, float, Any]:
        start = time.perf_counter()
        response = self._get_client().chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        latency = time.perf_counter() - start
        answer = response.choices[0].message.content or ""
        return answer, latency, response

    @staticmethod
    def _as_bool(value: Any, default: bool = False) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            if v in {"true", "yes", "1"}:
                return True
            if v in {"false", "no", "0"}:
                return False
        return default

    def _planner(self, query: str) -> tuple[list[dict[str, Any]], float, dict[str, int], float]:
        system_prompt = (
            "You are the Planner agent in a multi-agent QA system. "
            "Break the query into 2-4 clear subtasks. "
            "Return strict JSON only: "
            '{"subtasks":[{"id":"s1","goal":"...","focus":"reasoning|factual|calculation",'
            '"needs_tool":false,"tool_name":"none|read_file|web_search|wikipedia"}]}. '
            "Set needs_tool=true only if external tools are genuinely required."
        )
        user_prompt = f"Question: {query}"
        raw, latency, response = self._call_with_messages(system_prompt, user_prompt, temperature=0.0)
        usage = self._get_usage(response)
        cost = self._compute_cost(usage["prompt_tokens"], usage["completion_tokens"])

        parsed = self._extract_json_object(raw) or {}
        subtasks = parsed.get("subtasks", [])
        if not isinstance(subtasks, list):
            subtasks = []

        normalized: list[dict[str, Any]] = []
        for i, st in enumerate(subtasks[: self.max_subtasks], start=1):
            if isinstance(st, dict):
                goal = str(st.get("goal", "")).strip()
                focus = str(st.get("focus", "reasoning")).strip() or "reasoning"
                needs_tool = self._as_bool(st.get("needs_tool"), default=False)
                tool_name = str(st.get("tool_name", "none")).strip() or "none"
            else:
                goal = str(st).strip()
                focus = "reasoning"
                needs_tool = False
                tool_name = "none"
            if goal:
                normalized.append(
                    {
                        "id": f"s{i}",
                        "goal": goal,
                        "focus": focus,
                        "needs_tool": needs_tool,
                        "tool_name": tool_name,
                    }
                )

        if not normalized:
            normalized = [
                {"id": "s1", "goal": query, "focus": "reasoning", "needs_tool": False, "tool_name": "none"}
            ]

        return normalized, latency, usage, cost

    def _worker(self, query: str, subtask: dict[str, Any]) -> tuple[dict[str, Any], float, dict[str, int], float]:
        system_prompt = (
            "You are a specialist Worker in a multi-agent QA system. "
            "Solve only your assigned subtask. "
            "Return strict JSON only: "
            '{"subtask_id":"...","result":"...","confidence":0.0,"evidence":"short justification"}'
        )
        user_prompt = (
            f"Original question: {query}\n"
            f"Subtask id: {subtask['id']}\n"
            f"Subtask goal: {subtask['goal']}\n"
            f"Subtask focus: {subtask['focus']}"
        )
        raw, latency, response = self._call_with_messages(system_prompt, user_prompt, temperature=0.2)
        usage = self._get_usage(response)
        cost = self._compute_cost(usage["prompt_tokens"], usage["completion_tokens"])

        parsed = self._extract_json_object(raw) or {}
        result = {
            "subtask_id": subtask["id"],
            "goal": subtask["goal"],
            "focus": subtask["focus"],
            "needs_tool": bool(subtask.get("needs_tool", False)),
            "tool_name": str(subtask.get("tool_name", "none")),
            "result": str(parsed.get("result", raw)).strip(),
            "confidence": float(parsed.get("confidence", 0.5)) if str(parsed.get("confidence", "")).strip() else 0.5,
            "evidence": str(parsed.get("evidence", "")).strip(),
        }
        return result, latency, usage, cost

    def _tool_worker(self, query: str, subtask: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        """Delegate tool-needed subtasks to ReAct agent to reuse existing tools stack."""
        try:
            from agent import react
        except Exception as exc:
            # If tool stack import fails, degrade gracefully to an LLM worker-style output.
            return (
                {
                    "subtask_id": subtask["id"],
                    "goal": subtask["goal"],
                    "focus": subtask["focus"],
                    "needs_tool": True,
                    "tool_name": str(subtask.get("tool_name", "none")),
                    "result": f"Tool worker unavailable: {exc}",
                    "confidence": 0.2,
                    "evidence": "Fell back due to tool worker import failure.",
                },
                {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "cost_usd": 0.0,
                    "latency_llm": 0.0,
                    "latency_tools": 0.0,
                    "tools_called": [],
                    "tools_results": [],
                    "num_tool_calls": 0,
                    "num_llm_calls": 0,
                },
            )

        worker_query = (
            f"Original question: {query}\n"
            f"Subtask goal: {subtask['goal']}\n"
            f"Preferred tool: {subtask.get('tool_name', 'none')}\n"
            "Use tools only if needed and return final JSON answer."
        )
        response = react.run(
            query=worker_query,
            model=self.model,
            dataset=self.dataset,
        )
        result = {
            "subtask_id": subtask["id"],
            "goal": subtask["goal"],
            "focus": subtask["focus"],
            "needs_tool": True,
            "tool_name": str(subtask.get("tool_name", "none")),
            "result": response.answer,
            "confidence": 0.7,
            "evidence": "Tool-enabled worker delegated to ReAct agent.",
            "react_tools_called": response.tools_called,
        }
        usage = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens": response.total_tokens,
            "cost_usd": response.cost_usd,
            "latency_llm": response.latency_llm,
            "latency_tools": response.latency_tools,
            "tools_called": response.tools_called,
            "tools_results": response.tools_results,
            "num_tool_calls": response.num_tool_calls,
            "num_llm_calls": response.num_llm_calls,
        }
        return result, usage

    def _judge(
        self,
        query: str,
        subtasks: list[dict[str, Any]],
        worker_outputs: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], float, dict[str, int], float]:
        system_prompt = (
            "You are the Judge/Synthesizer in a multi-agent QA system. "
            "Combine worker outputs, resolve conflicts, and provide a final answer. "
            "Return strict JSON only: "
            '{"answer":"<value>","consensus_score":0.0,"rationale":"short summary"}'
        )
        user_prompt = (
            f"Question: {query}\n"
            f"Plan: {json.dumps(subtasks, ensure_ascii=True)}\n"
            f"Worker outputs: {json.dumps(worker_outputs, ensure_ascii=True)}"
        )
        raw, latency, response = self._call_with_messages(system_prompt, user_prompt, temperature=0.0)
        usage = self._get_usage(response)
        cost = self._compute_cost(usage["prompt_tokens"], usage["completion_tokens"])

        parsed = self._extract_json_object(raw) or {}
        answer = str(parsed.get("answer", "")).strip()
        if not answer:
            answer = raw.strip() or "No answer produced."
        return {
            "answer": answer,
            "consensus_score": float(parsed.get("consensus_score", 0.0))
            if str(parsed.get("consensus_score", "")).strip()
            else 0.0,
            "rationale": str(parsed.get("rationale", "")).strip(),
            "raw_judge_output": raw,
        }, latency, usage, cost

    def run(self, query: str, **kwargs) -> AgentResponse:
        expected = kwargs.get("expected_answer")
        sub_agent_responses: list[dict[str, Any]] = []
        sub_agents_run: list[str] = []
        tools_called: list[str] = []
        tools_results: list[dict[str, Any]] = []
        num_tool_calls = 0

        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        total_cost = 0.0
        total_latency_llm = 0.0
        total_latency_tools = 0.0
        total_llm_calls = 0

        try:
            subtasks, lat, usage, cost = self._planner(query)
            sub_agents_run.append("planner")
            total_latency_llm += lat
            prompt_tokens += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            total_tokens += usage["total_tokens"]
            total_cost += cost
            total_llm_calls += 1

            worker_outputs: list[dict[str, Any]] = []
            for subtask in subtasks[: self.max_workers]:
                use_tool_worker = self.enable_tools_when_needed and bool(subtask.get("needs_tool", False))
                if use_tool_worker:
                    out, tool_usage = self._tool_worker(query, subtask)
                    worker_outputs.append(out)
                    sub_agents_run.append(f"worker_{subtask['id']}_tool")
                    sub_agent_responses.append(
                        {
                            "role": "worker",
                            "mode": "tool_enabled",
                            "subtask_id": subtask["id"],
                            "goal": subtask["goal"],
                            "output": out,
                        }
                    )
                    prompt_tokens += int(tool_usage["prompt_tokens"])
                    completion_tokens += int(tool_usage["completion_tokens"])
                    total_tokens += int(tool_usage["total_tokens"])
                    total_cost += float(tool_usage["cost_usd"])
                    total_latency_llm += float(tool_usage["latency_llm"])
                    total_latency_tools += float(tool_usage["latency_tools"])
                    total_llm_calls += int(tool_usage["num_llm_calls"])
                    tools_called.extend(tool_usage["tools_called"])
                    tools_results.extend(tool_usage["tools_results"])
                    num_tool_calls += int(tool_usage["num_tool_calls"])
                else:
                    out, w_lat, w_usage, w_cost = self._worker(query, subtask)
                    worker_outputs.append(out)
                    sub_agents_run.append(f"worker_{subtask['id']}")
                    sub_agent_responses.append(
                        {
                            "role": "worker",
                            "mode": "llm_only",
                            "subtask_id": subtask["id"],
                            "goal": subtask["goal"],
                            "output": out,
                        }
                    )
                    total_latency_llm += w_lat
                    prompt_tokens += w_usage["prompt_tokens"]
                    completion_tokens += w_usage["completion_tokens"]
                    total_tokens += w_usage["total_tokens"]
                    total_cost += w_cost
                    total_llm_calls += 1

            judged, j_lat, j_usage, j_cost = self._judge(query, subtasks, worker_outputs)
            sub_agents_run.append("judge")
            sub_agent_responses.append({"role": "judge", "output": judged})
            total_latency_llm += j_lat
            prompt_tokens += j_usage["prompt_tokens"]
            completion_tokens += j_usage["completion_tokens"]
            total_tokens += j_usage["total_tokens"]
            total_cost += j_cost
            total_llm_calls += 1

            answer = judged["answer"]
            return AgentResponse(
                query=query,
                answer=answer,
                model=self.model,
                agent="multiagent",
                dataset=self.dataset,
                latency_total=total_latency_llm + total_latency_tools,
                latency_llm=total_latency_llm,
                latency_tools=total_latency_tools,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=total_cost,
                reasoning_steps=[s["goal"] for s in subtasks],
                num_llm_calls=total_llm_calls,
                num_steps=len(subtasks),
                tools_available=["read_file", "web_search", "wikipedia"],
                tools_called=tools_called,
                tools_results=tools_results,
                num_tool_calls=num_tool_calls,
                max_steps=self.max_subtasks + 2,
                steps_taken=total_llm_calls + num_tool_calls,
                is_stopped_early=False,
                orchestration_type="planner_workers_judge_conditional_tools_v1",
                sub_agents_run=sub_agents_run,
                selected_agent="judge",
                sub_agent_responses=sub_agent_responses,
                consensus_score=judged.get("consensus_score"),
                expected_answer=expected,
                is_correct=answer.strip() == str(expected).strip() if expected else None,
                error=None,
                is_failed=False,
            )
        except Exception as exc:  # defensive for long-running batches
            return AgentResponse(
                query=query,
                answer='{"answer": "ERROR"}',
                model=self.model,
                agent="multiagent",
                dataset=self.dataset,
                latency_total=total_latency_llm + total_latency_tools,
                latency_llm=total_latency_llm,
                latency_tools=total_latency_tools,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=total_cost,
                num_llm_calls=total_llm_calls,
                tools_available=["read_file", "web_search", "wikipedia"],
                tools_called=tools_called,
                tools_results=tools_results,
                num_tool_calls=num_tool_calls,
                orchestration_type="planner_workers_judge_conditional_tools_v1",
                sub_agents_run=sub_agents_run,
                sub_agent_responses=sub_agent_responses,
                expected_answer=expected,
                is_correct=None,
                error=str(exc),
                is_failed=True,
            )


def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
    agent = MultiAgentAgent(model=model, dataset=dataset, **kwargs)
    return agent.run(query=query, **kwargs)