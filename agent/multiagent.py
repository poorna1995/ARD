
from __future__ import annotations

from typing import Any

from agent.base import BaseAgent, AgentResponse
from agent.dataset_profile import apply_multiagent_profile, extract_hop_metadata
from agent.multiagent_judge import JudgeMixin
from agent.multiagent_workers import WorkerChainBlockedError
from agent.multiagent_planner import PlannerMixin
from agent.multiagent_workers import WorkerMixin
from prompts.prompts import MultiagentPrompts, build_multiagent_prompts
from agent.react import react_tool_names_for_dataset, tools_for_dataset


class MultiAgentAgent(BaseAgent, PlannerMixin, WorkerMixin, JudgeMixin):

    def __init__(self, model: str, dataset: str, **kwargs: Any) -> None:
        cfg_result = self._normalize_config(
            model=model,
            dataset=dataset,
            kwargs=kwargs,
            strategy="multiagent",
        )
        cfg = cfg_result.config
        n_hops, _ = extract_hop_metadata(kwargs)
        params = apply_multiagent_profile(
            cfg.dataset, cfg.agent_params, n_hops=n_hops,
        )

        self.dataset      = cfg.dataset
        self._n_hops      = n_hops
        self.max_workers  = int(params.get("max_workers", 4))
        self.max_subtasks = int(params.get("max_subtasks", 5))
        self.enable_tools_when_needed = bool(
            params.get("enable_tools_when_needed", True)
        )
        self._react_tools = tools_for_dataset(self.dataset)

        self.planner_model = str(params.get("planner_model", "gpt-4o-mini"))
        self.worker_model  = str(params.get("worker_model", "gpt-4o-mini"))
        self.judge_model   = str(params.get("judge_model", cfg.model))

        self.planner_max_tokens   = 512   
        self.worker_max_tokens    = 1024   
        self.judge_max_tokens     = 768   
        self.extractor_max_tokens = 300    

        # [OPT-TOOL1] Cap tool steps per worker — prevents runaway ReAct loops
        self.tool_max_steps = int(params.get("tool_max_steps", 5))

        self.worker_retry_threshold = float(params.get("worker_retry_threshold", 0.4))

        # [OPT-FWD1] Tighter forward caps — workers produce compact results
        self.worker_result_forward_chars = int(params.get("worker_result_forward_chars", 160))
        self.memory_fact_chars = int(
            params.get("memory_fact_chars", self.worker_result_forward_chars)
        )
        self.worker_evidence_forward_chars = int(params.get("worker_evidence_forward_chars", 80))

        planner_tools = react_tool_names_for_dataset(cfg.dataset)
        tool_enum = "|".join(["none"] + planner_tools)
        tools_csv = ", ".join(planner_tools)
        self._prompts: MultiagentPrompts = build_multiagent_prompts(
            cfg.dataset,
            tools_csv=tools_csv,
            tool_enum=tool_enum,
        )

        effective_max_tokens = cfg.max_tokens if "max_tokens" in kwargs else 2048
        super().__init__(
            model=cfg.model,
            system_prompt="",
            user_prompt="{query}",
            temperature=cfg.temperature,
            max_tokens=effective_max_tokens,
            seed=cfg.seed,
        )

    def run(self, query: str, **kwargs: Any) -> AgentResponse:  # type: ignore[override]
        expected = kwargs.get("expected_answer")
        raw_ctx = kwargs.get("context")
        self._episode_context = raw_ctx if isinstance(raw_ctx, dict) else {}

        sub_agent_responses: list[dict[str, Any]] = []
        sub_agents_run:      list[str]             = []
        tools_called:        list[str]             = []
        tools_results:       list[dict[str, Any]]  = []
        num_tool_calls = 0

        prompt_tokens = completion_tokens = total_tokens = 0
        total_cost = total_latency_llm = total_latency_tools = 0.0
        total_llm_calls = 0

        try:
            # ── 1. Planner ────────────────────────────────────────────────────
            subtasks, mem, lat, usage, cost = self._planner(query, **kwargs)
            mem._FACT_CHARS = self.memory_fact_chars

            sub_agents_run.append("planner")
            total_latency_llm += lat
            prompt_tokens     += usage["prompt_tokens"]
            completion_tokens += usage["completion_tokens"]
            total_tokens      += usage["total_tokens"]
            total_cost        += cost
            total_llm_calls   += 1

            print(f"[DIAG] Query: {query[:60]}")
            print(f"[DIAG] Subtasks: {len(subtasks)} | "
                  f"Tool-needed: {sum(1 for s in subtasks if s.get('needs_tool'))}")
            for s in subtasks:
                sq = s.get("search_query", "") or ""
                print(f"       [{s['id']}] needs_tool={s['needs_tool']} "
                      f"tool={s.get('tool_name','none')} "
                      f"depends_on={s.get('depends_on',[])} "
                      f"search_query={sq!r} "
                      f"url={s.get('candidate_url','') or '—'} "
                      f"| {s['goal'][:80]}")

            # ── 2. Workers ────────────────────────────────────────────────────
            worker_outputs: list[dict[str, Any]] = []

            for subtask in subtasks:

                use_tool = (
                    self.enable_tools_when_needed
                    and bool(subtask.get("needs_tool", False))
                )

                if use_tool:
                    out, tool_usage = self._tool_worker(query, subtask, mem)
                    worker_outputs.append(out)
                    sub_agents_run.append(f"worker_{subtask['id']}_tool")
                    sub_agent_responses.append({
                        "role": "worker", "mode": "tool_enabled",
                        "subtask_id": subtask["id"], "goal": subtask["goal"],
                        "output": out,
                    })
                    prompt_tokens       += int(tool_usage["prompt_tokens"])
                    completion_tokens   += int(tool_usage["completion_tokens"])
                    total_tokens        += int(tool_usage["total_tokens"])
                    total_cost          += float(tool_usage["cost_usd"])
                    total_latency_llm   += float(tool_usage["latency_llm"])
                    total_latency_tools += float(tool_usage["latency_tools"])
                    total_llm_calls     += int(tool_usage["num_llm_calls"])
                    tools_called.extend(tool_usage["tools_called"])
                    tools_results.extend(tool_usage["tools_results"])
                    num_tool_calls      += int(tool_usage["num_tool_calls"])
                else:
                    out, w_lat, w_usage, w_cost = self._worker(query, subtask, mem)
                    worker_outputs.append(out)
                    sub_agents_run.append(f"worker_{subtask['id']}")
                    sub_agent_responses.append({
                        "role": "worker", "mode": "llm_only",
                        "subtask_id": subtask["id"], "goal": subtask["goal"],
                        "output": out,
                    })
                    total_latency_llm += w_lat
                    prompt_tokens     += w_usage["prompt_tokens"]
                    completion_tokens += w_usage["completion_tokens"]
                    total_tokens      += w_usage["total_tokens"]
                    total_cost        += w_cost
                    total_llm_calls   += 1

                self._guard_worker_output(subtask, out, subtasks)
                self._update_memory(mem, subtask, out)

            # ── 3. Judge ──────────────────────────────────────────────────────
            judged, j_lat, j_usage, j_cost = self._judge(query, subtasks, mem, **kwargs)
            sub_agents_run.append("judge")
            sub_agent_responses.append({"role": "judge", "output": judged})
            total_latency_llm += j_lat
            prompt_tokens     += j_usage["prompt_tokens"]
            completion_tokens += j_usage["completion_tokens"]
            total_tokens      += j_usage["total_tokens"]
            total_cost        += j_cost
            total_llm_calls   += 1

            predicted_answer = judged["predicted_answer"]
            is_failed = bool(judged.get("is_failed"))

            return self._core_response(
                query=query,
                agent="multiagent",
                agent_id="multiagent_007",
                predicted_answer=predicted_answer,
                latency_total=total_latency_llm + total_latency_tools,
                latency_llm=total_latency_llm,
                expected_answer=expected,
                is_failed=is_failed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                confidence=judged.get("confidence"),
                complexity=judged.get("complexity"),
                latency_tools=total_latency_tools,
                reasoning_steps=[s["goal"] for s in subtasks],
                num_llm_calls=total_llm_calls,
                num_steps=len(subtasks),
                tools_available=react_tool_names_for_dataset(self.dataset),
                tools_called=tools_called,
                tools_results=tools_results,
                num_tool_calls=num_tool_calls,
                max_steps=self.max_subtasks + 2,
                steps_taken=total_llm_calls + num_tool_calls,
                is_stopped_early=False,
                orchestration_type="planner_workers_judge_working_memory_v6_optimised",
                sub_agents_run=sub_agents_run,
                selected_agent="judge",
                sub_agent_responses=sub_agent_responses,
                consensus_score=judged.get("consensus_score"),
            )

        except WorkerChainBlockedError as exc:
            print(f"[MULTIAGENT] {exc}")
            return self._core_response(
                query=query,
                agent="multiagent",
                agent_id="multiagent_007",
                predicted_answer="",
                latency_total=total_latency_llm + total_latency_tools,
                latency_llm=total_latency_llm,
                expected_answer=expected,
                is_failed=True,
                error=str(exc),
                finalize=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_tools=total_latency_tools,
                num_llm_calls=total_llm_calls,
                tools_available=react_tool_names_for_dataset(self.dataset),
                tools_called=tools_called,
                tools_results=tools_results,
                num_tool_calls=num_tool_calls,
                orchestration_type="planner_workers_judge_working_memory_v6_optimised",
                sub_agents_run=sub_agents_run,
                sub_agent_responses=sub_agent_responses,
            )

        except Exception as exc:
            return self._core_response(
                query=query,
                agent="multiagent",
                agent_id="multiagent_007",
                predicted_answer="",
                latency_total=total_latency_llm + total_latency_tools,
                latency_llm=total_latency_llm,
                expected_answer=expected,
                is_failed=True,
                error=str(exc),
                finalize=False,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_tools=total_latency_tools,
                num_llm_calls=total_llm_calls,
                tools_available=react_tool_names_for_dataset(self.dataset),
                tools_called=tools_called,
                tools_results=tools_results,
                num_tool_calls=num_tool_calls,
                orchestration_type="planner_workers_judge_working_memory_v6_optimised",
                sub_agents_run=sub_agents_run,
                sub_agent_responses=sub_agent_responses,
            )


# ── Module-level entry point ───────────────────────────────────────────────────

def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
    agent = MultiAgentAgent(model=model, dataset=dataset, **kwargs)
    return agent.run(query=query, **kwargs)

