from __future__ import annotations

from typing import Any

from agent.dataset_profile import validate_planner_subtasks
from agent.multiagent_memory import WorkingMemory


class PlannerMixin:
    @staticmethod
    def _normalize_subtask_depends_on(
        depends_on: Any,
        *,
        valid_ids: frozenset[str],
        self_id: str,
    ) -> list[str]:
        if not isinstance(depends_on, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for dep in depends_on:
            dep_id = str(dep).strip()
            if not dep_id or dep_id == self_id or dep_id in seen:
                continue
            if dep_id not in valid_ids:
                continue
            seen.add(dep_id)
            out.append(dep_id)
        return out

    @staticmethod
    def _topo_sort_subtasks(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run parents before children; preserve planner order among independent tasks."""
        if not subtasks:
            return subtasks

        by_id: dict[str, dict[str, Any]] = {}
        for i, st in enumerate(subtasks, start=1):
            sid = str(st.get("id", f"s{i}")).strip() or f"s{i}"
            if sid in by_id:
                sid = f"{sid}_{i}"
            row = dict(st)
            row["id"] = sid
            by_id[sid] = row

        valid = frozenset(by_id.keys())
        for sid, st in by_id.items():
            st["depends_on"] = PlannerMixin._normalize_subtask_depends_on(
                st.get("depends_on"),
                valid_ids=valid,
                self_id=sid,
            )

        orig_index = {sid: i for i, sid in enumerate(by_id)}
        indegree = {sid: 0 for sid in by_id}
        children: dict[str, list[str]] = {sid: [] for sid in by_id}
        for sid, st in by_id.items():
            for dep in st["depends_on"]:
                indegree[sid] += 1
                children[dep].append(sid)

        ready = sorted(
            [sid for sid in by_id if indegree[sid] == 0],
            key=lambda s: orig_index[s],
        )
        ordered: list[str] = []
        while ready:
            nid = ready.pop(0)
            ordered.append(nid)
            for child in sorted(children[nid], key=lambda s: orig_index[s]):
                indegree[child] -= 1
                if indegree[child] == 0:
                    ready.append(child)
                    ready.sort(key=lambda s: orig_index[s])

        if len(ordered) < len(by_id):
            for sid in sorted(by_id, key=lambda s: orig_index[s]):
                if sid not in ordered:
                    ordered.append(sid)

        return [by_id[sid] for sid in ordered]

    def _planner_user_prompt(self, query: str, **kwargs: Any) -> str:
        user = self._format_question(query, **kwargs)
        n_hops = getattr(self, "_n_hops", None) or kwargs.get("n_hops")
        if n_hops and self.dataset in ("musique", "hotpot"):
            try:
                hops = int(n_hops)
            except (TypeError, ValueError):
                hops = None
            if hops and hops >= 2:
                cap = min(int(self.max_subtasks), hops)
                user += (
                    f"\nHop count: {hops}. "
                    f"Create up to {cap} retrieve subtask(s), one per hop.\n"
                    "Decompose inside-out: innermost nested clause first, "
                    "final/root question last. "
                    "depends_on must chain each hop to the immediately prior subtask."
                )
        return user

    def _planner(
        self, query: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], WorkingMemory, float, dict[str, int], float]:
        raw, latency, response = self._call_with_messages(
            self._prompts.planner_system,
            self._planner_user_prompt(query, **kwargs),
            temperature=0.1,
            max_tokens=self.planner_max_tokens,
            model=self.planner_model,
        )
        usage = self._get_usage(response)
        cost = self._compute_cost_for_model(
            self.planner_model, usage["prompt_tokens"], usage["completion_tokens"]
        )

        parsed = self._extract_json_object(raw) or {}
        raw_subs = parsed.get("subtasks", [])
        if not isinstance(raw_subs, list):
            raw_subs = []

        normalized: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for i, st in enumerate(raw_subs[: self.max_subtasks], start=1):
            if isinstance(st, dict):
                goal = str(st.get("goal", "")).strip()
                focus = str(st.get("focus", "reasoning")).strip() or "reasoning"
                needs_tool = self._as_bool(st.get("needs_tool"), default=False)
                tool_name = str(st.get("tool_name", "none")).strip() or "none"
                candidate_url = str(st.get("candidate_url", "")).strip()
                search_query = str(st.get("search_query", "")).strip()
                depends_on = st.get("depends_on", [])
                if not isinstance(depends_on, list):
                    depends_on = []
                sub_id = str(st.get("id", f"s{i}")).strip() or f"s{i}"
            else:
                goal = str(st).strip()
                focus = "reasoning"
                needs_tool = False
                tool_name = "none"
                candidate_url = ""
                search_query = ""
                depends_on = []
                sub_id = f"s{i}"

            if sub_id in used_ids:
                sub_id = f"{sub_id}_{i}"
            used_ids.add(sub_id)

            if goal:
                normalized.append(
                    {
                        "id": sub_id,
                        "goal": goal,
                        "focus": focus,
                        "needs_tool": needs_tool,
                        "tool_name": tool_name,
                        "candidate_url": candidate_url,
                        "search_query": search_query,
                        "depends_on": [str(d).strip() for d in depends_on if str(d).strip()],
                    }
                )

        if not normalized:
            if self.dataset == "math":
                normalized = [
                    {
                        "id": "s1",
                        "goal": query,
                        "focus": "calculation",
                        "needs_tool": True,
                        "tool_name": "math_tool",
                        "candidate_url": "",
                        "search_query": "",
                        "depends_on": [],
                    }
                ]
            elif self.dataset in ("hotpot", "musique"):
                normalized = [
                    {
                        "id": "s1",
                        "goal": query,
                        "focus": "factual",
                        "needs_tool": True,
                        "tool_name": "retrieve",
                        "candidate_url": "",
                        "search_query": "",
                        "depends_on": [],
                    }
                ]
            elif self.dataset == "mmlu":
                normalized = [
                    {
                        "id": "s1",
                        "goal": query,
                        "focus": "reasoning",
                        "needs_tool": False,
                        "tool_name": "none",
                        "candidate_url": "",
                        "search_query": "",
                        "depends_on": [],
                    }
                ]
            else:
                normalized = [
                    {
                        "id": "s1",
                        "goal": query,
                        "focus": "reasoning",
                        "needs_tool": True,
                        "tool_name": "web_search",
                        "candidate_url": "",
                        "search_query": "",
                        "depends_on": [],
                    }
                ]

        normalized = validate_planner_subtasks(
            normalized,
            dataset=self.dataset,
            n_hops=getattr(self, "_n_hops", None),
            hop_name=kwargs.get("hop_name"),
        )
        normalized = self._topo_sort_subtasks(normalized)

        mem = WorkingMemory(
            goal=query,
            open_questions=[s["goal"] for s in normalized],
            final_constraint=str(parsed.get("final_constraint", "")).strip(),
        )

        return normalized, mem, latency, usage, cost

