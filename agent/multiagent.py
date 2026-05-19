
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.base import BaseAgent, AgentResponse
from agent.dataset_profile import (
    apply_multiagent_profile,
    extract_hop_metadata,
    is_open_wiki_qa,
    observation_text_from_tools_results,
    validate_planner_subtasks,
)
from evaluator.eval import is_correct
from evaluator.parse import (
    extract_last_json_dict,
    optional_float,
    parse_llm_output_detailed,
)
from prompts.prompts import MultiagentPrompts, build_multiagent_prompts

# Nested ReAct tool workers use ``tools_for_dataset`` / ``react_tool_names_for_dataset``
# from ``agent.react`` (single source of truth with standalone ReAct).
from agent.react import react_tool_names_for_dataset, tools_for_dataset

# ── WorkingMemory ──────────────────────────────────────────────────────────────

@dataclass
class WorkingMemory:
    goal:             str            = ""
    known_facts: dict[str, dict] = field(default_factory=dict)
    # Goal text for each subtask_id at the time its fact was recorded (for dedup).
    subtask_goals:    dict[str, str] = field(default_factory=dict)
    open_questions:   list[str]      = field(default_factory=list)
    tool_summaries:   dict[str, str] = field(default_factory=dict)
    final_constraint: str            = ""

    # [OPT-WM1] Tighter char budgets — 120/80 instead of 200/120.
    # Facts forwarded to the judge are already distilled; 120 chars is enough
    # for a zip code, species name, date, or short phrase.
    _FACT_CHARS:    int = 120
    _SUMMARY_CHARS: int = 80

    def record_fact(self, subtask_id: str, result: str, confidence: float, tool_backed: bool = False) -> None:
        existing = self.known_facts.get(subtask_id)
        if existing is None or confidence >= existing.get("confidence", 0):
            self.known_facts[subtask_id] = {
                "text": result[:self._FACT_CHARS],
                "confidence": confidence,
                "tool_backed": tool_backed,
            }

    def record_tool_summary(self, subtask_id: str, summary: str) -> None:
        self.tool_summaries[subtask_id] = summary[: self._SUMMARY_CHARS]

    def close_question(self, goal_text: str) -> None:
        self.open_questions = [q for q in self.open_questions if q != goal_text]

    def slice_for(self, dep_ids: list[str]) -> str:
        """Compact JSON slice — only the deps this worker actually needs."""
        if dep_ids:
            facts    = {k: v["text"] for k, v in self.known_facts.items() if k in dep_ids}
            tool_sum = {k: v for k, v in self.tool_summaries.items() if k in dep_ids}
        else:
            facts = tool_sum = {}

        payload: dict[str, Any] = {"goal": self.goal}
        if facts:
            payload["known_facts"] = facts
        if tool_sum:
            payload["tool_summaries"] = tool_sum
        if self.open_questions:
            payload["open_questions"] = self.open_questions
        if self.final_constraint:
            payload["final_constraint"] = self.final_constraint
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))

    def full(self) -> str:
        return self.slice_for(list(self.known_facts.keys()))


# ── MultiAgentAgent ────────────────────────────────────────────────────────────

class MultiAgentAgent(BaseAgent):

    _STRUCTURED_SOURCE_PATTERNS: list[str] = [
        r"usgs\.gov", r"gbif\.org",  r"epa\.gov",       r"census\.gov",
        r"data\.gov", r"ncbi\.nlm\.nih\.gov", r"wikipedia\.org",
    ]

    def __init__(self, model: str, dataset: str, **kwargs: Any) -> None:
        cfg_result = self._normalize_config(
            model=model,
            dataset=dataset,
            kwargs=kwargs,
            strategy="multiagent",
        )
        cfg = cfg_result.config
        n_hops, hop_name = extract_hop_metadata(kwargs)
        params = apply_multiagent_profile(
            cfg.dataset, cfg.agent_params, n_hops=n_hops,
        )

        self.dataset      = cfg.dataset
        self._n_hops      = n_hops
        self._hop_name    = hop_name
        self.max_workers  = int(params.get("max_workers", 4))
        self.max_subtasks = int(params.get("max_subtasks", 5))
        self.enable_tools_when_needed = bool(
            params.get("enable_tools_when_needed", True)
        )
        self._react_tools = tools_for_dataset(self.dataset)

        self.planner_model = str(params.get("planner_model", "gpt-4o-mini"))
        self.worker_model  = str(params.get("worker_model", "gpt-4o-mini"))
        self.judge_model   = str(params.get("judge_model", cfg.model))

        self.planner_max_tokens   = 512    # [OPT-TOK1] 768 → 512: planner output is short JSON
        self.worker_max_tokens    = 1024   # [OPT-TOK2] 2048 → 1024: workers return compact JSON
        self.judge_max_tokens     = 768    # [OPT-TOK3] 2048 → 512: judge reads compact memory
        self.extractor_max_tokens = 300    # [OPT-TOK4] 512 → 300: extractor returns ≤250 words

        # [OPT-TOOL1] Cap tool steps per worker — prevents runaway ReAct loops
        self.tool_max_steps = int(params.get("tool_max_steps", 5))

        self.worker_retry_threshold = float(params.get("worker_retry_threshold", 0.4))

        # [OPT-FWD1] Tighter forward caps — workers produce compact results
        self.worker_result_forward_chars = int(params.get("worker_result_forward_chars", 200))
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

    # ── Generic utilities ──────────────────────────────────────────────────────

    @staticmethod
    def _coerce_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        if isinstance(value, (dict, list)):
            try:
                return json.dumps(value, ensure_ascii=False)
            except Exception:
                return str(value)
        return str(value)

    @staticmethod
    def _extract_json_object(text: Any) -> dict[str, Any] | None:
        return extract_last_json_dict(MultiAgentAgent._coerce_text(text))

    def _build_strategy_hint(self, subtask: dict[str, Any]) -> str:
        """Short ReAct nudge from planner tool_name and optional search_query."""
        tool_name     = str(subtask.get("tool_name", "none")).strip().lower()
        candidate_url = str(subtask.get("candidate_url", "") or "").strip()
        search_query  = str(subtask.get("search_query", "") or "").strip()

        if tool_name == "math_tool":
            return (
                "\nStrategy: math_tool[<python_expression>] for every numeric step; "
                "pure Python only — no units or words."
            )
        if tool_name in ("wikipedia_search", "wikipedia"):
            if search_query:
                return (
                    f"\nStrategy: first Action must be "
                    f"wikipedia_search[{search_query}] — entity/event name only."
                )
            return "\nStrategy: wikipedia_search[entity_name] — entity name only."
        if tool_name == "read_file":
            return "\nStrategy: read_file[exact_filename] when a file is named in the question."
        if tool_name == "web_fetch" and candidate_url:
            return f"\nStrategy: web_fetch[{candidate_url}] first; fall back to web_search."
        if tool_name == "web_fetch":
            return "\nStrategy: build an authoritative URL and web_fetch it."
        if tool_name == "web_search":
            if search_query:
                return f"\nStrategy: first Action should be web_search[{search_query}]."
            return "\nStrategy: web_search."
        if tool_name and tool_name != "none":
            return f"\nStrategy: call {tool_name} for this subtask."
        return ""

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

    @staticmethod
    def _strip_answer_preamble(answer: Any) -> str:
        answer = MultiAgentAgent._coerce_text(answer)
        for pat in [
            r"(?i)^the\s+final\s+answer\s+is\s*[:\-]?\s*",
            r"(?i)^final\s+answer\s*[:\-]\s*",
            r"(?i)^the\s+answer\s+is\s*[:\-]?\s*",
            r"(?i)^answer\s*[:\-]\s*",
            r"(?i)^result\s*[:\-]\s*",
        ]:
            answer = re.sub(pat, "", answer).strip()
        return answer

    @staticmethod
    def _empty_tool_usage() -> dict[str, Any]:
        return {
            "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            "cost_usd": 0.0, "latency_llm": 0.0, "latency_tools": 0.0,
            "tools_called": [], "tools_results": [],
            "num_tool_calls": 0, "num_llm_calls": 0,
        }

    # ── LLM call ───────────────────────────────────────────────────────────────

    def _call_with_messages(
        self,
        system_prompt: str,
        user_prompt:   str,
        temperature:   float | None = None,
        max_tokens:    int   | None = None,
        model:         str   | None = None,
    ) -> tuple[str, float, Any]:
        effective_model = model or self.model
        client          = self._get_client_for_model(effective_model)
        start           = time.perf_counter()
        response = client.chat.completions.create(
            model       = effective_model,
            messages    = [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature = self.temperature if temperature is None else temperature,
            max_tokens  = self.max_tokens  if max_tokens  is None else max_tokens,
        )
        latency = time.perf_counter() - start
        content = response.choices[0].message.content
        return self._coerce_text(content), latency, response

    # ── Chunk + keyword-retrieve ───────────────────────────────────────────────

    @staticmethod
    def _retrieve_top_chunks(
        text: str, query: str,
        chunk_words: int = 350, overlap: int = 100, top_k: int = 3,  # [OPT-CHK1] top_k 4→3
    ) -> str:
        words = text.split()
        if len(words) <= chunk_words * top_k:
            return text
        step   = chunk_words - overlap
        chunks = [(i, " ".join(words[i: i + chunk_words]))
                  for i in range(0, len(words), step)]
        terms  = set(re.sub(r"[^\w\s]", "", query).lower().split())
        ranked = sorted(
            chunks,
            key=lambda x: sum(1 for t in terms if t in x[1].lower()),
            reverse=True,
        )
        top = sorted(ranked[:top_k], key=lambda x: x[0])
        return "\n---\n".join(c for _, c in top)

    # ── Semantic extraction ────────────────────────────────────────────────────

    def _semantic_extract(self, raw_text: str, goal: str) -> str:
        """Distil only facts that answer *goal*.

        [OPT-EXT1] Raised the skip threshold from 120 → 200 words.
        Short tool snippets (50–150 words) were being fed to the extractor
        even when they were already compact; raising the bar avoids an extra
        LLM call that costs ~400 tokens and adds latency.
        """
        focused = self._retrieve_top_chunks(raw_text, goal)
        if len(focused.split()) < 200:          # [OPT-EXT1] was 120
            return focused

        try:
            out, _, _ = self._call_with_messages(
                self._prompts.extractor_system, f"Q: {goal}\n\nDoc:\n{focused}",
                temperature=0.0, max_tokens=self.extractor_max_tokens,
                model=self.worker_model,
            )
            out = out.strip()
            return focused if out == "NO_RELEVANT_CONTENT" else out
        except Exception as exc:
            print(f"[EXTRACT] semantic extraction failed ({exc}); using chunks.")
            return focused

    # ── Structured extraction for known-schema sources ─────────────────────────

    @staticmethod
    def _is_structured_source(url: str) -> bool:
        url_lower = (url or "").lower()
        return any(re.search(p, url_lower)
                   for p in MultiAgentAgent._STRUCTURED_SOURCE_PATTERNS)

    def _structured_extract(self, raw_text: str, fields: list[str], goal: str) -> str:
        fields_str = ", ".join(f'"{f}"' for f in fields)
        system = self._prompts.structured_extractor_template.format(
            fields_str=fields_str,
        )
        focused = self._retrieve_top_chunks(raw_text, goal)
        try:
            raw, _, _ = self._call_with_messages(
                system, f"Doc:\n{focused}",
                temperature=0.0, max_tokens=self.extractor_max_tokens,
                model=self.worker_model,
            )
            parsed = self._extract_json_object(raw)
            return json.dumps(parsed, ensure_ascii=False) if parsed else focused
        except Exception as exc:
            print(f"[EXTRACT] structured extraction failed ({exc}); using chunks.")
            return focused

    @staticmethod
    def _infer_fields_from_goal(goal: str) -> list[str]:
        g = goal.lower()
        fields: list[str] = []
        if any(w in g for w in ["zip", "postal", "location", "where", "found"]):
            fields += ["zip_code", "location", "state", "county", "locality"]
        if any(w in g for w in ["species", "taxon", "scientific", "common name"]):
            fields += ["species", "scientific_name", "common_name", "taxon"]
        if any(w in g for w in ["date", "year", "when", "first", "last"]):
            fields += ["date", "year", "collection_date", "observation_date"]
        if any(w in g for w in ["count", "number", "how many", "total"]):
            fields += ["count", "total", "number_of_records"]
        if any(w in g for w in ["nonnative", "invasive", "introduced"]):
            fields += ["status", "establishment_means", "occurrence_status"]
        if any(w in g for w in ["population", "habitat", "range"]):
            fields += ["habitat", "range", "population_size"]
        if not fields:
            fields = ["value", "name", "description", "date", "location"]
        return list(dict.fromkeys(fields))

    def _react_worker_payload(
        self,
        response: Any,
        subtask: dict[str, Any],
    ) -> tuple[str, str, float]:
        """
        (compressed result, raw storage text, confidence).
        Uses last tool observation when ReAct exhausts steps without finish.
        """
        source_url = str(subtask.get("candidate_url", "") or "").strip()
        tools_results = getattr(response, "tools_results", None)
        pred = self._coerce_text(getattr(response, "predicted_answer", "")).strip()

        if is_open_wiki_qa(self.dataset):
            obs_text = observation_text_from_tools_results(tools_results)
            if obs_text:
                compressed = self._process_tool_result(
                    raw=obs_text, subtask=subtask, source_url=source_url,
                )
                raw = obs_text[: self.worker_result_forward_chars * 2]
                return compressed or obs_text[:300], raw, 0.6
            if pred:
                return (
                    self._process_tool_result(raw=pred, subtask=subtask, source_url=source_url),
                    pred,
                    0.55,
                )
            return "", "", 0.2

        if pred:
            return (
                self._process_tool_result(raw=pred, subtask=subtask, source_url=source_url),
                pred,
                0.75,
            )

        obs_text = observation_text_from_tools_results(tools_results)
        if obs_text:
            compressed = self._process_tool_result(
                raw=obs_text, subtask=subtask, source_url=source_url,
            )
            raw = obs_text[: self.worker_result_forward_chars * 2]
            return compressed or obs_text[:300], raw, 0.55

        return "", "", 0.2

    # ── Smart tool-result processor ────────────────────────────────────────────

    def _process_tool_result(
        self, raw: Any, subtask: dict[str, Any], source_url: str = "",
    ) -> str:
        raw_text = self._coerce_text(raw)
        if not raw_text or len(raw_text.split()) < 80:
            return raw_text
        goal = subtask.get("goal", "")
        if self._is_structured_source(source_url):
            return self._structured_extract(raw_text, self._infer_fields_from_goal(goal), goal)
        return self._semantic_extract(raw_text, goal)

    # ── WorkingMemory update ───────────────────────────────────────────────────

    def _update_memory(
        self,
        mem:     WorkingMemory,
        subtask: dict[str, Any],
        result:  dict[str, Any],
    ) -> None:
        sid        = result.get("subtask_id", subtask["id"])
        confidence = float(result.get("confidence", 0.5))
        raw_result = str(result.get("result", "")).strip()

        # [OPT-EXT1] Only re-extract when genuinely long (>60 words)
        signal = (
            self._semantic_extract(raw_result, subtask["goal"])
            if len(raw_result.split()) > 60
            else raw_result
        )

        mem.record_fact(sid, signal, confidence)
        mem.subtask_goals[sid] = str(subtask.get("goal", "")).strip()

        if result.get("needs_tool") and raw_result:
            one_liner = raw_result.split(".")[0][: WorkingMemory._SUMMARY_CHARS]
            mem.record_tool_summary(sid, one_liner)

        mem.close_question(subtask["goal"])

    # ── Prior-context builder ──────────────────────────────────────────────────

    @staticmethod
    def _build_prior_context(subtask: dict[str, Any], mem: WorkingMemory) -> str:
        dep_ids: list[str] = subtask.get("depends_on", [])
        if not dep_ids:
            return ""
        relevant = {k: v for k, v in mem.known_facts.items() if k in dep_ids}
        if not relevant:
            return ""
        lines = [f"  [{sid}] {fact}" for sid, fact in relevant.items()]
        return "\n\nContext from earlier steps (deps only):\n" + "\n".join(lines)

    # ── [OPT-DEDUP] Shared result cache: skip redundant subtasks ──────────────

    @staticmethod
    def _goals_overlap(a: str, b: str, threshold: float = 0.55) -> bool:
        """True when two goal strings share >threshold of their word stems."""
        wa = set(re.sub(r"[^\w]", " ", a.lower()).split())
        wb = set(re.sub(r"[^\w]", " ", b.lower()).split())
        if not wa or not wb:
            return False
        return len(wa & wb) / min(len(wa), len(wb)) > threshold

    def _find_reusable_result(
        self, subtask: dict[str, Any], mem: WorkingMemory
    ) -> str | None:
        """
        [OPT-DEDUP] If a previous worker already answered a near-identical goal,
        return that fact and skip the new worker entirely.

        This catches the common pattern where the planner produces subtasks like
        'Find the arXiv article' and 'Get the figure labels from the arXiv article'
        when the first already fetched everything needed — as seen in the trace
        where worker s1 and s2 both searched the same paper.
        """
        for sid, fact in mem.known_facts.items():
            if sid == subtask["id"]:
                continue
            fact_text = self._coerce_text(fact.get("text", "") if isinstance(fact, dict) else fact).strip()
            if not fact_text:
                continue
            # Only reuse if the existing result actually contains a substantive answer
            # (not an empty or error string) and goals clearly overlap.
            prior_goal = mem.subtask_goals.get(sid, "")
            if not prior_goal:
                continue
            if (
                len(fact_text.split()) >= 3
                and self._goals_overlap(subtask["goal"], prior_goal)
            ):
                # Be conservative — only reuse when the subtask goal keywords are
                # a strict subset of an already-answered goal.
                subtask_words = set(re.sub(r"[^\w]", " ", subtask["goal"].lower()).split())
                fact_words    = set(re.sub(r"[^\w]", " ", fact_text.lower()).split())
                if len(subtask_words & fact_words) / max(len(subtask_words), 1) > 0.6:
                    return fact_text
        return None

    # ── [OPT-SKIP] Early-exit: should we even run a tool worker? ──────────────

    @staticmethod
    def _llm_can_answer(subtask: dict[str, Any], mem: WorkingMemory) -> bool:
        """
        [OPT-SKIP] Return True when the subtask can be answered from existing
        working-memory facts alone — sparing an expensive tool worker.

        Heuristic: the subtask has at least one dependency whose fact is non-empty
        AND the dependency fact already contains key terms from the subtask goal.
        This avoids sending a ReAct worker to fetch data that a prior worker
        already retrieved (the s2 scenario in the trace).
        """
        dep_ids: list[str] = subtask.get("depends_on", [])
        if not dep_ids:
            return False
        goal_words = set(re.sub(r"[^\w]", " ", subtask["goal"].lower()).split())
        for dep_id in dep_ids:
            fact = mem.known_facts.get(dep_id, {})
            fact_text = MultiAgentAgent._coerce_text(
                fact.get("text", "") if isinstance(fact, dict) else fact
            ).strip()
            if not fact_text:
                continue
            fact_words = set(re.sub(r"[^\w]", " ", fact_text.lower()).split())
            # If >50 % of the goal's words already appear in a dependency fact,
            # there is a good chance the LLM worker can answer from context.
            overlap = len(goal_words & fact_words) / max(len(goal_words), 1)
            if overlap > 0.50:
                return True
        return False

    # ── Subtask ordering / planner fields ────────────────────────────────────────

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
            st["depends_on"] = MultiAgentAgent._normalize_subtask_depends_on(
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

    # ── Planner ────────────────────────────────────────────────────────────────

    def _planner(
        self, query: str, **kwargs: Any,
    ) -> tuple[list[dict[str, Any]], WorkingMemory, float, dict[str, int], float]:
        """
        [OPT-PLN1] Tighter system prompt (saves ~80 tokens).
        [OPT-PLN2] Enforce max 3 subtasks unless query clearly requires more —
        fewer subtasks = fewer worker calls = fewer tokens.
        """
        raw, latency, response = self._call_with_messages(
            self._prompts.planner_system,
            self._format_question(query, **kwargs),
            temperature=0.1, max_tokens=self.planner_max_tokens,
            model=self.planner_model,
        )
        usage = self._get_usage(response)
        cost  = self._compute_cost_for_model(
            self.planner_model, usage["prompt_tokens"], usage["completion_tokens"],
        )

        parsed   = self._extract_json_object(raw) or {}
        raw_subs = parsed.get("subtasks", [])
        if not isinstance(raw_subs, list):
            raw_subs = []

        normalized: list[dict[str, Any]] = []
        used_ids: set[str] = set()
        for i, st in enumerate(raw_subs[: self.max_subtasks], start=1):
            if isinstance(st, dict):
                goal               = str(st.get("goal", "")).strip()
                focus              = str(st.get("focus", "reasoning")).strip() or "reasoning"
                needs_tool    = self._as_bool(st.get("needs_tool"), default=False)
                tool_name     = str(st.get("tool_name", "none")).strip() or "none"
                candidate_url = str(st.get("candidate_url", "")).strip()
                search_query  = str(st.get("search_query", "")).strip()
                depends_on    = st.get("depends_on", [])
                if not isinstance(depends_on, list):
                    depends_on = []
                sub_id = str(st.get("id", f"s{i}")).strip() or f"s{i}"
            else:
                goal = str(st).strip()
                focus = "reasoning"; needs_tool = False; tool_name = "none"
                candidate_url = ""; search_query = ""; depends_on = []
                sub_id = f"s{i}"

            if sub_id in used_ids:
                sub_id = f"{sub_id}_{i}"
            used_ids.add(sub_id)

            if goal:
                normalized.append({
                    "id": sub_id, "goal": goal, "focus": focus,
                    "needs_tool": needs_tool, "tool_name": tool_name,
                    "candidate_url": candidate_url,
                    "search_query": search_query,
                    "depends_on": [str(d).strip() for d in depends_on if str(d).strip()],
                })

        if not normalized:
            if self.dataset == "math":
                normalized = [{
                    "id": "s1", "goal": query, "focus": "calculation",
                    "needs_tool": True, "tool_name": "math_tool",
                    "candidate_url": "", "search_query": "", "depends_on": [],
                }]
            elif self.dataset in ("hotpot", "musique"):
                normalized = [{
                    "id": "s1", "goal": query, "focus": "factual",
                    "needs_tool": True, "tool_name": "wikipedia_search",
                    "candidate_url": "", "search_query": "", "depends_on": [],
                }]
            else:
                normalized = [{
                    "id": "s1", "goal": query, "focus": "reasoning",
                    "needs_tool": True, "tool_name": "web_search",
                    "candidate_url": "", "search_query": "", "depends_on": [],
                }]

        normalized = self._topo_sort_subtasks(normalized)

        run_hops, run_name = extract_hop_metadata(kwargs)
        n_hops = run_hops if run_hops is not None else self._n_hops
        hop_name = run_name if run_name is not None else self._hop_name
        normalized = validate_planner_subtasks(
            normalized,
            dataset=self.dataset,
            n_hops=n_hops,
            hop_name=hop_name,
        )

        mem = WorkingMemory(
            goal             = query,
            open_questions   = [s["goal"] for s in normalized],
            final_constraint = str(parsed.get("final_constraint", "")).strip(),
        )

        return normalized, mem, latency, usage, cost

    # ── LLM-only Worker ────────────────────────────────────────────────────────

    def _worker(
        self,
        query:   str,
        subtask: dict[str, Any],
        mem:     WorkingMemory,
        attempt: int = 1,
    ) -> tuple[dict[str, Any], float, dict[str, int], float]:
        """
        [OPT-WRK1] Shorter worker system prompt — saves ~40 tokens per call.
        [OPT-WRK2] Pass only the compact dep-slice, not the full memory.
        """

        prior_context = self._build_prior_context(subtask, mem)

        user_prompt = (
            f"Memory: {mem.slice_for(subtask.get('depends_on', []))}\n"
            f"Subtask: {subtask['id']} | {subtask['goal']} | focus={subtask['focus']}"
            f"{prior_context}"
        )

        raw, latency, response = self._call_with_messages(
            self._prompts.worker_system, user_prompt,
            temperature=0.2, max_tokens=self.worker_max_tokens,
            model=self.worker_model,
        )
        usage = self._get_usage(response)
        cost  = self._compute_cost_for_model(
            self.worker_model, usage["prompt_tokens"], usage["completion_tokens"],
        )

        parsed = self._extract_json_object(raw) or {}
        try:
            confidence = float(parsed.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5

        raw_result = str(parsed.get("result", raw)).strip()

        result_signal = (
            self._semantic_extract(raw_result, subtask["goal"])
            if len(raw_result.split()) > 60 else raw_result
        )

        result: dict[str, Any] = {
            "subtask_id":    subtask["id"],
            "goal":          subtask["goal"],
            "focus":         subtask["focus"],
            "needs_tool":    False,
            "tool_name":     str(subtask.get("tool_name", "none")),
            "result":        result_signal,
            "result_raw":    raw_result,
            "confidence":    confidence,
            "evidence":      str(parsed.get("evidence", ""))[:self.worker_evidence_forward_chars],
            "self_critique": str(parsed.get("self_critique", "")).strip(),
            "attempt":       attempt,
        }

        if attempt == 1 and confidence < self.worker_retry_threshold:
            print(f"[WORKER][{subtask['id']}] Low confidence ({confidence:.2f}), retrying …")
            r2, l2, u2, c2 = self._worker(query, subtask, mem, attempt=2)
            if r2["confidence"] >= result["confidence"]:
                return r2, latency + l2, {
                    "prompt_tokens":     usage["prompt_tokens"]     + u2["prompt_tokens"],
                    "completion_tokens": usage["completion_tokens"] + u2["completion_tokens"],
                    "total_tokens":      usage["total_tokens"]      + u2["total_tokens"],
                }, cost + c2

        return result, latency, usage, cost

    # ── Tool-enabled Worker ────────────────────────────────────────────────────

    def _tool_worker(
        self,
        query:   str,
        subtask: dict[str, Any],
        mem:     WorkingMemory,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """
        [OPT-TOOL1] Enforce tool_max_steps in the ReAct query hint.
        [OPT-TOOL2] Pass only the dep-slice memory string (not full mem).
        [OPT-TOOL3] Shorter strategy hint.
        """
        try:
            from agent import react
        except Exception as exc:
            return self._tool_worker_error(subtask, str(exc))

        prior_context = self._build_prior_context(subtask, mem)
        strategy_hint = self._build_strategy_hint(subtask)
        tool_name     = str(subtask.get("tool_name", "none")).strip().lower()
        if tool_name == "math_tool":
            stop_hint = (
                "If math_tool returns a usable result, verify it before finish."
            )
        else:
            stop_hint = (
                "If the first fetch/search succeeds, return immediately. "
                "Do NOT repeat searches with minor query variations."
            )

        search_query = str(subtask.get("search_query", "") or "").strip()
        search_nudge = ""
        if search_query:
            search_nudge = (
                f"\nPlanner search_query (use for the first retrieval step): "
                f"{search_query}"
            )

        # [OPT-TOOL2] Compact memory slice + [OPT-TOOL1] step budget hint
        worker_query = (
            f"Memory: {mem.slice_for(subtask.get('depends_on', []))}\n"
            f"Goal: {subtask['goal']}\n"
            f"Tool: {subtask.get('tool_name', 'none')}"
            f"{strategy_hint}"
            f"{search_nudge}"
            f"{prior_context}"
            f"\nIMPORTANT: Use at most {self.tool_max_steps} tool calls. {stop_hint}"
        )

        wt = self._react_tools
        run_variants: list[dict[str, Any]] = [
            {
                "query": worker_query,
                "model": self.model,
                "dataset": self.dataset,
                "max_steps": self.tool_max_steps,
                "tools": wt,
            },
            {
                "query": worker_query,
                "model": self.model,
                "dataset": self.dataset,
                "max_steps": self.tool_max_steps,
            },
            {
                "query": worker_query,
                "model": self.model,
                "dataset": self.dataset,
            },
        ]
        response = None
        for kwargs in run_variants:
            try:
                response = react.run(**kwargs)
                break
            except TypeError:
                continue
            except Exception as exc:
                print(f"[TOOL_WORKER][{subtask['id']}] ReAct failed ({exc}); llm-only fallback.")
                return self._tool_worker_llm_fallback(query, subtask, mem, exc)
        if response is None:
            exc = TypeError("react.run rejected all kwargs variants (signature mismatch).")
            print(f"[TOOL_WORKER][{subtask['id']}] ReAct failed ({exc}); llm-only fallback.")
            return self._tool_worker_llm_fallback(query, subtask, mem, exc)

        compressed, response_answer_text, worker_conf = self._react_worker_payload(
            response, subtask,
        )

        result: dict[str, Any] = {
            "subtask_id":         subtask["id"],
            "goal":               subtask["goal"],
            "focus":              subtask["focus"],
            "needs_tool":         True,
            "tool_name":          str(subtask.get("tool_name", "none")),
            "result":             compressed,
            "result_raw":         response_answer_text,
            "confidence":         worker_conf,
            "evidence":           "ReAct tool worker; result semantically extracted.",
            "react_tools_called": response.tools_called,
        }
        usage: dict[str, Any] = {
            "prompt_tokens":     response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
            "total_tokens":      response.total_tokens,
            "cost_usd":          response.cost_usd,
            "latency_llm":       response.latency_llm,
            "latency_tools":     response.latency_tools,
            "tools_called":      response.tools_called,
            "tools_results":     response.tools_results,
            "num_tool_calls":    response.num_tool_calls,
            "num_llm_calls":     response.num_llm_calls,
        }
        return result, usage

    def _tool_worker_llm_fallback(
        self,
        query:   str,
        subtask: dict[str, Any],
        mem:     WorkingMemory,
        exc:     Exception,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        fb, fb_lat, fb_usage, fb_cost = self._worker(query, subtask, mem)
        fb["evidence"] += f" [tool failed: {exc}; llm-only fallback]"
        fb["confidence"] = min(fb["confidence"], 0.45)
        return fb, {
            **self._empty_tool_usage(),
            "prompt_tokens":     fb_usage["prompt_tokens"],
            "completion_tokens": fb_usage["completion_tokens"],
            "total_tokens":      fb_usage["total_tokens"],
            "cost_usd":          fb_cost,
            "latency_llm":       fb_lat,
            "num_llm_calls":     1,
        }

    @staticmethod
    def _tool_worker_error(subtask: dict[str, Any], reason: str) -> tuple[dict, dict]:
        return (
            {
                "subtask_id": subtask["id"], "goal": subtask["goal"],
                "focus": subtask["focus"], "needs_tool": True,
                "tool_name": str(subtask.get("tool_name", "none")),
                "result": f"Tool worker unavailable: {reason}",
                "confidence": 0.1, "evidence": reason,
            },
            {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                "cost_usd": 0.0, "latency_llm": 0.0, "latency_tools": 0.0,
                "tools_called": [], "tools_results": [],
                "num_tool_calls": 0, "num_llm_calls": 0,
            },
        )

    # ── Judge / Synthesizer ────────────────────────────────────────────────────

    def _judge_answer_fallback(self, query: str, malformed: str) -> str:
        try:
            out, _, _ = self._call_with_messages(
                self._prompts.judge_fallback_system,
                f"Q: {query}\n\nText:\n{malformed[:400]}",  # [OPT-JDG1] 600→400
                temperature=0.0, max_tokens=64,
                model=self.worker_model,
            )
            return out.strip()
        except Exception as exc:
            print(f"[JUDGE][fallback] extraction failed: {exc}")
            return ""

    def _judge(
        self,
        query:    str,
        subtasks: list[dict[str, Any]],
        mem:      WorkingMemory,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], float, dict[str, int], float]:
        """
        [OPT-JDG1] Tighter system prompt (~50 tokens saved).
        [OPT-JDG2] Judge receives only high-confidence facts (≥0.4) — filters
        noise from failed workers before they inflate the context.
        """
        # [OPT-JDG2] Filter low-confidence noise before serialising memory
        high_conf_facts = {
            k: v["text"] for k, v in mem.known_facts.items()
            if v["confidence"] >= 0.4 and v["text"].strip()
            and v["text"].lower() not in {"none", "null", "unknown"}
        }
        unique_facts  = set(high_conf_facts.values())
        conflict_note = (
            " WARNING: conflicting results — prefer tool-backed evidence."
            if len(unique_facts) > 1 else ""
        )

        system_prompt = self._prompts.judge_system + conflict_note
        # [OPT-JDG2] Pass only filtered memory
        filtered_mem_str = json.dumps(
            {"goal": mem.goal, "known_facts": high_conf_facts,
             "final_constraint": mem.final_constraint},
            ensure_ascii=False, separators=(",", ":"),
        )
        user_prompt = (
            f"{self._format_question(query, **kwargs)}\n\n"
            f"Working memory:\n{filtered_mem_str}"
        )

        raw, latency, response = self._call_with_messages(
            system_prompt, user_prompt,
            temperature=0.0, max_tokens=self.judge_max_tokens,
            model=self.judge_model,
        )
        usage = self._get_usage(response)
        cost  = self._compute_cost_for_model(
            self.judge_model, usage["prompt_tokens"], usage["completion_tokens"],
        )

        extraction_method   = "none"
        consensus_score     = 0.0
        rationale           = ""
        conflict_resolution = ""
        judge_confidence: float | None = None
        judge_complexity: float | None = None

        parsed_out = parse_llm_output_detailed(raw)
        meta = extract_last_json_dict(raw) or {}

        predicted_answer = ""
        if parsed_out.structured:
            predicted_answer = parsed_out.predicted_answer
            extraction_method = "parse_llm_output"
            judge_confidence = parsed_out.confidence
            judge_complexity = parsed_out.complexity

        if meta:
            try:
                consensus_score = float(meta.get("consensus_score", 0.0))
            except (TypeError, ValueError):
                consensus_score = 0.0
            rationale = str(meta.get("rationale", "")).strip()
            conflict_resolution = str(meta.get("conflict_resolution", "")).strip()
            if judge_confidence is None:
                judge_confidence = optional_float(meta.get("confidence"))
            if judge_complexity is None:
                judge_complexity = optional_float(meta.get("complexity"))

        if not predicted_answer:
            predicted_answer = self._judge_answer_fallback(query, raw)
            extraction_method = "llm_fallback" if predicted_answer else "failed"

        if not predicted_answer:
            predicted_answer = "unknown"
            extraction_method = "failed"

        predicted_answer = self._strip_answer_preamble(predicted_answer)

        return (
            {
                "predicted_answer":    predicted_answer,
                "consensus_score":     round(consensus_score, 4),
                "confidence":          judge_confidence,
                "complexity":          judge_complexity,
                "rationale":           rationale,
                "conflict_resolution": conflict_resolution,
                "raw_judge_output":    raw,
                "extraction_method":   extraction_method,
            },
            latency, usage, cost,
        )

    # ── Orchestrator ───────────────────────────────────────────────────────────

    def run(self, query: str, **kwargs: Any) -> AgentResponse:  # type: ignore[override]
        expected = kwargs.get("expected_answer")

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

            for subtask in subtasks[: self.max_workers]:

                # [OPT-SKIP] Check if a prior worker's result already covers this
                if subtask.get("depends_on"):
                    reused = self._find_reusable_result(subtask, mem)
                    if reused:
                        print(f"[OPT-SKIP][{subtask['id']}] Reusing prior result; "
                              f"skipping worker.")
                        out: dict[str, Any] = {
                            "subtask_id": subtask["id"],
                            "goal":       subtask["goal"],
                            "focus":      subtask["focus"],
                            "needs_tool": False,
                            "tool_name":  "none",
                            "result":     reused,
                            "result_raw": reused,
                            "confidence": 0.7,
                            "evidence":   "reused from prior worker",
                            "self_critique": "derived from dependency",
                            "attempt":    0,
                        }
                        worker_outputs.append(out)
                        sub_agents_run.append(f"worker_{subtask['id']}_reused")
                        sub_agent_responses.append({
                            "role": "worker", "mode": "reused",
                            "subtask_id": subtask["id"], "goal": subtask["goal"],
                            "output": out,
                        })
                        self._update_memory(mem, subtask, out)
                        continue

                # [OPT-SKIP] If the subtask needs a tool but deps already contain
                # the answer, demote it to LLM-only.
                use_tool = (
                    self.enable_tools_when_needed
                    and bool(subtask.get("needs_tool", False))
                    and not self._llm_can_answer(subtask, mem)
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

            response_obj = AgentResponse(
                query   = query,
                predicted_answer=predicted_answer,
                model   = self.model,
                agent   = "multiagent",
                dataset = self.dataset,
                latency_total  = total_latency_llm + total_latency_tools,
                latency_llm    = total_latency_llm,
                latency_tools  = total_latency_tools,
                prompt_tokens     = prompt_tokens,
                completion_tokens = completion_tokens,
                total_tokens      = total_tokens,
                cost_usd          = total_cost,
                reasoning_steps   = [s["goal"] for s in subtasks],
                num_llm_calls     = total_llm_calls,
                num_steps         = len(subtasks),
                tools_available   = react_tool_names_for_dataset(self.dataset),
                tools_called      = tools_called,
                tools_results     = tools_results,
                num_tool_calls    = num_tool_calls,
                max_steps            = self.max_subtasks + 2,
                steps_taken          = total_llm_calls + num_tool_calls,
                is_stopped_early     = False,
                orchestration_type   = "planner_workers_judge_working_memory_v6_optimised",
                sub_agents_run       = sub_agents_run,
                selected_agent       = "judge",
                sub_agent_responses  = sub_agent_responses,
                consensus_score      = judged.get("consensus_score"),
                confidence           = judged.get("confidence"),
                complexity           = judged.get("complexity"),
                expected_answer = expected,
                is_correct      = (
                    is_correct(predicted_answer, str(expected), dataset=self.dataset)
                    if expected else None
                ),
                error     = None,
                is_failed = False,
            )
            return self._finalize_response(response_obj)

        except Exception as exc:
            response_obj = AgentResponse(
                query   = query,
                predicted_answer='{"answer": "ERROR"}',
                model   = self.model,
                agent   = "multiagent",
                dataset = self.dataset,
                latency_total  = total_latency_llm + total_latency_tools,
                latency_llm    = total_latency_llm,
                latency_tools  = total_latency_tools,
                prompt_tokens     = prompt_tokens,
                completion_tokens = completion_tokens,
                total_tokens      = total_tokens,
                cost_usd          = total_cost,
                num_llm_calls     = total_llm_calls,
                tools_available   = react_tool_names_for_dataset(self.dataset),
                tools_called      = tools_called,
                tools_results     = tools_results,
                num_tool_calls    = num_tool_calls,
                orchestration_type  = "planner_workers_judge_working_memory_v6_optimised",
                sub_agents_run      = sub_agents_run,
                sub_agent_responses = sub_agent_responses,
                expected_answer = expected,
                is_correct      = None,
                error           = str(exc),
                is_failed       = True,
            )
            return self._finalize_response(response_obj)


# ── Module-level entry point ───────────────────────────────────────────────────

def run(query: str, model: str, dataset: str, **kwargs: Any) -> AgentResponse:
    agent = MultiAgentAgent(model=model, dataset=dataset, **kwargs)
    return agent.run(query=query, **kwargs)

