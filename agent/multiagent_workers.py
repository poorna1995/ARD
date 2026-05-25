from __future__ import annotations

import json
import re
import time
from typing import Any

from agent.dataset_profile import (
    is_open_wiki_qa,
    looks_country_span,
    looks_temporal,
    observation_text_from_tools_results,
    query_relation_type,
    subtask_relation_type,
    terminal_subtask_ids,
)
from agent.multiagent_memory import WorkingMemory


class WorkerChainBlockedError(RuntimeError):
    """Raised when a dependent or terminal worker cannot produce a usable result."""

    def __init__(self, subtask_id: str, goal: str, *, reason: str = "") -> None:
        detail = f" [{reason}]" if reason else ""
        super().__init__(f"worker chain blocked at {subtask_id!r}{detail}: {goal[:120]}")
from evaluator.parse import extract_last_json_dict


class WorkerMixin:
    _TOOL_ERROR_PREFIX = "tool worker unavailable"

    _STRUCTURED_SOURCE_PATTERNS: list[str] = [
        r"usgs\.gov",
        r"gbif\.org",
        r"epa\.gov",
        r"census\.gov",
        r"data\.gov",
        r"ncbi\.nlm\.nih\.gov",
        r"wikipedia\.org",
    ]

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
        return extract_last_json_dict(WorkerMixin._coerce_text(text))

    def _build_strategy_hint(self, subtask: dict[str, Any]) -> str:
        tool_name = str(subtask.get("tool_name", "none")).strip().lower()
        candidate_url = str(subtask.get("candidate_url", "") or "").strip()
        search_query = str(subtask.get("search_query", "") or "").strip()

        if tool_name == "math_tool":
            return (
                "\nStrategy: math_tool[<python_expression>] for every numeric step; "
                "pure Python only — no units or words."
            )
        if tool_name == "retrieve":
            if search_query:
                return (
                    f"\nStrategy: first Action must be retrieve[{search_query}] — "
                    "short semantic phrase over Context; copy exact answer span."
                )
            return (
                "\nStrategy: retrieve[semantic_phrase] over Context; "
                "copy exact answer span from Observation."
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
        answer = WorkerMixin._coerce_text(answer)
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
        }

    def _call_with_messages(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
        model: str | None = None,
    ) -> tuple[str, float, Any]:
        effective_model = model or self.model
        client = self._get_client_for_model(effective_model)
        start = time.perf_counter()
        response = client.chat.completions.create(
            model=effective_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=self.temperature if temperature is None else temperature,
            max_tokens=self.max_tokens if max_tokens is None else max_tokens,
        )
        latency = time.perf_counter() - start
        content = response.choices[0].message.content
        return self._coerce_text(content), latency, response

    @staticmethod
    def _retrieve_top_chunks(
        text: str,
        query: str,
        chunk_words: int = 350,
        overlap: int = 100,
        top_k: int = 3,
    ) -> str:
        words = text.split()
        if len(words) <= chunk_words * top_k:
            return text
        step = chunk_words - overlap
        chunks = [
            (i, " ".join(words[i : i + chunk_words])) for i in range(0, len(words), step)
        ]
        terms = set(re.sub(r"[^\w\s]", "", query).lower().split())
        ranked = sorted(
            chunks,
            key=lambda x: sum(1 for t in terms if t in x[1].lower()),
            reverse=True,
        )
        top = sorted(ranked[:top_k], key=lambda x: x[0])
        return "\n---\n".join(c for _, c in top)

    def _semantic_extract(self, raw_text: str, goal: str) -> str:
        focused = self._retrieve_top_chunks(raw_text, goal)
        if len(focused.split()) < 200:
            return focused
        try:
            out, _, _ = self._call_with_messages(
                self._prompts.extractor_system,
                f"Q: {goal}\n\nDoc:\n{focused}",
                temperature=0.0,
                max_tokens=self.extractor_max_tokens,
                model=self.worker_model,
            )
            out = out.strip()
            return focused if out == "NO_RELEVANT_CONTENT" else out
        except Exception as exc:
            print(f"[EXTRACT] semantic extraction failed ({exc}); using chunks.")
            return focused

    @staticmethod
    def _is_structured_source(url: str) -> bool:
        url_lower = (url or "").lower()
        return any(re.search(p, url_lower) for p in WorkerMixin._STRUCTURED_SOURCE_PATTERNS)

    def _structured_extract(self, raw_text: str, fields: list[str], goal: str) -> str:
        fields_str = ", ".join(f'"{f}"' for f in fields)
        system = self._prompts.structured_extractor_template.format(fields_str=fields_str)
        focused = self._retrieve_top_chunks(raw_text, goal)
        try:
            raw, _, _ = self._call_with_messages(
                system,
                f"Doc:\n{focused}",
                temperature=0.0,
                max_tokens=self.extractor_max_tokens,
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
        tools_results = getattr(response, "tools_results", None)
        pred = self._coerce_text(getattr(response, "predicted_answer", "")).strip()
        obs_text = observation_text_from_tools_results(tools_results)
        cap = self.worker_result_forward_chars

        if pred:
            clean = self._strip_answer_preamble(pred)
            return clean[:cap], pred, 0.75

        if obs_text:
            exact = obs_text[:cap]
            raw = obs_text[: cap * 2]
            return exact, raw, 0.55

        return "", "", 0.2

    def _process_tool_result(self, raw: Any, subtask: dict[str, Any], source_url: str = "") -> str:
        raw_text = self._coerce_text(raw)
        cap = int(getattr(self, "worker_result_forward_chars", 200))
        return raw_text[:cap] if raw_text else ""

    @staticmethod
    def _worker_result_text(result: dict[str, Any]) -> str:
        return str(result.get("result", "") or "").strip()

    @classmethod
    def _worker_output_ok(cls, result: dict[str, Any]) -> bool:
        text = cls._worker_result_text(result)
        if not text:
            return False
        if text.lower().startswith(cls._TOOL_ERROR_PREFIX):
            return False
        return True

    def _guard_worker_output(
        self,
        subtask: dict[str, Any],
        result: dict[str, Any],
        subtasks: list[dict[str, Any]],
    ) -> None:
        """Block the run when a dependent or terminal worker has no usable result."""
        if self._worker_output_ok(result):
            return
        sid = str(subtask.get("id", "")).strip()
        depends_on = subtask.get("depends_on") or []
        terminals = set(terminal_subtask_ids(subtasks))
        upstream = {
            str(d).strip()
            for st in subtasks
            for d in (st.get("depends_on") or [])
            if str(d).strip()
        }
        if depends_on or sid in terminals or sid in upstream:
            raise WorkerChainBlockedError(
                sid,
                str(subtask.get("goal", "")),
                reason="empty_result_after_retry",
            )

    def _update_memory(
        self,
        mem: WorkingMemory,
        subtask: dict[str, Any],
        result: dict[str, Any],
    ) -> None:
        sid = result.get("subtask_id", subtask["id"])
        confidence = float(result.get("confidence", 0.5))
        verified = str(result.get("result", "")).strip()
        raw_span = str(result.get("result_raw", "")).strip()
        evidence = str(result.get("evidence", "")).strip()
        cap = self.worker_result_forward_chars
        if verified:
            signal = verified[:cap]
        elif raw_span:
            signal = raw_span[:cap]
        elif evidence and evidence != "ReAct tool worker; exact evidence preserved.":
            signal = evidence[:cap]
        else:
            signal = ""
        if not signal:
            mem.close_question(subtask["goal"])
            return
        tool_backed = bool(result.get("needs_tool"))
        mem.record_fact(sid, signal, confidence, tool_backed=tool_backed)
        mem.close_question(subtask["goal"])

    def _verify_worker_result(
        self,
        subtask: dict[str, Any],
        candidate: str,
        evidence: str,
        *,
        query: str = "",
    ) -> tuple[str, float, float, dict[str, int], float]:
        """Lightweight retrieve→verify: relation check + trim to shortest exact evidence span."""
        candidate = str(candidate or "").strip()
        evidence = str(evidence or "").strip()
        empty_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if not candidate or not evidence:
            return candidate, 0.5, 0.0, empty_usage, 0.0

        st_rel = subtask_relation_type(str(subtask.get("goal", "")))
        q_rel = query_relation_type(query) if query else None
        if st_rel == "when" and not looks_temporal(candidate):
            return "", 0.0, 0.0, empty_usage, 0.0
        if st_rel == "where" and q_rel == "where" and looks_country_span(candidate):
            return "", 0.0, 0.0, empty_usage, 0.0
        if st_rel == "country" and q_rel == "country" and "," in candidate:
            parts = [p.strip() for p in candidate.split(",") if p.strip()]
            if len(parts) >= 2:
                candidate = parts[-1]

        cap = self.worker_result_forward_chars
        question_line = f"Question: {query}\n" if str(query or "").strip() else ""
        user_prompt = (
            f"{question_line}"
            f"Subtask: {subtask['goal']}\n"
            f"Candidate: {candidate}\n\n"
            f"Evidence:\n{evidence[: cap * 2]}"
        )
        raw, latency, response = self._call_with_messages(
            self._prompts.worker_verify_system,
            user_prompt,
            temperature=0.0,
            max_tokens=min(256, self.worker_max_tokens),
            model=self.worker_model,
        )
        usage = self._get_usage(response)
        cost = self._compute_cost_for_model(
            self.worker_model, usage["prompt_tokens"], usage["completion_tokens"]
        )
        parsed = self._extract_json_object(raw) or {}
        verified = str(parsed.get("result", "")).strip()
        try:
            conf = float(parsed.get("confidence", 0.65) or 0.65)
        except (TypeError, ValueError):
            conf = 0.65
        if conf <= 0.0 or not verified:
            return "", 0.2, latency, usage, cost
        return verified[:cap], conf, latency, usage, cost

    @staticmethod
    def _evidence_ambiguity_note(evidence: str) -> str:
        years = set(re.findall(r"\b(?:19|20)\d{2}\b", evidence or ""))
        if len(years) >= 2:
            return (
                " Evidence contains multiple years — verify which satisfies the "
                "subtask before Final Answer."
            )
        return ""

    @staticmethod
    def _build_prior_context(subtask: dict[str, Any], mem: WorkingMemory) -> str:
        dep_ids: list[str] = subtask.get("depends_on", [])
        if not dep_ids:
            return ""
        relevant = {k: v.get("text", "") for k, v in mem.known_facts.items() if k in dep_ids}
        if not relevant:
            return ""
        lines = [f"  [{sid}] {fact}" for sid, fact in relevant.items()]
        return "\n\nContext from earlier steps (deps only):\n" + "\n".join(lines)

    def _find_reusable_result(self, subtask: dict[str, Any], mem: WorkingMemory) -> str | None:
        for sid, fact in mem.known_facts.items():
            if sid == subtask["id"]:
                continue
            fact_text = self._coerce_text(fact.get("text", "") if isinstance(fact, dict) else fact).strip()
            if not fact_text or len(fact_text.split()) < 3:
                continue
            subtask_words = set(re.sub(r"[^\w]", " ", subtask["goal"].lower()).split())
            fact_words = set(re.sub(r"[^\w]", " ", fact_text.lower()).split())
            if len(subtask_words & fact_words) / max(len(subtask_words), 1) > 0.6:
                return fact_text
        return None

    @staticmethod
    def _llm_can_answer(subtask: dict[str, Any], mem: WorkingMemory) -> bool:
        dep_ids: list[str] = subtask.get("depends_on", [])
        if not dep_ids:
            return False
        goal_words = set(re.sub(r"[^\w]", " ", subtask["goal"].lower()).split())
        for dep_id in dep_ids:
            fact = mem.known_facts.get(dep_id, {})
            fact_text = WorkerMixin._coerce_text(
                fact.get("text", "") if isinstance(fact, dict) else fact
            ).strip()
            if not fact_text:
                continue
            fact_words = set(re.sub(r"[^\w]", " ", fact_text.lower()).split())
            overlap = len(goal_words & fact_words) / max(len(goal_words), 1)
            if overlap > 0.50:
                return True
        return False

    def _worker(
        self,
        query: str,
        subtask: dict[str, Any],
        mem: WorkingMemory,
        attempt: int = 1,
    ) -> tuple[dict[str, Any], float, dict[str, int], float]:
        prior_context = self._build_prior_context(subtask, mem)
        problem_block = ""
        if (self.dataset or "").strip().lower() == "math":
            problem_block = f"Problem (full):\n{query}\n\n"
        user_prompt = (
            f"{problem_block}"
            f"Memory: {mem.slice_for(subtask.get('depends_on', []))}\n"
            f"Subtask: {subtask['id']} | {subtask['goal']} | focus={subtask['focus']}"
            f"{prior_context}"
        )
        raw, latency, response = self._call_with_messages(
            self._prompts.worker_system,
            user_prompt,
            temperature=0.2,
            max_tokens=self.worker_max_tokens,
            model=self.worker_model,
        )
        usage = self._get_usage(response)
        cost = self._compute_cost_for_model(
            self.worker_model, usage["prompt_tokens"], usage["completion_tokens"]
        )

        parsed = self._extract_json_object(raw) or {}
        try:
            confidence = float(parsed.get("confidence", 0.5) or 0.5)
        except (TypeError, ValueError):
            confidence = 0.5

        raw_result = str(parsed.get("result", raw)).strip()
        evidence_text = str(parsed.get("evidence", raw_result)).strip()
        result_signal = raw_result[: self.worker_result_forward_chars]
        if result_signal and evidence_text:
            verified, vconf, vlat, vusage, vcost = self._verify_worker_result(
                subtask, result_signal, evidence_text, query=query,
            )
            if verified:
                result_signal = verified
                confidence = max(confidence, vconf)
            elif vconf <= 0.2:
                result_signal = ""
                confidence = min(confidence, vconf)
            latency += vlat
            usage = {
                "prompt_tokens": usage["prompt_tokens"] + vusage["prompt_tokens"],
                "completion_tokens": usage["completion_tokens"] + vusage["completion_tokens"],
                "total_tokens": usage["total_tokens"] + vusage["total_tokens"],
            }
            cost += vcost
        result: dict[str, Any] = {
            "subtask_id": subtask["id"],
            "goal": subtask["goal"],
            "focus": subtask["focus"],
            "needs_tool": False,
            "tool_name": str(subtask.get("tool_name", "none")),
            "result": result_signal,
            "result_raw": raw_result,
            "confidence": confidence,
            "evidence": str(parsed.get("evidence", ""))[: self.worker_evidence_forward_chars],
            "self_critique": str(parsed.get("self_critique", "")).strip(),
            "attempt": attempt,
        }

        if attempt == 1 and (
            confidence < self.worker_retry_threshold
            or not self._worker_output_ok(result)
        ):
            reason = (
                "empty result"
                if not self._worker_output_ok(result)
                else f"low confidence ({confidence:.2f})"
            )
            print(f"[WORKER][{subtask['id']}] {reason}, retrying …")
            r2, l2, u2, c2 = self._worker(query, subtask, mem, attempt=2)
            if r2["confidence"] >= result["confidence"] or (
                self._worker_output_ok(r2) and not self._worker_output_ok(result)
            ):
                return (
                    r2,
                    latency + l2,
                    {
                        "prompt_tokens": usage["prompt_tokens"] + u2["prompt_tokens"],
                        "completion_tokens": usage["completion_tokens"] + u2["completion_tokens"],
                        "total_tokens": usage["total_tokens"] + u2["total_tokens"],
                    },
                    cost + c2,
                )
        return result, latency, usage, cost

    def _tool_worker(
        self, query: str, subtask: dict[str, Any], mem: WorkingMemory
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        try:
            from agent import react
        except Exception as exc:
            return self._tool_worker_error(subtask, str(exc))

        prior_context = self._build_prior_context(subtask, mem)
        strategy_hint = self._build_strategy_hint(subtask)
        tool_name = str(subtask.get("tool_name", "none")).strip().lower()
        if tool_name == "math_tool":
            stop_hint = (
                "Use math_tool for every numeric step. "
                "Final Answer must match the Problem's requested representation "
                "(base notation like 1_6, matrix form, exact symbolic value)."
            )
        elif self.dataset == "gaia":
            stop_hint = (
                "After web_search, web_fetch a promising URL before searching again. "
                "Verify extracted value matches question type (number-only → digits). "
                "After 2 failed searches or repeated observations, finalize best grounded answer."
            )
        elif is_open_wiki_qa(self.dataset):
            stop_hint = (
                "retrieve → verify shortest exact span answers subtask → Final Answer. "
                "If Evidence has multiple years or competing entities, retrieve again "
                "with a narrower query before finalizing."
            )
        else:
            stop_hint = (
                "Verify tool output matches subtask before Final Answer. "
                "If the same tool+query yields the same observation twice, finalize from "
                "existing evidence (no new evidence obtained)."
            )

        search_query = str(subtask.get("search_query", "") or "").strip()
        search_nudge = (
            f"\nPlanner search_query (use for the first retrieval step): {search_query}"
            if search_query
            else ""
        )
        problem_block = ""
        if (self.dataset or "").strip().lower() == "math":
            problem_block = f"Problem (full):\n{query}\n\n"
        worker_query = (
            f"{problem_block}"
            f"Memory: {mem.slice_for(subtask.get('depends_on', []))}\n"
            f"Subtask goal: {subtask['goal']}\n"
            f"Tool: {subtask.get('tool_name', 'none')}"
            f"{strategy_hint}"
            f"{search_nudge}"
            f"{prior_context}"
            f"\nIMPORTANT: Use at most {self.tool_max_steps} tool calls. {stop_hint}"
        )

        react_kwargs: dict[str, Any] = {
            "query": worker_query,
            "model": self.model,
            "dataset": self.dataset,
            "max_steps": self.tool_max_steps,
        }
        episode_ctx = getattr(self, "_episode_context", None)
        if isinstance(episode_ctx, dict) and episode_ctx:
            react_kwargs["context"] = episode_ctx

        try:
            response = react.run(**react_kwargs)
        except Exception as exc:
            print(f"[TOOL_WORKER][{subtask['id']}] ReAct failed ({exc}); llm-only fallback.")
            return self._tool_worker_llm_fallback(query, subtask, mem, exc)

        compressed, response_answer_text, worker_conf = self._react_worker_payload(response, subtask)
        evidence = observation_text_from_tools_results(response.tools_results) or response_answer_text
        verify_lat = 0.0
        verify_usage = self._empty_tool_usage()
        verify_cost = 0.0
        if compressed and evidence:
            verified, vconf, verify_lat, verify_usage, verify_cost = self._verify_worker_result(
                subtask, compressed, evidence, query=query,
            )
            if verified:
                compressed = verified
                worker_conf = max(worker_conf, vconf)
            elif vconf <= 0.2:
                raw_fb = self._strip_answer_preamble(response_answer_text).strip()
                if raw_fb:
                    compressed = raw_fb[: self.worker_result_forward_chars]
                    worker_conf = max(0.35, min(worker_conf, 0.55))
                else:
                    compressed = ""
                    worker_conf = min(worker_conf, vconf)
        result: dict[str, Any] = {
            "subtask_id": subtask["id"],
            "goal": subtask["goal"],
            "focus": subtask["focus"],
            "needs_tool": True,
            "tool_name": str(subtask.get("tool_name", "none")),
            "result": compressed,
            "result_raw": response_answer_text,
            "confidence": worker_conf,
            "evidence": (evidence or response_answer_text)[: self.worker_evidence_forward_chars],
            "react_tools_called": response.tools_called,
            "attempt": 1,
        }
        retry_response = None
        if not self._worker_output_ok(result):
            print(f"[TOOL_WORKER][{subtask['id']}] Empty result, retrying …")
            try:
                retry_response = react.run(**react_kwargs)
            except Exception as exc:
                print(f"[TOOL_WORKER][{subtask['id']}] Retry failed ({exc}).")
            else:
                c2, r2, conf2 = self._react_worker_payload(retry_response, subtask)
                ev2 = observation_text_from_tools_results(retry_response.tools_results) or r2
                if c2 and ev2:
                    v2, vc2, vl2, vu2, vcost2 = self._verify_worker_result(
                        subtask, c2, ev2, query=query,
                    )
                    if v2:
                        c2, conf2 = v2, max(conf2, vc2)
                    elif vc2 <= 0.2:
                        raw_fb = self._strip_answer_preamble(r2).strip()
                        if raw_fb:
                            c2 = raw_fb[: self.worker_result_forward_chars]
                            conf2 = max(0.35, min(conf2, 0.55))
                        else:
                            c2 = ""
                            conf2 = min(conf2, vc2)
                    verify_lat += vl2
                    verify_usage = {
                        "prompt_tokens": int(verify_usage.get("prompt_tokens", 0))
                        + int(vu2.get("prompt_tokens", 0)),
                        "completion_tokens": int(verify_usage.get("completion_tokens", 0))
                        + int(vu2.get("completion_tokens", 0)),
                        "total_tokens": int(verify_usage.get("total_tokens", 0))
                        + int(vu2.get("total_tokens", 0)),
                    }
                    verify_cost += vcost2
                if self._worker_output_ok({"result": c2}):
                    result["result"] = c2
                    result["result_raw"] = r2
                    result["confidence"] = conf2
                    result["evidence"] = (ev2 or r2)[: self.worker_evidence_forward_chars]
                    result["react_tools_called"] = retry_response.tools_called
                    result["attempt"] = 2
                    response = retry_response

        if not self._worker_output_ok(result):
            raw_fb = self._strip_answer_preamble(
                str(result.get("result_raw", ""))
            ).strip()
            if raw_fb:
                result["result"] = raw_fb[: self.worker_result_forward_chars]
                result["confidence"] = max(
                    0.35,
                    min(float(result.get("confidence", 0.2) or 0.2), 0.55),
                )

        react_prompt = int(response.prompt_tokens)
        react_completion = int(response.completion_tokens)
        react_total = int(response.total_tokens)
        react_cost = float(response.cost_usd)
        react_llm_lat = float(response.latency_llm)
        react_tool_lat = float(response.latency_tools)
        react_llm_calls = int(response.num_llm_calls)
        react_tool_calls = int(response.num_tool_calls)
        tools_called = list(response.tools_called)
        tools_results = list(response.tools_results)
        if retry_response is not None and result.get("attempt") == 2:
            react_prompt += int(retry_response.prompt_tokens)
            react_completion += int(retry_response.completion_tokens)
            react_total += int(retry_response.total_tokens)
            react_cost += float(retry_response.cost_usd)
            react_llm_lat += float(retry_response.latency_llm)
            react_tool_lat += float(retry_response.latency_tools)
            react_llm_calls += int(retry_response.num_llm_calls)
            react_tool_calls += int(retry_response.num_tool_calls)
            tools_called.extend(retry_response.tools_called)
            tools_results.extend(retry_response.tools_results)

        usage: dict[str, Any] = {
            "prompt_tokens": react_prompt + int(verify_usage.get("prompt_tokens", 0)),
            "completion_tokens": react_completion + int(verify_usage.get("completion_tokens", 0)),
            "total_tokens": react_total + int(verify_usage.get("total_tokens", 0)),
            "cost_usd": react_cost + verify_cost,
            "latency_llm": react_llm_lat + verify_lat,
            "latency_tools": react_tool_lat,
            "tools_called": tools_called,
            "tools_results": tools_results,
            "num_tool_calls": react_tool_calls,
            "num_llm_calls": react_llm_calls + (1 if verify_lat > 0 else 0),
        }
        return result, usage

    def _tool_worker_llm_fallback(
        self,
        query: str,
        subtask: dict[str, Any],
        mem: WorkingMemory,
        exc: Exception,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        fb, fb_lat, fb_usage, fb_cost = self._worker(query, subtask, mem)
        fb["evidence"] += f" [tool failed: {exc}; llm-only fallback]"
        fb["confidence"] = min(fb["confidence"], 0.45)
        return fb, {
            **self._empty_tool_usage(),
            "prompt_tokens": fb_usage["prompt_tokens"],
            "completion_tokens": fb_usage["completion_tokens"],
            "total_tokens": fb_usage["total_tokens"],
            "cost_usd": fb_cost,
            "latency_llm": fb_lat,
            "num_llm_calls": 1,
        }

    @staticmethod
    def _tool_worker_error(subtask: dict[str, Any], reason: str) -> tuple[dict, dict]:
        return (
            {
                "subtask_id": subtask["id"],
                "goal": subtask["goal"],
                "focus": subtask["focus"],
                "needs_tool": True,
                "tool_name": str(subtask.get("tool_name", "none")),
                "result": f"Tool worker unavailable: {reason}",
                "confidence": 0.1,
                "evidence": reason,
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

