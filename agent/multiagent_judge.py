from __future__ import annotations

import re
from typing import Any

from agent.dataset_profile import (
    looks_country_span,
    looks_temporal,
    query_relation_type,
    terminal_subtask_ids,
)
from evaluator.parse import extract_last_json_dict, optional_float, parse_llm_output_detailed


class JudgeMixin:
    @staticmethod
    def _matches_fact_span(predicted: str, fact_text: str) -> bool:
        pred = str(predicted or "").strip()
        fact = str(fact_text or "").strip()
        if not pred or not fact:
            return False
        if pred == fact or pred.lower() == fact.lower():
            return True
        pl, fl = pred.lower(), fact.lower()
        return pl in fl or fl in pl

    @staticmethod
    def _ground_judge_in_facts(
        predicted: str,
        high_conf_facts: dict[str, dict[str, Any]],
        terminals: list[str],
        query: str,
    ) -> str:
        pred = str(predicted or "").strip()
        pool = high_conf_facts
        if terminals:
            term_set = set(terminals)
            pool = {
                sid: row
                for sid, row in high_conf_facts.items()
                if sid in term_set and str(row.get("text", "")).strip()
            }
            if not pool:
                return ""
        if not pool:
            return pred
        if pred and any(
            JudgeMixin._matches_fact_span(pred, str(v.get("text", "")))
            for v in pool.values()
        ):
            return pred

        terminal_set = set(terminals)
        ranked: list[tuple[float, str]] = []
        for sid, fact in pool.items():
            text = str(fact.get("text", "")).strip()
            if not text:
                continue
            score = float(fact.get("confidence", 0))
            if sid in terminal_set:
                score += 0.15
            if fact.get("tool_backed"):
                score += 0.05
            ranked.append((score, text))
        if not ranked:
            return ""

        rel = query_relation_type(query)
        if rel == "when":
            temporal = [(s, t) for s, t in ranked if looks_temporal(t)]
            if temporal:
                return max(temporal, key=lambda x: x[0])[1]
        if rel == "where":
            non_country = [(s, t) for s, t in ranked if not looks_country_span(t)]
            if non_country:
                return max(non_country, key=lambda x: x[0])[1]
        return max(ranked, key=lambda x: x[0])[1]

    @staticmethod
    def _enforce_answer_type(
        query: str,
        predicted: str,
        high_conf_facts: dict[str, dict[str, Any]],
        terminals: list[str],
    ) -> str:
        pred = str(predicted or "").strip()
        if not pred:
            return pred
        pool = high_conf_facts
        if terminals:
            term_set = set(terminals)
            pool = {
                sid: row
                for sid, row in high_conf_facts.items()
                if sid in term_set and str(row.get("text", "")).strip()
            }
        rel = query_relation_type(query)
        if rel == "country" and "," in pred:
            parts = [p.strip() for p in pred.split(",") if p.strip()]
            if len(parts) >= 2:
                return parts[-1]
        if rel == "when" and not looks_temporal(pred):
            temporal = [
                str(v.get("text", "")).strip()
                for v in pool.values()
                if looks_temporal(str(v.get("text", "")))
            ]
            if temporal:
                term_id = terminals[0] if len(terminals) == 1 else None
                if term_id and pool.get(term_id, {}).get("text"):
                    term_text = str(pool[term_id]["text"]).strip()
                    if looks_temporal(term_text):
                        return term_text
                return temporal[0]
        if rel == "where" and looks_country_span(pred):
            for sid in reversed(terminals or list(pool.keys())):
                text = str(pool.get(sid, {}).get("text", "")).strip()
                if text and not looks_country_span(text):
                    return text
        return pred

    @staticmethod
    def _prefer_terminal_answer(
        subtasks: list[dict[str, Any]],
        high_conf_facts: dict[str, dict[str, Any]],
        terminals: list[str],
        predicted: str,
    ) -> str:
        """If judge picked an intermediate-hop fact, prefer the terminal subtask fact."""
        pred = str(predicted or "").strip()
        if not pred or not terminals:
            return pred
        terminal_set = set(terminals)
        for sid, fact in high_conf_facts.items():
            if sid in terminal_set:
                continue
            if str(fact.get("text", "")).strip() == pred:
                for term_id in terminals:
                    term_text = str(
                        high_conf_facts.get(term_id, {}).get("text", "")
                    ).strip()
                    if term_text:
                        return term_text
                return ""
        if len(terminal_set) != 1:
            return pred
        term_id = terminals[0]
        term_row = high_conf_facts.get(term_id, {})
        term_text = str(term_row.get("text", "")).strip()
        if not term_text or term_text == pred:
            return pred
        for sid, fact in high_conf_facts.items():
            if sid in terminal_set:
                continue
            if str(fact.get("text", "")).strip() != pred:
                continue
            if term_row.get("tool_backed") or fact.get("tool_backed"):
                return term_text
        return pred

    @staticmethod
    def _raw_has_extractable_answer(raw: str) -> bool:
        if not raw or not str(raw).strip():
            return False
        text = str(raw).strip()
        if extract_last_json_dict(text):
            return True
        if re.search(r'(?i)"answer"\s*:', text):
            return True
        if re.search(r"(?i)final\s+answer\s*:", text):
            return True
        return False

    @staticmethod
    def _coerce_judge_answer(text: str, dataset: str) -> str:
        ds = (dataset or "").strip().lower()
        if not text:
            return text
        if ds == "math":
            from evaluator.grade import normalize_math_answer

            return normalize_math_answer(text)
        if ds == "mmlu":
            from evaluator.grade import normalize_mmlu_answer

            return normalize_mmlu_answer(text)
        return text.strip()

    def _judge_answer_fallback(self, query: str, malformed: str) -> str:
        try:
            out, _, _ = self._call_with_messages(
                self._prompts.judge_fallback_system,
                f"Q: {query}\n\nText:\n{malformed[:400]}",
                temperature=0.0,
                max_tokens=64,
                model=self.worker_model,
            )
            return out.strip()
        except Exception as exc:
            print(f"[JUDGE][fallback] extraction failed: {exc}")
            return ""

    def _judge(
        self,
        query: str,
        subtasks: list[dict[str, Any]],
        mem: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], float, dict[str, int], float]:
        high_conf_facts = {
            k: v
            for k, v in mem.known_facts.items()
            if v.get("confidence", 0) >= 0.4
            and str(v.get("text", "")).strip()
            and str(v.get("text", "")).strip().lower() not in {"none", "null", "unknown"}
        }
        unique_facts = {str(v.get("text", "")).strip() for v in high_conf_facts.values()}
        terminals = terminal_subtask_ids(subtasks)
        term_set = set(terminals)
        terminal_facts = (
            {
                sid: high_conf_facts[sid]
                for sid in terminals
                if sid in high_conf_facts and str(high_conf_facts[sid].get("text", "")).strip()
            }
            if terminals
            else high_conf_facts
        )
        terminal_note = ""
        if terminals:
            terminal_note = (
                f" Terminal subtasks (sink nodes — prefer for final answer): "
                f"{', '.join(terminals)}."
            )
        conflict_note = (
            " WARNING: conflicting results — prefer tool-backed terminal subtask evidence."
            if len(unique_facts) > 1
            else ""
        )

        system_prompt = self._prompts.judge_system + conflict_note + terminal_note
        annotated_facts = {
            sid: {**fact, "terminal": sid in terminals}
            for sid, fact in high_conf_facts.items()
        }
        filtered_mem_str = self._coerce_text(
            {
                "goal": mem.goal,
                "known_facts": annotated_facts,
                "final_constraint": mem.final_constraint,
            }
        )
        user_prompt = (
            f"{self._format_question(query, **kwargs)}\n\n"
            f"Working memory:\n{filtered_mem_str}"
        )

        raw, latency, response = self._call_with_messages(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_tokens=self.judge_max_tokens,
            model=self.judge_model,
        )
        usage = self._get_usage(response)
        cost = self._compute_cost_for_model(
            self.judge_model, usage["prompt_tokens"], usage["completion_tokens"]
        )

        extraction_method = "none"
        consensus_score = 0.0
        rationale = ""
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
            if self._raw_has_extractable_answer(raw):
                predicted_answer = self._judge_answer_fallback(query, raw)
                extraction_method = "llm_fallback" if predicted_answer else "failed"
            else:
                extraction_method = "failed"
        if not predicted_answer:
            predicted_answer = "unknown"
            extraction_method = "failed"

        terminal_missing = bool(terminals) and not terminal_facts

        predicted_answer = self._strip_answer_preamble(predicted_answer)
        predicted_answer = self._coerce_judge_answer(predicted_answer, self.dataset)

        if terminal_missing:
            predicted_answer = ""
            extraction_method = "terminal_facts_missing"
            rationale = (
                rationale or "Terminal subtasks produced no facts; cannot answer."
            )
        else:
            if terminals and predicted_answer:
                matches_terminal = any(
                    sid in term_set
                    and self._matches_fact_span(
                        predicted_answer, str(fact.get("text", ""))
                    )
                    for sid, fact in high_conf_facts.items()
                )
                if not matches_terminal and any(
                    sid not in term_set
                    and self._matches_fact_span(
                        predicted_answer, str(fact.get("text", ""))
                    )
                    for sid, fact in high_conf_facts.items()
                ):
                    predicted_answer = ""
                    extraction_method = "rejected_intermediate_hop"
                    rationale = (
                        rationale or "Rejected answer from non-terminal subtask."
                    )
            predicted_answer = self._ground_judge_in_facts(
                predicted_answer, high_conf_facts, terminals, query
            )
            predicted_answer = self._prefer_terminal_answer(
                subtasks, high_conf_facts, terminals, predicted_answer
            )
            predicted_answer = self._enforce_answer_type(
                query, predicted_answer, high_conf_facts, terminals
            )
            if terminals and predicted_answer:
                matches_terminal = any(
                    sid in term_set
                    and self._matches_fact_span(
                        predicted_answer, str(fact.get("text", ""))
                    )
                    for sid, fact in high_conf_facts.items()
                )
                if not matches_terminal and any(
                    sid not in term_set
                    and self._matches_fact_span(
                        predicted_answer, str(fact.get("text", ""))
                    )
                    for sid, fact in high_conf_facts.items()
                ):
                    predicted_answer = ""
                    extraction_method = "rejected_intermediate_hop"

        is_failed = (
            terminal_missing
            or not predicted_answer.strip()
            or predicted_answer.strip().lower() in {"unknown", "none", "null"}
        )
        if is_failed and not predicted_answer.strip():
            predicted_answer = "unknown"
        return (
            {
                "predicted_answer": predicted_answer,
                "is_failed": is_failed,
                "consensus_score": round(consensus_score, 4),
                "confidence": judge_confidence,
                "complexity": judge_complexity,
                "rationale": rationale,
                "conflict_resolution": conflict_resolution,
                "raw_judge_output": raw,
                "extraction_method": extraction_method,
            },
            latency,
            usage,
            cost,
        )
