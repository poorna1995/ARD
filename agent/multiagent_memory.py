from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class WorkingMemory:
    goal: str = ""
    known_facts: dict[str, dict] = field(default_factory=dict)
    open_questions: list[str] = field(default_factory=list)
    tool_summaries: dict[str, str] = field(default_factory=dict)
    final_constraint: str = ""

    # Per-fact char cap — set from dataset profile on each run (see multiagent.run).
    _FACT_CHARS: int = 160
    _SUMMARY_CHARS: int = 80

    def record_fact(
        self, subtask_id: str, result: str, confidence: float, tool_backed: bool = False
    ) -> None:
        existing = self.known_facts.get(subtask_id)
        if existing is None or confidence >= existing.get("confidence", 0):
            self.known_facts[subtask_id] = {
                "text": result[: self._FACT_CHARS],
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
            facts = {k: v["text"] for k, v in self.known_facts.items() if k in dep_ids}
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

