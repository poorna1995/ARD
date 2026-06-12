"""Agent runtime kwargs and answer-validation helpers."""

from __future__ import annotations

BAD_ANSWERS = frozenset(
    {
        "",
        "none",
        "null",
        "n/a",
        "na",
        "unknown",
        "...",
        "<value>",
        "<short value>",
        "<answer>",
        "value",
        "answer",
        "final answer",
        "your answer",
        "your answer here",
        "insert answer",
        "tbd",
    }
)

HOP_KEYS = frozenset({"n_hops", "hop_name"})

RUNTIME_KWARGS = frozenset(
    {
        "expected_answer",
        "phase",
        "n_hops",
        "hop_name",
        "context",
        "evidence_mode",
        "allowed_tools",
        "policy",
    }
)

__all__ = ["BAD_ANSWERS", "HOP_KEYS", "RUNTIME_KWARGS"]
