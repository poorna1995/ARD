"""
Dataset runtime policy — two independent axes.

**Evidence environment** (``evidence_mode``)
    Where factual evidence may come from for this episode.
    Does *not* imply which tools are callable.

**Tool permissions** (``allowed_tools``)
    Which ReAct tools the executor may invoke.
    Set explicitly per dataset or override; never derived from evidence_mode.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Literal

# Where evidence lives for this task (routing / prompt context shaping).
EvidenceMode = Literal["provided_context", "open_web", "internal"]

_OPEN_WEB_TOOLS = frozenset({"web_search", "web_fetch", "wikipedia_search"})
_RETRIEVE_TOOLS = frozenset({"retrieve"})

_DEFAULT_ALLOWED_TOOLS: dict[str, frozenset[str]] = {
    "math": frozenset({"math_tool"}),
    "hotpot": frozenset({"retrieve"}),
    "musique": frozenset({"retrieve"}),
    "gaia": frozenset({
        "read_file",
        "web_search",
        "web_fetch",
        "wikipedia_search",
        "python_exec",
        "arxiv_search",
        "github_search",
        "pdb_parse",
        "math_tool",
    }),
    "mmlu_pro": frozenset(),
}

_DEFAULT_MAX_STEPS: dict[str, int] = {
    "gaia": 14,
    "hotpot": 7,
    "musique": 8,
    "math": 8,
    "mmlu_pro": 6,
}

_DEFAULT_EVIDENCE_MODE: dict[str, EvidenceMode] = {
    "hotpot": "provided_context",
    "musique": "provided_context",
    "gaia": "open_web",
    "math": "internal",
    "mmlu_pro": "internal",
}


def default_allowed_tools(dataset: str) -> frozenset[str]:
    ds = (dataset or "").strip().lower()
    return _DEFAULT_ALLOWED_TOOLS.get(ds, frozenset())


def policy_warnings(policy: DatasetPolicy) -> list[str]:
    """
    Optional consistency checks (advisory only — does not mutate policy).

    Call after manual overrides when wiring experiments.
    """
    warnings: list[str] = []
    tools = {t.lower() for t in policy.allowed_tools}
    mode = policy.evidence_mode

    if mode == "provided_context" and tools & _OPEN_WEB_TOOLS:
        warnings.append(
            "evidence_mode=provided_context but open-web tools are allowed "
            f"({sorted(tools & _OPEN_WEB_TOOLS)})."
        )
    if mode == "provided_context" and policy.dataset in ("hotpot", "musique"):
        if not (tools & _RETRIEVE_TOOLS):
            warnings.append(
                "evidence_mode=provided_context for open-QA dataset but "
                "retrieve is not in allowed_tools."
            )
    if mode == "open_web" and tools & _RETRIEVE_TOOLS:
        warnings.append(
            "evidence_mode=open_web but retrieve is allowed (usually redundant)."
        )
    if mode == "internal" and tools & (_OPEN_WEB_TOOLS | _RETRIEVE_TOOLS):
        warnings.append(
            "evidence_mode=internal but evidence-fetching tools are allowed."
        )

    return warnings


@dataclass(frozen=True)
class DatasetPolicy:
    dataset: str
    allowed_tools: frozenset[str] = field(default_factory=frozenset)
    max_steps: int | None = None
    evidence_mode: EvidenceMode = "internal"

    def with_overrides(self, **kwargs: object) -> DatasetPolicy:
        return replace(self, **kwargs)


def default_dataset_policy(dataset: str) -> DatasetPolicy:
    ds = (dataset or "").strip().lower()
    return DatasetPolicy(
        dataset=ds,
        allowed_tools=default_allowed_tools(ds),
        max_steps=_DEFAULT_MAX_STEPS.get(ds),
        evidence_mode=_DEFAULT_EVIDENCE_MODE.get(ds, "internal"),
    )
