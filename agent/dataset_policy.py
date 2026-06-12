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

from config.local.constants import EvidenceMode, POLICY, TOOLS


def default_allowed_tools(dataset: str) -> frozenset[str]:
    ds = (dataset or "").strip().lower()
    return POLICY.allowed.get(ds, frozenset())


def policy_warnings(policy: DatasetPolicy) -> list[str]:
    """
    Optional consistency checks (advisory only — does not mutate policy).

    Call after manual overrides when wiring experiments.
    """
    warnings: list[str] = []
    tools = {t.lower() for t in policy.allowed_tools}
    mode = policy.evidence_mode

    if mode == "provided_context" and tools & TOOLS.open_web:
        warnings.append(
            "evidence_mode=provided_context but open-web tools are allowed "
            f"({sorted(tools & TOOLS.open_web)})."
        )
    if mode == "provided_context" and policy.dataset in ("hotpot", "musique"):
        if not (tools & TOOLS.retrieve):
            warnings.append(
                "evidence_mode=provided_context for open-QA dataset but "
                "retrieve is not in allowed_tools."
            )
    if mode == "open_web" and tools & TOOLS.retrieve:
        warnings.append(
            "evidence_mode=open_web but retrieve is allowed (usually redundant)."
        )
    if mode == "internal" and tools & (TOOLS.open_web | TOOLS.retrieve):
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
        max_steps=POLICY.max_steps.get(ds),
        evidence_mode=POLICY.evidence.get(ds, "internal"),
    )
