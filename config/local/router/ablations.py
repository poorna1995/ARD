"""Feature ablation cases for router experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AblationCase:
    label: str
    feature_set: str
    description: str = ""


_SHARED_ABLATIONS: tuple[tuple[str, str, str], ...] = (
    ("emb-only", "emb_only", "PCA-16 embedding only"),
    ("dataset-only", "dataset_only", "dataset one-hot only"),
)

CVEC5_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase("cvec5+emb+dataset", "cvec5_emb_ds", "5-dim C(Q) + emb + dataset"),
    AblationCase("cvec5+emb", "cvec5_emb", "5-dim C(Q) + emb (production features)"),
    AblationCase("cvec5+dataset", "cvec5_ds", "5-dim C(Q) + dataset"),
    AblationCase("cvec5-only", "cvec5", "5-dim C(Q) only"),
    AblationCase("heur+emb", "heur_emb", "heuristic v2 + PCA-16 (no decompose)"),
    *(AblationCase(*row) for row in _SHARED_ABLATIONS),
)

TRUST_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase(
        "cvec5+emb (baseline)",
        "cvec5_emb",
        "production feature columns",
    ),
    AblationCase(
        "cvec5+emb+trust",
        "cvec5_emb_trust",
        "+ plan-trust scalars",
    ),
)

__all__ = [
    "AblationCase",
    "CVEC5_ABLATION_CASES",
    "TRUST_ABLATION_CASES",
]
