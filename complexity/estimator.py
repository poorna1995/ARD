"""
estimator.py
============
Model-Agnostic Task Complexity Estimator (TCE).

Public API
----------
ComplexityResult   — structured output for a single scored query
TaskComplexityEstimator
    .score(query)        → ComplexityResult
    .score_batch(queries) → pd.DataFrame

Composite formula:

    C(q) = w_S·S + w_R·R + w_T·T + w_D·D + w_TT·TT

All five sub-scores are computed entirely from classical NLP — zero LLM calls.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.config import DEFAULT_ESTIMATOR_WEIGHTS
from .features import (
    DomainSkills,
    ReasoningDepth,
    SurfaceFeatures,
    TaskType,
    ToolDependency,
)
from src.utils.helpers import assign_band, clip_to_range, get_doc


# ─────────────────────────────────────────────────────────────────────────────
# Result container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ComplexityResult:
    """
    Full result for one scored query.

    Attributes
    ----------
    query            : original query string
    C                : composite complexity score ∈ [0, 1]
    S, R, T, D, TT   : individual layer scores
    raw_*            : unscaled sub-features per layer (hidden from repr)
    complexity_band  : human-readable tier (LOW / MEDIUM / HIGH / VERY HIGH)
    """

    query:   str
    C:       float
    S:       float
    R:       float
    T:       float
    D:       float
    TT:      float
    raw_S:   dict = field(repr=False)
    raw_R:   dict = field(repr=False)
    raw_T:   dict = field(repr=False)
    raw_D:   dict = field(repr=False)
    raw_TT:  dict = field(repr=False)
    complexity_band: str = field(init=False)

    def __post_init__(self) -> None:
        self.complexity_band = assign_band(self.C)

    def summary(self) -> str:
        """One-line human-readable summary."""
        return (
            f"C={self.C:.3f} {self.complexity_band:<10}  "
            f"[S={self.S:.2f}  R={self.R:.2f}  T={self.T:.2f}  "
            f"D={self.D:.2f}  TT={self.TT:.2f}]"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Estimator
# ─────────────────────────────────────────────────────────────────────────────

class TaskComplexityEstimator:
    """
    Score a query (or a batch of queries) for task complexity.

    Parameters
    ----------
    weights : dict, optional
        Custom per-layer weights with keys ``'S', 'R', 'T', 'D', 'TT'``.
        Values are automatically re-normalised to sum to 1.0, so you can
        pass raw priorities rather than strict probabilities.

    Examples
    --------
    >>> tce = TaskComplexityEstimator()
    >>> result = tce.score("Design a distributed caching system for 100M DAU.")
    >>> print(result.summary())
    C=0.712 VERY HIGH   [S=0.43  R=0.80  T=0.50  D=0.25  TT=1.00]
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        raw_w = weights or DEFAULT_ESTIMATOR_WEIGHTS
        total = sum(raw_w.values())
        if total == 0:
            raise ValueError("All estimator weights are zero.")
        self.weights: Dict[str, float] = {k: v / total for k, v in raw_w.items()}

    # ── Core scoring ─────────────────────────────────────────────────────────

    def score(self, query: str) -> ComplexityResult:
        """
        Estimate the complexity of a single query string.

        The spaCy Doc is parsed once and shared between L1 (SurfaceFeatures)
        and L2 (ReasoningDepth), which both require it.  L3–L5 are regex-only
        and receive the raw text directly.

        Parameters
        ----------
        query : str

        Returns
        -------
        ComplexityResult
        """
        doc = get_doc(query)

        S,  raw_S  = SurfaceFeatures(query, doc=doc).score()
        R,  raw_R  = ReasoningDepth(query,  doc=doc).score()
        T,  raw_T  = ToolDependency(query).score()
        D,  raw_D  = DomainSkills(query).score()
        TT, raw_TT = TaskType(query).score()

        w = self.weights
        C = clip_to_range(
            w["S"] * S + w["R"] * R + w["T"] * T + w["D"] * D + w["TT"] * TT
        )

        return ComplexityResult(
            query=query,
            C=C, S=S, R=R, T=T, D=D, TT=TT,
            raw_S=raw_S, raw_R=raw_R, raw_T=raw_T, raw_D=raw_D, raw_TT=raw_TT,
        )

    # ── Batch scoring ─────────────────────────────────────────────────────────

    def score_batch(self, queries: List[str]) -> pd.DataFrame:
        """
        Score a list of queries and return a tidy ``pd.DataFrame``.

        Each row corresponds to one query.  The ``query`` column is
        truncated to 80 characters for readability.

        Parameters
        ----------
        queries : list of str

        Returns
        -------
        pd.DataFrame
        """
        records = []
        for q in queries:
            r = self.score(q)
            records.append({
                "query":           q[:80] + ("..." if len(q) > 80 else ""),
                "C":               round(r.C,  3),
                "S":               round(r.S,  3),
                "R":               round(r.R,  3),
                "T":               round(r.T,  3),
                "D":               round(r.D,  3),
                "TT":              round(r.TT, 3),
                "band":            r.complexity_band,
                "task_type":       r.raw_TT["primary_type"],
                "bloom_level":     r.raw_R["bloom_level"],
                "tool_categories": ", ".join(r.raw_T["matched_categories"]) or "none",
                "domains":         ", ".join(r.raw_D["matched_domains"]) or "none",
            })
        return pd.DataFrame(records)