"""
features/surface.py
===================
L1 — Surface Features

Extracts three length / richness signals from raw query text using only
classical NLP — no LLM inference:

  token_count  — raw word count (excluding punctuation and whitespace)
  ner_density  — fraction of tokens that belong to named-entity spans
  mattr        — Moving-Average Type-Token Ratio (window=25)

Combined into a single score S ∈ [0, 1]:

    S = 0.45 * scaled_token_count
      + 0.30 * scaled_ner_density
      + 0.25 * scaled_mattr
"""

from __future__ import annotations

from typing import Optional, Tuple

import spacy

from ...config import (
    MATTR_WINDOW,
    SURFACE_MATTR_HI,
    SURFACE_MATTR_LO,
    SURFACE_NER_HI,
    SURFACE_NER_LO,
    SURFACE_TC_HI,
    SURFACE_TC_LO,
    SURFACE_W_MATTR,
    SURFACE_W_NER,
    SURFACE_W_TC,
)
# from ...utils import clip_to_range, get_doc, min_max_scale

from src.utils.helpers import clip_to_range, get_doc, min_max_scale


class SurfaceFeatures:
    """
    Compute surface-level complexity signals for a single query.

    Parameters
    ----------
    text : str
        Raw query string.
    doc  : spacy.tokens.Doc, optional
        Pre-parsed spaCy Doc.  Pass this when you have already parsed the
        text to avoid a redundant NLP call (e.g. from the estimator loop).
    """

    def __init__(
        self,
        text: str,
        doc: Optional[spacy.tokens.Doc] = None,
    ) -> None:
        self.text = text
        self.doc  = doc if doc is not None else get_doc(text)

    # ── Sub-feature extractors ────────────────────────────────────────────────

    def _token_count(self) -> int:
        """Count meaningful tokens — skip punctuation and whitespace."""
        return sum(1 for t in self.doc if not t.is_punct and not t.is_space)

    def _ner_density(self) -> float:
        """
        Fraction of tokens belonging to a named-entity span.

        Example: "Barack Obama visited Paris in 2023"
          Named entities: [Barack Obama] [Paris] [2023] → 4 entity tokens / 6
          NER density = 4/6 ≈ 0.667

        A high density means the query targets specific real-world facts,
        making it harder to answer without factual grounding.
        """
        total              = max(len(self.doc), 1)
        entity_token_count = sum(len(ent) for ent in self.doc.ents)
        return round(entity_token_count / total, 4)

    def _mattr(self, window: int = MATTR_WINDOW) -> float:
        """
        Moving-Average Type-Token Ratio.

        Algorithm:
          1. Collect all lowercase non-punctuation tokens.
          2. Slide a window of *window* tokens across the sequence.
          3. At each position compute ``unique_tokens / window``.
          4. Average across all positions.

        Short texts (fewer tokens than *window*) fall back to plain TTR.

        Higher MATTR → richer vocabulary → more complex / specific query.
        """
        tokens = [t.lower_ for t in self.doc if not t.is_punct and not t.is_space]
        n      = len(tokens)
        if n == 0:
            return 0.0
        if n <= window:
            return round(len(set(tokens)) / n, 4)
        ttr_scores = [
            len(set(tokens[i : i + window])) / window
            for i in range(n - window + 1)
        ]
        return round(sum(ttr_scores) / len(ttr_scores), 4)

    # ── Public API ────────────────────────────────────────────────────────────

    def raw_features(self) -> dict:
        """Return all sub-features in their original (unscaled) values."""
        return {
            "token_count": self._token_count(),
            "ner_density": self._ner_density(),
            "mattr":       self._mattr(),
        }

    def score(self) -> Tuple[float, dict]:
        """
        Compute the surface complexity score S ∈ [0, 1].

        Returns
        -------
        (S, raw_features)
        """
        raw = self.raw_features()

        s_tc    = min_max_scale(raw["token_count"], SURFACE_TC_LO,    SURFACE_TC_HI)
        s_ner   = min_max_scale(raw["ner_density"], SURFACE_NER_LO,   SURFACE_NER_HI)
        s_mattr = min_max_scale(raw["mattr"],       SURFACE_MATTR_LO, SURFACE_MATTR_HI)

        S = SURFACE_W_TC * s_tc + SURFACE_W_NER * s_ner + SURFACE_W_MATTR * s_mattr
        return clip_to_range(S), raw