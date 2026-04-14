"""
features/reasoning.py
=====================
L2 — Reasoning Depth

Measures cognitive and structural complexity via six sub-features:

  bloom_level      (1–6)   — Bloom's Taxonomy cognitive operation demanded
  syntactic_depth          — mean max dependency-tree depth per sentence
  multi_hop        (0/1)   — does the query chain across ≥2 entity types?
  negation_count           — number of negation arcs ("not", "never", …)
  conditional_count        — number of conditional/causal conjunctions
  modifier_density         — ratio of adjective+adverb modifiers to tokens

Combined into R ∈ [0, 1].
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import spacy

from ...config import (
    BLOOM_PATTERNS,
    CONDITIONAL_TOKENS,
    REASONING_COND_HI,
    REASONING_COND_LO,
    REASONING_DEPTH_HI,
    REASONING_DEPTH_LO,
    REASONING_MOD_HI,
    REASONING_MOD_LO,
    REASONING_NEG_HI,
    REASONING_NEG_LO,
    REASONING_W,
)
from src.utils.helpers import clip_to_range, get_doc, min_max_scale, tree_depth


class ReasoningDepth:
    """
    Compute reasoning-depth complexity signals for a single query.

    Parameters
    ----------
    text : str
        Raw query string.
    doc  : spacy.tokens.Doc, optional
        Pre-parsed spaCy Doc (reuse from L1 to skip duplicate parsing).
    """

    def __init__(
        self,
        text: str,
        doc: Optional[spacy.tokens.Doc] = None,
    ) -> None:
        self.text = text
        self.doc  = doc if doc is not None else get_doc(text)

    # ── Sub-feature extractors ────────────────────────────────────────────────

    def _bloom_level(self) -> int:
        """
        Return the HIGHEST Bloom level (1–6) matched anywhere in the text.

        Levels are checked from 6 down to 1; the first match wins.
        Default is 1 (Remember) when no verb matches.
        """
        for lvl in range(6, 0, -1):
            if BLOOM_PATTERNS[lvl].search(self.text):
                return lvl
        return 1

    def _syntactic_depth(self) -> float:
        """
        Mean maximum dependency-tree depth over all sentences.

        Deeper trees indicate more embedded clause structures
        (relative clauses, complement clauses, etc.) and harder grammar.
        """
        depths = [tree_depth(sent.root) for sent in self.doc.sents]
        return round(float(np.mean(depths)) if depths else 0.0, 4)

    def _is_multi_hop(self) -> bool:
        """
        Heuristic: query is multi-hop if it has ≥2 distinct entity TYPES
        AND at least one relational / comparison token.

        We check entity *types* (e.g. GPE, ORG, DATE), not entity names,
        so "France vs Germany" counts even though both are GPE.
        """
        entity_types = {ent.label_ for ent in self.doc.ents}
        if len(entity_types) < 2:
            return False
        relational = {
            "and", "between", "both", "versus", "vs", "compared",
            "relate", "link", "connect", "differ", "same", "than",
        }
        query_tokens = {t.lower_ for t in self.doc}
        return bool(query_tokens & relational)

    def _negation_count(self) -> int:
        """Count tokens labelled with a 'neg' dependency arc (not, never, no)."""
        return sum(1 for t in self.doc if t.dep_ == "neg")

    def _conditional_count(self) -> int:
        """Count how many conditional/causal conjunctions appear in the query."""
        tokens_lower = {t.lower_ for t in self.doc}
        return len(tokens_lower & CONDITIONAL_TOKENS)

    def _modifier_density(self) -> float:
        """
        Ratio of (adjective + adverb modifiers) to total tokens.

        amod  = adjectival modifier  ("large" in "large model")
        advmod = adverbial modifier  ("quickly" in "runs quickly")

        Denser modifiers → more constrained / nuanced query.
        """
        total = max(len(self.doc), 1)
        mods  = sum(1 for t in self.doc if t.dep_ in {"amod", "advmod"})
        return round(mods / total, 4)

    # ── Public API ────────────────────────────────────────────────────────────

    def raw_features(self) -> dict:
        """Return all sub-features in their original (unscaled) values."""
        return {
            "bloom_level":       self._bloom_level(),
            "syntactic_depth":   self._syntactic_depth(),
            "multi_hop":         int(self._is_multi_hop()),
            "negation_count":    self._negation_count(),
            "conditional_count": self._conditional_count(),
            "modifier_density":  self._modifier_density(),
        }

    def score(self) -> Tuple[float, dict]:
        """
        Compute the reasoning depth score R ∈ [0, 1].

        Returns
        -------
        (R, raw_features)
        """
        raw = self.raw_features()

        # Bloom levels 1–6: subtract 1, divide by 5 → [0, 1]
        s_bloom = (raw["bloom_level"] - 1) / 5.0
        s_depth = min_max_scale(raw["syntactic_depth"],   REASONING_DEPTH_LO, REASONING_DEPTH_HI)
        s_hop   = float(raw["multi_hop"])                 # already 0 or 1
        s_neg   = min_max_scale(raw["negation_count"],    REASONING_NEG_LO,   REASONING_NEG_HI)
        s_cond  = min_max_scale(raw["conditional_count"], REASONING_COND_LO,  REASONING_COND_HI)
        s_mod   = min_max_scale(raw["modifier_density"],  REASONING_MOD_LO,   REASONING_MOD_HI)

        W = REASONING_W
        R = (
            W["bloom"]     * s_bloom
            + W["syn_depth"] * s_depth
            + W["multi_hop"] * s_hop
            + W["negation"]  * s_neg
            + W["condition"] * s_cond
            + W["modifier"]  * s_mod
        )
        return clip_to_range(R), raw




