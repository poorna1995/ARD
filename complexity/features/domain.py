"""
features/domain.py
==================
L4 — Domain Skills

Measures cross-domain breadth and temporal reasoning:

  domain_count     — number of distinct knowledge domains hit
  temporal_signals — number of distinct temporal signal patterns matched

D ∈ [0, 1]:

    D = 0.65 * min(domain_count / 4, 1.0)
      + 0.35 * min(temporal_signals / 3, 1.0)
"""

from __future__ import annotations

from typing import Tuple

from ...config import (
    DOMAIN_MAX_CATS,
    DOMAIN_MAX_TEMP,
    DOMAIN_PATTERNS,
    DOMAIN_W_DOMAIN,
    DOMAIN_W_TEMPORAL,
    TEMPORAL_PATTERNS,
    
)
from src.utils.helpers import clip_to_range, min_max_scale


class DomainSkills:
    """
    Compute domain-breadth and temporal-reasoning signals for a single query.

    Parameters
    ----------
    text : str
        Raw query string (no spaCy Doc needed — regex only).
    """

    def __init__(self, text: str) -> None:
        self.text = text

    def raw_features(self) -> dict:
        """
        Identify which knowledge domains and temporal patterns fire.

        Returns
        -------
        dict with keys:
          matched_domains  : sorted list of domain names that matched
          domain_count     : int
          temporal_signals : int — number of distinct temporal patterns matched
        """
        domains_hit  = {d for d, pat in DOMAIN_PATTERNS.items() if pat.search(self.text)}
        temp_signals = sum(1 for pat in TEMPORAL_PATTERNS if pat.search(self.text))
        return {
            "matched_domains":  sorted(domains_hit),
            "domain_count":     len(domains_hit),
            "temporal_signals": temp_signals,
        }

    def score(self) -> Tuple[float, dict]:
        """
        Compute the domain-skills score D ∈ [0, 1].

        Returns
        -------
        (D, raw_features)
        """
        raw        = self.raw_features()
        s_domain   = min_max_scale(raw["domain_count"],     0, DOMAIN_MAX_CATS)
        s_temporal = min_max_scale(raw["temporal_signals"], 0, DOMAIN_MAX_TEMP)
        D          = DOMAIN_W_DOMAIN * s_domain + DOMAIN_W_TEMPORAL * s_temporal
        return clip_to_range(D), raw