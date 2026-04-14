# """
# features/tool.py
# ================
# L3 — Tool Dependency

# Identifies which external tool categories a task likely requires and
# returns T ∈ [0, 1].

#     T = min(category_count / MAX_CATS, 1.0)

#   1 category → T = 0.25   (light tool use)
#   2 categories → T = 0.50
#   3 categories → T = 0.75
#   4+ categories → T = 1.0  (complex agentic task)
# """

# from __future__ import annotations

# from typing import Tuple

# from ...config import TOOL_MAX_CATS, TOOL_PATTERNS
# from src.utils.helpers import clip_to_range


# class ToolDependency:
#     """
#     Compute tool-dependency complexity for a single query.

#     Parameters
#     ----------
#     text : str
#         Raw query string.
#     """

#     def __init__(self, text: str) -> None:
#         self.text = text

#     def raw_features(self) -> dict:
#         """
#         Scan for each tool-category pattern and record which ones fire.

#         Returns
#         -------
#         dict with keys:
#           matched_categories : list of category names that matched
#           category_count     : int — number of matched categories
#           category_flags     : dict[str, bool] — one flag per category
#         """
#         flags = {
#             cat: bool(pat.search(self.text))
#             for cat, pat in TOOL_PATTERNS.items()
#         }
#         matched = [cat for cat, hit in flags.items() if hit]
#         return {
#             "matched_categories": matched,
#             "category_count":     len(matched),
#             "category_flags":     flags,
#         }

#     def score(self) -> Tuple[float, dict]:
#         """
#         Compute the tool-dependency score T ∈ [0, 1].

#         Returns
#         -------
#         (T, raw_features)
#         """
#         raw = self.raw_features()
#         T   = clip_to_range(raw["category_count"] / TOOL_MAX_CATS)
#         return T, raw





"""
features/tool.py
================
L3 — Tool Dependency

Identifies which external tool categories a task likely requires and
returns T ∈ [0, 1].

Primary path:
  - regex category detection

Fallback path:
  - sentence-embedding similarity against lightweight category exemplars
    when regex finds no matches

    T = min(category_count / MAX_CATS, 1.0)

  1 category → T = 0.25   (light tool use)
  2 categories → T = 0.50
  3 categories → T = 0.75
  4+ categories → T = 1.0  (complex agentic task)
"""

from __future__ import annotations

from math import sqrt
from typing import Dict, List, Optional, Tuple

from ...config import TOOL_MAX_CATS, TOOL_PATTERNS
from src.utils.helpers import clip_to_range

try:
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    SentenceTransformer = None


class ToolDependency:
    """
    Compute tool-dependency complexity for a single query.

    Parameters
    ----------
    text : str
        Raw query string.
    """

    # Low-cost, conservative threshold for embedding fallback.
    EMBEDDING_THRESHOLD: float = 0.55

    # Cached embedding model instance.
    _embedder = None

    def __init__(self, text: str) -> None:
        self.text = text

    @staticmethod
    def _normalize_category_name(category: str) -> str:
        """
        Convert a category key into a short natural-language phrase.
        """
        return category.replace("_", " ").replace("-", " ").strip()

    @classmethod
    def _get_embedder(cls):
        """
        Lazily load the sentence-transformers model once.

        Returns None if sentence-transformers is unavailable or loading fails.
        """
        if cls._embedder is not None:
            return cls._embedder

        if SentenceTransformer is None:
            return None

        try:
            cls._embedder = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            cls._embedder = None

        return cls._embedder

    @staticmethod
    def _dot(a, b) -> float:
        return float(sum(x * y for x, y in zip(a, b)))

    @classmethod
    def _category_exemplar(cls, category: str) -> str:
        """
        Lightweight exemplar sentence for a category.

        The fallback intentionally stays generic and cheap; it does not invent
        new tool categories.
        """
        pretty = cls._normalize_category_name(category)
        return (
            f"This task likely needs {pretty}. "
            f"Use {pretty} tooling when the answer depends on external tools."
        )

    def _embedding_fallback(self, categories: List[str]) -> Dict[str, object]:
        """
        Infer likely categories using embedding similarity.

        Returns a dict with:
          matched_categories : list[str]
          embedding_scores   : dict[str, float]
          used_embedding     : bool
        """
        model = self._get_embedder()
        if model is None or not categories:
            return {
                "matched_categories": [],
                "embedding_scores": {},
                "used_embedding": False,
            }

        exemplars = [self._category_exemplar(cat) for cat in categories]

        try:
            vectors = model.encode(
                [self.text, *exemplars],
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception:
            return {
                "matched_categories": [],
                "embedding_scores": {},
                "used_embedding": False,
            }

        query_vec = vectors[0]
        exemplar_vecs = vectors[1:]

        scores = {
            cat: self._dot(query_vec, vec)
            for cat, vec in zip(categories, exemplar_vecs)
        }

        matched = [
            cat
            for cat, score in scores.items()
            if score >= self.EMBEDDING_THRESHOLD
        ]

        return {
            "matched_categories": matched,
            "embedding_scores": scores,
            "used_embedding": True,
        }

    def raw_features(self) -> dict:
        """
        Scan for each tool-category pattern and record which ones fire.

        Returns
        -------
        dict with keys:
          matched_categories       : list[str] of category names that matched
          category_count           : int — number of matched categories
          category_flags           : dict[str, bool] — one flag per category
          regex_matched_categories : list[str]
          regex_category_flags     : dict[str, bool]
          embedding_matched_categories : list[str]
          embedding_scores         : dict[str, float]
          used_embedding_fallback   : bool
        """
        categories = list(TOOL_PATTERNS.keys())

        regex_flags = {
            cat: bool(pat.search(self.text))
            for cat, pat in TOOL_PATTERNS.items()
        }
        regex_matched = [cat for cat, hit in regex_flags.items() if hit]

        embedding_matched: List[str] = []
        embedding_scores: Dict[str, float] = {}
        used_embedding_fallback = False

        # Fallback only when regex finds nothing.
        if not regex_matched:
            fb = self._embedding_fallback(categories)
            embedding_matched = fb["matched_categories"]
            embedding_scores = fb["embedding_scores"]
            used_embedding_fallback = bool(fb["used_embedding"])

        matched = regex_matched if regex_matched else embedding_matched

        combined_flags = {cat: cat in matched for cat in categories}

        return {
            "matched_categories": matched,
            "category_count": len(matched),
            "category_flags": combined_flags,
            "regex_matched_categories": regex_matched,
            "regex_category_flags": regex_flags,
            "embedding_matched_categories": embedding_matched,
            "embedding_scores": embedding_scores,
            "used_embedding_fallback": used_embedding_fallback,
        }

    def score(self) -> Tuple[float, dict]:
        """
        Compute the tool-dependency score T ∈ [0, 1].

        Returns
        -------
        (T, raw_features)
        """
        raw = self.raw_features()
        T = clip_to_range(raw["category_count"] / TOOL_MAX_CATS)
        return T, raw