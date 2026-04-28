# from __future__ import annotations

# import logging
# import re
# from collections import deque
# from dataclasses import dataclass, field
# from typing import Any, Iterable, Sequence

# import numpy as np

# logger = logging.getLogger(__name__)


# # =========================================================
# # DATA CLASS
# # =========================================================

# @dataclass
# class QueryComplexityScore:
#     """
#     Holds every complexity score for a single query.

#     Attributes
#     ----------
#     overall          : Weighted aggregate score, 0–10.
#     task_length      : Vocabulary richness + length dimension, 0–10.
#     num_requirements : Requirement-count dimension, 0–10.
#     domain_breadth   : Subject-domain coverage dimension, 0–10.
#     tool_requirements: Tool-invocation demand dimension, 0–10.
#     reasoning_depth  : Bloom level + multi-hop dimension, 0–10.
#     task_type        : Primary task-verb complexity, 0–10.
#     complexity_band  : Routing label – "LLM" | "CoT" | "SA" | "MA".
#     breakdown        : Raw feature values used to produce the scores.
#     """

#     overall:           float
#     task_length:       float
#     num_requirements:  float
#     domain_breadth:    float
#     tool_requirements: float
#     reasoning_depth:   float
#     task_type:         float
#     complexity_band:   str
#     breakdown:         dict[str, Any] = field(default_factory=dict)


# # =========================================================
# # ANALYZER
# # =========================================================

# class QueryComplexityAnalyzer:
#     """
#     Scores the complexity of a natural-language query across six
#     weighted dimensions and maps the result to a routing band.

#     Default static band boundaries (before dataset fitting)
#     --------------------------------------------------------
#     LLM  – simple, direct answer       score  < 2
#     CoT  – needs chain-of-thought      score  < 5
#     SA   – needs search / agents       score  < 7
#     MA   – multi-agent / very hard     score >= 7

#     After calling ``fit_thresholds`` or ``analyze_dataset`` the
#     boundaries are replaced by empirical percentiles from real data.
#     """

#     DEFAULT_WEIGHTS: dict[str, float] = {
#         "task_length":       0.10,
#         "num_requirements":  0.18,
#         "domain_breadth":    0.18,
#         "tool_requirements": 0.20,
#         "reasoning_depth":   0.22,
#         "task_type":         0.12,
#     }

#     # ------------------------------------------------------------------
#     # Bloom's taxonomy – ordered highest → lowest so _get_bloom_level()
#     # can return the FIRST (= most demanding) match.
#     # ------------------------------------------------------------------
#     BLOOM_KEYWORDS: list[tuple[str, str]] = [
#         ("L6_create",     r"\b(create|design|build|compose|formulate|invent|devise)\b"),
#         ("L5_evaluate",   r"\b(evaluate|assess|judge|critique|recommend|justify|defend)\b"),
#         ("L4_analyze",    r"\b(analyze|analyse|compare|contrast|differentiate|examine)\b"),
#         ("L3_apply",      r"\b(apply|use|implement|demonstrate|calculate|solve|execute)\b"),
#         ("L2_understand", r"\b(explain|describe|summarize|interpret|classify|paraphrase)\b"),
#         ("L1_remember",   r"\b(list|recall|define|identify|name|state|who|what|when|where)\b"),
#     ]

#     # Score (0–10) assigned to each Bloom level.
#     BLOOM_SCORES: dict[str, float] = {
#         "L1_remember":  1.0,
#         "L2_understand":3.0,
#         "L3_apply":     5.0,
#         "L4_analyze":   7.0,
#         "L5_evaluate":  8.5,
#         "L6_create":   10.0,
#     }

#     # Domain keywords that indicate subject-area breadth.
#     DOMAIN_KEYWORDS: list[str] = [
#         "code", "data", "finance", "legal", "research", "science",
#         "medical", "statistics", "machine learning", "database",
#     ]

#     # Keywords suggesting a tool or computation must be invoked.
#     TOOL_KEYWORDS: list[str] = [
#         "calculate", "plot", "search", "api", "sql", "graph",
#         "visualize", "query", "fetch", "simulate",
#     ]

#     # Multi-hop patterns ordered highest complexity → lowest.
#     # Each entry is (compiled pattern, score).
#     MULTI_HOP_PATTERNS: list[tuple[re.Pattern[str], float]] = [
#         (re.compile(r"\bthen\b.+\bafter\b|\bafter\b.+\bthen\b",    re.I | re.S), 9.0),
#         (re.compile(r"\bif\b.+\bthen\b",                            re.I | re.S), 7.5),
#         (re.compile(r"\bgiven that\b|\bassuming\b|\bprovided that\b",re.I),        6.0),
#         (re.compile(r"\bif\b",                                       re.I),        5.0),
#         (re.compile(r"\bfirst\b.+\bthen\b|\bstep \d\b",             re.I | re.S), 4.0),
#     ]

#     # Task-type keywords ordered highest complexity → lowest.
#     # Each entry is (keyword, score). Checked in order so more specific
#     # terms (e.g. "compare") win over broad ones (e.g. "what").
#     TASK_TYPE_SCORES: list[tuple[str, float]] = [
#         ("compare",   8.0),
#         ("evaluate",  7.5),
#         ("analyze",   7.0),
#         ("generate",  6.5),
#         ("design",    6.5),
#         ("write",     5.5),
#         ("summarize", 4.5),
#         ("list",      3.5),
#         ("what",      3.0),
#         ("who",       2.5),
#         ("when",      2.0),
#         ("where",     2.0),
#     ]

#     # =========================================================
#     # CONSTRUCTION
#     # =========================================================

#     def __init__(
#         self,
#         weights: dict[str, float] | None = None,
#         threshold_percentiles: tuple[int, int, int] = (30, 70, 90),
#         mattr_window: int = 25,
#     ) -> None:
#         """
#         Parameters
#         ----------
#         weights
#             Per-dimension weights; must contain the same keys as
#             ``DEFAULT_WEIGHTS``.  If *None*, defaults are used.
#         threshold_percentiles
#             Three strictly-increasing integers in (0, 100) that define the
#             LLM / CoT / SA band boundaries after dataset fitting.
#         mattr_window
#             Sliding-window width for the MATTR computation.  Clamped to
#             at least 1.
#         """
#         self._validate_percentiles(threshold_percentiles)

#         self.threshold_percentiles = threshold_percentiles
#         self.mattr_window          = max(1, mattr_window)
#         self.weights               = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
#         self.thresholds: dict[str, float] | None = None

#         if weights is not None:
#             self._warn_if_weights_off(weights)

#     # =========================================================
#     # PUBLIC API
#     # =========================================================
#     def get_user_percentiles() -> tuple[int, int, int]:
#         """Prompt the user for three band-boundary percentiles and validate them."""
#         print("=" * 60)
#         print("  Query Complexity Analyzer")
#         print("=" * 60)
#         print()
#         print("Enter three percentiles that define the band boundaries.")
#         print("  p_low  → scores below this percentile are classified LLM")
#         print("  p_mid  → scores below this percentile are classified CoT")
#         print("  p_high → scores below this percentile are classified SA")
#         print("           (everything above p_high is classified MA)")
#         print()
#         print("Values must be integers, strictly increasing, and in (0, 100).")
#         print("Example: 30  70  90")
#         print()

#         while True:
#             raw = input("Enter p_low  p_mid  p_high: ").strip().split()

#             if len(raw) != 3:
#                 print(f"  ✗ Expected 3 values, got {len(raw)}. Try again.\n")
#                 continue

#             try:
#                 p_low, p_mid, p_high = int(raw[0]), int(raw[1]), int(raw[2])
#             except ValueError:
#                 print("  ✗ All three values must be whole numbers. Try again.\n")
#                 continue

#             if not (0 < p_low < p_mid < p_high < 100):
#                 print("  ✗ Values must be strictly increasing and between 0 and 100. Try again.\n")
#                 continue

#             print(f"\n  ✓ Using percentiles: LLM < p{p_low}  |  CoT < p{p_mid}  |  SA < p{p_high}  |  MA")
#             print()
#             return p_low, p_mid, p_high

#     def analyze(self, task: str) -> QueryComplexityScore:
#         """Score a single query and return a :class:`QueryComplexityScore`."""

#         task_lower = task.lower()

#         # ---- raw feature extraction ----
#         word_count   = len(task.split())
#         mattr        = self._compute_mattr(task)
#         requirements = self._count_requirements(task)
#         domain_hits  = self._count_keyword_matches(task_lower, self.DOMAIN_KEYWORDS)
#         tool_hits    = self._count_keyword_matches(task_lower, self.TOOL_KEYWORDS)
#         bloom_level  = self._get_bloom_level(task_lower)
#         multi_hop    = self._multi_hop_score(task_lower)
#         task_type    = self._task_type_score(task_lower)

#         # ---- dimension scores (all on 0–10 scale) ----
#         s_length    = 0.6 * self._score_word_count(word_count) + 0.4 * self._score_mattr(mattr)
#         s_req       = self._score_requirements(requirements)
#         s_domain    = min(10.0, domain_hits * 2.0)
#         s_tools     = min(10.0, tool_hits   * 2.5)
#         s_reasoning = self.BLOOM_SCORES[bloom_level] * 0.6 + multi_hop * 0.4

#         # ---- weighted overall (capped at 10) ----
#         overall = min(10.0, (
#             self.weights["task_length"]       * s_length    +
#             self.weights["num_requirements"]  * s_req       +
#             self.weights["domain_breadth"]    * s_domain    +
#             self.weights["tool_requirements"] * s_tools     +
#             self.weights["reasoning_depth"]   * s_reasoning +
#             self.weights["task_type"]         * task_type
#         ))

#         band = self._classify(overall)

#         return QueryComplexityScore(
#             overall           = round(overall,     4),
#             task_length       = round(s_length,    4),
#             num_requirements  = round(s_req,       4),
#             domain_breadth    = round(s_domain,    4),
#             tool_requirements = round(s_tools,     4),
#             reasoning_depth   = round(s_reasoning, 4),
#             task_type         = round(task_type,   4),
#             complexity_band   = band,
#             breakdown={                                # FIX: was always empty {}
#                 "word_count":   word_count,
#                 "mattr":        round(mattr, 4),
#                 "requirements": requirements,
#                 "domain_hits":  domain_hits,
#                 "tool_hits":    tool_hits,
#                 "bloom_level":  bloom_level,
#                 "multi_hop":    round(multi_hop, 2),
#             },
#         )

#     def analyze_dataset(
#         self, queries: Sequence[str]
#     ) -> list[dict[str, Any]]:
#         """
#         Score every query, fit percentile thresholds from the results,
#         then re-classify each query using the fitted thresholds.

#         Returns a list of ``{"query", "score", "band"}`` dicts.

#         .. note::
#             The :attr:`~QueryComplexityScore.complexity_band` stored inside
#             each :class:`QueryComplexityScore` uses static defaults because
#             thresholds are unknown at analysis time.  The ``"band"`` key in
#             the returned dicts reflects the **fitted** thresholds and is
#             therefore the authoritative label.
#         """
#         scores = [self.analyze(q) for q in queries]
#         self.fit_thresholds(scores)

#         return [
#             {
#                 "query": q,
#                 "score": s.overall,
#                 "band":  self._classify(s.overall),  # FIX: re-classify with fitted thresholds
#             }
#             for q, s in zip(queries, scores)
#         ]

#     # =========================================================
#     # THRESHOLD FITTING & CLASSIFICATION
#     # =========================================================

#     def fit_thresholds(self, scores: Iterable[QueryComplexityScore]) -> None:
#         """
#         Derive band boundaries from the empirical distribution of scores.

#         Call this once you have a representative corpus so that the
#         static defaults are replaced by data-driven cut points.

#         Raises
#         ------
#         ValueError
#             If *scores* is empty.
#         """
#         values = np.array([s.overall for s in scores], dtype=float)

#         if values.size == 0:
#             raise ValueError("Cannot fit thresholds: score list is empty.")

#         p_low, p_mid, p_high = self.threshold_percentiles
#         self.thresholds = {
#             "LLM": float(np.percentile(values, p_low)),
#             "CoT": float(np.percentile(values, p_mid)),
#             "SA":  float(np.percentile(values, p_high)),
#         }
#         logger.debug("Fitted band thresholds: %s", self.thresholds)

#     def classify_score(self, score: float) -> str:
#         """Public convenience wrapper around :meth:`_classify`."""
#         return self._classify(score)

#     def _classify(self, score: float) -> str:
#         """Map a numeric score to a routing band label."""
#         t = self.thresholds

#         if t is None:
#             # Static defaults used before dataset fitting.
#             if score < 2: return "LLM"
#             if score < 5: return "CoT"
#             if score < 7: return "SA"
#             return "MA"

#         if score < t["LLM"]: return "LLM"
#         if score < t["CoT"]: return "CoT"
#         if score < t["SA"]:  return "SA"
#         return "MA"

#     # =========================================================
#     # FEATURE EXTRACTION
#     # =========================================================

#     def _compute_mattr(self, text: str) -> float:
#         """
#         Moving Average Type-Token Ratio (MATTR).

#         Slides a window of width ``mattr_window`` across the token list
#         and averages the TTR of each window position.  This gives a
#         length-insensitive measure of vocabulary richness.

#         FIX: the original implementation computed plain TTR
#         (``len(set) / len(words)``) and never used ``mattr_window``.
#         """
#         words = re.findall(r"\b[a-z]+\b", text.lower())
#         n     = len(words)

#         if n == 0:
#             return 0.0

#         w = min(self.mattr_window, n)          # window cannot exceed text length

#         # --- initialise the first window ---
#         window: deque[str]   = deque(words[:w])
#         freq:   dict[str, int] = {}
#         for word in window:
#             freq[word] = freq.get(word, 0) + 1

#         ttr_sum   = len(freq) / w              # TTR of the first window
#         n_windows = 1

#         # --- slide the window across remaining tokens ---
#         for i in range(w, n):
#             # remove leftmost token
#             outgoing = words[i - w]
#             freq[outgoing] -= 1
#             if freq[outgoing] == 0:
#                 del freq[outgoing]

#             # add new rightmost token
#             incoming = words[i]
#             freq[incoming] = freq.get(incoming, 0) + 1

#             ttr_sum   += len(freq) / w
#             n_windows += 1

#         return ttr_sum / n_windows

#     def _count_requirements(self, task: str) -> int:
#         """
#         Count explicit requirement signals in *task*.

#         Signals counted
#         ---------------
#         * Question marks       – each "?" is a distinct sub-question.
#         * Conjunctions         – "and", "also", "moreover", "additionally".
#         * Enumeration markers  – "first", "second", ordinals, "finally", etc.

#         FIX: the original used ``max(1, ...)`` which forced a minimum of 1
#         even for trivially simple queries, inflating ``_score_requirements``
#         for every input.  The floor is removed; 0 is now a valid return.
#         """
#         question_marks = task.count("?")
#         conjunctions   = len(re.findall(
#             r"\b(and|also|moreover|additionally|furthermore)\b", task, re.I
#         ))
#         enum_markers   = len(re.findall(
#             r"\b(first|second|third|fourth|finally|lastly|\d+[.)]\s)", task, re.I
#         ))
#         return question_marks + conjunctions + enum_markers

#     def _count_keyword_matches(self, text: str, keywords: list[str]) -> int:
#         """Return how many keywords from *keywords* appear in *text*."""
#         return sum(1 for kw in keywords if kw in text)

#     def _get_bloom_level(self, text: str) -> str:
#         """
#         Return the highest Bloom's taxonomy level triggered by *text*.

#         FIX: the original used an if/elif chain where the order of the
#         conditions determined the result (e.g. "evaluate" was checked
#         before "create"), so a query containing "create" could be
#         mis-classified as L5.  We now iterate from L6 → L1 explicitly
#         and return on the first (highest) match.
#         """
#         for level, pattern in self.BLOOM_KEYWORDS:
#             if re.search(pattern, text):
#                 return level
#         return "L1_remember"

#     def _multi_hop_score(self, text: str) -> float:
#         """
#         Score the degree of conditional / multi-step reasoning required.

#         Patterns are checked highest-complexity first; the score of the
#         first matching pattern is returned.

#         FIX: the original used plain ``"then" in text and "after" in text``
#         substring checks that could fire on unrelated sentences, and gave
#         a disproportionately high score (6) for any use of "if" alone.
#         Patterns are now compiled regexes, reducing false positives.
#         """
#         for pattern, score in self.MULTI_HOP_PATTERNS:
#             if pattern.search(text):
#                 return score
#         return 2.0      # baseline: no multi-hop structure detected

#     def _task_type_score(self, text: str) -> float:
#         """
#         Assign a complexity score based on the primary task verb or
#         question word found in *text*.

#         FIX: the original checked "what" before "compare", so a query
#         like "what should I compare …" returned 3 instead of 8.
#         Keywords are now ordered highest → lowest complexity.
#         """
#         for keyword, score in self.TASK_TYPE_SCORES:
#             if keyword in text:
#                 return score
#         return 5.0      # neutral default when no keyword matches

#     # =========================================================
#     # SCORING HELPERS  (raw feature → 0–10 dimension score)
#     # =========================================================

#     def _score_word_count(self, wc: int) -> float:
#         """
#         Piecewise mapping: word count → 0–10.

#         FIX: the original jumped straight from 2 at <20 words to 4 at
#         <50 words with no differentiation for very short queries (1–9 words
#         vs 10–19 words both returned 2).
#         """
#         if wc <  10: return 1.0
#         if wc <  20: return 2.5
#         if wc <  50: return 4.0
#         if wc < 100: return 6.0
#         if wc < 200: return 8.0
#         return 10.0

#     def _score_mattr(self, m: float) -> float:
#         """MATTR value (0–1) → 0–10 linear scale."""
#         return min(10.0, m * 10.0)

#     def _score_requirements(self, r: int) -> float:
#         """
#         Map requirement count to 0–10.

#         FIX: 0 requirements now correctly scores 0 instead of being
#         lifted to 2 by the removed ``max(1, ...)`` floor in
#         ``_count_requirements``.
#         """
#         return min(10.0, r * 2.0)

#     # =========================================================
#     # VALIDATION HELPERS
#     # =========================================================

#     @staticmethod
#     def _validate_percentiles(percentiles: tuple[int, int, int]) -> None:
#         if len(percentiles) != 3:
#             raise ValueError(
#                 "threshold_percentiles must contain exactly 3 values; "
#                 f"got {len(percentiles)}."
#             )
#         if not (0 < percentiles[0] < percentiles[1] < percentiles[2] < 100):
#             raise ValueError(
#                 "Percentiles must be strictly increasing and lie within (0, 100). "
#                 f"Got: {percentiles}."
#             )

#     @staticmethod
#     def _warn_if_weights_off(weights: dict[str, float]) -> None:
#         total = sum(weights.values())
#         if abs(total - 1.0) > 1e-6:
#             logger.warning(
#                 "Custom weights sum to %.6f instead of 1.0. "
#                 "Overall scores will be scaled accordingly.",
#                 total,
#             )
   
# import pandas as pd
# import logging

# logging.basicConfig(level=logging.INFO)
# if __name__ == "__main__":
#     percentiles = get_user_percentiles()
#     analyzer    = QueryComplexityAnalyzer(threshold_percentiles=percentiles)

#     df      = pd.read_parquet("../datasets/processed/gaia.parquet")
#     queries = df["query"].tolist()

#     # Score every query, then fit thresholds from the corpus distribution.
#     scores = [analyzer.analyze(q) for q in queries]
#     analyzer.fit_thresholds(scores)

#     # Show the fitted numeric boundaries derived from the chosen percentiles.
#     t = analyzer.thresholds
#     print("Fitted band boundaries")
#     print(f"  LLM  : score < {t['LLM']:.4f}   (p{percentiles[0]})")
#     print(f"  CoT  : score < {t['CoT']:.4f}   (p{percentiles[1]})")
#     print(f"  SA   : score < {t['SA']:.4f}   (p{percentiles[2]})")
#     print(f"  MA   : score >= {t['SA']:.4f}")
#     print()
#     print("-" * 60)

#     for query, score in zip(queries, scores):
#         band = analyzer.classify_score(score.overall)

#         print(f"{score.overall:.2f}  {band:<4}  {query}")
#         print(f"  Overall          : {score.overall}")
#         print(f"  Band             : {band}")
#         print(f"  task_length      : {score.task_length}")
#         print(f"  num_requirements : {score.num_requirements}")
#         print(f"  domain_breadth   : {score.domain_breadth}")
#         print(f"  tool_requirements: {score.tool_requirements}")
#         print(f"  reasoning_depth  : {score.reasoning_depth}")
#         print(f"  task_type        : {score.task_type}")
#         print()
#         print("  --- Raw features ---")
#         for key, value in score.breakdown.items():
#             print(f"    {key:<15}: {value}")
#         print("-" * 60)




from __future__ import annotations

import logging
import re
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np

logger = logging.getLogger(__name__)


# =========================================================
# DATA CLASS
# =========================================================

@dataclass
class QueryComplexityScore:
    """
    Holds every complexity score for a single query.

    Attributes
    ----------
    overall          : Weighted aggregate score, 0–10.
    task_length      : Vocabulary richness + length dimension, 0–10.
    num_requirements : Requirement-count dimension, 0–10.
    domain_breadth   : Subject-domain coverage dimension, 0–10.
    tool_requirements: Tool-invocation demand dimension, 0–10.
    reasoning_depth  : Bloom level + multi-hop dimension, 0–10.
    task_type        : Primary task-verb complexity, 0–10.
    complexity_band  : Routing label – "LLM" | "CoT" | "SA" | "MA".
    breakdown        : Raw feature values used to produce the scores.
    """

    overall:           float
    task_length:       float
    num_requirements:  float
    domain_breadth:    float
    tool_requirements: float
    reasoning_depth:   float
    task_type:         float
    complexity_band:   str
    breakdown:         dict[str, Any] = field(default_factory=dict)


# =========================================================
# ANALYZER
# =========================================================

class QueryComplexityAnalyzer:
    """
    Scores the complexity of a natural-language query across six
    weighted dimensions and maps the result to a routing band.

    Default static band boundaries (before dataset fitting)
    --------------------------------------------------------
    LLM  – simple, direct answer       score  < 2
    CoT  – needs chain-of-thought      score  < 5
    SA   – needs search / agents       score  < 7
    MA   – multi-agent / very hard     score >= 7

    After calling ``fit_thresholds`` or ``analyze_dataset`` the
    boundaries are replaced by empirical percentiles from real data.
    """

    DEFAULT_WEIGHTS: dict[str, float] = {
        "task_length":       0.10,
        "num_requirements":  0.18,
        "domain_breadth":    0.18,
        "tool_requirements": 0.20,
        "reasoning_depth":   0.22,
        "task_type":         0.12,
    }

    # ------------------------------------------------------------------
    # Bloom's taxonomy – ordered highest → lowest so _get_bloom_level()
    # can return the FIRST (= most demanding) match.
    # ------------------------------------------------------------------
    BLOOM_KEYWORDS: list[tuple[str, str]] = [
        ("L6_create",     r"\b(create|design|build|compose|formulate|invent|devise)\b"),
        ("L5_evaluate",   r"\b(evaluate|assess|judge|critique|recommend|justify|defend)\b"),
        ("L4_analyze",    r"\b(analyze|analyse|compare|contrast|differentiate|examine)\b"),
        ("L3_apply",      r"\b(apply|use|implement|demonstrate|calculate|solve|execute)\b"),
        ("L2_understand", r"\b(explain|describe|summarize|interpret|classify|paraphrase)\b"),
        ("L1_remember",   r"\b(list|recall|define|identify|name|state|who|what|when|where)\b"),
    ]

    # Score (0–10) assigned to each Bloom level.
    BLOOM_SCORES: dict[str, float] = {
        "L1_remember":  1.0,
        "L2_understand":3.0,
        "L3_apply":     5.0,
        "L4_analyze":   7.0,
        "L5_evaluate":  8.5,
        "L6_create":   10.0,
    }

    # # Domain keywords that indicate subject-area breadth.
    # DOMAIN_KEYWORDS: list[str] = [
    #     "code", "data", "finance", "legal", "research", "science",
    #     "medical", "statistics", "machine learning", "database",
    # ]

    # # Keywords suggesting a tool or computation must be invoked.
    # TOOL_KEYWORDS: list[str] = [
    #     "calculate", "plot", "search", "api", "sql", "graph",
    #     "visualize", "query", "fetch", "simulate",
    # ]

    DOMAIN_KEYWORDS: list[str] = [
        # programming / software
        r"cod(e|ing|er)", r"programm(ing|er)", r"software", r"algorithm",
        r"python", r"java\b", r"javascript", r"function\b", r"script\b",
        # data / machine learning
        r"data\b", r"dataset", r"analytics?", r"analysis", r"pipeline",
        r"model(ing)?", r"predict(ion)?", r"classif(y|ication)",
        r"clustering?", r"regression", r"neural", r"machine.?learning",
        r"deep.?learning", r"\bllm\b", r"embedding",
        # mathematics / statistics
        r"statistics?", r"probabilit(y|ies)", r"distribution\b",
        r"equation", r"formula", r"math(ematics)?", r"calculus",
        r"algebra", r"geometry", r"derivative", r"integral\b",
        # finance / economics
        r"financ(e|ial)", r"econom(y|ic)", r"budget\b", r"revenue",
        r"invest(ment)?", r"stock\b", r"gdp\b", r"market\b",
        r"tax(ation)?", r"accounting", r"profit\b", r"\bloss\b",
        # legal / policy
        r"legal\b", r"\blaw\b", r"regulat(ion|ory)", r"compliance",
        r"contract\b", r"polic(y|ies)", r"legislation", r"\bcourt\b",
        # natural science
        r"science\b", r"biolog(y|ical)", r"chemistr(y|ical)", r"physics",
        r"molecule", r"element\b", r"reaction\b", r"organism",
        r"astrono(my|mical)", r"quantum",
        # medicine / health
        r"medical\b", r"health\b", r"disease\b", r"symptom",
        r"diagnosis", r"treatment", r"\bdrug\b", r"clinical", r"patient\b",
        # databases / infrastructure
        r"database", r"\bsql\b", r"schema\b", r"\bserver\b",
        r"\bcloud\b", r"infrastructure", r"architect(ure)?",
        # research / academic
        r"research\b", r"stud(y|ies)", r"experiment", r"hypothesis",
        r"\bpaper\b", r"journal\b", r"peer.?review",
    ]
 
    # Tool keywords — patterns whose presence implies an external tool,
    # computation, or system call is needed to answer the query.
    TOOL_KEYWORDS: list[str] = [
        # arithmetic / computation
        r"calculat(e|ion)", r"comput(e|ation)", r"solv(e|ing)",
        r"estimat(e|ion)", r"measur(e|ment)",
        # visualisation
        r"\bplot\b", r"\bchart\b", r"\bgraph\b", r"visuali(ze|se|sation|zation)",
        r"diagram\b", r"\bfigure\b",
        # data retrieval / search
        r"\bsearch\b", r"look.?up", r"retriev(e|al)", r"\bfetch\b",
        r"\bfind\b", r"\bquery\b",
        # APIs / web
        r"\bapi\b", r"endpoint", r"\bhttp\b", r"\brequest\b",
        r"webhook", r"\bscrape\b", r"\bcrawl\b",
        # SQL / database ops
        r"\bsql\b", r"\bselect\b", r"\bjoin\b", r"\binsert\b",
        # simulation / optimisation / training
        r"simulat(e|ion)", r"optimiz(e|ation)", r"\btrain\b", r"fine.?tun(e|ing)",
        # file / system operations
        r"\bpars(e|ing)\b", r"\bconvert\b", r"\bexecut(e|ion)\b",
    ]
    # Multi-hop patterns ordered highest complexity → lowest.
    # Each entry is (compiled pattern, score).
    MULTI_HOP_PATTERNS: list[tuple[re.Pattern[str], float]] = [
        (re.compile(r"\bthen\b.+\bafter\b|\bafter\b.+\bthen\b",    re.I | re.S), 9.0),
        (re.compile(r"\bif\b.+\bthen\b",                            re.I | re.S), 7.5),
        (re.compile(r"\bgiven that\b|\bassuming\b|\bprovided that\b",re.I),        6.0),
        (re.compile(r"\bif\b",                                       re.I),        5.0),
        (re.compile(r"\bfirst\b.+\bthen\b|\bstep \d\b",             re.I | re.S), 4.0),
    ]

    # Task-type keywords ordered highest complexity → lowest.
    # Each entry is (keyword, score). Checked in order so more specific
    # terms (e.g. "compare") win over broad ones (e.g. "what").
    TASK_TYPE_SCORES: list[tuple[str, float]] = [
        ("compare",   8.0),
        ("evaluate",  7.5),
        ("analyze",   7.0),
        ("generate",  6.5),
        ("design",    6.5),
        ("write",     5.5),
        ("summarize", 4.5),
        ("list",      3.5),
        ("what",      3.0),
        ("who",       2.5),
        ("when",      2.0),
        ("where",     2.0),
    ]

    # =========================================================
    # CONSTRUCTION
    # =========================================================

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold_percentiles: tuple[int, int, int] = (30, 70, 90),
        mattr_window: int = 25,
    ) -> None:
        """
        Parameters
        ----------
        weights
            Per-dimension weights; must contain the same keys as
            ``DEFAULT_WEIGHTS``.  If *None*, defaults are used.
        threshold_percentiles
            Three strictly-increasing integers in (0, 100) that define the
            LLM / CoT / SA band boundaries after dataset fitting.
        mattr_window
            Sliding-window width for the MATTR computation.  Clamped to
            at least 1.
        """
        self._validate_percentiles(threshold_percentiles)

        self.threshold_percentiles = threshold_percentiles
        self.mattr_window          = max(1, mattr_window)
        self.weights               = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
        self.thresholds: dict[str, float] | None = None

        if weights is not None:
            self._warn_if_weights_off(weights)

    # =========================================================
    # PUBLIC API
    # =========================================================

    def analyze(self, task: str) -> QueryComplexityScore:
        """Score a single query and return a :class:`QueryComplexityScore`."""

        task_lower = task.lower()

        # ---- raw feature extraction ----
        word_count   = len(task.split())
        mattr        = self._compute_mattr(task)
        requirements = self._count_requirements(task)
        domain_hits  = self._count_keyword_matches(task_lower, self.DOMAIN_KEYWORDS)
        tool_hits    = self._count_keyword_matches(task_lower, self.TOOL_KEYWORDS)
        bloom_level  = self._get_bloom_level(task_lower)
        multi_hop    = self._multi_hop_score(task_lower)
        task_type    = self._task_type_score(task_lower)

        # ---- dimension scores (all on 0–10 scale) ----
        s_length    = 0.6 * self._score_word_count(word_count) + 0.4 * self._score_mattr(mattr)
        s_req       = self._score_requirements(requirements)
        s_domain    = min(10.0, domain_hits * 2.0)
        s_tools     = min(10.0, tool_hits   * 2.5)
        s_reasoning = self.BLOOM_SCORES[bloom_level] * 0.6 + multi_hop * 0.4

        # ---- weighted overall (capped at 10) ----
        overall = min(10.0, (
            self.weights["task_length"]       * s_length    +
            self.weights["num_requirements"]  * s_req       +
            self.weights["domain_breadth"]    * s_domain    +
            self.weights["tool_requirements"] * s_tools     +
            self.weights["reasoning_depth"]   * s_reasoning +
            self.weights["task_type"]         * task_type
        ))

        band = self._classify(overall)

        return QueryComplexityScore(
            overall           = round(overall,     4),
            task_length       = round(s_length,    4),
            num_requirements  = round(s_req,       4),
            domain_breadth    = round(s_domain,    4),
            tool_requirements = round(s_tools,     4),
            reasoning_depth   = round(s_reasoning, 4),
            task_type         = round(task_type,   4),
            complexity_band   = band,
            breakdown={                                # FIX: was always empty {}
                "word_count":   word_count,
                "mattr":        round(mattr, 4),
                "requirements": requirements,
                "domain_hits":  domain_hits,
                "tool_hits":    tool_hits,
                "bloom_level":  bloom_level,
                "multi_hop":    round(multi_hop, 2),
            },
        )

    def analyze_dataset(
        self, queries: Sequence[str]
    ) -> list[dict[str, Any]]:
        """
        Score every query, fit percentile thresholds from the results,
        then re-classify each query using the fitted thresholds.

        Returns a list of ``{"query", "score", "band"}`` dicts.

        .. note::
            The :attr:`~QueryComplexityScore.complexity_band` stored inside
            each :class:`QueryComplexityScore` uses static defaults because
            thresholds are unknown at analysis time.  The ``"band"`` key in
            the returned dicts reflects the **fitted** thresholds and is
            therefore the authoritative label.
        """
        scores = [self.analyze(q) for q in queries]
        self.fit_thresholds(scores)

        return [
            {
                "query": q,
                "score": s.overall,
                "band":  self._classify(s.overall),  # FIX: re-classify with fitted thresholds
            }
            for q, s in zip(queries, scores)
        ]

    # =========================================================
    # THRESHOLD FITTING & CLASSIFICATION
    # =========================================================

    def fit_thresholds(
        self,
        scores: Iterable[QueryComplexityScore],
        method: str = "percentile",
    ) -> None:
        """
        Derive band boundaries from the empirical distribution of scores.

        Parameters
        ----------
        scores
            Iterable of :class:`QueryComplexityScore` objects.
        method
            ``"percentile"`` – places cuts at the user-supplied rank positions.
            ``"gmm"``        – fits a 4-component Gaussian Mixture Model and
                               places cuts midway between adjacent cluster
                               centres (requires ``scikit-learn``).

        Raises
        ------
        ValueError
            If *scores* is empty or *method* is unrecognised.
        """
        values = np.array([s.overall for s in scores], dtype=float)

        if values.size == 0:
            raise ValueError("Cannot fit thresholds: score list is empty.")

        if method == "percentile":
            p_low, p_mid, p_high = self.threshold_percentiles
            self.thresholds = {
                "LLM": float(np.percentile(values, p_low)),
                "CoT": float(np.percentile(values, p_mid)),
                "SA":  float(np.percentile(values, p_high)),
            }

        elif method == "gmm":
            try:
                from sklearn.mixture import GaussianMixture
            except ImportError:
                raise ImportError(
                    "scikit-learn is required for method='gmm'. "
                    "Install it with: pip install scikit-learn"
                )
            gmm = GaussianMixture(n_components=4, random_state=0)
            gmm.fit(values.reshape(-1, 1))
            centres = sorted(gmm.means_.flatten())
            self.thresholds = {
                "LLM": (centres[0] + centres[1]) / 2,
                "CoT": (centres[1] + centres[2]) / 2,
                "SA":  (centres[2] + centres[3]) / 2,
            }

        else:
            raise ValueError(
                f"Unknown method '{method}'. Choose 'percentile' or 'gmm'."
            )

        logger.debug("Fitted band thresholds (%s): %s", method, self.thresholds)

    def classify_score(self, score: float) -> str:
        """Public convenience wrapper around :meth:`_classify`."""
        return self._classify(score)

    def _classify(self, score: float) -> str:
        """Map a numeric score to a routing band label."""
        t = self.thresholds

        if t is None:
            # Static defaults used before dataset fitting.
            if score < 2: return "LLM"
            if score < 5: return "CoT"
            if score < 7: return "SA"
            return "MA"

        if score < t["LLM"]: return "LLM"
        if score < t["CoT"]: return "CoT"
        if score < t["SA"]:  return "SA"
        return "MA"

    # =========================================================
    # FEATURE EXTRACTION
    # =========================================================

    def _compute_mattr(self, text: str) -> float:
        """
        Moving Average Type-Token Ratio (MATTR).

        Slides a window of width ``mattr_window`` across the token list
        and averages the TTR of each window position.  This gives a
        length-insensitive measure of vocabulary richness.

        FIX: the original implementation computed plain TTR
        (``len(set) / len(words)``) and never used ``mattr_window``.
        """
        words = re.findall(r"\b[a-z]+\b", text.lower())
        n     = len(words)

        if n == 0:
            return 0.0

        w = min(self.mattr_window, n)          # window cannot exceed text length

        # --- initialise the first window ---
        window: deque[str]   = deque(words[:w])
        freq:   dict[str, int] = {}
        for word in window:
            freq[word] = freq.get(word, 0) + 1

        ttr_sum   = len(freq) / w              # TTR of the first window
        n_windows = 1

        # --- slide the window across remaining tokens ---
        for i in range(w, n):
            # remove leftmost token
            outgoing = words[i - w]
            freq[outgoing] -= 1
            if freq[outgoing] == 0:
                del freq[outgoing]

            # add new rightmost token
            incoming = words[i]
            freq[incoming] = freq.get(incoming, 0) + 1

            ttr_sum   += len(freq) / w
            n_windows += 1

        return ttr_sum / n_windows

    def _count_requirements(self, task: str) -> int:
        """
        Count explicit requirement signals in *task*.

        Signals counted
        ---------------
        * Question marks       – each "?" is a distinct sub-question.
        * Conjunctions         – "and", "also", "moreover", "additionally".
        * Enumeration markers  – "first", "second", ordinals, "finally", etc.

        FIX: the original used ``max(1, ...)`` which forced a minimum of 1
        even for trivially simple queries, inflating ``_score_requirements``
        for every input.  The floor is removed; 0 is now a valid return.
        """
        question_marks = task.count("?")
        conjunctions   = len(re.findall(
            r"\b(and|also|moreover|additionally|furthermore)\b", task, re.I
        ))
        enum_markers   = len(re.findall(
            r"\b(first|second|third|fourth|finally|lastly|\d+[.)]\s)", task, re.I
        ))
        return question_marks + conjunctions + enum_markers

    def _count_keyword_matches(self, text: str, keywords: list[str]) -> int:
        """Return how many keywords from *keywords* appear in *text*."""
        return sum(1 for kw in keywords if kw in text)

    def _get_bloom_level(self, text: str) -> str:
        """
        Return the highest Bloom's taxonomy level triggered by *text*.

        FIX: the original used an if/elif chain where the order of the
        conditions determined the result (e.g. "evaluate" was checked
        before "create"), so a query containing "create" could be
        mis-classified as L5.  We now iterate from L6 → L1 explicitly
        and return on the first (highest) match.
        """
        for level, pattern in self.BLOOM_KEYWORDS:
            if re.search(pattern, text):
                return level
        return "L1_remember"

    def _multi_hop_score(self, text: str) -> float:
        """
        Score the degree of conditional / multi-step reasoning required.

        Patterns are checked highest-complexity first; the score of the
        first matching pattern is returned.

        FIX: the original used plain ``"then" in text and "after" in text``
        substring checks that could fire on unrelated sentences, and gave
        a disproportionately high score (6) for any use of "if" alone.
        Patterns are now compiled regexes, reducing false positives.
        """
        for pattern, score in self.MULTI_HOP_PATTERNS:
            if pattern.search(text):
                return score
        return 2.0      # baseline: no multi-hop structure detected

    def _task_type_score(self, text: str) -> float:
        """
        Assign a complexity score based on the primary task verb or
        question word found in *text*.

        FIX: the original checked "what" before "compare", so a query
        like "what should I compare …" returned 3 instead of 8.
        Keywords are now ordered highest → lowest complexity.
        """
        for keyword, score in self.TASK_TYPE_SCORES:
            if keyword in text:
                return score
        return 5.0      # neutral default when no keyword matches

    # =========================================================
    # SCORING HELPERS  (raw feature → 0–10 dimension score)
    # =========================================================

    def _score_word_count(self, wc: int) -> float:
        """
        Piecewise mapping: word count → 0–10.

        FIX: the original jumped straight from 2 at <20 words to 4 at
        <50 words with no differentiation for very short queries (1–9 words
        vs 10–19 words both returned 2).
        """
        if wc <  10: return 1.0
        if wc <  20: return 2.5
        if wc <  50: return 4.0
        if wc < 100: return 6.0
        if wc < 200: return 8.0
        return 10.0

    def _score_mattr(self, m: float) -> float:
        """MATTR value (0–1) → 0–10 linear scale."""
        return min(10.0, m * 10.0)

    def _score_requirements(self, r: int) -> float:
        """
        Map requirement count to 0–10.

        FIX: 0 requirements now correctly scores 0 instead of being
        lifted to 2 by the removed ``max(1, ...)`` floor in
        ``_count_requirements``.
        """
        return min(10.0, r * 2.0)

    # =========================================================
    # VALIDATION HELPERS
    # =========================================================

    @staticmethod
    def _validate_percentiles(percentiles: tuple[int, int, int]) -> None:
        if len(percentiles) != 3:
            raise ValueError(
                "threshold_percentiles must contain exactly 3 values; "
                f"got {len(percentiles)}."
            )
        if not (0 < percentiles[0] < percentiles[1] < percentiles[2] < 100):
            raise ValueError(
                "Percentiles must be strictly increasing and lie within (0, 100). "
                f"Got: {percentiles}."
            )

    @staticmethod
    def _warn_if_weights_off(weights: dict[str, float]) -> None:
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            logger.warning(
                "Custom weights sum to %.6f instead of 1.0. "
                "Overall scores will be scaled accordingly.",
                total,
            )



from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
GAIA_GOLDEN_PATH = REPO_ROOT / "datasets" / "golden" / "gaia.parquet"


# =========================================================
# USER INPUT
# =========================================================

def get_user_percentiles() -> tuple[int, int, int]:
    """Prompt the user for three band-boundary percentiles and validate them."""
    print("=" * 60)
    print("  Query Complexity Analyzer")
    print("=" * 60)
    print()
    print("Enter three percentiles that define the band boundaries.")
    print("  p_low  → scores below this percentile are classified LLM")
    print("  p_mid  → scores below this percentile are classified CoT")
    print("  p_high → scores below this percentile are classified SA")
    print("           (everything above p_high is classified MA)")
    print()
    print("Values must be integers, strictly increasing, and in (0, 100).")
    print("Example: 30  70  90")
    print()

    while True:
        raw = input("Enter p_low  p_mid  p_high: ").strip().split()

        if len(raw) != 3:
            print(f"  ✗ Expected 3 values, got {len(raw)}. Try again.\n")
            continue

        try:
            p_low, p_mid, p_high = int(raw[0]), int(raw[1]), int(raw[2])
        except ValueError:
            print("  ✗ All three values must be whole numbers. Try again.\n")
            continue

        if not (0 < p_low < p_mid < p_high < 100):
            print("  ✗ Values must be strictly increasing and between 0 and 100. Try again.\n")
            continue

        print(f"\n  ✓ Using percentiles: LLM < p{p_low}  |  CoT < p{p_mid}  |  SA < p{p_high}  |  MA")
        print()
        return p_low, p_mid, p_high


def get_threshold_method() -> str:
    """Ask whether to use GMM or percentile-based threshold fitting."""
    print("Choose threshold fitting method:")
    print("  1 → GMM        (finds natural clusters in your data — recommended)")
    print("  2 → Percentile (places cuts at exact rank positions)")
    print()

    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            print("  ✓ Using GMM\n")
            return "gmm"
        if choice == "2":
            print("  ✓ Using percentile\n")
            return "percentile"
        print("  ✗ Please enter 1 or 2.\n")


# =========================================================
# ENTRY POINT
# =========================================================

if __name__ == "__main__":

    # --- user configuration ---
    percentiles = get_user_percentiles()
    method      = get_threshold_method()

    # --- load data ---
    analyzer = QueryComplexityAnalyzer(threshold_percentiles=percentiles)
    df       = pd.read_parquet(GAIA_GOLDEN_PATH)
    print(df.head())
    queries  = df["query"].tolist()


    # # --- score every query ---
    # scores = [analyzer.analyze(q) for q in queries]

    # # --- fit thresholds from the corpus distribution ---
    # analyzer.fit_thresholds(scores, method=method)

    # # --- show the fitted numeric boundaries ---
    # t = analyzer.thresholds
    # print("Fitted band boundaries")=
    # print(f"  LLM  : score < {t['LLM']:.4f}")
    # print(f"  CoT  : score < {t['CoT']:.4f}")
    # print(f"  SA   : score < {t['SA']:.4f}")
    # print(f"  MA   : score >= {t['SA']:.4f}")
    # print()
    # print("-" * 60)

    # # --- print results ---
    # for query, score in zip(queries, scores):
    #     band = analyzer.classify_score(score.overall)

    #     print(f"{score.overall:.2f}  {band:<4}  {query}")
    #     print(f"  Overall          : {score.overall}")
    #     print(f"  Band             : {band}")
    #     print(f"  task_length      : {score.task_length}")
    #     print(f"  num_requirements : {score.num_requirements}")
    #     print(f"  domain_breadth   : {score.domain_breadth}")
    #     print(f"  tool_requirements: {score.tool_requirements}")
    #     print(f"  reasoning_depth  : {score.reasoning_depth}")
    #     print(f"  task_type        : {score.task_type}")
    #     print()
    #     print("  --- Raw features ---")
    #     for key, value in score.breakdown.items():
    #         print(f"    {key:<15}: {value}")
    #     print("-" * 60)