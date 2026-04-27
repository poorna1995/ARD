# from __future__ import annotations

# import re
# import logging
# from collections import deque
# from dataclasses import dataclass, field
# from typing import Any
# import statistics

# logger = logging.getLogger(__name__)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Data Class
# # ══════════════════════════════════════════════════════════════════════════════

# @dataclass
# class TaskComplexityScore:
#     overall:         float
#     task_length:     float
#     reasoning_depth: float
#     tool_dependency: float
#     domain_breadth:  float
#     task_type:       float
#     complexity_band: str
#     breakdown:       dict[str, Any] = field(default_factory=dict)


# # ══════════════════════════════════════════════════════════════════════════════
# #  Analyzer
# # ══════════════════════════════════════════════════════════════════════════════

# class TaskComplexityAnalyzer:

#     # ── Top-level weights (must sum to 1.0) ───────────────────────────────────

#     DEFAULT_WEIGHTS: dict[str, float] = {
#         "task_length":     0.12,
#         "reasoning_depth": 0.30,
#         "tool_dependency": 0.25,
#         "domain_breadth":  0.15,
#         "task_type":       0.18,
#     }

#     # ── Sub-dimension weights ─────────────────────────────────────────────────

#     TASK_LENGTH_WEIGHTS: dict[str, float] = {
#         "word_count": 0.60,
#         "mattr":      0.40,
#     }

#     REASONING_DEPTH_WEIGHTS: dict[str, float] = {
#         "bloom":    0.50,   # cognitive level of the task
#         "multihop": 0.30,   # chained steps / conditions
#         "negation": 0.20,   # linguistic negation complexity
#     }

#     # ── Bloom's Taxonomy ──────────────────────────────────────────────────────

#     BLOOM_KEYWORDS: list[tuple[str, str]] = [
#         ("L6_create",     r"\b(create|design|build|compose|formulate|invent|devise)\b"),
#         ("L5_evaluate",   r"\b(evaluate|assess|judge|critique|recommend|justify|defend)\b"),
#         ("L4_analyze",    r"\b(analyze|analyse|compare|contrast|differentiate|examine)\b"),
#         ("L3_apply",      r"\b(apply|use|implement|demonstrate|calculate|solve|execute)\b"),
#         ("L2_understand", r"\b(explain|describe|summarize|interpret|classify|paraphrase)\b"),
#         ("L1_remember",   r"\b(list|recall|define|identify|name|state|who|what|when|where)\b"),
#     ]

#     BLOOM_SCORES: dict[str, float] = {
#         "L1_remember":   1.0,
#         "L2_understand": 3.0,
#         "L3_apply":      5.0,
#         "L4_analyze":    7.0,
#         "L5_evaluate":   8.5,
#         "L6_create":    10.0,
#     }

#     # ── Multi-hop patterns (ordered high → low score) ─────────────────────────

#     MULTI_HOP_PATTERNS: list[tuple[re.Pattern[str], float]] = [
#         # sequential
#         (re.compile(r"\bthen\b.+\bafter\b|\bafter\b.+\bthen\b",               re.I | re.S), 9.0),
#         # enumeration
#         (re.compile(r"\bfirst\b.+\bsecond\b.+\bthird\b",                      re.I | re.S), 8.5),
#         # causal chain
#         (re.compile(r"\bbecause\b.+\btherefore\b|\bsince\b.+\bthus\b",        re.I | re.S), 8.0),
#         # conditional
#         (re.compile(r"\bif\b.+\bthen\b",                                       re.I | re.S), 7.5),
#         # dependency
#         (re.compile(r"\bdepends on\b|\brequires\b.+\bbefore\b",               re.I | re.S), 7.0),
#         # contrast + conclusion
#         (re.compile(r"\balthough\b.+\bnevertheless\b|\bdespite\b.+\bstill\b", re.I | re.S), 7.0),
#         # temporal ordering
#         (re.compile(r"\bbefore\b.+\bafter\b|\bprior to\b",                    re.I | re.S), 6.5),
#         # conditional negation
#         (re.compile(r"\bunless\b|\buntil\b.+\bthen\b",                        re.I | re.S), 6.5),
#         # assumption
#         (re.compile(r"\bgiven that\b|\bassuming\b|\bprovided that\b",          re.I),        6.0),
#         # purpose
#         (re.compile(r"\bin order to\b|\bso that\b|\bsuch that\b",             re.I),        6.0),
#         # temporal continuation
#         (re.compile(r"\bonce\b.+\bthen\b|\bwhen\b.+\bthen\b",                re.I | re.S), 6.0),
#         # consequence
#         (re.compile(r"\bas a result\b|\bconsequently\b|\btherefore\b",        re.I),        5.5),
#         # simple conditional
#         (re.compile(r"\bif\b",                                                 re.I),        5.0),
#         # branching
#         (re.compile(r"\botherwi?se\b",                                         re.I),        5.0),
#         # temporal
#         (re.compile(r"\bsubsequently\b|\bafterward(s)?\b",                    re.I),        4.5),
#         # contrast
#         (re.compile(r"\bhowever\b|\bnonetheless\b|\byet\b",                   re.I),        4.0),
#         # step-based
#         (re.compile(r"\bfirst\b.+\bthen\b|\bstep \d\b",                      re.I | re.S), 4.0),
#         # closing marker
#         (re.compile(r"\bfinally\b|\blastly\b|\bin the end\b",                 re.I),        3.5),
#     ]

#     # ── Negation patterns (ordered high → low score) ──────────────────────────

#     NEGATION_PATTERNS: list[tuple[re.Pattern[str], float]] = [
#         # double negation → affirmative
#         (re.compile(r"\bnot\b.+\bwithout\b|\bnever\b.+\bunless\b",            re.I | re.S), 9.0),
#         # negated condition
#         (re.compile(r"\bif not\b|\bif no\b|\bif neither\b",                   re.I),        8.0),
#         # exception-based
#         (re.compile(r"\bexcept\b.+\bif\b|\bexcept\b.+\bwhen\b",              re.I | re.S), 7.5),
#         (re.compile(r"\bunless\b.+\botherwise\b",                             re.I | re.S), 7.5),
#         # negated conjunction
#         (re.compile(r"\bneither\b.+\bnor\b",                                  re.I | re.S), 7.0),
#         # counterfactual
#         (re.compile(r"\bhad not\b|\bwere not\b|\bwould not\b.+\bif\b",        re.I | re.S), 7.0),
#         (re.compile(r"\bwithout\b.+\bwould\b|\bwithout\b.+\bcould\b",         re.I | re.S), 6.5),
#         # denial before instruction
#         (re.compile(r"\bdo not\b.+\buntil\b|\bdon'?t\b.+\buntil\b",          re.I | re.S), 6.0),
#         (re.compile(r"\bdo not\b.+\bbefore\b|\bdon'?t\b.+\bbefore\b",         re.I | re.S), 6.0),
#         # simple exclusion
#         (re.compile(r"\bnot\b.+\bbut\b|\bnot\b.+\binstead\b",                re.I | re.S), 5.5),
#         (re.compile(r"\brather than\b|\binstead of\b",                        re.I),        4.5),
#         # bare negators — keep last
#         (re.compile(r"\bcannot\b|\bcan'?t\b|\bwon'?t\b|\bshouldn'?t\b",      re.I),        3.5),
#         (re.compile(r"\bno\b|\bnot\b|\bnever\b|\bnone\b",                     re.I),        2.0),
#     ]

#     # ── Domain keywords ───────────────────────────────────────────────────────

#     DOMAINS: dict[str, list[str]] = {
#         "technical":  ["code", "programming", "software", "algorithm", "debug", "implement", "script", "api"],
#         "research":   ["study", "research", "investigate", "analyze", "survey", "review", "literature"],
#         "business":   ["market", "sales", "revenue", "business", "strategy", "roi", "profit"],
#         "creative":   ["design", "create", "write", "compose", "generate", "draft", "brainstorm"],
#         "data":       ["data", "statistics", "analytics", "metrics", "dataset", "visualization", "analysis"],
#         "scientific": ["experiment", "hypothesis", "theory", "scientific", "methodology", "findings"],
#         "legal":      ["legal", "law", "regulation", "compliance", "contract", "policy"],
#         "financial":  ["financial", "accounting", "budget", "investment", "forecast", "valuation"],
#     }

#     # ── Tool indicators ───────────────────────────────────────────────────────

#     TOOL_INDICATORS: dict[str, list[str]] = {
#         "web_search":    ["search", "find online", "look up", "google", "web"],
#         "code_executor": ["run code", "execute", "test", "python", "script"],
#         "calculator":    ["calculate", "compute", "math", "equation", "formula"],
#         "file_ops":      ["file", "document", "read", "write", "save", "load"],
#         "api":           ["api", "request", "fetch data", "endpoint", "rest"],
#         "database":      ["database", "sql", "query", "table", "data"],
#         "image":         ["image", "picture", "photo", "visualization", "chart", "graph"],
#         "video":         ["video", "movie", "stream", "multimedia"],
#     }

#     # ── Task-type scores (ordered high → low) ─────────────────────────────────

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

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Constructor
#     # ══════════════════════════════════════════════════════════════════════════

#     def __init__(
#         self,
#         weights: dict[str, float] | None = None,
#         threshold_percentiles: tuple[int, int, int] = (30, 70, 90),
#         mattr_window: int = 15,
#     ) -> None:
#         self._validate_percentiles(threshold_percentiles)
#         raw             = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
#         normalized      = self._normalize_weights(raw)
#         self._check_weights(normalized)
#         self.weights               = normalized
#         self.threshold_percentiles = threshold_percentiles
#         self.thresholds: dict[str, float] | None = None
#         self.mattr_window          = mattr_window
#         self._score_history: list[float] = []   # accumulates overall scores for threshold fitting

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Public API
#     # ══════════════════════════════════════════════════════════════════════════

#     def analyze(self, query: str) -> TaskComplexityScore:
#         """Score the complexity of *query* and return a TaskComplexityScore."""
#         text = query.lower()

#         task_length     = self._score_task_length(text)
#         reasoning_depth = self._score_reasoning_depth(text)
#         tool_dependency = self._score_tool_dependency(text)
#         domain_breadth  = self._score_domain_breadth(text)
#         task_type       = self._score_task_type(text)

#         scores = {
#             "task_length":     task_length,
#             "reasoning_depth": reasoning_depth,
#             "tool_dependency": tool_dependency,
#             "domain_breadth":  domain_breadth,
#             "task_type":       task_type,
#         }

#         overall = sum(self.weights[dim] * score for dim, score in scores.items())

#         # ── update thresholds from accumulated score history ───────────────────
#         self._score_history.append(overall)
#         self._recompute_thresholds()

#         band = self._complexity_band(overall)

#         result = TaskComplexityScore(
#             overall         = round(overall, 3),
#             task_length     = round(task_length, 3),
#             reasoning_depth = round(reasoning_depth, 3),
#             tool_dependency = round(tool_dependency, 3),
#             domain_breadth  = round(domain_breadth, 3),
#             task_type       = round(task_type, 3),
#             complexity_band = band,
#             breakdown       = scores,
#         )

#         self._print_report(query, result)
#         return result

#     def fit(self, queries: list[str]) -> "TaskComplexityAnalyzer":
#         """
#         Pre-compute thresholds from a reference corpus of queries.

#         Call this before analyze() when you have a representative dataset,
#         so that complexity bands reflect the actual score distribution.

#         Returns self for chaining: analyzer.fit(corpus).analyze(query)
#         """
#         print(f"\n  [Fitting thresholds on {len(queries)} queries...]")
#         for q in queries:
#             text    = q.lower()
#             scores  = {
#                 "task_length":     self._score_task_length(text),
#                 "reasoning_depth": self._score_reasoning_depth(text),
#                 "tool_dependency": self._score_tool_dependency(text),
#                 "domain_breadth":  self._score_domain_breadth(text),
#                 "task_type":       self._score_task_type(text),
#             }
#             overall = sum(self.weights[dim] * s for dim, s in scores.items())
#             self._score_history.append(overall)

#         self._recompute_thresholds()
#         print(f"  [Thresholds set] → {self.thresholds}")
#         return self

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 1 — Task Length
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_task_length(self, text: str) -> float:
#         """Weighted combination of word-count bracket and MATTR vocabulary richness."""
#         word_count  = self._compute_word_count(text)
#         wc_score    = self._score_word_count(word_count)
#         mattr       = self._compute_mattr(text)
#         mattr_score = mattr * 10.0          # scale 0–1 → 0–10

#         print(f"\n  [Task Length]")
#         print(f"    Word count  : {word_count}  →  bracket score = {wc_score}")
#         print(f"    MATTR       : {mattr:.4f}  →  scaled score  = {mattr_score:.4f}")

#         score = (
#             self.TASK_LENGTH_WEIGHTS["word_count"] * wc_score
#             + self.TASK_LENGTH_WEIGHTS["mattr"]    * mattr_score
#         )
#         print(f"    Score       : {score:.4f}")
#         return score

#     @staticmethod
#     def _compute_word_count(text: str) -> int:
#         return len(text.split())

#     @staticmethod
#     def _score_word_count(word_count: int) -> float:
#         if word_count < 10:  return 1.0
#         if word_count < 60:  return 3.0
#         if word_count < 150: return 5.0
#         if word_count < 250: return 7.0
#         return 10.0

#     def _compute_mattr(self, text: str) -> float:
#         """Moving Average Type-Token Ratio — length-insensitive vocabulary richness."""
#         words = re.findall(r"\b[a-z]+\b", text)
#         n     = len(words)
#         if n == 0:
#             return 0.0

#         w    = min(self.mattr_window, n)
#         freq: dict[str, int] = {}

#         for word in words[:w]:
#             freq[word] = freq.get(word, 0) + 1

#         ttr_sum, n_windows = len(freq) / w, 1

#         for i in range(w, n):
#             outgoing = words[i - w]
#             freq[outgoing] -= 1
#             if freq[outgoing] == 0:
#                 del freq[outgoing]

#             incoming = words[i]
#             freq[incoming] = freq.get(incoming, 0) + 1

#             ttr_sum   += len(freq) / w
#             n_windows += 1

#         return ttr_sum / n_windows

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 2 — Reasoning Depth  (Bloom + Multi-hop + Negation)
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_reasoning_depth(self, text: str) -> float:
#         """Weighted combination of Bloom level, multi-hop complexity, and negation."""
#         bloom_level    = self._get_bloom_level(text)
#         bloom_score    = self.BLOOM_SCORES[bloom_level]
#         multihop_score = self._score_multihop(text)
#         negation_score = self._score_negation(text)

#         print(f"\n  [Reasoning Depth]")
#         print(f"    Bloom       : {bloom_level}  →  {bloom_score}")
#         print(f"    Multi-hop   : {multihop_score}")
#         print(f"    Negation    : {negation_score}")

#         score = (
#             self.REASONING_DEPTH_WEIGHTS["bloom"]    * bloom_score
#             + self.REASONING_DEPTH_WEIGHTS["multihop"] * multihop_score
#             + self.REASONING_DEPTH_WEIGHTS["negation"] * negation_score
#         )
#         print(f"    Score       : {score:.4f}")
#         return score

#     def _get_bloom_level(self, text: str) -> str:
#         for level, pattern in self.BLOOM_KEYWORDS:
#             match = re.search(pattern, text)
#             if match:
#                 print(f"    Bloom match : level={level!r}, word={match.group()!r}")
#                 return level
#         print(f"    Bloom match : none → defaulting to 'L1_remember'")
#         return "L1_remember"

#     def _score_multihop(self, text: str) -> float:
#         for pattern, score in self.MULTI_HOP_PATTERNS:
#             match = pattern.search(text)
#             if match:
#                 print(f"    Multi-hop   : matched={match.group()!r}, score={score}")
#                 return score
#         print(f"    Multi-hop   : none found → 0.0")
#         return 0.0

#     def _score_negation(self, text: str) -> float:
#         for pattern, score in self.NEGATION_PATTERNS:
#             match = pattern.search(text)
#             if match:
#                 print(f"    Negation    : matched={match.group()!r}, score={score}")
#                 return score
#         print(f"    Negation    : none found → 0.0")
#         return 0.0

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 3 — Tool Dependency
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_tool_dependency(self, text: str) -> float:
#         """Score based on how many distinct tool types are needed."""
#         matched = {
#             tool: [kw for kw in kws if kw in text]
#             for tool, kws in self.TOOL_INDICATORS.items()
#             if any(kw in text for kw in kws)
#         }

#         print(f"\n  [Tool Dependency]")
#         if matched:
#             for tool, kws in matched.items():
#                 print(f"    {tool:<16}: {kws}")
#         else:
#             print(f"    No tools matched.")

#         count = len(matched)
#         score = {0: 0.0, 1: 3.0, 2: 5.0, 3: 7.0}.get(count, 10.0)
#         print(f"    Tools matched : {count}  →  score = {score}")
#         return score

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 4 — Domain Breadth
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_domain_breadth(self, text: str) -> float:
#         """Score based on how many distinct knowledge domains are touched."""
#         matched = {
#             domain: [kw for kw in kws if kw in text]
#             for domain, kws in self.DOMAINS.items()
#             if any(kw in text for kw in kws)
#         }

#         print(f"\n  [Domain Breadth]")
#         if matched:
#             for domain, kws in matched.items():
#                 print(f"    {domain:<12}: {kws}")
#         else:
#             print(f"    No domains matched.")

#         count = len(matched)
#         score = min(count * 2.0, 10.0)
#         print(f"    Domains matched : {count}  →  score = {score}")
#         return score

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 5 — Task Type
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_task_type(self, text: str) -> float:
#         """Match the highest-scoring task-type keyword found in the text."""
#         print(f"\n  [Task Type]")
#         for keyword, score in self.TASK_TYPE_SCORES:
#             if re.search(rf"\b{keyword}\b", text, re.I):
#                 print(f"    Matched : {keyword!r}  →  score = {score}")
#                 return score
#         print(f"    No keyword matched → defaulting to 1.0")
#         return 1.0

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Report Printer
#     # ══════════════════════════════════════════════════════════════════════════

#     def _print_report(self, query: str, result: TaskComplexityScore) -> None:
#         SEP  = "═" * 62
#         SEP2 = "─" * 62

#         print(f"\n{SEP}")
#         print(f"  TASK COMPLEXITY REPORT")
#         print(f"{SEP}")
#         print(f"  Query : {query}")
#         print(f"{SEP2}")
#         print(f"  {'Dimension':<20} {'Weight':>8}  {'Score':>8}  {'Weighted':>10}")
#         print(f"  {SEP2}")
#         for dim, score in result.breakdown.items():
#             weight   = self.weights[dim]
#             weighted = weight * score
#             print(f"  {dim:<20} {weight:>8.2f}  {score:>8.3f}  {weighted:>10.4f}")
#         print(f"  {SEP2}")
#         print(f"  {'OVERALL':<20} {'':>8}  {result.overall:>8.3f}")
#         print(f"  {'Complexity Band':<20} {'':>8}  {result.complexity_band:>10}")

#         # ── threshold info ─────────────────────────────────────────────────────
#         print(f"\n  Percentile thresholds ({self.threshold_percentiles[0]}th / "
#               f"{self.threshold_percentiles[1]}th / {self.threshold_percentiles[2]}th) :")
#         if self.thresholds:
#             p_low, p_mid, p_high = self.threshold_percentiles
#             print(f"    < {self.thresholds['simple']:.3f}  →  simple")
#             print(f"    < {self.thresholds['moderate']:.3f}  →  moderate")
#             print(f"    < {self.thresholds['complex']:.3f}  →  complex")
#             print(f"    ≥ {self.thresholds['complex']:.3f}  →  highly_complex")
#             print(f"    Scores seen so far : {len(self._score_history)}")
#         else:
#             print(f"    No history yet — using hardcoded fallback thresholds.")
#         print(f"{SEP}\n")

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Helpers
#     # ══════════════════════════════════════════════════════════════════════════

#     def _recompute_thresholds(self) -> None:
#         """
#         Derive band cut-offs from the accumulated score history using
#         the configured threshold_percentiles (default: 30th, 70th, 90th).

#         With n scores the three percentile values map to four bands:

#             score < p30            →  simple
#             p30  ≤ score < p70     →  moderate
#             p70  ≤ score < p90     →  complex
#             score ≥ p90            →  highly_complex
#         """
#         if not self._score_history:
#             self.thresholds = None
#             return

#         p_low, p_mid, p_high = self.threshold_percentiles
#         history = sorted(self._score_history)
#         n       = len(history)

#         def _percentile(data: list[float], p: int) -> float:
#             """Linear interpolation percentile (same as numpy default)."""
#             idx = (p / 100) * (len(data) - 1)
#             lo, hi = int(idx), min(int(idx) + 1, len(data) - 1)
#             return data[lo] + (data[hi] - data[lo]) * (idx - lo)

#         self.thresholds = {
#             "simple":        _percentile(history, p_low),   # below this → simple
#             "moderate":      _percentile(history, p_mid),   # below this → moderate
#             "complex":       _percentile(history, p_high),  # below this → complex
#             # above complex  → highly_complex
#         }

#     def _complexity_band(self, score: float) -> str:
#         """
#         Map *score* to a band label.

#         Uses dynamically computed percentile thresholds when available
#         (set by fit() or accumulated via analyze()); falls back to
#         hardcoded sensible defaults when no history exists yet.
#         """
#         if self.thresholds:
#             if score < self.thresholds["simple"]:   return "simple"
#             if score < self.thresholds["moderate"]: return "moderate"
#             if score < self.thresholds["complex"]:  return "complex"
#             return "highly_complex"

#         # ── fallback defaults (used before any scores are accumulated) ─────────
#         if score < 3.0: return "simple"
#         if score < 5.5: return "moderate"
#         if score < 7.5: return "complex"
#         return "highly_complex"

#     @staticmethod
#     def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
#         total = sum(weights.values())
#         if total == 0:
#             raise ValueError("Weights must not all be zero.")
#         return {k: v / total for k, v in weights.items()}

#     @staticmethod
#     def _check_weights(weights: dict[str, float]) -> None:
#         total = sum(weights.values())
#         if abs(total - 1.0) > 1e-6:
#             raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

#     @staticmethod
#     def _validate_percentiles(percentiles: tuple[int, int, int]) -> None:
#         if len(percentiles) != 3:
#             raise ValueError(f"Expected 3 percentiles, got {len(percentiles)}.")
#         if not (0 < percentiles[0] < percentiles[1] < percentiles[2] < 100):
#             raise ValueError(f"Percentiles must be strictly increasing within (0, 100). Got: {percentiles}.")


# # ══════════════════════════════════════════════════════════════════════════════
# #  Entry Point
# # ══════════════════════════════════════════════════════════════════════════════
# import pandas as pd
# if __name__ == "__main__":
#     df = pd.read_parquet("../datasets/processed/gaia.parquet")
#     corpus = df["query"].tolist()
#     analyzer = TaskComplexityAnalyzer(threshold_percentiles=(30, 70, 90))

#     # ── Mode 1: fit on corpus first, then analyze ─────────────────────────────
#     print("\n" + "█" * 62)
#     print("  MODE 1 — fit() on corpus, then analyze")
#     print("█" * 62)
#     analyzer.fit(corpus)




from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  Data Class
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TaskComplexityScore:
    overall:         float
    task_length:     float
    reasoning_depth: float
    tool_dependency: float
    domain_breadth:  float
    task_type:       float
    complexity_band: str
    breakdown:       dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "overall":         self.overall,
            "task_length":     self.task_length,
            "reasoning_depth": self.reasoning_depth,
            "tool_dependency": self.tool_dependency,
            "domain_breadth":  self.domain_breadth,
            "task_type":       self.task_type,
            "complexity_band": self.complexity_band,
        }


# ══════════════════════════════════════════════════════════════════════════════
#  Analyzer
# ══════════════════════════════════════════════════════════════════════════════

class TaskComplexityAnalyzer:

    DEFAULT_WEIGHTS: dict[str, float] = {
        "task_length":     0.12,
        "reasoning_depth": 0.30,
        "tool_dependency": 0.25,
        "domain_breadth":  0.15,
        "task_type":       0.18,
    }

    TASK_LENGTH_WEIGHTS: dict[str, float] = {
        "word_count": 0.60,
        "mattr":      0.40,
    }

    REASONING_DEPTH_WEIGHTS: dict[str, float] = {
        "bloom":    0.50,
        "multihop": 0.30,
        "negation": 0.20,
    }

    BLOOM_KEYWORDS: list[tuple[str, str]] = [
        ("L6_create",     r"\b(create|design|build|compose|formulate|invent|devise)\b"),
        ("L5_evaluate",   r"\b(evaluate|assess|judge|critique|recommend|justify|defend)\b"),
        ("L4_analyze",    r"\b(analyze|analyse|compare|contrast|differentiate|examine)\b"),
        ("L3_apply",      r"\b(apply|use|implement|demonstrate|calculate|solve|execute)\b"),
        ("L2_understand", r"\b(explain|describe|summarize|interpret|classify|paraphrase)\b"),
        ("L1_remember",   r"\b(list|recall|define|identify|name|state|who|what|when|where)\b"),
    ]

    BLOOM_SCORES: dict[str, float] = {
        "L1_remember":   1.0,
        "L2_understand": 3.0,
        "L3_apply":      5.0,
        "L4_analyze":    7.0,
        "L5_evaluate":   8.5,
        "L6_create":    10.0,
    }

    MULTI_HOP_PATTERNS: list[tuple[re.Pattern, float]] = [
        (re.compile(r"\bthen\b.+\bafter\b|\bafter\b.+\bthen\b",               re.I | re.S), 9.0),
        (re.compile(r"\bfirst\b.+\bsecond\b.+\bthird\b",                      re.I | re.S), 8.5),
        (re.compile(r"\bbecause\b.+\btherefore\b|\bsince\b.+\bthus\b",        re.I | re.S), 8.0),
        (re.compile(r"\bif\b.+\bthen\b",                                       re.I | re.S), 7.5),
        (re.compile(r"\bdepends on\b|\brequires\b.+\bbefore\b",               re.I | re.S), 7.0),
        (re.compile(r"\balthough\b.+\bnevertheless\b|\bdespite\b.+\bstill\b", re.I | re.S), 7.0),
        (re.compile(r"\bbefore\b.+\bafter\b|\bprior to\b",                    re.I | re.S), 6.5),
        (re.compile(r"\bunless\b|\buntil\b.+\bthen\b",                        re.I | re.S), 6.5),
        (re.compile(r"\bgiven that\b|\bassuming\b|\bprovided that\b",          re.I),        6.0),
        (re.compile(r"\bin order to\b|\bso that\b|\bsuch that\b",             re.I),        6.0),
        (re.compile(r"\bonce\b.+\bthen\b|\bwhen\b.+\bthen\b",                re.I | re.S), 6.0),
        (re.compile(r"\bas a result\b|\bconsequently\b|\btherefore\b",        re.I),        5.5),
        (re.compile(r"\bif\b",                                                 re.I),        5.0),
        (re.compile(r"\botherwi?se\b",                                         re.I),        5.0),
        (re.compile(r"\bsubsequently\b|\bafterward(s)?\b",                    re.I),        4.5),
        (re.compile(r"\bhowever\b|\bnonetheless\b|\byet\b",                   re.I),        4.0),
        (re.compile(r"\bfirst\b.+\bthen\b|\bstep \d\b",                      re.I | re.S), 4.0),
        (re.compile(r"\bfinally\b|\blastly\b|\bin the end\b",                 re.I),        3.5),
    ]

    NEGATION_PATTERNS: list[tuple[re.Pattern, float]] = [
        (re.compile(r"\bnot\b.+\bwithout\b|\bnever\b.+\bunless\b",            re.I | re.S), 9.0),
        (re.compile(r"\bif not\b|\bif no\b|\bif neither\b",                   re.I),        8.0),
        (re.compile(r"\bexcept\b.+\bif\b|\bexcept\b.+\bwhen\b",              re.I | re.S), 7.5),
        (re.compile(r"\bunless\b.+\botherwise\b",                             re.I | re.S), 7.5),
        (re.compile(r"\bneither\b.+\bnor\b",                                  re.I | re.S), 7.0),
        (re.compile(r"\bhad not\b|\bwere not\b|\bwould not\b.+\bif\b",        re.I | re.S), 7.0),
        (re.compile(r"\bwithout\b.+\bwould\b|\bwithout\b.+\bcould\b",         re.I | re.S), 6.5),
        (re.compile(r"\bdo not\b.+\buntil\b|\bdon'?t\b.+\buntil\b",          re.I | re.S), 6.0),
        (re.compile(r"\bdo not\b.+\bbefore\b|\bdon'?t\b.+\bbefore\b",         re.I | re.S), 6.0),
        (re.compile(r"\bnot\b.+\bbut\b|\bnot\b.+\binstead\b",                re.I | re.S), 5.5),
        (re.compile(r"\brather than\b|\binstead of\b",                        re.I),        4.5),
        (re.compile(r"\bcannot\b|\bcan'?t\b|\bwon'?t\b|\bshouldn'?t\b",      re.I),        3.5),
        (re.compile(r"\bno\b|\bnot\b|\bnever\b|\bnone\b",                     re.I),        2.0),
    ]

    DOMAINS: dict[str, list[str]] = {
        "technical":  ["code", "programming", "software", "algorithm", "debug", "implement", "script", "api"],
        "research":   ["study", "research", "investigate", "analyze", "survey", "review", "literature"],
        "business":   ["market", "sales", "revenue", "business", "strategy", "roi", "profit"],
        "creative":   ["design", "create", "write", "compose", "generate", "draft", "brainstorm"],
        "data":       ["data", "statistics", "analytics", "metrics", "dataset", "visualization", "analysis"],
        "scientific": ["experiment", "hypothesis", "theory", "scientific", "methodology", "findings"],
        "legal":      ["legal", "law", "regulation", "compliance", "contract", "policy"],
        "financial":  ["financial", "accounting", "budget", "investment", "forecast", "valuation"],
    }

    # TOOL_INDICATORS: dict[str, list[str]] = {
    #     "web_search":    ["search", "find online", "look up", "google", "web",
    #     "latest", "current", "today", "recent", "breaking",
    #     "news", "update", "live", "right now", "real time"],
    #     "code_executor": [ "run code", "execute", "test", "python", "script",
    #     "debug", "runtime", "output", "compile", "sql"],
    #     "calculator":    ["calculate", "compute", "math", "equation", "formula"],
    #     "file_ops":      ["file", "document", "read", "write", "save", "load",
    #     "open file", "export", "import"],
    #     "api":           ["api", "request", "fetch data", "endpoint", "rest",
    #     "real-time data", "live data", "stream data", "webhook"]
    #     "database":      ["database", "sql", "query", "table", "data"],
    #     "image":         ["image", "picture", "photo", "visualization", "chart", "graph"],
    #     "video":         ["video", "movie", "stream", "multimedia"],
    # }

    TOOL_INDICATORS: dict[str, dict] = {
    "web_search": {
        "weight": 1.0,          # how much this tool adds to complexity
        "strong": [             # high-confidence signals (score: 1.0 each)
            "search the web", "find online", "look it up",
            "breaking news", "real time", "live data",
            "latest news", "current events"
        ],
        "weak": [               # low-confidence signals (score: 0.3 each)
            "search", "latest", "current", "recent",
            "today", "news", "update",
            "wikipedia", "wiki", "wikipedia page", "wiki page"
        ],
    },

    "code_executor": {
        "weight": 1.4,          # running code is more complex than a search
        "strong": [
            "run code", "execute script", "run this python",
            "debug this", "compile and run", "run the program"
        ],
        "weak": [
            "python", "script", "execute", "runtime", "output"
        ],
    },

    "calculator": {
        "weight": 0.8,
        "strong": [
            "calculate", "solve the equation", "evaluate the formula",
            "compute the result", "numerical solution"
        ],
        "weak": [
            "compute", "math", "equation", "formula", "solve"
        ],
    },

    "file_ops": {
        "weight": 0.9,
        "strong": [
            "open file", "read the file", "write to file",
            "save to disk", "export file", "load from file"
        ],
        "weak": [
            "file", "document", "save", "load", "export", "import"
        ],
    },

    "api": {
        "weight": 1.3,
        "strong": [
            "call the api", "api endpoint", "rest api",
            "fetch from api", "webhook", "api request",
            "live data feed", "stream data"
        ],
        "weak": [
            "api", "endpoint", "request", "rest", "fetch"
        ],
    },

    "database": {
        "weight": 1.2,
        "strong": [
            "sql query", "run a query", "fetch records",
            "insert into", "update row", "retrieve from database",
            "join tables"
        ],
        "weak": [
            "database", "sql", "table", "query"
        ],
    },

    "image": {
        "weight": 0.9,
        "strong": [
            "generate image", "plot the graph", "create chart",
            "visualize the data", "render diagram", "draw a graph"
        ],
        "weak": [
            "image", "chart", "graph", "plot", "diagram", "photo"
        ],
    },

    "video": {
        "weight": 1.1,
        "strong": [
            "live stream", "broadcast", "video call",
            "record video", "stream this"
        ],
        "weak": [
            "video", "movie", "stream", "multimedia"
        ],
    },
}


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

    # ══════════════════════════════════════════════════════════════════════════
    #  Constructor
    # ══════════════════════════════════════════════════════════════════════════

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        threshold_percentiles: tuple[int, int, int] = (30, 70, 90),
        mattr_window: int = 15,
        verbose: bool = False,
    ) -> None:
        self._validate_percentiles(threshold_percentiles)
        raw        = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
        normalized = self._normalize_weights(raw)
        self._check_weights(normalized)
        self.weights               = normalized
        self.threshold_percentiles = threshold_percentiles
        self.thresholds: dict[str, float] | None = None
        self.mattr_window          = mattr_window
        self.verbose               = verbose
        self._score_history: list[float] = []

    # ══════════════════════════════════════════════════════════════════════════
    #  Public API
    # ══════════════════════════════════════════════════════════════════════════

    def analyze(self, query: str) -> TaskComplexityScore:
        text = query.lower()

        task_length     = self._score_task_length(text)
        reasoning_depth = self._score_reasoning_depth(text)
        tool_dependency = self._score_tool_dependency(text)
        domain_breadth  = self._score_domain_breadth(text)
        task_type       = self._score_task_type(text)

        scores = {
            "task_length":     task_length,
            "reasoning_depth": reasoning_depth,
            "tool_dependency": tool_dependency,
            "domain_breadth":  domain_breadth,
            "task_type":       task_type,
        }

        overall = sum(self.weights[dim] * score for dim, score in scores.items())

        self._score_history.append(overall)
        self._recompute_thresholds()

        band = self._complexity_band(overall)

        return TaskComplexityScore(
            overall         = round(overall, 3),
            task_length     = round(task_length, 3),
            reasoning_depth = round(reasoning_depth, 3),
            tool_dependency = round(tool_dependency, 3),
            domain_breadth  = round(domain_breadth, 3),
            task_type       = round(task_type, 3),
            complexity_band = band,
            breakdown       = scores,
        )

    def fit(self, queries: list[str]) -> "TaskComplexityAnalyzer":
        """Pre-compute thresholds from a reference corpus."""
        for q in queries:
            text   = q.lower()
            scores = {
                "task_length":     self._score_task_length(text),
                "reasoning_depth": self._score_reasoning_depth(text),
                "tool_dependency": self._score_tool_dependency(text),
                "domain_breadth":  self._score_domain_breadth(text),
                "task_type":       self._score_task_type(text),
            }
            overall = sum(self.weights[dim] * s for dim, s in scores.items())
            self._score_history.append(overall)

        self._recompute_thresholds()
        return self

    def analyze_batch(self, queries: list[str]) -> list[dict]:
        """Analyze a list of queries, return list of result dicts."""
        results = []
        for q in queries:
            score = self.analyze(q)
            d = score.to_dict()
            d["query"] = q
            results.append(d)
        return results

    # ══════════════════════════════════════════════════════════════════════════
    #  Dimension 1 — Task Length
    # ══════════════════════════════════════════════════════════════════════════

    def _score_task_length(self, text: str) -> float:
        word_count  = self._compute_word_count(text)
        wc_score    = self._score_word_count(word_count)
        mattr       = self._compute_mattr(text)
        mattr_score = mattr * 10.0

        return (
            self.TASK_LENGTH_WEIGHTS["word_count"] * wc_score
            + self.TASK_LENGTH_WEIGHTS["mattr"]    * mattr_score
        )

    @staticmethod
    def _compute_word_count(text: str) -> int:
        return len(text.split())

    @staticmethod
    def _score_word_count(word_count: int) -> float:
        if word_count < 10:  return 1.0
        if word_count < 60:  return 3.0
        if word_count < 150: return 5.0
        if word_count < 250: return 7.0
        return 10.0

    def _compute_mattr(self, text: str) -> float:
        words = re.findall(r"\b[a-z]+\b", text)
        n     = len(words)
        if n == 0:
            return 0.0

        w    = min(self.mattr_window, n)
        freq: dict[str, int] = {}

        for word in words[:w]:
            freq[word] = freq.get(word, 0) + 1

        ttr_sum, n_windows = len(freq) / w, 1

        for i in range(w, n):
            outgoing = words[i - w]
            freq[outgoing] -= 1
            if freq[outgoing] == 0:
                del freq[outgoing]

            incoming = words[i]
            freq[incoming] = freq.get(incoming, 0) + 1

            ttr_sum   += len(freq) / w
            n_windows += 1

        return ttr_sum / n_windows

    # ══════════════════════════════════════════════════════════════════════════
    #  Dimension 2 — Reasoning Depth
    # ══════════════════════════════════════════════════════════════════════════

    def _score_reasoning_depth(self, text: str) -> float:
        bloom_level    = self._get_bloom_level(text)
        bloom_score    = self.BLOOM_SCORES[bloom_level]
        multihop_score = self._score_multihop(text)
        negation_score = self._score_negation(text)

        return (
            self.REASONING_DEPTH_WEIGHTS["bloom"]    * bloom_score
            + self.REASONING_DEPTH_WEIGHTS["multihop"] * multihop_score
            + self.REASONING_DEPTH_WEIGHTS["negation"] * negation_score
        )

    def _get_bloom_level(self, text: str) -> str:
        for level, pattern in self.BLOOM_KEYWORDS:
            if re.search(pattern, text):
                return level
        return "L1_remember"

    def _score_multihop(self, text: str) -> float:
        for pattern, score in self.MULTI_HOP_PATTERNS:
            if pattern.search(text):
                return score
        return 0.0

    def _score_negation(self, text: str) -> float:
        for pattern, score in self.NEGATION_PATTERNS:
            if pattern.search(text):
                return score
        return 0.0

    # ══════════════════════════════════════════════════════════════════════════
    #  Dimension 3 — Tool Dependency
    # ══════════════════════════════════════════════════════════════════════════

    # def _score_tool_dependency(self, text: str) -> float:
    #     count = sum(
    #         1 for kws in self.TOOL_INDICATORS.values()
    #         if any(kw in text for kw in kws)
    #     )
    #     return {0: 0.0, 1: 3.0, 2: 5.0, 3: 7.0}.get(count, 10.0)


    STRONG_SIGNAL_SCORE = 1.0
    WEAK_SIGNAL_SCORE   = 0.3
    WEAK_SIGNAL_CAP     = 0.6   # at most 2 weak hits count per tool

    def _score_tool_dependency(self, text: str) -> float:
        """
        Confidence-weighted tool dependency score.

        Each tool contributes: tool_weight × confidence_score
        where confidence comes from strong/weak keyword tiers.
        Final score is normalized to 0–10.
        """
        total_confidence = 0.0
        matched_tools    = []

        for tool, cfg in self.TOOL_INDICATORS.items():
            strong_hit = any(kw in text for kw in cfg["strong"])
            weak_score = min(
                sum(self.WEAK_SIGNAL_SCORE for kw in cfg["weak"] if kw in text),
                self.WEAK_SIGNAL_CAP
            )

            if strong_hit:
                confidence = self.STRONG_SIGNAL_SCORE
            elif weak_score > 0:
                confidence = weak_score
            else:
                continue

            weighted = cfg["weight"] * confidence
            total_confidence += weighted
            matched_tools.append((tool, round(weighted, 3)))

        # log what matched (helpful for debugging)
        if matched_tools:
            for tool, score in matched_tools:
                print(f"    {tool:<16}: weighted_score = {score}")
        else:
            print(f"    No tools matched.")

        # normalize: empirically, 3+ fully-confident tools ≈ max complexity
        # clip to 0–10
        normalized = min(total_confidence / 3.0 * 10.0, 10.0)
        print(f"    Raw confidence: {total_confidence:.3f}  →  score = {normalized:.3f}")
        return normalized

    # ══════════════════════════════════════════════════════════════════════════
    #  Dimension 4 — Domain Breadth
    # ══════════════════════════════════════════════════════════════════════════

    def _score_domain_breadth(self, text: str) -> float:
        count = sum(
            1 for kws in self.DOMAINS.values()
            if any(kw in text for kw in kws)
        )
        return min(count * 2.0, 10.0)

    # ══════════════════════════════════════════════════════════════════════════
    #  Dimension 5 — Task Type
    # ══════════════════════════════════════════════════════════════════════════

    def _score_task_type(self, text: str) -> float:
        for keyword, score in self.TASK_TYPE_SCORES:
            if re.search(rf"\b{keyword}\b", text, re.I):
                return score
        return 1.0

    # ══════════════════════════════════════════════════════════════════════════
    #  Helpers
    # ══════════════════════════════════════════════════════════════════════════

    def _recompute_thresholds(self) -> None:
        if not self._score_history:
            self.thresholds = None
            return

        p_low, p_mid, p_high = self.threshold_percentiles
        history = sorted(self._score_history)

        def _percentile(data: list[float], p: int) -> float:
            idx = (p / 100) * (len(data) - 1)
            lo  = int(idx)
            hi  = min(lo + 1, len(data) - 1)
            return data[lo] + (data[hi] - data[lo]) * (idx - lo)

        self.thresholds = {
            "LLM":   _percentile(history, p_low),
            "COT": _percentile(history, p_mid),
            "SA":  _percentile(history, p_high),
        }

    def _complexity_band(self, score: float) -> str:
        if self.thresholds:
            if score < self.thresholds["LLM"]:   return "LLM"
            if score < self.thresholds["COT"]: return "CoT"
            if score < self.thresholds["SA"]:  return "SA"
            return "MA"

        if score < 3.0: return "LLM"
        if score < 5.5: return "COT"
        if score < 7.5: return "SA"
        return "MA"

    @staticmethod
    def _normalize_weights(weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total == 0:
            raise ValueError("Weights must not all be zero.")
        return {k: v / total for k, v in weights.items()}

    @staticmethod
    def _check_weights(weights: dict[str, float]) -> None:
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Weights must sum to 1.0, got {total:.6f}")

    @staticmethod
    def _validate_percentiles(percentiles: tuple) -> None:
        if len(percentiles) != 3:
            raise ValueError(f"Expected 3 percentiles, got {len(percentiles)}.")
        if not (0 < percentiles[0] < percentiles[1] < percentiles[2] < 100):
            raise ValueError(f"Percentiles must be strictly increasing within (0, 100). Got: {percentiles}.")


# ══════════════════════════════════════════════════════════════════════════════
#  CLI entry point — accepts a parquet or CSV file, writes results to JSON
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    from pathlib import Path
    import pandas as pd

    repo_root = Path(__file__).resolve().parents[1]
    input_path = repo_root / "datasets" / "golden" / "gaia.parquet"
    output_path = repo_root / "results1" / "unified_baseline" / "gaia" / "features_complexity.parquet"

    df = pd.read_parquet(input_path)

    queries = df["query"].dropna().astype(str).tolist()

    analyzer = TaskComplexityAnalyzer(threshold_percentiles=(20, 50, 90))

    # Fit first for accurate band thresholds, then analyze
    print("Fitting thresholds on corpus...")
    analyzer.fit(queries)
    print(f"Thresholds: {analyzer.thresholds}")

    print("Analyzing all queries...")
    results = analyzer.analyze_batch(queries)

    # Band distribution summary
    from collections import Counter
    band_counts = Counter(r["complexity_band"] for r in results)
    print("\nComplexity distribution:")
    for band, count in sorted(band_counts.items()):
        pct = 100 * count / len(results)
        print(f"  {band:<16}: {count:>5}  ({pct:.1f}%)")
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_parquet(output_path)
    print(f"\nResults saved to: {output_path}")