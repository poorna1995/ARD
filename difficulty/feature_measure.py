

# from __future__ import annotations

# import re
# import json
# import logging
# from dataclasses import dataclass, field
# from typing import Any

# logger = logging.getLogger(__name__)






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

#     def to_dict(self) -> dict:
#         return {
#             "overall":         self.overall,
#             "task_length":     self.task_length,
#             "reasoning_depth": self.reasoning_depth,
#             "tool_dependency": self.tool_dependency,
#             "domain_breadth":  self.domain_breadth,
#             "task_type":       self.task_type,
#             "complexity_band": self.complexity_band,
#         }


# # ══════════════════════════════════════════════════════════════════════════════
# #  Analyzer
# # ══════════════════════════════════════════════════════════════════════════════

# class TaskComplexityAnalyzer:

#     DEFAULT_WEIGHTS: dict[str, float] = {
#         "task_length":     0.12,
#         "reasoning_depth": 0.30,
#         "tool_dependency": 0.25,
#         "domain_breadth":  0.15,
#         "task_type":       0.18,
#     }

#     TASK_LENGTH_WEIGHTS: dict[str, float] = {
#         "word_count": 0.60,
#         "mattr":      0.40,
#     }

#     REASONING_DEPTH_WEIGHTS: dict[str, float] = {
#         "bloom":    0.50,
#         "multihop": 0.30,
#         "negation": 0.20,
#     }

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

#     MULTI_HOP_PATTERNS: list[tuple[re.Pattern, float]] = [
#         (re.compile(r"\bthen\b.+\bafter\b|\bafter\b.+\bthen\b",               re.I | re.S), 9.0),
#         (re.compile(r"\bfirst\b.+\bsecond\b.+\bthird\b",                      re.I | re.S), 8.5),
#         (re.compile(r"\bbecause\b.+\btherefore\b|\bsince\b.+\bthus\b",        re.I | re.S), 8.0),
#         (re.compile(r"\bif\b.+\bthen\b",                                       re.I | re.S), 7.5),
#         (re.compile(r"\bdepends on\b|\brequires\b.+\bbefore\b",               re.I | re.S), 7.0),
#         (re.compile(r"\balthough\b.+\bnevertheless\b|\bdespite\b.+\bstill\b", re.I | re.S), 7.0),
#         (re.compile(r"\bbefore\b.+\bafter\b|\bprior to\b",                    re.I | re.S), 6.5),
#         (re.compile(r"\bunless\b|\buntil\b.+\bthen\b",                        re.I | re.S), 6.5),
#         (re.compile(r"\bgiven that\b|\bassuming\b|\bprovided that\b",          re.I),        6.0),
#         (re.compile(r"\bin order to\b|\bso that\b|\bsuch that\b",             re.I),        6.0),
#         (re.compile(r"\bonce\b.+\bthen\b|\bwhen\b.+\bthen\b",                re.I | re.S), 6.0),
#         (re.compile(r"\bas a result\b|\bconsequently\b|\btherefore\b",        re.I),        5.5),
#         (re.compile(r"\bif\b",                                                 re.I),        5.0),
#         (re.compile(r"\botherwi?se\b",                                         re.I),        5.0),
#         (re.compile(r"\bsubsequently\b|\bafterward(s)?\b",                    re.I),        4.5),
#         (re.compile(r"\bhowever\b|\bnonetheless\b|\byet\b",                   re.I),        4.0),
#         (re.compile(r"\bfirst\b.+\bthen\b|\bstep \d\b",                      re.I | re.S), 4.0),
#         (re.compile(r"\bfinally\b|\blastly\b|\bin the end\b",                 re.I),        3.5),
#     ]

#     NEGATION_PATTERNS: list[tuple[re.Pattern, float]] = [
#         (re.compile(r"\bnot\b.+\bwithout\b|\bnever\b.+\bunless\b",            re.I | re.S), 9.0),
#         (re.compile(r"\bif not\b|\bif no\b|\bif neither\b",                   re.I),        8.0),
#         (re.compile(r"\bexcept\b.+\bif\b|\bexcept\b.+\bwhen\b",              re.I | re.S), 7.5),
#         (re.compile(r"\bunless\b.+\botherwise\b",                             re.I | re.S), 7.5),
#         (re.compile(r"\bneither\b.+\bnor\b",                                  re.I | re.S), 7.0),
#         (re.compile(r"\bhad not\b|\bwere not\b|\bwould not\b.+\bif\b",        re.I | re.S), 7.0),
#         (re.compile(r"\bwithout\b.+\bwould\b|\bwithout\b.+\bcould\b",         re.I | re.S), 6.5),
#         (re.compile(r"\bdo not\b.+\buntil\b|\bdon'?t\b.+\buntil\b",          re.I | re.S), 6.0),
#         (re.compile(r"\bdo not\b.+\bbefore\b|\bdon'?t\b.+\bbefore\b",         re.I | re.S), 6.0),
#         (re.compile(r"\bnot\b.+\bbut\b|\bnot\b.+\binstead\b",                re.I | re.S), 5.5),
#         (re.compile(r"\brather than\b|\binstead of\b",                        re.I),        4.5),
#         (re.compile(r"\bcannot\b|\bcan'?t\b|\bwon'?t\b|\bshouldn'?t\b",      re.I),        3.5),
#         (re.compile(r"\bno\b|\bnot\b|\bnever\b|\bnone\b",                     re.I),        2.0),
#     ]

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

#     # TOOL_INDICATORS: dict[str, list[str]] = {
#     #     "web_search":    ["search", "find online", "look up", "google", "web",
#     #     "latest", "current", "today", "recent", "breaking",
#     #     "news", "update", "live", "right now", "real time"],
#     #     "code_executor": [ "run code", "execute", "test", "python", "script",
#     #     "debug", "runtime", "output", "compile", "sql"],
#     #     "calculator":    ["calculate", "compute", "math", "equation", "formula"],
#     #     "file_ops":      ["file", "document", "read", "write", "save", "load",
#     #     "open file", "export", "import"],
#     #     "api":           ["api", "request", "fetch data", "endpoint", "rest",
#     #     "real-time data", "live data", "stream data", "webhook"]
#     #     "database":      ["database", "sql", "query", "table", "data"],
#     #     "image":         ["image", "picture", "photo", "visualization", "chart", "graph"],
#     #     "video":         ["video", "movie", "stream", "multimedia"],
#     # }

#     TOOL_INDICATORS: dict[str, dict] = {
#     "web_search": {
#         "weight": 1.0,          # how much this tool adds to complexity
#         "strong": [             # high-confidence signals (score: 1.0 each)
#             "search the web", "find online", "look it up",
#             "breaking news", "real time", "live data",
#             "latest news", "current events"
#         ],
#         "weak": [               # low-confidence signals (score: 0.3 each)
#             "search", "latest", "current", "recent",
#             "today", "news", "update",
#             "wikipedia", "wiki", "wikipedia page", "wiki page"
#         ],
#     },

#     "code_executor": {
#         "weight": 1.4,          # running code is more complex than a search
#         "strong": [
#             "run code", "execute script", "run this python",
#             "debug this", "compile and run", "run the program"
#         ],
#         "weak": [
#             "python", "script", "execute", "runtime", "output"
#         ],
#     },

#     "calculator": {
#         "weight": 0.8,
#         "strong": [
#             "calculate", "solve the equation", "evaluate the formula",
#             "compute the result", "numerical solution"
#         ],
#         "weak": [
#             "compute", "math", "equation", "formula", "solve"
#         ],
#     },

#     "file_ops": {
#         "weight": 0.9,
#         "strong": [
#             "open file", "read the file", "write to file",
#             "save to disk", "export file", "load from file"
#         ],
#         "weak": [
#             "file", "document", "save", "load", "export", "import"
#         ],
#     },

#     "api": {
#         "weight": 1.3,
#         "strong": [
#             "call the api", "api endpoint", "rest api",
#             "fetch from api", "webhook", "api request",
#             "live data feed", "stream data"
#         ],
#         "weak": [
#             "api", "endpoint", "request", "rest", "fetch"
#         ],
#     },

#     "database": {
#         "weight": 1.2,
#         "strong": [
#             "sql query", "run a query", "fetch records",
#             "insert into", "update row", "retrieve from database",
#             "join tables"
#         ],
#         "weak": [
#             "database", "sql", "table", "query"
#         ],
#     },

#     "image": {
#         "weight": 0.9,
#         "strong": [
#             "generate image", "plot the graph", "create chart",
#             "visualize the data", "render diagram", "draw a graph"
#         ],
#         "weak": [
#             "image", "chart", "graph", "plot", "diagram", "photo"
#         ],
#     },

#     "video": {
#         "weight": 1.1,
#         "strong": [
#             "live stream", "broadcast", "video call",
#             "record video", "stream this"
#         ],
#         "weak": [
#             "video", "movie", "stream", "multimedia"
#         ],
#     },
# }


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
#     def __init__(
#         self,
#         weights: dict[str, float] | None = None,
#         threshold_percentiles: tuple[int, int, int] = (30, 70, 90),
#         mattr_window: int = 15,
#         verbose: bool = False,
#     ) -> None:
#         self._validate_percentiles(threshold_percentiles)
#         raw        = weights if weights is not None else dict(self.DEFAULT_WEIGHTS)
#         normalized = self._normalize_weights(raw)
#         self._check_weights(normalized)
#         self.weights               = normalized
#         self.threshold_percentiles = threshold_percentiles
#         self.thresholds: dict[str, float] | None = None
#         self.mattr_window          = mattr_window
#         self.verbose               = verbose
#         self._score_history: list[float] = []
#     def analyze(self, query: str) -> TaskComplexityScore:
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

#         self._score_history.append(overall)
#         self._recompute_thresholds()

#         band = self._complexity_band(overall)

#         return TaskComplexityScore(
#             overall         = round(overall, 3),
#             task_length     = round(task_length, 3),
#             reasoning_depth = round(reasoning_depth, 3),
#             tool_dependency = round(tool_dependency, 3),
#             domain_breadth  = round(domain_breadth, 3),
#             task_type       = round(task_type, 3),
#             complexity_band = band,
#             breakdown       = scores,
#         )

#     def fit(self, queries: list[str]) -> "TaskComplexityAnalyzer":
#         """Pre-compute thresholds from a reference corpus."""
#         for q in queries:
#             text   = q.lower()
#             scores = {
#                 "task_length":     self._score_task_length(text),
#                 "reasoning_depth": self._score_reasoning_depth(text),
#                 "tool_dependency": self._score_tool_dependency(text),
#                 "domain_breadth":  self._score_domain_breadth(text),
#                 "task_type":       self._score_task_type(text),
#             }
#             overall = sum(self.weights[dim] * s for dim, s in scores.items())
#             self._score_history.append(overall)

#         self._recompute_thresholds()
#         return self

#     def analyze_batch(self, queries: list[str]) -> list[dict]:
#         """Analyze a list of queries, return list of result dicts."""
#         results = []
#         for q in queries:
#             score = self.analyze(q)
#             d = score.to_dict()
#             d["query"] = q
#             results.append(d)
#         return results

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 1 — Task Length
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_task_length(self, text: str) -> float:
#         word_count  = self._compute_word_count(text)
#         wc_score    = self._score_word_count(word_count)
#         mattr       = self._compute_mattr(text)
#         mattr_score = mattr * 10.0

#         return (
#             self.TASK_LENGTH_WEIGHTS["word_count"] * wc_score
#             + self.TASK_LENGTH_WEIGHTS["mattr"]    * mattr_score
#         )

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
#     #  Dimension 2 — Reasoning Depth
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_reasoning_depth(self, text: str) -> float:
#         bloom_level    = self._get_bloom_level(text)
#         bloom_score    = self.BLOOM_SCORES[bloom_level]
#         multihop_score = self._score_multihop(text)
#         negation_score = self._score_negation(text)

#         return (
#             self.REASONING_DEPTH_WEIGHTS["bloom"]    * bloom_score
#             + self.REASONING_DEPTH_WEIGHTS["multihop"] * multihop_score
#             + self.REASONING_DEPTH_WEIGHTS["negation"] * negation_score
#         )

#     def _get_bloom_level(self, text: str) -> str:
#         for level, pattern in self.BLOOM_KEYWORDS:
#             if re.search(pattern, text):
#                 return level
#         return "L1_remember"

#     def _score_multihop(self, text: str) -> float:
#         for pattern, score in self.MULTI_HOP_PATTERNS:
#             if pattern.search(text):
#                 return score
#         return 0.0

#     def _score_negation(self, text: str) -> float:
#         for pattern, score in self.NEGATION_PATTERNS:
#             if pattern.search(text):
#                 return score
#         return 0.0

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 3 — Tool Dependency
#     # ══════════════════════════════════════════════════════════════════════════

#     # def _score_tool_dependency(self, text: str) -> float:
#     #     count = sum(
#     #         1 for kws in self.TOOL_INDICATORS.values()
#     #         if any(kw in text for kw in kws)
#     #     )
#     #     return {0: 0.0, 1: 3.0, 2: 5.0, 3: 7.0}.get(count, 10.0)


#     STRONG_SIGNAL_SCORE = 1.0
#     WEAK_SIGNAL_SCORE   = 0.3
#     WEAK_SIGNAL_CAP     = 0.6   # at most 2 weak hits count per tool

#     def _score_tool_dependency(self, text: str) -> float:
#         """
#         Confidence-weighted tool dependency score.

#         Each tool contributes: tool_weight × confidence_score
#         where confidence comes from strong/weak keyword tiers.
#         Final score is normalized to 0–10.
#         """
#         total_confidence = 0.0
#         matched_tools    = []

#         for tool, cfg in self.TOOL_INDICATORS.items():
#             strong_hit = any(kw in text for kw in cfg["strong"])
#             weak_score = min(
#                 sum(self.WEAK_SIGNAL_SCORE for kw in cfg["weak"] if kw in text),
#                 self.WEAK_SIGNAL_CAP
#             )

#             if strong_hit:
#                 confidence = self.STRONG_SIGNAL_SCORE
#             elif weak_score > 0:
#                 confidence = weak_score
#             else:
#                 continue

#             weighted = cfg["weight"] * confidence
#             total_confidence += weighted
#             matched_tools.append((tool, round(weighted, 3)))

#         # log what matched (helpful for debugging)
#         if matched_tools:
#             for tool, score in matched_tools:
#                 print(f"    {tool:<16}: weighted_score = {score}")
#         else:
#             print(f"    No tools matched.")

#         # normalize: empirically, 3+ fully-confident tools ≈ max complexity
#         # clip to 0–10
#         normalized = min(total_confidence / 3.0 * 10.0, 10.0)
#         print(f"    Raw confidence: {total_confidence:.3f}  →  score = {normalized:.3f}")
#         return normalized

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 4 — Domain Breadth
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_domain_breadth(self, text: str) -> float:
#         count = sum(
#             1 for kws in self.DOMAINS.values()
#             if any(kw in text for kw in kws)
#         )
#         return min(count * 2.0, 10.0)

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Dimension 5 — Task Type
#     # ══════════════════════════════════════════════════════════════════════════

#     def _score_task_type(self, text: str) -> float:
#         for keyword, score in self.TASK_TYPE_SCORES:
#             if re.search(rf"\b{keyword}\b", text, re.I):
#                 return score
#         return 1.0

#     # ══════════════════════════════════════════════════════════════════════════
#     #  Helpers
#     # ══════════════════════════════════════════════════════════════════════════

#     def _recompute_thresholds(self) -> None:
#         if not self._score_history:
#             self.thresholds = None
#             return

#         p_low, p_mid, p_high = self.threshold_percentiles
#         history = sorted(self._score_history)

#         def _percentile(data: list[float], p: int) -> float:
#             idx = (p / 100) * (len(data) - 1)
#             lo  = int(idx)
#             hi  = min(lo + 1, len(data) - 1)
#             return data[lo] + (data[hi] - data[lo]) * (idx - lo)

#         self.thresholds = {
#             "raw":   _percentile(history, p_low),
#             "cot": _percentile(history, p_mid),
#             "react":  _percentile(history, p_high),
#         }

#     def _complexity_band(self, score: float) -> str:
#         if self.thresholds:
#             if score < self.thresholds["raw"]:   return "raw"
#             if score < self.thresholds["cot"]: return "cot"
#             if score < self.thresholds["react"]:  return "react"
#             return "multiagent"

#         if score < 3.0: return "raw"
#         if score < 5.5: return "cot"
#         if score < 7.5: return "react"
#         return "multiagent"

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
#     def _validate_percentiles(percentiles: tuple) -> None:
#         if len(percentiles) != 3:
#             raise ValueError(f"Expected 3 percentiles, got {len(percentiles)}.")
#         if not (0 < percentiles[0] < percentiles[1] < percentiles[2] < 100):
#             raise ValueError(f"Percentiles must be strictly increasing within (0, 100). Got: {percentiles}.")



# if __name__ == "__main__":
#     import sys
#     from pathlib import Path
#     import pandas as pd

#     repo_root = Path(__file__).resolve().parents[1]
#     input_path = repo_root / "datasets" / "golden" / "gaia.parquet"
#     output_path = repo_root / "results1" / "unified_baseline" / "gaia" / "features_complexity.parquet"

#     df = pd.read_parquet(input_path)

#     queries = df["query"].dropna().astype(str).tolist()

#     analyzer = TaskComplexityAnalyzer(threshold_percentiles=(20, 50, 90))

#     # Fit first for accurate band thresholds, then analyze
#     print("Fitting thresholds on corpus...")
#     analyzer.fit(queries)
#     thresholds = analyzer.thresholds
#     print(f"Thresholds: {analyzer.thresholds}")

#     print("Analyzing all queries...")
#     results = analyzer.analyze_batch(queries)

#     # Band distribution summary
#     from collections import Counter
#     band_counts = Counter(r["complexity_band"] for r in results)
#     print("\nComplexity distribution:")
#     for band, count in sorted(band_counts.items()):
#         pct = 100 * count / len(results)
#         print(f"  {band:<16}: {count:>5}  ({pct:.1f}%)")
#     # Save results
#     output_path.parent.mkdir(parents=True, exist_ok=True)
#     df = pd.DataFrame(results)
#     df.to_parquet(output_path)
#     # thresholds to json file
#     thresholds_path = output_path.parent / "thresholds.json"
#     with open(thresholds_path, "w") as f:
#         json.dump(thresholds, f)
#     print(f"\nResults saved to: {output_path}")




from __future__ import annotations

import re
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class TaskComplexityScore:
    overall:         float
    task_length:     float
    reasoning_depth: float
    tool_dependency: float
    domain_breadth:  float
    task_type:       float
    # complexity_band: str

    breakdown:       dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in (
            "overall", "task_length", "reasoning_depth",
            "tool_dependency", "domain_breadth", "task_type", 
            # "complexity_band"
            
        )}


# ── Analyzer ──────────────────────────────────────────────────────────────────

class TaskComplexityAnalyzer:

    # ---------- weights ----------

    DEFAULT_WEIGHTS = {
        "task_length": 0.12, "reasoning_depth": 0.30,
        "tool_dependency": 0.25, "domain_breadth": 0.15, "task_type": 0.18,
    }
    TASK_LENGTH_WEIGHTS     = {"word_count": 0.60, "mattr": 0.40}
    REASONING_DEPTH_WEIGHTS = {"bloom": 0.50, "multihop": 0.30, "negation": 0.20}

    # ---------- bloom taxonomy ----------

    BLOOM_KEYWORDS = [
        ("L6_create",     r"\b(create|design|build|compose|formulate|invent|devise)\b"),
        ("L5_evaluate",   r"\b(evaluate|assess|judge|critique|recommend|justify|defend)\b"),
        ("L4_analyze",    r"\b(analyze|analyse|compare|contrast|differentiate|examine)\b"),
        ("L3_apply",      r"\b(apply|use|implement|demonstrate|calculate|solve|execute)\b"),
        ("L2_understand", r"\b(explain|describe|summarize|interpret|classify|paraphrase)\b"),
        ("L1_remember",   r"\b(list|recall|define|identify|name|state|who|what|when|where)\b"),
    ]
    BLOOM_SCORES = {
        "L1_remember": 1.0, "L2_understand": 3.0, "L3_apply":   5.0,
        "L4_analyze":  7.0, "L5_evaluate":   8.5, "L6_create": 10.0,
    }

    # ---------- multi-hop patterns ----------

    MULTI_HOP_PATTERNS = [
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

    # ---------- negation patterns ----------

    NEGATION_PATTERNS = [
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

    # ---------- domains ----------

    DOMAINS = {
        "technical":  ["code", "programming", "software", "algorithm", "debug", "implement", "script", "api"],
        "research":   ["study", "research", "investigate", "analyze", "survey", "review", "literature"],
        "business":   ["market", "sales", "revenue", "business", "strategy", "roi", "profit"],
        "creative":   ["design", "create", "write", "compose", "generate", "draft", "brainstorm"],
        "data":       ["data", "statistics", "analytics", "metrics", "dataset", "visualization", "analysis"],
        "scientific": ["experiment", "hypothesis", "theory", "scientific", "methodology", "findings"],
        "legal":      ["legal", "law", "regulation", "compliance", "contract", "policy"],
        "financial":  ["financial", "accounting", "budget", "investment", "forecast", "valuation"],
    }

    # ---------- tool indicators ----------

    TOOL_INDICATORS = {
        "web_search":    {"weight": 1.0, "strong": ["search the web", "find online", "look it up", "breaking news", "real time", "live data", "latest news", "current events"],           "weak": ["search", "latest", "current", "recent", "today", "news", "update", "wikipedia", "wiki"]},
        "code_executor": {"weight": 1.4, "strong": ["run code", "execute script", "run this python", "debug this", "compile and run", "run the program"],                                  "weak": ["python", "script", "execute", "runtime", "output"]},
        "calculator":    {"weight": 0.8, "strong": ["calculate", "solve the equation", "evaluate the formula", "compute the result", "numerical solution"],                               "weak": ["compute", "math", "equation", "formula", "solve"]},
        "file_ops":      {"weight": 0.9, "strong": ["open file", "read the file", "write to file", "save to disk", "export file", "load from file"],                                      "weak": ["file", "document", "save", "load", "export", "import"]},
        "api":           {"weight": 1.3, "strong": ["call the api", "api endpoint", "rest api", "fetch from api", "webhook", "api request", "live data feed", "stream data"],             "weak": ["api", "endpoint", "request", "rest", "fetch"]},
        "database":      {"weight": 1.2, "strong": ["sql query", "run a query", "fetch records", "insert into", "update row", "retrieve from database", "join tables"],                   "weak": ["database", "sql", "table", "query"]},
        "image":         {"weight": 0.9, "strong": ["generate image", "plot the graph", "create chart", "visualize the data", "render diagram", "draw a graph"],                         "weak": ["image", "chart", "graph", "plot", "diagram", "photo"]},
        "video":         {"weight": 1.1, "strong": ["live stream", "broadcast", "video call", "record video", "stream this"],                                                             "weak": ["video", "movie", "stream", "multimedia"]},
    }

    STRONG_SIGNAL_SCORE = 1.0
    WEAK_SIGNAL_SCORE   = 0.3
    WEAK_SIGNAL_CAP     = 0.6

    # ---------- task type ----------

    TASK_TYPE_SCORES = [
        ("compare", 8.0), ("evaluate", 7.5), ("analyze", 7.0),
        ("generate", 6.5), ("design", 6.5), ("write", 5.5),
        ("summarize", 4.5), ("list", 3.5), ("what", 3.0),
        ("who", 2.5), ("when", 2.0), ("where", 2.0),
    ]

    # ── Init ──────────────────────────────────────────────────────────────────

    def __init__(self,weights: dict[str, float] | None = None,threshold_percentiles: tuple[int, int, int] = (30, 70, 90),mattr_window: int = 15,
        verbose: bool = False) -> None:
        self._validate_percentiles(threshold_percentiles)
        normalized = self._normalize_weights(weights or dict(self.DEFAULT_WEIGHTS))
        self._check_weights(normalized)
        self.weights               = normalized
        self.threshold_percentiles = threshold_percentiles
        self.thresholds: dict[str, float] | None = None
        self.mattr_window          = mattr_window
        self.verbose               = verbose
        self._score_history: list[float] = []
        # ↓ add these two lines
        self._feature_history: dict[str, list[float]] = {
            "task_length": [], "reasoning_depth": [], "tool_dependency": [],
            "domain_breadth": [], "task_type": [],
        }
        self._query_scores: dict[str, float] = {}
        # verbose: bool = False) -> None:
        # self._validate_percentiles(threshold_percentiles)
        # normalized = self._normalize_weights(weights or dict(self.DEFAULT_WEIGHTS))
        # self._check_weights(normalized)
        # self.weights               = normalized
        # self.threshold_percentiles = threshold_percentiles
        # self.thresholds: dict[str, float] | None = None
        # self.mattr_window          = mattr_window
        # self.verbose               = verbose
        # self._score_history: list[float] = []

    # ── Public API ────────────────────────────────────────────────────────────

    # def analyze(self, query: str) -> TaskComplexityScore:
    #     text   = query.lower()
    #     scores = self._compute_scores(text)
    #     overall = sum(self.weights[d] * s for d, s in scores.items())

    #     self._score_history.append(overall)
    #     self._recompute_thresholds()

    #     return TaskComplexityScore(
    #         overall         = round(overall, 3),
    #         task_length     = round(scores["task_length"], 3),
    #         reasoning_depth = round(scores["reasoning_depth"], 3),
    #         tool_dependency = round(scores["tool_dependency"], 3),
    #         domain_breadth  = round(scores["domain_breadth"], 3),
    #         task_type       = round(scores["task_type"], 3),
    #         # complexity_band = self._complexity_band(overall),
    #         breakdown       = scores,
    #     )
    def analyze(self, query: str) -> TaskComplexityScore:
        text    = query.lower()
        scores  = self._compute_scores(text)
        overall = sum(self.weights[d] * s for d, s in scores.items())

        self._score_history.append(overall)
        for feat, val in scores.items():        # ↓ add this loop
            self._feature_history[feat].append(val)
        self._recompute_thresholds()

        return TaskComplexityScore(
            overall         = round(overall, 3),
            task_length     = round(scores["task_length"], 3),
            reasoning_depth = round(scores["reasoning_depth"], 3),
            tool_dependency = round(scores["tool_dependency"], 3),
            domain_breadth  = round(scores["domain_breadth"], 3),
            task_type       = round(scores["task_type"], 3),
            breakdown       = scores,
        )
    def fit(self, queries: list[str]) -> "TaskComplexityAnalyzer":
        self._query_scores: dict[str, float] = {}
        self._feature_history: dict[str, list[float]] = {
            "task_length": [], "reasoning_depth": [], "tool_dependency": [],
            "domain_breadth": [], "task_type": [],
        }

        for q in queries:
            scores  = self._compute_scores(q.lower())
            overall = sum(self.weights[d] * s for d, s in scores.items())

            self._score_history.append(overall)
            self._query_scores[q] = overall

            for feat, val in scores.items():          # ← record every feature
                self._feature_history[feat].append(val)

        self._recompute_thresholds()
        return self


    def _recompute_thresholds(self) -> None:
        if not self._score_history:
            self.thresholds = None
            return

        bands = ("raw", "cot", "react")
        pcts  = self.threshold_percentiles
        hist  = sorted(self._score_history)

        self.thresholds = {
            "overall": {
                band: self._percentile(hist, p)
                for band, p in zip(bands, pcts)
            }
        }

        # Only computed after fit() — guard with hasattr
        if hasattr(self, "_feature_history"):
            for feat, values in self._feature_history.items():
                if not values:
                    continue
                sorted_vals = sorted(values)
                self.thresholds[feat] = {
                    band: self._percentile(sorted_vals, p)
                    for band, p in zip(bands, pcts)
                }
                
    def get_score(self, query: str) -> float:
        """
        For routing. Cached for corpus queries, fresh + tracked for new ones.
        """
        if hasattr(self, "_query_scores") and query in self._query_scores:
            return self._query_scores[query]

        scores  = self._compute_scores(query.lower())
        overall = sum(self.weights[d] * s for d, s in scores.items())
        self._query_scores[query] = overall
        self._score_history.append(overall)
        self._recompute_thresholds()
        return overall

    def analyze_batch(self, queries: list[str]) -> list[dict]:
        return [{**self.analyze(q).to_dict(), "query": q} for q in queries]

    # ── Shared score builder ──────────────────────────────────────────────────

    def _compute_scores(self, text: str) -> dict[str, float]:
        return {
            "task_length":     self._score_task_length(text),
            "reasoning_depth": self._score_reasoning_depth(text),
            "tool_dependency": self._score_tool_dependency(text),
            "domain_breadth":  self._score_domain_breadth(text),
            "task_type":       self._score_task_type(text),
        }

    # ── Dimension 1 — Task Length ─────────────────────────────────────────────

    def _score_task_length(self, text: str) -> float:
        wc_score    = self._score_word_count(len(text.split()))
        mattr_score = self._compute_mattr(text) * 10.0
        return (self.TASK_LENGTH_WEIGHTS["word_count"] * wc_score
              + self.TASK_LENGTH_WEIGHTS["mattr"]      * mattr_score)

    @staticmethod
    def _score_word_count(n: int) -> float:
        if n < 10:  return 1.0
        if n < 60:  return 3.0
        if n < 150: return 5.0
        if n < 250: return 7.0
        return 10.0

    def _compute_mattr(self, text: str) -> float:
        words = re.findall(r"\b[a-z]+\b", text)
        n = len(words)
        if n == 0:
            return 0.0

        w    = min(self.mattr_window, n)
        freq: dict[str, int] = {}
        for word in words[:w]:
            freq[word] = freq.get(word, 0) + 1

        ttr_sum, n_windows = len(freq) / w, 1
        for i in range(w, n):
            out = words[i - w]
            freq[out] -= 1
            if freq[out] == 0:
                del freq[out]
            inc = words[i]
            freq[inc] = freq.get(inc, 0) + 1
            ttr_sum   += len(freq) / w
            n_windows += 1

        return ttr_sum / n_windows

    # ── Dimension 2 — Reasoning Depth ────────────────────────────────────────

    def _score_reasoning_depth(self, text: str) -> float:
        bloom_level = next(
            (lvl for lvl, pat in self.BLOOM_KEYWORDS if re.search(pat, text)),
            "L1_remember"
        )
        bloom_score    = self.BLOOM_SCORES[bloom_level]
        multihop_score = next((s for p, s in self.MULTI_HOP_PATTERNS if p.search(text)), 0.0)
        negation_score = next((s for p, s in self.NEGATION_PATTERNS  if p.search(text)), 0.0)

        return (self.REASONING_DEPTH_WEIGHTS["bloom"]    * bloom_score
              + self.REASONING_DEPTH_WEIGHTS["multihop"] * multihop_score
              + self.REASONING_DEPTH_WEIGHTS["negation"] * negation_score)

    # ── Dimension 3 — Tool Dependency ────────────────────────────────────────

    def _score_tool_dependency(self, text: str) -> float:
        total = 0.0
        for tool, cfg in self.TOOL_INDICATORS.items():
            strong = any(kw in text for kw in cfg["strong"])
            weak   = min(
                sum(self.WEAK_SIGNAL_SCORE for kw in cfg["weak"] if kw in text),
                self.WEAK_SIGNAL_CAP,
            )
            confidence = self.STRONG_SIGNAL_SCORE if strong else weak
            if confidence:
                total += cfg["weight"] * confidence
                logger.debug("  %-16s weighted = %.3f", tool, cfg["weight"] * confidence)

        normalized = min(total / 3.0 * 10.0, 10.0)
        logger.debug("  Raw confidence: %.3f  → score = %.3f", total, normalized)
        return normalized

    # ── Dimension 4 — Domain Breadth ─────────────────────────────────────────

    def _score_domain_breadth(self, text: str) -> float:
        count = sum(1 for kws in self.DOMAINS.values() if any(kw in text for kw in kws))
        return min(count * 2.0, 10.0)

    # ── Dimension 5 — Task Type ───────────────────────────────────────────────

    def _score_task_type(self, text: str) -> float:
        return next(
            (s for kw, s in self.TASK_TYPE_SCORES if re.search(rf"\b{kw}\b", text, re.I)),
            1.0,
        )

    # ── Threshold helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _percentile(data: list[float], p: int) -> float:
        idx = (p / 100) * (len(data) - 1)
        lo  = int(idx)
        hi  = min(lo + 1, len(data) - 1)
        return data[lo] + (data[hi] - data[lo]) * (idx - lo)

    # def _complexity_band(self, score: float) -> str:
    #     if self.thresholds:
    #         if score < self.thresholds["raw"]:   return "raw"
    #         if score < self.thresholds["cot"]:   return "cot"
    #         if score < self.thresholds["react"]: return "react"
    #         return "multiagent"
    #     # static fallback
    #     if score < 3.0: return "raw"
    #     if score < 5.5: return "cot"
    #     if score < 7.5: return "react"
    #     return "multiagent"

    # ── Weight validation ─────────────────────────────────────────────────────

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
            raise ValueError(f"Percentiles must be strictly increasing within (0,100). Got: {percentiles}.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    from collections import Counter
    from pathlib import Path
    import pandas as pd

    repo_root   = Path(__file__).resolve().parents[1]
    input_path  = repo_root / "datasets" / "golden" / "gaia.parquet"
    output_path = repo_root / "results1" / "unified_baseline" / "gaia" / "features_complexity.parquet"

    df      = pd.read_parquet(input_path)
    queries = df["query"].dropna().astype(str).tolist()

    analyzer = TaskComplexityAnalyzer(threshold_percentiles=(20, 50, 90))

    print("Fitting thresholds on corpus...")
    analyzer.fit(queries)
    print(f"Thresholds: {analyzer.thresholds}")

    print("Analyzing all queries...")
    results = [{"query": q, "overall": analyzer.get_score(q)} for q in queries]
    for r in results:
        print(r)

    # band_counts = Counter(r["complexity_band"] for r in results)
    # print("\nComplexity distribution:")
    # for band, count in sorted(band_counts.items()):
    #     print(f"  {band:<16}: {count:>5}  ({100 * count / len(results):.1f}%)")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_parquet(output_path)

    thresholds_path = output_path.parent / "thresholds.json"
    with open(thresholds_path, "w") as f:
        json.dump(analyzer.thresholds, f)

    print(f"\nResults saved to: {output_path}")




    # def fit(self, queries: list[str]) -> "TaskComplexityAnalyzer":
    #     """Pre-compute thresholds from a reference corpus."""
    #     for q in queries:
    #         scores  = self._compute_scores(q.lower())
    #         overall = sum(self.weights[d] * s for d, s in scores.items())
    #         self._score_history.append(overall)
    #     self._recompute_thresholds()
    #     return self

    # def fit(self, queries: list[str]) -> "TaskComplexityAnalyzer":
    #     self._query_scores = {}   # query text → overall score
        
    #     for q in queries:
    #         scores  = self._compute_scores(q.lower())
    #         overall = sum(self.weights[d] * s for d, s in scores.items())
    #         self._score_history.append(overall)
    #         self._query_scores[q] = overall   # ← store it

    #     self._recompute_thresholds()
    #     return self