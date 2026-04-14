"""
config.py
=========
Central configuration for the Task Complexity Estimator (TCE).

All numeric bounds, weight vectors, keyword signal dictionaries, and their
pre-compiled regex patterns live here so every feature layer imports from
one source of truth instead of duplicating literals.
"""

from __future__ import annotations

import math
import re
from typing import Dict, List, Tuple

# ─────────────────────────────────────────────────────────────────────────────
# L1 — Surface Features
# ─────────────────────────────────────────────────────────────────────────────

# Normalisation bounds (set from P5/P95 of benchmark data).
SURFACE_TC_LO:    float = 8.0
SURFACE_TC_HI:    float = 167.0
SURFACE_NER_LO:   float = 0.0
SURFACE_NER_HI:   float = 0.29
SURFACE_MATTR_LO: float = 0.7265
SURFACE_MATTR_HI: float = 1.0

# Sub-feature weights — must sum to 1.0
SURFACE_W_TC:    float = 0.45
SURFACE_W_NER:   float = 0.30
SURFACE_W_MATTR: float = 0.25


# MATTR sliding window size
MATTR_WINDOW: int = 25

# ─────────────────────────────────────────────────────────────────────────────
# L2 — Reasoning Depth
# ─────────────────────────────────────────────────────────────────────────────

REASONING_W: Dict[str, float] = {
    "bloom":     0.30,
    "syn_depth": 0.20,
    "multi_hop": 0.15,
    "negation":  0.10,
    "condition": 0.15,
    "modifier":  0.10,
}

# Normalisation ranges for each reasoning sub-feature
REASONING_DEPTH_LO:    float = 1.0
REASONING_DEPTH_HI:    float = 15.0
REASONING_NEG_LO:      int   = 0
REASONING_NEG_HI:      int   = 4
REASONING_COND_LO:     int   = 0
REASONING_COND_HI:     int   = 5
REASONING_MOD_LO:      float = 0.0
REASONING_MOD_HI:      float = 0.5

# Bloom verb lists per cognitive level (1 = Remember … 6 = Create)
BLOOM_VERBS: Dict[int, List[str]] = {
    1: [
        "define", "list", "recall", "name", "identify", "recognize", "state",
        "label", "match", "select", "locate", "arrange", "duplicate",
        "memorize", "repeat", "reproduce", "copy", "quote", "order", "record",
        "relate", "underline", "who is", "when was", "where is", "what year",
        "what is the capital",
    ],
    2: [
        "explain", "describe", "interpret", "paraphrase", "restate",
        "translate", "clarify", "elaborate", "simplify", "convert",
        "classify", "categorize", "sort", "group", "infer", "predict",
        "conclude from", "anticipate", "extrapolate", "summarize", "abstract",
        "generalize", "outline", "illustrate", "give an example", "represent",
        "compare", "report", "review", "discuss", "indicate", "rewrite",
        "what is the difference",
    ],
    3: [
        "apply", "use", "implement", "execute", "carry out", "perform",
        "administer", "operate", "employ", "utilize", "solve", "compute",
        "calculate", "estimate", "determine", "find", "derive the value",
        "work out", "demonstrate", "show how to", "illustrate how",
        "construct", "build", "produce", "make", "prepare", "complete",
        "modify", "develop a solution", "simulate", "run", "test", "sketch",
        "map", "write a function", "code", "script",
    ],
    4: [
        "analyze", "analyse", "differentiate", "distinguish", "discriminate",
        "separate", "decompose", "deconstruct", "break down", "parse",
        "dissect", "organize", "structure", "integrate", "diagram",
        "correlate", "attribute", "deduce", "detect", "examine",
        "investigate", "compare and contrast", "contrast", "examine differences",
        "compare the trade-offs", "trade-off", "trade-offs",
        "what are the trade-offs", "explain why", "what causes", "trace",
        "map the relationship", "identify assumptions", "identify biases",
        "debate", "prioritize",
    ],
    5: [
        "evaluate", "assess", "judge", "critique", "criticize",
        "review critically", "appraise", "rate", "rank", "check", "monitor",
        "validate", "verify", "debug", "judge whether", "measure", "inspect",
        "score", "justify", "defend", "argue for", "argue against", "support",
        "refute", "counter", "recommend", "decide", "choose between",
        "determine the best", "conclude", "is it better to", "should we",
        "which is more", "what is the optimal",
    ],
    6: [
        "design", "plan", "devise", "propose", "formulate", "architect",
        "blueprint", "create", "develop", "generate", "produce", "invent",
        "originate", "pioneer", "compose", "author", "draft", "synthesize",
        "combine", "assemble", "compile", "derive", "prove", "demonstrate that",
        "show that", "formally show", "from first principles", "from scratch",
        "hypothesize", "theorize", "conjecture", "postulate", "novel",
        "new approach", "new method", "original", "come up with",
        "propose a new", "imagine a new",
    ],
}

# Compile each Bloom level — longest phrases first to prevent partial matches
BLOOM_PATTERNS: Dict[int, re.Pattern] = {
    lvl: re.compile(
        r"\b(?:" + "|".join(
            re.escape(v) for v in sorted(verbs, key=len, reverse=True)
        ) + r")\b",
        re.IGNORECASE,
    )
    for lvl, verbs in BLOOM_VERBS.items()
}

# Conditional / causal tokens that signal hypothetical reasoning
CONDITIONAL_TOKENS: frozenset = frozenset([
    "if", "unless", "provided", "assuming", "given", "suppose", "suppose that",
    "in the event", "whenever", "only if", "even if", "as long as",
    "because", "since", "therefore", "thus", "hence", "consequently",
    "as a result", "due to", "owing to", "so that", "in order to",
])

# ─────────────────────────────────────────────────────────────────────────────
# L3 — Tool Dependency
# ─────────────────────────────────────────────────────────────────────────────

TOOL_SIGNALS: Dict[str, List[str]] = {
    "code_execution": [
        "run", "execute", "compile", "script", "program", "code", "function",
        "algorithm", "implement", "debug", "unit test", "pytest", "bash",
        "terminal", "command line", "cli", "shell", "subprocess", "notebook",
        "jupyter", "pip install", "npm", "yarn", "makefile", "dockerfile",
    "virtual env", "venv", "conda", "poetry", "lint", "format",
    "type check", "mypy", "black", "ruff",
    ],
    "web_search": [
        "search", "browse", "look up", "find online", "google", "bing",
        "retrieve", "latest", "current", "real-time", "live", "up-to-date",
        "news", "fetch", "scrape", "crawl", "url", "website", "webpage",
        ".com", ".org", ".in", "wikipedia", "wiki", "stackoverflow", "reddit", "docs",
        "documentation", "arxiv", "paper", "research", "source",
    ],
    "file_io": [
        "read file", "write file", "open file", "save to", "load from", "csv",
        "json", "excel", "pdf", "word doc", "docx", "txt", "download",
        "upload", "import", "export", "parse document", "extract from",
        "dataset",     "yaml", "yml", "xml", "toml", "parquet", "feather", "hdf5",
        "zip", "tar", "gz", "folder", "directory", "s3", "blob",
        "gcs", "azure storage", "gdrive",
    ],
    
    "api_call": [
        "api", "rest", "graphql", "endpoint", "request", "response", "http",
        "https", "curl", "webhook", "oauth", "token", "authenticate",
        "authorize", "call service", "microservice", "grpc",
    ],
    "database": [
        "database", "sql", "query", "select", "insert", "update", "delete",
        "join", "table", "schema", "mongodb", "postgres", "mysql", "sqlite",
        "redis", "elasticsearch", "nosql", "supabase", "firebase", "dynamodb", "cassandra", "neo4j",
        "graph database", "migration", "orm", "sqlalchemy", "prisma",
        "vector database",
    ],
    "external_service": [
        "slack", "email", "calendar", "send message", "notify", "alert",
        "github", "jira", "trello", "confluence", "zapier", "ifttt",
        "gpt", "llm", "openai", "anthropic", "claude", "gemini",
    ],
    "math_computation": [
        "calculate", "compute", "integral", "derivative", "matrix",
        "solve equation", "eigenvalue", "optimize", "gradient", "probability",
        "statistics", "regression", "fourier", "laplace",
    ],
    "multimodal": [
        "image", "photo", "picture", "diagram", "chart", "graph", "plot",
        "video", "audio", "speech", "transcribe", "ocr", "caption",
        "visualize",
    ],
}

TOOL_MAX_CATS: int = 4  # 4+ categories → T = 1.0

# Compile tool patterns — longest terms first
TOOL_PATTERNS: Dict[str, re.Pattern] = {
    cat: re.compile(
        r"\b(?:" + "|".join(
            re.escape(t) for t in sorted(terms, key=len, reverse=True)
        ) + r")\b",
        re.IGNORECASE,
    )
    for cat, terms in TOOL_SIGNALS.items()
}

# ─────────────────────────────────────────────────────────────────────────────
# L4 — Domain Skills
# ─────────────────────────────────────────────────────────────────────────────

DOMAIN_SIGNALS: Dict[str, List[str]] = {
    "mathematics": [
        "theorem", "proof", "lemma", "calculus", "algebra", "geometry",
        "topology", "number theory", "combinatorics", "probability",
        "statistics", "integral", "derivative", "polynomial", "matrix",
        "vector space", "differential equation",
    ],
    "computer_science": [
        "algorithm", "data structure", "complexity", "big-o", "recursion",
        "sorting", "graph", "tree", "dynamic programming", "machine learning",
        "neural network", "transformer", "attention", "gradient",
        "backpropagation", "compiler", "operating system", "concurrency",
        "parallelism", "distributed", "blockchain",
    ],
    "science": [
        "physics", "chemistry", "biology", "quantum", "thermodynamics",
        "genetics", "protein", "molecule", "atom", "electron", "cell", "dna",
        "rna", "enzyme", "evolution", "ecology", "astronomy", "relativity",
        "nuclear",
    ],
    "medicine": [
        "diagnosis", "symptom", "treatment", "drug", "clinical", "patient",
        "surgery", "pathology", "pharmacology", "dosage", "side effect",
        "trial", "prognosis", "cancer", "diabetes", "blood pressure",
    ],
    "law": [
        "statute", "regulation", "legal", "law", "court", "jurisdiction",
        "precedent", "contract", "liability", "intellectual property",
        "gdpr", "compliance", "patent", "copyright", "tort",
    ],
    "finance": [
        "stock", "bond", "option", "derivative", "portfolio", "hedge",
        "equity", "valuation", "balance sheet", "income statement",
        "cash flow", "roi", "interest rate", "inflation", "gdp",
        "monetary policy",
    ],
    "history_culture": [
        "war", "empire", "revolution", "civilization", "ancient", "medieval",
        "renaissance", "colonialism", "culture", "philosophy", "religion",
        "mythology", "archaeology", "linguistics",
    ],
    "engineering": [
        "circuit", "electrical", "mechanical", "structural", "thermal",
        "fluid", "signal processing", "control system", "robotics",
        "embedded", "fpga", "microcontroller", "antenna", "semiconductor",
    ],
}

DOMAIN_MAX_CATS:   int   = 4   # 4+ domains  → D_domain   = 1.0
DOMAIN_MAX_TEMP:   int   = 3   # 3+ patterns → D_temporal = 1.0
DOMAIN_W_DOMAIN:   float = 0.65
DOMAIN_W_TEMPORAL: float = 0.35

DOMAIN_PATTERNS: Dict[str, re.Pattern] = {
    dom: re.compile(
        r"\b(?:" + "|".join(
            re.escape(k) for k in sorted(kw, key=len, reverse=True)
        ) + r")\b",
        re.IGNORECASE,
    )
    for dom, kw in DOMAIN_SIGNALS.items()
}

# 4 distinct temporal signal patterns
TEMPORAL_PATTERNS: List[re.Pattern] = [
    re.compile(r"\b(\d{4}|\d{1,2}/\d{1,2}/\d{2,4})\b"),
    re.compile(
        r"\b(before|after|since|until|during|between|by"
        r"|\d+\s*(years?|months?|days?|decades?))\b",
        re.I,
    ),
    re.compile(
        r"\b(historical|future|predict|forecast|trend|evolve|timeline)\b",
        re.I,
    ),
    re.compile(r"\b(current|latest|recent|today|now|as of)\b", re.I),
]

# ─────────────────────────────────────────────────────────────────────────────
# L5 — Task Type
# ─────────────────────────────────────────────────────────────────────────────

# Each rule is (type_name, score, compiled_pattern).
# Ordered from highest to lowest complexity — first match wins.
TASK_TYPE_RULES: List[Tuple[str, float, re.Pattern]] = [
    (
        "generative", 1.0,
        re.compile(
            r"\b(write|compose|draft|generate|create|design|build|develop"
            r"|produce|author|invent|synthesize|formulate|devise|architect"
            r"|plan a|propose a|construct|make a|come up with)\b",
            re.I,
        ),
    ),
    (
        "reasoning", 0.85,
        re.compile(
            r"\b(prove|derive|show that|explain why|analyze|analyse|justify"
            r"|evaluate|assess|critique|reason|investigate|demonstrate that"
            r"|how does|why does|what causes|what leads to|infer|conclude)\b",
            re.I,
        ),
    ),
    (
        "alt_choice", 0.5,
        re.compile(
            r"\b(which is better|compare|choose between|select the best"
            r"|best option|trade-off|versus|vs\.?|alternatives|should i use"
            r"|recommend|rank|what would you prefer|which approach)\b",
            re.I,
        ),
    ),
    (
        "single_answer", 0.3,
        re.compile(
            r"\b(what is|who is|where is|when was|what year|how many|how much"
            r"|what was|name the|list the|define|what does .* mean"
            r"|capital of)\b",
            re.I,
        ),
    ),
    (
        "boolean_decision", 0.2,
        re.compile(
            r"\b(is it|is there|are there|does .* exist|can .* be|is .* true"
            r"|do .* have|has .* ever|was .* ever|would .* work"
            r"|is .* possible)\b",
            re.I,
        ),
    ),
]

TASK_TYPE_UNKNOWN_SCORE: float = 0.2

# ─────────────────────────────────────────────────────────────────────────────
# Composite Estimator Weights
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ESTIMATOR_WEIGHTS: Dict[str, float] = {
    "S":  0.15,   # Surface Features
    "R":  0.35,   # Reasoning Depth  ← dominant signal
    "T":  0.20,   # Tool Dependency
    "D":  0.15,   # Domain Skills
    "TT": 0.15,   # Task Type
}

# ─────────────────────────────────────────────────────────────────────────────
# Complexity Band Thresholds
# ─────────────────────────────────────────────────────────────────────────────

BAND_THRESHOLDS: List[Tuple[float, str]] = [
    (0.20, "LOW"),
    (0.45, "MEDIUM"),
    (0.60, "HIGH"),
    (1.01, "VERY HIGH"),
]

# ─────────────────────────────────────────────────────────────────────────────
# Pipeline / Dataset Configuration
# ─────────────────────────────────────────────────────────────────────────────

QUERY_COL_MAP: Dict[str, str] = {
    "gaia":      "query",
    "aime":      "query",
    "mmlu_pro":  "query",
    "musique":   "query",
    "swe_bench": "query",
}

EXPECTED_DIFFICULTY_RANK: Dict[str, int] = {
    "musique":   1,
    "mmlu_pro":  2,
    "aime":      3,
    "gaia":      4,
    "swe_bench": 5,
}

# Calibration percentile strategies
CALIBRATION_STRATEGIES: Dict[str, Tuple[int, int]] = {
    "P2_P98":  (2,  98),
    "P5_P95":  (5,  95),
    "P10_P90": (10, 90),
}

SURFACE_FEATURE_NAMES: List[str] = ["token_count", "ner_density", "mattr"]
SURFACE_ATTR_MAP: Dict[str, Tuple[str, str]] = {
    "token_count": ("SURFACE_TC_LO",    "SURFACE_TC_HI"),
    "ner_density": ("SURFACE_NER_LO",   "SURFACE_NER_HI"),
    "mattr":       ("SURFACE_MATTR_LO", "SURFACE_MATTR_HI"),
}