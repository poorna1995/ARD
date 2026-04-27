DEFAULT_WEIGHTS: dict[str, float] = {
    "task_length": 0.15,
    "num_requirements": 0.20,
    "domain_breadth": 0.20,
    "tool_requirements": 0.225,
    "reasoning_depth": 0.225,
}

# Sub-feature weights
TASK_LENGTH_SUBWEIGHTS: dict[str, float] = {"word_count": 0.60, "mattr": 0.40}
REQUIREMENTS_SUBWEIGHTS: dict[str, float] = {
    "requirement_count": 0.70,
    "temporal_ordering": 0.30,
}
DOMAIN_SUBWEIGHTS: dict[str, float] = {"domain_keywords": 0.70, "temporal_domain": 0.30}
REASONING_SUBWEIGHTS: dict[str, float] = {"bloom_level": 0.60, "multi_hop": 0.40}

# Domains
DOMAINS: dict[str, list[str]] = {
    "technical": ["code", "programming", "software", "algorithm", "debug", "implement", "script", "api"],
    "research": ["study", "research", "investigate", "analyze", "survey", "review", "literature"],
    "business": ["market", "sales", "revenue", "business", "strategy", "roi", "profit"],
    "creative": ["design", "create", "write", "compose", "generate", "draft", "brainstorm"],
    "data": ["data", "statistics", "analytics", "metrics", "dataset", "visualization", "analysis"],
    "scientific": ["experiment", "hypothesis", "theory", "scientific", "methodology", "findings"],
    "legal": ["legal", "law", "regulation", "compliance", "contract", "policy"],
    "financial": ["financial", "accounting", "budget", "investment", "forecast", "valuation"],
}

TEMPORAL_DOMAIN_INDICATORS: dict[str, list[str]] = {
    "historical": ["history", "historical", "past", "previously", "formerly", "ancient", "archival", "retrospective"],
    "forecasting": ["forecast", "predict", "projection", "future", "upcoming", "anticipated", "outlook", "trend"],
    "time_series": ["over time", "time series", "longitudinal", "quarterly", "annually", "year-on-year", "month-over-month", "periodic"],
    "scheduling": ["schedule", "deadline", "calendar", "timeline", "milestone", "roadmap", "sprint", "phase"],
    "duration": ["era", "decade", "century", "epoch", "period", "duration", "interval", "lifespan"],
}

TOOL_INDICATORS: dict[str, list[str]] = {
    "web_search": ["search", "find online", "look up", "google", "web"],
    "code_executor": ["run code", "execute", "test", "python", "script"],
    "calculator": ["calculate", "compute", "math", "equation", "formula"],
    "file_ops": ["file", "document", "read", "write", "save", "load"],
    "api": ["api", "request","github" "fetch data", "endpoint", "rest"],
    "database": ["database", "sql", "query", "table"],
    "image": ["image", "picture", "photo", "visualization", "chart", "graph"],
    "video": ["video", "movie", "stream", "multimedia"],
}

TEMPORAL_ORDERING_INDICATORS: dict[str, list[str]] = {
    "relative_time": ["before", "after", "during", "since", "until", "while", "when", "once", "as soon as"],
    "absolute_time": ["by january", "by february", "by march", "last year", "next month", "yesterday", "tomorrow", "in 20"],
    "ordering_words": ["first", "then", "next", "subsequently", "finally", "afterward", "chronological"],
    "conditional_time": ["if by", "no later than", "at most", "within", "in the next", "over the next"],
}

BLOOM_LEVELS: dict[str, list[str]] = {
    "L1_remember": [
        "list", "name", "recall", "define", "identify", "recognize", "recognise", "label", "match", "state",
        "what is", "who is", "when did", "where is", "show me", "give me", "tell me", "find the", "look up",
    ],
    "L2_understand": [
        "explain", "describe", "summarize", "summarise", "interpret", "paraphrase", "classify", "categorize",
        "illustrate", "outline", "how does", "what does", "why does", "what are", "clarify", "rephrase",
        "translate", "give an example",
    ],
    "L3_apply": [
        "apply", "use", "demonstrate", "implement", "execute", "solve", "calculate", "compute", "show how",
        "carry out", "perform", "produce", "build", "develop", "construct", "write code", "run", "deploy", "test",
    ],
    "L4_analyze": [
        "analyze", "analyse", "compare", "contrast", "differentiate", "examine", "break down", "distinguish",
        "deconstruct", "investigate", "dissect", "inspect", "correlate", "map out", "trace", "attribute", "infer",
    ],
    "L5_evaluate": [
        "evaluate", "assess", "critique", "judge", "justify", "argue", "defend", "recommend", "prioritize",
        "prioritise", "rank", "weigh", "decide", "measure", "validate", "verify", "rate", "appraise",
        "choose the best", "which is better",
    ],
    "L6_create": [
        "synthesize", "synthesise", "design", "create", "compose", "formulate", "plan", "generate",
        "construct a strategy", "architect", "devise", "invent", "produce a report", "write a", "draft a",
        "develop a plan", "create a framework", "end-to-end", "holistic", "comprehensive plan",
        "propose a solution", "innovate",
    ],
}

BLOOM_LEVEL_SCORES: dict[str, float] = {
    "L1_remember": 1.5,
    "L2_understand": 3.5,
    "L3_apply": 5.5,
    "L4_analyze": 7.0,
    "L5_evaluate": 8.5,
    "L6_create": 10.0,
}

MULTI_HOP_INDICATORS: dict[str, list[str]] = {
    "chain_connectors": ["therefore", "consequently", "thus", "hence", "as a result", "which means", "it follows", "this implies"],
    "conditional_logic": ["if", "given that", "assuming", "provided that", "in case", "only if"],
    "reference_chains": ["based on", "referring to", "from the above", "using the result", "with this", "from this"],
    "sequential_steps": ["step 1", "step 2", "first find", "then use", "finally apply", "now calculate"],
    "cross_domain_links": ["combine", "integrate", "merge results", "cross-reference", "reconcile", "correlate"],
}

TASK_TYPES: dict[str, list[str]] = {
    "boolean": ["is ", "does ", "can ", "are ", "will ", "should "],
    "retrieval": ["what is", "who is", "when did", "where is", "list all", "find the"],
    "multiple_choice": ["which of", "select ", "choose ", "pick the"],
    "generation": ["write", "draft", "compose", "generate", "create"],
    "reasoning": ["why ", "how does", "explain why", "reason ", "justify"],
    "synthesis": ["synthesize", "combine", "integrate findings", "consolidate", "create a comprehensive"],
}

_TASK_TYPE_SCORES: dict[str, float] = {
    "boolean": 1.0,
    "retrieval": 2.5,
    "multiple_choice": 3.5,
    "generation": 6.0,
    "reasoning": 7.5,
    "synthesis": 9.5,
    "general": 5.0,
}