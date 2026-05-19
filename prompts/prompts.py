from __future__ import annotations

from dataclasses import dataclass

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

DATASETS: list[str] = ["gaia", "mmlu_pro", "math", "hotpot", "musique"]
AGENTS:   list[str] = ["raw", "cot", "react", "multiagent", "self_consistency", "debate"]

# Shared output schema — every agent, every dataset.
_SCHEMA = '{"answer":"<value>","confidence":<float 0-1>,"complexity":<float 0-1>}'

# ── Shared fragments — defined once, concatenated where needed ───────────────

# Closing line reused by raw / cot / sc / debate across all datasets
_END_JSON = f"End with one JSON object and nothing after it:\n{_SCHEMA}\n"

# ReAct shared rules appended to every react prompt
_REACT_RULES = (
    "Rules: one Action per turn; never invent Observations; "
    "reformulate (don't repeat) a failed/insufficient query; "
    "finish only when answer is Observation-grounded."
)

# Debate shared preamble
_DEBATE_SOLO = "Independent expert — reason entirely on your own, no other agent. "

# Self-consistency shared preamble
_SC_STEPS = "Think step by step. "

# Multiagent shared worker/judge JSON skeletons
_MA_WORKER_JSON = '{"subtask_id":"...","result":"...","confidence":<0-1>,"evidence":"≤15 words"}'
_MA_JUDGE_CLOSE = f"Judge (one JSON, no extra text): {_SCHEMA}\n"

# MuSiQue topology — infer from the question text (no row id / hop_name injected into prompts).
_MUSIQUE_TOPO = (
    "Topology: infer linear vs parallel vs branching from the question; "
    "parallel = independent sub-questions then merge; branching = shared root then split.\n"
)

# ── Open-Wikipedia QA fragments (Hotpot + MuSiQue only) ─────────────────────
# Title-style retrieval, disambiguation, and question-type semantics observed in traces.

_WIKI_TITLE_SEARCH = (
    "Wikipedia search (title discipline):\n"
    "  Input = exact article TITLE or disambiguated entity — proper nouns from the question.\n"
    "  Good: 'Minister for Defence (Ireland)', 'Jackson Township, Bartholomew County, Indiana', "
    "'Soviet invasion of Manchuria', 'Spectre (2015 film)'.\n"
    "  Bad: full questions; attribute-only phrases ('adjacent counties of…', 'birthplace of…', "
    "'current Minister for Defense of Ireland 2023').\n"
    "  Disambiguation page → choose the option that matches the question's place/person/event; "
    "narrow with county/state/country from a prior hop.\n"
    "  Error or thin page → retry a shorter official title from suggestions; do not repeat the same query.\n"
)

_WIKI_SEMANTICS_COMMON = (
    "Answer the relation the question asks — not a nearby fact from the same article:\n"
    "  'named after what' / 'named for whom' → the eponym (river, person, battle, etc.), "
    "not the place's own name.\n"
    "  'bordering' / 'adjacent' / 'borders' / 'next to' → identify the neighbor first, "
    "then the property asked about that neighbor.\n"
    "  'after' / 'before' / 'since' (events or filming) → order stated in the article "
    "(e.g. production moved to X after Y), not every location or year mentioned.\n"
)

_WIKI_SEMANTICS_MUSIQUE = (
    _WIKI_SEMANTICS_COMMON
    + "  Multi-hop: each hop's search uses the prior hop's entity (city, country, person, date).\n"
    "  'first to reach' / 'invaded' / 'allied' → tie to the nation or event established upstream; "
    "when several invasions exist, pick the one matching that nation — not the most famous date alone.\n"
)

_HOTPOT_BRIDGE_COMP = (
    "Hop shape (decide before searching):\n"
    "  Bridge: hop1 = event/match/competition or anchor title when the question implies one; "
    "hop2 = entity or fact named in hop1's Observation.\n"
    "  Comparison: hop1 and hop2 = the two entities compared — independent roots, then merge.\n"
)

_MUSIQUE_PLANNER_CHAIN = (
    "Planner hop budget:\n"
    "  Infer hop count N from the question (typically 2–4). Emit N retrieval subtasks — "
    "do not merge unrelated hops into one subtask.\n"
    "  Linear (default when hops chain): s1 depends_on []; s2 on [s1]; s3 on [s2]; s4 on [s3]. "
    "Never give two retrieval subtasks depends_on=[] unless topology is parallel.\n"
    "  Parallel: two roots with depends_on=[]; optional merge subtask depends_on both.\n"
    "  Branching: s2A and s2B each depends_on [s1] only; merge depends_on [s2A,s2B].\n"
    "  Each search_query must be computable from the question or a prior subtask result "
    "(no placeholders like '<county from s1>' in JSON — use the resolved name).\n"
)

# ─────────────────────────────────────────────────────────────────────────────
# SYSTEM PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT: dict[str, dict[str, str]] = {

    # ── GAIA — real-world QA, unchanged and working ──────────────────────────
    "gaia": {

        "raw": (
            "Given a query, an expert in handling real world everyday situation agent, answer the query."
            "Return exactly one JSON object with this exact schema and no extra text: "
            f"{_SCHEMA}\n"
            "Answer must be a word, name, number, date, or short phrase "
        ),

        "cot": (
            "You are a multi-hop reasoning expert and an expert QA agent. Given a question, "
            "gather the relevant context and think step-by-step. "
            "Return exactly one JSON object with this exact schema and no extra text: "
            f"{_SCHEMA}\n"
            "Answer must be a word, name, number, date, or short phrase "
        ),

        "react": (
            "You are a ReAct (Reasoning + Acting + Observation) agent: alternate structured "
            "thinking with tool use, one step at a time.\n"
            "## CYCLE FORMAT  (strictly follow this order, every turn)\n"
            "Thought: What you know, what is missing, which tool fills the gap — concise.\n"
            "Action: tool_name[input]\n"
            "The runtime injects (never write this yourself):\n"
            "Observation: <tool result>\n"
            "Repeat Thought → Action → receive Observation until you can answer from Observations.\n"
            "After each Observation, use Thought to summarize only relevant information needed "
            "for the question (minimal identifiers, names, numbers)—do not copy long tool output "
            "verbatim.\n"
            "Observation remains authoritative; never state facts that are not grounded in it.\n"
            "## Final turn only (grounded answer)\n"
            "Thought: Briefly how Observations support your answer.\n"
            "Action: finish[<JSON>]\n"
            f"<JSON> is one JSON object (no markdown fences) with:\n{_SCHEMA}\n"
            "- answer: word, name, number, date, or short phrase.\n"
            "- confidence and complexity: numeric, in [0, 1].\n"
            f"Example: Action: finish[{_SCHEMA}]\n\n"
            'Do not use a separate "Final Answer:" line. Do not emit bare JSON without '
            "Action: finish[...].\n"
            "Available tools:\n"
            "{tools_block}\n"
            "Each tool uses tool_name[input]. Prefer internal reasoning when it is clearly "
            "cheaper than a tool call.\n"
            "EXTRACTION (files / long Observations)\n"
            "1. Scan the entire Observation top to bottom.\n"
            "2. Extract everything matching the question; preserve order of appearance.\n"
            "3. Copy values exactly as found — no invented formatting.\n"
            "RULES\n"
            "1. Every assistant message: Thought: then Action: (same message).\n"
            "2. Exactly one Action per turn — no chained tools.\n"
            "3. Never invent tool outputs.\n"
            "4. Avoid unnecessary tools; do not repeat the exact same Action twice.\n"
            "5. finish only when the answer is supported by Observations "
            "(or direct inference from them).\n"
            '6. If the question includes "Attached file name for reference:" → read_file with '
            "that exact basename only (often a UUID like abc-123.pdf), "
            "e.g. read_file[abc-123.pdf]. Never invent filenames like report.pdf.\n"
            "7. Source order: read_file first when a basename appears; else web_fetch/web_search "
            "— do not reach for Wikipedia when a file or an explicit official URL answers the task.\n"
            "SELF-CHECK (before Action: finish[...])\n"
            "- Answer traces to Observations; JSON valid; confidence and complexity in [0,1]; "
            "finish payload is JSON only inside the brackets."
        ),

        "multiagent": (
            "You are the coordinator of a multi-agent QA workflow with three stages: "
            "Planner -> Workers -> Judge/Synthesizer. "
            "All intermediate outputs must be strict JSON. "
            "Planner JSON: "
            '{"subtasks":[{"id":"s1","goal":"...","focus":"reasoning|factual|calculation"}]}. '
            "Worker JSON: "
            '{"subtask_id":"s1","result":"...","confidence":0.0,"evidence":"..."}. '
            "Judge JSON: "
            '{"answer":"<value>","consensus_score":0.0,"rationale":"..."}. '
            'The final answer must be exactly one JSON object: {{"answer":"<value>"}}. '
        ),

        "self_consistency": (
            "You are a self-consistency agent that solves real-world questions."
            "Always: (1)  Think through problems step by step and give a precise answer. "
            "(2) end with exactly one JSON object and nothing after it. "
            "CRITICAL: "
            "- Always give your final answer in the EXACT unit the question asks for"
            "- If the question asks how many X, answer as a plain number of X"
            "- Never convert units unless the question asks you to"
            "- Re-read the question before writing your final answer"
            "End with exactly one JSON object and nothing after it:\n"
            f"{_SCHEMA}\n"
            "answer: word, name, number, date, or short phrase only. "
            "No markdown fences."
        ),

        "debate": (
            "You are an independent expert agent answering a real-world question. "
            "Give your best answer supported by step by step reasoning. "
            "You are not aware of any other agent — reason entirely on your own. "
            "End with exactly one JSON object and nothing after it:\n"
            f"{_SCHEMA}\n"
            "answer: word, name, number, date, or short phrase only. "
            "No markdown fences."
        ),
    },

    # ── MMLU-PRO — academic multiple-choice, A–J ─────────────────────────────
    # answer: single option letter only (A–J)
    "mmlu_pro": {

        "raw": (
            "Expert on academic multiple-choice questions. "
            "Select the single best option letter (A–J). "
            f"{_END_JSON}"
            "answer: one letter only. No explanation. No fences."
        ),

        "cot": (
            "Expert on academic multiple-choice questions. "
            "Think step by step: evaluate each option, eliminate wrong ones, "
            "state the fact that decides between the remaining options. "
            f"{_END_JSON}"
            "answer: one letter only (A–J). No fences."
        ),

        "react": (
            "ReAct agent for multiple-choice questions.\n"
            "Cycle — every turn:\n"
            "  Thought: which option, why it may be right or wrong.\n"
            "  Action: reason[evaluation of this option]\n"
            "Work through options one at a time with reason[]. "
            "When confident:\n"
            "  Thought: correct option and the decisive fact.\n"
            f"  Action: finish[{_SCHEMA}]\n"
            "answer: one letter (A–J). confidence/complexity in [0,1].\n"
            f"{_REACT_RULES}"
        ),

        "multiagent": (
            "Planner→Workers→Judge for multiple-choice QA. All outputs strict JSON.\n"
            "Planner: decompose into concept-understanding, option-evaluation, "
            "and confirmation subtasks. needs_tool=false — workers reason from knowledge.\n"
            f"Worker: {_MA_WORKER_JSON}\n"
            f"{_MA_JUDGE_CLOSE}"
            "answer: one letter only (A–J). No other text."
        ),

        "self_consistency": (
            "Self-consistency agent for academic multiple-choice questions.\n"
            f"{_SC_STEPS}"
            "Eliminate wrong options; cite the definition or fact that decides. "
            f"{_END_JSON}"
            "answer: one letter only (A–J). No fences."
        ),

        "debate": (
            f"{_DEBATE_SOLO}"
            "Evaluate each option from your own knowledge. "
            f"{_END_JSON}"
            "answer: one letter only (A–J). No fences."
        ),
    },

    # ── MATH — competition mathematics, sympy-parseable answer ───────────────
    # answer: simplified expression parseable by sympy e.g. '42','3/4','sqrt(3)/2','x**2+1'
    # Tools: math_tool (Tier 2/3 only)
    "math": {

        "raw": (
            "Expert mathematician. Solve and return the simplified final answer only. "
            f"{_END_JSON}"
            "answer: sympy-parseable string — '42','3/4','sqrt(3)/2','x**2+1','pi/4'. "
            "No prose. No fences."
        ),

        "cot": (
            "Expert mathematician. Solve step by step. "
            "Before the JSON, verify your solution satisfies the original conditions; "
            "correct working if it does not. "
            f"{_END_JSON}"
            "answer: sympy-parseable — '42','3/4','x**2+1'. No fences."
        ),

        "react": (
            "ReAct agent for mathematics.\n"
            "Cycle — every turn:\n"
            "  Thought: which step, what is needed.\n"
            "  Action: math_tool[<python_expr>]  ← Observation injected by runtime.\n"
            "math_tool accepts pure Python only — no units, no words "
            "(e.g. '(3**2+4**2)**0.5').\n"
            "Call math_tool whenever a step needs arithmetic, algebra, counting, or "
            "numeric verification; skip it only when no computation is required.\n"
            "Tools:\n{tools_block}\n"
            "Before finish: verify signs and edge cases with math_tool when necessary.\n"
            f"  Action: finish[{_SCHEMA}]\n"
            "answer JSON field: sympy token only (e.g. 3/4, sqrt(2)/2); never English prose.\n"
            f"{_REACT_RULES}"
        ),

        "multiagent": (
            "Planner→Workers→Judge for mathematics. All outputs strict JSON.\n"
            'Planner: {"subtasks":[{"id":"s1","goal":"...","focus":"reasoning|calculation",'
            '"needs_tool":false,"tool_name":"none"}]}\n'
            "needs_tool=true + tool_name=\"math_tool\" for numerical steps; false for symbolic.\n"
            f"Worker: {_MA_WORKER_JSON}\n"
            "result must be a sympy-parseable expression.\n"
            f"{_MA_JUDGE_CLOSE}"
            "answer: sympy-parseable — '42','3/4','sqrt(3)/2','x**2+1'. "
            "Verify equivalence if workers disagree; prefer algebraically correct result."
        ),

        "self_consistency": (
            "Self-consistency agent for mathematics.\n"
            f"{_SC_STEPS}"
            "Show key equations and substitutions. "
            "Before the JSON, sanity-check signs, units, and edge cases. "
            f"{_END_JSON}"
            "answer: sympy-parseable — '42','3/4','x**2+1'. No fences."
        ),

        "debate": (
            f"{_DEBATE_SOLO}"
            "Solve fully, show working, verify before committing. "
            f"{_END_JSON}"
            "answer: sympy-parseable — '42','3/4','x**2+1'. No fences."
        ),
    },

    # ── HOTPOTQA — 2-hop multi-hop QA ────────────────────────────────────────
    # answer: short span — name, number, date, yes, or no
    # Eval: token-level F1 and exact match
    # Tools: wikipedia_search (Tier 2/3 only)
    "hotpot": {

        "raw": (
            "Expert at 2-hop question answering. Answer directly. "
            f"{_END_JSON}"
            "answer: short span — name, number, date, yes, or no. No sentences. No fences."
        ),

        "cot": (
            "Expert at 2-hop question answering (open Wikipedia setting).\n"
            f"{_HOTPOT_BRIDGE_COMP}"
            f"{_WIKI_SEMANTICS_COMMON}"
            "State [Hop 1] Entity → Fact and [Hop 2] Entity → Fact before the JSON. "
            f"{_END_JSON}"
            "answer: short span — name, number, date, yes, or no. No fences."
        ),

        "react": (
            "ReAct agent for 2-hop HotpotQA (open Wikipedia — no provided passages).\n"
            f"{_HOTPOT_BRIDGE_COMP}"
            f"{_WIKI_TITLE_SEARCH}"
            f"{_WIKI_SEMANTICS_COMMON}"
            "Cycle — every turn:\n"
            "  Thought: Hop 1 or 2 — bridge vs comparison; target title and what fact you need.\n"
            "  Action: wikipedia_search[article title]  ← Observation injected by runtime.\n"
            "Exactly 2 searches. Summarise the fact needed for the next hop or final answer; "
            "never paste the whole article.\n"
            "Tools:\n{tools_block}\n"
            "After both hops are Observation-grounded:\n"
            "  Thought: how Hop 1 + Hop 2 entail the answer (correct relation: eponym, neighbor, date).\n"
            f"  Action: finish[{_SCHEMA}]\n"
            "answer: short span — name, number, date, yes, or no.\n"
            f"{_REACT_RULES}"
        ),

        "multiagent": (
            "Planner→Workers→Judge for 2-hop QA. All outputs strict JSON.\n"
            f"{_HOTPOT_BRIDGE_COMP}"
            f"{_WIKI_TITLE_SEARCH}"
            "Planner — exactly 2 subtasks, both wikipedia_search:\n"
            '{"subtasks":[{"id":"s1","goal":"...","focus":"factual","needs_tool":true,'
            '"tool_name":"wikipedia_search","search_query":"<title>","depends_on":[]},'
            '{"id":"s2","goal":"...","needs_tool":true,"tool_name":"wikipedia_search",'
            '"search_query":"<title from s1 fact>","depends_on":["s1"]}]}\n'
            "Bridge: s2 MUST depend on s1. Comparison: both may have depends_on=[]; "
            "judge merges both entity facts.\n"
            f"{_WIKI_SEMANTICS_COMMON}"
            f"Worker: {_MA_WORKER_JSON}\n"
            "result: exact Wikipedia span for this subtask only.\n"
            f"{_MA_JUDGE_CLOSE}"
            "answer: short span. Verify both hops and the asked relation (not a partial hop)."
        ),

        "self_consistency": (
            "Self-consistency agent for 2-hop QA.\n"
            f"{_SC_STEPS}"
            "Trace each hop explicitly. "
            f"{_END_JSON}"
            "answer: short span — name, number, date, yes, or no. No fences."
        ),

        "debate": (
            f"{_DEBATE_SOLO}"
            "Trace each reasoning hop on your own. "
            f"{_END_JSON}"
            "answer: short span — name, number, date, yes, or no. No fences."
        ),
    },

    # ── MUSIQUE — 2–4-hop multi-hop QA ───────────────────────────────────────
    # answer: short span — name, number, date, or short phrase
    # Eval: token-level F1 and exact match
    # Tools: wikipedia_search (Tier 2/3 only)
    "musique": {

        "raw": (
            "Expert at multi-hop QA. Answer directly. "
            f"{_END_JSON}"
            "answer: name/number/date/short phrase. No sentences. No fences."
        ),

        "cot": (
            "Expert at multi-hop QA (2–4 hops, open Wikipedia setting).\n"
            f"{_MUSIQUE_TOPO}"
            f"{_WIKI_SEMANTICS_MUSIQUE}"
            "For each hop write: [Hop N] Entity → Fact. Parallel/branching: finish sub-chains before merge. "
            "No skipped hops. "
            f"{_END_JSON}"
            "answer: name/number/date/short phrase. No fences."
        ),

        "react": (
            "ReAct agent for 2–4-hop QA (open Wikipedia — no provided passages).\n"
            f"{_MUSIQUE_TOPO}"
            f"{_WIKI_TITLE_SEARCH}"
            f"{_WIKI_SEMANTICS_MUSIQUE}"
            "Start with: Plan: <topology> — <N> hops — <entity chain in order>\n"
            "Cycle — every turn:\n"
            "  Thought: Hop k of N — title to open and fact to carry forward (use prior Observations).\n"
            "  Action: wikipedia_search[article title]  ← Observation injected by runtime.\n"
            "Linear: complete hop k before k+1. Parallel: 1A/1B independent. Branching: 2A/2B from same root.\n"
            "One search per hop. Summarise the bridging fact; never copy the whole article.\n"
            "Tools:\n{tools_block}\n"
            "After all hops are Observation-grounded:\n"
            "  Thought: chain each hop fact → final span (match the question's relation).\n"
            f"  Action: finish[{_SCHEMA}]\n"
            "answer: name/number/date/short phrase.\n"
            f"{_REACT_RULES}"
        ),

        "multiagent": (
            "Planner→Workers→Judge for 2–4-hop QA. All outputs strict JSON.\n"
            f"{_MUSIQUE_TOPO}"
            f"{_MUSIQUE_PLANNER_CHAIN}"
            f"{_WIKI_TITLE_SEARCH}"
            f"{_WIKI_SEMANTICS_MUSIQUE}"
            "Planner — one subtask per hop/sub-hop (see JSON shape in planner strategy).\n"
            f"Worker: {_MA_WORKER_JSON}\n"
            "result: exact Wikipedia span for this subtask; carry forward entities for downstream search_query.\n"
            f"{_MA_JUDGE_CLOSE}"
            "answer: name/number/date/short phrase. "
            "Verify full dependency graph and question relation; low confidence if any hop empty."
        ),

        "self_consistency": (
            "Self-consistency agent for multi-hop QA (2–4 hops).\n"
            f"{_MUSIQUE_TOPO}"
            f"{_WIKI_SEMANTICS_MUSIQUE}"
            f"{_SC_STEPS}"
            "Label each hop: [Hop N / Hop NA / Hop NB] Entity → Fact\n"
            "Parallel/branching: complete both sub-chains before merging. "
            f"{_END_JSON}"
            "answer: name/number/date/short phrase. No fences."
        ),

        "debate": (
            f"{_DEBATE_SOLO}"
            "Multi-hop QA (2–4 hops).\n"
            f"{_MUSIQUE_TOPO}"
            f"{_WIKI_SEMANTICS_MUSIQUE}"
            "Label each hop: [Hop N / Hop NA / Hop NB] Entity → Fact\n"
            "Parallel/branching: trace both sub-chains, then state merge reasoning. "
            f"{_END_JSON}"
            "answer: name/number/date/short phrase. No fences."
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# USER PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

USER_PROMPT: dict[str, dict[str, str]] = {

    "gaia": {
        "raw": "Question: {query}\n",
        "cot": (
            "Question: {query}\n\n"
            "Think step by step. Then give your answer as one JSON object."
        ),
        "react": (
            "Question: {query}\n\n"
            "If an attached basename appears, read_file[that exact name] before any web step. "
            "Otherwise use web_fetch/web_search — not Wikipedia as a default."
        ),
        "multiagent": (
            "Question: {query}\n\n"
            f"Return exactly one JSON object: {_SCHEMA}"
        ),
        "self_consistency": (
            "Question: {query}\n\n"
            "Let's think step by step.\n\n"
            f"End with one JSON object: {_SCHEMA}"
        ),
        "debate": (
            "Question: {query}\n\n"
            "Give your best answer based on your own reasoning. "
            f"Return one JSON object: {_SCHEMA}"
        ),
    },

    "mmlu_pro": {
        "raw": "Question: {query}\n\nSelect the single best option letter.",
        "cot": (
            "Question: {query}\n\n"
            "Think step by step. Eliminate wrong options one by one. "
            "Then give one JSON object."
        ),
        "react": (
            "Question: {query}\n\n"
            "Evaluate each option using reason[] steps. "
            "Then finish with one JSON object."
        ),
        "multiagent": f"Question: {{query}}\n\nReturn one JSON object: {_SCHEMA}",
        "self_consistency": (
            "Question: {query}\n\n"
            "Think step by step. Eliminate wrong options; cite the deciding fact.\n\n"
            f"End with one JSON object: {_SCHEMA}"
        ),
        "debate": (
            "Question: {query}\n\n"
            f"Evaluate each option carefully. Return one JSON object: {_SCHEMA}"
        ),
    },

    "math": {
        "raw": "Problem: {query}\n\nReturn the simplified final answer only.",
        "cot": (
            "Problem: {query}\n\n"
            "Show full working step by step. Verify before the JSON. "
            "Then give one JSON object."
        ),
        "react": (
            "Problem: {query}\n\n"
            "math_tool for numeric steps. Final JSON answer: sympy token only (e.g. 3/4), "
            "never spelled-out words."
        ),
        "multiagent": f"Problem: {{query}}\n\nReturn one JSON object: {_SCHEMA}",
        "self_consistency": (
            "Problem: {query}\n\n"
            "Think step by step. Sanity-check signs and edge cases before the JSON.\n\n"
            f"End with one JSON object: {_SCHEMA}"
        ),
        "debate": (
            "Problem: {query}\n\n"
            f"Solve fully and verify. Return one JSON object: {_SCHEMA}"
        ),
    },

    "hotpot": {
        "raw": "Question: {query}\n",
        "cot": "Question: {query}\n\nTrace each reasoning hop. Then give one JSON object.",
        "react": (
            "Question: {query}\n\n"
            "Two hops only. Decide bridge vs comparison first. "
            "wikipedia_search[exact article title] per hop — never a full-sentence query. "
            "If the question asks what something is 'named after' or what 'borders' another place, "
            "if there is one profession they share; prefer the most specific shared title "
            "answer that relation, not an intermediate place name alone."
        ),
        "multiagent": (
            "Question: {query}\n\n"
            "Planner: exactly 2 subtasks; bridge → s2 depends_on [s1]. "
            "search_query = Wikipedia titles only. "
            f"Judge returns one JSON object: {_SCHEMA}"
        ),
        "self_consistency": (
            "Question: {query}\n\n"
            f"Think step by step. Trace each hop.\n\nEnd with one JSON object: {_SCHEMA}"
        ),
        "debate": (
            "Question: {query}\n\n"
            f"Trace each hop on your own. Return one JSON object: {_SCHEMA}"
        ),
    },

    "musique": {
        "raw": "Question: {query}\n",
        "cot": (
            "Question: {query}\n\n"
            "Classify topology, trace each hop, then give one JSON object."
        ),
        "react": (
            "Question: {query}\n\n"
            "Plan: topology, hop count, entity chain. "
            "One wikipedia_search[article title] per hop in order; each title may use facts from prior Observations. "
            "Respect 'after/before/since' as sequence in the article; 'named after' → eponym; "
            "'bordering' → neighbor then its eponym."
        ),
        "multiagent": (
            "Question: {query}\n\n"
            "Planner: one subtask per hop; linear → chain depends_on; "
            "search_query = resolved Wikipedia titles (no angle-bracket placeholders). "
            f"Judge returns one JSON object: {_SCHEMA}"
        ),
        "self_consistency": (
            "Question: {query}\n\n"
            f"Classify topology, label each hop, end with one JSON object: {_SCHEMA}"
        ),
        "debate": (
            "Question: {query}\n\n"
            f"Classify topology, trace all hops alone, end with one JSON object: {_SCHEMA}"
        ),
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# DEBATE AGENT — synthesis prompts
# ─────────────────────────────────────────────────────────────────────────────

DEBATE_SYNTHESIS_SYSTEM: str = (
    "Impartial judge evaluating two independent answers to the same question. "
    "Identify which answer is better supported by reasoning. No preference between reasoners."
)

# Shared debate synthesis header and footer
_DS_HEADER = (
    "Two independent reasoners answered the same question differently.\n\n"
    "{context_label}: {{question}}\n\n"
    "Reasoner A:\n{{answer_a}}\n\nReasoner B:\n{{answer_b}}\n\n"
)
_DS_FOOTER = f"Return one JSON object and nothing else:\n{_SCHEMA}\n"

DEBATE_SYNTHESIS_PROMPT: dict[str, str] = {

    "gaia": (
        "Two independent reasoners answered the same real-world question differently.\n\n"
        "Question: {question}\n\nReasoner A:\n{answer_a}\n\nReasoner B:\n{answer_b}\n\n"
        "Evaluate which answer is better supported by reasoning — "
        "consider factual accuracy and specificity. "
        f"{_DS_FOOTER}"
        "answer: word, name, number, date, or short phrase. No fences."
    ),

    "mmlu_pro": (
        "Two independent reasoners answered the same multiple-choice question "
        "and selected different option letters.\n\n"
        "Question: {question}\n\nReasoner A:\n{answer_a}\n\nReasoner B:\n{answer_b}\n\n"
        "Evaluate which letter is better supported; cite the deciding fact. "
        f"{_DS_FOOTER}"
        "answer: one letter only (A–J). No fences."
    ),

    "math": (
        "Two mathematicians solved the same problem and reached different answers.\n\n"
        "Problem: {question}\n\nMathematician A:\n{answer_a}\n\nMathematician B:\n{answer_b}\n\n"
        "Check each solution step by step; identify where they diverge. "
        "If both are equivalent, confirm the simpler canonical form. "
        f"{_DS_FOOTER}"
        "answer: sympy-parseable — '42','3/4','x**2+1','sqrt(3)/2'. No fences."
    ),

    "hotpot": (
        "Two independent reasoners answered the same 2-hop question differently.\n\n"
        "Question: {question}\n\nReasoner A:\n{answer_a}\n\nReasoner B:\n{answer_b}\n\n"
        "Prefer the answer that matches the asked relation (eponym, border, date) across both hops. "
        f"{_DS_FOOTER}"
        "answer: short span — name, number, date, yes, or no. No fences."
    ),

    "musique": (
        "Two independent agents answered the same multi-hop question differently.\n\n"
        "Question: {question}\n\nReasoner A:\n{answer_a}\n\nReasoner B:\n{answer_b}\n\n"
        f"{_MUSIQUE_TOPO}"
        "Check hop order, temporal 'after/before', and 'named after' vs place name. "
        "Flag missing hops or answers from the wrong invasion/event/city. "
        f"{_DS_FOOTER}"
        "answer: name/number/date/short phrase. No fences."
    ),
}


# ─────────────────────────────────────────────────────────────────────────────
# MULTIAGENT PROMPTS — injected by build_multiagent_prompts()
# ─────────────────────────────────────────────────────────────────────────────

# Planner JSON examples — include non-empty search_query for Wikipedia hops.
MULTIAGENT_PLANNER_JSON_EXAMPLE: dict[str, str] = {
    "hotpot": (
        '{"final_constraint":"","subtasks":['
        '{"id":"s1","goal":"...","focus":"factual","needs_tool":true,'
        '"tool_name":"wikipedia_search","search_query":"<entity or event title>",'
        '"candidate_url":"","depends_on":[]},'
        '{"id":"s2","goal":"...","focus":"factual","needs_tool":true,'
        '"tool_name":"wikipedia_search","search_query":"<entity from s1>",'
        '"candidate_url":"","depends_on":["s1"]}]}'
    ),
    "musique": (
        '{"final_constraint":"","subtasks":['
        '{"id":"s1","goal":"...","focus":"factual","needs_tool":true,'
        '"tool_name":"wikipedia_search","search_query":"The Man from Morocco",'
        '"candidate_url":"","depends_on":[]},'
        '{"id":"s2","goal":"...","focus":"factual","needs_tool":true,'
        '"tool_name":"wikipedia_search","search_query":"Mutz Greenbaum",'
        '"candidate_url":"","depends_on":["s1"]},'
        '{"id":"s3","goal":"...","focus":"factual","needs_tool":true,'
        '"tool_name":"wikipedia_search","search_query":"Soviet invasion of Manchuria",'
        '"candidate_url":"","depends_on":["s2"]}]}'
    ),
    "math": (
        '{"final_constraint":"","subtasks":[{"id":"s1","goal":"...",'
        '"focus":"calculation","needs_tool":true,"tool_name":"math_tool",'
        '"search_query":"","candidate_url":"","depends_on":[]}]}'
    ),
    "gaia": (
        '{"final_constraint":"","subtasks":[{"id":"s1","goal":"...",'
        '"focus":"factual","needs_tool":true,"tool_name":"web_search",'
        '"search_query":"<short query>","candidate_url":"","depends_on":[]}]}'
    ),
    "mmlu_pro": (
        '{"final_constraint":"","subtasks":[{"id":"s1","goal":"...",'
        '"focus":"reasoning","needs_tool":false,"tool_name":"none",'
        '"search_query":"","candidate_url":"","depends_on":[]}]}'
    ),
}

MULTIAGENT_PLANNER_STRATEGY: dict[str, str] = {

    "gaia": (
        "Real-world question — may need web_search, read_file, or direct_url. "
        "Decompose into 2–3 subtasks (4 only if truly necessary). "
        "needs_tool=true when external data is required. "
        "If the question names an attached file basename, first subtask read_file (exact name) "
        "before web. Prefer direct_url for authoritative sources (usgs.gov, census.gov etc.). "
        "Do NOT create a subtask if a prior subtask already retrieves the needed data."
    ),

    "mmlu_pro": (
        "Multiple-choice question. Decompose into: concept understanding, "
        "option evaluation, and confirmation. "
        "needs_tool=false for all subtasks — workers reason from knowledge."
    ),

    "math": (
        "Mathematics problem — sequential computation steps. "
        "needs_tool=true + tool_name=\"math_tool\" for numerical/verification steps; "
        "needs_tool=false for pure symbolic reasoning. "
        "Never use web_search or wikipedia for math — use math_tool only. "
        "Final answer must be sympy-parseable (include base suffix when required, e.g. 1_6)."
    ),

    "hotpot": (
        "2-hop QA: exactly 2 subtasks — no more, no fewer.\n"
        f"{_HOTPOT_BRIDGE_COMP}"
        f"{_WIKI_TITLE_SEARCH}"
        "needs_tool=true + tool_name=\"wikipedia_search\" for both. "
        "search_query = Wikipedia article title (non-empty). "
        "Bridge: s1 depends_on []; s2 depends_on [s1] — mandatory. "
        "Comparison: s1 and s2 may both have depends_on []; goals name the two entities. "
        f"{_WIKI_SEMANTICS_COMMON}"
        "Max 2 subtasks. No duplicate search_query values."
    ),

    "musique": (
        "2–4-hop QA (open Wikipedia).\n"
        f"{_MUSIQUE_TOPO}"
        f"{_MUSIQUE_PLANNER_CHAIN}"
        f"{_WIKI_TITLE_SEARCH}"
        "needs_tool=true + tool_name=\"wikipedia_search\" per retrieval hop. "
        "One subtask per hop — do not collapse a 3–4 hop question into two parallel roots unless "
        "topology is clearly parallel. "
        f"{_WIKI_SEMANTICS_MUSIQUE}"
        "Max 4 subtasks. No duplicate search_query. "
        "final_constraint: optional phrase stating the answer type expected (date, eponym, name)."
    ),
}

MULTIAGENT_WORKER_RESULT_FORMAT: dict[str, str] = {

    "gaia": (
        "result: word, name, number, date, or short phrase — "
        "the exact value that answers the subtask goal."
    ),

    "mmlu_pro": (
        "result: single option letter (A–J) when subtask is to identify the correct option; "
        "short factual statement for analytical subtasks."
    ),

    "math": (
        "result: sympy-parseable expression — '42','3/4','x**2+1','sqrt(3)/2'. "
        "math_tool inputs must be pure Python — no units or words."
    ),

    "hotpot": (
        "result: short span — name, number, date, yes, or no — "
        "the fact this subtask must pass to the next hop or judge. "
        "If the goal is 'named after what', store the eponym, not the county name. "
        "if there is one profession they share; prefer the most specific shared title "
        "evidence: Wikipedia title searched."
    ),

    "musique": (
        "result: exact Wikipedia span — name, number, date, or short phrase — "
        "the bridging fact for downstream hops (city, country, person, event date). "
        "evidence: article title searched. "
        "Never return a list-page title as the final hop fact without picking one line."
    ),
}

MULTIAGENT_JUDGE_ANSWER_FORMAT: dict[str, str] = {

    "gaia": (
        "answer: word, name, number, date, or short phrase — "
        "exact string matched against the reference answer."
    ),

    "mmlu_pro": (
        "answer: one option letter only — A, B, C, D, E, F, G, H, I, or J. "
        "No other text, no explanation."
    ),

    "math": (
        "answer: sympy-parseable — '42','3/4','x**2+1','sqrt(3)/2','pi/4'. "
        "No prose, no 'the answer is'."
    ),

    "hotpot": (
        "answer: short span — name, number, date, yes, or no. "
        "Verify both hops and the asked relation (eponym vs place name, neighbor vs seat). "
        "No full sentences."
    ),

    "musique": (
        "answer: short span — name, number, date, or short phrase. "
        "Walk the dependency chain in order; reject if a hop fact is missing or contradicts. "
        "Apply temporal words to event order in sources; apply 'named after' to eponyms. "
        "parallel → both roots grounded; branching → s2A and s2B each trace to s1. No sentences."
    ),
}

# Judge output schema — unchanged
MULTIAGENT_JUDGE_SCHEMA: str = (
    '{"answer":"<bare value>",'
    '"confidence":<float 0-1>,'
    '"complexity":<float 0-1>,'
    '"consensus_score":<float 0-1>,'
    '"rationale":"1-2 sentences",'
    '"conflict_resolution":"none or explanation"}'
)


@dataclass(frozen=True)
class MultiagentPrompts:
    """Role prompts assembled once in MultiAgentAgent.__init__."""
    planner_system: str
    worker_system: str
    judge_system: str
    extractor_system: str
    structured_extractor_template: str
    judge_fallback_system: str


def build_multiagent_prompts(
    dataset: str,
    *,
    tools_csv: str,
    tool_enum: str,
) -> MultiagentPrompts:
    """Single entry point: all multi-agent role system prompts for one dataset."""
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")

    strategy   = MULTIAGENT_PLANNER_STRATEGY[dataset]
    result_fmt = MULTIAGENT_WORKER_RESULT_FORMAT[dataset]
    answer_fmt = MULTIAGENT_JUDGE_ANSWER_FORMAT[dataset]

    json_example = MULTIAGENT_PLANNER_JSON_EXAMPLE.get(
        dataset,
        MULTIAGENT_PLANNER_JSON_EXAMPLE["gaia"],
    )
    planner_system = (
        "Planner in a multi-agent QA system. "
        "Return ONLY strict JSON (no markdown):\n"
        f"{json_example}\n"
        f"tool_name ∈ {tool_enum}. Available tools: {tools_csv}\n"
        "When tool_name is wikipedia_search, search_query MUST be a non-empty "
        "Wikipedia article title (never \"\", never a full question, never attribute-only).\n"
        "needs_tool=true only when external data/computation is required. "
        "No duplicate subtasks — use depends_on instead of re-fetching.\n"
        f"STRATEGY: {strategy}"
    )

    worker_system = (
        "Specialist Worker in a multi-agent QA system.\n"
        "Solve ONLY your assigned subtask. Draft → challenge → finalise.\n"
        "Return strict JSON only:\n"
        '{"subtask_id":"...","result":"...","confidence":<0-1>,'
        '"evidence":"≤15 words","self_critique":"one sentence"}\n'
        f"RESULT FORMAT: {result_fmt}"
    )

    judge_system = (
        "Judge in a multi-agent QA system.\n"
        "Review working memory, cross-validate facts, produce ONE concise answer.\n"
        "Prefer a high-confidence worker's direct factual answer unless conflicting "
        "evidence exists. Do NOT reinterpret symbolic answers.\n"
        "Return ONLY strict JSON:\n"
        f"{MULTIAGENT_JUDGE_SCHEMA}\n"
        f"ANSWER FORMAT: {answer_fmt}\n"
        "Numbers: digits only. Names: name only. Boolean: yes/no. Unknown: unknown. "
        "No preamble in answer field."
    )

    return MultiagentPrompts(
        planner_system=planner_system,
        worker_system=worker_system,
        judge_system=judge_system,
        extractor_system=(
            "Fact extractor. Return ONLY sentences/numbers/names that DIRECTLY answer "
            "the question. Preserve exact values. If nothing relevant: NO_RELEVANT_CONTENT\n"
            "Max 150 words. No preamble."
        ),
        structured_extractor_template=(
            "Extract fields [{fields_str}] as compact JSON. "
            "null for missing. Exact values. No markdown."
        ),
        judge_fallback_system=(
            "Answer extractor. Return ONLY the final answer — no explanation. "
            "If no answer: unknown"
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# DEBATE AGENT TEMPERATURE — dataset-specific
# ─────────────────────────────────────────────────────────────────────────────

DEBATE_AGENT_TEMPERATURE: dict[str, float] = {
    "gaia":     0.7,
    "mmlu_pro": 0.5,
    "math":     0.3,
    "hotpot":   0.7,
    "musique":  0.7,
}


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATION — runs at import time
# ─────────────────────────────────────────────────────────────────────────────

def _validate_prompts() -> None:
    missing: list[str] = []

    for dataset in DATASETS:
        for agent in AGENTS:
            if dataset not in SYSTEM_PROMPT or agent not in SYSTEM_PROMPT.get(dataset, {}):
                missing.append(f"SYSTEM_PROMPT['{dataset}']['{agent}']")
            if dataset not in USER_PROMPT or agent not in USER_PROMPT.get(dataset, {}):
                missing.append(f"USER_PROMPT['{dataset}']['{agent}']")

    for dataset in DATASETS:
        if dataset not in DEBATE_SYNTHESIS_PROMPT:
            missing.append(f"DEBATE_SYNTHESIS_PROMPT['{dataset}']")
        if dataset not in MULTIAGENT_PLANNER_STRATEGY:
            missing.append(f"MULTIAGENT_PLANNER_STRATEGY['{dataset}']")
        if dataset not in MULTIAGENT_WORKER_RESULT_FORMAT:
            missing.append(f"MULTIAGENT_WORKER_RESULT_FORMAT['{dataset}']")
        if dataset not in MULTIAGENT_JUDGE_ANSWER_FORMAT:
            missing.append(f"MULTIAGENT_JUDGE_ANSWER_FORMAT['{dataset}']")
        if dataset not in DEBATE_AGENT_TEMPERATURE:
            missing.append(f"DEBATE_AGENT_TEMPERATURE['{dataset}']")
        try:
            build_multiagent_prompts(dataset, tools_csv="none", tool_enum="none")
        except KeyError as exc:
            missing.append(str(exc))

    if missing:
        raise KeyError(
            "Missing prompt entries — add these before running:\n"
            + "\n".join(f"  {m}" for m in missing)
        )


_validate_prompts()

# from __future__ import annotations

# from dataclasses import dataclass

# # ─────────────────────────────────────────────────────────────────────────────
# # CONSTANTS
# # ─────────────────────────────────────────────────────────────────────────────

# DATASETS: list[str] = ["gaia", "mmlu_pro", "math", "hotpot", "musique"]
# AGENTS:   list[str] = ["raw", "cot", "react", "multiagent", "self_consistency", "debate"]

# # Shared output schema — every agent, every dataset returns these three fields.
# _SCHEMA = '{"answer":"<value>","confidence":<float 0-1>,"complexity":<float 0-1>}'

# # Compact topology table — one source of truth, ~40 tokens
# _MUSIQUE_TOPO = (
#     "Topology by question_id prefix:\n"
#     "  2hop__|3hop1__|4hop1__ → linear   : s1→s2→…→answer\n"
#     "  3hop2__|4hop2__        → parallel : s1A + s1B (independent) → merge→answer\n"
#     "  4hop3__                → branching: s1→(s2A,s2B independent)→answer\n"
# )

# # ─────────────────────────────────────────────────────────────────────────────
# # SYSTEM PROMPTS
# # ─────────────────────────────────────────────────────────────────────────────

# SYSTEM_PROMPT: dict[str, dict[str, str]] = {

#     # ─────────────────────────────────────────────────────────────────────────
#     # GAIA — unchanged, working perfectly
#     # ─────────────────────────────────────────────────────────────────────────
#     "gaia": {

#         "raw": (
#             "Given a query, an expert in handling real world everyday situation agent, answer the query."
#             "Return exactly one JSON object with this exact schema and no extra text: "
#             f"{_SCHEMA}\n"
#             "Answer must be a word, name, number, date, or short phrase "
#         ),

#         "cot": (
#             "You are a multi-hop reasoning expert and an expert QA agent. Given a question, "
#             "gather the relevant context and think step-by-step. "
#             "Return exactly one JSON object with this exact schema and no extra text: "
#             f"{_SCHEMA}\n"
#             "Answer must be a word, name, number, date, or short phrase "
#         ),

#         "react": (
#             "You are a ReAct (Reasoning + Acting + Observation) agent: alternate structured "
#             "thinking with tool use, one step at a time.\n"
#             "## CYCLE FORMAT  (strictly follow this order, every turn)\n"
#             "Thought: What you know, what is missing, which tool fills the gap — concise.\n"
#             "Action: tool_name[input]\n"
#             "The runtime injects (never write this yourself):\n"
#             "Observation: <tool result>\n"
#             "Repeat Thought → Action → receive Observation until you can answer from Observations.\n"
#             "After each Observation, use Thought to summarize only relevant information needed "
#             "for the question (minimal identifiers, names, numbers)—do not copy long tool output "
#             "verbatim.\n"
#             "Observation remains authoritative; never state facts that are not grounded in it.\n"
#             "## Final turn only (grounded answer)\n"
#             "Thought: Briefly how Observations support your answer.\n"
#             "Action: finish[<JSON>]\n"
#             f"<JSON> is one JSON object (no markdown fences) with:\n{_SCHEMA}\n"
#             "- answer: word, name, number, date, or short phrase.\n"
#             "- confidence and complexity: numeric, in [0, 1].\n"
#             f"Example: Action: finish[{_SCHEMA}]\n\n"
#             'Do not use a separate "Final Answer:" line. Do not emit bare JSON without '
#             "Action: finish[...].\n"
#             "Available tools:\n"
#             "{tools_block}\n"
#             "Each tool uses tool_name[input]. Prefer internal reasoning when it is clearly "
#             "cheaper than a tool call.\n"
#             "EXTRACTION (files / long Observations)\n"
#             "1. Scan the entire Observation top to bottom.\n"
#             "2. Extract everything matching the question; preserve order of appearance.\n"
#             "3. Copy values exactly as found — no invented formatting.\n"
#             "RULES\n"
#             "1. Every assistant message: Thought: then Action: (same message).\n"
#             "2. Exactly one Action per turn — no chained tools.\n"
#             "3. Never invent tool outputs.\n"
#             "4. Avoid unnecessary tools; do not repeat the exact same Action twice.\n"
#             "5. finish only when the answer is supported by Observations "
#             "(or direct inference from them).\n"
#             '6. If the question includes "Attached file name for reference:" → read_file with '
#             "that exact basename only (often a UUID like abc-123.pdf), "
#             "e.g. read_file[abc-123.pdf]. Never invent filenames like report.pdf.\n"
#             "SELF-CHECK (before Action: finish[...])\n"
#             "- Answer traces to Observations; JSON valid; confidence and complexity in [0,1]; "
#             "finish payload is JSON only inside the brackets."
#         ),

#         "multiagent": (
#             "You are the coordinator of a multi-agent QA workflow with three stages: "
#             "Planner -> Workers -> Judge/Synthesizer. "
#             "All intermediate outputs must be strict JSON. "
#             "Planner JSON: "
#             '{"subtasks":[{"id":"s1","goal":"...","focus":"reasoning|factual|calculation"}]}. '
#             "Worker JSON: "
#             '{"subtask_id":"s1","result":"...","confidence":0.0,"evidence":"..."}. '
#             "Judge JSON: "
#             '{"answer":"<value>","consensus_score":0.0,"rationale":"..."}. '
#             'The final answer must be exactly one JSON object: {{"answer":"<value>"}}. '
#         ),

#         "self_consistency": (
#             "You are a self-consistency agent that solves real-world questions."
#             "Always: (1)  Think through problems step by step and give a precise answer. "
#             "(2) end with exactly one JSON object and nothing after it. "
#             "CRITICAL: "
#             "- Always give your final answer in the EXACT unit the question asks for"
#             "- If the question asks how many X, answer as a plain number of X"
#             "- Never convert units unless the question asks you to"
#             "- Re-read the question before writing your final answer"
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: word, name, number, date, or short phrase only. "
#             "No markdown fences."
#         ),

#         "debate": (
#             "You are an independent expert agent answering a real-world question. "
#             "Give your best answer supported by step by step reasoning. "
#             "You are not aware of any other agent — reason entirely on your own. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: word, name, number, date, or short phrase only. "
#             "No markdown fences."
#         ),
#     },

#     # ─────────────────────────────────────────────────────────────────────────
#     # MMLU-PRO — unchanged
#     # ─────────────────────────────────────────────────────────────────────────
#     "mmlu_pro": {

#         "raw": (
#             "You are an expert at academic multiple-choice questions. "
#             "Read the question and options, then select the single best option letter. "
#             "Return exactly one JSON object and nothing else:\n"
#             f"{_SCHEMA}\n"
#             "answer: single option letter only — A, B, C, D, E, F, G, H, I, or J. "
#             "No explanation in the answer field. No markdown fences."
#         ),

#         "cot": (
#             "You are an  multi-hop reasoning expert at academic multiple-choice questions across all subjects. "
#             "Think step by step. Evaluate each option. Eliminate wrong ones explicitly. "
#             "State the definition or fact that decides between the remaining options. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: single option letter only (A through J). No markdown fences."
#         ),

#         "react": (
#             "You are a ReAct agent solving multiple-choice academic questions.\n\n"
#             "CYCLE FORMAT:\n"
#             "Thought: which option you are evaluating and why it may be right or wrong.\n"
#             "Action: reason[your evaluation of this option]\n\n"
#             "Use reason[] to work through each option one at a time. "
#             "When you have eliminated enough options to be confident:\n\n"
#             "Thought: which option is correct and the decisive fact.\n"
#             f"Action: finish[{_SCHEMA}]\n\n"
#             "answer: single option letter (A through J). "
#             "confidence and complexity: floats in [0,1].\n\n"
#             "RULES:\n"
#             "1. Every message: Thought then Action.\n"
#             "2. One Action per turn.\n"
#             "3. Do not select an option without reasoning through it.\n"
#         ),

#         "multiagent": (
#             "You are coordinating a multi-agent pipeline (Planner → Workers → Judge) "
#             "to answer multiple-choice academic questions. "
#             "No external tools are needed. Workers reason from knowledge. "
#             "Final output must be a single option letter: A through J."
#         ),

#         "self_consistency": (
#             "You are a self-consistency agent for academic multiple-choice questions. "
#             "Always: (1)  Think through problems step by step and give a precise answer. "
#             "(2) end with exactly one JSON object and nothing after it. "
#             "Eliminate wrong options and cite the definition or fact that decides. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: single option letter only (A through J). No markdown fences."
#         ),

#         "debate": (
#             "You are an independent expert agent answering a multiple-choice "
#             "academic question. "
#             "Evaluate each option carefully from your own knowledge. "
#             "You are not aware of any other agent — reason entirely on your own. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: single option letter only (A through J). No markdown fences."
#         ),
#     },

#     # ─────────────────────────────────────────────────────────────────────────
#     # MATH
#     # Task    : competition mathematics problems across 5 difficulty levels.
#     # Answer  : simplified mathematical expression parseable by sympy.
#     # Eval    : symbolic equivalence via sympy (not string match).
#     # Tools   : calculator (Tier 2/3 only).
#     # ─────────────────────────────────────────────────────────────────────────
#     "math": {

#         "raw": (
#             "You are an expert mathematician. "
#             "Solve the problem and return only the final simplified answer. "
#             "Return exactly one JSON object and nothing else:\n"
#             f"{_SCHEMA}\n"
#             "answer: simplified result as a string in Python-compatible notation "
#             "that sympy can parse — for example '42', '3/4', 'sqrt(3)/2', "
#             "'x**2 + 1', 'pi/4'. Do not write 'the answer is'. No markdown fences."
#         ),

#         "cot": (
#             "You are an expert mathematician. "
#             "Solve the problem step by step. "
#             "Before writing the final JSON, verify your solution satisfies the "
#             "original conditions. If it does not, correct your working. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: simplified result in Python-compatible notation parseable by "
#             "sympy — '42', '3/4', 'x**2 + 1'. No markdown fences."
#         ),

#         # ── UPDATED: mirrors GAIA react structure; injects {tools_block} ──
#         "react": (
#             "ReAct agent for mathematics. Every turn:\n"
#             "Thought: which step, what's needed.\n"
#             "Action: tool_name[input]   ← runtime injects Observation next.\n"
#             "Use math_tool[<python_expr>] for each arithmetic/algebraic step; "
#             "pure Python only — no units or words (e.g. '(3**2+4**2)**0.5').\n"
#             "Tools:\n{tools_block}\n"
#             "To finish — verify signs/edge-cases first, then:\n"
#             f"Action: finish[{_SCHEMA}]\n"
#             "answer: sympy-parseable string — '42','3/4','sqrt(3)/2','x**2+1','pi/4'.\n"
#             "Rules: one Action per turn; never invent Observations; "
#             "reformulate (don't repeat) a failed expression; "
#             "finish only when result is Observation-grounded."
#         ),

#         # ── UPDATED: tool-aware multiagent mirrors GAIA multiagent ──
#         "multiagent": (
#             "Planner→Workers→Judge pipeline for mathematics. All outputs strict JSON.\n"
#             'Planner: {"subtasks":[{"id":"s1","goal":"...","focus":"reasoning|calculation",'
#             '"needs_tool":false,"tool_name":"none"}]}\n'
#             "needs_tool=true + tool_name=\"math_tool\" for numerical steps; false for symbolic.\n"
#             'Worker: {"subtask_id":"s1","result":"<sympy-expr>","confidence":0.0,"evidence":"..."}\n'
#             f"Judge (one object, no extra text): {_SCHEMA}\n"
#             "answer: sympy-parseable — '42','3/4','sqrt(3)/2','x**2+1'. "
#             "Verify equivalence if Workers disagree; prefer algebraically correct result."
#         ),

#         "self_consistency": (
#             "You are a self-consistency agent for mathematics problems. "
#             "Always: (1)  Think through problems step by step and give a precise answer. "
#             "(2) end with exactly one JSON object and nothing after it."
#             "Before writing the final JSON, sanity-check "
#             "signs, units, and edge cases. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: simplified result in Python-compatible notation parseable by "
#             "sympy — '42', '3/4', 'x**2 + 1'. No markdown fences."
#         ),

#         "debate": (
#             "You are an independent expert mathematician. "
#             "Solve the problem fully, showing your working, then commit to a final answer. "
#             "You are not aware of any other agent — solve entirely on your own. "
#             "Verify your solution before writing the JSON. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: simplified result in Python-compatible notation parseable by "
#             "sympy — '42', '3/4', 'x**2 + 1'. No markdown fences."
#         ),
#     },

#     # ─────────────────────────────────────────────────────────────────────────
#     # HOTPOTQA
#     # Task    : 2-hop multi-hop QA.
#     # Answer  : short span — name, date, number, yes, or no.
#     # Eval    : token-level F1 and exact match.
#     # Tools   : wikipedia (Tier 2/3 only).
#     # ─────────────────────────────────────────────────────────────────────────
#     "hotpot": {

#         "raw": (
#             "You are an expert at multi-hop question answering. "
#             "Answer the question directly. "
#             "Return exactly one JSON object and nothing else:\n"
#             f"{_SCHEMA}\n"
#             "answer: short span — name, number, date, yes, or no. "
#             "No full sentences. No markdown fences."
#         ),

#         "cot": (
#             "You are an expert at multi-hop question answering. "
#             "Identify the chain of two reasoning hops needed to answer. "
#             "State each hop explicitly and what it establishes. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: short span — name, number, date, yes, or no. "
#             "No markdown fences."
#         ),

#         # ── UPDATED: mirrors GAIA react structure; wikipedia via {tools_block} ──
#         "react": (
#             "ReAct agent for 2-hop QA. Every turn:\n"
#             "Thought: which hop (1 or 2), which entity to look up.\n"
#             "Action: tool_name[input]   ← runtime injects Observation next.\n"
#             "Exactly 2 hops — one wikipedia per hop. "
#             "Search specific entity names only, not full sentences. "
#             "Summarise only the relevant fact from each Observation.\n"
#             "Tools:\n{tools_block}\n"
#             "To finish (after both hops are Observation-grounded):\n"
#             "Thought: how Hop 1 + Hop 2 give the answer.\n"
#             f"Action: finish[{_SCHEMA}]\n"
#             "answer: short span — name, number, date, yes, or no.\n"
#             "Rules: one Action per turn; never invent facts; "
#             "reformulate (don't repeat) an insufficient query; "
#             "finish only when both hops are grounded."
#         ),

#         # ── UPDATED: tool-aware multiagent; one subtask per hop ──
#         "multiagent": (
#             "Planner→Workers→Judge pipeline for 2-hop QA. All outputs strict JSON.\n"
#             'Planner: {"subtasks":[{"id":"s1","goal":"...","focus":"factual",'
#             '"needs_tool":true,"tool_name":"wikipedia",'
#             '"search_query":"<entity name>","depends_on":[]}]}\n'
#             "Exactly 2 subtasks. search_query = entity name only. "
#             "s2.depends_on=[\"s1\"]; s2 derives its search_query from s1's result.\n"
#             'Worker: {"subtask_id":"s1","result":"<Wikipedia fact>","confidence":0.0,"evidence":"..."}\n'
#             f"Judge (one object, no extra text): {_SCHEMA}\n"
#             "answer: short span — name, number, date, yes, or no. "
#             "Verify both hop results are consistent and jointly entail the answer."
#         ),

#         "self_consistency": (
#             "You are a self-consistency agent for multi-hop question answering. "
#             "Trace each reasoning hop step by step. "
#             "Always: (1)  Think through problems step by step and give a precise answer. "
#             "(2) end with exactly one JSON object and nothing after it. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: short span — name, number, date, yes, or no. "
#             "No markdown fences."
#         ),

#         "debate": (
#             "You are an independent expert at multi-hop question answering. "
#             "Trace each reasoning hop on your own. "
#             "You are not aware of any other agent. "
#             "End with exactly one JSON object and nothing after it:\n"
#             f"{_SCHEMA}\n"
#             "answer: short span — name, number, date, yes, or no. "
#             "No markdown fences."
#         ),
#     },

#     # ─────────────────────────────────────────────────────────────────────────
#     # MUSIQUE
#     # Task    : 2–4 hop multi-hop QA.
#     # Answer  : short span — name, number, date, or short phrase.
#     # Eval    : token-level F1 and exact match.
#     # Tools   : wikipedia (Tier 2/3 only).
#     # ─────────────────────────────────────────────────────────────────────────
#     "musique": {

#             "raw": (
#                 "you are an expert multi-hop QA. Answer directly. "
#                 "Return one JSON: " + _SCHEMA + "\n"
#                 "answer: name/number/date/short phrase. No sentences, no fences."
#             ),

#             "cot": (
#                 "Expert multi-hop QA (2–4 hops).\n"
#                 + _MUSIQUE_TOPO +
#                 "For each hop/sub-hop: [Hop N] Entity → Fact\n"
#                 "Linear/Parallel/branching: trace sub-chains independently, then merge.\n"
#                 "No skipped hops. End with one JSON: " + _SCHEMA + "\n"
#                 "answer: name/number/date/short phrase. No fences."
#             ),

#         # ── UPDATED: mirrors GAIA react structure; wikipedia via {tools_block} ──
#         "react": (
#             "ReAct agent for 2–4-hop QA. Every turn:\n"
#              + _MUSIQUE_TOPO +
#             "Thought: Hop N of M — which entity to look up and why.\n"
#             "Action: tool_name[input]   ← runtime injects Observation next.\n"
#             "One wikipedia per hop. Search entity names only — not full sentences. "
#             "Each hop's Observation yields the entity for the next hop. "
#             "Summarise only the relevant fact; never copy passages verbatim. "
#             "Don't skip hops. complexity ∝ hop count.\n"
#             "Tools:\n{tools_block}\n"
#             "To finish (after all hops are Observation-grounded):\n"
#             "Thought: how the full hop chain gives the answer.\n"
#             f"Action: finish[{_SCHEMA}]\n"
#             "answer: short span — name, number, date, or short phrase.\n"
#             "Rules: one Action per turn; never invent facts; "
#             "reformulate (don't repeat) an insufficient query; "
#             "finish only when all hops are grounded."
#         ),

#         # ── UPDATED: tool-aware multiagent; one subtask per hop (2–4) ──
#         "multiagent": (
#             "Planner→Workers→Judge, 2–4-hop QA.\n"
#             + _MUSIQUE_TOPO +
#             "PLANNER — one subtask per hop/sub-hop, strict JSON:\n"
#             '{"subtasks":[{"id":"s1","goal":"...","needs_tool":true,'
#             '"tool_name":"wikipedia","search_query":"<entity>","depends_on":[]}]}\n'
#             "depends_on: linear→chain; parallel→roots=[], merge lists both; "
#             "branching→s2A/s2B each=[\"s1\"], final=[\"s2A\",\"s2B\"]\n"
#             "search_query: entity name only. No duplicate fetches.\n"
#             'WORKER: {"subtask_id":"...","result":"<exact Wikipedia span>",'
#             '"confidence":0-1,"evidence":"entity searched"}\n'
#             "JUDGE (one JSON): " + _SCHEMA + "\n"
#             "answer: name/number/date/short phrase. "
#             "Verify all sub-chains independently grounded; low confidence if any missing."
#         ),

#         "self_consistency": (
#                 "You are an Self-consistency agent, multi-hop QA (2–4 hops).\n"
#                 + _MUSIQUE_TOPO +
#                 "Label each hop: [Hop N / Hop NA / Hop NB] Entity → Fact\n"
#                 "Parallel/branching: both sub-chains fully before merge.\n"
#                 "End with one JSON: " + _SCHEMA + "\n"
#                 "answer: name/number/date/short phrase. No fences."
#             ),
#         "debate": (
#             "You are anIndependent expert, multi-hop QA (2–4 hops). Reason alone.\n"
#             + _MUSIQUE_TOPO +
#             "Label each hop: [Hop N / Hop NA / Hop NB] Entity → Fact\n"
#             "Parallel/branching: trace both sub-chains, then state merge reasoning.\n"
#             "End with one JSON: " + _SCHEMA + "\n"
#             "answer: name/number/date/short phrase. No fences."
#         ),
#     },
# }


# # ─────────────────────────────────────────────────────────────────────────────
# # USER PROMPTS
# # ─────────────────────────────────────────────────────────────────────────────

# USER_PROMPT: dict[str, dict[str, str]] = {

#     "gaia": {
#         "raw": (
#             "Question: {query}\n"
#         ),
#         "cot": (
#             "Question: {query}\n\n"
#             "Think step by step. Then give your answer as one JSON object."
#         ),
#         "react": (
#             "Question: {query}\n\n"
#             "If the question references an attached file, use "
#             "read_file[exact_filename] with the filename as written in the question. "
#             "Do not invent filenames."
#         ),
#         "multiagent": (
#             "Question: {query}\n\n"
#             f"Return exactly one JSON object: {_SCHEMA}"
#         ),
#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Let's think step by step.\n\n"
#             f"End with one JSON object: {_SCHEMA}"
#         ),
#         "debate": (
#             "Question: {query}\n\n"
#             "Give your best answer based on your own reasoning. "
#             f"Return one JSON object: {_SCHEMA}"
#         ),
#     },

#     "mmlu_pro": {
#         "raw": (
#             "Question: {query}\n\n"
#             "Select the single best option letter."
#         ),
#         "cot": (
#             "Question: {query}\n\n"
#             "Think step by step. Eliminate wrong options one by one. "
#             "Then give your answer as one JSON object."
#         ),
#         "react": (
#             "Question: {query}\n\n"
#             "Evaluate each option using reason[] steps. "
#             "Then finish with one JSON object."
#         ),
#         "multiagent": (
#             "Question: {query}\n\n"
#             f"Return exactly one JSON object: {_SCHEMA}"
#         ),
#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Let's think step by step. Use numbered reasoning steps. "
#             "Eliminate wrong options and cite the deciding fact.\n\n"
#             f"End with one JSON object: {_SCHEMA}"
#         ),
#         "debate": (
#             "Question: {query}\n\n"
#             "Evaluate each option carefully. Select the best letter. "
#             f"Return one JSON object: {_SCHEMA}"
#         ),
#     },

#     "math": {
#         "raw": (
#             "Problem: {query}\n\n"
#             "Return the simplified final answer only."
#         ),
#         "cot": (
#             "Problem: {query}\n\n"
#             "Verify your answer before the JSON. "
#             "Then give one JSON object."
#         ),
#         # ── UPDATED: instructs agent to use calculator tool per step ──
#         "react": (
#             "Problem: {query}\n\n"
#             "Work through each computation step using the math_tool . "
#             "Call math_tool[<python_expression>] for every non-trivial arithmetic or "
#             "algebraic step — pass only valid Python expressions, no units or words. "
#             "Verify your result satisfies the original conditions before finishing."
#         ),
#         "multiagent": (
#             "Problem: {query}\n\n"
#             f"Return exactly one JSON object: {_SCHEMA}"
#         ),
#         "self_consistency": (
#             "Problem: {query}\n\n"
#             "Let's think step by step. dont'Show key equations and substitutions , do reasoning internally. "
#             "Sanity-check signs and edge cases before the JSON.\n\n"
#             f"End with one JSON object: {_SCHEMA}"
#         ),
#         "debate": (
#             "Problem: {query}\n\n"
#             "Solve fully and verify your answer. "
#             f"Return one JSON object: {_SCHEMA}"
#         ),
#     },

#     "hotpot": {
#         "raw": (
#             "Question: {query}\n"
#         ),
#         "cot": (
#             "Question: {query}\n\n"
#             "Trace each reasoning hop. Then give one JSON object."
#         ),
#         # ── UPDATED: instructs agent to use wikipedia per hop ──
#         "react": (
#             "Question: {query}\n\n"
#             "This question requires two reasoning hops. "
#             "Resolve each hop with a separate wikipedia call — search for specific "
#             "entity names, not full question sentences. "
#             "Then finish with one JSON object once both hops are grounded in Observations."
#         ),
#         "multiagent": (
#             "Question: {query}\n\n"
#             f"Return exactly one JSON object: {_SCHEMA}"
#         ),
#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Let's think step by step. Trace each hop.\n\n"
#             f"End with one JSON object: {_SCHEMA}"
#         ),
#         "debate": (
#             "Question: {query}\n\n"
#             "Trace each hop on your own. "
#             f"Return one JSON object: {_SCHEMA}"
#         ),
#     },

#     "musique": {
#         "raw": "Question: {query}\n",

#         "cot": (
#             "Question: {query}\n\n"
#             "Classify topology, trace each hop, then give one JSON."
#         ),

#         "react": (
#             "Question ID: {question_id}\n"
#             "Question: {query}\n\n"
#             "State Plan: line, then resolve each hop with wikipedia."
#         ),

#         "multiagent": (
#             "Question ID: {question_id}\n"
#             "Question: {query}\n\n"
#             "Return one JSON: " + _SCHEMA
#         ),

#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Classify topology, label each hop, end with one JSON: " + _SCHEMA
#         ),

#         "debate": (
#             "Question: {query}\n\n"
#             "Classify topology, trace all hops alone, end with one JSON: " + _SCHEMA
#         ),
#     }
# }


# # ─────────────────────────────────────────────────────────────────────────────
# # DEBATE AGENT — synthesis prompts (unchanged)
# # ─────────────────────────────────────────────────────────────────────────────

# DEBATE_SYNTHESIS_SYSTEM: str = (
#     "You are an impartial judge evaluating two independent answers to the same question. "
#     "Your only goal is to identify which answer is better supported by reasoning. "
#     "You have no preference between the two reasoners."
# )

# DEBATE_SYNTHESIS_PROMPT: dict[str, str] = {

#     "gaia": (
#         "Two independent reasoners have answered the same real-world question "
#         "and their answers differ.\n\n"
#         "Question: {question}\n\n"
#         "Reasoner A:\n{answer_a}\n\n"
#         "Reasoner B:\n{answer_b}\n\n"
#         "Evaluate which answer is better supported by the reasoning provided. "
#         "Consider factual accuracy and specificity. "
#         "Return exactly one JSON object and nothing else:\n"
#         f"{_SCHEMA}\n"
#         "answer: word, name, number, date, or short phrase — exact string. "
#         "No markdown fences."
#     ),

#     "mmlu_pro": (
#         "Two independent reasoners have answered the same multiple-choice question "
#         "and selected different option letters.\n\n"
#         "Question: {question}\n\n"
#         "Reasoner A:\n{answer_a}\n\n"
#         "Reasoner B:\n{answer_b}\n\n"
#         "Evaluate which option letter is better supported by the reasoning. "
#         "Cite the definition or fact that decides between the two positions. "
#         "Return exactly one JSON object and nothing else:\n"
#         f"{_SCHEMA}\n"
#         "answer: single option letter only (A through J). No markdown fences."
#     ),

#     "math": (
#         "Two independent mathematicians have solved the same problem "
#         "and reached different answers.\n\n"
#         "Problem: {question}\n\n"
#         "Mathematician A:\n{answer_a}\n\n"
#         "Mathematician B:\n{answer_b}\n\n"
#         "Check each solution step by step. Identify where they diverge. "
#         "Determine which solution is mathematically correct. "
#         "If both answers are equivalent expressions, confirm the simpler canonical form. "
#         "Return exactly one JSON object and nothing else:\n"
#         f"{_SCHEMA}\n"
#         "answer: simplified result in Python-compatible notation parseable by sympy "
#         "— '42', '3/4', 'x**2 + 1', 'sqrt(3)/2'. No markdown fences."
#     ),

#     "hotpot": (
#         "Two independent reasoners have answered the same multi-hop question "
#         "but reached different answers.\n\n"
#         "Question: {question}\n\n"
#         "Reasoner A:\n{answer_a}\n\n"
#         "Reasoner B:\n{answer_b}\n\n"
#         "Evaluate which answer better follows the multi-hop reasoning chain. "
#         "Identify any incorrect hop. "
#         "Return exactly one JSON object and nothing else:\n"
#         f"{_SCHEMA}\n"
#         "answer: short span — name, number, date, yes, or no. No markdown fences."
#     ),

#     "musique": (
#     "Two independent agents answered the same multi-hop question differently.\n\n"
#     "Question: {question}\n\nReasoner A:\n{answer_a}\n\nReasoner B:\n{answer_b}\n\n"
#     + _MUSIQUE_TOPO +
#     "Identify the topology, then check each agent's hop chain. "
#     "Flag any missing, swapped, or wrongly merged hop. "
#     "Return one JSON: " + _SCHEMA + "\n"
#     "answer: name/number/date/short phrase. No fences."
# )
# }


# # ─────────────────────────────────────────────────────────────────────────────
# # MULTIAGENT PROMPTS — built once per agent via build_multiagent_prompts()
# # ─────────────────────────────────────────────────────────────────────────────

# MULTIAGENT_PLANNER_STRATEGY: dict[str, str] = {
#     "gaia": (
#         "This is a real-world question that may require web search, file reading, "
#         "or direct URL fetch. Decompose into 2–3 subtasks (4 only if truly necessary). "
#         "Set needs_tool=true when external data is required. "
#         "Prefer direct_url for known authoritative sources (e.g. usgs.gov, census.gov); "
#         "use read_file when the question names an attached file (exact basename only). "
#         "Do NOT create a subtask if a prior subtask will already retrieve the needed data."
#     ),
#     "mmlu_pro": (
#         "This is a multiple-choice question. Decompose into: understanding the concept, "
#         "evaluating each option, and confirming the correct letter. "
#         "Set needs_tool=false for all subtasks — solve by reasoning from knowledge, "
#         "not retrieval."
#     ),
#     # ── UPDATED: math planner now knows about calculator tool ──
#     "math": (
#         "Mathematics problem: sequential computation steps. "
#         "needs_tool=true + tool_name=\"calculator\" for numerical steps; false for symbolic. "
#         "Final answer must be sympy-parseable."
#     ),
#     # ── UPDATED: hotpot planner now knows about wikipedia tool ──
#     "hotpot": (
#         "2-hop QA: exactly 2 subtasks, both wikipedia. "
#         "search_query = entity name only. s2.depends_on=[\"s1\"]. "
#         "Answer: short span — name, number, date, yes, or no."
#     ),
#     # ── UPDATED: musique planner now knows about wikipedia tool ──
#     "musique": (
#         "2–4-hop QA. " + _MUSIQUE_TOPO +
#         "One subtask per hop/sub-hop. search_query = entity name only. "
#         "Max 4 subtasks. No duplicate fetches."
#     ),
# }

# MULTIAGENT_WORKER_RESULT_FORMAT: dict[str, str] = {
#     "gaia": (
#         "result must be a word, name, number, date, or short phrase — "
#         "the exact value that answers the subtask goal."
#     ),
#     "mmlu_pro": (
#         "If your subtask is to identify the correct option, result must be the "
#         "single option letter (A through J). "
#         "For analytical subtasks, result is a short factual statement."
#     ),
#     # ── UPDATED: math worker must return sympy-parseable expressions ──
#     "math": (
#         "result: sympy-parseable expression — '42','3/4','x**2+1','sqrt(3)/2'. "
#         "calculator inputs must be pure Python — no units or words."
#     ),
#     # ── UPDATED: hotpot worker must return Wikipedia-grounded short facts ──
#     "hotpot": (
#         "result: short span — name, number, date, yes, or no — "
#         "copied exactly from the Wikipedia passage. Do not paraphrase or invent."
#     ),
#     # ── UPDATED: musique worker must return Wikipedia-grounded short facts ──
#     "musique": (
#         "result: exact Wikipedia span (name/number/date/phrase). "
#         "evidence: entity name searched — for downstream hop verification."
#     ),
# }

# MULTIAGENT_JUDGE_ANSWER_FORMAT: dict[str, str] = {
#     "gaia": (
#         "answer must be a word, name, number, date, or short phrase — "
#         "exact string that will be matched against the reference answer."
#     ),
#     "mmlu_pro": (
#         "answer must be the single option letter only — A, B, C, D, E, F, G, H, I, or J. "
#         "No other text. No explanation."
#     ),
#     "math": (
#         "answer must be a simplified mathematical expression in Python-compatible "
#         "notation that sympy can parse — for example '42', '3/4', 'x**2 + 1', "
#         "'sqrt(3)/2', 'pi/4'. Do not write 'the answer is'. Do not write prose."
#     ),
#     "hotpot": (
#         "answer must be a short span — name, number, date, yes, or no. "
#         "Verify that both hop results are consistent and together entail this answer. "
#         "No full sentences."
#     ),
#     "musique": (
#         "answer: short span. Verify full dependency graph: "
#         "parallel→both roots grounded independently; "
#         "branching→s2A and s2B each trace to s1. No sentences."
#     )
# }

# # Judge output schema — unchanged
# MULTIAGENT_JUDGE_SCHEMA: str = (
#     '{"answer":"<bare value>",'
#     '"confidence":<float 0-1, certainty in this answer>,'
#     '"complexity":<float 0-1, difficulty of the question>,'
#     '"consensus_score":<float 0-1>,'
#     '"rationale":"1-2 sentences",'
#     '"conflict_resolution":"none or explanation"}'
# )


# @dataclass(frozen=True)
# class MultiagentPrompts:
#     """Role prompts assembled once in MultiAgentAgent.__init__."""
#     planner_system: str
#     worker_system: str
#     judge_system: str
#     extractor_system: str
#     structured_extractor_template: str
#     judge_fallback_system: str


# def build_multiagent_prompts(
#     dataset: str,
#     *,
#     tools_csv: str,
#     tool_enum: str,
# ) -> MultiagentPrompts:
#     """Single entry point: all multi-agent role system prompts for one dataset."""
#     if dataset not in DATASETS:
#         raise KeyError(f"Unknown dataset {dataset!r}; expected one of {DATASETS}")

#     strategy   = MULTIAGENT_PLANNER_STRATEGY[dataset]
#     result_fmt = MULTIAGENT_WORKER_RESULT_FORMAT[dataset]
#     answer_fmt = MULTIAGENT_JUDGE_ANSWER_FORMAT[dataset]

#     planner_system = (
#         "Planner in a multi-agent QA system. "
#         "Return ONLY strict JSON (no markdown):\n"
#         '{"final_constraint":"","subtasks":[{"id":"s1","goal":"...",'
#         '"focus":"reasoning|factual|calculation","needs_tool":true,'
#         f'"tool_name":"none","search_query":"","depends_on":[]}}]}}\n'
#         f"tool_name ∈ {tool_enum}. Available tools: {tools_csv}\n"
#         "needs_tool=true only when external data/computation is required. "
#         "No duplicate subtasks — use depends_on instead of re-fetching.\n"
#         f"STRATEGY: {strategy}"
#     )

#     worker_system = (
#         "Specialist Worker in a multi-agent QA system.\n"
#         "Solve ONLY your assigned subtask. Draft → challenge → finalise.\n"
#         "Return strict JSON only:\n"
#         '{"subtask_id":"...","result":"...","confidence":<0-1>,'
#         '"evidence":"≤15 words","self_critique":"one sentence"}\n'
#         f"RESULT FORMAT: {result_fmt}"
#     )

#     judge_system = (
#         "Judge in a multi-agent QA system.\n"
#         "Review working memory, cross-validate facts, produce ONE concise answer.\n"
#         "If a high-confidence worker produced a direct factual answer, "
#         "prefer it unless conflicting evidence exists. "
#         "Do NOT reinterpret symbolic answers.\n"
#         "Return ONLY strict JSON:\n"
#         f"{MULTIAGENT_JUDGE_SCHEMA}\n"
#         f"ANSWER FORMAT: {answer_fmt}\n"
#         "Numbers: digits only. Names: name only. Boolean: yes/no. Unknown: unknown. "
#         "No preamble in answer field."
#     )

#     return MultiagentPrompts(
#         planner_system=planner_system,
#         worker_system=worker_system,
#         judge_system=judge_system,
#         extractor_system=(
#             "Fact extractor. Return ONLY sentences/numbers/names that DIRECTLY answer "
#             "the question. Preserve exact values. If nothing is relevant: NO_RELEVANT_CONTENT\n"
#             "Max 150 words. No preamble."
#         ),
#         structured_extractor_template=(
#             'Extract fields [{fields_str}] as compact JSON. '
#             "null for missing. Exact values. No markdown."
#         ),
#         judge_fallback_system=(
#             "Answer extractor. Return ONLY the final answer — no explanation. "
#             "If no answer: unknown"
#         ),
#     )


# # ─────────────────────────────────────────────────────────────────────────────
# # DEBATE AGENT TEMPERATURE — dataset-specific (unchanged)
# # ─────────────────────────────────────────────────────────────────────────────

# DEBATE_AGENT_TEMPERATURE: dict[str, float] = {
#     "gaia":     0.7,
#     "mmlu_pro": 0.5,
#     "math":     0.3,
#     "hotpot":   0.7,
#     "musique":  0.7,
# }


# # ─────────────────────────────────────────────────────────────────────────────
# # VALIDATION — runs at import time
# # ─────────────────────────────────────────────────────────────────────────────

# def _validate_prompts() -> None:
#     missing: list[str] = []

#     for dataset in DATASETS:
#         for agent in AGENTS:
#             if dataset not in SYSTEM_PROMPT or agent not in SYSTEM_PROMPT.get(dataset, {}):
#                 missing.append(f"SYSTEM_PROMPT['{dataset}']['{agent}']")
#             if dataset not in USER_PROMPT or agent not in USER_PROMPT.get(dataset, {}):
#                 missing.append(f"USER_PROMPT['{dataset}']['{agent}']")

#     for dataset in DATASETS:
#         if dataset not in DEBATE_SYNTHESIS_PROMPT:
#             missing.append(f"DEBATE_SYNTHESIS_PROMPT['{dataset}']")
#         if dataset not in MULTIAGENT_PLANNER_STRATEGY:
#             missing.append(f"MULTIAGENT_PLANNER_STRATEGY['{dataset}']")
#         if dataset not in MULTIAGENT_WORKER_RESULT_FORMAT:
#             missing.append(f"MULTIAGENT_WORKER_RESULT_FORMAT['{dataset}']")
#         if dataset not in MULTIAGENT_JUDGE_ANSWER_FORMAT:
#             missing.append(f"MULTIAGENT_JUDGE_ANSWER_FORMAT['{dataset}']")
#         if dataset not in DEBATE_AGENT_TEMPERATURE:
#             missing.append(f"DEBATE_AGENT_TEMPERATURE['{dataset}']")
#         try:
#             build_multiagent_prompts(dataset, tools_csv="none", tool_enum="none")
#         except KeyError as exc:
#             missing.append(str(exc))

#     if missing:
#         raise KeyError(
#             "Missing prompt entries — add these before running:\n"
#             + "\n".join(f"  {m}" for m in missing)
#         )


# _validate_prompts()










# from __future__ import annotations


# DATASETS = ["gaia", "mmlu_pro", "math", "hotpot", "musique"]
# AGENTS = ['raw', 'cot', 'react', 'multiagent', 'self_consistency','debate']



# SYSTEM_PROMPT: dict[str, dict[str, str]] = {
#     "gaia": {
#         "raw": ( "Given a query, an expert in handling real world everyday situation agent, answer the query."
#             "Return exactly one JSON object with this exact schema and no extra text: "
#             '{{\"answer\":\"<short value>\",\"confidence\":<0_to_1_float>(2 short pointers),\"complexity\":<0_to_1_float>(2 short pointers)>}}. '
#             "Answer must be a word, name, number, date, or short phrase "
#         ),  

#         "cot": (
#             "You are a multi-hop reasoning expert and an expert QA agent. Given a question, and gather the relevant context, think step-by-step. "
#             "Return exactly one JSON object with this exact schema and no extra text: "
#             '{{\"answer\":\"<short value>\",\"confidence\":<0_to_1_float>(2 short pointers),\"complexity\":<0_to_1_float>(2 short pointers)}}.'
#             "Answer must be a word, name, number, date, or short phrase "
#         ),

#         "react": (
#             """You are a ReAct (Reasoning + Acting + Observation) agent: alternate structured thinking with tool use, one step at a time.

#             ## CYCLE FORMAT  (strictly follow this order, every turn)
#             Thought: What you know, what is missing, which tool fills the gap — concise.
#             Action: tool_name[input]

#             The runtime injects (never write this yourself):
#             Observation: <tool result>

#             Repeat Thought → Action → receive Observation until you can answer from Observations.

#             After each Observation, use Thought to summarize only relevant information needed for the question (minimal identifiers, names, numbers)—do not copy long tool output verbatim.
#             Observation remains authoritative; never state facts that are not grounded in it.

#             ## Final turn only (grounded answer)
#             Thought: Briefly how Observations support your answer.
#             Action: finish[<JSON>]
#             <JSON> is one JSON object (no markdown fences) with:
#             {{"answer":"<short value>","confidence":<float 0-1>,"complexity":<float 0-1>}}
#             - answer: word, name, number, date, or short phrase.
#             - confidence and complexity: numeric, in [0, 1].

#             Example: Action: finish[{{"answer":"42","confidence":0.9,"complexity":0.35}}]

#             Do not use a separate "Final Answer:" line. Do not emit bare JSON without Action: finish[...].

#             Available tools:
#             {tools_block}
#             Each tool uses tool_name[input]. Prefer internal reasoning when it is clearly cheaper than a tool call.
#             EXTRACTION (files / long Observations)
#             1. Scan the entire Observation top to bottom.
#             2. Extract everything matching the question; preserve order of appearance.
#             3. Copy values exactly as found — no invented formatting.

#             RULES
#             1. Every assistant message: Thought: then Action: (same message).
#             2. Exactly one Action per turn — no chained tools.
#             3. Never invent tool outputs.
#             4. Avoid unnecessary tools; do not repeat the exact same Action twice.
#             5. finish only when the answer is supported by Observations (or direct inference from them).
#             6. If the question includes \"Attached file name for reference:\" → read_file with that exact basename only (often a UUID like abc-123.pdf), e.g. read_file[abc-123.pdf]. Never invent filenames like report.pdf or standards.pdf.

#             SELF-CHECK (before Action: finish[...])
#             - Answer traces to Observations; JSON valid; confidence and complexity in [0,1]; finish payload is JSON only inside the brackets."""
#         ),
#         "multiagent": (
#             "You are the coordinator of a multi-agent QA workflow with three stages: "
#             "Planner -> Workers -> Judge/Synthesizer. "
#             "All intermediate outputs must be strict JSON. "
#             "Planner JSON: "
#             '{"subtasks":[{"id":"s1","goal":"...","focus":"reasoning|factual|calculation"}]}. '
#             "Worker JSON: "
#             '{"subtask_id":"s1","result":"...","confidence":0.0,"evidence":"..."}. '
#             "Judge JSON: "
#             '{"answer":"<value>","consensus_score":0.0,"rationale":"..."}. '
#             'The final answer must be exactly one JSON object: {{"answer":"<value>"}}. '
#             "Do not output markdown or extra commentary."
#         ),
#         "self_consistency": (
#             "You are a self-consistency agent that solves real-world questions."
#             "Always: (1)  Think through problems step by step and give a precise answer. "
#             "(2) end with exactly one JSON object and nothing after it. "
#             "CRITICAL: "
#             "- Always give your final answer in the EXACT unit the question asks for"
#             "- If the question asks how many X, answer as a plain number of X"
#             "- Never convert units unless the question asks you to"
#             "- Re-read the question before writing your final answer"
#             "Schema: "
#             '{{"answer":"<short value>","confidence":<number 0-1>,"complexity":<number 0-1>}}. '
#             "The answer must be only what the question requests: a word, name, number, date, "
#             "or short phrase — no full sentences inside the answer field. "
#             "Do not use markdown code fences around the JSON."
#         ),
#     },




#     "mmlu_pro": {
#         "raw":        "...",
#         "cot":        "...",
#         "react":      "...",
#         "multiagent": (
#             "You are a multi-agent coordinator using Planner -> Workers -> Judge. "
#             "Planner, worker, and judge outputs must be strict JSON. "
#             'Final output must be exactly {{"answer":"<value>"}} with no extra text.'
#         ),
#         "self_consistency": (
#             "Self-consistency decoding (Wang et al., 2023): step-by-step chain, then one JSON line. "
#             "You solve difficult multiple-choice questions. Use explicit step-by-step reasoning, "
#             "then output exactly one JSON object (no trailing commentary). Schema: "
#             '{{"answer":"<choice letter or exact option text>","confidence":<0-1>,"complexity":<0-1>}}. '
#             "If the question lists options A/B/C/…, the answer field should be the single best "
#             "letter unless the question demands the full option string. No markdown fences."
#         ),
#     },
#     "math": {
#         "raw":        "...",
#         "cot":        "...",
#         "react":      "...",
#         "multiagent": (
#             "You are a multi-agent coordinator using Planner -> Workers -> Judge. "
#             "Planner, worker, and judge outputs must be strict JSON. "
#             'Final output must be exactly {{"answer":"<value>"}} with no extra text.'
#         ),
#         "self_consistency": (
#             "Self-consistency decoding (Wang et al., 2023): step-by-step chain, then one JSON line. "
#             "You solve hard mathematics problems. Show clear intermediate reasoning, then "
#             "give one final JSON line. Schema: "
#             '{{"answer":"<simplified result>","confidence":<0-1>,"complexity":<0-1>}}. '
#             "Put the final numeric result or simplified expression in answer as a string "
#             '(e.g. "42", "3/4", "x=2"). No markdown fences.'
#         ),
#     },
#     "swe_bench_verified": {
#         "raw":        "...",
#         "cot":        "...",
#         "react":      "...",
#         "multiagent": (
#             "You are a multi-agent coordinator using Planner -> Workers -> Judge. "
#             "Planner, worker, and judge outputs must be strict JSON. "
#             'Final output must be exactly {{"answer":"<value>"}} with no extra text.'
#         ),
#         "self_consistency": (
#             "Self-consistency decoding (Wang et al., 2023): step-by-step chain, then one JSON line. "
#             "You analyze software engineering tasks (bugs, tests, patches). Reason step by "
#             "step, then output one JSON object. Schema: "
#             '{{"answer":"<short technical phrase>","confidence":<0-1>,"complexity":<0-1>}}. '
#             "The answer should capture the root cause, the key fix, or the failing behavior in "
#             "a short phrase suitable for aggregation — not a patch diff. No markdown fences."
#         ),
#     },
# }



# USER_PROMPT: dict[str, dict[str, str]] = {
#     "gaia": {
#         "raw": (
#             "Question: {query}\n"
#         ),
#         "cot": (
#             "Question: {query}\n\n"
#             "Think step by step. Show your concise reasoning before answering.\n"
#             "The final answer must be one JSON object: {{\"answer\": \"<value>\"}}."
#         ),
#         "react": (
#             "Question: {query}\n" 
#             "The answer <value> must be a word, number, name, date, or short phrase."
#             "check the attachement file only if it is provided and is relevant to the question."
#         ),

#         "multiagent": (
#             "Question: {query}\n\n"
#             "Use Planner -> Workers -> Judge internally. "
#             'Return exactly one JSON object: {{"answer":"<value>"}}.'
#         ),
#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Let's think step by step.\n\n"
#             '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
#         ),
#     },
#     "mmlu_pro": {
#         "raw": "...",
#         "cot": "...",
#         "react": "...",
#         "multiagent": (
#             "Question: {query}\n\n"
#             "Use Planner -> Workers -> Judge internally. "
#             'Return exactly one JSON object: {{"answer":"<value>"}}.'
#         ),
#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Let's think step by step.\n\n"
#             "Use numbered reasoning steps. Eliminate wrong options when applicable; cite the "
#             "definition or fact that decides between the remaining choices.\n\n"
#             "Last line only — one JSON object:\n"
#             '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
#         ),
#     },
#     "math": {
#         "raw": "...",
#         "cot": "...",
#         "react": "...",
#         "multiagent": (
#             "Question: {query}\n\n"
#             "Use Planner -> Workers -> Judge internally. "
#             'Return exactly one JSON object: {{"answer":"<value>"}}.'
#         ),
#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Let's think step by step.\n\n"
#             "Use numbered steps. Show key equations or substitutions. Before the JSON, "
#             "sanity-check units, signs, and edge cases when relevant.\n\n"
#             "Last line only — one JSON object:\n"
#             '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
#         ),
#     },
#     "swe_bench_verified": {
#         "raw": "...",
#         "cot": "...",
#         "react": "...",
#         "multiagent": (
#             "Question: {query}\n\n"
#             "Use Planner -> Workers -> Judge internally. "
#             'Return exactly one JSON object: {{"answer":"<value>"}}.'
#         ),
#         "self_consistency": (
#             "Question: {query}\n\n"
#             "Let's think step by step.\n\n"
#             "Use numbered steps: symptoms → likely location → hypothesis → what would fix or "
#             "verify. Stay grounded in the statement; do not invent file paths or stack traces "
#             "not implied by the question.\n\n"
#             "Last line only — one JSON object:\n"
#             '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
#         ),
#     },
# }

