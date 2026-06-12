"""
All prompt text construction for adaptive reasoning-strategy routing.

Layers elsewhere: policy (tools, max_steps), tools (retrieval), router (strategy).
"""

from __future__ import annotations

from dataclasses import dataclass

from config.local.constants import ALL_AGENTS, BAD_ANSWERS, DS

# ── Task identity (dataset objective only) ───────────────────────────────────

TASK_DESCRIPTION: dict[str, str] = {
    "gaia": "Answer the real-world question.",
    "mmlu": "Answer the multiple-choice question.",
    "math": "Solve the mathematical problem.",
    "hotpot": "Answer the multi-hop question.",
    "musique": "Answer the multi-hop question.",
}

DATASETS: list[str] = sorted(DS.names)
AGENTS: list[str] = list(ALL_AGENTS)


# ── Output contract (grading / fair comparison across strategies) ────────────

_JSON_EXAMPLES: dict[str, str] = {
    "gaia": '{"answer":"March 2012","confidence":0.85,"complexity":0.5}',
    "mmlu": '{"answer":"C","confidence":0.8,"complexity":0.3}',
    "math": '{"answer":"21","confidence":0.9,"complexity":0.4}',
    "hotpot": '{"answer":"Paris","confidence":0.9,"complexity":0.3}',
    "musique": '{"answer":"1954","confidence":0.85,"complexity":0.4}',
}

ANSWER_FORMAT: dict[str, str] = {
    "gaia": "word, name, number, date, or short phrase",
    "mmlu": "single option letter A-J",
    "math":     "exact mathematical expression preserving the representation "
    "requested by the problem "
    "(e.g. 42, 3/4, sqrt(2)/2, 1_6)",
    "hotpot": "short answer span: name, number, date, yes, or no",
    "musique": "short factual answer span: name, number, date, or short phrase",
}


def task_block(dataset: str) -> str:
    return (
        f"Task:\n"
        f"{TASK_DESCRIPTION[dataset]}\n\n"
        f"{representation_rules(dataset)}"
    )

def json_footer(dataset: str) -> str:
    return (
        "Return exactly one JSON object and nothing after it:\n"
        f"Example: {_JSON_EXAMPLES[dataset]}\n"
        f"answer: {ANSWER_FORMAT[dataset]}\n"
        "No markdown fences."
    )


# Legacy alias for modules that import OUTPUT_SCHEMA
OUTPUT_SCHEMA = _JSON_EXAMPLES["math"]


# ── Single-shot / CoT / debate reasoning formats ──────────────────────────────

def build_raw_system(dataset: str) -> str:
    return (
        "You are a reasoning agent.\n\n"
        f"{task_block(dataset)}\n\n"
        "Answer concisely and directly.\n"
        "Do not explain your reasoning.\n"
        "Preserve the representation requested by the problem.\n\n"
        f"{json_footer(dataset)}"
    )


def build_cot_system(dataset: str) -> str:
    return (
        "You are a reasoning agent.\n\n"
        f"{task_block(dataset)}\n\n"
        "Think step by step before answering.\n"
        "Verify intermediate reasoning before the final answer.\n"
        "Preserve the representation requested by the problem.\n\n"
        f"{json_footer(dataset)}"
    )


def build_self_consistency_system(dataset: str) -> str:
    return build_cot_system(dataset)


def build_debate_system(dataset: str) -> str:
    return (
        "You are a reasoning agent.\n\n"
        f"{task_block(dataset)}\n\n"
        "Independent expert — reason on your own; you do not see other agents.\n\n"
        f"{json_footer(dataset)}"
    )


def representation_rules(dataset: str) -> str:

    if dataset == "math":
        return (
            "Representation rules:\n"
            "- preserve the exact representation requested by the problem\n"
            "- base notation: 1_6 (do not convert to decimal unless asked)\n"
            "- fractions: 3/4; radicals: sqrt(2); coordinates: (a,b)\n"
            "- return the simplest final form\n"
            "- use plain-text mathematical formatting\n"
            "- do NOT use LaTeX, Unicode math symbols, or markdown\n"
        )

    if dataset == "mmlu":
        return (
            "Representation rules:\n"
            "- return EXACTLY one option letter: A,B,C,D,E,F,G,H,I,J\n"
            "- do not explain the answer\n"
        )

    if dataset == "hotpot":
        return (
            "Representation rules:\n"
            "- use ONLY the provided Context paragraphs as evidence\n"
            "- copy the shortest supporting span from evidence\n"
            "- do not paraphrase or add words not in evidence\n"
            "- preserve punctuation, titles, and suffixes\n"
            "- when multiple candidates appear, pick the shortest span that answers the question\n"
            "- do not return intermediate hops when the question asks for the final entity\n"
        )

    if dataset == "musique":
        return (
            "Representation rules:\n"
            "- use ONLY the provided Context paragraphs as evidence\n"
            "- copy the exact evidence span — do not paraphrase\n"
            "- preserve punctuation, titles, and suffixes (e.g. Leo Varadkar, TD)\n"
            "- when multiple candidate answers appear, verify which satisfies the relation\n"
            "- do not return intermediate hops\n"
        )

    if dataset == "gaia":
        return (
            "Representation rules:\n"
            "- Final Answer JSON must use exactly the key answer (no other field names)\n"
            "- never use None, null, unknown, or N/A as the answer value\n"
            "- if the question asks for a number only, return only that number in answer\n"
            "- preserve units, dates, names, and formatting exactly from evidence\n"
            "- do not add unnecessary explanation in the answer field\n"
        )

    return ""

# ── ReAct reasoning format + tool syntax ─────────────────────────────────────

_REACT_PROTOCOL = (
    "Each completion must be ONE of:\n\n"
    "1. Tool step\n"
    "Thought: (your reasoning)\n"
    "Action: tool[input]\n\n"
    "2. Final step\n"
    "Thought: (brief justification from Observations)\n"
    "Final Answer:\n"
    "(one JSON object — use a real answer, never template placeholders)\n\n"
    "Never generate Observation lines. "
    "One Action or one Final Answer per completion — not both."
)

_REACT_RULES = (
    "Rules:\n"
    "- use only facts from runtime Observations or Context\n"
    "- preserve the representation requested by the problem\n"
    "- Final Answer must be one JSON object with an answer field (see example schema)\n"
    "- avoid repeating the same or nearly identical tool queries\n"
    "- the runtime adds Observation lines; never write them yourself"
)

_REACT_OPEN_WEB_META = (
    "Open-web:\n"
    "- after web_search, use web_fetch on a promising URL before searching again\n"
    "- python_exec must print() results so output appears in Observations\n"
    "- do not print None as a substitute for a real answer; keep searching or reason from fetched text\n"
    "- when Observations already contain enough evidence, stop tools and use Final Answer\n\n"
)

_REACT_GAIA_META = (
    "GAIA workflow:\n"
    "- read the question literally (e.g. number-only means digits only in answer)\n"
    "- prefer primary sources named in the question over generic search snippets\n"
    "- avoid guessing from unrelated pages (e.g. YouTube policy pages when the question cites a specific site)\n\n"
)

# ── ReAct runtime feedback (parser / executor → model on format retry) ───────

REACT_BAD_FINAL_ANSWERS = BAD_ANSWERS

REACT_FORMAT_RETRY_MESSAGES: dict[str, str] = {
    "empty_output": "Output was empty. Use Thought + Action or Thought + Final Answer.",
    "invented_observation": (
        "Do not write Observation lines. The runtime adds Observations after Actions."
    ),
    "multiple_actions": "Exactly one Action: line per turn.",
    "multiple_final_answers": "Exactly one Final Answer: line per turn.",
    "action_and_final_together": (
        "Use either Action: tool[input] OR Final Answer: in one turn, not both."
    ),
    "action_with_final": "Tool turns cannot include Final Answer.",
    "missing_thought": "Start with Thought: before Action or Final Answer.",
    "missing_action_or_final": (
        "Start with Thought:, then Action: tool[input] or Final Answer: plus JSON. "
        "Bare tool[input] is accepted only as shorthand (e.g. retrieve[query])."
    ),
    "malformed_action": (
        "Action must be tool_name[input] with balanced [...]. "
        "python_exec may span multiple lines inside the brackets."
    ),
    "malformed_final": "Final Answer must be followed by one JSON object.",
    "extra_content_after_action": (
        "After Action: tool[input], stop. Wait for the runtime Observation. "
        "For python_exec only, the input may span multiple lines inside [...]."
    ),
    "repeated_search": (
        "You are repeating the same or nearly identical tool query. Reformulate the "
        "query, try a different tool or entity, or answer from evidence already in "
        "the scratchpad."
    ),
    "finalize_pressure": (
        "You already have enough evidence. Stop searching and provide Final Answer now."
    ),
    "tool_stagnation": (
        "No new evidence obtained. Use existing evidence or finalize now."
    ),
    "math_finalize_nudge": (
        "You have computed enough. Use Final Answer on the next turn."
    ),
    "empty_action_input": "Action input inside [...] cannot be empty.",
    "placeholder_answer": (
        "Final Answer JSON must contain a real computed answer "
        '(e.g. {{"answer":"1_6","confidence":0.9,"complexity":0.4}}), '
        "not placeholders like <value> or empty templates."
    ),
    "missing_answer_field": (
        "Final Answer JSON must include an answer field with the result value "
        '(e.g. {{"answer":"8","confidence":0.9,"complexity":0.3}}).'
    ),
}

def react_format_retry_observation(reason: str) -> str:
    """Observation text fed back to the model after a protocol violation."""
    return REACT_FORMAT_RETRY_MESSAGES.get(reason, reason)


def react_loop_observation(reason: str) -> str:
    """Runtime loop-control notice appended as an Observation (not a format retry)."""
    return REACT_FORMAT_RETRY_MESSAGES.get(reason, reason)


REACT_FINALIZE_PRESSURE = REACT_FORMAT_RETRY_MESSAGES["finalize_pressure"]
REACT_TOOL_STAGNATION = REACT_FORMAT_RETRY_MESSAGES["tool_stagnation"]
REACT_MATH_FINALIZE_NUDGE = REACT_FORMAT_RETRY_MESSAGES["math_finalize_nudge"]


def _react_final_block(dataset: str) -> str:
    return (
        "Final Answer:\n"
        f"{json_footer(dataset)}"
    )

_REACT_RETRIEVE_EXAMPLE = (
    "Example valid tool turn:\n"
    "Thought: I need evidence about the setting years of Dreamland.\n"
    "Action: retrieve[Dreamland novel setting years]\n\n"
    "Invalid retrieve query (paragraph numbers):\n"
    "Action: retrieve[1]\n"
    "Do not use paragraph numbers as retrieve queries.\n"
    "Preferred format is Thought: then Action:; bare retrieve[query] is accepted only as one-line shorthand.\n\n"
)

_REACT_PROVIDED_CONTEXT = (
    "Evidence: use ONLY the Context paragraphs below (not the open web).\n"
    "Use retrieve with short semantic queries over those paragraphs.\n"
    "Do not answer from outside knowledge.\n\n"
    f"{_REACT_RETRIEVE_EXAMPLE}"
)

_REACT_TOOL_EXAMPLE = (
    "Example valid tool turn:\n"
    "Thought: I need the release year.\n"
    "Action: web_search[movie release year]\n\n"
)

_REACT_GAIA_PYTHON_EXAMPLE = (
    "Example valid multiline python_exec turn:\n"
    "Thought: I will load the spreadsheet and count matching rows.\n"
    "Action: python_exec[\n"
    "import pandas as pd\n"
    "df = pd.read_excel('file.xlsx')\n"
    "print(len(df))\n"
    "]\n\n"
)

_REACT_MATH_EXAMPLE = (
    "Example valid tool turn:\n"
    "Thought: I should compute the absolute difference.\n"
    "Action: math_tool[abs(3-5)]\n\n"
    "Example valid final turn:\n"
    "Thought: The computation supports the result.\n"
    'Final Answer:\n{{"answer":"2","confidence":0.9,"complexity":0.4}}\n\n'
)

_REACT_MMLU_EXAMPLE = (
    "Example valid final turn:\n"
    "Thought: Option C is best supported by the passage.\n"
    'Final Answer:\n{{"answer":"C","confidence":0.8,"complexity":0.3}}\n\n'
)

_REACT_LOOP = (
    f"{_REACT_PROTOCOL}\n\n"
    "{provided_context}"
    "The runtime supplies Observations after each Action.\n\n"
    "Tools:\n{{tools_block}}\n\n"
    "{react_final}\n\n"
    f"{_REACT_RULES}"
)

_REACT_MMLU_LOOP = (
    f"{_REACT_PROTOCOL}\n\n"
    f"{_REACT_MMLU_EXAMPLE}"
    "No tools — Thought and Final Answer only.\n\n"
    "{react_final}\n\n"
    f"{_REACT_RULES}"
)

_REACT_MATH_LOOP = (
    f"{_REACT_PROTOCOL}\n\n"
    f"{_REACT_MATH_EXAMPLE}"
    "The runtime supplies Observations after each Action.\n\n"
    "Tools:\n{{tools_block}}\n\n"
    "{react_final}\n\n"
    f"{_REACT_RULES}"
)


def build_react_system(dataset: str) -> str:
    final = _react_final_block(dataset)
    if dataset in ("hotpot", "musique"):
        provided = _REACT_PROVIDED_CONTEXT
    elif dataset == "gaia":
        provided = (
            _REACT_TOOL_EXAMPLE
            + _REACT_GAIA_PYTHON_EXAMPLE
            + _REACT_OPEN_WEB_META
            + _REACT_GAIA_META
        )
    else:
        provided = ""
    if dataset == "mmlu":
        body = _REACT_MMLU_LOOP.format(react_final=final)
    elif dataset == "math":
        body = _REACT_MATH_LOOP.format(react_final=final)
    else:
        body = _REACT_LOOP.format(react_final=final, provided_context=provided)
    return (
        "You are a ReAct agent.\n\n"
        f"{task_block(dataset)}\n\n"
        f"{body}"
    )


# ── Verify: debate synthesis + multi-agent roles ─────────────────────────────

_MA_PLANNER_JSON = (
    '{"final_constraint":"","subtasks":['
    '{"id":"s1","goal":"...","focus":"reasoning|factual|calculation",'
    '"needs_tool":false,"tool_name":"none","search_query":"",'
    '"candidate_url":"","depends_on":[]}]}'
)
_MA_WORKER_JSON = (
    '{"subtask_id":"s1","result":"1954","confidence":0.75,'
    '"evidence":"found in paragraph 2","self_critique":"verified year"}'
)
_MA_JUDGE_JSON = (
    '{"answer":"Paris","confidence":0.85,"complexity":0.35,'
    '"consensus_score":0.9,'
    '"rationale":"Worker factual lane matched evidence.",'
    '"conflict_resolution":"none"}'
)

DEBATE_SYNTHESIS_SYSTEM = (
    "Impartial judge. Choose the better-supported answer.\n"
    f"Return one JSON object. Example: {_JSON_EXAMPLES['hotpot']}\n"
)

DEBATE_AGENT_TEMPERATURE: dict[str, float] = {
    "gaia": 0.7,
    "mmlu": 0.5,
    "math": 0.3,
    "hotpot": 0.7,
    "musique": 0.7,
}


def build_debate_synthesis_prompt(dataset: str) -> str:
    return (
        "Two reasoners disagreed.\n\n"
        "Question: {question}\n\nReasoner A:\n{answer_a}\n\nReasoner B:\n{answer_b}\n\n"
        f"Return one JSON object. Example: {_JSON_EXAMPLES[dataset]}\n"
        f"answer: {ANSWER_FORMAT[dataset]}. No fences."
    )


DEBATE_SYNTHESIS_PROMPT: dict[str, str] = {
    ds: build_debate_synthesis_prompt(ds) for ds in DATASETS
}


def build_verify_system(dataset: str) -> str:
    """Multi-agent judge role system prompt."""
    return (
        "Judge. Reconcile known_facts from workers and return one final answer.\n"
        "Do NOT perform new reasoning, invent facts, or paraphrase evidence spans.\n"
        "Prefer facts from terminal subtasks (sink nodes — last in the dependency chain).\n"
        "Do NOT return intermediate-hop entities when a terminal subtask answers the question.\n"
        "Answer MUST be copied verbatim from a known_facts text span — never invent facts.\n"
        "Match root question answer type: When→date/year/era; Where→place/region span; "
        "Which country→country name only (not city, country).\n"
        "Prefer the shortest exact supported span from tool-backed evidence.\n"
        "Apply representation rules to the answer field before returning.\n"
        "Return strict JSON:\n"
        f"{_MA_JUDGE_JSON}\n"
        f"{task_block(dataset)}\n"
        f"answer: {ANSWER_FORMAT[dataset]}."
    )


def build_multiagent_system(dataset: str) -> str:
    return (
        "Multi-agent QA: Planner → Workers → Judge. Strict JSON at each stage.\n\n"
        f"{task_block(dataset)}\n\n"
        f"Final answer type: {ANSWER_FORMAT[dataset]}."
    )


@dataclass(frozen=True)
class MultiagentPrompts:
    planner_system: str
    worker_system: str
    worker_verify_system: str
    judge_system: str
    extractor_system: str
    structured_extractor_template: str
    judge_fallback_system: str


_MA_MAX_SUBTASKS: dict[str, str] = {
    "math": "1–3",
    "hotpot": "2",
    "musique": "2–4",
    "gaia": "3–6",
    "mmlu": "1",
}


def _ma_planner_rules(dataset: str) -> str:
    cap = _MA_MAX_SUBTASKS.get(dataset, "3")
    extra = ""
    if dataset == "mmlu":
        extra = "Collapse to a single reasoning subtask with needs_tool=false.\n"
    elif dataset == "math":
        extra = (
            "Prefer 1 subtask with math_tool unless decomposition is clearly required.\n"
            "Subtask goal must state the exact final answer target from the problem "
            "(e.g. |A-B| in base 6), not only intermediate steps (e.g. find A and B).\n"
        )
    elif dataset in ("hotpot", "musique"):
        extra = (
            "Use retrieve for evidence subtasks; chain hops with depends_on.\n"
            "Decompose inside-out: innermost nested clause = earliest subtask; "
            "root question = terminal subtask.\n"
            "One subtask per hop — do not collapse multi-hop chains into one goal.\n"
            "Preserve relation types: WHERE/place → location; WHEN → date/year/era; "
            "WHO → person; country-question → country name only.\n"
            "Never swap place and time (e.g. 'place where X died' → location, NOT date).\n"
            "Terminal subtask must match the root question relation.\n"
            "Do not bundle two hops in one subtask.\n"
        )
    return (
        "Do NOT solve the problem, infer answers, or summarize evidence.\n"
        "Only: break into minimal subtasks, assign tools, define depends_on.\n"
        f"Max subtasks: {cap}. Subtasks must be executable independently.\n"
        f"{extra}"
    )


def _ma_worker_rules(dataset: str) -> str:
    common = (
        "Solve ONLY your subtask. Copy exact evidence spans — do NOT paraphrase, "
        "summarize, or semantically rewrite.\n"
        "Before returning result: verify the span answers the subtask type "
        "(WHEN→year/date phrase from evidence; WHO→name span; WHAT→shortest action/entity span).\n"
        "Use the shortest exact supported span — omit leading subjects (e.g. Students) "
        "when the question asks what/how/when, not who the subject is.\n"
        "If evidence shows multiple candidate years/entities, do NOT finalize until "
        "you confirm which one satisfies the subtask relation.\n"
    )
    if dataset == "math":
        return (
            common
            + "Use math_tool for computation; result must match the Problem's requested form.\n"
            + "Final answer must be the exact quantity the Problem asks for (not an intermediate value).\n"
        )
    if dataset in ("hotpot", "musique"):
        return (
            common
            + "retrieve → copy the shortest exact supported span from Context.\n"
            + "Preserve punctuation, titles, suffixes (e.g. Leo Varadkar, TD).\n"
        )
    if dataset == "gaia":
        return (
            common
            + "Tool-heavy execution: fetch after search, python_exec for computation.\n"
            + "If question asks for a number only, return digits only.\n"
            + "After 2 failed searches or repeated observations, finalize best grounded answer.\n"
        )
    if dataset == "mmlu":
        return "Reason from the question only; no tools. Return the option letter.\n"
    return common


def _ma_worker_verify_rules(dataset: str) -> str:
    return (
        "Verify one candidate answer against Evidence, Subtask, and Question relation type.\n"
        "Return strict JSON:\n"
        f"{_MA_WORKER_JSON}\n"
        "Relation-aware checks — reject with result=\"\" and confidence=0 when:\n"
        "- Subtask asks WHEN/year/date but candidate is only an event title (not a date span)\n"
        "- Subtask asks named after what/whom but candidate is a place/county name, not the namesake\n"
        "- Candidate entity type mismatches what the subtask relation requires\n"
        "- Wrong granularity (city, country when country-only or region-only is required)\n"
        "When valid: pick the shortest exact span from Evidence that satisfies the relation.\n"
        "Do not add words not present in Evidence.\n"
        f"{representation_rules(dataset)}"
    )


def build_multiagent_prompts(
    dataset: str,
    *,
    tools_csv: str,
    tool_enum: str,
) -> MultiagentPrompts:
    if dataset not in DATASETS:
        raise KeyError(f"Unknown dataset {dataset!r}")

    return MultiagentPrompts(
        planner_system=(
            "Planner. Return ONLY strict JSON:\n"
            f"{_MA_PLANNER_JSON}\n"
            f"tool_name ∈ {tool_enum}. Tools: {tools_csv}\n"
            "needs_tool=true when external data or computation is required.\n"
            "Use depends_on for ordering.\n"
            f"{_ma_planner_rules(dataset)}\n"
            f"{task_block(dataset)}\n"
            f"answer type: {ANSWER_FORMAT[dataset]}."
        ),
        worker_system=(
            f"{_ma_worker_rules(dataset)}"
            "Return strict JSON:\n"
            f"{_MA_WORKER_JSON}\n"
            f"result: exact subtask answer ({ANSWER_FORMAT[dataset]})."
        ),
        worker_verify_system=_ma_worker_verify_rules(dataset),
        judge_system=build_verify_system(dataset),
        extractor_system=(
            "Extract facts that answer the question. "
            "If none: NO_RELEVANT_CONTENT. Max 150 words."
        ),
        structured_extractor_template=(
            "Extract [{fields_str}] as compact JSON. null if missing."
        ),
        judge_fallback_system=(
            "Extract ONLY an answer explicitly marked in the text "
            '(JSON "answer" field or Final Answer line). '
            "Return the exact span. If none: unknown"
        ),
    )


MULTIAGENT_PLANNER_JSON_EXAMPLE = {ds: _MA_PLANNER_JSON for ds in DATASETS}
MULTIAGENT_PLANNER_STRATEGY = {ds: f"Task: {TASK_DESCRIPTION[ds]}" for ds in DATASETS}
MULTIAGENT_WORKER_RESULT_FORMAT = {
    ds: f"result: subtask answer ({ANSWER_FORMAT[ds]})." for ds in DATASETS
}
MULTIAGENT_JUDGE_ANSWER_FORMAT = {ds: f"answer: {ANSWER_FORMAT[ds]}." for ds in DATASETS}
MULTIAGENT_JUDGE_SCHEMA = _MA_JUDGE_JSON

# ── Prompt tables (dataset × strategy) ───────────────────────────────────────

def user_prompt(dataset: str) -> str:
    return "Problem: {query}\n" if dataset == "math" else "Question: {query}\n"


def _build_system_prompts() -> dict[str, dict[str, str]]:
    builders = {
        "raw": build_raw_system,
        "cot": build_cot_system,
        "self_consistency": build_self_consistency_system,
        "debate": build_debate_system,
        "react": build_react_system,
        "multiagent": build_multiagent_system,
    }
    return {
        ds: {agent: fn(ds) for agent, fn in builders.items()}
        for ds in DATASETS
    }


SYSTEM_PROMPT = _build_system_prompts()
USER_PROMPT = {ds: {agent: user_prompt(ds) for agent in AGENTS} for ds in DATASETS}

_SCHEMA = OUTPUT_SCHEMA
_END_JSON = f"End with one JSON object and nothing after it:\n{_JSON_EXAMPLES['math']}\n"


def _validate_prompts() -> None:
    missing: list[str] = []
    for ds in DATASETS:
        for agent in AGENTS:
            if agent not in SYSTEM_PROMPT.get(ds, {}):
                missing.append(f"SYSTEM_PROMPT['{ds}']['{agent}']")
            if agent not in USER_PROMPT.get(ds, {}):
                missing.append(f"USER_PROMPT['{ds}']['{agent}']")
        if ds not in DEBATE_SYNTHESIS_PROMPT:
            missing.append(f"DEBATE_SYNTHESIS_PROMPT['{ds}']")
        try:
            build_multiagent_prompts(ds, tools_csv="none", tool_enum="none")
        except KeyError as exc:
            missing.append(str(exc))
    if missing:
        raise KeyError("Missing prompts:\n" + "\n".join(f"  {m}" for m in missing))


_validate_prompts()
