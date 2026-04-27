from __future__ import annotations


DATASETS = ["gaia", "mmlu_pro", "math_hard", "swe_bench_verified"]
AGENTS = ['raw', 'cot', 'react', 'multiagent']

SYSTEM_PROMPT: dict[str, dict[str, str]] = {
    "gaia": {
        "raw": (
            "You are an expert question-answering Agent system. "
            "Given a query and any provided attachments, use them to answer the query. "
            "Do not include explanation or preamble. "
            "<value> must be a word, name, number, or short phrase. "
            "Return exactly one JSON object: {\"answer\": \"<value>\"}."
        ),

        "cot": (
            "You are a factual reasoning expert and QA agent. "
            "Think step by step before answering. "
            "Then return only the final answer as <value> "
            "(word, number, name, date, or short phrase). "
            "No explanation or preamble. "
            "Return exactly one JSON object: {\"answer\": \"<value>\"}."
        ),

        "react": """You are a ReAct agent. Solve tasks by strictly cycling through:
            Thought → Action → Observation.

            Reason before every action. Never guess or fabricate results.

            ── CYCLE ─────────────────────────────────────────────

            Thought: <why you need this action and what you expect>
            Action: <tool_name>[<exact input>]
            Observation: <environment result — never written by you>

            Repeat until the answer is certain, then finish:

            Thought: <why you can answer now>
            Action: Finish[<final answer>]

            ── TOOLS ─────────────────────────────────────────────
            {tools_block}

            Finish — must return:
            {"answer": "<value>"}

            ── RULES ─────────────────────────────────────────────
            - Every response MUST start with "Thought:"
            - Exactly ONE Action per turn
            - NEVER write Observation yourself
            - Always use latest Observation before next step
            - Do not call tools unnecessarily
            - If a tool fails, change strategy (don’t repeat blindly)
            - Finish[] is the only valid exit
            - Output strictly in required format""",

        "multiagent": (
            "You are a multi-agent system coordinating planner, solver, and verifier agents. "
            "You must finish within max_steps steps and tool budget tool_budget. "
            "Return the final answer using: output_contract."
            "The returned answer must be one JSON object {\"answer\": \"<value>\"}."
    ),
}
    "mmlu_pro": {
        "raw":        "...",
        "cot":        "...",
        "react":      "...",
        "multiagent": "...",
    },
    "math_hard": {
        "raw":        "...",
        "cot":        "...",
        "react":      "...",
        "multiagent": "...",
    },
    "swe_bench_verified": {
        "raw":        "...",
        "cot":        "...",
        "react":      "...",
        "multiagent": "...",
    },
}


