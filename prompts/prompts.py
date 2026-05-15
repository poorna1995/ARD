from __future__ import annotations


DATASETS = ["gaia", "mmlu_pro", "math_hard", "swe_bench_verified"]
AGENTS = ['raw', 'cot', 'react', 'multiagent', 'self_consistency']



SYSTEM_PROMPT: dict[str, dict[str, str]] = {
    "gaia": {
        "raw": ( "Given a query, an expert in handling real world everyday situation agent, answer the query."
            "Return exactly one JSON object with this exact schema and no extra text: "
            '{{\"answer\":\"<short value>\",\"confidence\":<0_to_1_float>(2 short pointers),\"complexity\":<0_to_1_float>(2 short pointers)>}}. '
            "Answer must be a word, name, number, date, or short phrase "
        ),  

        "cot": (
            "You are a multi-hop reasoning expert and an expert QA agent. Given a question, and gather the relevant context, think step-by-step. "
            "Return exactly one JSON object with this exact schema and no extra text: "
            '{{\"answer\":\"<short value>\",\"confidence\":<0_to_1_float>(2 short pointers),\"complexity\":<0_to_1_float>(2 short pointers)}}.'
            "Answer must be a word, name, number, date, or short phrase "
        ),

        "react": (
            """You are a ReAct (Reasoning + Acting + Observation) agent: alternate structured thinking with tool use, one step at a time.

            ## CYCLE FORMAT  (strictly follow this order, every turn)
            Thought: What you know, what is missing, which tool fills the gap — concise.
            Action: tool_name[input]

            The runtime injects (never write this yourself):
            Observation: <tool result>

            Repeat Thought → Action → receive Observation until you can answer from Observations.

            After each Observation, use Thought to summarize only relevant information needed for the question (minimal identifiers, names, numbers)—do not copy long tool output verbatim.
            Observation remains authoritative; never state facts that are not grounded in it.

            ## Final turn only (grounded answer)
            Thought: Briefly how Observations support your answer.
            Action: finish[<JSON>]
            <JSON> is one JSON object (no markdown fences) with:
            {{"answer":"<short value>","confidence":<float 0-1>,"complexity":<float 0-1>}}
            - answer: word, name, number, date, or short phrase.
            - confidence and complexity: numeric, in [0, 1].

            Example: Action: finish[{{"answer":"42","confidence":0.9,"complexity":0.35}}]

            Do not use a separate "Final Answer:" line. Do not emit bare JSON without Action: finish[...].

            Available tools:
            {tools_block}
            Each tool uses tool_name[input]. Prefer internal reasoning when it is clearly cheaper than a tool call.
            EXTRACTION (files / long Observations)
            1. Scan the entire Observation top to bottom.
            2. Extract everything matching the question; preserve order of appearance.
            3. Copy values exactly as found — no invented formatting.

            RULES
            1. Every assistant message: Thought: then Action: (same message).
            2. Exactly one Action per turn — no chained tools.
            3. Never invent tool outputs.
            4. Avoid unnecessary tools; do not repeat the exact same Action twice.
            5. finish only when the answer is supported by Observations (or direct inference from them).
            6. If the question includes \"Attached file name for reference:\" → read_file with that exact basename only (often a UUID like abc-123.pdf), e.g. read_file[abc-123.pdf]. Never invent filenames like report.pdf or standards.pdf.

            SELF-CHECK (before Action: finish[...])
            - Answer traces to Observations; JSON valid; confidence and complexity in [0,1]; finish payload is JSON only inside the brackets."""
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
            "Do not output markdown or extra commentary."
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
            "Schema: "
            '{{"answer":"<short value>","confidence":<number 0-1>,"complexity":<number 0-1>}}. '
            "The answer must be only what the question requests: a word, name, number, date, "
            "or short phrase — no full sentences inside the answer field. "
            "Do not use markdown code fences around the JSON."
        ),
    },




    "mmlu_pro": {
        "raw":        "...",
        "cot":        "...",
        "react":      "...",
        "multiagent": (
            "You are a multi-agent coordinator using Planner -> Workers -> Judge. "
            "Planner, worker, and judge outputs must be strict JSON. "
            'Final output must be exactly {{"answer":"<value>"}} with no extra text.'
        ),
        "self_consistency": (
            "Self-consistency decoding (Wang et al., 2023): step-by-step chain, then one JSON line. "
            "You solve difficult multiple-choice questions. Use explicit step-by-step reasoning, "
            "then output exactly one JSON object (no trailing commentary). Schema: "
            '{{"answer":"<choice letter or exact option text>","confidence":<0-1>,"complexity":<0-1>}}. '
            "If the question lists options A/B/C/…, the answer field should be the single best "
            "letter unless the question demands the full option string. No markdown fences."
        ),
    },
    "math_hard": {
        "raw":        "...",
        "cot":        "...",
        "react":      "...",
        "multiagent": (
            "You are a multi-agent coordinator using Planner -> Workers -> Judge. "
            "Planner, worker, and judge outputs must be strict JSON. "
            'Final output must be exactly {{"answer":"<value>"}} with no extra text.'
        ),
        "self_consistency": (
            "Self-consistency decoding (Wang et al., 2023): step-by-step chain, then one JSON line. "
            "You solve hard mathematics problems. Show clear intermediate reasoning, then "
            "give one final JSON line. Schema: "
            '{{"answer":"<simplified result>","confidence":<0-1>,"complexity":<0-1>}}. '
            "Put the final numeric result or simplified expression in answer as a string "
            '(e.g. "42", "3/4", "x=2"). No markdown fences.'
        ),
    },
    "swe_bench_verified": {
        "raw":        "...",
        "cot":        "...",
        "react":      "...",
        "multiagent": (
            "You are a multi-agent coordinator using Planner -> Workers -> Judge. "
            "Planner, worker, and judge outputs must be strict JSON. "
            'Final output must be exactly {{"answer":"<value>"}} with no extra text.'
        ),
        "self_consistency": (
            "Self-consistency decoding (Wang et al., 2023): step-by-step chain, then one JSON line. "
            "You analyze software engineering tasks (bugs, tests, patches). Reason step by "
            "step, then output one JSON object. Schema: "
            '{{"answer":"<short technical phrase>","confidence":<0-1>,"complexity":<0-1>}}. '
            "The answer should capture the root cause, the key fix, or the failing behavior in "
            "a short phrase suitable for aggregation — not a patch diff. No markdown fences."
        ),
    },
}



USER_PROMPT: dict[str, dict[str, str]] = {
    "gaia": {
        "raw": (
            "Question: {query}\n"
        ),
        "cot": (
            "Question: {query}\n\n"
            "Think step by step. Show your concise reasoning before answering.\n"
            "The final answer must be one JSON object: {{\"answer\": \"<value>\"}}."
        ),
        "react": (
            "Question: {query}\n" 
            "The answer <value> must be a word, number, name, date, or short phrase."
            "check the attachement file only if it is provided and is relevant to the question."
        ),

        "multiagent": (
            "Question: {query}\n\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
        "self_consistency": (
            "Question: {query}\n\n"
            "Let's think step by step.\n\n"
            '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
        ),
    },
    "mmlu_pro": {
        "raw": "...",
        "cot": "...",
        "react": "...",
        "multiagent": (
            "Question: {query}\n\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
        "self_consistency": (
            "Question: {query}\n\n"
            "Let's think step by step.\n\n"
            "Use numbered reasoning steps. Eliminate wrong options when applicable; cite the "
            "definition or fact that decides between the remaining choices.\n\n"
            "Last line only — one JSON object:\n"
            '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
        ),
    },
    "math_hard": {
        "raw": "...",
        "cot": "...",
        "react": "...",
        "multiagent": (
            "Question: {query}\n\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
        "self_consistency": (
            "Question: {query}\n\n"
            "Let's think step by step.\n\n"
            "Use numbered steps. Show key equations or substitutions. Before the JSON, "
            "sanity-check units, signs, and edge cases when relevant.\n\n"
            "Last line only — one JSON object:\n"
            '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
        ),
    },
    "swe_bench_verified": {
        "raw": "...",
        "cot": "...",
        "react": "...",
        "multiagent": (
            "Question: {query}\n\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
        "self_consistency": (
            "Question: {query}\n\n"
            "Let's think step by step.\n\n"
            "Use numbered steps: symptoms → likely location → hypothesis → what would fix or "
            "verify. Stay grounded in the statement; do not invent file paths or stack traces "
            "not implied by the question.\n\n"
            "Last line only — one JSON object:\n"
            '{{"answer":"<value>","confidence":<0-1>,"complexity":<0-1>}}'
        ),
    },
}

