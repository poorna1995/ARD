from __future__ import annotations


DATASETS = ["gaia", "mmlu_pro", "math_hard", "swe_bench_verified"]
AGENTS = ['raw', 'cot', 'react', 'multiagent']

SYSTEM_PROMPT: dict[str, dict[str, str]] = {
    "gaia": {
        "raw": ( "Given a query, an expert in handling real world everyday situation agent, answer the query."
            "Return exactly one JSON object: {\"answer\": \"<value>\"}."
            "<value> must be a word, name, number, or short phrase. "
        ),

        "cot": (
            "You are a multi-hop reasoning expert and an expert QA agent. Given a question, and the context, think step-by-step. "
            "answer <value> must be in = (word, number, name, date, or short phrase). "
            "Return exactly one JSON object: {{\"answer\": \"<value>\"}}."
        ),

        "react": (
            """You are a ReAct agent. Solve tasks by alternating Thought → Action → Observation. Never skip steps.
            ## FORMAT (strict)
            Thought: <Why this step. What's missing. Which tool fills the gap.>
            Action: tool_name[input_string]
            Observation: <runtime fills this — never write it yourself, concise summary of the information>
            ... repeat until done ...
            Thought: I have enough to answer.
            Final Answer: <Grounded response. Cite observations. No invented facts.> 
            Finish[] must return answer: {{"answer": "<value>"}}.
            {tools_block} - Tool descriptions added read it carefully.

            ## Rules
            - Always start with "Thought:".
            - ONE Action per turn.
            - One tool call per step
            - Ground every claim in an Observation
            - only fetch or retrive relevant information from the context as a summary.
            - Never invent tool outputs 
            - Never call a tool you don't need
            - Never repeat the same call twice
            - - If a file is mentioned in the query, extract ONLY the filename (string ending with .xlsx, .csv, etc.)
                - Example:
                Query: "Attached file name: data.xlsx"
                Action: read_file[data.xlsx]
            """
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
            "Question: {query}\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
    },
    "mmlu_pro": {
        "raw": "...",
        "cot": "...",
        "react": "...",
        "multiagent": (
            "Question: {query}\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
    },
    "math_hard": {
        "raw": "...",
        "cot": "...",
        "react": "...",
        "multiagent": (
            "Question: {query}\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
    },
    "swe_bench_verified": {
        "raw": "...",
        "cot": "...",
        "react": "...",
        "multiagent": (
            "Question: {query}\n"
            "Use Planner -> Workers -> Judge internally. "
            'Return exactly one JSON object: {{"answer":"<value>"}}.'
        ),
    },
}




# """You are a ReAct agent. Solve tasks by strictly cycling through Thought → Action → Observation.

                #     Thought: <why you need this action and what you expect>
                #     Action: <tool_name>[<exact input>]
                #     Observation: <environment result — never written by you>

                #     Repeat until the answer is certain, then finish:

                #     Thought: <why you can answer now>
                #     Action: Finish[<final answer>]

                #     {tools_block}

                #     Rules:
                #     - Every response MUST start with "Thought:"
                #     - Exactly ONE Action per turn
                #     - NEVER write Observation yourself
                #     - once context is found, summarize (highlight the most important information) the context and use it to the next step u
                #     - dont accumulate irrelevant information
                #     - Always use the latest Observation before the next step
                #     - Do not call tools unnecessarily
                #     - Only call Finish[] when you have VERIFIED the answer from a tool result
                #     - Output strictly in required format

                #     Finish[] must return: {{"answer": "<value>"}}.
                #     """