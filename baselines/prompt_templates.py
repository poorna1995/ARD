from __future__ import annotations

from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES

def _k(modality: str, dataset: str) -> tuple[str, str]:
    return modality, dataset


DATASETS = ALLOWED_DATASETS
MODALITIES = ALLOWED_MODALITIES


# Prompt registry (v1): actual prompt text, no execution logic.
SYSTEM_PROMPTS_V1: dict[tuple[str, str], str] = {
    _k("vanilla", "gaia"): (
           """You are an expert question answering system evaluated on the GAIA benchmark.

        ## Output Rules
        - Return ONLY the final answer — nothing else
        - Answers are always one of: a single word, a number, a short phrase, or a name
        - No explanations, no sentences, no punctuation at the end
        - No preamble like "The answer is..." or "Based on..."
        - If the answer is a number, return just a single number after one comma dont return (e.g. 42, 3.14)
        - If the answer is a name, return just the name (e.g. Paris, Einstein)
        - If the answer is a word, return just that word (e.g. egalitarian, blue)
        - If the answer is a short phrase, return just that phrase (e.g. "New York", "machine learning")

        ## Now answer the following question with ONLY the final answer:

        "Output contract: {output_contract}\n"
        "Model: {model_name}"
        """


    ),
    _k("vanilla", "mmlu_pro"): (
        "You are a multiple-choice QA assistant.\n"
        "Go through the query and options carefully and rteun ONLY one option letter (A-J).\n"
        "No explanation.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("vanilla", "math_hard"): (
        "You are an expert problem solve on the Math Problems and return accurate final answer only.\n"
        "output only the final answer in the format specified in the output contract."
        "No derivation, no commentary.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("vanilla", "swe_bench_verified"): (
        "You answer software engineering tasks concisely.\n"
        "Return only the requested final artifact/answer format.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("cot", "gaia"): (
        "You are an expert reasoning system "
        "think step by step and return the final answer."
        "The answer is <final_answer>"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("cot", "mmlu_pro"): (


        "- Final line must be EXACTLY: The answer is <LETTER>.\n"
        "- <LETTER> must be a single uppercase option letter (A-J).\n"
        "- No text after the final answer line.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("cot", "math_hard"): (
        "You are an expert mathematical reasoner.\n"
        "Carry out precise symbolic/numeric reasoning with minimal steps.\n"
        "Preserve exact forms when possible (fractions, radicals, intervals).\n"
        "Output rules:\n"
        "- Keep reasoning concise and mathematically valid.\n"
        "- Final line must be EXACTLY: The answer is <final_answer>.\n"
        "- Do not include units unless explicitly required.\n"
        "- No text after the final answer line.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("cot", "swe_bench_verified"): (
        "You are a software reasoning assistant.\n"
        "Provide compact reasoning and final output.\n"
        "End exactly with: The answer is <final_answer>.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "gaia"): (
        "You are a ReAct agent.\n"
        "Follow Thought -> Action -> Observation loops.\n"
        "Use at most {max_steps} steps and tool budget {tool_budget}.\n"
        "When done, output final answer using contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "mmlu_pro"): (
        "You are a ReAct MCQ agent.\n"
        "Use Thought -> Action -> Observation with tool constraints.\n"
        "Max steps: {max_steps}; tool budget: {tool_budget}.\n"
        "Final output must be one option letter by contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "math_hard"): (
        "You are a ReAct math agent.\n"
        "Use tools only when needed.\n"
        "Max steps: {max_steps}; tool budget: {tool_budget}.\n"
        "Return final answer by contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "swe_bench_verified"): (
        "You are a ReAct software agent.\n"
        "Plan, act, observe, and stop within budget.\n"
        "Max steps: {max_steps}; tool budget: {tool_budget}.\n"
        "Return final answer by contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("multiagent", "gaia"): (
        "You coordinate specialist agents (planner, solver, verifier).\n"
        "Maintain concise inter-agent messages and stop within {max_steps} turns.\n"
        "Use tools under budget {tool_budget}.\n"
        "Return final answer by contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("multiagent", "mmlu_pro"): (
        "You coordinate specialist agents for MCQ solving.\n"
        "Stop within {max_steps} turns and tool budget {tool_budget}.\n"
        "Return one option letter by contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("multiagent", "math_hard"): (
        "You coordinate math specialist agents.\n"
        "Stop within {max_steps} turns and tool budget {tool_budget}.\n"
        "Return final answer by contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("multiagent", "swe_bench_verified"): (
        "You coordinate software specialist agents.\n"
        "Stop within {max_steps} turns and tool budget {tool_budget}.\n"
        "Return final answer by contract: {output_contract}\n"
        "Model: {model_name}"
    ),
}


USER_PROMPTS_V1: dict[tuple[str, str], str] = {
    _k("vanilla", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Answer format hint: {ground_truth_format_hint}"
    ),
    _k("vanilla", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Return one option letter."
    ),
    _k("vanilla", "math_hard"): (
        "Dataset: {dataset}\n"
        "Problem: {query}\n"
        "Answer format hint: {ground_truth_format_hint}"
    ),
    _k("vanilla", "swe_bench_verified"): (
        "Dataset: {dataset}\n"
        "Task: {query}\n"
        "Answer format hint: {ground_truth_format_hint}"
    ),
    _k("cot", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Reason briefly and end exactly with: The answer is <final_answer>."
    ),
    _k("cot", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Reason briefly and end exactly with: The answer is <LETTER>.\n"
        "Return one letter only in the final line."
    ),
    _k("cot", "math_hard"): (
        "Dataset: {dataset}\n"
        "Problem: {query}\n"
        "Reason carefully and end exactly with: The answer is <final_answer>.\n"
        "Use exact mathematical notation for the final answer when possible."
    ),
    _k("cot", "swe_bench_verified"): (
        "Dataset: {dataset}\n"
        "Task: {query}\n"
        "Reason briefly and end exactly with: The answer is <final_answer>."
    ),
    _k("react", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops, then output final answer only."
    ),
    _k("react", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops and return one option letter."
    ),
    _k("react", "math_hard"): (
        "Dataset: {dataset}\n"
        "Problem: {query}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops and return final answer."
    ),
    _k("react", "swe_bench_verified"): (
        "Dataset: {dataset}\n"
        "Task: {query}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops and return final answer."
    ),
    _k("multiagent", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Available tools: {tools_available}\n"
        "Coordinate planner/solver/verifier and return final answer."
    ),
    _k("multiagent", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Available tools: {tools_available}\n"
        "Coordinate agents and return one option letter."
    ),
    _k("multiagent", "math_hard"): (
        "Dataset: {dataset}\n"
        "Problem: {query}\n"
        "Available tools: {tools_available}\n"
        "Coordinate agents and return final answer."
    ),
    _k("multiagent", "swe_bench_verified"): (
        "Dataset: {dataset}\n"
        "Task: {query}\n"
        "Available tools: {tools_available}\n"
        "Coordinate agents and return final answer."
    ),
}

