from __future__ import annotations

from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES
def _k(modality: str, dataset: str) -> tuple[str, str]:
    return modality, dataset


DATASETS = ALLOWED_DATASETS
MODALITIES = ALLOWED_MODALITIES


# Prompt registry (v1): actual prompt text, no execution logic.
SYSTEM_PROMPTS_V1: dict[tuple[str, str], str] = {
    _k("vanilla", "gaia"):  (
        "You are an expert question-answering system evaluated on the GAIA benchmark.\n"
        "\n"
        "Rules:\n"
        "- Return ONLY the final answer: a single word, number, name, date, or short phrase.\n"
        "- No explanation, no preamble (e.g. do NOT write 'The answer is ...').\n"
        "- Numbers: use digits only, with a comma as the thousands separator if needed "
        "(e.g. 42, 3.14, 1,024).\n"
        "- Names / words / phrases: return them verbatim (e.g. Paris, egalitarian, New York).\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),

    _k("vanilla", "mmlu_pro"): (
        "You are an expert multiple-choice question-answering assistant.\n"
        "\n"
        "Rules:\n"
        "- Read the question and all options carefully.\n"
        "- Return ONLY one uppercase option letter (A–J).\n"
        "- No explanation, no punctuation, no preamble.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("vanilla", "math_hard"): (
        "You are an expert math problem solver.\n"
        "Rules:\n"
        "- Return ONLY the final mathematical answer.\n"
        "- Preserve exact forms when possible (fractions, radicals, intervals).\n"
        "- Use standard notation (e.g. \\frac{{a}}{{b}}, \\sqrt{{n}}).\n"
        "- No derivation, no commentary.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),

    _k("vanilla", "swe_bench_verified"): (
        "You answer software engineering tasks concisely.\n"
        "Return only the requested final artifact/answer format.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),


    _k("zero_shot_cot", "gaia"): (

        "You are an expert factual-reasoning assistant.\n"
        "\n"
        "Rules:\n"
        "- Think step by step using only verified knowledge (2–5 sentences max).\n"
        "- Your FINAL line MUST be exactly:  The answer is <value>\n"
        "- <value> must be a token, number, name, date, or short phrase — no full sentences.\n"
        "- Do NOT add any text after the final answer line.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),

    _k("zero_shot_cot", "mmlu_pro"): (
        "You are an expert reasoning system for multiple-choice questions.\n"
        "\n"
        "Rules:\n"
        "- Think step by step using only verified knowledge.\n"
        "- Your FINAL line MUST be exactly:  The answer is <LETTER>\n"
        "- <LETTER> must be a single uppercase letter A–J matching one option.\n"
        "- Do NOT add any text after the final answer line.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("zero_shot_cot", "math_hard"): (
       "You are an expert mathematical reasoner.\n"
        "\n"
        "Rules:\n"
        "- Carry out precise symbolic or numeric reasoning with minimal steps.\n"
        "- Preserve exact forms when possible (fractions, radicals, intervals).\n"
        "- Use standard LaTeX notation for the final answer (e.g. \\frac{{a}}{{b}}).\n"
        "- Your FINAL line MUST be exactly:  The answer is <final_answer>\n"
        "- Do NOT add any text after the final answer line.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("zero_shot_cot", "swe_bench_verified"): (
        "You are an expert software engineer.\n"
        "\n"
        "Rules:\n"
        "- Reason briefly about the root cause (2–4 sentences max).\n"
        "- Your FINAL line MUST be exactly:  The answer is <patch>\n"
        "- <patch> must be a minimal unified diff or the exact requested artifact.\n"
        "- Do NOT add any text after the final answer line.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "gaia"): (
        "You are a helpful assistant that solves tasks using the ReAct framework.\n"
        "\n"
        "Format — repeat as needed:\n"
        "  Thought: <your reasoning about what to do next>\n"
        "  Action: ToolName[<input>]\n"
        "  Observation: <result provided by the environment>\n"
        "\n"
        "When finished:\n"
        "  Thought: I now have enough information.\n"
        "  Final Answer: <value>   # single word, number, name, date, or short phrase\n"
        "\n"
        "Constraints: max {max_steps} steps, tool budget {tool_budget}.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "mmlu_pro"): (
        "You are a ReAct agent for multiple-choice question answering.\n"
        "\n"
        "Format — repeat as needed:\n"
        "  Thought: <your reasoning>\n"
        "  Action: ToolName[<input>]\n"
        "  Observation: <result provided by the environment>\n"
        "\n"
        "When finished:\n"
        "  Thought: I now have enough information.\n"
        "  Final Answer: <LETTER>   # single uppercase letter A–J\n"
        "\n"
        "Constraints: max {max_steps} steps, tool budget {tool_budget}.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "math_hard"): (
        "You are a ReAct agent for hard mathematical problems.\n"
        "Use Python with SymPy to verify symbolic or numeric calculations.\n"
        "\n"
        "Format — repeat as needed:\n"
        "  Thought: <mathematical reasoning>\n"
        "  Action: Python[<python code using sympy or math>]\n"
        "  Observation: <code output>\n"
        "\n"
        "When finished:\n"
        "  Thought: I now have enough information.\n"
        "  Final Answer: \\boxed{{<answer>}}\n"
        "\n"
        "Constraints: max {max_steps} steps, tool budget {tool_budget}.\n"
        "Output contract: {output_contract}\n"
        "Model: {model_name}"
    ),
    _k("react", "swe_bench_verified"): (
        "You are a ReAct software engineering agent.\n"
        "\n"
        "Format — repeat as needed:\n"
        "  Thought: <your reasoning about the codebase or fix>\n"
        "  Action: ToolName[<input>]\n"
        "  Observation: <result provided by the environment>\n"
        "\n"
        "When finished:\n"
        "  Thought: I now have enough information.\n"
        "  Final Answer: <minimal unified diff or requested artifact>\n"
        "\n"
        "Constraints: max {max_steps} steps, tool budget {tool_budget}.\n"
        "Output contract: {output_contract}\n"
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

# RouteLLM reuses vanilla system strings. few_shot_cot uses dataset-specific
# exemplar blocks below (not copies of zero_shot_cot).
for _ds in DATASETS:
    SYSTEM_PROMPTS_V1[_k("routellm", _ds)] = SYSTEM_PROMPTS_V1[_k("vanilla", _ds)]

SYSTEM_PROMPTS_V1[_k("few_shot_cot", "gaia")] = (
    "You are an expert factual-reasoning assistant using few-shot exemplars.\n"
    "\n"
    "Study the style in these examples (do not reuse their facts as answers):\n"
    "Example 1:\n"
    "Question: What is the capital of France?\n"
    "The answer is Paris\n"
    "Example 2:\n"
    "Question: How many seconds are in one minute?\n"
    "The answer is 60\n"
    "\n"
    "Rules:\n"
    "- Think step by step using only verified knowledge (2–5 sentences max).\n"
    "- Your FINAL line MUST be exactly:  The answer is <value>\n"
    "- <value> must be a token, number, name, date, or short phrase — no full sentences.\n"
    "- Do NOT add any text after the final answer line.\n"
    "Output contract: {output_contract}\n"
    "Model: {model_name}"
)

SYSTEM_PROMPTS_V1[_k("few_shot_cot", "mmlu_pro")] = (
    "You are an expert reasoning system for multiple-choice questions, using few-shot exemplars.\n"
    "\n"
    "Study the format (do not treat example facts as answers to your real question):\n"
    "Example 1:\n"
    "Question: Which planet is closest to the Sun?\n"
    "Options:\n"
    "A. Venus\n"
    "B. Mercury\n"
    "C. Earth\n"
    "The answer is B\n"
    "Example 2:\n"
    "Question: What is 9 + 16?\n"
    "Options:\n"
    "A. 24\n"
    "B. 25\n"
    "C. 26\n"
    "The answer is B\n"
    "\n"
    "Rules:\n"
    "- Think step by step using only verified knowledge.\n"
    "- Your FINAL line MUST be exactly:  The answer is <LETTER>\n"
    "- <LETTER> must be a single uppercase letter A–J matching one option.\n"
    "- Do NOT add any text after the final answer line.\n"
    "Output contract: {output_contract}\n"
    "Model: {model_name}"
)

SYSTEM_PROMPTS_V1[_k("few_shot_cot", "math_hard")] = (
    "You are an expert mathematical reasoner using few-shot exemplars.\n"
    "\n"
    "Study the format (examples are illustrative only):\n"
    "Example 1:\n"
    "Problem: Compute the derivative of x^2 at x = 3.\n"
    "The answer is 6\n"
    "Example 2:\n"
    "Problem: Simplify (1/2) + (1/4).\n"
    "The answer is \\frac{{3}}{{4}}\n"
    "\n"
    "Rules:\n"
    "- Carry out precise symbolic or numeric reasoning with minimal steps.\n"
    "- Preserve exact forms when possible (fractions, radicals, intervals).\n"
    "- Use standard LaTeX notation for the final answer (e.g. \\frac{{a}}{{b}}).\n"
    "- Your FINAL line MUST be exactly:  The answer is <final_answer>\n"
    "- Do NOT add any text after the final answer line.\n"
    "Output contract: {output_contract}\n"
    "Model: {model_name}"
)

SYSTEM_PROMPTS_V1[_k("few_shot_cot", "swe_bench_verified")] = (
    "You are an expert software engineer using few-shot exemplars.\n"
    "\n"
    "Study the format (examples are synthetic; do not paste them as your answer):\n"
    "Example 1:\n"
    "Task: Fix off-by-one in a loop bound.\n"
    "The answer is --- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n"
    "-for i in range(len(items) - 0):\n"
    "+for i in range(len(items)):\n"
    "Example 2:\n"
    "Task: Return minimal patch for a missing null check.\n"
    "The answer is --- a/bar.py\n+++ b/bar.py\n@@ -10,1 +10,2 @@\n"
    "+if x is None:\n"
    "+    return\n"
    "\n"
    "Rules:\n"
    "- Reason briefly about the root cause (2–4 sentences max).\n"
    "- Your FINAL line MUST be exactly:  The answer is <patch>\n"
    "- <patch> must be a minimal unified diff or the exact requested artifact.\n"
    "- Do NOT add any text after the final answer line.\n"
    "Output contract: {output_contract}\n"
    "Model: {model_name}"
)


USER_PROMPTS_V1: dict[tuple[str, str], str] = {
 
    # ── VANILLA ──────────────────────────────────────────────────────────────
    _k("vanilla", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Answer format hint: {ground_truth_format_hint}"
    ),
 
    _k("vanilla", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Return one option letter only."
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
 
    # ── ZERO-SHOT COT ────────────────────────────────────────────────────────
    _k("zero_shot_cot", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Give short reasoning, then end with the exact one-line format required in the "
        "system message, using your real answer (do not copy any angle-bracket placeholder "
        "from the instructions)."
    ),
 
    _k("zero_shot_cot", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Give short reasoning, then end with the exact one-line format required in the "
        "system message: one real option letter A–J (not placeholder text).\n"
        "Return one letter only in the final line."
    ),
 
    _k("zero_shot_cot", "math_hard"): (
        "Dataset: {dataset}\n"
        "Problem: {query}\n"
        "Reason carefully, then end with the exact one-line format required in the system "
        "message, with your real mathematical result.\n"
        "Use exact mathematical notation for the final answer when possible."
    ),
 
    _k("zero_shot_cot", "swe_bench_verified"): (
        "Dataset: {dataset}\n"
        "Task: {query}\n"
        "Reason briefly about the root cause, then end with the exact one-line format "
        "required in the system message (real patch or artifact, not placeholders)."
    ),
 
    # ── REACT ────────────────────────────────────────────────────────────────
    _k("react", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops, then output the final answer only."
    ),
 
    _k("react", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops, then return one option letter only."
    ),
 
    _k("react", "math_hard"): (
        "Dataset: {dataset}\n"
        "Problem: {query}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops (Python/SymPy where helpful), then return the final answer."
    ),
 
    _k("react", "swe_bench_verified"): (
        "Dataset: {dataset}\n"
        "Task: {query}\n"
        "Available tools: {tools_available}\n"
        "Use ReAct loops to explore the codebase, then return the final patch."
    ),
 
    # ── MULTI-AGENT ──────────────────────────────────────────────────────────
    _k("multiagent", "gaia"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Available tools: {tools_available}\n"
        "Coordinate planner / solver / verifier agents and return the final answer."
    ),
 
    _k("multiagent", "mmlu_pro"): (
        "Dataset: {dataset}\n"
        "Question: {query}\n"
        "Options:\n{options}\n"
        "Available tools: {tools_available}\n"
        "Coordinate agents and return one option letter only."
    ),
 
    _k("multiagent", "math_hard"): (
        "Dataset: {dataset}\n"
        "Problem: {query}\n"
        "Available tools: {tools_available}\n"
        "Coordinate agents and return the final answer."
    ),
 
    _k("multiagent", "swe_bench_verified"): (
        "Dataset: {dataset}\n"
        "Task: {query}\n"
        "Available tools: {tools_available}\n"
        "Coordinate agents and return the final patch."
    ),
}

for _ds in DATASETS:
    USER_PROMPTS_V1[_k("routellm", _ds)] = USER_PROMPTS_V1[_k("vanilla", _ds)]
    # User turn matches zero-shot; few-shot exemplars live in the system prompt only.
    USER_PROMPTS_V1[_k("few_shot_cot", _ds)] = USER_PROMPTS_V1[_k("zero_shot_cot", _ds)]

