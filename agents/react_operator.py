from __future__ import annotations
 
import re
import json
import uuid
import math
import serpapi
import time
import logging
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Optional
from dotenv import load_dotenv
from enum import Enum   # ✅ add this
import pypdf
import pdfplumber
from unstructured.partition.auto import partition
import pandas as pd
import pytesseract
from PIL import Image
load_dotenv()
import os
from groq import Groq

import openai

# openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # add this
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
 

client_serpapi = serpapi.Client(api_key=os.getenv("SERP_API_KEY"))
# ── logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level  = logging.INFO,
    format = "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt= "%H:%M:%S",
)
logger = logging.getLogger("react_agent")
 
 
class StepType(Enum):
    THOUGHT     = "Thought"
    ACTION      = "Action"
    OBSERVATION = "Observation"
    REFLECTION  = "Reflection"
    FINISH      = "Finish"


class SubApproach(Enum):
    TOOLING    = "tooling"       # External tool-heavy queries
    PLANNING   = "planning"      # Multi-step sub-goal decomposition
    REFLECTION = "reflection"    # Self-critique + re-ranking


class EscalationReason(Enum):
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TOOL_FAILURE       = "tool_failure"
    LOW_CONFIDENCE     = "low_confidence"
    PARSE_ERROR        = "parse_error"

# Change this constant at the top of your file
UPLOAD_DIR: str = "../datasets/gaia_files/2023/validation"   # ✅ point to actual folder

@dataclass
class ReActStep:
    step_index: int
    step_type: StepType
    content: str
    tool_name: Optional[str] = None          # populated if step_type == ACTION
    tool_input: Optional[Any] = None
    tool_output: Optional[Any] = None
    token_count: int = 0
    latency_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
 
# ── Loop control ──────────────────────────────────────────────
MAX_STEPS: int = 15           # Hard limit on Thought/Action/Observation iterations
MAX_TOKENS_PER_STEP: int = 512
MAX_CONTEXT_TOKENS: int = 8192  # Scratchpad window before truncation

# ── Stopping signals ──────────────────────────────────────────
FINISH_TOKEN: str = "FINISH"         # Agent emits this when answer is ready
THINK_PREFIX: str = "Thought:"
ACTION_PREFIX: str = "Action:"
OBS_PREFIX: str = "Observation:"
ACTION_INPUT_PREFIX: str = "Action Input:"

# ── Confidence thresholds ─────────────────────────────────────
MIN_CONFIDENCE_FOR_REACT: float = 0.45   # Below this → CoT only
MAX_CONFIDENCE_FOR_REACT: float = 0.85   # Above this → Base LLM only
REFLECTION_TRIGGER_STEP: int = 5         # Trigger self-reflection after N steps without FINISH

# ── Provider ─────────────────────────────────────────────────
MODEL_NAME: str = "llama-3.3-70b-versatile"             # or "llama-3.1-70b"
TEMPERATURE: float = 0.0                # Deterministic for reproducibility
TOOL_CALL_TIMEOUT_SEC: int = 30

# ── Dataset-specific ─────────────────────────────────────────
DATASET_CONFIGS: dict = {
    "GAIA":        {"max_steps": 15, "tools": ["search", "file_reader", "calculator", "python_exec"]},
    "SWE_verified": {"max_steps": 20, "tools": ["bash_exec", "file_reader", "python_exec", "grep"]},
    "mmlupro":     {"max_steps": 10, "tools": ["calculator", "search", "python_exec"]},
    "MathHard_L5": {"max_steps": 12, "tools": ["sympy_solver", "calculator", "python_exec"]},
}

# ── Tracing ───────────────────────────────────────────────────
TRACE_DIR: str = "./traces"
LOG_LEVEL: str = "INFO"         # DEBUG | INFO | WARNING
WANDB_PROJECT: str = "react-adaptive-routing"

@dataclass
class Scratchpad:
    """
    The persistent context window for one ReAct episode.
    Implements the T/A/O triplet accumulation from Yao et al. Section 3.
    """
    query: str
    system_prompt: str
    steps: list[ReActStep] = field(default_factory=list)
    total_tokens: int = 0

    def append(self, step: ReActStep) -> None:
        self.steps.append(step)
        self.total_tokens += step.token_count

    def to_prompt_string(self) -> str:
        """Serialise the full scratchpad into a single LLM context string."""
        lines = [f"Question: {self.query}\n"]
        for s in self.steps:
            if s.step_type == StepType.THOUGHT:
                lines.append(f"Thought {s.step_index}: {s.content}")
            elif s.step_type == StepType.ACTION:
                lines.append(f"Action {s.step_index}: {s.tool_name}")
                lines.append(f"Action Input {s.step_index}: {s.tool_input}")
            elif s.step_type == StepType.OBSERVATION:
                lines.append(f"Observation {s.step_index}: {s.content}")
            elif s.step_type == StepType.REFLECTION:
                lines.append(f"Reflection: {s.content}")
            elif s.step_type == StepType.FINISH:
                lines.append(f"Final Answer: {s.content}")
        return "\n".join(lines)

    def token_budget_remaining(self) -> int:
        return MAX_CONTEXT_TOKENS - self.total_tokens

    def needs_truncation(self) -> bool:
        return self.total_tokens > MAX_CONTEXT_TOKENS


@dataclass
class ToolSchema:
    name: str
    description: str
    input_schema: dict          # JSON Schema for input validation
    returns: str                # Human-readable description of return type
    is_stateful: bool = False   # e.g. bash session keeps state across calls
    timeout_sec: int = TOOL_CALL_TIMEOUT_SEC


@dataclass
class ToolResult:
    tool_name: str
    success: bool
    output: Any
    error_message: Optional[str] = None
    latency_ms: float = 0.0
    raw_output: Optional[str] = None   # unparsed string from tool

@dataclass
class ReActAgentState:
    episode_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    query: str = ""
    dataset: str = ""
    modality: str = ""
    confidence_score: float = 0.0     # From complexity estimator upstream
    sub_approach: SubApproach = SubApproach.TOOLING
    scratchpad: Optional[Scratchpad] = None
    current_step: int = 0
    finished: bool = False
    escalated: bool = False
    escalation_reason: Optional[EscalationReason] = None
    final_answer: Optional[str] = None
    ground_truth: Optional[str] = None    # For evaluation
    total_latency_ms: float = 0.0
    total_cost_usd: float = 0.0


@dataclass
class EpisodeTrace:
    episode_id: str
    query: str
    dataset: str
    ground_truth: Optional[str]
    final_answer: Optional[str]
    sub_approach: str
    steps: list[ReActStep]
    n_steps: int
    finished: bool
    escalated: bool
    escalation_reason: Optional[str]
    correct: Optional[bool]           # Set by evaluator post-hoc
    total_tokens: int
    total_latency_ms: float
    total_cost_usd: float
    model: str
    timestamp: str
# ══════════════════════════════════════════════════════════════════════════════
#  Tool registry
# ══════════════════════════════════════════════════════════════════════════════
 
# ✅ After — {n} escaped as {{n}}, only real format vars left as single braces
REACT_SYSTEM_PROMPT_BASE = """You are a ReAct agent. You solve problems by interleaving Thought and Action steps.

    Format strictly:
    Thought {{n}}: <your reasoning about what to do next>
    Action {{n}}: <tool name — one of: {tool_names}>
    Action Input {{n}}: <input to the tool>
    Observation {{n}}: <tool result — filled by the system>
    ...
    Final Answer: <your answer>

    Rules:
    - Always begin with a Thought step.
    - Never skip directly to a Final Answer without at least one Action.
    - Keep each Thought to 2–4 sentences.
    - When you have enough information, output exactly: Final Answer: <answer>

    Dataset context: {dataset}
    Modality: {modality}
    Task type: {task_type}

    Available tools:
    {tool_descriptions}

    Few-shot example:
    {few_shot_example}
"""

REFLECTION_PROMPT = """You have completed {n} steps without reaching a Final Answer.
Review your scratchpad below and identify:
1. Any incorrect assumptions made.
2. Any tool results you misinterpreted.
3. An alternative action sequence.

Scratchpad:
{scratchpad}

Output your reflection as: Reflection: <your critique and revised plan>
"""


from abc import ABC, abstractmethod


class BaseTool(ABC):
    schema: ToolSchema

    @abstractmethod
    def run(self, tool_input: Any) -> ToolResult:
        """Execute the tool. Must return a ToolResult."""
        ...

    def validate_input(self, tool_input: Any) -> bool:
        """Optional JSON Schema validation before execution."""
        return True


# class WebSearchTool(BaseTool):
#     """
#     For GAIA and mmlupro factual queries.
#     Uses SerpAPI or Tavily under the hood.
#     """
#     schema = ToolSchema(
#         name="search",
#         description="Search the web for factual information. Use for current events, definitions, or factual lookups.",
#         input_schema={"type": "string", "description": "Search query string"},
#         returns="Top-3 search result snippets as a single string",
#     )
#     def run(self, tool_input: str) -> ToolResult:
#         results = client_serpapi.search({
#             "q": tool_input,
#             "location": "Austin, Texas, United States",
#             "hl": "en",
#             "gl": "us",
#             "google_domain": "google.com"
#         })
#         return ToolResult(
#             tool_name="search",
#             success=True,
#             output=results,
#             latency_ms=results.get("latency", 0.0),
#             raw_output=json.dumps(results)
#         )

class WebSearchTool(BaseTool):
    """
    For GAIA and mmlupro factual queries.
    Uses SerpAPI under the hood.
    """
    schema = ToolSchema(
        name="search",
        description="Search the web for factual information. Use for current events, definitions, or factual lookups.",
        input_schema={"type": "string", "description": "Search query string"},
        returns="Top-3 search result snippets as a single string",
    )

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            results = client_serpapi.search({
                "q": tool_input,
                "location": "Austin, Texas, United States",
                "hl": "en",
                "gl": "us",
                "google_domain": "google.com",
            })

            results_dict = dict(results)                    # ✅ convert once, use everywhere
            latency_ms = (time.time() - start) * 1000

            organic = results_dict.get("organic_results", [])[:3]
            snippets = "\n".join(
                f"{i+1}. {r.get('title', '')}: {r.get('snippet', '')}"
                for i, r in enumerate(organic)
            )

            return ToolResult(
                tool_name="search",
                success=True,
                output=snippets,
                latency_ms=latency_ms,
                raw_output=json.dumps(results_dict),        # ✅ now serializable
            )

        except Exception as e:
            latency_ms = (time.time() - start) * 1000
            return ToolResult(
                tool_name="search",
                success=False,
                output=None,
                error_message=str(e),
                latency_ms=latency_ms,
            )
class PythonExecTool(BaseTool):
    """
    Sandboxed Python execution. For GAIA file processing, mmlupro reasoning.
    Runs in subprocess with timeout; no network access inside sandbox.
    """
    schema = ToolSchema(
        name="python_exec",
        description="Execute Python code. Use for arithmetic, data manipulation, file parsing.",
        input_schema={"type": "string", "description": "Valid Python code string"},
        returns="stdout of execution, truncated to 2000 chars",
        timeout_sec=20,
    )
    def run(self, tool_input: str) -> ToolResult: ...


class BashExecTool(BaseTool):
    """
    For SWE-bench verified. Runs inside the checked-out repo environment.
    Stateful: session persists across steps within one episode.
    """
    schema = ToolSchema(
        name="bash_exec",
        description="Run bash commands inside the target repository. Use for running tests, grep, git diff.",
        input_schema={"type": "string", "description": "Bash command string"},
        returns="stdout + stderr, truncated to 3000 chars",
        is_stateful=True,
        timeout_sec=60,
    )
    def run(self, tool_input: str) -> ToolResult: ...


class SympySolverTool(BaseTool):
    """
    For MathHard Level 5. Passes expression to SymPy for symbolic computation.
    """
    schema = ToolSchema(
        name="sympy_solver",
        description="Solve mathematical expressions symbolically. Input a Python expression using SymPy syntax.",
        input_schema={"type": "string", "description": "SymPy-compatible expression string"},
        returns="Symbolic or numerical solution as string",
    )
    def run(self, tool_input: str) -> ToolResult: ...


class CalculatorTool(BaseTool):
    schema = ToolSchema(
        name="calculator",
        description="Evaluate arithmetic expressions. Use for simple numeric computation.",
        input_schema={"type": "string", "description": "Arithmetic expression, e.g. '(3 + 4) * 2 / 7'"},
        returns="Numeric result as string",
    )
    def run(self, tool_input: str) -> ToolResult: ...


# class FileReaderTool(BaseTool):
#     """
#     For GAIA file-based tasks. Reads uploaded file content by filename.
#     """
#     schema = ToolSchema(
#         name="file_reader",
#         description="Read the contents of an attached file. Input the filename.",
#         input_schema={"type": "string", "description": "Filename string"},
#         returns="File content string, truncated to 4000 chars",
#     )
#     def run(self, tool_input: str) -> ToolResult: ...


class FileReaderTool(BaseTool):
    schema = ToolSchema(
        name="file_reader",
        description="Read the contents of an attached file. Input the filename.",
        input_schema={"type": "string", "description": "Filename string"},
        returns="File content string, truncated to 4000 chars",
    )

    # ✅ Store method names as strings — safe at class load time
    EXTRACTORS: dict[str, list[str]] = {
        ".pdf":  ["_pdfplumber", "_pypdf"],
        ".xlsx": ["_xlsx"],
        ".png":  ["_tesseract"],
        ".pdb":  ["_pdb_records"],
        ".csv":  ["_csv"],
    }

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            file_path = os.path.join(UPLOAD_DIR, os.path.basename(tool_input))

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{tool_input}' not found in {UPLOAD_DIR}")

            ext = os.path.splitext(file_path)[1].lower()
            content = self._extract(file_path, ext)

            return ToolResult(
                tool_name="file_reader",
                success=True,
                output=content[:4000],
                latency_ms=(time.time() - start) * 1000,
                raw_output=content,
            )

        except Exception as e:
            return ToolResult(
                tool_name="file_reader",
                success=False,
                output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    # ── Generic fallback runner ───────────────────────────────────────────────
    def _extract(self, file_path: str, ext: str) -> str:
        method_names = self.EXTRACTORS.get(ext, ["_plain_text"])

        # ✅ Resolve string names → real methods at call time, not class load time
        extractors = [getattr(self, name) for name in method_names]

        for extractor in extractors:
            try:
                content = extractor(file_path)
                if content and content.strip():     # only accept non-empty result
                    return content
            except Exception:
                continue                            # silently try next extractor

        # Last resort for all formats
        return self._unstructured(file_path)

    # ── Extractors ────────────────────────────────────────────────────────────
    @staticmethod
    def _pdfplumber(file_path: str) -> str:
        with pdfplumber.open(file_path) as pdf:
            return "\n".join(page.extract_text() or "" for page in pdf.pages)

    @staticmethod
    def _pypdf(file_path: str) -> str:
        reader = pypdf.PdfReader(file_path)
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    @staticmethod
    def _xlsx(file_path: str) -> str:
        xl = pd.ExcelFile(file_path)
        parts = [
            f"--- Sheet: {name} ---\n{xl.parse(name).to_string(index=False)}"
            for name in xl.sheet_names
        ]
        return "\n\n".join(parts)

    @staticmethod
    def _tesseract(file_path: str) -> str:
        return pytesseract.image_to_string(Image.open(file_path).convert("RGB"))

    @staticmethod
    def _pdb_records(file_path: str) -> str:
        USEFUL_RECORDS = {"HEADER", "TITLE", "REMARK", "SEQRES", "ATOM", "HETATM", "COMPND"}
        with open(file_path, "r", encoding="utf-8") as f:
            return "\n".join(
                line.rstrip() for line in f
                if line[:6].strip() in USEFUL_RECORDS
            )

    @staticmethod
    def _csv(file_path: str) -> str:
        return pd.read_csv(file_path).to_string()

    @staticmethod
    def _plain_text(file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _unstructured(file_path: str) -> str:
        return "\n".join(str(el) for el in partition(filename=file_path))



class GrepTool(BaseTool):
    """
    For SWE-bench. Searches across repo files for a pattern.
    """
    schema = ToolSchema(
        name="grep",
        description="Search for a pattern in repository files. Returns matching lines with file paths.",
        input_schema={"type": "object", "properties": {
            "pattern": {"type": "string"},
            "path": {"type": "string", "default": "."},
        }},
        returns="Matching lines with file:line_number format",
    )
    def run(self, tool_input: dict) -> ToolResult: ...



class ToolRegistry:
    def __init__(self, dataset: str):
        self._tools: dict[str, BaseTool] = {}
        self._load_for_dataset(dataset)

    def _load_for_dataset(self, dataset: str) -> None:
        if dataset not in DATASET_CONFIGS:
            raise ValueError(f"Unknown dataset '{dataset}'. Valid options: {list(DATASET_CONFIGS.keys())}")
        all_tools = {
            "search":       WebSearchTool(),
            "python_exec":  PythonExecTool(),
            "bash_exec":    BashExecTool(),
            "sympy_solver": SympySolverTool(),
            "calculator":   CalculatorTool(),
            "file_reader":  FileReaderTool(),
            "grep":         GrepTool(),
        }
        self._tools = {k: v for k, v in all_tools.items() if k in tool_names}

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def descriptions(self) -> str:
        return "\n".join(
            f"- {t.schema.name}: {t.schema.description}"
            for t in self._tools.values()
        )

    def names(self) -> list[str]:
        return list(self._tools.keys())



def select_sub_approach(
    query: str,
    confidence_score: float,
    dataset: str,
    modality: str,
) -> SubApproach:
    """
    Heuristic + rule-based selection of ReAct sub-approach.
    Called once per episode before the loop starts.

    Rules (in order of priority):
    1. MathHard Level 5 + confidence < 0.6 → Reflection (algebraic errors common)
    2. SWE-bench → Planning (multi-file edits require ordered sub-goals)
    3. GAIA Level 3 (multi-hop) → Planning
    4. confidence < 0.55 → Reflection
    5. Default → Tooling
    """
    if dataset == "MathHard_L5" and confidence_score < 0.6:
        return SubApproach.REFLECTION
    if dataset in ("SWE_verified",):
        return SubApproach.PLANNING
    if dataset == "GAIA" and confidence_score < 0.65:
        return SubApproach.PLANNING
    if confidence_score < 0.55:
        return SubApproach.REFLECTION
    return SubApproach.TOOLING

def build_system_prompt(
    dataset: str,
    modality: str,
    tool_registry: ToolRegistry,
    sub_approach: SubApproach,
    few_shot_examples: dict,   # keyed by dataset name
) -> str:
    task_type_map = {
        SubApproach.TOOLING:    "direct tool use",
        SubApproach.PLANNING:   "multi-step planning with sub-goals",
        SubApproach.REFLECTION: "iterative self-critique and re-ranking",
    }
    return REACT_SYSTEM_PROMPT_BASE.format(
        tool_names=", ".join(tool_registry.names()),
        dataset=dataset,
        modality=modality,
        task_type=task_type_map[sub_approach],
        tool_descriptions=tool_registry.descriptions(),
        few_shot_example=few_shot_examples.get(dataset, ""),
    )



def call_llm(
    system_prompt: str,
    user_content: str,
    model: str = MODEL_NAME,
    temperature: float = TEMPERATURE,
    max_tokens: int = MAX_TOKENS_PER_STEP,

) -> tuple[str, int, float]:
    """
        Returns (response_text, token_count, latency_ms).
        Wraps your provider SDK (OpenAI, Together, vLLM, etc.).
    """
    import time
    start = time.time()
    # -- replace with your SDK call --
    response = groq_client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )
    latency_ms = (time.time() - start) * 1000
    text = response.choices[0].message.content
    tokens = response.usage.total_tokens
    return text, tokens, latency_ms


def parse_llm_output(text: str, step_index: int) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Parses one LLM generation into (thought, action_name, action_input, is_finish).

    Returns:
        thought       : str or None
        action_name   : str or None
        action_input  : str or None
        is_finish     : bool — True if FINISH_TOKEN found
    """
    # Check for final answer
    finish_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    if finish_match:
        return None, None, None, True

    thought_match = re.search(
        r"Thought\s*\d*\s*:\s*(.+?)(?=Action\s*\d*\s*:|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    action_match = re.search(
        r"Action\s*\d*\s*:\s*(.+?)(?=Action\s+Input\s*\d*\s*:|$)",
        text, re.DOTALL | re.IGNORECASE
    )
    input_match = re.search(
        r"Action\s+Input\s*\d*\s*:\s*(.+?)(?=Observation\s*\d*\s*:|$)",
        text, re.DOTALL | re.IGNORECASE
    )

    thought     = thought_match.group(1).strip() if thought_match else None
    action_name = action_match.group(1).strip()  if action_match  else None
    action_input= input_match.group(1).strip()   if input_match   else None

    return thought, action_name, action_input, False


def extract_final_answer(text: str) -> Optional[str]:
    match = re.search(r"Final Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def should_trigger_reflection(state: ReActAgentState) -> bool:
    """
    Returns True if the agent should pause for self-reflection.
    Triggered when REFLECTION_TRIGGER_STEP steps pass with no FINISH.
    """
    return (
        state.current_step >= REFLECTION_TRIGGER_STEP
        and not state.finished
        and state.sub_approach != SubApproach.REFLECTION  # avoid double-trigger
    )


def run_reflection(
    scratchpad: Scratchpad,
    system_prompt: str,
    step_index: int,
) -> ReActStep:
    """
    Inserts a Reflection step into the scratchpad.
    Prompts the LLM to critique its own trace.
    """
    reflection_prompt = REFLECTION_PROMPT.format(
        n=step_index,
        scratchpad=scratchpad.to_prompt_string(),
    )
    text, tokens, latency = call_llm(system_prompt, reflection_prompt)
    return ReActStep(
        step_index=step_index,
        step_type=StepType.REFLECTION,
        content=text,
        token_count=tokens,
        latency_ms=latency,
    )



def run_react_episode(
    query: str,
    dataset: str,
    modality: str,
    confidence_score: float,
    ground_truth: Optional[str] = None,
    few_shot_examples: dict = {},
) -> EpisodeTrace:
    """
    Full ReAct episode. Entry point after the router dispatches here.
    Returns a complete EpisodeTrace for logging and evaluation.
    """
    import time
    episode_start = time.time()

    # ── Initialise state ─────────────────────────────────────
    state = ReActAgentState(
        query=query,
        dataset=dataset,
        modality=modality,
        confidence_score=confidence_score,
        ground_truth=ground_truth,
    )
    state.sub_approach = select_sub_approach(query, confidence_score, dataset, modality)

    tool_registry  = ToolRegistry(dataset)
    system_prompt  = build_system_prompt(dataset, modality, tool_registry, state.sub_approach, few_shot_examples)
    scratchpad     = Scratchpad(query=query, system_prompt=system_prompt)
    state.scratchpad = scratchpad

    tracer = EpisodeTracer(state.episode_id)
    tracer.log_start(state)

    # ── Main loop ─────────────────────────────────────────────
    for step in range(1, MAX_STEPS + 1):
        state.current_step = step

        # Reflection mid-episode if stuck
        if should_trigger_reflection(state):
            reflection_step = run_reflection(scratchpad, system_prompt, step)
            scratchpad.append(reflection_step)
            tracer.log_step(reflection_step)
            state.sub_approach = SubApproach.REFLECTION 

        # Context window guard
        if scratchpad.needs_truncation():
            scratchpad = truncate_scratchpad(scratchpad)

        # LLM generation
        prompt_content = scratchpad.to_prompt_string()
        raw_text, tokens, latency = call_llm(system_prompt, prompt_content)

        # Parse output
        thought, action_name, action_input, is_finish = parse_llm_output(raw_text, step)

        if is_finish:
            answer = extract_final_answer(raw_text)
            finish_step = ReActStep(
                step_index=step,
                step_type=StepType.FINISH,
                content=answer or raw_text,
                token_count=tokens,
                latency_ms=latency,
            )
            scratchpad.append(finish_step)
            tracer.log_step(finish_step)
            state.finished = True
            state.final_answer = answer
            break

        # Record Thought
        if thought:
            t_step = ReActStep(step_index=step, step_type=StepType.THOUGHT,
                               content=thought, token_count=tokens, latency_ms=latency)
            scratchpad.append(t_step)
            tracer.log_step(t_step)

        # Execute Action
        if action_name and action_input:
            a_step = ReActStep(step_index=step, step_type=StepType.ACTION,
                               content=raw_text, tool_name=action_name,
                               tool_input=action_input)
            scratchpad.append(a_step)
            tracer.log_step(a_step)

            tool = tool_registry.get(action_name)
            if tool is None:
                obs_content = f"ERROR: Tool '{action_name}' not found. Available: {tool_registry.names()}"
            else:
                result: ToolResult = tool.run(action_input)
                obs_content = result.output if result.success else f"TOOL ERROR: {result.error_message}"

            o_step = ReActStep(step_index=step, step_type=StepType.OBSERVATION,
                               content=str(obs_content), tool_name=action_name,
                               tool_output=obs_content)
            scratchpad.append(o_step)
            tracer.log_step(o_step)
        else:
            # Parse failure
            state.escalated = True
            state.escalation_reason = EscalationReason.PARSE_ERROR
            tracer.log_escalation(state)
            break

    # ── Hard limit reached ────────────────────────────────────
    if not state.finished and not state.escalated:
        state.escalated = True
        state.escalation_reason = EscalationReason.MAX_STEPS_EXCEEDED
        tracer.log_escalation(state)

    state.total_latency_ms = (time.time() - episode_start) * 1000

    # ── Build trace ───────────────────────────────────────────
    trace = EpisodeTrace(
        episode_id=state.episode_id,
        query=query,
        dataset=dataset,
        ground_truth=ground_truth,
        final_answer=state.final_answer,
        sub_approach=state.sub_approach.value,
        steps=scratchpad.steps,
        n_steps=state.current_step,
        finished=state.finished,
        escalated=state.escalated,
        escalation_reason=state.escalation_reason.value if state.escalation_reason else None,
        correct=None,
        total_tokens=scratchpad.total_tokens,
        total_latency_ms=state.total_latency_ms,
        total_cost_usd=estimate_cost(scratchpad.total_tokens, MODEL_NAME),
        model=MODEL_NAME,
        timestamp=datetime.utcnow().isoformat(),
    )
    tracer.log_end(trace)
    return trace




def truncate_scratchpad(scratchpad: Scratchpad) -> Scratchpad:
    """
    Drops the oldest Thought/Action/Observation triplets
    while always preserving the first triplet and the last two.
    Implements the sliding-window strategy from Yao et al. Appendix.
    """
    steps = scratchpad.steps
    if len(steps) <= 6:
        return scratchpad
    preserved = steps[:3] + steps[-6:]
    new_sp = Scratchpad(query=scratchpad.query, system_prompt=scratchpad.system_prompt)
    new_sp.steps = preserved
    new_sp.total_tokens = sum(s.token_count for s in preserved)
    return new_sp



COST_PER_1K_TOKENS: dict = {
    "gpt-4o":                    0.005,
    "llama-3.1-70b":             0.0009,
    "llama-3.3-70b-versatile":   0.0009,   # ← add this
    "llama-3.1-8b-instant":      0.0002,
    "mixtral-8x7b-32768":        0.0006,
}

def estimate_cost(total_tokens: int, model: str) -> float:
    rate = COST_PER_1K_TOKENS.get(model, 0.005)
    return round((total_tokens / 1000) * rate, 6)




import json
import logging
import os


logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("react_agent")


class EpisodeTracer:
    """
    Writes structured traces to disk (JSONL) and optionally to W&B.
    One tracer instance per episode.
    """
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.trace_path = os.path.join(TRACE_DIR, f"{episode_id}.jsonl")
        os.makedirs(TRACE_DIR, exist_ok=True)

    def _write(self, record: dict) -> None:
        with open(self.trace_path, "a") as f:
            f.write(json.dumps(record) + "\n")

    def log_start(self, state: ReActAgentState) -> None:
        record = {
            "event": "episode_start",
            "episode_id": state.episode_id,
            "query": state.query,
            "dataset": state.dataset,
            "confidence_score": state.confidence_score,
            "sub_approach": state.sub_approach.value,
            "model": MODEL_NAME,
        }
        logger.info(f"[START] {state.episode_id} | dataset={state.dataset} | approach={state.sub_approach.value}")
        self._write(record)

    def log_step(self, step: ReActStep) -> None:
        record = {
            "event": "step",
            "step_index": step.step_index,
            "step_type": step.step_type.value,
            "content": step.content[:500],   # truncated for log
            "tool_name": step.tool_name,
            "token_count": step.token_count,
            "latency_ms": round(step.latency_ms, 2),
            "timestamp": step.timestamp,
        }
        logger.debug(f"  [{step.step_type.value}] step={step.step_index} | tool={step.tool_name}")
        self._write(record)

    def log_escalation(self, state: ReActAgentState) -> None:
        record = {
            "event": "escalation",
            "episode_id": state.episode_id,
            "reason": state.escalation_reason.value if state.escalation_reason else "unknown",
            "step": state.current_step,
        }
        logger.warning(f"[ESCALATE] {state.episode_id} | reason={record['reason']} | step={state.current_step}")
        self._write(record)

    def log_end(self, trace: EpisodeTrace) -> None:
        record = {
            "event": "episode_end",
            "episode_id": trace.episode_id,
            "finished": trace.finished,
            "escalated": trace.escalated,
            "n_steps": trace.n_steps,
            "total_tokens": trace.total_tokens,
            "total_latency_ms": round(trace.total_latency_ms, 2),
            "total_cost_usd": trace.total_cost_usd,
            "final_answer": trace.final_answer,
        }
        logger.info(
            f"[END] {trace.episode_id} | steps={trace.n_steps} | "
            f"finished={trace.finished} | tokens={trace.total_tokens} | "
            f"cost=${trace.total_cost_usd}"
        )
        self._write(record)

    def log_to_wandb(self, trace: EpisodeTrace) -> None:
        """Optional: call after episode to push metrics to Weights & Biases."""
        try:
            import wandb
            wandb.log({
                "n_steps": trace.n_steps,
                "total_tokens": trace.total_tokens,
                "total_cost_usd": trace.total_cost_usd,
                "total_latency_ms": trace.total_latency_ms,
                "finished": int(trace.finished),
                "escalated": int(trace.escalated),
                "sub_approach": trace.sub_approach,
                "dataset": trace.dataset,
                "correct": int(trace.correct) if trace.correct is not None else -1,
            })
        except ImportError:
            logger.warning("wandb not installed — skipping remote logging")



def aggregate_traces(trace_dir: str) -> dict:
    """
    Reads all episode JSONL files in trace_dir and computes summary stats.
    Call after a full dataset run.

    Returns dict with:
        n_episodes, accuracy, avg_steps, avg_tokens, avg_cost,
        escalation_rate, finish_rate, per_dataset_breakdown
    """
    import glob
    import statistics

    episodes = []
    for path in glob.glob(os.path.join(trace_dir, "*.jsonl")):
        end_record = None
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("event") == "episode_end":
                    end_record = rec
        if end_record:
            episodes.append(end_record)

    if not episodes:
        return {}

    return {
        "n_episodes": len(episodes),
        "finish_rate": sum(e["finished"] for e in episodes) / len(episodes),
        "escalation_rate": sum(e["escalated"] for e in episodes) / len(episodes),
        "avg_steps": statistics.mean(e["n_steps"] for e in episodes),
        "avg_tokens": statistics.mean(e["total_tokens"] for e in episodes),
        "avg_cost_usd": statistics.mean(e["total_cost_usd"] for e in episodes),
        "avg_latency_ms": statistics.mean(e["total_latency_ms"] for e in episodes),
    }



class ReActEvaluator:
    """
    Post-hoc correctness scoring. Attach ground truth and score.
    Dataset-specific exact-match or fuzzy-match logic.
    """

    def evaluate(self, trace: EpisodeTrace) -> bool:
        if trace.final_answer is None or trace.ground_truth is None:
            return False
        if trace.dataset == "MathHard_L5":
            return self._math_match(trace.final_answer, trace.ground_truth)
        return self._exact_match(trace.final_answer, trace.ground_truth)

    def _exact_match(self, pred: str, gold: str) -> bool:
        return pred.strip().lower() == gold.strip().lower()

    def _math_match(self, pred: str, gold: str) -> bool:
        """Numeric or symbolic equivalence check."""
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            return self._exact_match(pred, gold)



if __name__ == "__main__":

    # ══════════════════════════════════════════════════════════════════════
    # 1. WebSearchTool smoke test
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TESTING: WebSearchTool")
    print("=" * 60)

    search_tool = WebSearchTool()
    result = search_tool.run("capital of France")

    print("success     :", result.success)
    print("latency_ms  :", result.latency_ms)
    print("output      :", result.output)
    print("error       :", result.error_message)

    assert result.success is True,          f"FAIL: {result.error_message}"
    assert isinstance(result.output, str),  f"FAIL: output is {type(result.output)}"
    assert result.latency_ms > 0,           "FAIL: latency not measured"
    assert "Paris" in result.output,        f"FAIL: Paris not in output — {result.output}"
    print("WebSearchTool: PASSED ✅\n")


    # ══════════════════════════════════════════════════════════════════════
    # 2. FileReaderTool smoke test
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TESTING: FileReaderTool")
    print("=" * 60)

    file_tool = FileReaderTool()

    # PDF
    pdf_file = "67e8878b-5cef-4375-804e-e6291fdbe78a.pdf"
    print(f"\n  [{pdf_file}]")
    r = file_tool.run(pdf_file)
    print("  success     :", r.success)
    print("  latency_ms  :", r.latency_ms)
    print("  output[:300]:", r.output[:300] if r.output else None)
    print("  error       :", r.error_message)
    assert r.success is True,             f"FAIL pdf: {r.error_message}"
    assert isinstance(r.output, str),     "FAIL pdf: output not a string"
    assert len(r.output) > 0,            "FAIL pdf: output is empty"
    assert len(r.output) <= 4000,        f"FAIL pdf: output too long ({len(r.output)})"
    print("  PDF: PASSED ✅")

    # XLSX
    xlsx_file = "076c8171-9b3b-49b9-a477-244d2a532826.xlsx"
    print(f"\n  [{xlsx_file}]")
    r = file_tool.run(xlsx_file)
    print("  success     :", r.success)
    print("  latency_ms  :", r.latency_ms)
    print("  output[:300]:", r.output[:300] if r.output else None)
    print("  error       :", r.error_message)
    assert r.success is True,             f"FAIL xlsx: {r.error_message}"
    assert isinstance(r.output, str),     "FAIL xlsx: output not a string"
    assert len(r.output) > 0,            "FAIL xlsx: output is empty"
    assert len(r.output) <= 4000,        f"FAIL xlsx: output too long ({len(r.output)})"
    print("  XLSX: PASSED ✅")

    # PNG
    png_file = "6359a0b1-8f7b-499b-9336-840f9ab90688.png"
    print(f"\n  [{png_file}]")
    r = file_tool.run(png_file)
    print("  success     :", r.success)
    print("  latency_ms  :", r.latency_ms)
    print("  output[:300]:", r.output[:300] if r.output else None)
    print("  error       :", r.error_message)
    assert r.success is True,             f"FAIL png: {r.error_message}"
    assert isinstance(r.output, str),     "FAIL png: output not a string"
    assert len(r.output) <= 4000,        f"FAIL png: output too long ({len(r.output)})"
    print("  PNG: PASSED ✅")

    # Missing file — must fail gracefully
    print(f"\n  [missing_file.pdf — expected failure]")
    r = file_tool.run("this_does_not_exist.pdf")
    print("  success     :", r.success)
    print("  error       :", r.error_message)
    assert r.success is False,            "FAIL: should be False for missing file"
    assert r.output is None,             "FAIL: output should be None"
    assert "not found" in r.error_message, "FAIL: error_message should say 'not found'"
    print("  Missing file error case: PASSED ✅\n")


    # ══════════════════════════════════════════════════════════════════════
    # 3. Full ReAct episode — GAIA query with attached PDF
    # ══════════════════════════════════════════════════════════════════════
    print("=" * 60)
    print("TESTING: Full ReAct Episode (GAIA query + file)")
    print("=" * 60)

    # Real GAIA-style query — agent must read the file AND search the web
    gaia_query = (
        "Based on the attached PDF file 67e8878b-5cef-4375-804e-e6291fdbe78a.pdf, "
        "what is the main topic discussed and are there any recent developments "
        "related to it? Use the file_reader to read the file first, "
        "then use search to find related recent information."
    )

    print(f"\nQuery : {gaia_query}")
    print(f"Dataset: GAIA")
    print(f"File   : {pdf_file}")
    print("-" * 60)

    trace = run_react_episode(
        query            = gaia_query,
        dataset          = "GAIA",
        modality         = "text",
        confidence_score = 0.65,
        ground_truth     = None,         # no ground truth for manual test
        few_shot_examples= {},
    )

    print("\n── Episode Result ──────────────────────────────────────")
    print("Final answer :", trace.final_answer)
    print("Steps taken  :", trace.n_steps)
    print("Total tokens :", trace.total_tokens)
    print("Cost USD     :", trace.total_cost_usd)
    print("Finished     :", trace.finished)
    print("Escalated    :", trace.escalated)
    print("Escalation   :", trace.escalation_reason)
    print("Trace file   :", f"./traces/{trace.episode_id}.jsonl")

    print("\n── Step-by-step trace ──────────────────────────────────")
    for step in trace.steps:
        print(f"  [{step.step_type.value:12}] step={step.step_index} | tool={step.tool_name}")
        print(f"    {step.content[:150].strip()}")

    # Assertions on episode health
    assert trace.episode_id is not None,     "FAIL: no episode_id"
    assert trace.n_steps > 0,               "FAIL: no steps taken"
    assert trace.total_tokens > 0,          "FAIL: no tokens used"
    assert trace.total_cost_usd >= 0,       "FAIL: negative cost"
    assert trace.finished or trace.escalated, "FAIL: neither finished nor escalated"

    if trace.finished:
        assert trace.final_answer is not None, "FAIL: finished but no final_answer"
        print("\nEpisode completed successfully ✅")
    else:
        print(f"\nEpisode escalated — reason: {trace.escalation_reason}")
        print("This is acceptable for a hard GAIA task ⚠️")

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)

# if __name__ == "__main__":

#     # Point to your actual GAIA files directory
#     UPLOAD_DIR = "../datasets/gaia_files/2023/validation"   # ✅ fix Bug 1

#     # ══════════════════════════════════════════════════════════════════════
#     # 1. WebSearchTool
#     # ══════════════════════════════════════════════════════════════════════
#     print("=" * 60)
#     print("TESTING: WebSearchTool")
#     print("=" * 60)

#     search_tool = WebSearchTool()
#     result = search_tool.run("capital of France")

#     print("success     :", result.success)
#     print("latency_ms  :", result.latency_ms)
#     print("output      :", result.output)
#     print("error       :", result.error_message)

#     assert result.success is True,              f"FAIL: success is False — {result.error_message}"
#     assert isinstance(result.output, str),      f"FAIL: output is {type(result.output)}, expected str"
#     assert result.latency_ms > 0,               f"FAIL: latency_ms is 0 — timing not measured"
#     assert "Paris" in result.output,            f"FAIL: 'Paris' not found in output — {result.output}"

#     print("WebSearchTool: PASSED ✅\n")

#     # ══════════════════════════════════════════════════════════════════════
#     # 2. FileReaderTool — one test per supported format
#     # ══════════════════════════════════════════════════════════════════════
#     print("=" * 60)
#     print("TESTING: FileReaderTool")
#     print("=" * 60)

#     file_tool = FileReaderTool()

#     def test_file(filename: str, expected_keyword: str = None):
#         print(f"\n  [{filename}]")
#         result = file_tool.run(filename)
#         print("  success     :", result.success)
#         print("  latency_ms  :", result.latency_ms)
#         print("  output[:200]:", result.output[:200] if result.output else None)
#         print("  error       :", result.error_message)

#         assert result.success is True,              f"FAIL {filename}: success is False — {result.error_message}"
#         assert isinstance(result.output, str),      f"FAIL {filename}: output is {type(result.output)}, expected str"
#         assert len(result.output) > 0,              f"FAIL {filename}: output is empty"
#         assert len(result.output) <= 4000,          f"FAIL {filename}: output exceeds 4000 chars — {len(result.output)}"
#         assert result.latency_ms > 0,               f"FAIL {filename}: latency_ms is 0"

#         if expected_keyword:
#             assert expected_keyword in result.output, \
#                 f"FAIL {filename}: '{expected_keyword}' not found in output"

#         print(f"  {filename}: PASSED ✅")

#     # ✅ Pass just filenames — UPLOAD_DIR handles the directory
#     test_file("67e8878b-5cef-4375-804e-e6291fdbe78a.pdf",  expected_keyword=None)
#     test_file("076c8171-9b3b-49b9-a477-244d2a532826.xlsx", expected_keyword=None)
#     test_file("6359a0b1-8f7b-499b-9336-840f9ab90688.png",  expected_keyword=None)

#     # ── Error case — genuinely missing file ─────────────────────────────
#     print("\n  [this_file_does_not_exist.pdf — expected to fail gracefully]")
#     result = file_tool.run("this_file_does_not_exist.pdf")  # ✅ fix Bug 2
#     print("  success     :", result.success)
#     print("  error       :", result.error_message)
#     assert result.success is False,             "FAIL: should be False for missing file"
#     assert result.output is None,               "FAIL: output should be None for missing file"
#     assert "not found" in result.error_message, "FAIL: error_message should say 'not found'"
#     print("  missing file error case: PASSED ✅")

#     print("\n" + "=" * 60)
#     print("ALL TESTS PASSED ✅")
#     print("=" * 60)





class PythonExecTool(BaseTool):
    schema = ToolSchema(
        name="python_exec",
        description="Execute Python code. Use for arithmetic, data manipulation, file parsing.",
        input_schema={"type": "string", "description": "Valid Python code string"},
        returns="stdout of execution, truncated to 2000 chars",
        timeout_sec=20,
    )
    def run(self, tool_input: str) -> ToolResult:
        import subprocess, sys
        start = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", tool_input],
                capture_output=True, text=True, timeout=20
            )
            output = (proc.stdout + proc.stderr)[:2000]
            return ToolResult(
                tool_name="python_exec",
                success=proc.returncode == 0,
                output=output,
                error_message=proc.stderr[:500] if proc.returncode != 0 else None,
                latency_ms=(time.time() - start) * 1000,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(tool_name="python_exec", success=False, output=None,
                              error_message="Timeout after 20s",
                              latency_ms=(time.time() - start) * 1000)
        except Exception as e:
            return ToolResult(tool_name="python_exec", success=False, output=None,
                              error_message=str(e),
                              latency_ms=(time.time() - start) * 1000)


class BashExecTool(BaseTool):
    schema = ToolSchema(
        name="bash_exec",
        description="Run bash commands inside the target repository.",
        input_schema={"type": "string", "description": "Bash command string"},
        returns="stdout + stderr, truncated to 3000 chars",
        is_stateful=True,
        timeout_sec=60,
    )
    def run(self, tool_input: str) -> ToolResult: ...


class SympySolverTool(BaseTool):
    schema = ToolSchema(
        name="sympy_solver",
        description="Solve mathematical expressions symbolically using SymPy.",
        input_schema={"type": "string", "description": "SymPy-compatible expression string"},
        returns="Symbolic or numerical solution as string",
    )
    def run(self, tool_input: str) -> ToolResult:
        


class CalculatorTool(BaseTool):
    schema = ToolSchema(
        name="calculator",
        description="Evaluate arithmetic expressions. Use for simple numeric computation.",
        input_schema={"type": "string", "description": "Arithmetic expression, e.g. '(3 + 4) * 2 / 7'"},
        returns="Numeric result as string",
    )
    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            # Safe eval — only allow math ops
            allowed = set("0123456789+-*/().% ")
            if not all(c in allowed for c in tool_input):
                raise ValueError(f"Unsafe expression: {tool_input}")
            result = eval(tool_input, {"__builtins__": {}}, {})  # noqa: S307
            return ToolResult(
                tool_name="calculator",
                success=True,
                output=str(result),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="calculator",
                success=False,
                output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )
