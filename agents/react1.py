from __future__ import annotations

import ast
import json
import logging
import operator
import os
import re
import subprocess
import sys
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

import pandas as pd
import pdfplumber
import pypdf
import pytesseract
import serpapi
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
from unstructured.partition.auto import partition

load_dotenv()

# ── Clients ───────────────────────────────────────────────────────────────────
groq_client    = Groq(api_key=os.getenv("GROQ_API_KEY"))
client_serpapi = serpapi.Client(api_key=os.getenv("SERP_API_KEY"))

# ── Logging — configured ONCE                                        [FIX F1] ─
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("react_agent")

# ── Directories (env-overridable) ─────────────────────────────────────────────
UPLOAD_DIR = os.getenv("UPLOAD_DIR", "../datasets/gaia_files/2023/validation")
TRACE_DIR  = os.getenv("TRACE_DIR",  "./traces")

# ── Search location (env-overridable)                               [FIX F13] ─
SERPAPI_LOCATION = os.getenv("SERPAPI_LOCATION", "United States")

# ── Model / loop constants ────────────────────────────────────────────────────
MODEL_NAME             = "llama-3.3-70b-versatile"
TEMPERATURE            = 0.0
MAX_STEPS              = 15
MAX_TOKENS_PER_STEP    = 512
# llama-3.3-70b-versatile supports 128 K context; 32 K is a safe working budget
MAX_CONTEXT_TOKENS     = 32_768
TOOL_CALL_TIMEOUT_SEC  = 30
WANDB_PROJECT          = "react-adaptive-routing"

DATASET_CONFIGS: dict = {
    "GAIA":         {"max_steps": 15, "tools": ["search", "file_reader", "calculator", "python_exec"]},
    "SWE_verified": {"max_steps": 20, "tools": ["bash_exec", "file_reader", "python_exec", "grep"]},
    "mmlupro":      {"max_steps": 10, "tools": ["calculator", "search", "python_exec"]},
    "MathHard_L5":  {"max_steps": 12, "tools": ["sympy_solver", "calculator", "python_exec"]},
}

COST_PER_1K_TOKENS: dict = {
    "llama-3.3-70b-versatile": 0.0009,
    "llama-3.1-70b":           0.0009,
    "llama-3.1-8b-instant":    0.0002,
    "gpt-4o":                  0.005,
}


# ── Enums ─────────────────────────────────────────────────────────────────────

class StepType(Enum):
    THOUGHT     = "Thought"
    ACTION      = "Action"
    OBSERVATION = "Observation"
    FINISH      = "Finish"


class EscalationReason(Enum):
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TOOL_FAILURE       = "tool_failure"
    LOW_CONFIDENCE     = "low_confidence"
    PARSE_ERROR        = "parse_error"


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    step_index:  int
    step_type:   StepType
    content:     str
    tool_name:   Optional[str] = None
    tool_input:  Optional[Any] = None
    tool_output: Optional[Any] = None
    token_count: int   = 0
    latency_ms:  float = 0.0
    timestamp: str = field(                                         # [FIX F8]
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


@dataclass
class ToolSchema:
    name:         str
    description:  str
    input_schema: dict
    returns:      str
    is_stateful:  bool = False
    timeout_sec:  int  = TOOL_CALL_TIMEOUT_SEC


@dataclass
class ToolResult:
    tool_name:     str
    success:       bool
    output:        Any
    error_message: Optional[str] = None
    latency_ms:    float = 0.0
    raw_output:    Optional[str] = None


@dataclass
class Scratchpad:
    """Persistent context window for one ReAct episode (Yao et al. §3)."""
    query:         str
    system_prompt: str
    steps:         list[ReActStep] = field(default_factory=list)
    total_tokens:  int = 0

    def append(self, step: ReActStep) -> None:
        self.steps.append(step)
        self.total_tokens += step.token_count

    def to_prompt_string(self) -> str:
        lines = [f"Question: {self.query}\n"]
        for s in self.steps:
            if s.step_type == StepType.THOUGHT:
                lines.append(f"Thought {s.step_index}: {s.content}")
            elif s.step_type == StepType.ACTION:
                lines.append(f"Action {s.step_index}: {s.tool_name}")
                lines.append(f"Action Input {s.step_index}: {s.tool_input}")
            elif s.step_type == StepType.OBSERVATION:
                lines.append(f"Observation {s.step_index}: {s.content}")
            elif s.step_type == StepType.FINISH:
                lines.append(f"Final Answer: {s.content}")
        return "\n".join(lines)

    def needs_truncation(self) -> bool:
        return self.total_tokens > MAX_CONTEXT_TOKENS

    def token_budget_remaining(self) -> int:
        return MAX_CONTEXT_TOKENS - self.total_tokens


@dataclass
class ReActAgentState:
    episode_id:        str = field(default_factory=lambda: str(uuid.uuid4()))
    query:             str = ""
    dataset:           str = ""
    modality:          str = ""
    confidence_score:  float = 0.0
    sub_approach:      str = 'react'
    scratchpad:        Optional[Scratchpad] = None
    current_step:      int = 0
    finished:          bool = False
    escalated:         bool = False
    escalation_reason: Optional[EscalationReason] = None
    final_answer:      Optional[str] = None
    ground_truth:      Optional[str] = None
    total_latency_ms:  float = 0.0
    total_cost_usd:    float = 0.0


@dataclass
class EpisodeTrace:
    episode_id:        str
    query:             str
    steps:             list[ReActStep]
    n_steps:           int
    finished:          bool
    escalated:         bool
    escalation_reason: Optional[str]
    correct:           Optional[bool]
    total_tokens:      int
    total_latency_ms:  float
    total_cost_usd:    float
    model:             str
    timestamp:         str
    dataset:           str
    ground_truth:      Optional[str]
    final_answer:      Optional[str]
    sub_approach:      str = 'react'
    
    
    
    
    
    


# ── System prompts ────────────────────────────────────────────────────────────
# {{n}} renders as literal {n} after .format() so the LLM sees "Thought {n}:"
# The parser uses \d* and matches both numbered and plain labels.

REACT_SYSTEM_PROMPT = """You are a ReAct agent. Solve problems by interleaving Thought and Action steps.

Format (n = step number):
Thought {{n}}: <1-3 sentences of reasoning>
Action {{n}}: <one tool from: {tool_names}>
Action Input {{n}}: <input to the tool>
Observation {{n}}: <filled by the system — do NOT write this yourself>

When you have enough information, output ONLY:
Final Answer: <your concise answer>

Rules:
- Always start with a Thought.
- Never output Final Answer without at least one Action/Observation exchange.
- One tool per Action step.

Dataset: {dataset} | Modality: {modality} | Approach: {task_type}

Available tools:
{tool_descriptions}

{few_shot_example}"""

# ── Token estimation                                                 [FIX F2] ─

def rough_token_count(text: str) -> int:
    """~4 chars per token — safe heuristic for Llama tokenizers."""
    return max(1, len(str(text)) // 4)


# ── Tools ─────────────────────────────────────────────────────────────────────

class BaseTool(ABC):
    schema: ToolSchema

    @abstractmethod
    def run(self, tool_input: Any) -> ToolResult: ...

    def validate_input(self, tool_input: Any) -> bool:
        return True


class WebSearchTool(BaseTool):
    schema = ToolSchema(
        name="search",
        description="Search the web for factual information. Use for current events or definitions.",
        input_schema={"type": "string"},
        returns="Top-3 search result snippets as a single string",
    )

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            results      = client_serpapi.search({
                "q":             tool_input,
                "location":      SERPAPI_LOCATION,                  # [FIX F13]
                "hl":            "en",
                "gl":            "us",
                "google_domain": "google.com",
            })
            results_dict = dict(results)
            organic      = results_dict.get("organic_results", [])[:3]
            snippets     = "\n".join(
                f"{i+1}. {r.get('title','')}: {r.get('snippet','')}"
                for i, r in enumerate(organic)
            )
            return ToolResult(
                tool_name="search", success=True, output=snippets,
                latency_ms=(time.time() - start) * 1000,
                raw_output=json.dumps(results_dict),
            )
        except Exception as e:
            return ToolResult(
                tool_name="search", success=False, output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


class PythonExecTool(BaseTool):                                     # [FIX F12]
    schema = ToolSchema(
        name="python_exec",
        description="Execute Python code. Use for arithmetic, data manipulation, file parsing.",
        input_schema={"type": "string"},
        returns="stdout of execution, truncated to 2000 chars",
        timeout_sec=20,
    )

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            proc   = subprocess.run(
                [sys.executable, "-c", tool_input],
                capture_output=True, text=True, timeout=20,
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
            return ToolResult(
                tool_name="python_exec", success=False, output=None,
                error_message="Timeout after 20s",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="python_exec", success=False, output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


class BashExecTool(BaseTool):                                       # [FIX F4]
    schema = ToolSchema(
        name="bash_exec",
        description="Run bash commands inside the target repository.",
        input_schema={"type": "string"},
        returns="stdout + stderr, truncated to 3000 chars",
        is_stateful=True,
        timeout_sec=60,
    )

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            proc   = subprocess.run(
                tool_input, shell=True,
                capture_output=True, text=True, timeout=60,
            )
            output = (proc.stdout + proc.stderr)[:3000]
            return ToolResult(
                tool_name="bash_exec",
                success=proc.returncode == 0,
                output=output,
                error_message=proc.stderr[:500] if proc.returncode != 0 else None,
                latency_ms=(time.time() - start) * 1000,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="bash_exec", success=False, output=None,
                error_message="Timeout after 60s",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="bash_exec", success=False, output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


# ── Safe arithmetic evaluator — replaces eval()              [FIX F15] ────────
_SAFE_OPS: dict = {
    ast.Add:  operator.add,
    ast.Sub:  operator.sub,
    ast.Mult: operator.mul,
    ast.Div:  operator.truediv,
    ast.Mod:  operator.mod,
    ast.Pow:  operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

def _ast_eval(node: ast.expr) -> float:
    if isinstance(node, ast.Expression):
        return _ast_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.BinOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {node.op}")
        return op(_ast_eval(node.left), _ast_eval(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _SAFE_OPS.get(type(node.op))
        if op is None:
            raise ValueError(f"Operator not allowed: {node.op}")
        return op(_ast_eval(node.operand))
    raise ValueError(f"Expression type not allowed: {type(node).__name__}")


class CalculatorTool(BaseTool):                                     # [FIX F12, F15]
    schema = ToolSchema(
        name="calculator",
        description="Evaluate arithmetic expressions. Use for simple numeric computation.",
        input_schema={"type": "string", "description": "e.g. '(3 + 4) * 2 / 7'"},
        returns="Numeric result as string",
    )

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            tree   = ast.parse(tool_input.strip(), mode="eval")
            result = _ast_eval(tree)
            return ToolResult(
                tool_name="calculator", success=True, output=str(result),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="calculator", success=False, output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


class SympySolverTool(BaseTool):                                    # [FIX F4, F11]
    schema = ToolSchema(
        name="sympy_solver",
        description="Solve mathematical expressions symbolically using SymPy.",
        input_schema={"type": "string", "description": "SymPy-compatible expression"},
        returns="Symbolic or numerical solution as string",
    )

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            import sympy
            expr   = sympy.sympify(tool_input)
            result = sympy.simplify(expr)
            return ToolResult(
                tool_name="sympy_solver", success=True, output=str(result),
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="sympy_solver", success=False, output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


class FileReaderTool(BaseTool):
    schema = ToolSchema(
        name="file_reader",
        description="Read the contents of an attached file. Input the filename only.",
        input_schema={"type": "string"},
        returns="File content string, truncated to 4000 chars",
    )

    EXTRACTORS: dict[str, list[str]] = {
        ".pdf":  ["_pdfplumber", "_pypdf"],
        ".xlsx": ["_xlsx"],
        ".png":  ["_tesseract"],
        ".pdb":  ["_pdb_records"],
        ".csv":  ["_csv"],
    }
    MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            filename  = os.path.basename(tool_input.strip())
            file_path = os.path.join(UPLOAD_DIR, filename)

            # Path traversal guard
            if not os.path.realpath(file_path).startswith(os.path.realpath(UPLOAD_DIR)):
                raise PermissionError("Path traversal detected.")

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File '{filename}' not found in {UPLOAD_DIR}")

            if os.path.getsize(file_path) > self.MAX_FILE_BYTES:
                raise ValueError(f"File exceeds {self.MAX_FILE_BYTES // 1_048_576} MB limit.")

            ext     = os.path.splitext(file_path)[1].lower()
            content = self._extract(file_path, ext)
            return ToolResult(
                tool_name="file_reader", success=True, output=content[:4000],
                latency_ms=(time.time() - start) * 1000,
                raw_output=content,
            )
        except Exception as e:
            return ToolResult(
                tool_name="file_reader", success=False, output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )

    def _extract(self, file_path: str, ext: str) -> str:
        for name in self.EXTRACTORS.get(ext, ["_plain_text"]):
            try:
                content = getattr(self, name)(file_path)
                if content and content.strip():
                    return content
            except Exception:
                continue
        return self._unstructured(file_path)

    @staticmethod
    def _pdfplumber(path: str) -> str:
        with pdfplumber.open(path) as pdf:
            return "\n".join(p.extract_text() or "" for p in pdf.pages)

    @staticmethod
    def _pypdf(path: str) -> str:
        reader = pypdf.PdfReader(path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    @staticmethod
    def _xlsx(path: str) -> str:
        xl = pd.ExcelFile(path)
        return "\n\n".join(
            f"--- Sheet: {name} ---\n{xl.parse(name).to_string(index=False)}"
            for name in xl.sheet_names
        )

    @staticmethod
    def _tesseract(path: str) -> str:
        return pytesseract.image_to_string(Image.open(path).convert("RGB"))

    @staticmethod
    def _pdb_records(path: str) -> str:
        USEFUL = {"HEADER", "TITLE", "REMARK", "SEQRES", "ATOM", "HETATM", "COMPND"}
        with open(path, encoding="utf-8") as f:
            return "\n".join(l.rstrip() for l in f if l[:6].strip() in USEFUL)

    @staticmethod
    def _csv(path: str) -> str:
        return pd.read_csv(path).to_string()

    @staticmethod
    def _plain_text(path: str) -> str:
        with open(path, encoding="utf-8") as f:
            return f.read()

    @staticmethod
    def _unstructured(path: str) -> str:
        return "\n".join(str(el) for el in partition(filename=path))


class GrepTool(BaseTool):                                           # [FIX F4, F16]
    schema = ToolSchema(
        name="grep",
        description=(
            "Search for a pattern in repository files. "
            "Input format: 'PATTERN [PATH]' — PATH defaults to current directory."
        ),
        input_schema={"type": "string"},                            # [FIX F16 — was dict]
        returns="Matching lines with file:line_number format, truncated to 3000 chars",
    )

    def run(self, tool_input: str) -> ToolResult:
        start = time.time()
        try:
            parts   = tool_input.strip().split(maxsplit=1)
            pattern = parts[0]
            path    = parts[1] if len(parts) > 1 else "."
            proc    = subprocess.run(
                ["grep", "-rn", "--", pattern, path],
                capture_output=True, text=True, timeout=30,
            )
            output  = (proc.stdout + proc.stderr)[:3000]
            return ToolResult(
                tool_name="grep", success=True,
                output=output or "No matches found.",
                latency_ms=(time.time() - start) * 1000,
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                tool_name="grep", success=False, output=None,
                error_message="Timeout after 30s",
                latency_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            return ToolResult(
                tool_name="grep", success=False, output=None,
                error_message=str(e),
                latency_ms=(time.time() - start) * 1000,
            )


# ── Tool registry ─────────────────────────────────────────────────────────────

class ToolRegistry:
    # Instantiated once at class level — tools are stateless singletons
    _ALL_TOOLS: dict[str, BaseTool] = {
        "search":       WebSearchTool(),
        "python_exec":  PythonExecTool(),
        "bash_exec":    BashExecTool(),
        "sympy_solver": SympySolverTool(),
        "calculator":   CalculatorTool(),
        "file_reader":  FileReaderTool(),
        "grep":         GrepTool(),
    }

    def __init__(self, dataset: str):
        if dataset not in DATASET_CONFIGS:                          # [FIX F6]
            raise ValueError(
                f"Unknown dataset '{dataset}'. "
                f"Valid: {list(DATASET_CONFIGS.keys())}"
            )
        tool_names   = DATASET_CONFIGS[dataset]["tools"]           # [FIX F5 — was NameError]
        self._tools  = {k: v for k, v in self._ALL_TOOLS.items() if k in tool_names}

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def descriptions(self) -> str:
        return "\n".join(
            f"- {t.schema.name}: {t.schema.description}"
            for t in self._tools.values()
        )

    def names(self) -> list[str]:
        return list(self._tools.keys())

# ── Prompt builder ────────────────────────────────────────────────────────────

def build_system_prompt(
    dataset:           str,
    modality:          str,
    tool_registry:     ToolRegistry,
    sub_approach:      SubApproach,
    few_shot_examples: Optional[dict] = None,                       # [FIX F3]
) -> str:
    few_shot_examples = few_shot_examples or {}
    task_type = "ReAct reasoning and acting"    
    return REACT_SYSTEM_PROMPT.format(
        tool_names        =", ".join(tool_registry.names()),
        dataset           =dataset,
        modality          =modality,
        task_type         =task_type,
        tool_descriptions =tool_registry.descriptions(),
        few_shot_example  =few_shot_examples.get(dataset, ""),
    )


# ── LLM call ─────────────────────────────────────────────────────────────────

def call_llm(
    system_prompt: str,
    user_content:  str,
    model:         str   = MODEL_NAME,
    temperature:   float = TEMPERATURE,
    max_tokens:    int   = MAX_TOKENS_PER_STEP,
) -> tuple[str, int, float]:
    """Returns (response_text, total_tokens, latency_ms)."""
    start    = time.time()
    response = groq_client.chat.completions.create(
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
    )
    return (
        response.choices[0].message.content,
        response.usage.total_tokens,
        (time.time() - start) * 1000,
    )


# ── Output parser ─────────────────────────────────────────────────────────────

def parse_llm_output(
    text:       str,
    step_index: int,
) -> tuple[Optional[str], Optional[str], Optional[str], bool]:
    """
    Returns (thought, action_name, action_input, is_finish).
    Accepts both numbered (Thought 1:) and plain (Thought:) labels.
    """
    # Accept Final Answer only after step 1 — ensures ≥1 tool call happened  [FIX F14]
    finish_match = re.search(r"Final Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    MIN_STEPS = 3

    if finish_match and step_index >= MIN_STEPS:
        return None, None, None, True

    thought_match = re.search(
        r"Thought\s*\d*\s*:\s*(.+?)(?=Action\s*\d*\s*:|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    action_match = re.search(
        r"Action\s*\d*\s*:\s*(.+?)(?=Action\s+Input\s*\d*\s*:|$)",
        text, re.DOTALL | re.IGNORECASE,
    )
    input_match = re.search(
        r"Action\s+Input\s*\d*\s*:\s*(.+?)(?=Observation\s*\d*\s*:|$)",
        text, re.DOTALL | re.IGNORECASE,
    )

    thought      = thought_match.group(1).strip() if thought_match else None
    action_name  = action_match.group(1).strip()  if action_match  else None
    action_input = input_match.group(1).strip()   if input_match   else None

    # Normalise tool name — LLM sometimes appends extra words after the name
    if action_name:
        action_name = action_name.split()[0].lower().strip(".,:")

    return thought, action_name, action_input, False


def extract_final_answer(text: str) -> Optional[str]:
    match = re.search(r"Final Answer\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


# ── Scratchpad truncation ─────────────────────────────────────────────────────

def truncate_scratchpad(scratchpad: Scratchpad) -> Scratchpad:
    """
    Preserves first 3 steps + last 6 steps.
    Only runs when len > 9 so at least one complete triplet exists on each end.
    """
    steps = scratchpad.steps
    if len(steps) <= 9:
        return scratchpad
    preserved   = steps[:3] + steps[-6:]
    new_sp      = Scratchpad(query=scratchpad.query, system_prompt=scratchpad.system_prompt)
    new_sp.steps        = preserved
    new_sp.total_tokens = sum(rough_token_count(s.content) for s in preserved)
    return new_sp


# ── Cost estimation ───────────────────────────────────────────────────────────

def estimate_cost(total_tokens: int, model: str) -> float:
    rate = COST_PER_1K_TOKENS.get(model, 0.005)
    return round((total_tokens / 1000) * rate, 6)


# ── Main episode runner ───────────────────────────────────────────────────────

def run_react_episode(
    query:             str,
    dataset:           str,
    modality:          str,
    confidence_score:  float,
    ground_truth:      Optional[str] = None,
    few_shot_examples: Optional[dict] = None,                       # [FIX F3]
) -> EpisodeTrace:
    episode_start     = time.time()
    few_shot_examples = few_shot_examples or {}

    state = ReActAgentState(
        query=query, dataset=dataset, modality=modality,
        confidence_score=confidence_score, ground_truth=ground_truth,
    )
    state.sub_approach = 'react'

    tool_registry    = ToolRegistry(dataset)
    system_prompt    = build_system_prompt(
        dataset, modality, tool_registry, state.sub_approach, few_shot_examples
    )
    scratchpad       = Scratchpad(query=query, system_prompt=system_prompt)
    state.scratchpad = scratchpad

    tracer = EpisodeTracer(state.episode_id)
    tracer.log_start(state)

    consecutive_parse_failures = 0                                  # [FIX F9]

    for step in range(1, MAX_STEPS + 1):
        state.current_step = step

    

        # Context window guard
        if scratchpad.needs_truncation():
            scratchpad       = truncate_scratchpad(scratchpad)
            state.scratchpad = scratchpad                           # [FIX F10]

        raw_text, tokens, latency = call_llm(system_prompt, scratchpad.to_prompt_string())
        thought, action_name, action_input, is_finish = parse_llm_output(raw_text, step)

        # ── FINISH ────────────────────────────────────────────────────────
        if is_finish:
            answer      = extract_final_answer(raw_text)
            finish_step = ReActStep(
                step_index=step, step_type=StepType.FINISH,
                content=answer or raw_text,
                token_count=tokens, latency_ms=latency,
            )
            scratchpad.append(finish_step)
            state.scratchpad = scratchpad
            tracer.log_step(finish_step)
            state.finished     = True
            state.final_answer = answer
            break

        # ── THOUGHT ───────────────────────────────────────────────────────
        if thought:
            t_step = ReActStep(
                step_index=step, step_type=StepType.THOUGHT,
                content=thought, token_count=tokens, latency_ms=latency,
            )
            scratchpad.append(t_step)
            state.scratchpad = scratchpad
            tracer.log_step(t_step)

        # ── ACTION + OBSERVATION ──────────────────────────────────────────
        if action_name and action_input:
            consecutive_parse_failures = 0                          # [FIX F9]

            a_step = ReActStep(
                step_index=step, step_type=StepType.ACTION,
                content=raw_text, tool_name=action_name,
                tool_input=action_input,
                token_count=rough_token_count(action_input),        # [FIX F2]
            )
            scratchpad.append(a_step)
            state.scratchpad = scratchpad
            tracer.log_step(a_step)

            tool = tool_registry.get(action_name)
            if tool is None:
                obs_content = (
                    f"ERROR: Tool '{action_name}' not found. "
                    f"Available: {tool_registry.names()}"
                )
            else:
                result: ToolResult = tool.run(action_input)
                obs_content = (
                    result.output if result.success
                    else f"TOOL ERROR: {result.error_message}"
                )

            o_step = ReActStep(
                step_index=step, step_type=StepType.OBSERVATION,
                content=str(obs_content),
                tool_name=action_name, tool_output=obs_content,
                token_count=rough_token_count(str(obs_content)),    # [FIX F2]
            )
            scratchpad.append(o_step)
            state.scratchpad = scratchpad
            tracer.log_step(o_step)

        else:
            # Parse failure — nudge once, escalate on second failure  [FIX F9]
            consecutive_parse_failures += 1
            logger.warning(f"Step {step}: parse failure #{consecutive_parse_failures}")

            if consecutive_parse_failures >= 2:
                state.escalated         = True
                state.escalation_reason = EscalationReason.PARSE_ERROR
                tracer.log_escalation(state)
                break

            nudge = ReActStep(
                step_index=step, step_type=StepType.OBSERVATION,
                content=(
                    "SYSTEM: Your last response did not contain a valid "
                    "Action/Action Input. Output a Thought followed by "
                    "exactly one Action and Action Input."
                ),
                token_count=rough_token_count("SYSTEM: nudge"),
            )
            scratchpad.append(nudge)
            state.scratchpad = scratchpad
            tracer.log_step(nudge)

    # Hard limit
    if not state.finished and not state.escalated:
        state.escalated         = True
        state.escalation_reason = EscalationReason.MAX_STEPS_EXCEEDED
        tracer.log_escalation(state)

    state.total_latency_ms = (time.time() - episode_start) * 1000

    trace = EpisodeTrace(
        episode_id        =state.episode_id,
        query             =query,
        dataset           =dataset,
        ground_truth      =ground_truth,
        final_answer      =state.final_answer,
        sub_approach      ='react',
        steps             =scratchpad.steps,
        n_steps           =state.current_step,
        finished          =state.finished,
        escalated         =state.escalated,
        escalation_reason =state.escalation_reason.value if state.escalation_reason else None,
        correct           =None,
        total_tokens      =scratchpad.total_tokens,
        total_latency_ms  =state.total_latency_ms,
        total_cost_usd    =estimate_cost(scratchpad.total_tokens, MODEL_NAME),
        model             =MODEL_NAME,
        timestamp         =datetime.now(timezone.utc).isoformat(),  # [FIX F8]
    )

    # Wire evaluator when ground truth is available                  [FIX F7]
    if ground_truth is not None:
        trace.correct = ReActEvaluator().evaluate(trace)

    tracer.log_end(trace)
    return trace


# ── Evaluator ─────────────────────────────────────────────────────────────────

class ReActEvaluator:                                               # [FIX F7]
    def evaluate(self, trace: EpisodeTrace) -> bool:
        if trace.final_answer is None or trace.ground_truth is None:
            return False
        if trace.dataset == "MathHard_L5":
            return self._math_match(trace.final_answer, trace.ground_truth)
        return self._exact_match(trace.final_answer, trace.ground_truth)

    def _exact_match(self, pred: str, gold: str) -> bool:
        return pred.strip().lower() == gold.strip().lower()

    def _math_match(self, pred: str, gold: str) -> bool:
        try:
            return abs(float(pred) - float(gold)) < 1e-6
        except ValueError:
            return self._exact_match(pred, gold)


# ── Tracer ────────────────────────────────────────────────────────────────────

class EpisodeTracer:
    def __init__(self, episode_id: str):
        self.episode_id = episode_id
        self.trace_path = os.path.join(TRACE_DIR, f"{episode_id}.jsonl")
        os.makedirs(TRACE_DIR, exist_ok=True)

    def _write(self, record: dict) -> None:
        try:
            with open(self.trace_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError as e:
            logger.error(f"Trace write failed: {e}")

    def log_start(self, state: ReActAgentState) -> None:
        self._write({
            "event":            "episode_start",
            "episode_id":       state.episode_id,
            "query":            state.query,
            "dataset":          state.dataset,
            "confidence_score": state.confidence_score,
            "sub_approach":     "react",
            "model":            MODEL_NAME,
        })
        logger.info(
            f"[START] {state.episode_id} | dataset={state.dataset} "
            f"| approach={state.sub_approach}"
        )

    def log_step(self, step: ReActStep) -> None:
        self._write({
            "event":       "step",
            "step_index":  step.step_index,
            "step_type":   step.step_type.value
            "content":     step.content[:500],
            "tool_name":   step.tool_name,
            "token_count": step.token_count,
            "latency_ms":  round(step.latency_ms, 2),
            "timestamp":   step.timestamp,
        })
        logger.debug(
            f"  [{step.step_type.value}] step={step.step_index} | tool={step.tool_name}"
        )

    def log_escalation(self, state: ReActAgentState) -> None:
        reason = state.escalation_reason.value if state.escalation_reason else "unknown"
        self._write({
            "event":      "escalation",
            "episode_id": state.episode_id,
            "reason":     reason,
            "step":       state.current_step,
        })
        logger.warning(
            f"[ESCALATE] {state.episode_id} | reason={reason} | step={state.current_step}"
        )

    def log_end(self, trace: EpisodeTrace) -> None:
        self._write({
            "event":            "episode_end",
            "episode_id":       trace.episode_id,
            "finished":         trace.finished,
            "escalated":        trace.escalated,
            "correct":          trace.correct,
            "n_steps":          trace.n_steps,
            "total_tokens":     trace.total_tokens,
            "total_latency_ms": round(trace.total_latency_ms, 2),
            "total_cost_usd":   trace.total_cost_usd,
            "final_answer":     trace.final_answer,
        })
        logger.info(
            f"[END] {trace.episode_id} | steps={trace.n_steps} | "
            f"finished={trace.finished} | correct={trace.correct} | "
            f"tokens={trace.total_tokens} | cost=${trace.total_cost_usd}"
        )

    def log_to_wandb(self, trace: EpisodeTrace) -> None:
        try:
            import wandb
            wandb.log({
                "n_steps":         trace.n_steps,
                "total_tokens":    trace.total_tokens,
                "total_cost_usd":  trace.total_cost_usd,
                "total_latency_ms":trace.total_latency_ms,
                "finished":        int(trace.finished),
                "escalated":       int(trace.escalated),
                "sub_approach":    trace.sub_approach,
                "dataset":         trace.dataset,
                "correct":         int(trace.correct) if trace.correct is not None else -1,
            })
        except ImportError:
            logger.warning("wandb not installed — skipping")


# ── Aggregate traces ──────────────────────────────────────────────────────────

def aggregate_traces(trace_dir: str) -> dict:
    import glob
    import statistics

    episodes = []
    for path in glob.glob(os.path.join(trace_dir, "*.jsonl")):
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("event") == "episode_end":
                    episodes.append(rec)
                    break  # one end-record per file

    if not episodes:
        return {}

    return {
        "n_episodes":      len(episodes),
        "finish_rate":     sum(e["finished"]  for e in episodes) / len(episodes),
        "escalation_rate": sum(e["escalated"] for e in episodes) / len(episodes),
        "accuracy":        sum(
            1 for e in episodes if e.get("correct") is True
        ) / len(episodes),
        "avg_steps":       statistics.mean(e["n_steps"]          for e in episodes),
        "avg_tokens":      statistics.mean(e["total_tokens"]      for e in episodes),
        "avg_cost_usd":    statistics.mean(e["total_cost_usd"]    for e in episodes),
        "avg_latency_ms":  statistics.mean(e["total_latency_ms"]  for e in episodes),
    }


# ── Smoke tests ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # 1. Calculator (no API key needed)
    print("=" * 60)
    print("TESTING: CalculatorTool")
    print("=" * 60)
    calc = CalculatorTool()
    r    = calc.run("(3 + 4) * 2 / 7")
    assert r.success, f"FAIL: {r.error_message}"
    assert abs(float(r.output) - 2.0) < 1e-6, f"FAIL: wrong result {r.output}"
    print(f"output: {r.output} — PASSED ✅\n")

    # 2. CalculatorTool unsafe input guard
    r = calc.run("__import__('os').system('echo pwned')")
    assert not r.success, "FAIL: should reject non-arithmetic input"
    print(f"unsafe input rejected — PASSED ✅\n")

    # 3. FileReaderTool — missing file
    print("=" * 60)
    print("TESTING: FileReaderTool (missing file)")
    print("=" * 60)
    ft = FileReaderTool()
    r  = ft.run("this_does_not_exist.pdf")
    assert not r.success and "not found" in r.error_message
    print(f"error: {r.error_message} — PASSED ✅\n")

    # 4. WebSearchTool
    print("=" * 60)
    print("TESTING: WebSearchTool")
    print("=" * 60)
    st = WebSearchTool()
    r  = st.run("capital of France")
    assert r.success and "Paris" in r.output, f"FAIL: {r.output}"
    print(f"output[:120]: {r.output[:120]} — PASSED ✅\n")

    # 5. Full episode
    print("=" * 60)
    print("TESTING: Full ReAct Episode")
    print("=" * 60)
    trace = run_react_episode(
        query            ="""According to github, when was Regression added to the oldest closed numpy.polynomial issue that has the Regression label in MM/DD/YY?""",
        dataset          ="GAIA",
        modality         ="text",
        confidence_score =0.7,

    )
    print(f"finished  : {trace.finished}")
    print(f"answer    : {trace.final_answer}")
    print(f"correct   : {trace.correct}")
    print(f"steps     : {trace.n_steps}")
    print(f"tokens    : {trace.total_tokens}")
    print(f"cost      : ${trace.total_cost_usd}")
    assert trace.finished or trace.escalated
    print("Episode — PASSED ✅\n")

    print("=" * 60)
    print("ALL TESTS PASSED ✅")
    print("=" * 60)