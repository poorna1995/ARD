"""
ReAct Agent — Reasoning + Acting framework over an LLM backend (Groq).

Structure
---------
  AgentConfig   — all tuneable knobs in one place
  Tool          — schema + callable unified
  ToolRegistry  — central tool store
  ReActParser   — Thought / Action / Final Answer parsing
  RunTracer     — records every step to a JSON file
  ReActAgent    — main reasoning loop
  build_agent() — factory / dependency wiring
  __main__      — CLI entry point
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# stdlib
# ---------------------------------------------------------------------------
import json
import httpx
import logging
import os
import re
import textwrap
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

# ---------------------------------------------------------------------------
# third-party
# ---------------------------------------------------------------------------
import pdfplumber
import serpapi
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

# ---------------------------------------------------------------------------
# Logging — configure once at module level
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("react_agent")


# ===========================================================================
# ── 1. CONFIG ───────────────────────────────────────────────────────────────
# ===========================================================================

@dataclass(frozen=True)
class AgentConfig:
    """All tuneable knobs in one place — no magic literals scattered around."""

    model: str = "llama-3.3-70b-versatile"
    max_steps: int = 10
    temperature: float = 0.0
    retry_attempts: int = 3
    retry_base_delay: float = 1.0       # seconds; doubles on each retry
    serp_location: str = "Austin, Texas, United States"
    serp_hl: str = "en"
    serp_gl: str = "us"
    serp_domain: str = "google.com"
    # ── tracing ──────────────────────────────────────────────────────────────
    trace_file: str = "runs.json"       # path where JSON runs are appended
    trace_enabled: bool = True          # set False to disable tracing entirely


# ===========================================================================
# ── 2. EXCEPTIONS ───────────────────────────────────────────────────────────
# ===========================================================================

class ReActError(Exception):
    """Base exception for all agent errors."""


class ToolNotFoundError(ReActError):
    """Raised when an LLM requests a tool that is not registered."""


class ToolExecutionError(ReActError):
    """Raised when a registered tool raises an unhandled exception."""


class MaxStepsExceededError(ReActError):
    """Raised when the agent exhausts its step budget without a final answer."""


class ParseError(ReActError):
    """Raised when the LLM output cannot be parsed into the ReAct format."""


# ===========================================================================
# ── 3. RETRY DECORATOR ──────────────────────────────────────────────────────
# ===========================================================================

def with_retry(attempts: int = 3, base_delay: float = 1.0):
    """Exponential-backoff retry for transient API/network errors."""

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(1, attempts + 1):
                try:
                    return fn(*args, **kwargs)
                except (ReActError, KeyboardInterrupt):
                    raise
                except Exception as exc:  # noqa: BLE001
                    last_exc = exc
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Attempt %d/%d failed for %s: %s — retrying in %.1fs",
                        attempt, attempts, fn.__name__, exc, delay,
                    )
                    time.sleep(delay)
            raise ReActError(
                f"{fn.__name__} failed after {attempts} attempts"
            ) from last_exc

        return wrapper
    return decorator


# ===========================================================================
# ── 4. TOOL LAYER ───────────────────────────────────────────────────────────
# ===========================================================================

@dataclass
class Tool:

    name: str
    description: str
    input_description: str
    returns: str
    fn: Callable[[str], str]

    def __call__(self, tool_input: str) -> str:
        return self.fn(tool_input)

    def prompt_description(self) -> str:
        """Formatted block used inside the system prompt."""
        return (
            f"  • {self.name}: {self.description}\n"
            f"    Input : {self.input_description}\n"
            f"    Output: {self.returns}"
        )


class ToolRegistry:
    """Centralised registry; add tools once, look them up by name."""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> "ToolRegistry":
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool
        logger.debug("Registered tool: %s", tool.name)
        return self  # fluent API

    def get(self, name: str) -> Tool:
        try:
            return self._tools[name]
        except KeyError:
            raise ToolNotFoundError(
                f"Tool '{name}' not found. Available: {self.names}"
            )

    @property
    def names(self) -> list[str]:
        return list(self._tools.keys())

    def system_prompt_block(self) -> str:
        """Render all tools into a formatted block for the system prompt."""
        return "\n\n".join(t.prompt_description() for t in self._tools.values())


# ---------------------------------------------------------------------------
# Built-in tool implementations
# ---------------------------------------------------------------------------

def _build_search_tool(config: AgentConfig, client: serpapi.Client) -> Tool:

    @with_retry(attempts=config.retry_attempts, base_delay=config.retry_base_delay)
    def search(query: str) -> str:
        logger.info("search(%r)", query)
        results = client.search({
            "q": query,
            "location": config.serp_location,
            "hl": config.serp_hl,
            "gl": config.serp_gl,
            "google_domain": config.serp_domain,
        })
        organic = results.get("organic_results", [])
        if not organic:
            return "No results found."

        snippets = []
        for item in organic[:3]:
            title   = item.get("title", "")
            snippet = item.get("snippet", "")
            link    = item.get("link", "")
            snippets.append(f"- {title}\n  {snippet}\n  {link}")

        return "\n".join(snippets)

    return Tool(
        name="search",
        description="Search the web and return the top relevant results.",
        input_description="A natural-language search query string.",
        returns="Top 3 organic results.",
        fn=search,
    )



import wikipedia

def _build_wikipedia_tool(config: AgentConfig) -> Tool:

    @with_retry(attempts=config.retry_attempts, base_delay=config.retry_base_delay)
    def wikipedia_search(query: str) -> str:
        logger.info("wikipedia(%r)", query)

        try:
            page = wikipedia.page(query, auto_suggest=True)
            return (
                f"**{page.title}**\n\n"
                f"{page.summary}\n"
                f"Source: {page.url}"
            )
        except wikipedia.DisambiguationError as e:
            # Retry with the first suggested option
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return (
                f"**{page.title}**\n\n"
                f"{page.summary}\n"
                f"Source: {page.url}"
            )
        except wikipedia.PageError:
            return "No Wikipedia results found."

    return Tool(
        name="wikipedia",
        description="Look up encyclopedic facts from Wikipedia.",
        input_description="A topic name or natural-language query.",
        returns="Wikipedia summary with title and source URL.",
        fn=wikipedia_search,
    )

# def _build_wikipedia_tool(config: AgentConfig) -> Tool:

#     @with_retry(attempts=config.retry_attempts, base_delay=config.retry_base_delay)
#     def wikipedia(query: str) -> str:
#         logger.info("wikipedia(%r)", query)

#         # Step 1: Search for the best matching page title
#         search_url = "https://en.wikipedia.org/w/rest.php/v1/search/page"
#         search_resp = httpx.get(search_url, params={"q": query, "limit": 1})
#         search_resp.raise_for_status()
#         pages = search_resp.json().get("pages", [])
#         if not pages:
#             return "No Wikipedia results found."

#         # Step 2: Fetch the summary of the top result
#         title = pages[0]["title"]
#         summary_url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{httpx.utils.quote(title)}"
#         summary_resp = httpx.get(summary_url)
#         summary_resp.raise_for_status()
#         data = summary_resp.json()

#         return (
#             f"**{data.get('title', '')}**\n"
#             f"{data.get('description', '')}\n\n"
#             f"{data.get('extract', '')}\n"
#             f"Source: {data.get('content_urls', {}).get('desktop', {}).get('page', '')}"
#         )

#     return Tool(
#         name="wikipedia",
#         description="Look up encyclopedic facts from Wikipedia.",
#         input_description="A topic name or natural-language query.",
#         returns="Wikipedia summary with title, description, and extract.",
#         fn=wikipedia,
#     )

def _read_pdf(path: str) -> str:
    import pdfplumber
    with pdfplumber.open(path) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    return text or "PDF contained no extractable text."
 
 
def _read_docx(path: str) -> str:
    from docx import Document
    doc = Document(path)
    lines: list[str] = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    for table in doc.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                lines.append(" | ".join(cells))
    return "\n".join(lines) or "Document contained no extractable text."
 
 
def _read_xlsx(path: str) -> str:
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    sections: list[str] = []
    for name in wb.sheetnames:
        rows = []
        for row in wb[name].iter_rows(values_only=True):
            if any(c is not None for c in row):
                rows.append("\t".join("" if c is None else str(c) for c in row))
        if rows:
            sections.append(f"=== Sheet: {name} ===\n" + "\n".join(rows))
    wb.close()
    return "\n\n".join(sections) or "Spreadsheet contained no data."
 
 
def _read_csv(path: str) -> str:
    sep = "\t" if path.endswith(".tsv") else ","
    rows: list[str] = []
    with open(path, newline="", encoding="utf-8-sig") as fh:
        for row in csv.reader(fh, delimiter=sep):
            rows.append(sep.join(row))
    return "\n".join(rows) or "CSV contained no data."
 
 
def _read_pptx(path: str) -> str:
    from pptx import Presentation
    prs = Presentation(path)
    slides: list[str] = []
    for i, slide in enumerate(prs.slides, start=1):
        lines = []
        for shape in slide.shapes:
            if shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    line = " ".join(r.text for r in para.runs).strip()
                    if line:
                        lines.append(line)
        if lines:
            slides.append(f"--- Slide {i} ---\n" + "\n".join(lines))
    return "\n\n".join(slides) or "Presentation contained no extractable text."
 
 
def _read_image_ocr(path: str) -> str:
    import pytesseract
    from PIL import Image
    text = pytesseract.image_to_string(Image.open(path)).strip()
    return text or "Image contained no recognisable text (OCR found nothing)."
 
 
 
def _read_jsonl(path: str) -> str:
    lines: list[str] = []
    with open(path, encoding="utf-8") as fh:
        for raw in fh:
            raw = raw.strip()
            if raw:
                lines.append(json.dumps(json.loads(raw), ensure_ascii=False))
    return "\n".join(lines) or "JSONL file was empty."
 
 
def _read_zip(path: str) -> str:
    with zipfile.ZipFile(path) as zf:
        entries = zf.infolist()
    if not entries:
        return "ZIP archive is empty."
    lines = [f"ZIP archive — {len(entries)} entries:"]
    for e in entries:
        size = f"{e.file_size:,} bytes" if e.file_size else "directory"
        lines.append(f"  {e.filename}  ({size})")
    return "\n".join(lines)
 
 
def _read_text(path: str) -> str:
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read().strip() or "File was empty."
 
_READERS: dict[str, Callable[[str], str]] = {
    ".pdf":    _read_pdf,
    ".docx":   _read_docx,
    ".xlsx":   _read_xlsx,
    ".csv":    _read_csv,
    ".pptx":   _read_pptx,
    ".png":    _read_image_ocr,
    ".jpg":    _read_image_ocr,
    ".jsonl":  _read_jsonl,
    ".zip":    _read_zip,
    # plain text / source code
    ".txt":  _read_text,
    ".py":   _read_text,
}
_SUPPORTED = ", ".join(sorted(_READERS))
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _build_pdf_reader_tool(config: AgentConfig) -> Tool:  # noqa: ARG001
    @with_retry(attempts=config.retry_attempts, base_delay=config.retry_base_delay)
    def file_reader(file_path: str) -> str:
        logger.info("file_reader(%r)", file_path)
        raw_path = Path(file_path).expanduser()

        candidate_paths: list[Path] = []
        if raw_path.is_absolute():
            candidate_paths.append(raw_path)
        else:
            candidate_paths.append(_PROJECT_ROOT / raw_path)
            if raw_path.parent == Path("."):
                candidate_paths.append(_GAIA_VALIDATION_DIR / raw_path.name)
                matched = list(_PROJECT_ROOT.rglob(raw_path.name))
                if len(matched) == 1:
                    candidate_paths.append(matched[0])

        path = next((p.resolve() for p in candidate_paths if p.is_file()), candidate_paths[0].resolve())
        if not path.is_file():
            return (
                f"File not found: {file_path} "
                f"(resolved to: {path})"
            )
        ext = path.suffix.lower()
        reader = _READERS.get(ext)
        if reader is None:
            return (f"Unsupported file type '{ext}'. " f"Supported extensions: {_SUPPORTED}")
        try:
            return reader(str(path))
        except ImportError as exc:
            return (
                f"Missing dependency for '{ext}' files: {exc}. Install the required package and retry."
            )
        except Exception as exc:
            logger.exception("file_reader failed for %r", file_path)
            return f"Failed to read '{path.name}': {exc}"

    return Tool(
        name="file_reader",
        description=(
            "Read and return the text content of a file. "
            f"Supported formats: {_SUPPORTED}. "
            "ZIP files return a content listing. "
            "Images (PNG/JPG/…) are processed with OCR."
        ),
        input_description="Absolute or relative path to the file.",
        returns=(
            "Full text content of the file, or an error message if the file "
            "could not be read."
        ),
        fn=file_reader,
    )

    # @with_retry(attempts=config.retry_attempts, base_delay=config.retry_base_delay)
    # def pdf_reader(file_path: str) -> str:
    #     logger.info("pdf_reader(%r)", file_path)
    #     if not os.path.isfile(file_path):
    #         return f"File not found: {file_path}"
    #     with pdfplumber.open(file_path) as pdf:
    #         pages = [page.extract_text() or "" for page in pdf.pages]
    #     text = "\n".join(pages).strip()
    #     return text if text else "PDF contained no extractable text."

    # return Tool(
    #     name="pdf_reader",
    #     description="Extract and return all text from a PDF file.",
    #     input_description="Absolute or relative path to the PDF file.",
    #     returns="Full text content of the PDF.",
    #     fn=pdf_reader,
    # )


# ===========================================================================
# ── 5. PARSER ───────────────────────────────────────────────────────────────
# ===========================================================================

@dataclass
class ParsedStep:
    """Structured result of one ReAct parse pass."""

    thought: str = ""
    action: str = ""
    action_input: str = ""
    final_answer: str = ""

    @property
    def is_final(self) -> bool:
        return bool(self.final_answer)

    @property
    def has_action(self) -> bool:
        return bool(self.action)

    def validate(self) -> None:
        """Raise ValueError for structurally invalid steps."""
        if self.has_action and not self.action_input:
            raise ValueError(
                f"Action '{self.action}' has no input. "
                "Use Action: tool[input] or Action Input: <input>."
            )
        if not self.thought:
            raise ValueError("Every step must begin with a Thought.")


class ReActParser:
    """
    Parses LLM output in ReAct format.

    Accepts both formats transparently:

        # Bracket (preferred — one line):
        Thought: I need to look this up.
        Action: search[Codex Alimentarius standards]

        # Two-line (legacy):
        Thought: I need to look this up.
        Action: search
        Action Input: Codex Alimentarius standards

        # Terminal:
        Thought: I have enough information.
        Final Answer: 42%
    """

    # ── section anchors ────────────────────────────────────────────────────
    _SECTION_START = re.compile(
        r"(?:Thought|Action Input|Action|Observation|Final Answer)\s*:",
        re.I,
    )

    # ── per-field patterns ─────────────────────────────────────────────────
    # Thought: capture until the next known section header (not end-of-string)
    _THOUGHT_RE = re.compile(
        r"Thought\s*:\s*"
        r"(.*?)"
        r"(?=(?:Action Input|Action|Final Answer|Observation)\s*:|$)",
        re.S | re.I,
    )

    # Bracket style — Action: tool_name[input text here]
    _ACTION_BRACKET_RE = re.compile(
        r"Action\s*:\s*"
        r"(\w[\w_-]*)"          # tool name: word chars, hyphens, underscores
        r"\[([^\]]*)\]",        # [input] — everything inside brackets
        re.I,
    )

    # Two-line style — Action: tool_name  (input on the next "Action Input:" line)
    _ACTION_NAME_RE = re.compile(
        r"Action\s*:\s*"
        r"(\w[\w_-]*)"          # tool name only
        r"\s*$",                # nothing else on this line
        re.I | re.M,
    )
    _ACTION_INPUT_RE = re.compile(
        r"Action Input\s*:\s*"
        r"(.*?)"
        r"(?=Observation\s*:|Action\s*:|Final Answer\s*:|$)",
        re.S | re.I,
    )

    # Final answer
    _FINAL_RE = re.compile(
        r"Final Answer\s*:\s*"
        r"(.*)",
        re.S | re.I,
    )

    # Finish[] — ReAct variant where termination is itself an action
    _FINISH_RE = re.compile(
        r"Action\s*:\s*Finish\[([^\]]*)\]",
        re.I,
    )

    # ── public API ─────────────────────────────────────────────────────────
    @classmethod
    def parse(cls, text: str) -> ParsedStep:
        step = ParsedStep()
        text = text.strip()

        # 1. Thought (always attempted first)
        m = cls._THOUGHT_RE.search(text)
        if m:
            step.thought = m.group(1).strip()

        # 2. Finish[answer]  — termination-as-action variant
        m = cls._FINISH_RE.search(text)
        if m:
            step.final_answer = m.group(1).strip()
            return step

        # 3. Final Answer: <text>
        m = cls._FINAL_RE.search(text)
        if m:
            step.final_answer = m.group(1).strip()
            return step

        # 4a. Bracket style: Action: tool[input]  (single line)
        m = cls._ACTION_BRACKET_RE.search(text)
        if m:
            step.action       = m.group(1).strip()
            step.action_input = m.group(2).strip()
            return step

        # 4b. Two-line style: Action: tool  +  Action Input: ...
        m = cls._ACTION_NAME_RE.search(text)
        if m:
            step.action = m.group(1).strip()
            mi = cls._ACTION_INPUT_RE.search(text)
            if mi:
                step.action_input = mi.group(1).strip()

        return step

# ===========================================================================
# ── 6. RUN TRACER ───────────────────────────────────────────────────────────
# ===========================================================================

@dataclass
class StepRecord:
    step: int                       # no default — must be first
    thought: str      = ""
    action: str       = ""
    action_input: str = ""
    observation: str  = ""
    final_answer: str = ""
    error: str        = ""

@dataclass
class RunRecord:
    run_id: str               = field(default_factory=lambda: __import__("uuid").uuid4().hex[:8])
    timestamp: str            = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    query: str                = ""
    model: str                = ""
    status: str               = "running"
    total_steps: int          = 0
    final_answer: str         = ""
    version: int              = 1
    error_message: str | None = None
    steps: list[StepRecord]   = field(default_factory=list)
 
    def add_step(self, record: StepRecord) -> None:
        self.steps.append(record)
        self.total_steps = len(self.steps)
 
    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["steps"] = [asdict(s) for s in self.steps]
        return d


class RunTracer:
 
    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        self.directory = Path(directory)
        self.enabled   = enabled
 
        if self.directory.exists() and not self.directory.is_dir():
            raise ValueError(
                f"RunTracer: '{self.directory}' already exists as a file. "
                "Delete or rename it before using it as a trace directory."
            )
        self.directory.mkdir(parents=True, exist_ok=True)
 
    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
 
    def _run_path(self, run_id: str) -> Path:
        """Canonical path for a run's .jsonl file."""
        return self.directory / f"{run_id}.jsonl"
 
    def _append_line(self, run_id: str, data: dict[str, Any]) -> None:
        """
        Append one JSON-encoded line to the run file.
 
        A single .write() call on a file opened in append mode is atomic
        on POSIX systems — no partial lines are ever visible to readers.
        """
        with self._run_path(run_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(data, ensure_ascii=False) + "\n")
 
    def _detect_version(self, query: str) -> int:
        """
        Scan existing .jsonl files and count how many prior runs share
        the same *query* string (first line = run_start header).
 
        Returns count + 1, so the first run is version 1.
        """
        count = 0
        for jsonl_file in self.directory.glob("*.jsonl"):
            try:
                first_line = jsonl_file.open(encoding="utf-8").readline()
                if not first_line:
                    continue
                header = json.loads(first_line)
                if header.get("type") == "run_start" and header.get("query") == query:
                    count += 1
            except (json.JSONDecodeError, OSError):
                # Corrupt or empty file — skip silently.
                continue
        return count + 1
 
    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------
 
    def start_run(
        self,
        query: str,
        model: str,
    ) -> RunRecord:
        if not self.enabled:
            return RunRecord(query=query, model=model, pdf_path=pdf_path)
 
        version = self._detect_version(query)
        record  = RunRecord(
            query    = query,
            model    = model,
            version  = version,
        )
 
        self._append_line(record.run_id, {
            "type":      "run_start",
            "run_id":    record.run_id,
            "timestamp": record.timestamp,
            "query":     record.query,
            "model":     record.model,
            "version":   record.version,
        })
 
        logger.info(
            "Tracer: started run %s (v%d) → %s",
            record.run_id, version, self._run_path(record.run_id),
        )
        return record
 
    # ------------------------------------------------------------------
    # Step recorders
    # ------------------------------------------------------------------
 
    def record_action_step(
        self,
        run: RunRecord,
        step_num: int,
        parsed: "ParsedStep",
        observation: str,
    ) -> None:
        """Record one Thought → Action → Observation cycle."""
        if not self.enabled:
            return
 
        run.add_step(StepRecord(
            step         = step_num,
            thought      = parsed.thought,
            action       = parsed.action,
            action_input = parsed.action_input,
            observation  = observation,
        ))
        self._append_line(run.run_id, {
            "type":         "step",
            "step":         step_num,
            "thought":      parsed.thought,
            "action":       parsed.action,
            "action_input": parsed.action_input,
            "observation":  observation,
        })
 
    def record_final_step(
        self,
        run: RunRecord,
        step_num: int,
        parsed: "ParsedStep",
    ) -> None:
        """Record the concluding Thought → Final Answer step."""
        if not self.enabled:
            return
 
        run.add_step(StepRecord(
            step         = step_num,
            thought      = parsed.thought,
            final_answer = parsed.final_answer,
        ))
        self._append_line(run.run_id, {
            "type":         "step",
            "step":         step_num,
            "thought":      parsed.thought,
            "final_answer": parsed.final_answer,
        })
 
    def record_error_step(
        self,
        run: RunRecord,
        step_num: int,
        thought: str,
        error: str,
    ) -> None:
        """Record a step where parsing failed or a tool raised an error."""
        if not self.enabled:
            return
 
        run.add_step(StepRecord(step=step_num, thought=thought, error=error))
        self._append_line(run.run_id, {
            "type":    "step",
            "step":    step_num,
            "thought": thought,
            "error":   error,
        })
 
    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------
 
    def save(self, run: RunRecord) -> None:
        """
        Write the closing ``run_end`` line.
 
        Always call this from a ``finally`` block so partial runs are
        never left open — even on ``MaxStepsExceededError`` or any other
        exception the caller propagates.
 
        ::
 
            try:
                answer = agent.run(query)
                run.status = "success"
                run.final_answer = answer
            except MaxStepsExceededError:
                run.status = "max_steps_exceeded"
                raise
            except Exception as exc:
                run.status = "error"
                run.error_message = str(exc)
                raise
            finally:
                tracer.save(run)   # ← always runs
        """
        if not self.enabled:
            return
 
        self._append_line(run.run_id, {
            "type":          "run_end",
            "run_id":        run.run_id,
            "status":        run.status,
            "total_steps":   run.total_steps,
            "final_answer":  run.final_answer,
            "error_message": run.error_message,
        })
 
        logger.info(
            "Tracer: run %s v%d (%s) closed — %d step(s) → %s",
            run.run_id, run.version, run.status,
            run.total_steps, self._run_path(run.run_id),
        )





# ===========================================================================
# ── 7. SYSTEM PROMPT ────────────────────────────────────────────────────────
# ===========================================================================
_SYSTEM_PROMPT_TEMPLATE = textwrap.dedent("""\
    You are a ReAct agent. Solve tasks by strictly cycling through Thought → Action → Observation.
    Reason before every action. Never guess or fabricate results.

    ── CYCLE ─────────────────────────────────────────────────────────────────

      Thought: <why you need this action and what you expect to learn>
      Action: <tool_name>[<exact input>]
      Observation: <result returned by the environment — never written by you>

    Repeat the cycle until the answer is certain, then close with:

      Thought: <why you can answer now>
      Action: Finish[<concise, direct answer to the original question>]

    ── TOOLS ─────────────────────────────────────────────────────────────────
    {tools_block}
    Finish[<answer>]  — terminates the loop and returns the final answer.

    ── RULES ─────────────────────────────────────────────────────────────────
    - Every response MUST start with "Thought:"
    - Emit exactly ONE Action per turn, then stop and wait for Observation
    - NEVER write the Observation yourself — it is injected by the environment
    - NEVER invent, assume, or carry over tool results
    - Re-read the latest Observation before writing the next Thought
    - Call a tool only when the answer cannot be determined from prior Observations
    - If a tool fails or returns unhelpful output, revise your strategy — do not repeat the same call
    - Finish[] is the only valid exit — use it once you can answer fully and precisely
    - Output nothing outside the required format
""")


def build_system_prompt(registry: ToolRegistry) -> str:
    return _SYSTEM_PROMPT_TEMPLATE.format(tools_block=registry.system_prompt_block())


# ===========================================================================
# ── 8. AGENT ────────────────────────────────────────────────────────────────
# ===========================================================================

class ReActAgent:
    def __init__(
        self,
        registry: ToolRegistry,
        groq_client: Groq,
        config: AgentConfig | None = None,
        tracer: RunTracer | None = None,
    ) -> None:
        self.registry = registry
        self.client   = groq_client
        self.config   = config or AgentConfig()
        self.parser   = ReActParser()
        self.tracer   = tracer or RunTracer(
            self.config.trace_file,
            enabled=self.config.trace_enabled,
        )
        self._system  = build_system_prompt(registry)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, query: str, pdf_path: str | None = None) -> str:
        run = self.tracer.start_run(
            query    = query,
            model    = self.config.model,
        )

        user_content = query
        if pdf_path:
            user_content += f"\n\nA PDF is available at: {pdf_path}"

        messages: list[dict[str, Any]] = [
            {"role": "user", "content": user_content}
        ]

        logger.info(
            "Agent started | run_id=%s | query=%r | max_steps=%d",
            run.run_id, query, self.config.max_steps,
        )

        try:
            for step in range(1, self.config.max_steps + 1):
                logger.info("── Step %d/%d ──────────────────────", step, self.config.max_steps)

                llm_text = self._call_llm(messages)
                logger.debug("LLM output:\n%s", llm_text)
                messages.append({"role": "assistant", "content": llm_text})

                parsed = self.parser.parse(llm_text)

                if parsed.thought:
                    logger.info("Thought: %s", parsed.thought)

                # ── Pattern B: Final Answer ───────────────────────────────
                if parsed.is_final:
                    self.tracer.record_final_step(run, step, parsed)
                    run.status       = "success"
                    run.final_answer = parsed.final_answer
                    logger.info("Final Answer reached in %d step(s).", step)
                    return parsed.final_answer

                # ── Pattern A: Action → Observation ──────────────────────
                if parsed.has_action:
                    observation = self._execute_tool(parsed.action, parsed.action_input)
                    self.tracer.record_action_step(run, step, parsed, observation)
                    logger.info("Observation: %s", observation[:200])
                    messages.append({"role": "user", "content": f"Observation: {observation}"})
                    continue

                # ── Unparseable: nudge the model ──────────────────────────
                logger.warning("Unparseable LLM output at step %d; nudging.", step)
                self.tracer.record_error_step(run, step, parsed.thought, "Unparseable LLM output")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your last response did not follow the required format. "
                        "Please use exactly one of the two patterns:\n"
                        "  Thought: …\n  Action: …\n  Action Input: …\n"
                        "OR\n"
                        "  Thought: …\n  Final Answer: …"
                    ),
                })

            run.status        = "max_steps_exceeded"
            run.error_message = f"No final answer within {self.config.max_steps} steps."
            raise MaxStepsExceededError(run.error_message)

        except Exception as exc:
            if run.status == "running":
                run.status        = "error"
                run.error_message = str(exc)
            raise

        finally:
            # Always persist — even on failure — so runs are never lost.
            self.tracer.save(run)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _call_llm(self, messages: list[dict[str, Any]]) -> str:
        @with_retry(
            attempts   = self.config.retry_attempts,
            base_delay = self.config.retry_base_delay,
        )
        def _call() -> str:
            response = self.client.chat.completions.create(
                model       = self.config.model,
                messages    = [{"role": "system", "content": self._system}] + messages,
                temperature = self.config.temperature,
            )
            return response.choices[0].message.content

        return _call()

    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        logger.info("Action: %s | Input: %r", tool_name, tool_input)
        try:
            tool = self.registry.get(tool_name)
        except ToolNotFoundError as exc:
            logger.error("%s", exc)
            return str(exc)

        try:
            return tool(tool_input)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Tool '%s' raised an error.", tool_name)
            raise ToolExecutionError(f"Tool '{tool_name}' failed: {exc}") from exc


# ===========================================================================
# ── 9. FACTORY / DEPENDENCY WIRING ──────────────────────────────────────────
# ===========================================================================

def build_agent(
    config: AgentConfig | None = None,
    tracer: RunTracer | None = None,
) -> ReActAgent:
    cfg = config or AgentConfig()

    serp_key = os.getenv("SERP_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")

    if not serp_key:
        raise EnvironmentError("SERP_API_KEY is not set.")
    if not groq_key:
        raise EnvironmentError("GROQ_API_KEY is not set.")

    serp_client = serpapi.Client(api_key=serp_key)
    groq_client = Groq(api_key=groq_key)

    registry = (
        ToolRegistry()
        .register(_build_search_tool(cfg, serp_client))
        .register(_build_wikipedia_tool(cfg))
        .register(_build_pdf_reader_tool(cfg))
    )

    return ReActAgent(
        registry    = registry,
        groq_client = groq_client,
        config      = cfg,
        tracer      = tracer,
    )


# ===========================================================================
# ── 10. CLI ENTRY POINT ─────────────────────────────────────────────────────
# ===========================================================================

if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description="Run the ReAct Agent.")
    cli.add_argument("query",          default="",                         help="The question or task for the agent.")                     
    cli.add_argument("--max-steps",    type=int,    default=10)
    cli.add_argument("--model",        default="llama-3.3-70b-versatile")
    cli.add_argument("--trace-file",   default="runs.json",                 help="JSON file to append run traces to.")
    cli.add_argument("--no-trace",     action="store_true",                 help="Disable JSON tracing.")
    cli.add_argument("--debug",        action="store_true")
    args = cli.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    cfg   = AgentConfig(
        model         = args.model,
        max_steps     = args.max_steps,
        trace_file    = args.trace_file,
        trace_enabled = not args.no_trace,
    )
    agent = build_agent(cfg)

    try:
        answer = agent.run(args.query)
        print("\n" + "=" * 60)
        print("FINAL ANSWER")
        print("=" * 60)
        print(answer)
        if cfg.trace_enabled:
            print(f"\n[trace saved → {cfg.trace_file}]")
    except MaxStepsExceededError as exc:
        logger.error("Agent gave up: %s", exc)
    except ReActError as exc:
        logger.error("Agent error: %s", exc)