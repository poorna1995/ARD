#
# agent/react.py
from __future__ import annotations
import os
import json
import re
import time
from typing import Any, Callable
from agent.base import BaseAgent, AgentResponse
from prompts.prompts import USER_PROMPT, SYSTEM_PROMPT
import serpapi
from dotenv import load_dotenv
load_dotenv()




# ── Tool decorator ─────────────────────────────────────────────────────────────

def tool(name: str, description: str):
    """Decorator to register a function as a ReAct tool."""
    def decorator(fn: Callable) -> Callable:
        fn._tool_name        = name
        fn._tool_description = description
        return fn
    return decorator


# ── Built-in Tools ─────────────────────────────────────────────────────────────

@tool("web_search", "Search the web for current information. Input: search query string.")
def web_search(query: str) -> str:
    try:
        from ddgs import DDGS
        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=3):
                results.append(
                    f"Title   : {r['title']}\n"
                    f"URL     : {r['href']}\n"
                    f"Snippet : {r['body']}"
                )
        return "\n\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


@tool("web_fetch", " Fetch and read the content of a URL once the searched the web. Input: full URL string.")
def web_fetch(url: str) -> str:
    try:
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
        return text[:3000]
    except Exception as e:
        return f"Fetch failed: {e}"


@tool(
    "wikipedia",
    (
        "Look up a topic on Wikipedia and return a concise summary. "
        "Input: the topic or entity name "
        "(e.g. 'Albert Einstein', 'Python programming language')."
    ),
)
def wikipedia_search(query: str) -> str:
    try:
        import wikipediaapi
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="ReActAgent/1.0 (research-bot)",
        )
        page = wiki.page(query)
        if not page.exists():
            return _wikipedia_fallback(query)
        summary = page.summary
        if len(summary) > 1500:
            summary = summary[:1500].rsplit(" ", 1)[0] + " …"
        return f"Wikipedia – {page.title}\nURL: {page.fullurl}\n\n{summary}"
    except ImportError:
        pass
    return _wikipedia_fallback(query)


def _wikipedia_fallback(query: str) -> str:
    try:
        import wikipedia
        wikipedia.set_lang("en")
        try:
            summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
            page    = wikipedia.page(query, auto_suggest=True)
            return f"Wikipedia – {page.title}\nURL: {page.url}\n\n{summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            options = ", ".join(e.options[:6])
            return (
                f"Disambiguation: '{query}' may refer to multiple topics. "
                f"Try being more specific. Suggestions: {options}"
            )
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for '{query}'."
    except ImportError:
        return (
            "Wikipedia tool unavailable. "
            "Install: pip install wikipedia-api  OR  pip install wikipedia"
        )


@tool(
    "read_file",
    (
        "Read the contents of an uploaded file by filename. "
        "Supports: .txt, .csv, .xlsx/.xls, .pdf, .docx, .pptx, "
        ".png/.jpg/.jpeg (OCR), .zip. "
        "Input: filename string."
    ),
)
def read_file(filename: str) -> str:
    filename = filename.strip().strip('"\'')
    _SEARCH_PATHS = [
        f"/Users/poorna/Desktop/research_work/datasets/gaia_files/2023/validation/{filename}",  
        filename,                                         
    ]
    path = next((p for p in _SEARCH_PATHS if os.path.exists(p)), None)
    if path is None:
        return f"File not found: {filename}"

    ext = os.path.splitext(filename)[1].lower()

    try:
        if ext in {".xlsx", ".xls"}:
            import openpyxl
            from openpyxl.styles.colors import COLOR_INDEX

            wb = openpyxl.load_workbook(path)
            parts = []

            for sheet_name in wb.sheetnames:
                ws = wb[sheet_name]
                lines = [f"=== Sheet: {sheet_name} ==="]

                for row in ws.iter_rows():
                    row_data = []
                    for cell in row:
                        # Resolve fill color
                        fill = cell.fill
                        color = None

                        if fill and fill.fgColor:
                            fg = fill.fgColor
                            if fg.type == "rgb":
                                color = fg.rgb          # e.g. "FF92D050" (ARGB)
                            elif fg.type == "indexed":
                                color = f"indexed:{fg.indexed}"
                            elif fg.type == "theme":
                                color = f"theme:{fg.theme}"

                        value = cell.value if cell.value is not None else ""
                        label = _color_to_name(color)   # see helper below
                        row_data.append(f"{cell.coordinate}:{label}")

                    lines.append("  ".join(row_data))

                parts.append("\n".join(lines))

            return "\n\n".join(parts)[:6000]
        if ext == ".csv":
            import pandas as pd
            df = pd.read_csv(path)
            return df.to_string(index=False)[:4000]

        if ext == ".pdf":
            import fitz
            doc   = fitz.open(path)
            pages = []
            for i, page in enumerate(doc):
                text = page.get_text("text").strip()
                if text:
                    pages.append(f"[Page {i + 1}]\n{text}")
                if sum(len(p) for p in pages) > 4000:
                    break
            doc.close()
            return "\n\n".join(pages)[:4000] if pages else "No extractable text in PDF."

        if ext == ".docx":
            from docx import Document
            doc   = Document(path)
            paras = [p.text for p in doc.paragraphs if p.text.strip()]
            for table in doc.tables:
                for row in table.rows:
                    paras.append(" | ".join(c.text.strip() for c in row.cells))
            return "\n".join(paras)[:4000]

        if ext == ".pptx":
            from pptx import Presentation
            prs    = Presentation(path)
            slides = []
            for i, slide in enumerate(prs.slides, start=1):
                texts = []
                for shape in slide.shapes:
                    if shape.has_text_frame:
                        for para in shape.text_frame.paragraphs:
                            line = para.text.strip()
                            if line:
                                texts.append(line)
                if texts:
                    slides.append(f"[Slide {i}]\n" + "\n".join(texts))
            return "\n\n".join(slides)[:4000] if slides else "No text found in presentation."

        if ext in {".png", ".jpg", ".jpeg"}:
            from PIL import Image
            import numpy as np

            img = Image.open(path).convert("RGBA")
            arr = np.array(img)

            # --- Color masks (tune thresholds as needed) ---
            r, g, b = arr[:,:,0], arr[:,:,1], arr[:,:,2]

            red_mask   = (r > 150) & (g < 100) & (b < 100)
            green_mask = (r < 100) & (g > 150) & (b < 100)

            # --- Try to extract numbers via OCR with bounding boxes ---
            try:
                import pytesseract
                data = pytesseract.image_to_data(
                    img.convert("RGB"),
                    output_type=pytesseract.Output.DICT
                )

                red_numbers, green_numbers = [], []

                for i, text in enumerate(data["text"]):
                    text = text.strip()
                    if not text:
                        continue
                    try:
                        num = float(text)
                    except ValueError:
                        continue

                    x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]

                    # Sample the dominant color in the bounding box
                    region = arr[y:y+h, x:x+w]
                    if region.size == 0:
                        continue

                    is_red   = red_mask[y:y+h, x:x+w].mean() > 0.3
                    is_green = green_mask[y:y+h, x:x+w].mean() > 0.3

                    if is_red:
                        red_numbers.append(num)
                    elif is_green:
                        green_numbers.append(num)

                return (
                    f"Red numbers  : {red_numbers}\n"
                    f"Green numbers: {green_numbers}"
                )

            except ImportError:
                return "pytesseract not installed — cannot extract colored numbers."
        # if ext in {".png", ".jpg", ".jpeg"}:
        #     from PIL import Image
        #     try:
        #         import pytesseract
        #         img  = Image.open(path)
        #         text = pytesseract.image_to_string(img).strip()
        #         return text[:4000] if text else "No text detected in image via OCR."
        #     except ImportError:
        #         img = Image.open(path)
        #         return (
        #             f"Image opened (pytesseract not installed — no OCR).\n"
        #             f"Format : {img.format}\n"
        #             f"Size   : {img.size[0]}×{img.size[1]} px\n"
        #             f"Mode   : {img.mode}"
        #         )

        if ext == ".zip":
            import zipfile
            with zipfile.ZipFile(path, "r") as zf:
                names = zf.namelist()
                lines = [f"ZIP archive – {len(names)} file(s):"]
                for name in names[:30]:
                    info = zf.getinfo(name)
                    lines.append(f"  {name:<45s}  {info.file_size:>10,} bytes")
                if len(names) > 30:
                    lines.append(f"  … and {len(names) - 30} more file(s).")
                extracted = []
                for name in names:
                    inner_ext = os.path.splitext(name)[1].lower()
                    info      = zf.getinfo(name)
                    if inner_ext in {".txt", ".csv", ".json", ".md", ".py"} and info.file_size < 50_000:
                        try:
                            content = zf.read(name).decode("utf-8", errors="ignore")
                            extracted.append(f"\n── {name} ──\n{content[:1000]}")
                        except Exception:
                            pass
                result = "\n".join(lines)
                if extracted:
                    result += "\n\nExtracted text files:" + "".join(extracted)
                return result[:4000]

        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()[:4000]

    except ImportError as e:
        return f"Missing library to read '{ext}' files: {e}"
    except Exception as e:
        return f"Failed to read file '{filename}': {e}"


@tool("finish", "Submit the final answer. Input: JSON string {\"answer\": \"<value>\"}.")
def finish(answer: str) -> str:
    return answer


# ── Default Tool Registry ──────────────────────────────────────────────────────

DEFAULT_TOOLS: dict[str, Callable] = {
    fn._tool_name: fn
    for fn in [web_search, web_fetch, wikipedia_search, read_file, finish]
}


# ── ReAct Step dataclass ───────────────────────────────────────────────────────

class ReActStep:
    def __init__(
        self,
        step_num:     int,
        thought:      str,
        action:       str,
        action_input: str,
        observation:  str = "",
    ):
        self.step_num     = step_num
        self.thought      = thought
        self.action       = action
        self.action_input = action_input
        self.observation  = observation

    def __repr__(self) -> str:
        return (
            f"[Step {self.step_num}]\n"
            f"  Thought     : {self.thought}\n"
            f"  Action      : {self.action}[{self.action_input}]\n"
            f"  Observation : {self.observation}\n"
        )


# ── ReAct Parser ───────────────────────────────────────────────────────────────

class ReActParser:
    """
    Parses LLM output into (thought, action_name, action_input).

    Enforces ONE canonical format:
        Thought: <text>
        Action: tool_name[input]

    Falls back gracefully to "finish" if no action is detected.
    """

    # Capture everything after "Thought:" up to the next "Action:" label
    THOUGHT_RE = re.compile(
        r"Thought\s*:\s*(.+?)(?=\nAction\s*:|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # Primary format — Action: tool_name[input]
    # Allows multi-line input inside the brackets
    ACTION_BRACKET_RE = re.compile(
        r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\[([^\]]*)\]",
        re.DOTALL | re.IGNORECASE,
    )

    # Fallback format — Action: tool_name\nAction Input: input
    ACTION_SPLIT_RE = re.compile(
        r"Action\s*:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\n+\s*(?:Action\s+)?Input\s*:\s*(.+?)(?=\nThought|\nObservation|\Z)",
        re.DOTALL | re.IGNORECASE,
    )

    # Bare Finish[...] shortcut
    FINISH_RE = re.compile(r"\bFinish\s*\[([^\]]*)\]", re.DOTALL | re.IGNORECASE)

    # JSON answer extraction  {"answer": "..."}
    JSON_ANSWER_RE = re.compile(r'\{[^}]*"answer"\s*:\s*"([^"]+)"[^}]*\}', re.DOTALL)

    @classmethod
    def parse(cls, text: str) -> tuple[str, str, str]:
        """
        Returns (thought, action_name, action_input).
        action_name is always lowercase and stripped.
        """
        # ── Thought ───────────────────────────────────────────────────────
        thought = ""
        t_match = cls.THOUGHT_RE.search(text)
        if t_match:
            thought = t_match.group(1).strip()

        # ── Primary: Action: tool[input] ──────────────────────────────────
        a_match = cls.ACTION_BRACKET_RE.search(text)
        if a_match:
            action_name  = a_match.group(1).strip().lower()
            action_input = a_match.group(2).strip()
            return thought, action_name, action_input

        # ── Secondary: Action: tool\nAction Input: input ──────────────────
        s_match = cls.ACTION_SPLIT_RE.search(text)
        if s_match:
            action_name  = s_match.group(1).strip().lower()
            action_input = s_match.group(2).strip()
            return thought, action_name, action_input

        # ── Bare Finish[...] ──────────────────────────────────────────────
        f_match = cls.FINISH_RE.search(text)
        if f_match:
            return thought, "finish", f_match.group(1).strip()

        # ── Nothing matched — treat entire output as final answer ─────────
        return thought, "finish", text.strip()

    @classmethod
    def extract_final_answer(cls, raw: str) -> str:
        """
        Extract the clean answer string from finish input.
        Handles:
          • {"answer": "Paris"}
          • Paris
        """
        j_match = cls.JSON_ANSWER_RE.search(raw)
        if j_match:
            return j_match.group(1).strip()

        # Try parsing as JSON directly
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict) and "answer" in obj:
                return str(obj["answer"]).strip()
        except (json.JSONDecodeError, ValueError):
            pass

        return raw.strip()


# ── ReAct Agent ────────────────────────────────────────────────────────────────

class ReactAgent(BaseAgent):
    """
    ReAct agent that interleaves Thought / Action / Observation.

    Key design decisions
    ─────────────────────
    • The scratchpad is sent as *alternating* assistant/user messages
      so the LLM sees the correct conversational structure.
    • Tool names are normalised to lowercase at registration and at
      dispatch time, preventing case-mismatch misses.
    • The parser enforces ONE canonical format:  Action: name[input]
    """

    def __init__(
        self,
        model:     str,
        dataset:   str,
        tools:     dict[str, Callable] | None = None,
        max_steps: int = 10,
        **kwargs,
    ):
        self.dataset   = dataset
        self.max_steps = max_steps
        # Normalise all tool keys to lowercase at registration time
        raw_tools  = tools or DEFAULT_TOOLS
        self.tools = {k.lower(): v for k, v in raw_tools.items()}
        self.parser = ReActParser()

        tools_block = self._build_tools_block()

        super().__init__(
            model         = model,
            system_prompt = SYSTEM_PROMPT[dataset]["react"].format(tools_block=tools_block),
            user_prompt   = USER_PROMPT[dataset]["react"],
            **kwargs,
        )

    # ── Helpers ────────────────────────────────────────────────────────────

    def _build_tools_block(self) -> str:
        lines = []
        for name, fn in self.tools.items():
            desc = getattr(fn, "_tool_description", "No description.")
            lines.append(f"  {name:15s}: {desc}")
        return "\n".join(lines)

    def _call_tool(self, action: str, action_input: str) -> tuple[str, float]:
        """
        Dispatch to the registered tool function.

        Returns (observation_string, wall_clock_seconds).
        """
        start = time.perf_counter()
        normalised = action.strip().lower()
        fn = self.tools.get(normalised)

        if fn is None:
            available = ", ".join(self.tools.keys())
            obs = (
                f"Error: unknown tool '{action}'. "
                f"Available tools: {available}. "
                f"Fix the Action name and try again."
            )
        else:
            try:
                obs = fn(action_input)
            except Exception as e:
                obs = f"Tool '{action}' raised an error: {e}. Try a different approach."

        return str(obs), time.perf_counter() - start

    # ── Conversation builder ───────────────────────────────────────────────

    def _build_messages(self, query: str, steps: list[ReActStep]) -> list[dict]:
        """
        Build a proper alternating message list for the LLM:

            system  : instructions + tool descriptions
            user    : original query
            assistant: Thought + Action   (step 1)
            user    : Observation          (step 1)
            assistant: Thought + Action   (step 2)
            user    : Observation          (step 2)
            …

        This is far superior to concatenating everything into one user
        message because the LLM understands role boundaries correctly.
        """
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": self.user_prompt.format(query=query)},
        ]

        for step in steps:
            # What the assistant said
            assistant_turn = (
                f"Thought: {step.thought}\n"
                f"Action: {step.action}[{step.action_input}]"
            )
            messages.append({"role": "assistant", "content": assistant_turn})

            # What the environment replied
            messages.append({
                "role":    "user",
                "content": f"Observation: {step.observation}",
            })

        return messages

    # ── LLM call ──────────────────────────────────────────────────────────

    def _call_llm(
        self,
        query: str,
        steps: list[ReActStep],
    ) -> tuple[str, float, Any]:
        messages = self._build_messages(query, steps)
        start    = time.perf_counter()
        response = self._get_client().chat.completions.create(
            model       = self.model,
            messages    = messages,
            temperature = self.temperature,
            max_tokens  = self.max_tokens,
        )
        latency = time.perf_counter() - start
        return response.choices[0].message.content, latency, response

    # ── Main loop ──────────────────────────────────────────────────────────

    def run(self, query: str, dataset: str = "", **kwargs) -> AgentResponse:
        steps:                list[ReActStep] = []
        tools_called:         list[str]       = []
        tools_results:        list[dict]      = []
        total_latency_llm       = 0.0
        total_latency_tool      = 0.0
        total_prompt_tokens     = 0
        total_completion_tokens = 0
        final_answer            = ""
        is_stopped_early        = False
        error                   = None
        is_failed               = False

        start_total = time.perf_counter()

        for step_num in range(1, self.max_steps + 1):

            # ── 1. Call the LLM ───────────────────────────────────────────
            try:
                llm_out, llm_latency, response = self._call_llm(query, steps)
            except Exception as e:
                error     = str(e)
                is_failed = True
                break

            total_latency_llm       += llm_latency
            total_prompt_tokens     += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens

            # ── 2. Parse Thought / Action / Input ─────────────────────────
            thought, action, action_input = self.parser.parse(llm_out)

            # ── 3. Finish early? ──────────────────────────────────────────
            if action == "finish":
                # Run the finish tool so the answer goes through the same path
                observation, tool_latency = self._call_tool("finish", action_input)
                total_latency_tool += tool_latency
                tools_called.append("finish")
                tools_results.append({
                    "step":        step_num,
                    "action":      "finish",
                    "input":       action_input,
                    "observation": observation,
                })
                steps.append(ReActStep(
                    step_num=step_num, thought=thought,
                    action="finish", action_input=action_input,
                    observation=observation,
                ))
                final_answer = self.parser.extract_final_answer(action_input)
                break

            # ── 4. Guard: don't call a tool if action is unrecognised ─────
            #    (the tool dispatcher will return an error observation, which
            #     the LLM will see and correct on the next step)

            # ── 5. Execute Tool ───────────────────────────────────────────
            observation, tool_latency = self._call_tool(action, action_input)
            total_latency_tool += tool_latency

            tools_called.append(action)
            tools_results.append({
                "step":        step_num,
                "action":      action,
                "input":       action_input,
                "observation": observation,
            })

            steps.append(ReActStep(
                step_num=step_num, thought=thought,
                action=action, action_input=action_input,
                observation=observation,
            ))

        else:
            # max_steps exhausted without a Finish action
            is_stopped_early = True
            final_answer = (
                self.parser.extract_final_answer(steps[-1].observation)
                if steps else ""
            )

        total_latency = time.perf_counter() - start_total
        expected      = kwargs.get("expected_answer")

        return AgentResponse(
            # Core
            query             = query,
            answer            = final_answer,
            model             = self.model,
            agent             = "react",
            dataset           = dataset or self.dataset,
            # Latency
            latency_total     = total_latency,
            latency_llm       = total_latency_llm,
            latency_tools     = total_latency_tool,
            # Tokens + Cost
            prompt_tokens     = total_prompt_tokens,
            completion_tokens = total_completion_tokens,
            total_tokens      = total_prompt_tokens + total_completion_tokens,
            cost_usd          = self._compute_cost(
                                    total_prompt_tokens,
                                    total_completion_tokens,
                                ),
            # Reasoning
            reasoning_steps   = [f"[{s.step_num}] {s.thought}" for s in steps],
            num_llm_calls     = len(steps),
            num_steps         = len(steps),
            # Tools
            tools_available   = list(self.tools.keys()),
            tools_called      = tools_called,
            tools_results     = tools_results,
            num_tool_calls    = len(tools_called),
            # ReAct specific
            max_steps         = self.max_steps,
            steps_taken       = len(steps),
            is_stopped_early  = is_stopped_early,
            # Evaluation
            expected_answer   = expected,
            is_correct        = (
                final_answer.lower().strip() == str(expected).lower().strip()
                if expected else None
            ),
            # Errors
            error             = error,
            is_failed         = is_failed,
        )


# ── Module-level entry point ───────────────────────────────────────────────────

def run(query: str, model: str, dataset: str, **kwargs) -> AgentResponse:
    agent = ReactAgent(model=model, dataset=dataset)
    return agent.run(query=query, dataset=dataset, **kwargs)


