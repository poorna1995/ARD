"""
Web search via DuckDuckGo (sync API suitable for ReAct tools).

Compared to an async `DDGAPIWrapper`:
- ReAct calls tools synchronously, so we stay sync here (no asyncio inside the tool).
- You can still offload blocking I/O with a thread + timeout (below) so slow searches
  fail fast instead of hanging the agent loop.
"""

from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from typing import Any

from agent.tools.decorator import tool

_DEFAULT_MAX = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "6"))
_TIMEOUT_SEC = float(os.environ.get("WEB_SEARCH_TIMEOUT_SEC", "45"))


def _import_ddgs():
    try:
        from ddgs import DDGS

        return DDGS
    except ImportError:
        pass
    try:
        from duckduckgo_search import DDGS  # type: ignore[import-untyped]

        return DDGS
    except ImportError as exc:
        raise ImportError(
            "Web search needs `ddgs` or `duckduckgo-search`. "
            "Install: uv pip install ddgs   or   uv pip install duckduckgo-search"
        ) from exc


def _search_ddgs(query: str, max_results: int) -> list[dict[str, Any]]:
    """Return structured rows like the LangChain-style wrapper (title / link / snippet)."""
    DDGS = _import_ddgs()
    rows: list[dict[str, Any]] = []
    with DDGS() as ddgs:
        # Same iteration pattern as reference: cap with zip + range.
        stream = ddgs.text(query, max_results=max_results)
        for _, item in zip(range(max_results), stream):
            rows.append(
                {
                    "title": item.get("title", ""),
                    "href": item.get("href", ""),
                    "body": item.get("body", ""),
                }
            )
    return rows


def _format_for_llm(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No results found."
    blocks = []
    for i, r in enumerate(rows, start=1):
        blocks.append(
            f"[{i}] {r['title']}\n"
            f"URL     : {r['href']}\n"
            f"Snippet : {r['body']}"
        )
    return "\n\n".join(blocks)


def _shorten_query_for_retry(q: str) -> str:
    """DDG often returns empty for very long or multi-sentence queries."""
    s = " ".join(q.split())
    if len(s) <= 120:
        return s
    # Prefer first quoted phrase if present
    m = re.search(r'"([^"]{10,200})"', s)
    if m:
        return m.group(1).strip()
    return s[:120].rsplit(" ", 1)[0].strip() if " " in s[:120] else s[:120]


def _web_search_impl(query: str, max_results: int) -> str:
    q = (query or "").strip()
    if not q:
        return "Error: empty search query."
    rows = _search_ddgs(q, max_results=max_results)
    if not rows:
        q2 = _shorten_query_for_retry(q)
        if q2 and q2 != q:
            rows = _search_ddgs(q2, max_results=max_results)
    return _format_for_llm(rows)


@tool(
    "web_search",
    (
        "Search the web (short factual lookup). Input: ONE tight query string — not a paragraph. "
        "Include concrete anchors from the task: exact titles, dates, arXiv ids (e.g. 1608.03637), "
        "site:arxiv.org, quoted rare phrases. "
        "Good: web_search[arxiv 1608.03637 physics.soc-ph] "
        "Good: web_search[\"Phase transition from egalitarian\" site:arxiv.org] "
        "Bad: web_search[paper about AI regulation figure axes society article 2016] (too vague / laundry list)."
    ),
)
def web_search(query: str) -> str:
    """
    Public tool entrypoint: human-readable string for the LLM scratchpad.

    Env:
      WEB_SEARCH_MAX_RESULTS (default 6)
      WEB_SEARCH_TIMEOUT_SEC (default 45) — thread pool timeout around DDG calls
    """
    max_results = max(1, min(_DEFAULT_MAX, 25))
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_web_search_impl, query, max_results)
            return fut.result(timeout=_TIMEOUT_SEC)
    except FuturesTimeout:
        return f"Search timed out after {_TIMEOUT_SEC:.0f}s."
    except ImportError as e:
        return str(e)
    except Exception as e:
        return f"Search failed: {e}"
