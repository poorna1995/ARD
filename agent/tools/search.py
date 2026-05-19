from __future__ import annotations

import json
import os
from pathlib import Path

from agent.tools.decorator import tool


def _load_env() -> None:
    """Load repo .env so SERP_API_KEY / SERPER_API_KEY are available in tool calls."""
    if os.environ.get("SERPER_API_KEY") or os.environ.get("SERP_API_KEY"):
        return
    try:
        from dotenv import load_dotenv

        root = Path(__file__).resolve().parents[2]
        load_dotenv(root / ".env")
    except Exception:
        pass


def _serper_search(query: str, api_key: str, num: int = 5) -> list[dict]:
    """Search via Serper (Google) API."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed: pip install requests")

    response = requests.post(
        "https://google.serper.dev/search",
        headers={
            "X-API-KEY": api_key,
            "Content-Type": "application/json",
        },
        json={"q": query, "num": num},
        timeout=15,
    )
    response.raise_for_status()
    data = response.json()

    results = []
    for item in data.get("organic", [])[:num]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return results


def _paid_google_search(query: str, api_key: str, num: int = 5) -> tuple[list[dict], str]:
    """
    Try Serper (serper.dev) first; on 401/403 retry SerpAPI (serpapi.com).
    Many accounts use one key name for the other provider.
    """
    try:
        return _serper_search(query, api_key, num), "serper"
    except Exception as e:
        err = str(e)
        if not any(x in err for x in ("401", "403", "Unauthorized", "Forbidden")):
            raise
        return _serpapi_search(query, api_key, num), "serpapi"


def _serpapi_search(query: str, api_key: str, num: int = 5) -> list[dict]:
    """Search via SerpAPI (serpapi.com) — uses SERP_API_KEY in .env."""
    try:
        import requests
    except ImportError:
        raise RuntimeError("requests not installed: pip install requests")

    response = requests.get(
        "https://serpapi.com/search",
        params={
            "q": query,
            "api_key": api_key,
            "engine": "google",
            "num": num,
        },
        timeout=25,
    )
    response.raise_for_status()
    data = response.json()
    if data.get("error"):
        raise RuntimeError(str(data["error"]))

    results = []
    for item in data.get("organic_results", [])[:num]:
        results.append(
            {
                "title": item.get("title", ""),
                "url": item.get("link", ""),
                "snippet": item.get("snippet", ""),
            }
        )
    return results


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
        raise RuntimeError(
            "Web search needs `ddgs` or `duckduckgo-search`. "
            "Install: uv pip install ddgs"
        ) from exc


def _duckduckgo_search(query: str, max_results: int = 5) -> list[dict]:
    """Free fallback using DuckDuckGo. No key required."""
    DDGS = _import_ddgs()
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", r.get("link", "")),
                    "snippet": r.get("body", r.get("snippet", "")),
                }
            )
    return results


@tool(
    "web_search",
    (
        "Search the web for current information. "
        "Input: a short, specific search query. "
        "Returns top result titles, URLs, and snippets. "
        "To read the full content of a result, follow up with web_fetch[<url>]. "
        "Uses SERPER_API_KEY (serper.dev) or SERP_API_KEY (serpapi.com) when set; "
        "auto-detects SerpAPI if a Serper key is rejected; otherwise DuckDuckGo. "
        "Examples: "
        "web_search[Nobel Prize Chemistry 2023 winner] "
        "web_search[current CEO of OpenAI] "
        "web_search[Python 3.12 release date]"
    ),
)
def web_search(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return json.dumps({"error": "Empty query. Provide a search term."})

    _load_env()
    paid_key = (
        os.environ.get("SERPER_API_KEY", "").strip()
        or os.environ.get("SERP_API_KEY", "").strip()
    )
    primary_err: str | None = None

    try:
        if paid_key:
            results, source = _paid_google_search(q, paid_key)
        else:
            results = _duckduckgo_search(q)
            source = "duckduckgo"
    except Exception as e:
        primary_err = str(e)
        try:
            results = _duckduckgo_search(q)
            source = "duckduckgo_fallback"
        except Exception as e2:
            return json.dumps(
                {
                    "error": f"Both search providers failed. Primary: {e}. Fallback: {e2}",
                    "query": q,
                }
            )

    if not results:
        return json.dumps(
            {
                "query": q,
                "results": [],
                "source": source,
                "note": "No results found. Try a different query.",
            }
        )

    payload: dict = {
        "query": q,
        "results": results,
        "source": source,
        "hint": "Use web_fetch[<url>] to read the full content of any result.",
    }
    if primary_err:
        payload["warning"] = f"Primary search failed ({primary_err}); used DuckDuckGo."
    return json.dumps(payload, ensure_ascii=False)


# """
# Web search via DuckDuckGo (sync API suitable for ReAct tools).

# Compared to an async `DDGAPIWrapper`:
# - ReAct calls tools synchronously, so we stay sync here (no asyncio inside the tool).
# - You can still offload blocking I/O with a thread + timeout (below) so slow searches
#   fail fast instead of hanging the agent loop.
# """

# from __future__ import annotations

# import os
# import re
# from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
# from typing import Any

# from agent.tools.decorator import tool

# _DEFAULT_MAX = int(os.environ.get("WEB_SEARCH_MAX_RESULTS", "6"))
# _TIMEOUT_SEC = float(os.environ.get("WEB_SEARCH_TIMEOUT_SEC", "45"))


# def _import_ddgs():
#     try:
#         from ddgs import DDGS

#         return DDGS
#     except ImportError:
#         pass
#     try:
#         from duckduckgo_search import DDGS  # type: ignore[import-untyped]

#         return DDGS
#     except ImportError as exc:
#         raise ImportError(
#             "Web search needs `ddgs` or `duckduckgo-search`. "
#             "Install: uv pip install ddgs   or   uv pip install duckduckgo-search"
#         ) from exc


# def _search_ddgs(query: str, max_results: int) -> list[dict[str, Any]]:
#     """Return structured rows like the LangChain-style wrapper (title / link / snippet)."""
#     DDGS = _import_ddgs()
#     rows: list[dict[str, Any]] = []
#     with DDGS() as ddgs:
#         # Same iteration pattern as reference: cap with zip + range.
#         stream = ddgs.text(query, max_results=max_results)
#         for _, item in zip(range(max_results), stream):
#             rows.append(
#                 {
#                     "title": item.get("title", ""),
#                     "href": item.get("href", ""),
#                     "body": item.get("body", ""),
#                 }
#             )
#     return rows


# def _format_for_llm(rows: list[dict[str, Any]]) -> str:
#     if not rows:
#         return "No results found."
#     blocks = []
#     for i, r in enumerate(rows, start=1):
#         blocks.append(
#             f"[{i}] {r['title']}\n"
#             f"URL     : {r['href']}\n"
#             f"Snippet : {r['body']}"
#         )
#     return "\n\n".join(blocks)


# def _shorten_query_for_retry(q: str) -> str:
#     """DDG often returns empty for very long or multi-sentence queries."""
#     s = " ".join(q.split())
#     if len(s) <= 120:
#         return s
#     # Prefer first quoted phrase if present
#     m = re.search(r'"([^"]{10,200})"', s)
#     if m:
#         return m.group(1).strip()
#     return s[:120].rsplit(" ", 1)[0].strip() if " " in s[:120] else s[:120]


# def _web_search_impl(query: str, max_results: int) -> str:
#     q = (query or "").strip()
#     if not q:
#         return "Error: empty search query."
#     rows = _search_ddgs(q, max_results=max_results)
#     if not rows:
#         q2 = _shorten_query_for_retry(q)
#         if q2 and q2 != q:
#             rows = _search_ddgs(q2, max_results=max_results)
#     return _format_for_llm(rows)


# @tool(
#     "web_search",
#     (
#         "Search the web (short factual lookup). Input: ONE tight query string — not a paragraph. "
#         "Include concrete anchors from the task: exact titles, dates, arXiv ids (e.g. 1608.03637), "
#         "site:arxiv.org, quoted rare phrases. "
#         "Good: web_search[arxiv 1608.03637 physics.soc-ph] "
#         "Good: web_search[\"Phase transition from egalitarian\" site:arxiv.org] "
#         "Bad: web_search[paper about AI regulation figure axes society article 2016] (too vague / laundry list)."
#     ),
# )
# def web_search(query: str) -> str:
#     """
#     Public tool entrypoint: human-readable string for the LLM scratchpad.

#     Env:
#       WEB_SEARCH_MAX_RESULTS (default 6)
#       WEB_SEARCH_TIMEOUT_SEC (default 45) — thread pool timeout around DDG calls
#     """
#     max_results = max(1, min(_DEFAULT_MAX, 25))
#     try:
#         with ThreadPoolExecutor(max_workers=1) as pool:
#             fut = pool.submit(_web_search_impl, query, max_results)
#             return fut.result(timeout=_TIMEOUT_SEC)
#     except FuturesTimeout:
#         return f"Search timed out after {_TIMEOUT_SEC:.0f}s."
#     except ImportError as e:
#         return str(e)
#     except Exception as e:
#         return f"Search failed: {e}"
