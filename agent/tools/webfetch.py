from __future__ import annotations

import json
import os
import re

from agent.tools.decorator import tool

_FETCH_TIMEOUT = float(os.environ.get("WEB_FETCH_TIMEOUT_SEC", "25"))
# Main article/body text cap (trafilatura output is usually clean enough for a larger window).
_CONTENT_MAX = max(500, int(os.environ.get("WEB_FETCH_MAX_CHARS", "4500")))

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "close",
}


def _dump(data: dict) -> str:
    return json.dumps(data, ensure_ascii=False)


def _extract_with_trafilatura(html: str, url: str) -> tuple[str, str] | None:
    """
    Prefer trafilatura for main-text extraction (boilerplate removal, better than tag soup).
    Returns (text, title_hint) or None if unavailable / insufficient.
    """
    try:
        import trafilatura
    except ImportError:
        return None

    try:
        text = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            favor_precision=True,
        )
    except Exception:
        return None

    text = (text or "").strip()
    if len(text) < 80:
        return None

    title_hint = ""
    try:
        meta = trafilatura.extract_metadata(html, default_url=url)
        if meta is not None and getattr(meta, "title", None):
            title_hint = str(meta.title).strip()
    except Exception:
        pass

    return text, title_hint


def _tidy_extracted_text(s: str) -> str:
    """Normalize spaces; keep line breaks when the extractor returned multiple lines."""
    s = (s or "").strip()
    if not s:
        return ""
    if "\n" in s:
        lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in s.splitlines()]
        lines = [ln for ln in lines if ln]
        return "\n".join(lines)
    return re.sub(r"\s+", " ", s).strip()


def _extract_with_bs4(html: str) -> str:
    """Previous heuristic: main/article + headings and paragraphs."""
    try:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
            tag.decompose()

        main_block = soup.find("article") or soup.find("main") or soup.body or soup
        paragraph_chunks = [
            p.get_text(" ", strip=True)
            for p in main_block.find_all(["h1", "h2", "h3", "p", "li"])
        ]
        paragraph_chunks = [c for c in paragraph_chunks if len(c) >= 30]
        text = " ".join(paragraph_chunks)
    except Exception:
        text = re.sub(r"<[^>]+>", " ", html)

    return re.sub(r"\s+", " ", text).strip()


def _normalize_fetch_url(raw: str) -> str:
    """Strip whitespace and matching outer quotes models often add inside tool brackets."""
    url = (raw or "").strip()
    for _ in range(3):
        if len(url) >= 2 and url[0] == url[-1] and url[0] in "\"'":
            url = url[1:-1].strip()
        else:
            break
    return url


@tool(
    "web_fetch",
    (
        "Fetch one URL and return cleaned main text (JSON: title, content, highlights, sources). "
        "Uses trafilatura when installed for article extraction; otherwise BeautifulSoup heuristics. "
        "Optional: pip install trafilatura. Env: WEB_FETCH_MAX_CHARS, WEB_FETCH_TIMEOUT_SEC. "
        "Input: the URL only — no extra quotes or JSON. "
        "Good: web_fetch[https://arxiv.org/abs/1608.03637] "
        "Good: web_fetch[https://example.org/page] "
        "Bad: web_fetch[\"https://arxiv.org/abs/1608.03637\"] (do not wrap URL in quotes)."
    ),
)
def web_fetch(url: str) -> str:
    url = _normalize_fetch_url(url)
    if not url:
        return _dump(
            {
                "title": "",
                "content": "",
                "highlights": [],
                "sources": [],
                "error": {"code": "EMPTY_URL", "message": "No URL provided."},
            }
        )

    try:
        import urllib.request

        req = urllib.request.Request(url, headers=dict(_DEFAULT_HEADERS))
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
            final_url = getattr(resp, "geturl", lambda: url)()

        title_match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        title = re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""

        extracted = _extract_with_trafilatura(html, final_url)
        if extracted is not None:
            text, meta_title = extracted
            if meta_title:
                title = meta_title
        else:
            text = _extract_with_bs4(html)

        text = _tidy_extracted_text(text)
        if not text:
            text = "No readable text extracted from page."

        content = text[:_CONTENT_MAX]
        highlights = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()][:3]

        # Only the former `data` payload — structured JSON for the LLM observation.
        data = {
            "title": title,
            "content": content,
            "highlights": highlights,
            "sources": [final_url],
        }
        return _dump(data)
    except Exception as e:
        return _dump(
            {
                "title": "",
                "content": "",
                "highlights": [],
                "sources": [url],
                "error": {"code": "FETCH_FAILED", "message": str(e)},
            }
        )
