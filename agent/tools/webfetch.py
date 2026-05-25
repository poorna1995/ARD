from __future__ import annotations

import json
import os
import re

from agent.tools.decorator import tool

_FETCH_TIMEOUT = float(os.environ.get("WEB_FETCH_TIMEOUT_SEC", "25"))
# Main article/body text cap (trafilatura output is usually clean enough for a larger window).
_CONTENT_MAX = max(500, int(os.environ.get("WEB_FETCH_MAX_CHARS", "8000")))

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


def _extract_wikipedia_infobox(html: str) -> str:
    """Pull infobox / wikitable text (jersey #, dates) often dropped from main-body extract."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        return ""

    soup = BeautifulSoup(html, "html.parser")
    chunks: list[str] = []
    for box in soup.select("table.infobox, table.wikitable"):
        rows = box.find_all("tr")
        for row in rows[:40]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            line = " | ".join(c.get_text(" ", strip=True) for c in cells if c.get_text(strip=True))
            if line:
                chunks.append(line)
    return "\n".join(chunks).strip()


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


_RAW_URL_MARKERS = (
    "raw.githubusercontent.com",
    "gist.githubusercontent.com",
    "gist.github.com/raw",
)

_PLAINTEXT_SUFFIXES = (
    ".txt",
    ".csv",
    ".tsv",
    ".json",
    ".jsonl",
    ".md",
    ".xml",
    ".yaml",
    ".yml",
    ".log",
)


def _github_blob_to_raw(url: str) -> str | None:
    m = re.match(
        r"https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.*)",
        url,
        flags=re.IGNORECASE,
    )
    if not m:
        return None
    return (
        f"https://raw.githubusercontent.com/{m.group(1)}/{m.group(2)}"
        f"/{m.group(3)}/{m.group(4)}"
    )


def _looks_like_plaintext_body(body: str) -> bool:
    sample = (body or "")[:800].lstrip()
    if not sample:
        return False
    if sample.startswith(("{", "[", "#")) and "<html" not in sample.lower():
        return True
    if "<html" in sample.lower() or "<!doctype" in sample.lower():
        return False
    if "<" in sample and ">" in sample and re.search(r"<[a-zA-Z][^>]*>", sample):
        return False
    return True


def _url_suggests_plaintext(url: str, content_type: str) -> bool:
    url_l = (url or "").lower()
    if any(marker in url_l for marker in _RAW_URL_MARKERS):
        return True
    if any(url_l.split("?", 1)[0].endswith(sfx) for sfx in _PLAINTEXT_SUFFIXES):
        return True
    ct = (content_type or "").lower()
    if ct.startswith("text/plain"):
        return True
    if "application/json" in ct or "text/csv" in ct:
        return True
    return False


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
        "Uses trafilatura when installed (tables included); Wikipedia pages also include infobox rows. "
        "Env: WEB_FETCH_MAX_CHARS (default 8000), WEB_FETCH_TIMEOUT_SEC. "
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

        fetch_url = url
        raw_alt = _github_blob_to_raw(url)
        if raw_alt:
            fetch_url = raw_alt

        req = urllib.request.Request(fetch_url, headers=dict(_DEFAULT_HEADERS))
        with urllib.request.urlopen(req, timeout=_FETCH_TIMEOUT) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            final_url = getattr(resp, "geturl", lambda: fetch_url)()
            content_type = resp.headers.get("Content-Type", "")

        if _url_suggests_plaintext(final_url, content_type) or _looks_like_plaintext_body(
            body
        ):
            text = _tidy_extracted_text(body)
            title = final_url.rsplit("/", 1)[-1] or final_url
            content = text[:_CONTENT_MAX]
            highlights = [ln.strip() for ln in content.splitlines() if ln.strip()][:3]
            return _dump(
                {
                    "title": title,
                    "content": content,
                    "highlights": highlights,
                    "sources": [final_url],
                    "format": "plaintext",
                }
            )

        title_match = re.search(
            r"<title[^>]*>(.*?)</title>", body, flags=re.IGNORECASE | re.DOTALL
        )
        title = (
            re.sub(r"\s+", " ", title_match.group(1)).strip() if title_match else ""
        )

        extracted = _extract_with_trafilatura(body, final_url)
        if extracted is not None:
            text, meta_title = extracted
            if meta_title:
                title = meta_title
        else:
            text = _extract_with_bs4(body)

        text = _tidy_extracted_text(text)
        if "wikipedia.org" in final_url.lower():
            infobox = _extract_wikipedia_infobox(body)
            if infobox:
                text = f"{text}\n\n[Infobox / tables]\n{infobox}".strip()
        if not text:
            hint = ""
            if raw_alt and raw_alt != url:
                hint = f" Tried raw URL: {raw_alt}."
            elif _github_blob_to_raw(url):
                hint = f" Try raw URL: {_github_blob_to_raw(url)}."
            text = f"No readable text extracted from page.{hint}"

        content = text[:_CONTENT_MAX]
        highlights = [s.strip() for s in re.split(r"(?<=[.!?])\s+", content) if s.strip()][:3]

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
