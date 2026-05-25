from __future__ import annotations

import json
import logging
import os
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from typing import TypeVar

from agent.tools.decorator import tool
_MAX_CONTENT_CHARS = int(os.environ.get("WIKIPEDIA_MAX_CHARS", "3000"))
_WIKI_RETRY_ATTEMPTS = int(os.environ.get("WIKIPEDIA_RETRY_ATTEMPTS", "5"))
_WIKI_RETRY_BACKOFF_S = float(os.environ.get("WIKIPEDIA_RETRY_BACKOFF_S", "1.0"))
_WIKI_MIN_INTERVAL_S = float(os.environ.get("WIKIPEDIA_MIN_INTERVAL_S", "0.25"))
_WIKI_USER_AGENT = os.environ.get(
    "WIKIPEDIA_USER_AGENT",
    "ResearchWorkBot/1.0 (research_work; set WIKIPEDIA_USER_AGENT with contact email)",
)
_WIKI_API_MAX_RETRIES = int(os.environ.get("WIKIPEDIA_API_MAX_RETRIES", "5"))
_WIKI_API_RETRY_WAIT_S = float(os.environ.get("WIKIPEDIA_API_RETRY_WAIT", "2.0"))
_WIKI_FALLBACK_MIN_WAIT_MS = int(os.environ.get("WIKIPEDIA_FALLBACK_MIN_WAIT_MS", "250"))

logger = logging.getLogger(__name__)
_T = TypeVar("_T")

# Process-wide throttle + shared clients (benchmark may use thread pools).
_wiki_throttle_lock = threading.Lock()
_wiki_last_request_at = 0.0
_wikiapi_client = None
_wikiapi_client_lock = threading.Lock()
_wikipedia_pkg_configured = False
_wikipedia_pkg_lock = threading.Lock()


def _throttle_before_request() -> None:
    """Minimum gap between any Wikipedia HTTP call (both libraries)."""
    global _wiki_last_request_at
    with _wiki_throttle_lock:
        now = time.monotonic()
        wait = _WIKI_MIN_INTERVAL_S - (now - _wiki_last_request_at)
        if wait > 0:
            time.sleep(wait)
        _wiki_last_request_at = time.monotonic()


def _get_wikipediaapi():
    """Shared wikipedia-api client (retries 429/5xx with backoff)."""
    global _wikiapi_client
    with _wikiapi_client_lock:
        if _wikiapi_client is None:
            import wikipediaapi

            _wikiapi_client = wikipediaapi.Wikipedia(
                user_agent=_WIKI_USER_AGENT,
                language="en",
                max_retries=_WIKI_API_MAX_RETRIES,
                retry_wait=_WIKI_API_RETRY_WAIT_S,
            )
        return _wikiapi_client


def _configure_wikipedia_package() -> None:
    """Configure goldsmith ``wikipedia`` once: HTTPS lang, UA, client-side rate limit."""
    global _wikipedia_pkg_configured
    with _wikipedia_pkg_lock:
        if _wikipedia_pkg_configured:
            return
        import wikipedia

        wikipedia.set_lang("en")
        wikipedia.set_user_agent(_WIKI_USER_AGENT)
        wikipedia.set_rate_limiting(
            True,
            min_wait=timedelta(milliseconds=_WIKI_FALLBACK_MIN_WAIT_MS),
        )
        _wikipedia_pkg_configured = True


def _is_rate_limit_error(exc: BaseException) -> bool:
    current: BaseException | None = exc
    while current is not None:
        try:
            import wikipediaapi

            if isinstance(current, wikipediaapi.WikiRateLimitError):
                return True
            if isinstance(current, wikipediaapi.WikiHttpError) and current.status_code == 429:
                return True
        except ImportError:
            pass
        msg = str(current).lower()
        if "429" in msg or "rate limit" in msg or "too many requests" in msg:
            return True
        current = current.__cause__
    return False


def _is_transient_wiki_error(exc: BaseException) -> bool:
    """True for empty/truncated JSON from Wikipedia REST under rate pressure."""
    if _is_rate_limit_error(exc):
        return True
    current: BaseException | None = exc
    while current is not None:
        if isinstance(current, json.JSONDecodeError):
            return True
        if "expecting value" in str(current).lower():
            return True
        try:
            import wikipediaapi

            if isinstance(
                current,
                (
                    wikipediaapi.WikiInvalidJsonError,
                    wikipediaapi.WikiConnectionError,
                    wikipediaapi.WikiHttpTimeoutError,
                ),
            ):
                return True
            if isinstance(current, wikipediaapi.WikiHttpError) and current.status_code >= 500:
                return True
        except ImportError:
            pass
        try:
            import wikipedia

            if isinstance(current, wikipedia.exceptions.HTTPTimeoutError):
                return True
        except ImportError:
            pass
        current = current.__cause__
    return False


def _retry_delay(exc: BaseException, attempt: int) -> float:
    """Longer waits after 429; honor Retry-After when exposed."""
    try:
        import wikipediaapi

        if isinstance(exc, wikipediaapi.WikiRateLimitError) and exc.retry_after:
            return float(exc.retry_after)
    except ImportError:
        pass
    if _is_rate_limit_error(exc):
        return max(_WIKI_RETRY_BACKOFF_S * (2 ** (attempt - 1)), 3.0)
    return _WIKI_RETRY_BACKOFF_S * attempt


def _wiki_retry(fn: Callable[[], _T], *, label: str) -> _T:
    """Retry only transient parse/network/rate-limit glitches."""
    last_exc: BaseException | None = None
    for attempt in range(1, _WIKI_RETRY_ATTEMPTS + 1):
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            try:
                import wikipedia

                if isinstance(
                    exc,
                    (
                        wikipedia.exceptions.DisambiguationError,
                        wikipedia.exceptions.PageError,
                    ),
                ):
                    raise
            except ImportError:
                pass
            if not _is_transient_wiki_error(exc) or attempt >= _WIKI_RETRY_ATTEMPTS:
                raise
            delay = _retry_delay(exc, attempt)
            logger.warning(
                "Wikipedia transient error for %r (attempt %d/%d): %s; retry in %.1fs",
                label,
                attempt,
                _WIKI_RETRY_ATTEMPTS,
                exc,
                delay,
            )
            time.sleep(delay)
    assert last_exc is not None
    raise last_exc


def _search_candidates(query: str) -> list[str]:
    """
    Return a list of Wikipedia page titles that match the query.
    Used when exact page lookup fails, to suggest alternatives.
    """
    try:
        import wikipedia

        _configure_wikipedia_package()

        def _search() -> list[str]:
            _throttle_before_request()
            return wikipedia.search(query, results=6)

        return _wiki_retry(_search, label=query)
    except Exception as e:
        logger.warning("Wikipedia candidate search failed for '%s': %s", query, e)
        return []


def _trim_content(text: str, max_chars: int) -> str:
    """Trim text to max_chars at the last complete sentence boundary."""
    if len(text) <= max_chars:
        return text
    content = text[:max_chars]
    last_period = content.rfind(". ")
    if last_period > max_chars // 2:
        content = content[: last_period + 1]
    return content + "\n[...truncated]"


def _page_dict_from_wikipediaapi(page, max_chars: int) -> dict | None:
    content = page.text or page.summary
    if not content:
        logger.warning("Both page.text and page.summary are empty for '%s'", page.title)
        return None
    content = _trim_content(content, max_chars)
    return {
        "title": page.title,
        "url": page.fullurl,
        "content": content,
        "source": "wikipedia",
    }


def _fetch_with_wikipediaapi(query: str, max_chars: int) -> dict | None:
    try:
        import wikipediaapi
    except ImportError:
        logger.error("wikipedia-api not installed. Run: pip install wikipedia-api")
        return None

    wiki = _get_wikipediaapi()

    def _load_page():
        _throttle_before_request()
        return wiki.page(query)

    try:
        page = _wiki_retry(_load_page, label=query)
    except wikipediaapi.WikiRateLimitError as e:
        logger.warning("Wikipedia rate limited loading page %r: %s", query, e)
        return None
    except wikipediaapi.WikipediaException as e:
        logger.error("wikipediaapi.page() failed for '%s': %s", query, e)
        return None

    if page.exists():
        return _page_dict_from_wikipediaapi(page, max_chars)

    logger.debug("Page does not exist: '%s', trying search candidates", query)
    candidates = _search_candidates(query)
    if not candidates:
        return None

    def _load_candidate():
        _throttle_before_request()
        return wiki.page(candidates[0])

    try:
        page = _wiki_retry(_load_candidate, label=candidates[0])
    except wikipediaapi.WikipediaException as e:
        logger.warning("wikipediaapi.page() failed for candidate %r: %s", candidates[0], e)
        return None

    if not page.exists():
        return None
    return _page_dict_from_wikipediaapi(page, max_chars)


def _fetch_with_wikipedia_package(query: str, max_chars: int) -> dict:
    """
    Fallback Wikipedia fetch using the wikipedia package.

    Unlike the primary fetcher, this intentionally lets
    DisambiguationError and PageError propagate so that
    wikipedia_search() can return helpful structured messages.
    """
    import wikipedia  # ImportError propagates intentionally

    _configure_wikipedia_package()

    def _load_page():
        _throttle_before_request()
        return wikipedia.page(query, auto_suggest=True)

    page = _wiki_retry(_load_page, label=query)
    content = _trim_content(page.content, max_chars)
    return {
        "title": page.title,
        "url": page.url,
        "content": content,
        "source": "wikipedia",
    }


@tool(
    "wikipedia_search",
    (
        "Look up a topic on Wikipedia and return a detailed article excerpt. "
        "Returns up to 3000 characters of article content — much more than a summary. "
        "Input: the topic or entity name. Be specific to avoid disambiguation. "
        "Examples: "
        "wikipedia_search[Albert Einstein] "
        "wikipedia_search[Python programming language] "
        "wikipedia_search[Battle of Waterloo] "
        "wikipedia_search[OpenAI company history]"
    ),
)
def wikipedia_search(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return json.dumps({"error": "Empty query. Provide a topic to look up."})

    result = _fetch_with_wikipediaapi(q, _MAX_CONTENT_CHARS)
    if result is not None:
        return json.dumps(result, ensure_ascii=False)

    try:
        import wikipedia
    except ImportError:
        return json.dumps(
            {
                "error": (
                    "Wikipedia tool unavailable. "
                    "Install: pip install wikipedia-api  or  pip install wikipedia"
                )
            }
        )

    try:
        result = _fetch_with_wikipedia_package(q, _MAX_CONTENT_CHARS)
        return json.dumps(result, ensure_ascii=False)

    except wikipedia.exceptions.DisambiguationError as e:
        options = e.options[:8]
        candidates = _search_candidates(q)
        return json.dumps(
            {
                "disambiguation": True,
                "query": q,
                "options": options,
                "search_suggestions": candidates,
                "message": (
                    f"'{q}' is ambiguous. Retry with one option from 'options' — "
                    "prefer titles grounded in the question's proper nouns, not attribute-only queries."
                ),
            },
            ensure_ascii=False,
        )

    except wikipedia.exceptions.PageError:
        candidates = _search_candidates(q)
        return json.dumps(
            {
                "error": f"No Wikipedia page found for '{q}'.",
                "suggestions": candidates,
                "hint": (
                    "Pick a suggested title; next query should reuse the question's proper nouns "
                    "— avoid attribute-only filters."
                ),
            },
            ensure_ascii=False,
        )

    except Exception as e:
        if _is_rate_limit_error(e) or _is_transient_wiki_error(e):
            logger.warning("Wikipedia rate limited or transient failure for '%s': %s", q, e)
            return json.dumps(
                {
                    "error": (
                        f"Wikipedia temporarily unavailable for '{q}' (rate limit or network). "
                        "Retry with a shorter entity-title query in a few seconds."
                    ),
                    "rate_limited": True,
                }
            )
        logger.error("Unexpected error fetching Wikipedia page '%s': %s", q, e)
        return json.dumps({"error": f"Unexpected error fetching '{q}': {e}"})
