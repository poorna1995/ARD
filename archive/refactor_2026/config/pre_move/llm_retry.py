"""Retry + rate-limit helpers for OpenAI / LLM calls."""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")

_last_llm_call_mono: float = 0.0
_log = logging.getLogger(__name__)


def llm_max_retries() -> int:
    return max(1, int(os.getenv("RESEARCH_LLM_MAX_RETRIES", "3")))


def llm_retry_base_sec() -> float:
    return max(0.0, float(os.getenv("RESEARCH_LLM_RETRY_BASE_SEC", "1.0")))


def llm_min_interval_sec() -> float:
    return max(0.0, float(os.getenv("RESEARCH_LLM_MIN_INTERVAL_SEC", "0")))


def throttle_llm() -> None:
    """Enforce minimum spacing between consecutive LLM requests."""
    global _last_llm_call_mono
    gap = llm_min_interval_sec()
    if gap <= 0:
        _last_llm_call_mono = time.monotonic()
        return
    now = time.monotonic()
    wait = gap - (now - _last_llm_call_mono)
    if wait > 0:
        time.sleep(wait)
    _last_llm_call_mono = time.monotonic()


def call_with_retry(fn: Callable[[], T], *, label: str = "llm") -> T:
    """Call ``fn`` with exponential backoff on transient OpenAI errors."""
    retries = llm_max_retries()
    base = llm_retry_base_sec()
    last_exc: Exception | None = None

    for attempt in range(1, retries + 1):
        throttle_llm()
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not _is_retryable(exc) or attempt >= retries:
                raise
            delay = base * (2 ** (attempt - 1))
            _log.warning(
                "%s attempt %d/%d failed (%s); retry in %.1fs",
                label,
                attempt,
                retries,
                exc.__class__.__name__,
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _is_retryable(exc: Exception) -> bool:
    name = exc.__class__.__name__
    if name in {"RateLimitError", "APIConnectionError", "APITimeoutError", "InternalServerError"}:
        return True
    # openai SDK wraps HTTP errors
    status = getattr(exc, "status_code", None)
    if status is not None and int(status) >= 500:
        return True
    return False
