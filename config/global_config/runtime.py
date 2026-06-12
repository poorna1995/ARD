"""Global runtime — env overrides, logging, secrets, LLM retry."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from dotenv import load_dotenv

from config.common import REPO_ROOT

load_dotenv()

T = TypeVar("T")
_log = logging.getLogger(__name__)
_last_llm_mono = 0.0


def _resolve_env_path(var: str, default: Path) -> Path:
    raw = os.getenv(var, "").strip()
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_router_path(default: Path) -> Path:
    """``RESEARCH_ROUTER_MODEL_PATH`` — absolute or repo-relative."""
    return _resolve_env_path("RESEARCH_ROUTER_MODEL_PATH", default)


def resolve_output_root(default: Path | None = None) -> Path:
    """``RESEARCH_OUTPUT_ROOT`` — absolute or repo-relative (default ``results/``)."""
    return _resolve_env_path("RESEARCH_OUTPUT_ROOT", default or REPO_ROOT / "results")


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(*, json_logs: bool | None = None, level: str | None = None) -> None:
    """Configure root logging once (idempotent)."""
    root = logging.getLogger()
    if getattr(root, "_research_configured", False):
        return

    use_json = json_logs if json_logs is not None else os.getenv("RESEARCH_LOG_JSON", "0") == "1"
    lvl = getattr(logging, (level or os.getenv("RESEARCH_LOG_LEVEL", "INFO")).upper(), logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        JsonLineFormatter() if use_json else logging.Formatter("%(levelname)s %(name)s: %(message)s")
    )
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    root._research_configured = True  # type: ignore[attr-defined]


REQUIRED_LLM_KEYS = ("OPENAI_API_KEY",)
REQUIRED_AGENT_KEYS = ("OPENAI_API_KEY",)
OPTIONAL_ENV_KEYS = (
    "GROQ_API_KEY",
    "SERPER_API_KEY",
    "SERP_API_KEY",
    "GITHUB_TOKEN",
    "ASSEMBLYAI_API_KEY",
    "HF_TOKEN",
    "HUGGINGFACE_TOKEN",
    "RESEARCH_ROUTER_MODEL_PATH",
    "RESEARCH_OUTPUT_ROOT",
    "RESEARCH_USE_NEW_PATHS",
)


@dataclass(frozen=True)
class EnvKeyStatus:
    missing_required: tuple[str, ...]
    present_optional: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.missing_required


def verify_env_keys(*, require: tuple[str, ...] = ()) -> EnvKeyStatus:
    missing = tuple(k for k in require if not os.getenv(k, "").strip())
    present = tuple(k for k in OPTIONAL_ENV_KEYS if os.getenv(k, "").strip())
    return EnvKeyStatus(missing_required=missing, present_optional=present)


def format_env_report(status: EnvKeyStatus) -> str:
    parts = []
    if status.missing_required:
        parts.append(f"missing required: {', '.join(status.missing_required)}")
    if status.present_optional:
        parts.append(f"optional set: {', '.join(status.present_optional)}")
    return "; ".join(parts) if parts else "secrets OK"


def pace_llm_calls() -> None:
    """Enforce minimum spacing between consecutive LLM requests."""
    global _last_llm_mono
    gap = max(0.0, float(os.getenv("RESEARCH_LLM_MIN_INTERVAL_SEC", "0")))
    if gap > 0:
        wait = gap - (time.monotonic() - _last_llm_mono)
        if wait > 0:
            time.sleep(wait)
    _last_llm_mono = time.monotonic()


def retry_with_backoff(fn: Callable[[], T], *, label: str = "llm") -> T:
    """Call ``fn`` with exponential backoff on transient OpenAI errors."""
    max_attempts = max(1, int(os.getenv("RESEARCH_LLM_MAX_RETRIES", "3")))
    base_delay = max(0.0, float(os.getenv("RESEARCH_LLM_RETRY_BASE_SEC", "1.0")))
    last_exc: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        pace_llm_calls()
        try:
            return fn()
        except Exception as exc:
            last_exc = exc
            if not _is_transient_error(exc) or attempt >= max_attempts:
                raise
            delay = base_delay * (2 ** (attempt - 1))
            _log.warning(
                "%s attempt %d/%d failed (%s); retry in %.1fs",
                label,
                attempt,
                max_attempts,
                exc.__class__.__name__,
                delay,
            )
            time.sleep(delay)

    assert last_exc is not None
    raise last_exc


def _is_transient_error(exc: Exception) -> bool:
    if exc.__class__.__name__ in {
        "RateLimitError",
        "APIConnectionError",
        "APITimeoutError",
        "InternalServerError",
    }:
        return True
    status = getattr(exc, "status_code", None)
    return status is not None and int(status) >= 500
