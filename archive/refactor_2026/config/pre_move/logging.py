"""Structured logging for CLI, orchestrator, and services."""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import UTC, datetime
from typing import Any


class JsonLogFormatter(logging.Formatter):
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


def setup_logging(*, json_logs: bool | None = None, level: str | None = None) -> None:
    """
    Configure root logging once (idempotent).

    ``RESEARCH_LOG_JSON=1`` → JSON lines on stderr.
    ``RESEARCH_LOG_LEVEL`` → default INFO.
    """
    root = logging.getLogger()
    if getattr(root, "_research_configured", False):
        return

    use_json = json_logs if json_logs is not None else os.getenv("RESEARCH_LOG_JSON", "0") == "1"
    lvl_name = (level or os.getenv("RESEARCH_LOG_LEVEL", "INFO")).upper()
    lvl = getattr(logging, lvl_name, logging.INFO)

    handler = logging.StreamHandler(sys.stderr)
    if use_json:
        handler.setFormatter(JsonLogFormatter())
    else:
        handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))

    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(lvl)
    root._research_configured = True  # type: ignore[attr-defined]
