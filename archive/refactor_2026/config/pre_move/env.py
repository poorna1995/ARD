"""Environment overrides and documented ``RESEARCH_*`` variables."""

from __future__ import annotations

import os
from pathlib import Path

from config.common import REPO_ROOT

# ── Env var catalog (global) ─────────────────────────────────────────────────

ENV_USE_NEW_PATHS = "RESEARCH_USE_NEW_PATHS"
ENV_ROUTER_MODEL = "RESEARCH_ROUTER_MODEL_PATH"
ENV_OUTPUT_ROOT = "RESEARCH_OUTPUT_ROOT"
ENV_LOG_JSON = "RESEARCH_LOG_JSON"
ENV_LOG_LEVEL = "RESEARCH_LOG_LEVEL"
ENV_LLM_MAX_RETRIES = "RESEARCH_LLM_MAX_RETRIES"
ENV_LLM_RETRY_BASE_SEC = "RESEARCH_LLM_RETRY_BASE_SEC"
ENV_LLM_MIN_INTERVAL_SEC = "RESEARCH_LLM_MIN_INTERVAL_SEC"


def router_model_path(default: Path) -> Path:
    """``RESEARCH_ROUTER_MODEL_PATH`` — absolute or repo-relative."""
    raw = os.getenv(ENV_ROUTER_MODEL, "").strip()
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def output_root(default: Path | None = None) -> Path:
    """``RESEARCH_OUTPUT_ROOT`` — absolute or repo-relative (default ``results/``)."""
    raw = os.getenv(ENV_OUTPUT_ROOT, "").strip()
    if not raw:
        return default if default is not None else REPO_ROOT / "results"
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


__all__ = [
    "ENV_LLM_MAX_RETRIES",
    "ENV_LLM_MIN_INTERVAL_SEC",
    "ENV_LLM_RETRY_BASE_SEC",
    "ENV_LOG_JSON",
    "ENV_LOG_LEVEL",
    "ENV_OUTPUT_ROOT",
    "ENV_ROUTER_MODEL",
    "ENV_USE_NEW_PATHS",
    "output_root",
    "router_model_path",
]
