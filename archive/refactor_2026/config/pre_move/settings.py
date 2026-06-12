"""Environment overrides for deployment (optional; defaults unchanged when unset)."""

from __future__ import annotations

import os
from pathlib import Path

from config.paths import REPO_ROOT


def router_model_path(default: Path) -> Path:
    """``RESEARCH_ROUTER_MODEL_PATH`` — absolute or repo-relative."""
    raw = os.getenv("RESEARCH_ROUTER_MODEL_PATH", "").strip()
    if not raw:
        return default
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p


def output_root(default: Path | None = None) -> Path:
    """``RESEARCH_OUTPUT_ROOT`` — absolute or repo-relative (default ``results/``)."""
    raw = os.getenv("RESEARCH_OUTPUT_ROOT", "").strip()
    if not raw:
        return default if default is not None else REPO_ROOT / "results"
    p = Path(raw)
    return p if p.is_absolute() else REPO_ROOT / p
