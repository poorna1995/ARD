"""Shared constants used by global and local configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

USE_NEW_DATA_LAYOUT = os.getenv("RESEARCH_USE_NEW_PATHS", "0") == "1"


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML required: pip install pyyaml") from exc
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


__all__ = ["REPO_ROOT", "USE_NEW_DATA_LAYOUT", "load_yaml"]
