"""
Dataset display names and alias resolution (universe A + B).

IN:  dataset string keys
MID: normalize aliases · map to paper display labels
OUT: canonical dataset id · human-readable name
"""

from __future__ import annotations

from routing.config import DATASET_ALIASES

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "hotpot": "HotpotQA",
    "musique": "MuSiQue",
    "math": "MATH",
    "mmlu": "MMLU",
    "mmlu_pro": "MMLU",
    "gaia": "GAIA",
}

# Back-compat alias used by benchmark / oracle_bounds.
DATASET_DISPLAY = DATASET_DISPLAY_NAMES

DISCUSSION_DATASET_ORDER = ["hotpot", "musique", "math", "mmlu", "gaia"]


def resolve_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get((name or "").strip().lower(), (name or "").strip().lower())
