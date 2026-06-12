"""HGBM hyperparameter search grid."""

from __future__ import annotations

from typing import Any

HGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
}

__all__ = ["HGBM_GRID"]
