"""Legacy ``routing.*`` submodule aliases → ``router`` / ``eval`` / ``research``."""

from __future__ import annotations

import importlib
import sys

# Old import path → new module (must be importable).
_ALIASES: dict[str, str] = {
    "routing.router": "router.router",
    "routing.config": "router.config",
    "routing.experiments": "router.experiments",
    "routing.datasets": "router.config",
    "routing.score_routes": "eval.score",
    "routing.benchmark": "eval.benchmark",
    "routing.oracle_bounds": "eval.oracle_bounds",
    "routing.analysis": "research.analysis",
    "routing.soft_train": "research.soft_train",
    "routing.baselines": "research.baselines",
    "routing.figures": "research.figures",
    "routing.selective": "research.selective",
    "routing.eval.outcomes": "eval.score",
    "routing.eval.metrics": "eval.score",
    "routing.features.eval_builder": "qce.features",
    "routing.features.heuristics": "router.config",
}


def install_legacy_submodules() -> None:
    for legacy, target in _ALIASES.items():
        if legacy in sys.modules:
            continue
        sys.modules[legacy] = importlib.import_module(target)
