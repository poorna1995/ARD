"""Orchestrator module entrypoints."""

from typing import Any

__all__ = ["run_pipeline"]


def run_pipeline(*args: Any, **kwargs: Any):
    from orchestrator.pipeline import run_pipeline as _run_pipeline

    return _run_pipeline(*args, **kwargs)
