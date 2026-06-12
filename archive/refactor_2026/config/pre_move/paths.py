"""Repository roots and path aliases (legacy vs opt-in new layout).

Toggle new layout::

    export RESEARCH_USE_NEW_PATHS=1

Legacy paths remain the default. ``LIVE_EVAL_BASELINE_ROOT`` always points at
``results/orchestrator/graph_main_tuned`` (universe B baselines are not relocated).
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

USE_NEW_DATA_LAYOUT = os.getenv("RESEARCH_USE_NEW_PATHS", "0") == "1"

_SPLIT_PARQUET_KEYS = {"train": "train", "val": "val", "test": "test"}


def eval_samples_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/input/samples/eval"
    return REPO_ROOT / "datasets/eval_samples"


def labels_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/input/labels/v1"
    return REPO_ROOT / "datasets/train_samples/v1"


def split_csv_path(split: str) -> Path:
    names = {
        "train": "qce_train.csv",
        "val": "qce_val.csv",
        "test": "qce_internal_test.csv",
    }
    if split not in names:
        raise ValueError(f"split must be one of {list(names)}")
    return labels_dir() / names[split]


def qce_features_split_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/intermediate/qce_features/by_split"
    return REPO_ROOT / "datasets/qce_features"


def qce_features_eval_dir(tag: str) -> Path:
    t = tag.strip().lower()
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/intermediate/qce_features/by_dataset" / t
    return REPO_ROOT / "datasets/qce_features"


def complexity_record_path(split: str) -> Path:
    key = _SPLIT_PARQUET_KEYS[split]
    return qce_features_split_dir() / f"complexity_record_{key}.parquet"


def embeddings_parquet_path(split: str) -> Path:
    key = _SPLIT_PARQUET_KEYS[split]
    return qce_features_split_dir() / f"query_embeddings_{key}.parquet"


def heuristics_parquet_path(split: str) -> Path:
    key = _SPLIT_PARQUET_KEYS[split]
    return qce_features_split_dir() / f"query_heuristics_{key}.parquet"


def eval_complexity_path(tag: str) -> Path:
    t = tag.strip().lower()
    return qce_features_eval_dir(t) / f"complexity_record_{t}.parquet"


def eval_embeddings_path(tag: str) -> Path:
    t = tag.strip().lower()
    return qce_features_eval_dir(t) / f"query_embeddings_{t}.parquet"


def decomposer_cache_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/intermediate/decomposer_cache"
    return REPO_ROOT / "datasets/decomposer_cache"


def decomposer_cache_path(tag: str) -> Path:
    t = tag.strip().lower()
    return decomposer_cache_dir() / f"qce_{t}_plans.jsonl"


def oracle_results_csv() -> Path:
    return labels_dir() / "oracle_results1.csv"


def router_production_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "models/router/production"
    return REPO_ROOT / "models/router/graph_main"


def router_selective_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "models/router/selective"
    return REPO_ROOT / "models/router/graph_main"


def live_eval_baseline_root() -> Path:
    """Universe B physical baselines — legacy location (never moved in Phase 5)."""
    return REPO_ROOT / "results/orchestrator/graph_main_tuned"


def results_runs_root() -> Path:
    return REPO_ROOT / "results/runs"


def orchestrator_default_root() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return results_runs_root()
    return REPO_ROOT / "results/orchestrator"


def new_run_dir(experiment_id: str, *, when: datetime | None = None) -> Path:
    """``results/runs/{YYYY-MM-DD}_{experiment_id}/`` (new layout only)."""
    ts = when or datetime.now(timezone.utc)
    safe = experiment_id.replace("/", "_").replace(" ", "_")
    return results_runs_root() / f"{ts.strftime('%Y-%m-%d')}_{safe}"


def legacy_embeddings_template() -> str:
    """Relative template kept for back-compat shims."""
    if USE_NEW_DATA_LAYOUT:
        return "datasets/intermediate/qce_features/by_split/query_embeddings_{split}.parquet"
    return "datasets/qce_features/query_embeddings_{split}.parquet"


def legacy_heuristics_template() -> str:
    if USE_NEW_DATA_LAYOUT:
        return "datasets/intermediate/qce_features/by_split/query_heuristics_{split}.parquet"
    return "datasets/qce_features/query_heuristics_{split}.parquet"
