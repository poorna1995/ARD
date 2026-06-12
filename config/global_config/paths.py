"""Repository roots and path aliases (legacy vs opt-in new layout).

Toggle new layout::

    export RESEARCH_USE_NEW_PATHS=1

Legacy paths remain the default. ``LIVE_EVAL_BASELINE_ROOT`` always points at
``results/orchestrator/graph_main_tuned`` (universe B baselines are not relocated).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from config.common import REPO_ROOT, USE_NEW_DATA_LAYOUT

_SPLIT_PARQUET_KEYS = {"train": "train", "val": "val", "test": "test"}


def eval_samples_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/input/samples/eval"
    return REPO_ROOT / "datasets/eval_samples"


def labels_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/input/labels/v1"
    return REPO_ROOT / "datasets/train_samples/v1"


def daar_root() -> Path:
    """D-AAR experiment root — isolated from baseline QCE/AAR under ``datasets/daar/``."""
    return REPO_ROOT / "datasets/daar"


def daar_source_dir() -> Path:
    """Offline execution pool (raw / master parquets) for D-AAR."""
    return daar_root() / "source"


def daar_oracle_dir() -> Path:
    return daar_root() / "oracle"


def daar_oracle_path() -> Path:
    return daar_oracle_dir() / "daar_oracle.parquet"


def daar_splits_dir() -> Path:
    """Stratified D-AAR train/val/test parquets."""
    return daar_root() / "splits"


def daar_features_dir() -> Path:
    return daar_root() / "features"


def daar_features_path() -> Path:
    return daar_features_dir() / "daar_query_features.parquet"


def daar_embeddings_path() -> Path:
    return daar_features_dir() / "daar_query_embeddings.parquet"


def daar_frames_dir() -> Path:
    return daar_root() / "frames"


def daar_frame_path(split: str) -> Path:
    return daar_frames_dir() / f"daar_{split}_frame.parquet"


def daar_models_dir() -> Path:
    return REPO_ROOT / "models" / "daar"


def daar_cache_dir() -> Path:
    """D-AAR decompose plans only — not ``datasets/decomposer_cache/qce_train_*``."""
    return daar_root() / "cache"


def daar_plans_cache_path() -> Path:
    return daar_cache_dir() / "plans.jsonl"


def daar_traces_dir() -> Path:
    return daar_root() / "traces"


def daar_trace_skeleton_path() -> Path:
    return daar_traces_dir() / "daar_trace_skeleton.parquet"


def daar_simulations_dir() -> Path:
    return daar_root() / "simulations"


def daar_cost_hand_path() -> Path:
    return daar_simulations_dir() / "daar_cost_hand.parquet"


def daar_routing_dir() -> Path:
    return daar_root() / "routing"


def daar_p0_predictions_path(split: str) -> Path:
    return daar_routing_dir() / f"daar_p0_{split}_predictions.parquet"


def daar_p1_predictions_path(split: str) -> Path:
    return daar_routing_dir() / f"daar_p1_{split}_predictions.parquet"


def daar_p1a_predictions_path(split: str) -> Path:
    return daar_routing_dir() / f"daar_p1a_{split}_predictions.parquet"


def daar_decomposed_predictions_path(split: str) -> Path:
    return daar_routing_dir() / f"daar_decomposed_{split}_predictions.parquet"


def daar_predictions_path(variant: str, split: str) -> Path:
    if variant == "p1":
        return daar_p1_predictions_path(split)
    if variant == "p1a":
        return daar_p1a_predictions_path(split)
    if variant == "decomposed":
        return daar_decomposed_predictions_path(split)
    if variant == "p0":
        return daar_p0_predictions_path(split)
    raise ValueError(f"unknown D-AAR variant {variant!r}")


def ensure_daar_dirs() -> None:
    for d in (
        daar_source_dir(),
        daar_oracle_dir(),
        daar_splits_dir(),
        daar_features_dir(),
        daar_frames_dir(),
        daar_cache_dir(),
        daar_traces_dir(),
        daar_simulations_dir(),
        daar_routing_dir(),
        daar_models_dir(),
    ):
        d.mkdir(parents=True, exist_ok=True)


def offline_dataset_dir() -> Path:
    """D-AAR offline execution pool alias (``datasets/daar/source/``)."""
    return daar_source_dir()


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


def eval_heuristics_path(tag: str) -> Path:
    t = tag.strip().lower()
    return qce_features_eval_dir(t) / f"query_heuristics_{t}.parquet"


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


# ── Input datasets (raw / processed / samples) ────────────────────────────────

CONFIG_DATASETS = REPO_ROOT / "input/datasets.yaml"


def data_root() -> Path:
    return REPO_ROOT / "datasets"


def raw_dir(dataset: str | None = None) -> Path:
    base = REPO_ROOT / ("datasets/input/raw" if USE_NEW_DATA_LAYOUT else "datasets/raw")
    return base / dataset if dataset else base


def processed_dir(dataset: str | None = None) -> Path:
    base = REPO_ROOT / ("datasets/input/processed" if USE_NEW_DATA_LAYOUT else "datasets/processed")
    return base / dataset if dataset else base


def train_samples_dir() -> Path:
    if USE_NEW_DATA_LAYOUT:
        return REPO_ROOT / "datasets/input/samples/train"
    return REPO_ROOT / "datasets/train_samples"


def eval_samples_path() -> Path:
    return eval_samples_dir()


def labels_path() -> Path:
    return labels_dir()
