"""Shared configuration (paths, settings). Router constants remain in routing.config."""

from config.paths import (
    REPO_ROOT,
    USE_NEW_DATA_LAYOUT,
    complexity_record_path,
    decomposer_cache_dir,
    embeddings_parquet_path,
    eval_samples_dir,
    labels_dir,
    live_eval_baseline_root,
    oracle_results_csv,
    orchestrator_default_root,
    router_production_dir,
    router_selective_dir,
)
from config.settings import output_root, router_model_path

__all__ = [
    "REPO_ROOT",
    "USE_NEW_DATA_LAYOUT",
    "complexity_record_path",
    "decomposer_cache_dir",
    "embeddings_parquet_path",
    "eval_samples_dir",
    "labels_dir",
    "live_eval_baseline_root",
    "oracle_results_csv",
    "orchestrator_default_root",
    "output_root",
    "router_model_path",
    "router_production_dir",
    "router_selective_dir",
]
