"""
Configuration layout: **common** → **global_config** → **local**.

Import patterns::

    from config.common import REPO_ROOT, USE_NEW_DATA_LAYOUT
    from config.global_config.paths import eval_samples_dir, raw_dir, processed_dir
    from config.local.constants import DS, TOOLS, PLAN, DIMS, AGENTS
"""

from config.common import REPO_ROOT, USE_NEW_DATA_LAYOUT, load_yaml
from config.local.constants import AGENTS, DIMS, DS, TOOLS
from config.global_config.runtime import resolve_output_root, resolve_router_path
from config.global_config.paths import (
    CONFIG_DATASETS,
    complexity_record_path,
    decomposer_cache_dir,
    embeddings_parquet_path,
    eval_samples_dir,
    labels_dir,
    live_eval_baseline_root,
    oracle_results_csv,
    orchestrator_default_root,
    processed_dir,
    raw_dir,
    router_production_dir,
    router_selective_dir,
    train_samples_dir,
)

__all__ = [
    "AGENTS",
    "CONFIG_DATASETS",
    "DIMS",
    "DS",
    "REPO_ROOT",
    "TOOLS",
    "USE_NEW_DATA_LAYOUT",
    "complexity_record_path",
    "decomposer_cache_dir",
    "embeddings_parquet_path",
    "eval_samples_dir",
    "labels_dir",
    "live_eval_baseline_root",
    "load_yaml",
    "oracle_results_csv",
    "orchestrator_default_root",
    "resolve_output_root",
    "processed_dir",
    "raw_dir",
    "resolve_router_path",
    "router_production_dir",
    "router_selective_dir",
    "train_samples_dir",
]
