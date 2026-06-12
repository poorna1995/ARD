"""Re-export — use ``config.local.constants.datasets``."""

from config.local.constants.datasets import (
    DATASET_DISPLAY,
    DATASET_DISPLAY_NAMES,
    DISCUSSION_DATASET_ORDER,
    DS,
    Datasets,
    disk_dir_name,
    eval_parquet_stem,
    resolve_dataset_name,
)

__all__ = [
    "DATASET_DISPLAY",
    "DATASET_DISPLAY_NAMES",
    "DISCUSSION_DATASET_ORDER",
    "DS",
    "Datasets",
    "disk_dir_name",
    "eval_parquet_stem",
    "resolve_dataset_name",
]
