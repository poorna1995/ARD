"""
INPUT zone — datasets, prompts, and everything needed before routing/training.

Disk layout (with ``RESEARCH_USE_NEW_PATHS=1``)::

    datasets/input/raw/{dataset}/
    datasets/input/processed/{dataset}/
    datasets/input/samples/eval/*.parquet
    datasets/input/samples/train/*.parquet
    datasets/input/labels/v1/qce_*.csv

Code modules: ``input.loaders``, ``input.samples``, ``input.preprocess``,
``input.prompts``, ``input.download``.

Config: ``input/datasets.yaml`` (benchmark download settings).
"""

from input.loaders import REGISTRY, get_loader
from config.global_config.paths import (
    CONFIG_DATASETS,
    data_root,
    eval_samples_dir,
    labels_dir,
    processed_dir,
    raw_dir,
    train_samples_dir,
)

__all__ = [
    "CONFIG_DATASETS",
    "REGISTRY",
    "data_root",
    "eval_samples_dir",
    "get_loader",
    "labels_dir",
    "processed_dir",
    "raw_dir",
    "train_samples_dir",
]
