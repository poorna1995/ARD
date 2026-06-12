"""On-disk paths for INPUT datasets (raw, processed, samples)."""

from __future__ import annotations

from pathlib import Path

from config.common import REPO_ROOT, USE_NEW_DATA_LAYOUT
from config.global_config.paths import eval_samples_dir, labels_dir, oracle_results_csv

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


__all__ = [
    "CONFIG_DATASETS",
    "REPO_ROOT",
    "data_root",
    "eval_samples_dir",
    "eval_samples_path",
    "labels_dir",
    "labels_path",
    "oracle_results_csv",
    "processed_dir",
    "raw_dir",
    "train_samples_dir",
]
