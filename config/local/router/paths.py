"""On-disk paths for router training, features, and experiments."""

from __future__ import annotations

from pathlib import Path

from config.common import REPO_ROOT
from config.global_config import paths as data_paths
from config.local.constants.datasets import DS

_PROD_ROUTER_DIR = data_paths.router_production_dir()
_MODELS = REPO_ROOT / "models"
_ROUTER_MODELS = _MODELS / "router"
_RESULTS = REPO_ROOT / "results"
_EXPERIMENTS = _RESULTS / "experiments"
_SPLITS = ("train", "val", "test")


def _joblib(stem: str, *, root: Path | None = None) -> Path:
    return (root or _PROD_ROUTER_DIR) / f"{stem}.joblib"


SPLIT_CSV: dict[str, Path] = {s: data_paths.split_csv_path(s) for s in _SPLITS}
SPLIT_PARQUET: dict[str, str] = {s: s for s in _SPLITS}

EMBEDDINGS_PARQUET = data_paths.legacy_embeddings_template()
HEURISTICS_PARQUET = data_paths.legacy_heuristics_template()
EMBEDDING_COL_PREFIX = "emb_"
HEURISTIC_COL_PREFIX = "heur_"
HEUR_CALIBRATOR_PATH = _MODELS / "qce_heuristics/train_calibrator.json"
SOFT_KL_ABLATION_DIR = _EXPERIMENTS / "feature_ablation_soft_kl"
TRAIN_NORM_JSON = _MODELS / "qce_graph/train_norm.json"
PCA_PATH = _MODELS / "query_embeddings/pca_16.joblib"
QCE_FEATURES_DIR = data_paths.qce_features_split_dir()
DECOMPOSER_CACHE_DIR = data_paths.decomposer_cache_dir()
EVAL_SAMPLES_DIR = data_paths.eval_samples_dir()
TUNE_OUT_DIR = _RESULTS / "router_tuning"
SEED_STABILITY_DIR = _EXPERIMENTS / "seed_stability_cvec5_emb"
BENCHMARK_OUT_ROOT = _EXPERIMENTS / "soft_hgbm_benchmark"
ROUTES_DIR = BENCHMARK_OUT_ROOT / "routes"

DATASET_ALIASES: dict[str, str] = dict(DS.aliases)
QCE_DATASETS: tuple[str, ...] = DS.router_train
LIVE_EVAL_BASELINE_ROOT = data_paths.live_eval_baseline_root()

__all__ = [
    "BENCHMARK_OUT_ROOT",
    "DATASET_ALIASES",
    "DECOMPOSER_CACHE_DIR",
    "EMBEDDINGS_PARQUET",
    "EMBEDDING_COL_PREFIX",
    "EVAL_SAMPLES_DIR",
    "HEURISTICS_PARQUET",
    "HEUR_CALIBRATOR_PATH",
    "HEURISTIC_COL_PREFIX",
    "LIVE_EVAL_BASELINE_ROOT",
    "PCA_PATH",
    "QCE_DATASETS",
    "QCE_FEATURES_DIR",
    "ROUTES_DIR",
    "SEED_STABILITY_DIR",
    "SOFT_KL_ABLATION_DIR",
    "SPLIT_CSV",
    "SPLIT_PARQUET",
    "TRAIN_NORM_JSON",
    "TUNE_OUT_DIR",
    "_EXPERIMENTS",
    "_PROD_ROUTER_DIR",
    "_ROUTER_MODELS",
    "_joblib",
]
