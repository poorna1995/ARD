"""Production router artifacts and deploy defaults."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.global_config import paths as data_paths
from config.local.router.paths import SOFT_KL_ABLATION_DIR, _EXPERIMENTS, _joblib

PRODUCTION_FEATURE_SET = "cvec5_emb"
PRODUCTION_SUPERVISION = "soft_kl"
PRIMARY_CLASSIFIER, SECONDARY_CLASSIFIER = "hgbm", "logreg"
PRODUCTION_HGBM_PARAMS: dict[str, Any] | None = None
DEFAULT_HGBM_PARAMS = PRODUCTION_HGBM_PARAMS
TUNED_HGBM_PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.03,
    "max_leaf_nodes": 15,
}
SELECTIVE_DECOMPOSE_COST_USD = 0.00025
SELECTIVE_TAU_GRID = tuple(round(0.30 + 0.05 * i, 2) for i in range(9))
DEFAULT_SELECTIVE_TAU = 0.50


def _soft_kl_stem(classifier: str, feature_set: str = PRODUCTION_FEATURE_SET) -> str:
    return f"{classifier}_{feature_set}_soft_kl"


def _router_path(
    classifier: str,
    feature_set: str = PRODUCTION_FEATURE_SET,
    *,
    root: Path | None = None,
) -> Path:
    return _joblib(_soft_kl_stem(classifier, feature_set), root=root)


PRIMARY_ROUTER_EXPERIMENT_ID = _soft_kl_stem(PRIMARY_CLASSIFIER)
SECONDARY_ROUTER_EXPERIMENT_ID = _soft_kl_stem(SECONDARY_CLASSIFIER)
PRIMARY_ROUTER_PATH = _router_path(PRIMARY_CLASSIFIER)
SECONDARY_ROUTER_PATH = _router_path(SECONDARY_CLASSIFIER)

PRODUCTION_ROUTER_EXPERIMENT_ID = PRIMARY_ROUTER_EXPERIMENT_ID
PRODUCTION_ROUTER_PATH = PRIMARY_ROUTER_PATH
PRODUCTION_CLASSIFIER = PRIMARY_CLASSIFIER
ROUTER_MODEL_PATH = PRIMARY_ROUTER_PATH
REFERENCE_SOFT_HGBM_EXPERIMENT_ID = PRIMARY_ROUTER_EXPERIMENT_ID
REFERENCE_SOFT_HGBM_PATH = PRIMARY_ROUTER_PATH
REFERENCE_SOFT_LOGREG_EXPERIMENT_ID = SECONDARY_ROUTER_EXPERIMENT_ID
REFERENCE_SOFT_LOGREG_PATH = SECONDARY_ROUTER_PATH
SELECTIVE_FULL_ROUTER_PATH = PRIMARY_ROUTER_PATH
ROUTER_ROLE_BY_CLASSIFIER = {PRIMARY_CLASSIFIER: "primary", SECONDARY_CLASSIFIER: "secondary"}

LEGACY_HARD_HGBM_EXPERIMENT_ID = f"hgbm_{PRODUCTION_FEATURE_SET}_default"
LEGACY_HARD_HGBM_PATH = _joblib(LEGACY_HARD_HGBM_EXPERIMENT_ID)
SOFT_KL_SEED_STABILITY_DIR = _EXPERIMENTS / "seed_stability_soft_kl"
SELECTIVE_CHEAP_ROUTER_PATH = _router_path("hgbm", "emb_only", root=data_paths.router_selective_dir())
SELECTIVE_QCE_OUT_DIR = _EXPERIMENTS / "selective_qce"
SELECTIVE_TAU_JSON = SELECTIVE_QCE_OUT_DIR / "recommended_tau.json"
TUNED_ROUTER_EXPERIMENT_ID = f"hgbm_{PRODUCTION_FEATURE_SET}_tuned"
TUNED_ROUTER_PATH = _joblib(TUNED_ROUTER_EXPERIMENT_ID)


def _norm_fs(feature_set: str) -> str:
    from config.local.router.features import normalize_feature_set

    return normalize_feature_set(feature_set)


def soft_kl_experiment_id(classifier: str, feature_set: str) -> str:
    """Soft-KL checkpoint stem, e.g. ``hgbm_heur_emb_soft_kl``."""
    fs = _norm_fs(feature_set)
    if fs == PRODUCTION_FEATURE_SET and classifier == PRIMARY_CLASSIFIER:
        return PRIMARY_ROUTER_EXPERIMENT_ID
    return _soft_kl_stem(classifier, fs)


def soft_kl_out_dir(feature_set: str) -> Path:
    fs = _norm_fs(feature_set)
    if fs == PRODUCTION_FEATURE_SET:
        return _EXPERIMENTS / "soft_kl_cvec5_emb"
    return SOFT_KL_ABLATION_DIR / fs


__all__ = [
    "DEFAULT_HGBM_PARAMS",
    "DEFAULT_SELECTIVE_TAU",
    "LEGACY_HARD_HGBM_EXPERIMENT_ID",
    "LEGACY_HARD_HGBM_PATH",
    "PRIMARY_CLASSIFIER",
    "PRIMARY_ROUTER_EXPERIMENT_ID",
    "PRIMARY_ROUTER_PATH",
    "PRODUCTION_CLASSIFIER",
    "PRODUCTION_FEATURE_SET",
    "PRODUCTION_HGBM_PARAMS",
    "PRODUCTION_ROUTER_EXPERIMENT_ID",
    "PRODUCTION_ROUTER_PATH",
    "PRODUCTION_SUPERVISION",
    "REFERENCE_SOFT_HGBM_EXPERIMENT_ID",
    "REFERENCE_SOFT_HGBM_PATH",
    "REFERENCE_SOFT_LOGREG_EXPERIMENT_ID",
    "REFERENCE_SOFT_LOGREG_PATH",
    "ROUTER_MODEL_PATH",
    "ROUTER_ROLE_BY_CLASSIFIER",
    "SECONDARY_CLASSIFIER",
    "SECONDARY_ROUTER_EXPERIMENT_ID",
    "SECONDARY_ROUTER_PATH",
    "SELECTIVE_CHEAP_ROUTER_PATH",
    "SELECTIVE_DECOMPOSE_COST_USD",
    "SELECTIVE_FULL_ROUTER_PATH",
    "SELECTIVE_QCE_OUT_DIR",
    "SELECTIVE_TAU_GRID",
    "SELECTIVE_TAU_JSON",
    "SOFT_KL_SEED_STABILITY_DIR",
    "TUNED_HGBM_PARAMS",
    "TUNED_ROUTER_EXPERIMENT_ID",
    "TUNED_ROUTER_PATH",
    "_soft_kl_stem",
    "soft_kl_experiment_id",
    "soft_kl_out_dir",
]
