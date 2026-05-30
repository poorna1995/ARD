"""
Router configuration — single source of truth for paths, feature sets, and HGBM variants.

Concept map
-------------

**Primary (deploy default, lowest test regret)**
  Soft ``p_*`` + KL + HGBM → ``PRIMARY_ROUTER_PATH`` (``hgbm_cvec5_emb_soft_kl``).

**Secondary**
  Same soft targets + logistic regression → ``SECONDARY_ROUTER_PATH``.

**Legacy**
  Hard ``oracle_agent`` + HGBM → ``LEGACY_HARD_HGBM_PATH`` (label-F1 baseline).

``ROUTER_MODEL_PATH`` / ``PRODUCTION_ROUTER_PATH`` alias the **primary** artifact.

**Comparison (paper / ablation only)**
  CV-tuned hard HGBM → ``TUNED_ROUTER_PATH``.

**Feature-set id** (``cvec5_emb``, ``cvec7_emb``, …)
  Selects which column groups enter the HGBM. Legacy ``graph_*`` ids normalize via
  ``FEATURE_SET_ALIASES``.

**Experiment id**
  Filename stem for saved ``.joblib`` / ``.json`` (e.g. ``hgbm_cvec5_emb_default``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

REPO_ROOT = Path(__file__).resolve().parents[1]

# ── Labels & agents ───────────────────────────────────────────────────────────

TARGET = "oracle_agent"
SOFT_DOMINANT = "soft_dominant_agent"
AGENTS: tuple[str, ...] = ("react", "cot", "raw", "multiagent")
PROBA_COLS: list[str] = [f"p_{a}" for a in AGENTS]

# ── Data paths ────────────────────────────────────────────────────────────────

_LABELS_ROOT = Path("datasets/train_samples/v1")
SPLIT_CSV: dict[str, Path] = {
    "train": _LABELS_ROOT / "qce_train.csv",
    "val": _LABELS_ROOT / "qce_val.csv",
    "test": _LABELS_ROOT / "qce_internal_test.csv",
}
SPLIT_PARQUET: dict[str, str] = {"train": "train", "val": "val", "test": "test"}
EMBEDDINGS_PARQUET = "datasets/qce_features/query_embeddings_{split}.parquet"
HEURISTICS_PARQUET = "datasets/qce_features/query_heuristics_{split}.parquet"
EMBEDDING_COL_PREFIX = "emb_"
HEURISTIC_COL_PREFIX = "heur_"
HEUR_CALIBRATOR_PATH = REPO_ROOT / "models/qce_heuristics/train_calibrator.json"
SOFT_KL_ABLATION_DIR = REPO_ROOT / "results/experiments/feature_ablation_soft_kl"

TRAIN_NORM_JSON = REPO_ROOT / "models/qce_graph/train_norm.json"
PCA_PATH = REPO_ROOT / "models/query_embeddings/pca_16.joblib"
QCE_FEATURES_DIR = REPO_ROOT / "datasets/qce_features"
DECOMPOSER_CACHE_DIR = REPO_ROOT / "datasets/decomposer_cache"
EVAL_SAMPLES_DIR = REPO_ROOT / "datasets/eval_samples"
TUNE_OUT_DIR = REPO_ROOT / "results/router_tuning"
SEED_STABILITY_DIR = REPO_ROOT / "results/experiments/seed_stability_cvec5_emb"

DATASET_ALIASES: dict[str, str] = {"mmlu": "mmlu_pro"}
DEFAULT_AGENT_MODEL = "gpt-4o-mini"
QCE_DATASETS: tuple[str, ...] = ("math", "hotpot", "musique")

# ── Router roles (soft p_* + KL, cvec5_emb, train n=808) ─────────────────────

PRODUCTION_FEATURE_SET = "cvec5_emb"
PRODUCTION_SUPERVISION = "soft_kl"

PRIMARY_ROUTER_EXPERIMENT_ID = "hgbm_cvec5_emb_soft_kl"
PRIMARY_ROUTER_PATH = (
    REPO_ROOT / "models/router/graph_main" / f"{PRIMARY_ROUTER_EXPERIMENT_ID}.joblib"
)
PRIMARY_CLASSIFIER = "hgbm"

SECONDARY_ROUTER_EXPERIMENT_ID = "logreg_cvec5_emb_soft_kl"
SECONDARY_ROUTER_PATH = (
    REPO_ROOT / "models/router/graph_main" / f"{SECONDARY_ROUTER_EXPERIMENT_ID}.joblib"
)
SECONDARY_CLASSIFIER = "logreg"

# Deploy / orchestrator / benchmark default = primary.
PRODUCTION_ROUTER_EXPERIMENT_ID = PRIMARY_ROUTER_EXPERIMENT_ID
PRODUCTION_ROUTER_PATH = PRIMARY_ROUTER_PATH
PRODUCTION_CLASSIFIER = PRIMARY_CLASSIFIER
# Deploy default: Soft HGBM (primary, lowest internal-test regret).
ROUTER_MODEL_PATH = PRIMARY_ROUTER_PATH  # …/hgbm_cvec5_emb_soft_kl.joblib

ROUTER_ROLE_BY_CLASSIFIER: dict[str, str] = {
    "hgbm": "primary",
    "logreg": "secondary",
}

# Back-compat aliases (older docs said "reference" for soft HGBM).
REFERENCE_SOFT_HGBM_EXPERIMENT_ID = PRIMARY_ROUTER_EXPERIMENT_ID
REFERENCE_SOFT_HGBM_PATH = PRIMARY_ROUTER_PATH
REFERENCE_SOFT_LOGREG_EXPERIMENT_ID = SECONDARY_ROUTER_EXPERIMENT_ID
REFERENCE_SOFT_LOGREG_PATH = SECONDARY_ROUTER_PATH

# Legacy: hard oracle_agent + default HGBM.
LEGACY_HARD_HGBM_EXPERIMENT_ID = "hgbm_cvec5_emb_default"
LEGACY_HARD_HGBM_PATH = (
    REPO_ROOT / "models/router/graph_main" / f"{LEGACY_HARD_HGBM_EXPERIMENT_ID}.joblib"
)

# ``None`` → ``make_pipeline`` defaults (max_depth=6, lr=0.05, max_leaf_nodes=31, …).
PRODUCTION_HGBM_PARAMS: dict[str, Any] | None = None
DEFAULT_HGBM_PARAMS = PRODUCTION_HGBM_PARAMS

SOFT_KL_SEED_STABILITY_DIR = REPO_ROOT / "results/experiments/seed_stability_soft_kl"

# ── Phase 2: selective QCE (opt-in; ROUTER_MODEL_PATH unchanged) ─────────────

SELECTIVE_CHEAP_ROUTER_PATH = (
    REPO_ROOT / "models/router/graph_main/hgbm_emb_only_soft_kl.joblib"
)
SELECTIVE_FULL_ROUTER_PATH = PRIMARY_ROUTER_PATH
SELECTIVE_DECOMPOSE_COST_USD = 0.00025
SELECTIVE_TAU_GRID: tuple[float, ...] = (
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
)
DEFAULT_SELECTIVE_TAU = 0.50
SELECTIVE_QCE_OUT_DIR = REPO_ROOT / "results/experiments/selective_qce"
SELECTIVE_TAU_JSON = SELECTIVE_QCE_OUT_DIR / "recommended_tau.json"

# Live eval: always-{agent} runs under {root}/{folder}/{name}_baseline_{agent}/
LIVE_EVAL_BASELINE_ROOT = REPO_ROOT / "results/orchestrator/graph_main_tuned"

# ── CV-tuned comparison (not production) ──────────────────────────────────────

TUNED_ROUTER_EXPERIMENT_ID = "hgbm_cvec5_emb_tuned"
TUNED_ROUTER_PATH = (
    REPO_ROOT / "models/router/graph_main" / f"{TUNED_ROUTER_EXPERIMENT_ID}.joblib"
)
TUNED_HGBM_PARAMS: dict[str, Any] = {
    "max_depth": 5,
    "learning_rate": 0.03,
    "max_leaf_nodes": 15,
}

# ── Legacy experiment ids (archived artifacts; not production) ────────────────

LEGACY_ROUTER_EXPERIMENT_ID = "hgbm_graph_emb_balanced"
LEGACY_GRAPH_MAIN_EXPERIMENT_ID = "hgbm_graph_main_balanced"
LEGACY_GRAPH_MAIN_DIM_EMB_EXPERIMENT_ID = "hgbm_graph_main_dim_emb_balanced"
LEGACY_ROUTER_V2_EXPERIMENT_ID = "hgbm_graph_emb_v2_balanced"
LEGACY_TUNED_EXPERIMENT_ID = "hgbm_graph_emb_tuned"

# Back-compat aliases used by older scripts
ROUTER_EXPERIMENT_ID = LEGACY_ROUTER_EXPERIMENT_ID
GRAPH_MAIN_EXPERIMENT_ID = LEGACY_GRAPH_MAIN_EXPERIMENT_ID
GRAPH_MAIN_DIM_EMB_EXPERIMENT_ID = LEGACY_GRAPH_MAIN_DIM_EMB_EXPERIMENT_ID
ROUTER_V2_EXPERIMENT_ID = LEGACY_ROUTER_V2_EXPERIMENT_ID
TUNED_EXPERIMENT_ID = LEGACY_TUNED_EXPERIMENT_ID
GRAPH_MAIN_MODEL_PATH = REPO_ROOT / "models/router/graph_main" / f"{LEGACY_GRAPH_MAIN_EXPERIMENT_ID}.joblib"
GRAPH_MAIN_DIM_EMB_MODEL_PATH = LEGACY_HARD_HGBM_PATH
TUNED_MODEL_PATH = REPO_ROOT / "models/router" / f"{LEGACY_TUNED_EXPERIMENT_ID}.joblib"

# ── HGBM tuning grid ──────────────────────────────────────────────────────────

HGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
}

# ── Feature sets ──────────────────────────────────────────────────────────────

FeatureSet = Literal[
    "cvec5_emb_ds",
    "cvec5_emb",
    "cvec5_ds",
    "cvec5",
    "cvec5_emb_trust",
    "cvec7_emb_ds",
    "cvec7_emb",
    "cvec7_ds",
    "cvec7",
    "cvec7_emb_trust",
    "emb_ds",
    "emb_only",
    "dataset_only",
    "graph_main",
    "graph_emb",
    "graph_emb_nods",
    "graph",
    "graph_nods",
    "graph_emb_v2",
    "graph_emb_v2_nods",
    "graph_v2",
    "graph_v2_nods",
    "emb",
    "emb_nods",
    "dataset",
]

CANONICAL_FEATURE_SETS: tuple[str, ...] = (
    "cvec5_emb_ds",
    "cvec5_emb",
    "cvec5_ds",
    "cvec5",
    "cvec5_emb_trust",
    "cvec7_emb_ds",
    "cvec7_emb",
    "cvec7_ds",
    "cvec7",
    "cvec7_emb_trust",
    "emb_ds",
    "emb_only",
    "heur_emb",
    "dataset_only",
)

FEATURE_SET_ALIASES: dict[str, str] = {
    "graph_main": "cvec5_emb_ds",
    "graph_emb": "cvec5_emb_ds",
    "graph_emb_nods": "cvec5_emb",
    "graph": "cvec5_ds",
    "graph_nods": "cvec5",
    "graph_emb_v2": "cvec7_emb_ds",
    "graph_emb_v2_nods": "cvec7_emb",
    "graph_v2": "cvec7_ds",
    "graph_v2_nods": "cvec7",
    "emb": "emb_ds",
    "emb_nods": "emb_only",
    "dataset": "dataset_only",
}

FEATURE_SET_CLI_CHOICES: tuple[str, ...] = CANONICAL_FEATURE_SETS + tuple(FEATURE_SET_ALIASES.keys())


@dataclass(frozen=True)
class FeatureSetSpec:
    """Which column groups a feature-set id includes."""

    cvec_version: int | None  # 1 → dim_*, 2 → dim7_*, None → no C(Q)
    use_emb: bool
    use_dataset: bool
    use_trust: bool
    use_heur: bool = False


FEATURE_SET_SPECS: dict[str, FeatureSetSpec] = {
    "cvec5_emb_ds": FeatureSetSpec(1, True, True, False, False),
    "cvec5_emb": FeatureSetSpec(1, True, False, False, False),
    "cvec5_emb_trust": FeatureSetSpec(1, True, False, True, False),
    "cvec5_ds": FeatureSetSpec(1, False, True, False, False),
    "cvec5": FeatureSetSpec(1, False, False, False, False),
    "cvec7_emb_ds": FeatureSetSpec(2, True, True, False, False),
    "cvec7_emb": FeatureSetSpec(2, True, False, False, False),
    "cvec7_emb_trust": FeatureSetSpec(2, True, False, True, False),
    "cvec7_ds": FeatureSetSpec(2, False, True, False, False),
    "cvec7": FeatureSetSpec(2, False, False, False, False),
    "emb_ds": FeatureSetSpec(None, True, True, False, False),
    "emb_only": FeatureSetSpec(None, True, False, False, False),
    "heur_emb": FeatureSetSpec(None, True, False, False, True),
    "dataset_only": FeatureSetSpec(None, False, True, False, False),
}

CVEC5_FEATURE_SETS = frozenset(k for k, s in FEATURE_SET_SPECS.items() if s.cvec_version == 1)
CVEC7_FEATURE_SETS = frozenset(k for k, s in FEATURE_SET_SPECS.items() if s.cvec_version == 2)
TRUST_FEATURE_SETS = frozenset(k for k, s in FEATURE_SET_SPECS.items() if s.use_trust)

# experiment_id when saving ad-hoc ``routing train`` (non-production sets)
EXPERIMENT_ID_BY_FEATURE_SET: dict[str, str] = {
    "cvec5_emb_ds": LEGACY_GRAPH_MAIN_EXPERIMENT_ID,
    "cvec5_emb": LEGACY_HARD_HGBM_EXPERIMENT_ID,
    "cvec5_emb_trust": "hgbm_cvec5_emb_trust_balanced",
    "cvec7_emb_ds": LEGACY_ROUTER_V2_EXPERIMENT_ID,
    "cvec7_emb": "hgbm_cvec7_emb_balanced",
    "cvec7_emb_trust": "hgbm_cvec7_emb_trust_balanced",
    "emb_only": "hgbm_emb_only_soft_kl",
    "heur_emb": "hgbm_heur_emb_soft_kl",
}


def soft_kl_experiment_id(classifier: str, feature_set: str) -> str:
    """Soft-KL checkpoint stem, e.g. ``hgbm_heur_emb_soft_kl``."""
    fs = normalize_feature_set(feature_set)
    if fs == PRODUCTION_FEATURE_SET and classifier == "hgbm":
        return PRIMARY_ROUTER_EXPERIMENT_ID
    return f"{classifier}_{fs}_soft_kl"


def soft_kl_out_dir(feature_set: str) -> Path:
    fs = normalize_feature_set(feature_set)
    if fs == PRODUCTION_FEATURE_SET:
        return REPO_ROOT / "results/experiments/soft_kl_cvec5_emb"
    return SOFT_KL_ABLATION_DIR / fs


def normalize_feature_set(feature_set: str) -> str:
    return FEATURE_SET_ALIASES.get(feature_set, feature_set)


def feature_set_spec(feature_set: str) -> FeatureSetSpec:
    fs = normalize_feature_set(feature_set)
    if fs not in FEATURE_SET_SPECS:
        raise ValueError(f"unknown feature_set {feature_set!r} (normalized={fs!r})")
    return FEATURE_SET_SPECS[fs]


def feature_set_needs_embeddings(feature_set: str) -> bool:
    return feature_set_spec(feature_set).use_emb


def feature_set_needs_heuristics(feature_set: str) -> bool:
    return feature_set_spec(feature_set).use_heur


def feature_set_cvec_version(feature_set: str) -> int | None:
    return feature_set_spec(feature_set).cvec_version


def model_dir_for_feature_set(feature_set: str) -> Path:
    fs = normalize_feature_set(feature_set)
    if fs in CVEC5_FEATURE_SETS:
        return REPO_ROOT / "models/router/graph_main"
    if fs in CVEC7_FEATURE_SETS:
        return REPO_ROOT / "models/router/cvec7"
    return REPO_ROOT / "models/router"


def experiment_id_for_feature_set(feature_set: str) -> str:
    fs = normalize_feature_set(feature_set)
    return EXPERIMENT_ID_BY_FEATURE_SET.get(fs, LEGACY_ROUTER_EXPERIMENT_ID)


def model_path_for_feature_set(feature_set: str) -> Path:
    fs = normalize_feature_set(feature_set)
    if fs == PRODUCTION_FEATURE_SET:
        return LEGACY_HARD_HGBM_PATH
    return model_dir_for_feature_set(fs) / f"{experiment_id_for_feature_set(fs)}.joblib"


def is_production_feature_set(feature_set: str) -> bool:
    return normalize_feature_set(feature_set) == PRODUCTION_FEATURE_SET


# ── Ablation tables (default HGBM; same splits) ────────────────────────────────

@dataclass(frozen=True)
class AblationCase:
    label: str
    feature_set: str
    description: str = ""


CVEC5_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase("cvec5+emb+dataset", "cvec5_emb_ds", "5-dim C(Q) + emb + dataset"),
    AblationCase("cvec5+emb", "cvec5_emb", "5-dim C(Q) + emb (production features)"),
    AblationCase("cvec5+dataset", "cvec5_ds", "5-dim C(Q) + dataset"),
    AblationCase("cvec5-only", "cvec5", "5-dim C(Q) only"),
    AblationCase("emb-only", "emb_only", "PCA-16 embedding only"),
    AblationCase("heur+emb", "heur_emb", "heuristic v2 + PCA-16 (no decompose)"),
    AblationCase("dataset-only", "dataset_only", "dataset one-hot only"),
)

CVEC7_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase("cvec7+emb+dataset", "cvec7_emb_ds", "7-dim C(Q) + emb + dataset"),
    AblationCase("cvec7+emb", "cvec7_emb", "7-dim C(Q) + emb"),
    AblationCase("cvec7+dataset", "cvec7_ds", "7-dim C(Q) + dataset"),
    AblationCase("cvec7-only", "cvec7", "7-dim C(Q) only"),
    AblationCase("emb-only", "emb_only", "PCA-16 embedding only"),
    AblationCase("dataset-only", "dataset_only", "dataset one-hot only"),
)

TRUST_ABLATION_CASES: tuple[AblationCase, ...] = (
    AblationCase("cvec5+emb (baseline)", "cvec5_emb", "production feature columns"),
    AblationCase("cvec5+emb+trust", "cvec5_emb_trust", "+ plan-trust scalars"),
    AblationCase("cvec7+emb (baseline)", "cvec7_emb", "7-dim + emb"),
    AblationCase("cvec7+emb+trust", "cvec7_emb_trust", "7-dim + emb + trust"),
)
