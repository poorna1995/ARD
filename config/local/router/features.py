"""Router feature-set ids, specs, and path helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from config.local.router.legacy import (
    LEGACY_GRAPH_MAIN_EXPERIMENT_ID,
    LEGACY_ROUTER_EXPERIMENT_ID,
)
from config.local.router.paths import _PROD_ROUTER_DIR, _ROUTER_MODELS, _joblib
from config.local.router.production import (
    LEGACY_HARD_HGBM_EXPERIMENT_ID,
    LEGACY_HARD_HGBM_PATH,
    PRODUCTION_FEATURE_SET,
    _soft_kl_stem,
)

# suffix None → bare cvec5; flags = use_emb, use_dataset, use_trust, use_heur
_CVEC_SUFFIX_FLAGS: tuple[tuple[str | None, bool, bool, bool, bool], ...] = (
    ("emb_ds", True, True, False, False),
    ("emb", True, False, False, False),
    ("emb_trust", True, False, True, False),
    ("ds", False, True, False, False),
    (None, False, False, False, False),
)
_NON_CVEC_ROWS: tuple[tuple[str, int | None, bool, bool, bool, bool], ...] = (
    ("emb_ds", None, True, True, False, False),
    ("emb_only", None, True, False, False, False),
    ("heur_emb", None, True, False, False, True),
    ("dataset_only", None, False, True, False, False),
)
_GRAPH_ALIAS_PATTERN: tuple[tuple[str, str | None], ...] = (
    ("graph_main", "emb_ds"),
    ("graph_emb", "emb_ds"),
    ("graph_emb_nods", "emb"),
    ("graph", "ds"),
    ("graph_nods", None),
)
_STATIC_FEATURE_ALIASES: tuple[tuple[str, str], ...] = (
    ("emb", "emb_ds"),
    ("emb_nods", "emb_only"),
    ("dataset", "dataset_only"),
)


def _cvec_name(suffix: str | None) -> str:
    return "cvec5" if suffix is None else f"cvec5_{suffix}"


def _cvec_rows() -> tuple[tuple[str, int | None, bool, bool, bool, bool], ...]:
    return tuple(
        (_cvec_name(suffix), 1, emb, ds, trust, heur)
        for suffix, emb, ds, trust, heur in _CVEC_SUFFIX_FLAGS
    )


_FEATURE_SET_ROWS: tuple[tuple[str, int | None, bool, bool, bool, bool], ...] = (
    _cvec_rows() + _NON_CVEC_ROWS
)

CANONICAL_FEATURE_SETS: tuple[str, ...] = tuple(row[0] for row in _FEATURE_SET_ROWS)

FEATURE_SET_ALIASES: dict[str, str] = dict(
    _STATIC_FEATURE_ALIASES
    + tuple((alias, _cvec_name(sfx)) for alias, sfx in _GRAPH_ALIAS_PATTERN)
)

FEATURE_SET_CLI_CHOICES: tuple[str, ...] = CANONICAL_FEATURE_SETS + tuple(FEATURE_SET_ALIASES)

FeatureSet = Literal[
    "cvec5_emb_ds",
    "cvec5_emb",
    "cvec5_ds",
    "cvec5",
    "cvec5_emb_trust",
    "emb_ds",
    "emb_only",
    "dataset_only",
    "graph_main",
    "graph_emb",
    "graph_emb_nods",
    "graph",
    "graph_nods",
    "emb",
    "emb_nodos",
    "dataset",
]


@dataclass(frozen=True)
class FeatureSetSpec:
    """Which column groups a feature-set id includes."""

    cvec_version: int | None  # 1 → dim_*, None → no C(Q)
    use_emb: bool
    use_dataset: bool
    use_trust: bool
    use_heur: bool = False


FEATURE_SET_SPECS: dict[str, FeatureSetSpec] = {
    row[0]: FeatureSetSpec(row[1], row[2], row[3], row[4], row[5])
    for row in _FEATURE_SET_ROWS
}

CVEC5_FEATURE_SETS = frozenset(k for k, s in FEATURE_SET_SPECS.items() if s.cvec_version == 1)
TRUST_FEATURE_SETS = frozenset(k for k, s in FEATURE_SET_SPECS.items() if s.use_trust)


def _balanced(stem: str) -> str:
    return f"hgbm_{stem}_balanced"


EXPERIMENT_ID_BY_FEATURE_SET: dict[str, str] = {
    "cvec5_emb_ds": LEGACY_GRAPH_MAIN_EXPERIMENT_ID,
    "cvec5_emb": LEGACY_HARD_HGBM_EXPERIMENT_ID,
    "cvec5_emb_trust": _balanced("cvec5_emb_trust"),
    "emb_only": _soft_kl_stem("hgbm", "emb_only"),
    "heur_emb": _soft_kl_stem("hgbm", "heur_emb"),
}


def normalize_feature_set(feature_set: str) -> str:
    return FEATURE_SET_ALIASES.get(feature_set, feature_set)


def _resolve_fs(feature_set: str) -> tuple[str, FeatureSetSpec]:
    fs = normalize_feature_set(feature_set)
    try:
        return fs, FEATURE_SET_SPECS[fs]
    except KeyError as exc:
        raise ValueError(
            f"unknown feature_set {feature_set!r} (normalized={fs!r})"
        ) from exc


def feature_set_spec(feature_set: str) -> FeatureSetSpec:
    return _resolve_fs(feature_set)[1]


def feature_set_needs_embeddings(feature_set: str) -> bool:
    return _resolve_fs(feature_set)[1].use_emb


def feature_set_needs_heuristics(feature_set: str) -> bool:
    return _resolve_fs(feature_set)[1].use_heur


def feature_set_cvec_version(feature_set: str) -> int | None:
    return _resolve_fs(feature_set)[1].cvec_version


def model_dir_for_feature_set(feature_set: str) -> Path:
    fs = normalize_feature_set(feature_set)
    if fs in CVEC5_FEATURE_SETS:
        return _PROD_ROUTER_DIR
    return _ROUTER_MODELS


def experiment_id_for_feature_set(feature_set: str) -> str:
    return EXPERIMENT_ID_BY_FEATURE_SET.get(
        normalize_feature_set(feature_set), LEGACY_ROUTER_EXPERIMENT_ID
    )


def model_path_for_feature_set(feature_set: str) -> Path:
    fs = normalize_feature_set(feature_set)
    if fs == PRODUCTION_FEATURE_SET:
        return LEGACY_HARD_HGBM_PATH
    root = model_dir_for_feature_set(fs)
    return _joblib(experiment_id_for_feature_set(fs), root=root)


def is_production_feature_set(feature_set: str) -> bool:
    return normalize_feature_set(feature_set) == PRODUCTION_FEATURE_SET


__all__ = [
    "CANONICAL_FEATURE_SETS",
    "CVEC5_FEATURE_SETS",
    "EXPERIMENT_ID_BY_FEATURE_SET",
    "FEATURE_SET_ALIASES",
    "FEATURE_SET_CLI_CHOICES",
    "FEATURE_SET_SPECS",
    "FeatureSet",
    "FeatureSetSpec",
    "TRUST_FEATURE_SETS",
    "experiment_id_for_feature_set",
    "feature_set_cvec_version",
    "feature_set_needs_embeddings",
    "feature_set_needs_heuristics",
    "feature_set_spec",
    "is_production_feature_set",
    "model_dir_for_feature_set",
    "model_path_for_feature_set",
    "normalize_feature_set",
]
