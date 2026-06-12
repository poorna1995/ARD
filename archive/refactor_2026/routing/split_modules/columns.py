"""
Feature column resolution for router training and inference.

IN:  merged QCE / eval feature DataFrames
MID: detect dim_* / emb_* / heur_* / trust columns · path helpers for split parquets
OUT: ordered feature column lists · ``overall`` complexity scalar
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config.paths import REPO_ROOT, embeddings_parquet_path, heuristics_parquet_path
from routing.config import (
    EMBEDDING_COL_PREFIX,
    FeatureSet,
    HEURISTIC_COL_PREFIX,
    PRODUCTION_FEATURE_SET,
    SPLIT_PARQUET,
    feature_set_spec,
    normalize_feature_set,
)

try:
    from qce.complexity import C_VECTOR_COLS as _C_VECTOR_COLS
    from qce.complexity import TRUST_SCALAR_COLS as _TRUST_SCALAR_COLS
except ImportError:
    _C_VECTOR_COLS = (
        "dim_structural",
        "dim_reasoning",
        "dim_evidence",
        "dim_tool",
        "dim_coordination_uncertainty",
    )
    _TRUST_SCALAR_COLS = (
        "plan_trust",
        "verify_fraction",
        "terminal_sink_ok",
        "sink_intermediate_risk",
    )

_C_VECTOR_COLS_V7 = (
    "dim7_structural",
    "dim7_compositional",
    "dim7_retrieval",
    "dim7_execution",
    "dim7_coordination",
    "dim7_verification",
    "dim7_uncertainty",
)


def query_embeddings_path(split: str, *, root: Path | None = None) -> Path:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    if root is not None:
        rel = embeddings_parquet_path(split).relative_to(REPO_ROOT)
        return Path(root) / rel
    return embeddings_parquet_path(split)


def query_heuristics_path(split: str, *, root: Path | None = None) -> Path:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    if root is not None:
        rel = heuristics_parquet_path(split).relative_to(REPO_ROOT)
        return Path(root) / rel
    return heuristics_parquet_path(split)


def embedding_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c.startswith(EMBEDDING_COL_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
    ]

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len(EMBEDDING_COL_PREFIX) :]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(cols, key=_key)


def load_query_embeddings(split: str, *, root: Path | None = None) -> pd.DataFrame:
    path = query_embeddings_path(split, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"missing query embeddings: {path}")
    emb = pd.read_parquet(path)
    if "training_id" not in emb.columns:
        raise ValueError(f"{path}: missing training_id")
    return emb[["training_id", *embedding_feature_cols(emb)]].copy()


def heuristic_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ``heur_*`` columns in canonical build order when complete."""
    present = [
        c
        for c in df.columns
        if c.startswith(HEURISTIC_COL_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not present:
        return []

    try:
        from routing.features.heuristics import HEUR_ROUTER_COLS

        ordered = [c for c in HEUR_ROUTER_COLS if c in df.columns]
        if len(ordered) == len(HEUR_ROUTER_COLS):
            return ordered
        missing = [c for c in HEUR_ROUTER_COLS if c not in df.columns]
        raise ValueError(
            f"heuristic parquet missing {len(missing)} columns (e.g. {missing[:3]}); "
            "re-run scripts/build_query_heuristics.py --split all"
        )
    except ValueError:
        raise
    except Exception:
        pass

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len(HEURISTIC_COL_PREFIX) :]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(present, key=_key)


def load_query_heuristics(split: str, *, root: Path | None = None) -> pd.DataFrame:
    path = query_heuristics_path(split, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"missing query heuristics: {path}")
    heur = pd.read_parquet(path)
    if "training_id" not in heur.columns:
        raise ValueError(f"{path}: missing training_id")
    return heur[["training_id", *heuristic_feature_cols(heur)]].copy()


def c_vector_feature_cols(df: pd.DataFrame, *, version: int = 1) -> list[str]:
    cols = _C_VECTOR_COLS_V7 if version == 2 else _C_VECTOR_COLS
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def trust_scalar_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in _TRUST_SCALAR_COLS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]


def router_feature_cols(
    df: pd.DataFrame, feature_set: FeatureSet = PRODUCTION_FEATURE_SET
) -> list[str]:
    """Resolve HGBM input columns for a feature-set id (see ``FEATURE_SET_SPECS``)."""
    spec = feature_set_spec(normalize_feature_set(feature_set))
    cols: list[str] = []

    if spec.cvec_version is not None:
        graph = c_vector_feature_cols(df, version=spec.cvec_version)
        label = "dim7_*" if spec.cvec_version == 2 else "dim_*"
        if not graph:
            raise ValueError(f"missing {label} — rebuild complexity_record_*.parquet")
        cols.extend(graph)

    if spec.use_emb:
        emb = embedding_feature_cols(df)
        if not emb:
            raise ValueError("missing emb_* — run scripts/build_query_embeddings.py --split all")
        cols.extend(emb)

    if spec.use_trust:
        trust = trust_scalar_feature_cols(df)
        if not trust:
            raise ValueError(
                f"missing trust scalars {_TRUST_SCALAR_COLS} — rebuild complexity_record_*.parquet"
            )
        cols.extend(trust)

    if spec.use_heur:
        heur = heuristic_feature_cols(df)
        if not heur:
            raise ValueError(
                "missing heur_* — run: uv run python scripts/build_query_heuristics.py --split all"
            )
        cols.extend(heur)

    if spec.use_dataset:
        if "dataset" not in df.columns:
            raise ValueError("missing dataset column")
        cols.append("dataset")

    return cols


def validate_feature_set_data(df: pd.DataFrame, feature_set: str) -> None:
    """Ensure parquet columns exist for ``feature_set`` (call after ``load_split``)."""
    router_feature_cols(df, normalize_feature_set(feature_set))


resolve_feature_cols = router_feature_cols


def overall_complexity(row: pd.Series | dict[str, Any]) -> float | None:
    vals: list[float] = []
    for c in _C_VECTOR_COLS:
        v = row.get(c) if isinstance(row, dict) else row.get(c, None)
        if v is not None and pd.notna(v):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else None
