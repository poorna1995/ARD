"""
QCE split and eval-sample data loaders.

IN:  QCE label CSVs (universe A) · eval_samples parquets (universe B)
MID: merge complexity / emb / heur · column alias normalization
OUT: training DataFrames with ``training_id``, features, labels
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config.paths import REPO_ROOT, complexity_record_path
from routing.config import AGENTS, EVAL_SAMPLES_DIR, SPLIT_CSV, TARGET
from routing.data.aliases import normalize_loader_frame
from routing.datasets import resolve_dataset_name
from routing.features.columns import (
    load_query_embeddings,
    load_query_heuristics,
    query_embeddings_path,
    query_heuristics_path,
)
from routing.config import feature_set_needs_embeddings, feature_set_needs_heuristics, normalize_feature_set


def _path_with_root(path: Path, root: Path | None) -> Path:
    if root is None or Path(root) == REPO_ROOT:
        return path
    return Path(root) / path.relative_to(REPO_ROOT)


def load_split(
    split: str,
    *,
    root: Path | None = None,
    with_embeddings: bool | None = None,
    with_heuristics: bool | None = None,
) -> pd.DataFrame:
    if split not in SPLIT_CSV:
        raise ValueError(f"split must be one of {list(SPLIT_CSV)}")
    root = Path(root or REPO_ROOT)
    labels = pd.read_csv(_path_with_root(SPLIT_CSV[split], root))
    feats = pd.read_parquet(_path_with_root(complexity_record_path(split), root))
    drop = [c for c in feats.columns if c in labels.columns and c != "training_id"]
    merged = labels.merge(
        feats.drop(columns=[c for c in drop if c in feats.columns], errors="ignore"),
        on="training_id",
        how="inner",
    )
    emb_path = query_embeddings_path(split, root=root)
    if with_embeddings is None:
        with_embeddings = emb_path.is_file()
    if with_embeddings:
        emb = load_query_embeddings(split, root=root)
        drop_emb = [c for c in emb.columns if c in merged.columns and c != "training_id"]
        merged = merged.merge(
            emb.drop(columns=[c for c in drop_emb if c in emb.columns], errors="ignore"),
            on="training_id",
            how="inner",
        )
    heur_path = query_heuristics_path(split, root=root)
    if with_heuristics is None:
        with_heuristics = heur_path.is_file()
    if with_heuristics:
        heur = load_query_heuristics(split, root=root)
        drop_h = [c for c in heur.columns if c in merged.columns and c != "training_id"]
        merged = merged.merge(
            heur.drop(columns=[c for c in drop_h if c in heur.columns], errors="ignore"),
            on="training_id",
            how="inner",
        )
    merged = merged[merged[TARGET].isin(AGENTS)].copy()
    if merged.empty:
        raise ValueError(f"{split}: no rows after filtering to {AGENTS}")
    return normalize_loader_frame(merged.reset_index(drop=True))


def load_split_for_feature_set(
    split: str,
    feature_set: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load labels + complexity + optional emb/heur columns required by ``feature_set``."""
    fs = normalize_feature_set(feature_set)
    return load_split(
        split,
        root=root,
        with_embeddings=feature_set_needs_embeddings(fs),
        with_heuristics=feature_set_needs_heuristics(fs),
    )


def eval_base_frame(parquet_path: Path, dataset: str | None = None) -> pd.DataFrame:
    path = Path(parquet_path)
    df = normalize_loader_frame(pd.read_parquet(path))
    if "query" not in df.columns:
        raise ValueError(f"{path}: missing 'query'")
    id_col = "training_id" if "training_id" in df.columns else "id"
    if id_col not in df.columns:
        raise ValueError(f"{path}: need id or training_id")
    ds = resolve_dataset_name(dataset or path.stem)
    out = pd.DataFrame(
        {
            "training_id": df[id_col].astype(str),
            "query": df["query"].astype(str),
            "dataset": ds,
            "expected_answer": df["answer"].astype(str) if "answer" in df.columns else "",
        }
    )
    for col in ("level", "file_name", "split", "n_hops", "hop_name", "type", "context", "metadata", "paragraphs"):
        if col in df.columns:
            out[col] = df[col]
    return out


def load_eval_parquet(dataset: str, *, data_path: Path | str | None = None) -> pd.DataFrame:
    path = Path(data_path) if data_path else EVAL_SAMPLES_DIR / f"{dataset.strip().lower()}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Eval parquet not found: {path}")
    return eval_base_frame(path, dataset=dataset)
