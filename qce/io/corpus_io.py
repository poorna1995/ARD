"""Load QCE corpora from CSV."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from qce.io.query_input import META_KEYS, resolve_ds

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_METADATA = REPO_ROOT / "datasets/train_samples/combined_raw.parquet"

REQUIRED_COLS = ("training_id", "query", "dataset")


def load_corpus(
    path: str | Path,
    *,
    labeled_only: bool = False,
    label_col: str = "label_status",
    attach_metadata: bool = True,
    metadata_path: str | Path = DEFAULT_RAW_METADATA,
) -> pd.DataFrame:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)

    corpus = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLS if c not in corpus.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")

    if labeled_only and label_col in corpus.columns:
        corpus = corpus[corpus[label_col].astype(str).str.strip().eq("labeled")].copy()

    corpus = corpus.dropna(subset=list(REQUIRED_COLS)).copy()
    corpus["training_id"] = corpus["training_id"].astype(str)
    corpus["query"] = corpus["query"].astype(str)
    if corpus["query"].str.strip().eq("").any():
        raise ValueError(f"{path}: empty query text")
    if corpus["training_id"].duplicated().any():
        raise ValueError(f"{path}: duplicate training_id")

    corpus["dataset"] = corpus["dataset"].astype(str).map(resolve_ds)

    if attach_metadata and Path(metadata_path).is_file():
        corpus = _attach_metadata(corpus, metadata_path)

    keep = list(REQUIRED_COLS) + [c for c in sorted(META_KEYS) if c in corpus.columns]
    extra = [c for c in corpus.columns if c not in keep and c.startswith(("expected_", "label_"))]
    return corpus[[c for c in keep + extra if c in corpus.columns]].reset_index(drop=True)


def _attach_metadata(corpus: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    raw = pd.read_parquet(raw_path)
    if "training_id" not in raw.columns:
        return corpus
    raw = raw.drop_duplicates(subset=["training_id"], keep="first")
    raw["training_id"] = raw["training_id"].astype(str)
    pull = [c for c in raw.columns if c in META_KEYS]
    if not pull:
        return corpus
    meta = raw[["training_id", *pull]]
    return corpus.merge(meta, on="training_id", how="left")
