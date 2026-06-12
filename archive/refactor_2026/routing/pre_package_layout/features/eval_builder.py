"""
Eval-sample feature build: decompose cache · complexity · embeddings (universe B).

IN:  base DataFrame (training_id, query, dataset) · tag (gaia, hotpot, …)
MID: LLM decompose · C(Q) parquet · PCA embeddings merge
OUT: merged feature DataFrame · on-disk parquets under datasets/qce_features/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.paths import eval_complexity_path, eval_embeddings_path, decomposer_cache_path
from routing.config import (
    DECOMPOSER_CACHE_DIR,
    PCA_PATH,
    QCE_FEATURES_DIR,
    TRAIN_NORM_JSON,
)
from routing.datasets import resolve_dataset_name


def feature_paths(tag: str) -> tuple[Path, Path, Path]:
    t = tag.strip().lower()
    return (
        eval_complexity_path(t),
        eval_embeddings_path(t),
        decomposer_cache_path(t),
    )


def load_decompose_cache(cache_path: Path, *, id_key: str = "training_id") -> dict[str, dict[str, Any]]:
    from qce.decompose import load_plans_jsonl

    return {
        str(rec.get(id_key, "") or ""): rec
        for rec in load_plans_jsonl(cache_path)
        if rec.get(id_key)
    }


def decompose_cache_status(
    rows: list[dict[str, Any]], cache_path: Path, *, id_key: str = "training_id"
) -> tuple[int, int, list[str]]:
    if not cache_path.is_file():
        return 0, len(rows), [str(r[id_key]) for r in rows]
    cached = load_decompose_cache(cache_path, id_key=id_key)
    missing = [str(r[id_key]) for r in rows if str(r[id_key]) not in cached]
    return len(rows) - len(missing), len(rows), missing


def plans_from_cache(
    rows: list[dict[str, Any]], cache_path: Path, *, id_key: str = "training_id"
) -> list[dict[str, Any]]:
    cached = load_decompose_cache(cache_path, id_key=id_key)
    plans, missing = [], []
    for row in rows:
        tid = str(row[id_key])
        if tid in cached:
            plans.append(cached[tid])
        else:
            missing.append(tid)
    if missing:
        raise KeyError(f"{len(missing)} ids missing from {cache_path}")
    return plans


def merge_feature_tables(base: pd.DataFrame, tag: str) -> pd.DataFrame:
    c_path, e_path, _ = feature_paths(tag)
    if not c_path.is_file() or not e_path.is_file():
        raise FileNotFoundError(f"QCE features missing for {tag!r}: {c_path}, {e_path}")
    c, e = pd.read_parquet(c_path), pd.read_parquet(e_path)
    if "training_id" not in c.columns or "training_id" not in e.columns:
        raise KeyError("QCE feature tables must include training_id")
    if c["training_id"].duplicated().any():
        c = c.drop_duplicates(subset=["training_id"], keep="first")
    if e["training_id"].duplicated().any():
        e = e.drop_duplicates(subset=["training_id"], keep="first")
    drop_c = [x for x in c.columns if x in base.columns and x != "training_id"]
    drop_e = [x for x in e.columns if x in base.columns and x != "training_id"]
    merged = base.merge(c.drop(columns=drop_c, errors="ignore"), on="training_id", how="left", validate="m:1")
    merged = merged.merge(e.drop(columns=drop_e, errors="ignore"), on="training_id", how="left", validate="m:1")
    if len(merged) != len(base):
        raise ValueError(f"feature merge changed rows unexpectedly: {len(base)} → {len(merged)}")
    return merged.reset_index(drop=True)


def build_eval_features(
    base: pd.DataFrame, tag: str, *, force_refresh: bool = False, verbose: bool = True
) -> None:
    from qce.complexity import complexity_dataframe, load_train_norm, write_complexity_parquet
    from qce.decompose import decompose_batch
    from scripts.build_query_embeddings import DEFAULT_MODEL, encode_queries, matrix_to_frame

    c_path, e_path, cache_path = feature_paths(tag.strip().lower())
    rows = base.to_dict(orient="records")
    for r in rows:
        r["dataset"] = resolve_dataset_name(str(r.get("dataset") or tag))

    n_hit, n_total, missing_ids = decompose_cache_status(rows, cache_path)
    if verbose:
        print(
            f"Decompose cache {cache_path.name}: {n_hit}/{n_total}"
            + (f" ({len(missing_ids)} missing)" if missing_ids else " (full hit)")
        )

    if force_refresh:
        if verbose:
            print(f"Decomposing {n_total} queries (force_refresh)…")
        plans = decompose_batch(rows, cache_path=cache_path, force_refresh=True)
    elif not missing_ids:
        if verbose:
            print("Skipping LLM decompose — all ids in cache.")
        plans = plans_from_cache(rows, cache_path)
    else:
        if verbose:
            print(f"Decomposing {len(missing_ids)} missing ({n_hit} cached)…")
        plans = decompose_batch(rows, cache_path=cache_path, force_refresh=False)
        _, _, still_missing = decompose_cache_status(rows, cache_path)
        if still_missing:
            raise RuntimeError(f"{len(still_missing)} ids still missing after decompose")

    if not TRAIN_NORM_JSON.is_file():
        raise FileNotFoundError(f"Missing {TRAIN_NORM_JSON}")
    c_df = complexity_dataframe(plans, norm=load_train_norm(TRAIN_NORM_JSON), fit_norm=False)
    c_path.parent.mkdir(parents=True, exist_ok=True)
    write_complexity_parquet(c_df, c_path)
    if verbose:
        print(f"Wrote {c_path} ({len(c_df)} rows)")

    if not PCA_PATH.is_file():
        raise FileNotFoundError(f"Missing {PCA_PATH}")
    pca = joblib.load(PCA_PATH)
    n_comp = int(getattr(pca, "n_components_", 16))
    if verbose:
        print(f"Encoding {len(base)} queries ({DEFAULT_MODEL})…")
    raw = encode_queries(
        base["query"].astype(str).tolist(),
        model_name=DEFAULT_MODEL,
        batch_size=32,
        normalize_embeddings=True,
    )
    matrix_to_frame(base["training_id"], pca.transform(raw), n_components=n_comp).to_parquet(e_path, index=False)
    if verbose:
        print(f"Wrote {e_path}")


def ensure_eval_features(
    base: pd.DataFrame,
    tag: str,
    *,
    build: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    c_path, e_path, _ = feature_paths(tag)
    if not c_path.is_file() or not e_path.is_file():
        if not build:
            raise FileNotFoundError(f"QCE features missing for {tag!r}; use --build-features")
        build_eval_features(base, tag, force_refresh=force_refresh, verbose=verbose)
    return merge_feature_tables(base, tag)


def build_eval_embeddings_only(
    base: pd.DataFrame,
    tag: str,
    *,
    verbose: bool = True,
) -> Path:
    """Write ``query_embeddings_{tag}.parquet`` only (no LLM decompose)."""
    from scripts.build_query_embeddings import DEFAULT_MODEL, encode_queries, matrix_to_frame

    _, e_path, _ = feature_paths(tag.strip().lower())
    if not PCA_PATH.is_file():
        raise FileNotFoundError(f"Missing {PCA_PATH}")

    pca = joblib.load(PCA_PATH)
    n_comp = int(getattr(pca, "n_components_", 16))
    if verbose:
        print(f"[selective] Encoding {len(base)} queries ({DEFAULT_MODEL})…")
    raw = encode_queries(
        base["query"].astype(str).tolist(),
        model_name=DEFAULT_MODEL,
        batch_size=32,
        normalize_embeddings=True,
    )
    e_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_to_frame(base["training_id"], pca.transform(raw), n_components=n_comp).to_parquet(
        e_path, index=False
    )
    if verbose:
        print(f"[selective] Wrote {e_path}")
    return e_path


def build_eval_complexity_subset(
    base_subset: pd.DataFrame,
    tag: str,
    *,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Path:
    """Decompose + C(Q) for ``base_subset`` rows; upsert into ``complexity_record_{tag}``."""
    from qce.complexity import complexity_dataframe, load_train_norm, write_complexity_parquet
    from qce.decompose import decompose_batch

    if base_subset.empty:
        c_path, _, _ = feature_paths(tag)
        return c_path

    c_path, _, cache_path = feature_paths(tag.strip().lower())
    rows = base_subset.to_dict(orient="records")
    for r in rows:
        r["dataset"] = resolve_dataset_name(str(r.get("dataset") or tag))

    if verbose:
        print(f"[selective] Decomposing {len(rows)} uncertain queries…")
    decompose_batch(rows, cache_path=cache_path, force_refresh=force_refresh)
    plans = plans_from_cache(rows, cache_path)
    new_c = complexity_dataframe(plans, norm=load_train_norm(TRAIN_NORM_JSON), fit_norm=False)

    if c_path.is_file():
        existing = pd.read_parquet(c_path)
        keep = existing[~existing["training_id"].isin(new_c["training_id"])]
        merged = pd.concat([keep, new_c], ignore_index=True)
    else:
        merged = new_c
    c_path.parent.mkdir(parents=True, exist_ok=True)
    write_complexity_parquet(merged, c_path)
    if verbose:
        print(f"[selective] Updated {c_path} (+{len(new_c)} rows)")
    return c_path


def merge_embeddings_only(base: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Merge base with embedding parquet only (no ``dim_*``)."""
    _, e_path, _ = feature_paths(tag)
    if not e_path.is_file():
        raise FileNotFoundError(f"missing embeddings for {tag!r}: {e_path}")
    emb = pd.read_parquet(e_path)
    drop_e = [x for x in emb.columns if x in base.columns and x != "training_id"]
    merged = base.merge(
        emb.drop(columns=[c for c in drop_e if c in emb.columns], errors="ignore"),
        on="training_id",
        how="left",
        validate="m:1",
    )
    if len(merged) != len(base):
        raise ValueError(f"embedding merge changed row count: {len(base)} → {len(merged)}")
    return merged.reset_index(drop=True)
