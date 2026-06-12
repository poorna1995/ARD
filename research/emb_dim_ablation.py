"""
Embedding PCA dimension ablation for cvec5_emb + soft HGBM.

Builds train-fitted PCA projections at dims {8, 16, 32, 64}, trains cost-soft KL
routers, and reports test utility regret (λ=25).

Usage::

  uv run python -m research.emb_dim_ablation
  uv run python -m research.emb_dim_ablation --dims 8 16 --skip-encode  # parquets exist
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from config.common import REPO_ROOT
from research.baselines import eval_pipeline_on_split
from research.soft_train import (
    fit_soft_kl_router,
    make_soft_kl_pipeline,
)
from router.config import PRODUCTION_FEATURE_SET, PRODUCTION_HGBM_PARAMS
from router.router import (
    load_split_for_feature_set,
    router_feature_cols,
    validate_feature_set_data,
)
from scripts.build_query_embeddings import (
    DEFAULT_LABELS_DIR,
    DEFAULT_N_COMPONENTS,
    _SPLIT_OUT,
    encode_queries,
    fit_pca_on_train,
    load_queries,
    matrix_to_frame,
)

OUT_ROOT = REPO_ROOT / "results/experiments/emb_dim_ablation"
FEATURES_ROOT = REPO_ROOT / "datasets/qce_features"
RAW_CACHE = OUT_ROOT / "raw_embeddings_384d.npz"
DEFAULT_DIMS = (8, 16, 32, 64)


def _emb_features_dir(n_components: int) -> Path:
    return FEATURES_ROOT / f"emb_dim_{n_components}"


def _load_or_encode_raw(
    *,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize_embeddings: bool = True,
    force: bool = False,
) -> dict[str, np.ndarray]:
    """Encode train/val/test once; cache under OUT_ROOT."""
    if RAW_CACHE.is_file() and not force:
        data = np.load(RAW_CACHE)
        return {k: data[k] for k in data.files}

    raw_by_split: dict[str, np.ndarray] = {}
    ids_by_split: dict[str, pd.Series] = {}
    for split in ("train", "val", "test"):
        df = load_queries(split, labels_dir=labels_dir)
        texts = df["query"].astype(str).tolist()
        raw_by_split[split] = encode_queries(
            texts,
            model_name=model_name,
            batch_size=batch_size,
            normalize_embeddings=normalize_embeddings,
        )
        ids_by_split[split] = df["training_id"].astype(str)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez(
        RAW_CACHE,
        train=raw_by_split["train"],
        val=raw_by_split["val"],
        test=raw_by_split["test"],
        train_ids=ids_by_split["train"].to_numpy(),
        val_ids=ids_by_split["val"].to_numpy(),
        test_ids=ids_by_split["test"].to_numpy(),
    )
    return {
        "train": raw_by_split["train"],
        "val": raw_by_split["val"],
        "test": raw_by_split["test"],
    }


def _production_embeddings_dir() -> Path:
    return FEATURES_ROOT


def build_embedding_parquets_for_dim(
    n_components: int,
    *,
    raw: dict[str, np.ndarray] | None = None,
    labels_dir: Path = DEFAULT_LABELS_DIR,
    random_state: int = 42,
    use_production_if_dim16: bool = True,
    verbose: bool = True,
) -> Path:
    """Fit PCA(n) on train raw matrix; write query_embeddings_{split}.parquet per split."""
    if use_production_if_dim16 and n_components == DEFAULT_N_COMPONENTS:
        prod = _production_embeddings_dir()
        if all((prod / f"query_embeddings_{s}.parquet").is_file() for s in ("train", "val", "test")):
            if verbose:
                print(f"  dim={n_components}: reuse production parquets in {prod.relative_to(REPO_ROOT)}")
            return prod

    out_dir = _emb_features_dir(n_components)
    model_dir = REPO_ROOT / "models/query_embeddings" / f"emb_dim_{n_components}"
    pca_path = model_dir / f"pca_{n_components}.joblib"

    if raw is None:
        if not RAW_CACHE.is_file():
            raise FileNotFoundError(f"Missing raw cache {RAW_CACHE}; run without --skip-encode")
        data = np.load(RAW_CACHE)
        train_raw = data["train"]
        val_raw = data["val"]
        test_raw = data["test"]
    else:
        train_raw = raw["train"]
        val_raw = raw["val"]
        test_raw = raw["test"]

    pca = fit_pca_on_train(train_raw, n_components=n_components, random_state=random_state)
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pca, pca_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, matrix in (
        ("train", train_raw),
        ("val", val_raw),
        ("test", test_raw),
    ):
        df = load_queries(split, labels_dir=labels_dir)
        reduced = pca.transform(matrix)
        frame = matrix_to_frame(df["training_id"], reduced, n_components=n_components)
        out_path = out_dir / f"query_embeddings_{_SPLIT_OUT[split]}.parquet"
        frame.to_parquet(out_path, index=False)
        if verbose:
            evr = float(np.sum(getattr(pca, "explained_variance_ratio_", [])))
            print(f"  dim={n_components} {split}: {len(frame)} rows → {out_path.name} (PCA EVR≈{evr:.3f})")
    return out_dir


def train_and_eval_dim(
    n_components: int,
    *,
    embeddings_dir: Path | None = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Soft HGBM + cvec5_emb at embedding dim n; return test metrics."""
    if embeddings_dir is not None:
        emb_dir = Path(embeddings_dir)
    elif n_components == DEFAULT_N_COMPONENTS and all(
        (_production_embeddings_dir() / f"query_embeddings_{s}.parquet").is_file()
        for s in ("train", "val", "test")
    ):
        emb_dir = _production_embeddings_dir()
    else:
        emb_dir = _emb_features_dir(n_components)
    fs = PRODUCTION_FEATURE_SET
    experiment_id = f"hgbm_cvec5_emb_d{n_components}_soft_kl_ablation"

    train_df = load_split_for_feature_set("train", fs, embeddings_dir=emb_dir)
    validate_feature_set_data(train_df, fs)
    feature_cols = router_feature_cols(train_df, fs)

    pipe = make_soft_kl_pipeline(
        feature_cols,
        "hgbm",
        random_state=seed,
        hgbm_params=PRODUCTION_HGBM_PARAMS,
    )
    pipe, n_queries, n_expanded = fit_soft_kl_router(pipe, train_df, feature_cols)

    test_row = eval_pipeline_on_split(
        "test",
        pipe,
        feature_cols,
        classifier="hgbm",
        hp_policy="soft_kl",
        experiment_id=experiment_id,
        feature_set=fs,
        embeddings_dir=emb_dir,
    )
    val_row = eval_pipeline_on_split(
        "val",
        pipe,
        feature_cols,
        classifier="hgbm",
        hp_policy="soft_kl",
        experiment_id=experiment_id,
        feature_set=fs,
        embeddings_dir=emb_dir,
    )

    if verbose:
        print(
            f"  dim={n_components}: test regret agent={test_row['mean_utility_regret']:.4f} "
            f"total={test_row['mean_utility_regret_total']:.4f} "
            f"EM={test_row['exec_em']:.2%}  n_features={len(feature_cols)}"
        )

    return {
        "emb_dim": n_components,
        "n_features": len(feature_cols),
        "feature_cols": ",".join(feature_cols),
        "n_train_queries": n_queries,
        "val_mean_utility_regret": val_row["mean_utility_regret"],
        "val_mean_utility_regret_total": val_row["mean_utility_regret_total"],
        "val_exec_em": val_row["exec_em"],
        "test_mean_utility_regret": test_row["mean_utility_regret"],
        "test_mean_utility_regret_total": test_row["mean_utility_regret_total"],
        "test_exec_em": test_row["exec_em"],
        "test_total_musd": test_row.get("mean_total_usd", 0) * 1000 if test_row.get("mean_total_usd") else None,
        "embeddings_dir": str(emb_dir.relative_to(REPO_ROOT)),
        "experiment_id": experiment_id,
    }


def run_ablation(
    dims: tuple[int, ...] = DEFAULT_DIMS,
    *,
    skip_encode: bool = False,
    force_encode: bool = False,
    seed: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    raw: dict[str, np.ndarray] | None = None
    if not skip_encode:
        if verbose:
            print("Encoding queries (sentence-transformers, one pass)...")
        raw = _load_or_encode_raw(force=force_encode)
    elif verbose:
        print(f"Using raw cache: {RAW_CACHE}")

    rows: list[dict[str, Any]] = []
    for dim in dims:
        if verbose:
            print(f"\n=== emb_dim={dim} ===")
        need_build = not all(
            (_emb_features_dir(dim) / f"query_embeddings_{s}.parquet").is_file()
            for s in ("train", "val", "test")
        )
        if need_build:
            build_embedding_parquets_for_dim(dim, raw=raw, verbose=verbose)
        rows.append(train_and_eval_dim(dim, seed=seed, verbose=verbose))

    table = pd.DataFrame(rows).sort_values("emb_dim")
    table.to_csv(OUT_ROOT / "ablation_test.csv", index=False)
    (OUT_ROOT / "ablation_summary.json").write_text(
        json.dumps({"dims": list(dims), "rows": rows}, indent=2),
        encoding="utf-8",
    )
    if verbose:
        print(f"\nWrote {OUT_ROOT / 'ablation_test.csv'}")
        print(table[["emb_dim", "test_mean_utility_regret", "test_mean_utility_regret_total", "test_exec_em"]].to_string(index=False))
    return table


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="PCA embedding dimension ablation (cvec5_emb soft HGBM).")
    p.add_argument("--dims", type=int, nargs="+", default=list(DEFAULT_DIMS))
    p.add_argument("--skip-encode", action="store_true", help="Require raw cache + per-dim parquets")
    p.add_argument("--force-encode", action="store_true", help="Re-run sentence-transformer encoding")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    run_ablation(
        tuple(sorted(set(args.dims))),
        skip_encode=args.skip_encode,
        force_encode=args.force_encode,
        seed=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
