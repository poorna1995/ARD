"""
Eval-sample feature build: decompose cache · complexity · embeddings (universe B).

IN:  base DataFrame (training_id, query, dataset) · tag (gaia, hotpot, …)
MID: LLM decompose · C(Q) parquet · PCA embeddings merge
OUT: merged feature DataFrame · on-disk parquets under qce_features/
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.common import REPO_ROOT
from config.global_config.paths import (
    decomposer_cache_path,
    eval_complexity_path,
    eval_embeddings_path,
    eval_heuristics_path,
)
from config.local.constants.heuristics import HEUR_ROUTER_COLS
from config.local.router.paths import HEUR_CALIBRATOR_PATH


def _router_paths():
    from router.config import PCA_PATH, TRAIN_NORM_JSON, resolve_dataset_name

    return PCA_PATH, TRAIN_NORM_JSON, resolve_dataset_name


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


def _heur_analyzer():
    import sys

    old = REPO_ROOT / "old"
    if str(old) not in sys.path:
        sys.path.insert(0, str(old))
    from difficulty.feature_measure import TaskComplexityAnalyzer

    if not HEUR_CALIBRATOR_PATH.is_file():
        raise FileNotFoundError(f"Missing heuristic calibrator: {HEUR_CALIBRATOR_PATH}")
    return TaskComplexityAnalyzer.from_calibrator(HEUR_CALIBRATOR_PATH, use_spacy=True)


def ensure_query_heuristics(
    df: pd.DataFrame,
    tag: str | None = None,
    *,
    verbose: bool = False,
) -> pd.DataFrame:
    """Attach ``heur_*`` columns from disk cache or query-only scorer."""
    missing = [c for c in HEUR_ROUTER_COLS if c not in df.columns]
    if not missing:
        return df
    if "query" not in df.columns:
        raise ValueError(f"cannot compute heur_* without query column (missing {missing[:3]}…)")

    out = df.copy()
    if tag:
        h_path = eval_heuristics_path(tag)
        if h_path.is_file():
            heur = pd.read_parquet(h_path)
            if "training_id" in heur.columns and "training_id" in out.columns:
                drop_h = [c for c in heur.columns if c in out.columns and c != "training_id"]
                out = out.merge(
                    heur.drop(columns=[c for c in drop_h if c in heur.columns], errors="ignore"),
                    on="training_id",
                    how="left",
                    validate="m:1",
                )
                if not [c for c in HEUR_ROUTER_COLS if c not in out.columns]:
                    return out.reset_index(drop=True)

    if verbose:
        print(f"Computing heur_* for {len(out)} queries…")
    analyzer = _heur_analyzer()
    if "dataset" in out.columns:
        datasets = out["dataset"].astype(str).tolist()
    else:
        datasets = [None] * len(out)
    rows = [
        analyzer.router_feature_row(
            str(q).strip(),
            dataset=(ds.strip() or None) if ds else None,
        )
        for q, ds in zip(out["query"].astype(str), datasets, strict=True)
    ]
    heur_df = pd.DataFrame(rows)
    for c in HEUR_ROUTER_COLS:
        out[c] = heur_df[c].values

    if tag and "training_id" in out.columns:
        h_path = eval_heuristics_path(tag)
        h_path.parent.mkdir(parents=True, exist_ok=True)
        out[["training_id", *HEUR_ROUTER_COLS]].to_parquet(h_path, index=False)
        if verbose:
            print(f"Wrote {h_path}")

    return out.reset_index(drop=True)


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
    h_path = eval_heuristics_path(tag)
    if h_path.is_file():
        heur = pd.read_parquet(h_path)
        if "training_id" in heur.columns:
            drop_h = [x for x in heur.columns if x in merged.columns and x != "training_id"]
            merged = merged.merge(
                heur.drop(columns=[c for c in drop_h if c in heur.columns], errors="ignore"),
                on="training_id",
                how="left",
                validate="m:1",
            )
    if len(merged) != len(base):
        raise ValueError(f"feature merge changed rows unexpectedly: {len(base)} → {len(merged)}")
    return merged.reset_index(drop=True)


def build_eval_features(
    base: pd.DataFrame, tag: str, *, force_refresh: bool = False, verbose: bool = True
) -> None:
    from qce.complexity import complexity_dataframe, load_train_norm, write_complexity_parquet
    from qce.decompose import decompose_batch
    from scripts.build_query_embeddings import DEFAULT_MODEL, encode_queries, matrix_to_frame

    _, TRAIN_NORM_JSON, resolve_dataset_name = _router_paths()
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

    PCA_PATH, _, _ = _router_paths()
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

    ensure_query_heuristics(base, tag.strip().lower(), verbose=verbose)


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

    PCA_PATH, _, _ = _router_paths()
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

    _, TRAIN_NORM_JSON, resolve_dataset_name = _router_paths()
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
