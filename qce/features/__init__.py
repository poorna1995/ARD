"""QCE feature build — decompose cache, complexity, embeddings merge."""

from qce.features.build import (
    build_eval_complexity_subset,
    build_eval_embeddings_only,
    build_eval_features,
    decompose_cache_status,
    ensure_eval_features,
    feature_paths,
    load_decompose_cache,
    merge_embeddings_only,
    merge_feature_tables,
    plans_from_cache,
)

__all__ = [
    "build_eval_complexity_subset",
    "build_eval_embeddings_only",
    "build_eval_features",
    "decompose_cache_status",
    "ensure_eval_features",
    "feature_paths",
    "load_decompose_cache",
    "merge_embeddings_only",
    "merge_feature_tables",
    "plans_from_cache",
]
