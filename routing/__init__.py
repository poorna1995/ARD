"""CLI entry package — ``research-route`` / ``python -m routing``.

Implementation lives in ``router``, ``eval``, ``research``.
Legacy ``from routing.router import …`` still works via ``routing._compat``.
"""

from routing._compat import install_legacy_submodules

install_legacy_submodules()

from router import (
    AGENTS,
    PROBA_COLS,
    PRODUCTION_FEATURE_SET,
    PRODUCTION_ROUTER_PATH,
    ROUTER_MODEL_PATH,
    TARGET,
    RuntimeRouter,
    attach_router_predictions,
    build_eval_features,
    ensure_eval_features,
    evaluate,
    load_eval_parquet,
    load_router,
    load_router_frame,
    load_split,
    load_split_for_feature_set,
    normalize_feature_set,
    resolve_dataset_name,
    router_feature_cols,
    run_agent_cascade,
    save_router,
    top_k_from_row,
    train_router,
    validate_feature_set_data,
)
from router.config import (
    LEGACY_HARD_HGBM_PATH,
    PRIMARY_ROUTER_PATH,
    PRODUCTION_HGBM_PARAMS,
    SECONDARY_ROUTER_PATH,
    TUNED_HGBM_PARAMS,
    TUNED_ROUTER_PATH,
)

__all__ = [
    "AGENTS",
    "PROBA_COLS",
    "LEGACY_HARD_HGBM_PATH",
    "PRIMARY_ROUTER_PATH",
    "PRODUCTION_FEATURE_SET",
    "PRODUCTION_HGBM_PARAMS",
    "PRODUCTION_ROUTER_PATH",
    "ROUTER_MODEL_PATH",
    "SECONDARY_ROUTER_PATH",
    "TARGET",
    "TUNED_HGBM_PARAMS",
    "TUNED_ROUTER_PATH",
    "RuntimeRouter",
    "attach_router_predictions",
    "build_eval_features",
    "ensure_eval_features",
    "evaluate",
    "load_eval_parquet",
    "load_router",
    "load_router_frame",
    "load_split",
    "load_split_for_feature_set",
    "normalize_feature_set",
    "resolve_dataset_name",
    "router_feature_cols",
    "run_agent_cascade",
    "run_full_analysis",
    "save_router",
    "top_k_from_row",
    "train_router",
    "validate_feature_set_data",
]


def __getattr__(name: str):
    if name == "run_full_analysis":
        from research.analysis import run_full_analysis

        return run_full_analysis
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
