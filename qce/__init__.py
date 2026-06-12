"""
Query Complexity Estimation — Phase 1 feature pipeline.

Stages::

    decompose  → LLM procedure plans (JSONL cache)
    graph      → procedure DAG + graph metrics
    complexity → C(Q) vector (dim_*)
    features   → merge complexity + embeddings for routing
    io         → query row normalization, corpus load
"""

from config.local.constants import DIMS, DS, GRAPH, PLAN, SCORE, STATUS, TOOLS

from qce.decompose import (
    DEFAULT_CACHE_PATH,
    DEFAULT_DECOMPOSE_MODEL,
    PLAN_OK,
    PLAN_PARSE_FAIL,
    decompose_batch,
    decompose_query,
    load_plans_jsonl,
    plan_summary,
    write_plans_jsonl,
)
from qce.complexity import (
    C_VECTOR_COLS,
    C_VECTOR_VER,
    ROUTER_MAIN_FEATURE_SET,
    SCALAR_COL,
    c_vector,
    calibration_summary,
    complexity_dataframe,
    complexity_from_plan,
    complexity_scalar,
    complexity_splits,
    fit_norm_from_jsonl,
    fit_norm_from_plans,
    gate_hard,
    graph_features_dataframe,
    load_train_norm,
    save_train_norm,
    router_feature_cols,
    router_input_matrix,
    router_scalar_col,
    router_scalar_vector,
)
from qce.graph import (
    END,
    START,
    GraphBuildResult,
    TrainNorm,
    build_graphs_from_plans,
    build_task_dag,
    complexity_graph_score,
    features_dataframe,
    fit_train_norm,
    rescore_dataframe,
)

# Dimension column tuples (canonical in config.local.constants).
DIM_COLS = DIMS.main
DIM5_COLS = DIMS.legacy

__all__ = [
    "C_VECTOR_COLS",
    "C_VECTOR_VER",
    "DIMS",
    "DIM5_COLS",
    "DIM_COLS",
    "DS",
    "GRAPH",
    "PLAN",
    "ROUTER_MAIN_FEATURE_SET",
    "SCALAR_COL",
    "SCORE",
    "STATUS",
    "TOOLS",
    "DEFAULT_CACHE_PATH",
    "DEFAULT_DECOMPOSE_MODEL",
    "END",
    "GraphBuildResult",
    "PLAN_OK",
    "PLAN_PARSE_FAIL",
    "START",
    "TrainNorm",
    "build_graphs_from_plans",
    "build_task_dag",
    "c_vector",
    "calibration_summary",
    "complexity_dataframe",
    "complexity_from_plan",
    "complexity_graph_score",
    "complexity_scalar",
    "complexity_splits",
    "fit_norm_from_jsonl",
    "fit_norm_from_plans",
    "load_train_norm",
    "save_train_norm",
    "decompose_batch",
    "decompose_query",
    "features_dataframe",
    "fit_train_norm",
    "load_plans_jsonl",
    "plan_summary",
    "rescore_dataframe",
    "gate_hard",
    "graph_features_dataframe",
    "router_feature_cols",
    "router_input_matrix",
    "router_scalar_col",
    "router_scalar_vector",
    "write_plans_jsonl",
]
