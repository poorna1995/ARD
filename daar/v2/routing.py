"""D-AAR 2.0 routing — hybrid structure, trace Gate 1b, residual cost."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.global_config.paths import (
    daar_models_dir,
    daar_routing_dir,
    daar_trace_skeleton_path,
    ensure_daar_dirs,
)
from config.local.constants.agents import ROUTER_AGENTS
from daar.gate1 import predict_solvable_proba
from daar.routing import (
    LAMBDA_GRID,
    THESIS_VARIANT,
    build_query_table,
    evaluate_lambda,
    load_gate1,
    load_gate1a,
    pick_best_lambda,
    route_agent,
    sweep_lambdas,
)
from daar.v2.constants import GATE1B_TRACE_MANIFEST, GATE1B_TRACE_STEM
from daar.v2.cost_residual import apply_cost_residual
from daar.v2.structure_distill import apply_hybrid_phi, load_structure_distill
from router.router import load_router, predict_agent_proba


def load_gate1b_trace(
    models_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    models_dir = models_dir or daar_models_dir()
    manifest_path = models_dir / GATE1B_TRACE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path} — run: uv run python scripts/train_daar2_gate1b_trace.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = Path(manifest.get("model_path", models_dir / f"{GATE1B_TRACE_STEM}.joblib"))
    pipe = load_router(model_path)["pipeline"]
    return pipe, manifest, list(manifest["feature_cols"])


def _refresh_gate1_scores(
    table: pd.DataFrame,
    query: pd.DataFrame,
    *,
    models_dir: Path | None,
    use_trace_gate1b: bool,
) -> pd.DataFrame:
    out = table.copy()
    gate1a_pipe, _g1a_m, gate1a_cols = load_gate1a(models_dir)
    if use_trace_gate1b:
        gate1b_pipe, _, gate1b_cols = load_gate1b_trace(models_dir)
    else:
        gate1b_pipe, _, gate1b_cols = load_gate1(THESIS_VARIANT, models_dir)

    q = query.copy()
    q["training_id"] = q["training_id"].astype(str)
    proba = predict_agent_proba(gate1b_pipe, q, gate1b_cols)
    p_solv = predict_solvable_proba(gate1a_pipe, q, gate1a_cols)
    if hasattr(p_solv, "__len__"):
        p_solv = list(p_solv)
    else:
        p_solv = [float(p_solv)]

    score = pd.DataFrame({"training_id": q["training_id"].values, "p_solv": p_solv})
    for agent in ROUTER_AGENTS:
        score[f"s_hat_{agent}"] = proba[f"p_{agent}"].astype(float).values
    score["s_hat_max"] = proba.max(axis=1).astype(float).values

    drop = [c for c in out.columns if c.startswith("s_hat_") or c == "p_solv" or c == "s_hat_max"]
    out = out.drop(columns=drop, errors="ignore")
    out["training_id"] = out["training_id"].astype(str)
    return out.merge(score, on="training_id", how="left")


def build_query_table_v2(
    split: str,
    *,
    hybrid_structure: bool = False,
    confidence_threshold: float = 0.35,
    use_residual_cost: bool = False,
    use_trace_gate1b: bool = False,
    models_dir: Path | None = None,
) -> pd.DataFrame:
    """Extend thesis table with 2.0 options (hybrid φ, residual ĉ, trace Gate 1b)."""
    from daar.gate1 import load_frame, query_level_frame

    table = build_query_table(split, variant=THESIS_VARIANT)
    if not (hybrid_structure or use_residual_cost or use_trace_gate1b):
        return table

    frame = load_frame(split)
    query = query_level_frame(frame)
    if use_trace_gate1b:
        from daar.v2.features import attach_query_trace_features

        traces = pd.read_parquet(daar_trace_skeleton_path())
        query = attach_query_trace_features(query, traces)

    if hybrid_structure:
        query = apply_hybrid_phi(query, confidence_threshold=confidence_threshold)
        table = _refresh_gate1_scores(
            table, query, models_dir=models_dir, use_trace_gate1b=use_trace_gate1b
        )
        meta = query[
            ["training_id", "structure_confidence", "used_qce_fallback", "structure_source"]
        ]
        table = table.merge(meta, on="training_id", how="left")
    elif use_trace_gate1b:
        table = _refresh_gate1_scores(
            table, query, models_dir=models_dir, use_trace_gate1b=True
        )

    if use_residual_cost:
        cvec_cols = [c for c in query.columns if c.startswith("dim_")]
        if cvec_cols:
            table = table.merge(
                query[["training_id", *cvec_cols]].drop_duplicates("training_id"),
                on="training_id",
                how="left",
                suffixes=("", "_phi"),
            )
        traces = pd.read_parquet(daar_trace_skeleton_path())
        table = apply_cost_residual(table, traces=traces)

    return table


def route_daar2(
    *,
    hybrid_structure: bool = True,
    confidence_threshold: float | None = None,
    use_residual_cost: bool = True,
    use_trace_gate1b: bool = True,
    models_dir: Path | None = None,
    theta_s: float | None = None,
    frozen_lambda: float | None = None,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
) -> dict[str, Any]:
    """Evaluate D-AAR 2.0 on val; compare to thesis-frozen v1 table."""
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()

    _obj, struct_manifest = load_structure_distill(models_dir)
    if confidence_threshold is None:
        confidence_threshold = float(
            struct_manifest.get("default_confidence_threshold", 0.35)
        )

    _gate1a_pipe, gate1a_manifest, _ = load_gate1a(models_dir)
    if theta_s is None:
        theta_s = float(
            gate1a_manifest.get("theta_s_joint_regret", gate1a_manifest.get("theta_s", 0.5))
        )
    if frozen_lambda is None:
        frozen_lambda = float(gate1a_manifest.get("lambda_at_theta_joint", 0.0))

    v1_table = build_query_table("val", variant=THESIS_VARIANT)
    v2_table = build_query_table_v2(
        "val",
        hybrid_structure=hybrid_structure,
        confidence_threshold=confidence_threshold,
        use_residual_cost=use_residual_cost,
        use_trace_gate1b=use_trace_gate1b,
        models_dir=models_dir,
    )

    v1_metrics = evaluate_lambda(
        v1_table, frozen_lambda, theta_s=theta_s, abstention="p_solv"
    )
    v2_metrics = evaluate_lambda(
        v2_table, frozen_lambda, theta_s=theta_s, abstention="p_solv"
    )

    fallback_rate = None
    if hybrid_structure and "used_qce_fallback" in v2_table.columns:
        fallback_rate = float(v2_table["used_qce_fallback"].mean())

    report: dict[str, Any] = {
        "version": "daar2",
        "hybrid_structure": hybrid_structure,
        "confidence_threshold": confidence_threshold,
        "use_residual_cost": use_residual_cost,
        "use_trace_gate1b": use_trace_gate1b,
        "theta_s": theta_s,
        "lambda": frozen_lambda,
        "val_v1_thesis": v1_metrics,
        "val_v2": v2_metrics,
        "qce_fallback_rate": fallback_rate,
        "structure_distill": {
            "val_mean_mae": struct_manifest.get("val_mean_mae"),
            "mae_p90": struct_manifest.get("mae_p90"),
        },
    }
    if use_residual_cost:
        try:
            res_manifest = json.loads(
                (models_dir / "cost_residual_manifest.json").read_text(encoding="utf-8")
            )
            report["cost_residual"] = res_manifest.get("per_agent")
        except FileNotFoundError:
            report["cost_residual"] = None

    out_path = daar_routing_dir() / "daar2_val_report.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)
    return report
