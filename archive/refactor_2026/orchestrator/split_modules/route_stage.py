"""Routing stage helpers (selective QCE, summaries, merge)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from agent.registry import STRATEGY_REGISTRY
from routing.config import (
    DEFAULT_SELECTIVE_TAU,
    SELECTIVE_CHEAP_ROUTER_PATH,
    SELECTIVE_FULL_ROUTER_PATH,
    SELECTIVE_TAU_JSON,
)
from orchestrator.load import EXECUTION_DONE_COLS
from routing.selective import SelectiveGateConfig


def resolve_selective_config(
    *,
    selective_qce: bool,
    selective_tau: float | None,
    selective_tau_margin: float | None,
    selective_cheap_router: str | Path | None,
    selective_full_router: str | Path | None,
) -> SelectiveGateConfig | None:
    if not selective_qce:
        return None
    tau = selective_tau
    if tau is None and SELECTIVE_TAU_JSON.is_file():
        try:
            data = json.loads(SELECTIVE_TAU_JSON.read_text(encoding="utf-8"))
            tau = float(data.get("recommended_tau", DEFAULT_SELECTIVE_TAU))
        except (json.JSONDecodeError, TypeError, ValueError):
            tau = DEFAULT_SELECTIVE_TAU
    if tau is None:
        tau = DEFAULT_SELECTIVE_TAU
    return SelectiveGateConfig(
        tau=tau,
        tau_margin=selective_tau_margin,
        cheap_router_path=Path(selective_cheap_router or SELECTIVE_CHEAP_ROUTER_PATH),
        full_router_path=Path(selective_full_router or SELECTIVE_FULL_ROUTER_PATH),
    )


def routing_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n": len(df),
        "router_distribution": df["router_pred"].value_counts().to_dict()
        if "router_pred" in df.columns
        else {},
    }
    if "selective_path" in df.columns:
        summary["selective_path_distribution"] = df["selective_path"].value_counts().to_dict()
    if "used_decompose" in df.columns:
        summary["pct_decompose"] = round(100.0 * float(df["used_decompose"].mean()), 1)
    if "selective_tau" in df.columns and len(df):
        summary["selective_tau"] = float(df["selective_tau"].iloc[0])
    if "oracle_agent" in df.columns and "router_pred" in df.columns:
        mask = df["oracle_agent"].astype(str).isin(STRATEGY_REGISTRY)
        sub = df[mask]
        summary["n_with_oracle"] = int(len(sub))
        if len(sub):
            summary["route_accuracy_vs_oracle"] = float(
                (sub["router_pred"] == sub["oracle_agent"]).mean()
            )
    if "is_correct" in df.columns:
        summary["execution_accuracy"] = float(
            pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).mean()
        )
    return summary


def row_execution_done(row: Mapping[str, Any], *, retry_failed: bool = False) -> bool:
    """True if this row already has a stored agent answer (skip on resume)."""
    if retry_failed and row.get("is_failed"):
        return False
    for key in EXECUTION_DONE_COLS:
        if str(row.get(key) or "").strip():
            return True
    return False


def done_training_ids(df: pd.DataFrame, *, retry_failed: bool = False) -> set[Any]:
    if df is None or df.empty or "training_id" not in df.columns:
        return set()
    return {
        r["training_id"]
        for r in df.to_dict(orient="records")
        if row_execution_done(r, retry_failed=retry_failed)
    }


def merge_pipeline_results(
    routes_df: pd.DataFrame,
    existing: pd.DataFrame | None,
    new_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge prior + new execution rows in ``routes_df`` order (full route set)."""
    by_id: dict[Any, dict[str, Any]] = {}
    if existing is not None and not existing.empty:
        for rec in existing.to_dict(orient="records"):
            by_id[rec.get("training_id")] = rec
    for rec in new_records:
        by_id[rec.get("training_id")] = rec

    out: list[dict[str, Any]] = []
    for row in routes_df.to_dict(orient="records"):
        tid = row.get("training_id")
        if tid in by_id:
            out.append(by_id[tid])
    return pd.DataFrame.from_records(out)

