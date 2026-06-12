"""Gate 2 on eval_samples — T(q,a) and ĉ from per-dataset QCE plan caches."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from config.global_config.paths import (
    daar_simulations_dir,
    daar_traces_dir,
    decomposer_cache_path,
    ensure_daar_dirs,
)
from config.local.constants.agents import ROUTER_AGENTS
from daar.cost_hand import simulate_cost_hand
from daar.routing import calibrate_route_cost_norms
from daar.trace_skeleton import build_trace_skeleton
from qce.decompose import load_plans_jsonl


def eval_trace_skeleton_path(tag: str) -> Path:
    return daar_traces_dir() / f"eval_{tag.strip().lower()}_trace_skeleton.parquet"


def eval_cost_hand_path(tag: str) -> Path:
    return daar_simulations_dir() / f"eval_{tag.strip().lower()}_cost_hand.parquet"


def load_frozen_cost_scales() -> dict[str, float]:
    """Train-calibrated scales from daar Gate 2 manifest (do not refit on eval)."""
    manifest = daar_simulations_dir() / "cost_hand_manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(
            f"missing {manifest} — run: uv run python scripts/simulate_daar_cost_hand.py"
        )
    data = json.loads(manifest.read_text(encoding="utf-8"))
    return dict(data["scales"])


def build_eval_trace_skeleton(
    tag: str,
    *,
    plans_path: Path | None = None,
    out_path: Path | None = None,
) -> pd.DataFrame:
    ensure_daar_dirs()
    tag = tag.strip().lower()
    plans_path = plans_path or decomposer_cache_path(tag)
    out_path = out_path or eval_trace_skeleton_path(tag)
    if not plans_path.is_file():
        raise FileNotFoundError(f"missing plans for {tag!r}: {plans_path}")

    plans = load_plans_jsonl(plans_path)
    rows = build_trace_skeleton(plans)
    df = pd.DataFrame(rows)
    df["training_id"] = df["training_id"].astype(str)
    df = df.sort_values(["training_id", "agent"]).reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def build_eval_cost_hand(
    tag: str,
    *,
    traces: pd.DataFrame | None = None,
    traces_path: Path | None = None,
    out_path: Path | None = None,
    scales: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Simulate ĉ on eval queries using frozen daar scales.

    Routing uses ``route_cost_hat = X1_hat + X2_dynamic_hat`` (C_static cancels
    within-query), so C_static is set to 0 for eval IDs without oracle frames.
    """
    ensure_daar_dirs()
    tag = tag.strip().lower()
    if traces is None:
        traces_path = traces_path or eval_trace_skeleton_path(tag)
        if not Path(traces_path).is_file():
            traces = build_eval_trace_skeleton(tag)
        else:
            traces = pd.read_parquet(traces_path)
    traces = traces.copy()
    traces["training_id"] = traces["training_id"].astype(str)

    scales = scales if scales is not None else load_frozen_cost_scales()
    ids = traces["training_id"].drop_duplicates()
    c_static = pd.Series(0.0, index=ids.astype(str))

    sim = simulate_cost_hand(traces, c_static, scales)
    sim["route_cost_hat"] = sim["X1_hat"] + sim["X2_dynamic_hat"]

    out_path = out_path or eval_cost_hand_path(tag)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sim.to_parquet(out_path, index=False)
    return sim


def load_eval_cost_hand(tags: list[str]) -> pd.DataFrame:
    """Concat per-dataset eval cost_hand parquets."""
    frames: list[pd.DataFrame] = []
    for tag in tags:
        path = eval_cost_hand_path(tag)
        if not path.is_file():
            build_eval_cost_hand(tag)
        df = pd.read_parquet(path)
        df["training_id"] = df["training_id"].astype(str)
        df["eval_dataset"] = tag.strip().lower()
        frames.append(df)
    if not frames:
        raise ValueError("no eval tags provided")
    return pd.concat(frames, ignore_index=True)


def _cost_norm_for_agent(norms: dict[str, float], agent: str, *, kind: str) -> float:
    if float(norms.get("cost_norm_mode", 0.0)) > 0:
        key = f"route_cost_{kind}_median_train_{agent}"
        if key in norms:
            return float(norms[key])
    return float(norms[f"route_cost_{kind}_median_train"])


def attach_gate2_costs(
    query_table: pd.DataFrame,
    cost_hand: pd.DataFrame,
    *,
    norms: dict[str, float] | None = None,
    per_agent: bool = True,
) -> pd.DataFrame:
    """Add ``route_cost_hat_*`` and ``route_cost_hat_norm_*`` wide columns."""
    out = query_table.copy()
    out["training_id"] = out["training_id"].astype(str)
    cost = cost_hand.copy()
    cost["training_id"] = cost["training_id"].astype(str)
    if "route_cost_hat" not in cost.columns:
        cost["route_cost_hat"] = cost["X1_hat"] + cost["X2_dynamic_hat"]

    if norms is None:
        norms = calibrate_route_cost_norms(per_agent=per_agent)

    long_df = out[["training_id"]].merge(
        cost[["training_id", "agent", "route_cost_hat", "X1_hat", "X2_dynamic_hat", "c_hat"]],
        on="training_id",
        how="left",
    )
    missing = long_df["route_cost_hat"].isna()
    if missing.any():
        n_miss = int(long_df.loc[missing, "training_id"].nunique())
        raise ValueError(
            f"Gate 2 cost missing for {n_miss} queries after merge — "
            "run scripts/build_eval_cost_hand.py"
        )

    for col, prefix in (
        ("route_cost_hat", "route_cost_hat"),
        ("X1_hat", "X1_hat"),
        ("X2_dynamic_hat", "X2_dynamic_hat"),
        ("c_hat", "c_hat"),
    ):
        wide = long_df.pivot(index="training_id", columns="agent", values=col)
        wide.columns = [f"{prefix}_{c}" for c in wide.columns]
        out = out.drop(columns=[c for c in out.columns if c.startswith(f"{prefix}_")], errors="ignore")
        out = out.merge(wide.reset_index(), on="training_id", how="left")

    for agent in ROUTER_AGENTS:
        hat_norm = _cost_norm_for_agent(norms, agent, kind="hat")
        out[f"route_cost_hat_norm_{agent}"] = out[f"route_cost_hat_{agent}"] / hat_norm
        if f"cost_{agent}" in out.columns:
            live = pd.to_numeric(out[f"cost_{agent}"], errors="coerce").fillna(0.0)
            live_med = float(live.median()) if live.notna().any() else 1.0
            denom = live_med if live_med > 0 else 1.0
            out[f"route_cost_oracle_norm_{agent}"] = live / denom
        else:
            out[f"route_cost_oracle_norm_{agent}"] = out[f"route_cost_hat_norm_{agent}"]

    return out
