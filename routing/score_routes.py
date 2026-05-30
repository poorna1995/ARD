"""
Score routed queries from precomputed outcomes (no agent re-runs).

- Live eval: join ``router_pred`` to always-{agent} baseline ``pipeline_results.parquet``.
- Internal QCE splits: join to ``oracle_results`` long table (same as soft_train eval).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from routing.analysis import (
    DEFAULT_ORACLE,
    attach_outcomes,
    metrics_for_routed,
    oracle_outcome_matrix,
)
from routing.config import AGENTS, LIVE_EVAL_BASELINE_ROOT, SPLIT_PARQUET
from src.utils.soft_labels import DEFAULT_UTILITY_LAMBDA


def _add_utility_cols(frame: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    out = frame.copy()
    lam = DEFAULT_UTILITY_LAMBDA
    for agent in agents:
        out[f"utility_{agent}"] = out[f"correct_{agent}"].astype(float) - lam * out[f"cost_{agent}"].astype(
            float
        )
    out["max_utility"] = out[[f"utility_{a}" for a in agents]].max(axis=1)
    return out.set_index("training_id")


def outcomes_from_baselines(
    dataset: str,
    training_ids: pd.Index | list[str],
    *,
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
) -> tuple[pd.DataFrame, list[str]]:
    """Wide per-id outcomes from four single-agent baseline runs."""
    from routing.oracle_bounds import build_oracle_frame

    frame, present, missing = build_oracle_frame(dataset, baseline_root=baseline_root)
    ids = set(str(x) for x in training_ids)
    frame = frame[frame["training_id"].astype(str).isin(ids)]
    if frame.empty:
        raise ValueError(f"{dataset}: no routed ids overlap baseline runs")
    return _add_utility_cols(frame, present), missing


def outcomes_from_oracle(
    training_ids: pd.Index | list[str],
    *,
    oracle_path: Path = DEFAULT_ORACLE,
) -> pd.DataFrame:
    return oracle_outcome_matrix(oracle_path, training_ids)


def routing_distribution(routes: pd.DataFrame, col: str = "router_pred") -> dict[str, int]:
    if col not in routes.columns:
        return {}
    return routes[col].astype(str).value_counts().to_dict()


def score_routes(
    routes: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    policy: str = "router",
    pred_col: str = "router_pred",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Attach ``exec_correct``, ``exec_cost_usd``, ``utility_regret``; return metrics summary.

    ``routes`` must have ``training_id`` and ``pred_col`` (defaults to ``router_pred``).
    """
    if "training_id" not in routes.columns:
        raise KeyError("routes missing training_id")
    if pred_col not in routes.columns and "assigned_agent" in routes.columns:
        routes = routes.copy()
        routes[pred_col] = routes["assigned_agent"]
    if pred_col not in routes.columns:
        raise KeyError(f"routes missing {pred_col!r}")

    work = routes.copy()
    work["training_id"] = work["training_id"].astype(str)
    work[pred_col] = work[pred_col].astype(str)
    if pred_col != "router_pred":
        work["router_pred"] = work[pred_col]

    scored = attach_outcomes(work, outcomes)
    m = metrics_for_routed(scored, "router_pred", name=policy)
    n = len(scored)
    dist = routing_distribution(scored)
    summary: dict[str, Any] = {
        "policy": policy,
        "n_routes": int(len(routes)),
        "n_scored": n,
        "em_pct": round(m.accuracy * 100, 2) if m.accuracy is not None else None,
        "mean_cost_usd": round(m.mean_cost_usd, 6) if m.mean_cost_usd is not None else None,
        "musd": round(m.mean_cost_usd * 1000, 4) if m.mean_cost_usd is not None else None,
        "mean_utility_regret": round(m.mean_utility_regret, 6)
        if m.mean_utility_regret is not None
        else None,
        "router_distribution": dist,
        "router_pct": {a: round(100.0 * dist.get(a, 0) / n, 1) if n else 0.0 for a in AGENTS},
    }
    return scored, summary


def score_routed_eval(
    routes: pd.DataFrame,
    dataset: str,
    *,
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
    policy: str = "router",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcomes, missing = outcomes_from_baselines(
        dataset, routes["training_id"], baseline_root=baseline_root
    )
    scored, summary = score_routes(routes, outcomes, policy=policy)
    if missing:
        summary["baselines_missing"] = missing
    summary["dataset"] = dataset
    summary["outcome_source"] = "live_baselines"
    return scored, summary


def score_routed_split(
    routes: pd.DataFrame,
    split: str,
    *,
    oracle_path: Path = DEFAULT_ORACLE,
    policy: str = "router",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    outcomes = outcomes_from_oracle(routes["training_id"], oracle_path=oracle_path)
    scored, summary = score_routes(routes, outcomes, policy=policy)
    summary["split"] = split
    summary["outcome_source"] = "oracle"
    return scored, summary


def score_routed(
    routes: pd.DataFrame,
    *,
    dataset: str | None = None,
    split: str | None = None,
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
    oracle_path: Path = DEFAULT_ORACLE,
    policy: str = "router",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch: ``split`` → oracle; else ``dataset`` → live baselines."""
    if split is not None:
        return score_routed_split(routes, split, oracle_path=oracle_path, policy=policy)
    if dataset is None:
        raise ValueError("provide dataset (live eval) or split (internal QCE)")
    return score_routed_eval(routes, dataset, baseline_root=baseline_root, policy=policy)


def write_score_summary(summary: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return path


def main_score_routes(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Score saved routes from baseline runs (live eval) or oracle (QCE split)."
    )
    p.add_argument("--routes-path", type=Path, required=True)
    p.add_argument("--dataset", default=None, help="Eval sample stem (gaia, hotpot, …).")
    p.add_argument("--split", choices=tuple(SPLIT_PARQUET), default=None)
    p.add_argument("--out", type=Path, default=None, help="Write lookup JSON (optional).")
    args = p.parse_args(argv)

    routes = pd.read_parquet(args.routes_path)
    scored, summary = score_routed(routes, dataset=args.dataset, split=args.split)
    print(
        f"n={summary['n_scored']} EM={summary['em_pct']}% mUSD={summary['musd']} "
        f"regret={summary.get('mean_utility_regret')} source={summary.get('outcome_source')}"
    )
    print("distribution:", summary.get("router_distribution"))
    if args.out:
        write_score_summary(summary, args.out)
        scored.to_parquet(args.out.with_suffix(".scored.parquet"), index=False)
        print(f"Wrote {args.out} and {args.out.with_suffix('.scored.parquet')}")
