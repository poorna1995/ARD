"""Score frozen D-AAR P1 and monolithic baseline on eval_samples (Universe B)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.local.constants.agents import ROUTER_AGENTS
from config.local.router.production import PRIMARY_ROUTER_PATH
from daar.routing import load_gate1
from eval.benchmark import list_eval_sample_datasets, load_routable_eval_frame
from eval.score import outcomes_from_baselines
from router.router import attach_router_predictions, load_router

FROZEN_P1_VAL = {
    "mean_agent_regret": 0.09333333333333334,
    "success_rate": 0.5111111111111111,
    "abstention_rate": 0.0,
    "split": "daar_val",
    "n": 225,
}


def _route_frame(
    df: pd.DataFrame,
    pipe: Any,
    feature_cols: list[str],
    *,
    experiment_id: str,
) -> pd.DataFrame:
    out = df.copy()
    out["training_id"] = out["training_id"].astype(str)
    return attach_router_predictions(out, pipe, feature_cols, experiment_id=experiment_id)


def _daar_agent_metrics(routed: pd.DataFrame, outcomes: pd.DataFrame) -> dict[str, Any]:
    """Same regret/success definitions as ``daar.routing`` (no abstention)."""
    merged = routed.merge(outcomes, on="training_id", how="inner")
    if merged.empty:
        raise ValueError("no overlap between routed ids and baseline outcomes")

    regrets: list[float] = []
    successes: list[float] = []
    for _, row in merged.iterrows():
        pred = str(row["router_pred"])
        r_best = max(int(row[f"correct_{a}"]) for a in ROUTER_AGENTS)
        r_pred = int(row[f"correct_{pred}"])
        regrets.append(float(r_best - r_pred))
        successes.append(float(r_pred))

    return {
        "n": len(merged),
        "mean_agent_regret": float(np.mean(regrets)),
        "success_rate": float(np.mean(successes)),
        "abstention_rate": 0.0,
    }


def eval_dataset(
    dataset: str,
    *,
    p1_pipe: Any,
    p1_feature_cols: list[str],
    baseline_pipe: Any,
    baseline_feature_cols: list[str],
    build_features: bool = False,
) -> dict[str, Any]:
    frame = load_routable_eval_frame(dataset, build_features=build_features)
    outcomes, missing = outcomes_from_baselines(dataset, frame["training_id"])

    p1_routed = _route_frame(
        frame,
        p1_pipe,
        p1_feature_cols,
        experiment_id="solvability_p1_soft_kl",
    )
    base_routed = _route_frame(
        frame,
        baseline_pipe,
        baseline_feature_cols,
        experiment_id=str(PRIMARY_ROUTER_PATH.stem),
    )

    return {
        "dataset": dataset,
        "n_eval_queries": len(frame),
        "baselines_missing": missing,
        "p1_frozen_rank": _daar_agent_metrics(p1_routed, outcomes),
        "baseline_hgbm_cvec5_emb_soft_kl": _daar_agent_metrics(base_routed, outcomes),
    }


def eval_daar_on_eval_samples(
    *,
    datasets: list[str] | None = None,
    build_features: bool = False,
    out_path: Path | None = None,
) -> dict[str, Any]:
    datasets = datasets or list_eval_sample_datasets()
    p1_pipe, _p1_manifest, p1_feature_cols = load_gate1("p1")
    baseline_obj = load_router(PRIMARY_ROUTER_PATH)
    baseline_pipe = baseline_obj["pipeline"]
    baseline_feature_cols = list(baseline_obj["feature_cols"])

    per_dataset = [
        eval_dataset(
            name,
            p1_pipe=p1_pipe,
            p1_feature_cols=p1_feature_cols,
            baseline_pipe=baseline_pipe,
            baseline_feature_cols=baseline_feature_cols,
            build_features=build_features,
        )
        for name in datasets
    ]

    def _pool(key: str) -> dict[str, Any]:
        ns = [r[key]["n"] for r in per_dataset]
        total = sum(ns)
        regret = sum(r[key]["mean_agent_regret"] * r[key]["n"] for r in per_dataset) / total
        success = sum(r[key]["success_rate"] * r[key]["n"] for r in per_dataset) / total
        return {
            "n": total,
            "mean_agent_regret": float(regret),
            "success_rate": float(success),
            "abstention_rate": 0.0,
        }

    report: dict[str, Any] = {
        "outcome_source": "eval_samples + live_eval_baselines (Universe B)",
        "note": (
            "P1 rank only (argmax ŝ); no Gate 2 cost_hand on eval_samples. "
            "Matches frozen P1 val routing with θ_s=0, λ≈0."
        ),
        "frozen_p1_daar_val_reference": FROZEN_P1_VAL,
        "per_dataset": per_dataset,
        "pooled": {
            "p1_frozen_rank": _pool("p1_frozen_rank"),
            "baseline_hgbm_cvec5_emb_soft_kl": _pool("baseline_hgbm_cvec5_emb_soft_kl"),
        },
    }

    out_path = out_path or (
        Path(__file__).resolve().parents[1]
        / "datasets/daar/routing/eval_samples_p1_vs_baseline.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", action="append", default=None, help="eval_samples stem (repeatable)")
    p.add_argument("--build-features", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    print(
        json.dumps(
            eval_daar_on_eval_samples(
                datasets=args.dataset,
                build_features=args.build_features,
                out_path=args.out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
