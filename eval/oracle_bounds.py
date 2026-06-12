"""
Oracle routing bounds from live-eval static baselines (always-{raw,cot,react,multiagent}).

Universe: B (live eval physical baselines)
IN:  pipeline_results.parquet under LIVE_EVAL_BASELINE_ROOT
MID: join four agents · compute Average / Best Agent / utility Oracle
OUT: oracle_bounds.csv · oracle_bounds_long.csv (Table 5 inputs)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from router.config import AGENTS, LIVE_EVAL_BASELINE_ROOT, REPO_ROOT
from eval.benchmark import list_eval_sample_datasets
from router.config import DATASET_DISPLAY
from eval.score import (
    DATASET_BASELINE_LAYOUT,
    baseline_parquet_path,
    build_oracle_frame,
    load_baseline_agent,
)
from router.router import load_eval_parquet
from src.utils.soft_labels import DEFAULT_UTILITY_LAMBDA

DEFAULT_BASELINE_ROOT = LIVE_EVAL_BASELINE_ROOT
DEFAULT_OUT_DIR = REPO_ROOT / "results/reports/live_eval"


def summarize_oracle_frame(
    frame: pd.DataFrame,
    agents: list[str],
    *,
    missing: list[str] | None = None,
) -> dict[str, Any]:
    """Dataset-level Average + Best Agent + per-agent always metrics."""
    n = len(frame)
    per_agent: dict[str, dict[str, float]] = {}
    for agent in agents:
        per_agent[agent] = {
            "em_pct": round(frame[f"correct_{agent}"].mean() * 100, 2),
            "musd": round(frame[f"cost_{agent}"].mean() * 1000, 4),
        }

    avg_em = sum(per_agent[a]["em_pct"] for a in agents) / len(agents)
    avg_musd = sum(per_agent[a]["musd"] for a in agents) / len(agents)

    best_agent_pick_counts = frame["best_agent"].value_counts().to_dict()
    utility_oracle_pick_counts = frame["utility_oracle_agent"].value_counts().to_dict()
    eval_n: int | None = None
    try:
        eval_n = len(load_eval_parquet(str(frame["dataset"].iloc[0])))
    except Exception:
        eval_n = None

    return {
        "dataset": frame["dataset"].iloc[0] if n else "",
        "dataset_display": DATASET_DISPLAY.get(frame["dataset"].iloc[0], frame["dataset"].iloc[0])
        if n
        else "",
        "n": n,
        "eval_n": eval_n,
        "n_complete": eval_n is not None and n >= eval_n,
        "agents_used": agents,
        "agents_missing": missing or [],
        "always": per_agent,
        "average_em_pct": round(avg_em, 2),
        "average_musd": round(avg_musd, 4),
        "best_agent_em_pct": round(frame["best_agent_correct"].mean() * 100, 2),
        "best_agent_musd": round(frame["best_agent_cost_usd"].mean() * 1000, 4),
        "oracle_em_pct": round(frame["em_oracle_upper"].mean() * 100, 2),
        "oracle_utility_em_pct": round(frame["utility_oracle_correct"].mean() * 100, 2),
        "oracle_utility_musd": round(frame["utility_oracle_cost_usd"].mean() * 1000, 4),
        "utility_lambda": DEFAULT_UTILITY_LAMBDA,
        "best_agent_pick_counts": {k: int(v) for k, v in best_agent_pick_counts.items()},
        "utility_oracle_pick_counts": {k: int(v) for k, v in utility_oracle_pick_counts.items()},
    }


def collect_oracle_bounds(
    datasets: list[str] | None = None,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    agents: tuple[str, ...] = AGENTS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    """
    Compute bounds for each dataset.

    Returns (summary_df, always_long_df, per_dataset_oracle_frames).
    """
    names = datasets or list_eval_sample_datasets()
    summary_rows: list[dict[str, Any]] = []
    always_rows: list[dict[str, Any]] = []
    frames: dict[str, pd.DataFrame] = {}

    for ds in names:
        if ds not in DATASET_BASELINE_LAYOUT:
            continue
        try:
            frame, present, missing = build_oracle_frame(ds, baseline_root=baseline_root, agents=agents)
        except FileNotFoundError as exc:
            print(f"SKIP {ds}: {exc}")
            continue
        frames[ds] = frame
        summary = summarize_oracle_frame(frame, present, missing=missing)
        summary_rows.append(summary)
        for agent in present:
            always_rows.append(
                {
                    "dataset": ds,
                    "dataset_display": summary["dataset_display"],
                    "method": f"always_{agent}",
                    "agent": agent,
                    "n": summary["n"],
                    "em_pct": summary["always"][agent]["em_pct"],
                    "musd": summary["always"][agent]["musd"],
                }
            )
        always_rows.append(
            {
                "dataset": ds,
                "dataset_display": summary["dataset_display"],
                "method": "average",
                "agent": "",
                "n": summary["n"],
                "em_pct": summary["average_em_pct"],
                "musd": summary["average_musd"],
            }
        )
        always_rows.append(
            {
                "dataset": ds,
                "dataset_display": summary["dataset_display"],
                "method": "best_agent",
                "agent": "best_agent",
                "n": summary["n"],
                "em_pct": summary["best_agent_em_pct"],
                "musd": summary["best_agent_musd"],
            }
        )
        always_rows.append(
            {
                "dataset": ds,
                "dataset_display": summary["dataset_display"],
                "method": "oracle",
                "agent": "utility_oracle",
                "n": summary["n"],
                "em_pct": summary["oracle_utility_em_pct"],
                "musd": summary["oracle_utility_musd"],
            }
        )
        if missing:
            print(f"NOTE {ds}: missing baselines for {missing} — bounds use {present} only.")
        eval_n = summary.get("eval_n")
        if eval_n is not None and summary["n"] < eval_n:
            print(
                f"WARN {ds}: only {summary['n']}/{eval_n} queries overlap across loaded baselines "
                f"(some runs incomplete)."
            )

    summary_df = pd.DataFrame(summary_rows)
    always_df = pd.DataFrame(always_rows)
    return summary_df, always_df, frames


def _summary_to_flat_csv(summary_df: pd.DataFrame) -> pd.DataFrame:
    """One row per dataset with always_*, average, best_agent columns."""
    rows: list[dict[str, Any]] = []
    for rec in summary_df.to_dict(orient="records"):
        row: dict[str, Any] = {
            "dataset": rec["dataset"],
            "dataset_display": rec["dataset_display"],
            "n": rec["n"],
            "agents_used": ",".join(rec["agents_used"]),
            "agents_missing": ",".join(rec["agents_missing"]),
            "average_em_pct": rec["average_em_pct"],
            "average_musd": rec["average_musd"],
            "best_agent_em_pct": rec["best_agent_em_pct"],
            "best_agent_musd": rec["best_agent_musd"],
            "oracle_em_pct": rec["oracle_em_pct"],
            "oracle_utility_em_pct": rec["oracle_utility_em_pct"],
            "oracle_utility_musd": rec["oracle_utility_musd"],
        }
        for agent, stats in rec["always"].items():
            row[f"always_{agent}_em_pct"] = stats["em_pct"]
            row[f"always_{agent}_musd"] = stats["musd"]
        rows.append(row)
    return pd.DataFrame(rows)


def write_oracle_bounds_artifacts(
    summary_df: pd.DataFrame,
    always_df: pd.DataFrame,
    frames: dict[str, pd.DataFrame],
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    flat = _summary_to_flat_csv(summary_df)
    flat.to_csv(out_dir / "oracle_bounds.csv", index=False)
    always_df.to_csv(out_dir / "oracle_bounds_long.csv", index=False)
    summary_df.to_json(out_dir / "oracle_bounds.json", orient="records", indent=2)
    detail_dir = out_dir / "oracle_bounds_per_query"
    detail_dir.mkdir(parents=True, exist_ok=True)
    for ds, frame in frames.items():
        frame.to_csv(detail_dir / f"{ds}_oracle_picks.csv", index=False)


def print_oracle_bounds_table(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        print("No oracle bounds computed (missing baseline parquets).")
        return
    print("\n=== Oracle bounds (Average / Best Agent / Oracle) ===\n")
    show = summary_df[
        [
            "dataset_display",
            "n",
            "average_em_pct",
            "average_musd",
            "best_agent_em_pct",
            "best_agent_musd",
            "oracle_utility_em_pct",
            "oracle_utility_musd",
            "agents_missing",
        ]
    ].copy()
    show["agents_missing"] = show["agents_missing"].apply(
        lambda xs: ",".join(xs) if isinstance(xs, list) and xs else ""
    )
    print(show.to_string(index=False))
    print()


def main_oracle_bounds(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Average + Best Agent bounds from always-{agent} live eval baselines."
    )
    p.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Eval dataset stems (default: all eval_samples).",
    )
    p.add_argument(
        "--baseline-root",
        type=Path,
        default=DEFAULT_BASELINE_ROOT,
        help=f"Root with {{dataset}}/{{dataset}}_baseline_{{agent}}/ (default: {DEFAULT_BASELINE_ROOT}).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Write CSV/JSON here (default: {DEFAULT_OUT_DIR}).",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=list(AGENTS),
        choices=list(AGENTS),
        help="Agent pool for bounds (default: all four).",
    )
    args = p.parse_args(argv)

    summary_df, always_df, frames = collect_oracle_bounds(
        args.datasets,
        baseline_root=args.baseline_root,
        agents=tuple(args.agents),
    )
    write_oracle_bounds_artifacts(summary_df, always_df, frames, out_dir=args.out_dir)
    print_oracle_bounds_table(summary_df)
    print(f"Wrote: {args.out_dir / 'oracle_bounds.csv'}")
    print(f"       {args.out_dir / 'oracle_bounds_long.csv'}")
    print(f"       {args.out_dir / 'oracle_bounds.json'}")
    print(f"       {args.out_dir / 'oracle_bounds_per_query'}/")


if __name__ == "__main__":
    main_oracle_bounds()
