"""
Oracle routing bounds from live-eval static baselines (always-{raw,cot,react,multiagent}).

Computes per-dataset:
  - always_{agent} EM / mUSD
  - Average — macro mean over available agents
  - Best Agent — per-query oracle (cheapest correct agent; else cheapest overall)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from routing.config import AGENTS, REPO_ROOT
from routing.benchmark import DATASET_DISPLAY, list_eval_sample_datasets
from routing.router import load_eval_parquet

DEFAULT_BASELINE_ROOT = REPO_ROOT / "results/orchestrator/graph_main_tuned"
DEFAULT_OUT_DIR = REPO_ROOT / "results/reports/live_eval"

# eval_samples stem -> on-disk baseline folder layout
DATASET_BASELINE_LAYOUT: dict[str, tuple[str, str]] = {
    "gaia": ("gaia", "gaia"),
    "hotpot": ("hotpot", "hotpot"),
    "math": ("math", "math"),
    "mmlu": ("mmlu_pro", "mmlu"),
    "musique": ("musique", "musique"),
}


def baseline_parquet_path(
    dataset: str,
    agent: str,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
) -> Path:
    """Path to ``{name}_baseline_{agent}/pipeline_results.parquet``."""
    if dataset not in DATASET_BASELINE_LAYOUT:
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {sorted(DATASET_BASELINE_LAYOUT)}")
    folder, name = DATASET_BASELINE_LAYOUT[dataset]
    return baseline_root / folder / f"{name}_baseline_{agent}" / "pipeline_results.parquet"


def load_baseline_agent(
    dataset: str,
    agent: str,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
) -> pd.DataFrame | None:
    """Load one always-{agent} baseline; None if parquet missing."""
    path = baseline_parquet_path(dataset, agent, baseline_root=baseline_root)
    if not path.is_file():
        return None
    df = pd.read_parquet(path)
    if "training_id" not in df.columns:
        raise KeyError(f"{path}: missing training_id")
    df = df.copy()
    df["training_id"] = df["training_id"].astype(str)
    correct_col = next(c for c in ("is_correct", "response_is_correct") if c in df.columns)
    pred_col = next(c for c in ("response_predicted_answer", "predicted_answer") if c in df.columns)
    out = pd.DataFrame(
        {
            "training_id": df["training_id"],
            "correct": pd.to_numeric(df[correct_col], errors="coerce").fillna(0).astype(int),
            "predicted_answer": df[pred_col].fillna("").astype(str),
            "cost_usd": pd.to_numeric(df["response_cost_usd"], errors="coerce").fillna(0.0),
        }
    )
    if len(out) != len(out.drop_duplicates("training_id")):
        out = out.drop_duplicates("training_id", keep="last")
    return out.set_index("training_id")


def _pick_oracle_agent(row: pd.Series, agents: list[str]) -> str:
    """Cheapest among correct agents; if none correct, cheapest overall."""
    winners = [a for a in agents if int(row[f"correct_{a}"])]
    if winners:
        return min(winners, key=lambda a: float(row[f"cost_{a}"]))
    return min(agents, key=lambda a: float(row[f"cost_{a}"]))


def build_oracle_frame(
    dataset: str,
    *,
    baseline_root: Path = DEFAULT_BASELINE_ROOT,
    agents: tuple[str, ...] = AGENTS,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Join baseline runs on ``training_id``.

    Returns (wide frame, agents loaded, agents missing).
    """
    loaded: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for agent in agents:
        frame = load_baseline_agent(dataset, agent, baseline_root=baseline_root)
        if frame is None:
            missing.append(agent)
            continue
        loaded[agent] = frame

    if not loaded:
        raise FileNotFoundError(
            f"{dataset}: no baseline parquets under {baseline_root} "
            f"(expected always-{{{','.join(agents)}}} runs)."
        )

    present = [a for a in agents if a in loaded]
    ids = sorted(set.intersection(*[set(loaded[a].index) for a in present]))
    if not ids:
        raise ValueError(f"{dataset}: no overlapping training_id across loaded baselines.")

    rows: list[dict[str, Any]] = []
    for tid in ids:
        rec: dict[str, Any] = {"training_id": tid, "dataset": dataset}
        for agent in present:
            sub = loaded[agent].loc[tid]
            rec[f"correct_{agent}"] = int(sub["correct"])
            rec[f"cost_{agent}"] = float(sub["cost_usd"])
            rec[f"pred_{agent}"] = str(sub["predicted_answer"])
        rec["oracle_agent"] = _pick_oracle_agent(pd.Series(rec), present)
        rec["oracle_correct"] = int(rec[f"correct_{rec['oracle_agent']}"])
        rec["oracle_cost_usd"] = float(rec[f"cost_{rec['oracle_agent']}"])
        rows.append(rec)

    return pd.DataFrame(rows), present, missing


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

    oracle_pick_counts = frame["oracle_agent"].value_counts().to_dict()
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
        "best_agent_em_pct": round(frame["oracle_correct"].mean() * 100, 2),
        "best_agent_musd": round(frame["oracle_cost_usd"].mean() * 1000, 4),
        "oracle_pick_counts": {k: int(v) for k, v in oracle_pick_counts.items()},
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
                "agent": "oracle",
                "n": summary["n"],
                "em_pct": summary["best_agent_em_pct"],
                "musd": summary["best_agent_musd"],
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
    print("\n=== Oracle bounds (Average / Best Agent) ===\n")
    show = summary_df[
        [
            "dataset_display",
            "n",
            "average_em_pct",
            "average_musd",
            "best_agent_em_pct",
            "best_agent_musd",
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
