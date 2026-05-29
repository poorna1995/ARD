"""
Benchmark eval_samples routing and orchestrator launch helpers.

Step 1 (route-dist): route all ``datasets/eval_samples/*.parquet`` with the frozen
production router and write per-dataset route tables + aggregate distribution.

Step 2 (run): execute agents via ``orchestrator.pipeline`` (see ``main_benchmark``).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from routing.config import (
    DATASET_ALIASES,
    EVAL_SAMPLES_DIR,
    PROBA_COLS,
    PRODUCTION_ROUTER_EXPERIMENT_ID,
    REPO_ROOT,
    ROUTER_MODEL_PATH,
)
from routing.router import (
    AGENTS,
    ensure_eval_features,
    load_eval_parquet,
    load_router_frame,
    merge_feature_tables,
)

BENCHMARK_OUT_ROOT = REPO_ROOT / "results/experiments/soft_logreg_benchmark"
ROUTES_DIR = BENCHMARK_OUT_ROOT / "routes"

DATASET_DISPLAY: dict[str, str] = {
    "hotpot": "HotpotQA",
    "musique": "MuSiQue",
    "math": "MATH",
    "mmlu": "MMLU",
    "mmlu_pro": "MMLU",
    "gaia": "GAIA",
}


def list_eval_sample_datasets() -> list[str]:
    """Parquet stems under ``datasets/eval_samples/`` (sorted)."""
    return sorted(p.stem for p in EVAL_SAMPLES_DIR.glob("*.parquet"))


def load_routable_eval_frame(dataset: str, *, build_features: bool = False) -> pd.DataFrame:
    """Base eval frame + merged QCE complexity / embeddings."""
    base = load_eval_parquet(dataset)
    tag = dataset.strip().lower()
    if build_features:
        base = ensure_eval_features(base, tag, build=True, verbose=False)
        return base
    return merge_feature_tables(base, tag)


def route_eval_dataset(
    dataset: str,
    *,
    router_path: Path = ROUTER_MODEL_PATH,
    build_features: bool = False,
) -> pd.DataFrame:
    """Route one eval_samples parquet; returns frame with ``router_pred`` and proba."""
    df = load_routable_eval_frame(dataset, build_features=build_features)
    return load_router_frame(df, router_path=router_path)


def router_distribution_row(dataset: str, routed: pd.DataFrame) -> dict[str, Any]:
    """One summary row: counts, fractions, mean confidence."""
    key = DATASET_ALIASES.get(dataset, dataset)
    n = len(routed)
    counts = routed["router_pred"].value_counts().to_dict() if n else {}
    row: dict[str, Any] = {
        "dataset_key": key,
        "dataset": DATASET_DISPLAY.get(key, key),
        "n": n,
        "router": str(ROUTER_MODEL_PATH.relative_to(REPO_ROOT))
        if ROUTER_MODEL_PATH.is_relative_to(REPO_ROOT)
        else str(ROUTER_MODEL_PATH),
        "experiment_id": PRODUCTION_ROUTER_EXPERIMENT_ID,
    }
    for agent in AGENTS:
        c = int(counts.get(agent, 0))
        row[f"n_{agent}"] = c
        row[f"pct_{agent}"] = round(100.0 * c / n, 1) if n else 0.0
    if n and "max_prob" in routed.columns:
        row["mean_max_prob"] = float(routed["max_prob"].mean())
        row["median_max_prob"] = float(routed["max_prob"].median())
    else:
        row["mean_max_prob"] = None
        row["median_max_prob"] = None
    return row


def collect_router_distributions(
    datasets: list[str] | None = None,
    *,
    router_path: Path = ROUTER_MODEL_PATH,
    build_features: bool = False,
    save_routes: bool = True,
    out_dir: Path = BENCHMARK_OUT_ROOT,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Route each dataset; return summary table and per-dataset routed frames."""
    names = datasets or list_eval_sample_datasets()
    if not names:
        raise FileNotFoundError(f"No eval parquets in {EVAL_SAMPLES_DIR}")

    routes_root = out_dir / "routes"
    if save_routes:
        routes_root.mkdir(parents=True, exist_ok=True)

    frames: dict[str, pd.DataFrame] = {}
    rows: list[dict[str, Any]] = []
    for name in names:
        routed = route_eval_dataset(name, router_path=router_path, build_features=build_features)
        frames[name] = routed
        rows.append(router_distribution_row(name, routed))
        if save_routes:
            out_cols = [
                "training_id",
                "dataset",
                "query",
                "expected_answer",
                "router_pred",
                "assigned_agent",
                "max_prob",
                "margin_top2",
                "router_second",
            ]
            proba_cols = list(PROBA_COLS)
            keep = [c for c in out_cols + proba_cols if c in routed.columns]
            routed[keep].to_parquet(routes_root / f"{name}_routes.parquet", index=False)

    summary = pd.DataFrame(rows)
    return summary, frames


def _format_distribution_markdown(summary: pd.DataFrame, *, router_label: str) -> str:
    lines = [
        "# Eval-sample router distribution",
        "",
        f"Router: `{router_label}`",
        "",
        "| Dataset | n | raw | cot | react | multiagent | mean max p |",
        "|---------|---|-----|-----|-------|------------|------------|",
    ]
    for _, r in summary.iterrows():
        def _cell(agent: str) -> str:
            return f"{int(r[f'n_{agent}'])} ({r[f'pct_{agent}']:.0f}%)"

        mean_p = "—" if r.get("mean_max_prob") is None or pd.isna(r["mean_max_prob"]) else f"{r['mean_max_prob']:.2f}"
        lines.append(
            f"| {r['dataset']} | {int(r['n'])} | {_cell('raw')} | {_cell('cot')} | "
            f"{_cell('react')} | {_cell('multiagent')} | {mean_p} |"
        )
    lines.append("")
    return "\n".join(lines)


def write_distribution_artifacts(
    summary: pd.DataFrame,
    *,
    out_dir: Path = BENCHMARK_OUT_ROOT,
    router_path: Path = ROUTER_MODEL_PATH,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "router_distribution.csv", index=False)
    meta = {
        "router": str(router_path),
        "experiment_id": PRODUCTION_ROUTER_EXPERIMENT_ID,
        "datasets": summary["dataset_key"].tolist(),
        "rows": summary.to_dict(orient="records"),
    }
    (out_dir / "router_distribution.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    md = _format_distribution_markdown(
        summary, router_label=router_path.name if router_path.is_file() else str(router_path)
    )
    md_path = out_dir / "router_distribution.md"
    md_path.write_text(md, encoding="utf-8")
    return md_path


def print_distribution_table(summary: pd.DataFrame) -> None:
    """Console-friendly distribution table."""
    print("\n=== Router distribution (eval_samples) ===\n")
    cols = ["dataset", "n"] + [f"pct_{a}" for a in AGENTS] + ["mean_max_prob"]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print()


def launch_orchestrator_benchmark(
    dataset: str,
    *,
    routes_path: Path,
    grade: bool = True,
    limit: int | None = None,
    model: str | None = None,
    extra_args: list[str] | None = None,
) -> int:
    """Run agent execution for one dataset from a saved routes parquet."""
    cmd = [
        sys.executable,
        str(REPO_ROOT / "orchestrator/pipeline.py"),
        "--dataset",
        dataset,
        "--routes-path",
        str(routes_path),
        "--grade",
    ]
    if limit is not None:
        cmd.extend(["--limit", str(limit)])
    if model:
        cmd.extend(["--model", model])
    if extra_args:
        cmd.extend(extra_args)
    print(f"\n>>> {' '.join(cmd)}\n", flush=True)
    return subprocess.call(cmd, cwd=REPO_ROOT)


def main_benchmark(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Benchmark eval_samples with production router.")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_dist = sub.add_parser(
        "route-dist",
        help="Route all eval_samples parquets; write distribution + route parquets.",
    )
    p_dist.add_argument(
        "--router",
        type=Path,
        default=ROUTER_MODEL_PATH,
        help=f"Router joblib (default: {ROUTER_MODEL_PATH.name}).",
    )
    p_dist.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Subset of eval datasets (default: all *.parquet).",
    )
    p_dist.add_argument("--out-dir", type=Path, default=BENCHMARK_OUT_ROOT)
    p_dist.add_argument(
        "--build-features",
        action="store_true",
        help="Build QCE features if missing (needs API for decompose/embed).",
    )
    p_dist.add_argument("--no-save-routes", action="store_true")

    p_run = sub.add_parser(
        "run",
        help="Execute agents on saved routes via orchestrator (one dataset per invocation).",
    )
    p_run.add_argument("--dataset", action="append", dest="datasets", required=True)
    p_run.add_argument("--routes-dir", type=Path, default=ROUTES_DIR)
    p_run.add_argument("--grade", action="store_true", default=True)
    p_run.add_argument("--no-grade", action="store_false", dest="grade")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--dry-run", action="store_true", help="Print orchestrator commands only.")

    p_oracle = sub.add_parser(
        "oracle-bounds",
        help="Average + Best Agent EM/mUSD from always-{raw,cot,react,multiagent} baselines.",
    )
    p_oracle.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="Eval dataset stems (default: all eval_samples).",
    )
    p_oracle.add_argument(
        "--baseline-root",
        type=Path,
        default=None,
    )
    p_oracle.add_argument("--out-dir", type=Path, default=None)

    args = p.parse_args(argv)

    if args.cmd == "oracle-bounds":
        from routing.oracle_bounds import main_oracle_bounds

        oracle_argv = []
        if args.datasets:
            for ds in args.datasets:
                oracle_argv.extend(["--dataset", ds])
        if args.baseline_root is not None:
            oracle_argv.extend(["--baseline-root", str(args.baseline_root)])
        if args.out_dir is not None:
            oracle_argv.extend(["--out-dir", str(args.out_dir)])
        main_oracle_bounds(oracle_argv)
        return

    if args.cmd == "route-dist":
        summary, _ = collect_router_distributions(
            args.datasets,
            router_path=args.router,
            build_features=args.build_features,
            save_routes=not args.no_save_routes,
            out_dir=args.out_dir,
        )
        md_path = write_distribution_artifacts(summary, out_dir=args.out_dir, router_path=args.router)
        print_distribution_table(summary)
        print(f"Wrote: {args.out_dir / 'router_distribution.csv'}")
        print(f"       {md_path}")
        if not args.no_save_routes:
            print(f"Routes: {args.out_dir / 'routes'}/")
        return

    for ds in args.datasets:
        routes_path = args.routes_dir / f"{ds.strip().lower()}_routes.parquet"
        if not routes_path.is_file():
            raise FileNotFoundError(f"Missing routes: {routes_path} (run route-dist first)")
        if args.dry_run:
            print(f"Would run orchestrator for {ds} with {routes_path}")
            continue
        code = launch_orchestrator_benchmark(ds, routes_path=routes_path, grade=args.grade, limit=args.limit)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main_benchmark()
