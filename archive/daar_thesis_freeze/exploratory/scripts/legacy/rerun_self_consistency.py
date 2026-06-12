#!/usr/bin/env python3
"""
Re-run only ``self_consistency`` on questions already in a benchmark checkpoint.

Other strategies (raw, cot, debate, react, multiagent) are left unchanged.
Rewrites ``checkpoint.jsonl`` and refreshes ``results.json`` / ``results_runs.parquet``.

Usage:
    # Dry-run: show how many questions would be updated
    uv run python scripts/rerun_self_consistency.py --dry-run

    # Replace all self_consistency runs that used 3 paths (default filter)
    uv run python scripts/rerun_self_consistency.py

    # Force re-run self_consistency for every checkpointed question
    uv run python scripts/rerun_self_consistency.py --all

    # Smoke test on 5 questions
    uv run python scripts/rerun_self_consistency.py --limit 5

    # Step 1 only: save old self_consistency rows before replacing (no LLM calls)
    uv run python scripts/rerun_self_consistency.py --archive-only
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_training_benchmark import (  # noqa: E402
    _run_strategy_runs,
    _strategy_kwargs,
    flatten_to_runs_df,
    flush_exports,
    load_jsonl,
)

STRATEGY = "self_consistency"


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _existing_num_paths(rec: dict[str, Any]) -> int | None:
    runs = (rec.get("strategies") or {}).get(STRATEGY) or []
    if not runs:
        return None
    ar = runs[0].get("agent_response") or {}
    n = ar.get("num_llm_calls")
    return int(n) if n is not None else None


def archive_self_consistency_runs(
    records: list[dict[str, Any]],
    indices: list[int],
    out_dir: Path,
    *,
    target_paths: int,
) -> tuple[Path, Path]:
    """
    Save self_consistency runs that will be replaced (JSONL + flat parquet).
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    archive_dir = out_dir / "archives"
    archive_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{len(indices)}q_{target_paths}paths_target"
    jsonl_path = archive_dir / f"self_consistency_before_rerun_{tag}_{ts}.jsonl"

    archived_rows: list[dict[str, Any]] = []
    flatten_records: list[dict[str, Any]] = []
    for i in indices:
        rec = records[i]
        sc_runs = (rec.get("strategies") or {}).get(STRATEGY)
        archived_rows.append(
            {
                "training_id": rec.get("training_id"),
                "dataset_source": rec.get("dataset_source"),
                "prompt_dataset": rec.get("prompt_dataset"),
                "query": rec.get("query"),
                "reference": rec.get("reference"),
                "model": rec.get("model"),
                "metadata": rec.get("metadata"),
                "num_llm_calls_before": _existing_num_paths(rec),
                "self_consistency_runs": sc_runs,
                "archived_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        flatten_records.append(
            {
                "training_id": rec.get("training_id"),
                "dataset_source": rec.get("dataset_source"),
                "prompt_dataset": rec.get("prompt_dataset"),
                "query": rec.get("query"),
                "reference": rec.get("reference"),
                "model": rec.get("model"),
                "metadata": rec.get("metadata") or {},
                "strategies": {STRATEGY: sc_runs or []},
            }
        )

    write_jsonl(jsonl_path, archived_rows)
    parquet_path = jsonl_path.with_suffix(".parquet")
    df = flatten_to_runs_df(flatten_records)
    if not df.empty:
        df.to_parquet(parquet_path, index=False)
    else:
        parquet_path = jsonl_path.with_suffix(".parquet")
        pd.DataFrame().to_parquet(parquet_path, index=False)

    return jsonl_path, parquet_path


def _should_rerun(rec: dict[str, Any], *, target_paths: int, rerun_all: bool) -> bool:
    if rerun_all:
        return True
    runs = (rec.get("strategies") or {}).get(STRATEGY) or []
    if runs:
        run = runs[0]
        ar = run.get("agent_response") or {}
        if run.get("is_failed") or ar.get("is_failed"):
            return True
        if not (run.get("answer") or ar.get("predicted_answer") or ar.get("answer")):
            return True
    n = _existing_num_paths(rec)
    if n is None:
        return True
    return n != target_paths


def _rerun_one(
    rec: dict[str, Any],
    *,
    model: str,
    num_paths: int,
    sample_temperature: float,
    temperature: float,
    max_tokens: int,
    seed: int,
    retry: bool,
    print_lock: threading.Lock | None,
) -> dict[str, Any]:
    tid = str(rec["training_id"])
    dataset = str(rec.get("prompt_dataset") or rec.get("dataset_source") or "math")
    query = str(rec["query"])
    reference = str(rec["reference"])

    skw = _strategy_kwargs(
        STRATEGY,
        expected=reference,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        num_paths=num_paths,
        sample_temperature=sample_temperature,
        react_max_steps=8,
    )
    _, runs = _run_strategy_runs(
        STRATEGY,
        query=query,
        model=model,
        dataset=dataset,
        run_kwargs=skw,
        retry=retry,
        runs=1,
        progress_label=tid,
        print_lock=print_lock,
    )
    rec = dict(rec)
    strategies = dict(rec.get("strategies") or {})
    strategies[STRATEGY] = runs
    rec["strategies"] = strategies
    rec["self_consistency_rerun_at"] = datetime.now(timezone.utc).isoformat()
    return rec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "logs/benchmark_runs/combined_raw",
        help="Benchmark run directory with checkpoint.jsonl",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--num-paths", type=int, default=5)
    parser.add_argument("--sample-temperature", type=float, default=0.7)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--retry-on-fail", action="store_true")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Re-run self_consistency for every question (not only num_paths != target)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print counts only; do not call LLM or write files",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help="Only save old self_consistency runs to archives/; do not rerun or edit checkpoint",
    )
    parser.add_argument(
        "--skip-archive",
        action="store_true",
        help="Do not write archives/ before rerun (not recommended)",
    )
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    checkpoint_path = out_dir / "checkpoint.jsonl"
    if not checkpoint_path.exists():
        print(f"No checkpoint: {checkpoint_path}", file=sys.stderr)
        return 1

    records = load_jsonl(checkpoint_path, dedupe=True)
    records.sort(key=lambda r: str(r.get("training_id", "")))

    model = args.model or (records[0].get("model") if records else "gpt-4o-mini")
    if model is None:
        model = "gpt-4o-mini"

    to_update_idx = [
        i
        for i, rec in enumerate(records)
        if _should_rerun(rec, target_paths=args.num_paths, rerun_all=args.all)
    ]
    if args.limit is not None:
        to_update_idx = to_update_idx[: args.limit]

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Total questions: {len(records)}")
    print(f"Will update self_consistency on: {len(to_update_idx)}")
    print(f"  num_paths={args.num_paths}, sample_temperature={args.sample_temperature}")
    print(f"  model={model}, workers={args.workers}")

    if not to_update_idx:
        print("Nothing to do.")
        return 0

    if args.dry_run:
        paths = [_existing_num_paths(records[i]) for i in to_update_idx[:20]]
        print(f"  sample existing num_llm_calls: {paths}")
        archive_dir = out_dir / "archives"
        print(f"  on run, would archive to: {archive_dir}/self_consistency_before_rerun_*.jsonl")
        return 0

    if not args.skip_archive:
        jsonl_arc, parquet_arc = archive_self_consistency_runs(
            records,
            to_update_idx,
            out_dir,
            target_paths=args.num_paths,
        )
        print(f"Archived {len(to_update_idx)} old self_consistency runs:")
        print(f"  {jsonl_arc}")
        print(f"  {parquet_arc}")
    else:
        print("Skipping per-strategy archive (--skip-archive)")

    if args.archive_only:
        print("Archive-only mode — checkpoint unchanged.")
        return 0

    backup = checkpoint_path.with_suffix(
        f".jsonl.bak.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    shutil.copy2(checkpoint_path, backup)
    print(f"Full checkpoint backup: {backup}")

    print_lock = threading.Lock() if args.workers > 1 else None
    t0 = time.perf_counter()

    def _task(idx: int) -> tuple[int, dict[str, Any]]:
        return idx, _rerun_one(
            records[idx],
            model=str(model),
            num_paths=args.num_paths,
            sample_temperature=args.sample_temperature,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            seed=args.seed,
            retry=args.retry_on_fail,
            print_lock=print_lock,
        )

    if args.workers <= 1:
        for idx in to_update_idx:
            _, records[idx] = _task(idx)
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(_task, idx) for idx in to_update_idx]
            for fut in as_completed(futures):
                idx, updated = fut.result()
                records[idx] = updated

    write_jsonl(checkpoint_path, records)
    flush_exports(out_dir, records)
    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s — updated {len(to_update_idx)} questions")
    print(f"  {out_dir / 'checkpoint.jsonl'}")
    print(f"  {out_dir / 'results_runs.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
