"""
Phase 1: Export immutable offline execution dataset from checkpoint JSONL.

Layer architecture:
  offline_dataset_raw.*      — FROZEN observables (re-run agents only if this is wrong)
  offline_dataset_derived.*  — recomputed via scripts/derive_ac_features.py
  offline_dataset_master.*   — raw ⨝ derived convenience join

Usage::

    uv run python scripts/export_offline_dataset_master.py
    uv run python scripts/derive_ac_features.py   # optional alone
"""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.global_config.paths import (
    offline_dataset_dir,
    qce_features_split_dir,
    split_csv_path,
)
from scripts.derive_ac_features import build_derived_frame

DEFAULT_CHECKPOINT = (
    Path(__file__).resolve().parents[1]
    / "logs/benchmark_runs/combined/checkpoint.jsonl"
)
AGENTS: tuple[str, ...] = ("raw", "cot", "react", "multiagent")
TOOL_COL_PREFIX = "n_tool_"
DEFAULT_ALPHA_D = 0.5
DEFAULT_ALPHA_M = 0.5

RAW_PRIORITY_COLS: tuple[str, ...] = (
    # A. Query
    "training_id", "query", "dataset", "split", "label_status",
    "meta_level", "meta_type", "meta_id", "expected_answer",
    "query_word_len", "query_char_len", "has_qce_features",
    # B. Agent
    "agent", "model", "prompt_dataset",
    # C. Success (observed)
    "is_correct", "is_graded", "is_failed", "execution_failed",
    "failure_type", "predicted_answer", "has_final_answer", "error",
    # D. Reasoning (raw counts)
    "num_reasoning_steps", "num_llm_calls", "num_format_retries",
    "steps_taken", "num_steps", "max_steps", "reasoning_chars",
    # E. Tools
    "tool_calls_total", "tools_called", "tool_call_counts",
    # F. Tokens
    "prompt_tokens", "completion_tokens", "total_tokens",
    # G. Latency
    "latency_total", "latency_llm", "latency_tools", "wall_clock_sec",
    # H. Economic
    "cost_usd",
    # I. Multi-agent
    "orchestration_type", "n_sub_agents", "sub_agents_run",
    "n_planner_calls", "n_worker_calls", "n_judge_calls",
    # J. Trace metadata
    "checkpoint_source", "started_at", "finished_at", "retried", "run_index",
    # Traces (large)
    "reasoning_steps_text", "reasoning_steps_json",
    "tools_results_json", "sub_agent_responses_json",
)


def _join_list(values: Any, *, sep: str = "|") -> str:
    if values is None:
        return ""
    if isinstance(values, list):
        return sep.join(str(v) for v in values) if values else ""
    return str(values)


def _tool_counts_str(tools_called: list[str] | None) -> str:
    if not tools_called:
        return ""
    return ";".join(f"{k}:{v}" for k, v in sorted(Counter(str(t) for t in tools_called).items()))


def _reasoning_text(steps: list[Any] | None, *, max_chars: int = 8000) -> str:
    if not steps:
        return ""
    text = "\n".join(f"[{i}] {step}" for i, step in enumerate(steps, start=1))
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _json_compact(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list) and not value:
        return "[]"
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _count_orchestration_roles(sub_agents_run: list[Any] | None) -> tuple[int, int, int]:
    planner = worker = judge = 0
    for item in sub_agents_run or []:
        s = str(item).lower()
        if "planner" in s:
            planner += 1
        elif "judge" in s:
            judge += 1
        elif "worker" in s:
            worker += 1
    return planner, worker, judge


def _tool_name_column(tool: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]+", "_", tool.strip().lower()).strip("_")
    return f"{TOOL_COL_PREFIX}{safe}"


def _parse_tool_counts(tool_call_counts: str) -> dict[str, int]:
    if not tool_call_counts or not isinstance(tool_call_counts, str):
        return {}
    out: dict[str, int] = {}
    for part in tool_call_counts.split(";"):
        if ":" not in part:
            continue
        name, _, count = part.partition(":")
        try:
            out[name.strip()] = int(count)
        except ValueError:
            continue
    return out


def load_checkpoint(path: Path) -> list[dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("training_id")
            if tid is not None:
                by_id[str(tid)] = rec
    return list(by_id.values())


def extract_run_row(
    rec: dict[str, Any],
    agent: str,
    run: dict[str, Any],
    *,
    checkpoint_source: str,
) -> dict[str, Any]:
    ar = run.get("agent_response") or {}
    meta = rec.get("metadata") or {}
    tools_called = ar.get("tools_called") or []
    if not isinstance(tools_called, list):
        tools_called = []
    reasoning_steps = ar.get("reasoning_steps") or []
    if not isinstance(reasoning_steps, list):
        reasoning_steps = []

    raw_correct = ar.get("is_correct")
    is_graded = raw_correct is not None
    is_correct = bool(raw_correct) if is_graded else False
    is_failed = bool(
        run.get("is_failed") if run.get("is_failed") is not None else ar.get("is_failed")
    )
    predicted = ar.get("predicted_answer") or run.get("answer")
    has_final = bool(predicted is not None and str(predicted).strip())

    sub_run = ar.get("sub_agents_run") or []
    n_planner, n_worker, n_judge = _count_orchestration_roles(
        sub_run if isinstance(sub_run, list) else []
    )

    return {
        "training_id": rec.get("training_id"),
        "dataset": rec.get("dataset_source") or meta.get("dataset_source"),
        "prompt_dataset": rec.get("prompt_dataset"),
        "model": rec.get("model"),
        "query": rec.get("query"),
        "expected_answer": rec.get("reference"),
        "meta_id": meta.get("id"),
        "meta_level": meta.get("level"),
        "meta_type": meta.get("type"),
        "meta_split": meta.get("split"),
        "dataset_training_id": meta.get("dataset_training_id"),
        "agent": agent,
        "is_correct": is_correct,
        "is_graded": is_graded,
        "is_failed": is_failed,
        "execution_failed": is_failed,
        "failure_type": ar.get("failure_type"),
        "predicted_answer": predicted,
        "has_final_answer": has_final,
        "prompt_tokens": ar.get("prompt_tokens"),
        "completion_tokens": ar.get("completion_tokens"),
        "total_tokens": ar.get("total_tokens"),
        "cost_usd": ar.get("cost_usd"),
        "tool_calls_total": ar.get("num_tool_calls"),
        "num_tool_calls": ar.get("num_tool_calls"),
        "num_llm_calls": ar.get("num_llm_calls"),
        "num_steps": ar.get("num_steps"),
        "steps_taken": ar.get("steps_taken"),
        "max_steps": ar.get("max_steps"),
        "num_format_retries": ar.get("num_format_retries"),
        "latency_total": ar.get("latency_total"),
        "latency_llm": ar.get("latency_llm"),
        "latency_tools": ar.get("latency_tools"),
        "wall_clock_sec": run.get("wall_clock"),
        "num_reasoning_steps": len(reasoning_steps),
        "reasoning_steps_text": _reasoning_text(reasoning_steps),
        "reasoning_steps_json": _json_compact(reasoning_steps),
        "reasoning_chars": sum(len(str(s)) for s in reasoning_steps),
        "tools_called": _join_list(tools_called, sep="|"),
        "tool_call_counts": _tool_counts_str(tools_called),
        "tools_results_json": _json_compact(ar.get("tools_results") or []),
        "orchestration_type": ar.get("orchestration_type"),
        "sub_agents_run": _join_list(sub_run, sep="|"),
        "n_sub_agents": len(sub_run) if isinstance(sub_run, list) else 0,
        "n_planner_calls": n_planner,
        "n_worker_calls": n_worker,
        "n_judge_calls": n_judge,
        "sub_agent_responses_json": _json_compact(ar.get("sub_agent_responses") or []),
        "error": ar.get("error") or run.get("error"),
        "retried": run.get("retried"),
        "run_index": run.get("run_index"),
        "checkpoint_source": checkpoint_source,
        "started_at": rec.get("started_at"),
        "finished_at": rec.get("finished_at"),
    }


def build_long_frame(
    records: list[dict[str, Any]],
    *,
    checkpoint_source: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for rec in records:
        strategies = rec.get("strategies") or {}
        for agent in AGENTS:
            runs = strategies.get(agent) or []
            if runs:
                rows.append(extract_run_row(rec, agent, runs[0], checkpoint_source=checkpoint_source))
    return pd.DataFrame(rows)


def _split_map() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        path = split_csv_path(split)
        if not path.is_file():
            continue
        part = pd.read_csv(path, usecols=["training_id", "label_status"])
        part["split"] = split
        rows.append(part)
    if not rows:
        return pd.DataFrame(columns=["training_id", "split", "label_status"])
    return pd.concat(rows, ignore_index=True).drop_duplicates("training_id")


def _load_qce_features() -> pd.DataFrame:
    feat_dir = qce_features_split_dir()
    parts: list[pd.DataFrame] = []
    for split in ("train", "val", "test"):
        heur_path = feat_dir / f"query_heuristics_{split}.parquet"
        comp_path = feat_dir / f"complexity_record_{split}.parquet"
        if not heur_path.is_file():
            continue
        heur = pd.read_parquet(heur_path)
        if comp_path.is_file():
            comp = pd.read_parquet(comp_path)
            comp_cols = [c for c in comp.columns if c not in ("training_id", "dataset")]
            heur = heur.merge(comp[["training_id", *comp_cols]], on="training_id", how="left")
        parts.append(heur)
    if not parts:
        return pd.DataFrame(columns=["training_id"])
    return pd.concat(parts, ignore_index=True).drop_duplicates("training_id")


def _add_tool_count_columns(df: pd.DataFrame) -> pd.DataFrame:
    parsed = df["tool_call_counts"].map(_parse_tool_counts)
    all_tools: set[str] = set()
    for counts in parsed:
        all_tools.update(counts.keys())
    out = df.copy()
    for tool in sorted(all_tools):
        col = _tool_name_column(tool)
        out[col] = parsed.map(lambda c, t=tool: c.get(t, 0)).astype(int)
    return out


def build_raw_frame(checkpoint: Path) -> pd.DataFrame:
    checkpoint = checkpoint.resolve()
    records = load_checkpoint(checkpoint)
    df = build_long_frame(records, checkpoint_source=str(checkpoint))

    df = _add_tool_count_columns(df)

    split_df = _split_map()
    df = df.merge(split_df, on="training_id", how="left")
    df["split"] = df["split"].fillna("pool")

    qce_df = _load_qce_features()
    if not qce_df.empty:
        qce_cols = [c for c in qce_df.columns if c != "dataset"]
        df = df.merge(qce_df[qce_cols], on="training_id", how="left")
    df["has_qce_features"] = df["training_id"].isin(qce_df["training_id"]) if not qce_df.empty else False

    if "query" in df.columns:
        q = df["query"].astype(str)
        df["query_word_len"] = q.str.split().str.len()
        df["query_char_len"] = q.str.len()

    rest = [c for c in df.columns if c not in RAW_PRIORITY_COLS]
    ordered = [c for c in RAW_PRIORITY_COLS if c in df.columns] + sorted(rest)
    return df[ordered].sort_values(["training_id", "agent"]).reset_index(drop=True)


def _json_safe_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in df.to_dict(orient="records"):
        clean: dict[str, Any] = {}
        for k, v in row.items():
            if v is None or (isinstance(v, float) and math.isnan(v)):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = float(v)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        records.append(clean)
    return records


def _write_parquet(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def export(
    checkpoint: Path,
    out_dir: Path,
    *,
    alpha_d: float = DEFAULT_ALPHA_D,
    alpha_m: float = DEFAULT_ALPHA_M,
    skip_derived: bool = False,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    raw_df = build_raw_frame(checkpoint)
    paths["raw"] = _write_parquet(raw_df, out_dir / "offline_dataset_raw.parquet")

    if skip_derived:
        return paths

    derived_df, norm_stats = build_derived_frame(raw_df, alpha_d=alpha_d, alpha_m=alpha_m)
    paths["normalization"] = out_dir / "offline_dataset_normalization.json"
    paths["normalization"].write_text(json.dumps(norm_stats, indent=2) + "\n", encoding="utf-8")

    derived_cols = [c for c in derived_df.columns if c not in raw_df.columns or c in ("training_id", "agent")]
    master_df = raw_df.merge(derived_df[derived_cols], on=["training_id", "agent"], how="left")
    paths["master"] = _write_parquet(master_df, out_dir / "offline_dataset_master.parquet")
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description="Export raw + derived offline dataset layers.")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--out-dir", type=Path, default=offline_dataset_dir())
    parser.add_argument("--alpha-d", type=float, default=DEFAULT_ALPHA_D)
    parser.add_argument("--alpha-m", type=float, default=DEFAULT_ALPHA_M)
    parser.add_argument("--raw-only", action="store_true", help="Skip derived layer.")
    args = parser.parse_args()
    if not args.raw_only and abs(args.alpha_d + args.alpha_m - 1.0) > 1e-6:
        parser.error(f"alpha_d + alpha_m must equal 1 (got {args.alpha_d + args.alpha_m})")

    paths = export(
        args.checkpoint,
        args.out_dir,
        alpha_d=args.alpha_d,
        alpha_m=args.alpha_m,
        skip_derived=args.raw_only,
    )
    raw = pd.read_parquet(paths["raw"])

    print("datasets/daar/source/ (parquet only)")
    print(f"  offline_dataset_raw.parquet     rows={len(raw)}  queries={raw.training_id.nunique()}")
    if not args.raw_only:
        print(f"  offline_dataset_master.parquet  {paths['master']}")
        print(f"  offline_dataset_normalization.json  {paths['normalization']}")


if __name__ == "__main__":
    main()
