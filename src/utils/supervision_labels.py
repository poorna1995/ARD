"""Build long-form strategy outcome tables from multi-agent benchmark checkpoints.

Per-query routing labels (``oracle_agent``, soft ``p_*``) are defined in
``src.utils.soft_labels`` via utility argmax ``Perf - λ·Cost`` on
``oracle_results*.csv``. This module only expands checkpoint JSONL into a
long-form matrix (one row per ``training_id`` × strategy) for oracle export.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Long-form matrix: one row per (training_id, strategy) for oracle / routing analysis.
STRATEGY_OUTCOME_COLUMNS: tuple[str, ...] = (
    "training_id",
    "query",
    "strategy",
    "expected_answer",
    "predicted_answer",
    "is_correct",
)


def _iter_strategy_run_rows(
    records: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expand checkpoint records to one dict per (question × strategy × run)."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        base = {
            "training_id": rec.get("training_id"),
            "dataset_source": rec.get("dataset_source"),
            "prompt_dataset": rec.get("prompt_dataset"),
            "query": rec.get("query"),
            "reference": rec.get("reference"),
            "model": rec.get("model"),
        }
        for strategy, runs in (rec.get("strategies") or {}).items():
            if not isinstance(runs, list):
                continue
            for run in runs:
                ar = run.get("agent_response") or {}
                is_failed = bool(run.get("is_failed") or ar.get("is_failed"))
                cost = ar.get("cost_usd")
                try:
                    cost_f = float(cost) if cost is not None else 0.0
                except (TypeError, ValueError):
                    cost_f = 0.0
                lat_ar = ar.get("latency_total")
                try:
                    lat_ar_f = float(lat_ar) if lat_ar is not None else 0.0
                except (TypeError, ValueError):
                    lat_ar_f = 0.0
                wc = run.get("wall_clock")
                try:
                    wc_f = float(wc) if wc is not None else 0.0
                except (TypeError, ValueError):
                    wc_f = 0.0
                latency = lat_ar_f if lat_ar_f > 0 else wc_f

                ic = ar.get("is_correct")
                if ic is None:
                    is_correct = False
                else:
                    is_correct = bool(ic)

                rows.append(
                    {
                        **base,
                        "strategy": str(strategy),
                        "run_index": run.get("run_index"),
                        "is_failed": is_failed,
                        "is_correct": is_correct and not is_failed,
                        "cost_usd": cost_f,
                        "latency_s": latency,
                        "wall_clock_s": wc_f,
                    }
                )
    return rows


def _pick_best_run_in_strategy(g: pd.DataFrame) -> pd.Series:
    """Prefer correct runs; else keep a single representative run for diagnostics."""
    correct = g[g["is_correct"] & ~g["is_failed"]]
    pool = correct if not correct.empty else g
    pool = pool.sort_values(
        by=["cost_usd", "latency_s"],
        ascending=[True, True],
        kind="mergesort",
    )
    return pool.iloc[0]


def build_strategy_outcome_matrix_from_runs_df(runs: pd.DataFrame) -> pd.DataFrame:
    """
    Long-form matrix: one row per (``training_id``, ``strategy``).

    Columns match ``STRATEGY_OUTCOME_COLUMNS`` for oracle / routing analysis.
    When multiple runs exist per (question, strategy), keeps the best run
    (correct over incorrect; then lowest cost, then lowest latency).
    """
    if runs.empty:
        return pd.DataFrame(columns=list(STRATEGY_OUTCOME_COLUMNS))

    df = runs.copy()
    if "is_failed" not in df.columns:
        df["is_failed"] = False
    for col in ("is_correct", "is_failed"):
        df[col] = df[col].fillna(False).astype(bool)

    if "expected_answer" not in df.columns and "reference" in df.columns:
        df["expected_answer"] = df["reference"]

    if "latency_s" not in df.columns:
        lt = pd.to_numeric(df.get("latency_total"), errors="coerce").fillna(0.0)
        wc = pd.to_numeric(df.get("wall_clock"), errors="coerce").fillna(0.0)
        df["latency_s"] = lt.where(lt > 0, wc)

    df["cost_usd"] = pd.to_numeric(df.get("cost_usd"), errors="coerce").fillna(0.0)

    picked: list[pd.Series] = []
    for _key, g in df.groupby(["training_id", "strategy"], sort=False):
        picked.append(_pick_best_run_in_strategy(g))

    out = pd.DataFrame(picked).reset_index(drop=True)
    for col in STRATEGY_OUTCOME_COLUMNS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[list(STRATEGY_OUTCOME_COLUMNS)]


def load_checkpoint_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL checkpoint; keep last record per ``training_id`` (resume-safe)."""
    path = Path(path)
    if not path.exists():
        return []
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            tid = rec.get("training_id") or rec.get("row_id")
            if tid is not None:
                by_id[str(tid)] = rec
    return list(by_id.values())


def write_supervision_parquet(
    labels: pd.DataFrame,
    path: str | Path,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(path, index=False)


def repo_root() -> Path:
    """Research repo root (parent of ``src/``). Works after importlib load."""
    return Path(__file__).resolve().parents[2]


def find_repo_root(*, start: Path | None = None) -> Path:
    """
    Locate repo root when cwd is the repo or ``notebooks/``.

    Walks ``start`` (default: ``Path.cwd()``) and its parent for
    ``src/utils/supervision_labels.py``.
    """
    here = (start or Path.cwd()).resolve()
    for candidate in (here, here.parent):
        if (candidate / "src/utils/supervision_labels.py").is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find research_work repo root from {here!s} "
        f"(expected src/utils/supervision_labels.py)"
    )


def default_checkpoint_path(repo: Path | None = None) -> Path:
    root = repo or repo_root()
    return root / "logs/benchmark_runs/combined/checkpoint.jsonl"

