"""Build per-query supervision targets for routing from multi-agent benchmark runs.

Concept
-------
The README pipeline calls for a *golden routing label* per query: which agent
strategy to use, together with cost and latency signals. After
``run_training_benchmark.py`` checkpoints each question with one or more
strategies, this module turns those executions into **one row per query** with:

- ``supervision_target_strategy`` — among strategies that answered *correctly*,
  the one with lowest estimated dollar cost, then lowest latency (tie-broken by
  a fixed strategy order so labels are deterministic).
- ``supervision_target_cost_usd`` / ``supervision_target_latency_s`` — from the
  chosen run's ``AgentResponse`` (latency falls back to per-run ``wall_clock``).

If no strategy is correct, the target strategy is empty and
``supervision_label_valid`` is False (still useful for filtering or contrastive
training).

Inputs can be checkpoint JSONL records (as produced by ``load_jsonl`` in
``scripts/run_training_benchmark.py``) or a flattened runs table
(``results_runs.parquet`` / ``results.parquet``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

# Prefer cheaper / simpler agents when cost and latency tie exactly.
DEFAULT_STRATEGY_TIEBREAK: tuple[str, ...] = (
    "raw",
    "cot",
    "self_consistency",
    "debate",
    "react",
    "multiagent",
)

# Long-form matrix: one row per (training_id, strategy) for oracle / routing analysis.
STRATEGY_OUTCOME_COLUMNS: tuple[str, ...] = (
    "training_id",
    "query",
    "strategy",
    "expected_answer",
    "predicted_answer",
    "is_correct",
)


def _tiebreak_index(strategy: str, order: tuple[str, ...]) -> int:
    try:
        return order.index(strategy)
    except ValueError:
        return len(order)


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


def build_supervision_labels_from_runs_df(
    runs: pd.DataFrame,
    *,
    strategy_tiebreak: tuple[str, ...] = DEFAULT_STRATEGY_TIEBREAK,
) -> pd.DataFrame:
    """
    One row per ``training_id`` with routing supervision fields.

    Expects columns at minimum:
    ``training_id``, ``strategy``, ``is_correct``, ``is_failed`` (optional),
    ``cost_usd``, ``latency_s`` (or ``latency_total`` + ``wall_clock``).
    """
    if runs.empty:
        return pd.DataFrame()

    df = runs.copy()
    if "is_failed" not in df.columns:
        df["is_failed"] = False
    for col in ("is_correct", "is_failed"):
        df[col] = df[col].fillna(False).astype(bool)

    if "latency_s" not in df.columns:
        lt = pd.to_numeric(df.get("latency_total"), errors="coerce").fillna(0.0)
        wc = pd.to_numeric(df.get("wall_clock"), errors="coerce").fillna(0.0)
        df["latency_s"] = lt.where(lt > 0, wc)

    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0.0)

    per_strategy: list[pd.Series] = []
    for _key, g in df.groupby(["training_id", "strategy"], sort=False):
        per_strategy.append(_pick_best_run_in_strategy(g))
    ps = pd.DataFrame(per_strategy).reset_index(drop=True)

    ps["_order"] = ps["strategy"].map(lambda s: _tiebreak_index(s, strategy_tiebreak))

    out_rows: list[dict[str, Any]] = []
    for tid, g in ps.groupby("training_id", sort=False):
        row0 = g.iloc[0]
        base = {
            "training_id": tid,
            "dataset_source": row0.get("dataset_source"),
            "prompt_dataset": row0.get("prompt_dataset"),
            "query": row0.get("query"),
            "reference": row0.get("reference"),
            "model": row0.get("model"),
            "n_strategies": int(len(g)),
        }

        correct = g[g["is_correct"] & ~g["is_failed"]]
        n_correct = int(correct["strategy"].nunique()) if not correct.empty else 0
        base["n_correct_strategies"] = n_correct
        base["supervision_label_valid"] = n_correct > 0

        if correct.empty:
            base["supervision_target_strategy"] = ""
            base["supervision_target_cost_usd"] = float("nan")
            base["supervision_target_latency_s"] = float("nan")
        else:
            c = correct.sort_values(
                by=["cost_usd", "latency_s", "_order"],
                ascending=[True, True, True],
            )
            pick = c.iloc[0]
            base["supervision_target_strategy"] = str(pick["strategy"])
            base["supervision_target_cost_usd"] = float(pick["cost_usd"])
            base["supervision_target_latency_s"] = float(pick["latency_s"])

        cost = base["supervision_target_cost_usd"]
        lat = base["supervision_target_latency_s"]
        strat = base["supervision_target_strategy"] or None
        base["supervision_target_json"] = json.dumps(
            {
                "strategy": strat,
                "cost_usd": None if cost != cost else float(cost),  # NaN → null
                "latency_s": None if lat != lat else float(lat),
                "valid": base["supervision_label_valid"],
            },
            ensure_ascii=False,
        )

        out_rows.append(base)

    return pd.DataFrame(out_rows)


def build_supervision_labels_from_checkpoint_records(
    records: list[dict[str, Any]],
    *,
    strategy_tiebreak: tuple[str, ...] = DEFAULT_STRATEGY_TIEBREAK,
) -> pd.DataFrame:
    rows = _iter_strategy_run_rows(records)
    if not rows:
        return pd.DataFrame()
    return build_supervision_labels_from_runs_df(
        pd.DataFrame(rows),
        strategy_tiebreak=strategy_tiebreak,
    )


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


def load_supervision_labels(
    checkpoint: str | Path | None = None,
    *,
    strategy_tiebreak: tuple[str, ...] = DEFAULT_STRATEGY_TIEBREAK,
) -> pd.DataFrame:
    """Load checkpoint JSONL and return one supervision row per ``training_id``."""
    path = Path(checkpoint) if checkpoint is not None else default_checkpoint_path()
    records = load_checkpoint_jsonl(path)
    return build_supervision_labels_from_checkpoint_records(
        records,
        strategy_tiebreak=strategy_tiebreak,
    )

