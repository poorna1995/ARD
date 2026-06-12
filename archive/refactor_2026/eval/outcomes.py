"""
Outcome matrices for routing evaluation — universe A (oracle CSV) and B (live baselines).

IN:  oracle_results1.csv · live baseline pipeline_results.parquet
MID: pivot / join per training_id × agent
OUT: wide DataFrame indexed by training_id (correct_*, cost_*, utility_*, max_utility)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.paths import oracle_results_csv
from router.config import AGENTS, LIVE_EVAL_BASELINE_ROOT
from src.utils.soft_labels import DEFAULT_UTILITY_LAMBDA

DEFAULT_ORACLE_PATH = oracle_results_csv()

# eval_samples stem -> on-disk baseline folder layout (universe B)
DATASET_BASELINE_LAYOUT: dict[str, tuple[str, str]] = {
    "gaia": ("gaia", "gaia"),
    "hotpot": ("hotpot", "hotpot"),
    "math": ("math", "math"),
    "mmlu": ("mmlu_pro", "mmlu"),
    "mmlu_pro": ("mmlu_pro", "mmlu"),
    "musique": ("musique", "musique"),
}


def load_oracle_long(path: Path) -> pd.DataFrame:
    """Load long-form oracle CSV (universe A)."""
    df = pd.read_csv(path)
    required = {"training_id", "agent", "is_correct", "cost_usd"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"oracle missing columns: {sorted(missing)}")
    return df[df["agent"].isin(AGENTS)].copy()


def _oracle_agent_perf(r: pd.Series) -> float:
    """Match ``soft_labels.regenerate_oracle_agent_labels`` perf definition."""
    ok = float(pd.to_numeric(r["is_correct"], errors="coerce") or 0.0) > 0
    failed = float(pd.to_numeric(r.get("is_failed"), errors="coerce") or 0.0) > 0
    has_answer = str(r.get("predicted_answer", "") or "").strip() != ""
    return float(ok and not failed and has_answer)


def from_oracle_csv(
    oracle_path: Path,
    training_ids: pd.Index | list[str],
    *,
    utility_lambda: float = DEFAULT_UTILITY_LAMBDA,
) -> pd.DataFrame:
    """Universe A: wide per-id outcomes from ``oracle_results*.csv``."""
    oracle = load_oracle_long(oracle_path)
    oracle = oracle[oracle["training_id"].isin(training_ids)].copy()
    oracle["cost_usd"] = pd.to_numeric(oracle["cost_usd"], errors="coerce").fillna(0.0)

    rows: list[dict[str, Any]] = []
    for tid, grp in oracle.groupby("training_id", sort=False):
        row: dict[str, Any] = {"training_id": tid}
        for agent in AGENTS:
            sub = grp[grp["agent"] == agent]
            if sub.empty:
                row[f"cost_{agent}"] = np.nan
                row[f"correct_{agent}"] = 0
                row[f"utility_{agent}"] = -float(utility_lambda) * 0.0
            else:
                r = sub.iloc[0]
                cost = float(r["cost_usd"])
                perf = _oracle_agent_perf(r)
                row[f"cost_{agent}"] = cost
                row[f"correct_{agent}"] = int(perf)
                row[f"utility_{agent}"] = perf - float(utility_lambda) * cost
        row["max_utility"] = max(float(row[f"utility_{a}"]) for a in AGENTS)
        rows.append(row)
    out = pd.DataFrame(rows)
    return out.set_index("training_id")


def add_utility_columns(frame: pd.DataFrame, agents: list[str]) -> pd.DataFrame:
    """Add ``utility_*`` and ``max_utility``; index by ``training_id``."""
    out = frame.copy()
    lam = DEFAULT_UTILITY_LAMBDA
    for agent in agents:
        out[f"utility_{agent}"] = out[f"correct_{agent}"].astype(float) - lam * out[f"cost_{agent}"].astype(
            float
        )
    out["max_utility"] = out[[f"utility_{a}" for a in agents]].max(axis=1)
    return out.set_index("training_id")


def baseline_parquet_path(
    dataset: str,
    agent: str,
    *,
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
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
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
) -> pd.DataFrame | None:
    """Load one always-{agent} live baseline; None if parquet missing."""
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


def _pick_best_agent(row: pd.Series, agents: list[str]) -> str:
    winners = [a for a in agents if int(row[f"correct_{a}"])]
    if winners:
        return min(winners, key=lambda a: float(row[f"cost_{a}"]))
    return min(agents, key=lambda a: float(row[f"cost_{a}"]))


def _pick_utility_oracle(
    row: pd.Series,
    agents: list[str],
    *,
    utility_lambda: float = DEFAULT_UTILITY_LAMBDA,
) -> str:
    best_agent: str | None = None
    best_utility = float("-inf")
    best_cost = float("inf")
    for agent in agents:
        perf = float(int(row[f"correct_{agent}"]))
        cost = float(row[f"cost_{agent}"])
        utility = perf - float(utility_lambda) * cost
        if (
            best_agent is None
            or utility > best_utility
            or (utility == best_utility and cost < best_cost)
            or (utility == best_utility and cost == best_cost and agent < best_agent)
        ):
            best_agent = agent
            best_utility = utility
            best_cost = cost
    assert best_agent is not None
    return best_agent


def from_live_baselines(
    dataset: str,
    *,
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
    agents: tuple[str, ...] = AGENTS,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """
    Universe B: join always-{agent} orchestrator runs on ``training_id``.

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
        rec["best_agent"] = _pick_best_agent(pd.Series(rec), present)
        rec["best_agent_correct"] = int(rec[f"correct_{rec['best_agent']}"])
        rec["best_agent_cost_usd"] = float(rec[f"cost_{rec['best_agent']}"])
        rec["utility_oracle_agent"] = _pick_utility_oracle(pd.Series(rec), present)
        rec["utility_oracle_correct"] = int(rec[f"correct_{rec['utility_oracle_agent']}"])
        rec["utility_oracle_cost_usd"] = float(rec[f"cost_{rec['utility_oracle_agent']}"])
        rec["em_oracle_upper"] = int(any(rec[f"correct_{a}"] for a in present))
        rows.append(rec)

    return pd.DataFrame(rows), present, missing


def from_live_baselines_for_ids(
    dataset: str,
    training_ids: pd.Index | list[str],
    *,
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
) -> tuple[pd.DataFrame, list[str]]:
    """Universe B outcomes matrix for routed ids (scoring)."""
    frame, present, missing = from_live_baselines(dataset, baseline_root=baseline_root)
    ids = set(str(x) for x in training_ids)
    frame = frame[frame["training_id"].astype(str).isin(ids)]
    if frame.empty:
        raise ValueError(f"{dataset}: no routed ids overlap baseline runs")
    return add_utility_columns(frame, present), missing


# Back-compat names (Phase 2 shims — remove in Phase 4)
oracle_outcome_matrix = from_oracle_csv
build_oracle_frame = from_live_baselines
