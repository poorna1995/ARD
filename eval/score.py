"""
Eval scoring — outcomes (A/B), metrics, and route scoring CLI.

Universe A: oracle CSV · Universe B: live baseline parquets
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.global_config.paths import oracle_results_csv, decomposer_cache_path
from config.local.constants.datasets import DS, disk_dir_name, resolve_dataset_name
from router.config import (
    AGENTS,
    LIVE_EVAL_BASELINE_ROOT,
    PRODUCTION_FEATURE_SET,
    SELECTIVE_DECOMPOSE_COST_USD,
    SPLIT_PARQUET,
    TARGET,
    feature_set_cvec_version,
    normalize_feature_set,
)
from src.utils.soft_labels import DEFAULT_UTILITY_LAMBDA

from qce.features.build import load_decompose_cache

DEFAULT_ORACLE_PATH = oracle_results_csv()
DEFAULT_ORACLE = DEFAULT_ORACLE_PATH


def _cell_scalar(value: Any) -> Any:
    """One scalar from a cell (duplicate index / merge can yield Series)."""
    if isinstance(value, pd.DataFrame):
        value = value.iloc[-1]
    if isinstance(value, pd.Series):
        value = value.iloc[-1]
    if isinstance(value, np.ndarray):
        value = value.flat[-1]
    return value


def _as_int(value: Any) -> int:
    return int(_cell_scalar(value))


def _as_float(value: Any) -> float:
    return float(_cell_scalar(value))


# --- outcomes ---

# eval_samples stem -> on-disk baseline folder layout (universe B)
DATASET_BASELINE_LAYOUT: dict[str, tuple[str, str]] = {
    ds: (disk_dir_name(ds), ds) for ds in sorted(DS.names)
}


def _baseline_layout(dataset: str) -> tuple[str, str]:
    canonical = resolve_dataset_name(dataset)
    if canonical not in DS.names:
        raise ValueError(f"Unknown dataset {dataset!r}. Known: {sorted(DS.names)}")
    return disk_dir_name(canonical), canonical


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
    folder, name = _baseline_layout(dataset)
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
    out = out.set_index("training_id")
    if out.index.duplicated().any():
        out = out[~out.index.duplicated(keep="last")]
    return out


def _pick_best_agent(row: pd.Series, agents: list[str]) -> str:
    winners = [a for a in agents if _as_int(row[f"correct_{a}"])]
    if winners:
        return min(winners, key=lambda a: _as_float(row[f"cost_{a}"]))
    return min(agents, key=lambda a: _as_float(row[f"cost_{a}"]))


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
        perf = _as_float(_as_int(row[f"correct_{agent}"]))
        cost = _as_float(row[f"cost_{agent}"])
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
            if isinstance(sub, pd.DataFrame):
                sub = sub.iloc[-1]
            rec[f"correct_{agent}"] = _as_int(sub["correct"])
            rec[f"cost_{agent}"] = _as_float(sub["cost_usd"])
            rec[f"pred_{agent}"] = str(_cell_scalar(sub["predicted_answer"]))
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

# --- metrics ---

@dataclass(frozen=True)
class StrategyMetrics:
    name: str
    accuracy: float
    mean_cost_usd: float
    n: int
    oracle_match_rate: float | None = None
    mean_utility_regret: float | None = None
    em_per_usd: float | None = None


def _drop_stale_outcome_cols(df: pd.DataFrame) -> pd.DataFrame:
    prefixes = ("correct_", "cost_", "pred_", "utility_")
    drop = [
        c
        for c in df.columns
        if c.startswith(prefixes)
        or c in ("max_utility", "best_agent", "best_agent_correct", "best_agent_cost_usd")
        or c.startswith("utility_oracle_")
        or c in ("em_oracle_upper",)
    ]
    return df.drop(columns=drop, errors="ignore")


def attach_outcomes(
    df: pd.DataFrame,
    outcomes: pd.DataFrame,
    *,
    decompose_cost_usd: float | pd.Series | None = None,
    feature_set: str | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """Merge outcomes; exec metrics and utility regret for ``router_pred``.

    Agent-only regret: ``max(0, max_utility - u_pred)`` (oracle costs exclude QCE).
    Total-cost regret adds the router's QCE decompose tax:
    ``max(0, max_utility - u_pred + λ × decompose_cost)``.

    Decompose cost resolves here (single path): ``used_decompose`` gate, else
    ``feature_set`` + ``split`` cache lookup, else explicit ``decompose_cost_usd``.
    """
    base = _drop_stale_outcome_cols(df)
    out = base.merge(outcomes, left_on="training_id", right_index=True, how="left")
    if decompose_cost_usd is None and "exec_decompose_cost_usd" in out.columns:
        decompose_cost_usd = out["exec_decompose_cost_usd"]
    fallback = float(
        SELECTIVE_DECOMPOSE_COST_USD if decompose_cost_usd is None else decompose_cost_usd
    )
    if isinstance(decompose_cost_usd, pd.Series):
        decompose = pd.to_numeric(decompose_cost_usd, errors="coerce").reindex(out.index).fillna(0.0)
    elif "used_decompose" in out.columns:
        used = pd.to_numeric(out["used_decompose"], errors="coerce").fillna(0.0)
        if split is not None:
            cache_path = decomposer_cache_path(split)
            if cache_path.is_file():
                cached = load_decompose_cache(cache_path)
                per_query = out["training_id"].astype(str).map(
                    lambda tid: float(cached[tid]["cost_usd"])
                    if tid in cached
                    else fallback
                )
                decompose = used.astype(float) * per_query.astype(float)
            else:
                decompose = used.astype(float) * fallback
        else:
            decompose = used.astype(float) * fallback
    elif feature_set is not None or split is not None:
        fs = feature_set
        if fs is None and "feature_set" in out.columns:
            fs = str(_cell_scalar(out["feature_set"].iloc[0]))
        if fs is None:
            fs = PRODUCTION_FEATURE_SET
        fs = normalize_feature_set(fs)
        if feature_set_cvec_version(fs) is None:
            decompose = pd.Series(0.0, index=out.index)
        elif split is not None:
            cache_path = decomposer_cache_path(split)
            if cache_path.is_file():
                cached = load_decompose_cache(cache_path)
                decompose = out["training_id"].astype(str).map(
                    lambda tid: float(cached[tid]["cost_usd"])
                    if tid in cached
                    else fallback
                )
            else:
                decompose = pd.Series(fallback, index=out.index)
        else:
            decompose = pd.Series(fallback, index=out.index)
    elif decompose_cost_usd is not None:
        decompose = pd.Series(float(decompose_cost_usd), index=out.index)
    else:
        decompose = None
    exec_correct = []
    exec_cost = []
    utility_regret = []
    utility_regret_total = []
    lam = DEFAULT_UTILITY_LAMBDA
    for i, row in out.iterrows():
        pred = str(_cell_scalar(row["router_pred"]))
        exec_correct.append(_as_int(row.get(f"correct_{pred}", 0)))
        exec_cost.append(_as_float(row.get(f"cost_{pred}", np.nan)))
        u_pred = _as_float(row.get(f"utility_{pred}", np.nan))
        u_max = _as_float(row.get("max_utility", np.nan))
        if pd.isna(u_pred) or pd.isna(u_max):
            utility_regret.append(np.nan)
            utility_regret_total.append(np.nan)
        else:
            gap = u_max - u_pred
            utility_regret.append(max(0.0, gap))
            if decompose is not None:
                utility_regret_total.append(max(0.0, gap + lam * float(decompose.loc[i])))
            else:
                utility_regret_total.append(max(0.0, gap))
    out["exec_correct"] = exec_correct
    out["exec_cost_usd"] = exec_cost
    out["utility_regret"] = utility_regret
    out["utility_regret_total"] = utility_regret_total
    if decompose is not None:
        out["exec_decompose_cost_usd"] = decompose
        out["exec_total_cost_usd"] = pd.to_numeric(out["exec_cost_usd"], errors="coerce") + decompose
    if TARGET in out.columns:
        out["oracle_label_match"] = (out["router_pred"] == out[TARGET]).astype(int)
    return out


def _strategy_metrics(
    name: str,
    accs: list[int],
    costs: list[float],
    regrets: list[float],
    oracle_match: list[int] | None,
    n: int,
) -> StrategyMetrics:
    mean_cost = float(pd.Series(costs).mean())
    acc = float(np.mean(accs))
    regret_series = pd.to_numeric(pd.Series(regrets), errors="coerce")
    mean_regret = float(regret_series.mean()) if regret_series.notna().any() else None
    em_per_usd = (acc / mean_cost) if mean_cost > 0 else None
    return StrategyMetrics(
        name=name,
        accuracy=acc,
        mean_cost_usd=mean_cost,
        n=n,
        oracle_match_rate=float(np.mean(oracle_match)) if oracle_match is not None else None,
        mean_utility_regret=mean_regret,
        em_per_usd=em_per_usd,
    )


def metrics_for_agent_column(
    df: pd.DataFrame,
    agent: str,
    *,
    name: str,
) -> StrategyMetrics:
    accs: list[int] = []
    costs: list[float] = []
    regrets: list[float] = []
    oracle_match: list[int] = []
    for _, row in df.iterrows():
        accs.append(_as_int(row.get(f"correct_{agent}", 0)))
        costs.append(_as_float(row.get(f"cost_{agent}", np.nan)))
        u = _as_float(row.get(f"utility_{agent}", np.nan))
        u_max = _as_float(row.get("max_utility", np.nan))
        regrets.append(max(0.0, u_max - u) if pd.notna(u) and pd.notna(u_max) else np.nan)
        oracle_match.append(int(agent == _cell_scalar(row[TARGET])))
    return _strategy_metrics(name, accs, costs, regrets, oracle_match, len(df))


def metrics_for_routed(
    df: pd.DataFrame,
    agent_col: str,
    *,
    name: str,
) -> StrategyMetrics:
    accs: list[int] = []
    costs: list[float] = []
    regrets: list[float] = []
    oracle_match: list[int] = []
    has_oracle_label = TARGET in df.columns
    for _, row in df.iterrows():
        agent = str(_cell_scalar(row[agent_col]))
        accs.append(_as_int(row.get(f"correct_{agent}", 0)))
        costs.append(_as_float(row.get(f"cost_{agent}", np.nan)))
        u = _as_float(row.get(f"utility_{agent}", np.nan))
        u_max = _as_float(row.get("max_utility", np.nan))
        regrets.append(max(0.0, u_max - u) if pd.notna(u) and pd.notna(u_max) else np.nan)
        if has_oracle_label:
            oracle_match.append(int(agent == _cell_scalar(row[TARGET])))
    return _strategy_metrics(name, accs, costs, regrets, oracle_match if has_oracle_label else None, len(df))


def baseline_strategies(df: pd.DataFrame) -> list[StrategyMetrics]:
    rows = [metrics_for_agent_column(df, a, name=f"always_{a}") for a in AGENTS]
    rows.append(metrics_for_routed(df, "router_pred", name="router_argmax"))
    return rows

# --- score ---

def outcomes_from_baselines(
    dataset: str,
    training_ids: pd.Index | list[str],
    *,
    baseline_root: Path = LIVE_EVAL_BASELINE_ROOT,
) -> tuple[pd.DataFrame, list[str]]:
    """Wide per-id outcomes from four single-agent baseline runs (universe B)."""
    return from_live_baselines_for_ids(
        dataset, training_ids, baseline_root=baseline_root
    )


def outcomes_from_oracle(
    training_ids: pd.Index | list[str],
    *,
    oracle_path: Path = DEFAULT_ORACLE,
) -> pd.DataFrame:
    """Universe A outcomes matrix."""
    return from_oracle_csv(oracle_path, training_ids)


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
    feature_set: str | None = None,
    split: str | None = None,
    decompose_cost_usd: float = SELECTIVE_DECOMPOSE_COST_USD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Attach ``exec_correct``, ``exec_cost_usd``, ``utility_regret``; return metrics summary.

    ``routes`` must have ``training_id`` and ``pred_col`` (defaults to ``router_pred``).
    When ``used_decompose`` is absent, infer from ``feature_set`` (or production ``cvec5_emb``).
    ``split`` resolves per-query decompose from ``decomposer_cache_path`` (QCE splits or eval dataset id).
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

    scored = attach_outcomes(
        work,
        outcomes,
        decompose_cost_usd=decompose_cost_usd,
        feature_set=feature_set,
        split=split,
    )
    m = metrics_for_routed(scored, "router_pred", name=policy)
    regret_total = pd.to_numeric(scored["utility_regret_total"], errors="coerce")
    mean_utility_regret_total = (
        float(regret_total.mean()) if regret_total.notna().any() else None
    )
    n = len(scored)
    dist = routing_distribution(scored)

    agent_cost = pd.to_numeric(scored["exec_cost_usd"], errors="coerce")
    mean_agent_usd = float(agent_cost.mean()) if n else None
    if "exec_decompose_cost_usd" in scored.columns:
        decompose_cost = pd.to_numeric(scored["exec_decompose_cost_usd"], errors="coerce")
        total_cost = pd.to_numeric(scored["exec_total_cost_usd"], errors="coerce")
        mean_decompose_usd = float(decompose_cost.mean()) if n else None
        mean_total_usd = float(total_cost.mean()) if n else None
        pct_decompose = (
            round(100.0 * float((decompose_cost > 0).mean()), 1) if n else None
        )
    else:
        mean_decompose_usd = None
        mean_total_usd = mean_agent_usd
        pct_decompose = None

    summary: dict[str, Any] = {
        "policy": policy,
        "n_routes": int(len(routes)),
        "n_scored": n,
        "em_pct": round(m.accuracy * 100, 2) if m.accuracy is not None else None,
        "mean_cost_usd": round(mean_agent_usd, 6) if mean_agent_usd is not None else None,
        "musd": round(mean_agent_usd * 1000, 4) if mean_agent_usd is not None else None,
        "mean_decompose_usd": round(mean_decompose_usd, 6) if mean_decompose_usd is not None else None,
        "mean_total_usd": round(mean_total_usd, 6) if mean_total_usd is not None else None,
        "total_musd": round(mean_total_usd * 1000, 4) if mean_total_usd is not None else None,
        "pct_decompose": pct_decompose,
        "mean_utility_regret": round(m.mean_utility_regret, 6)
        if m.mean_utility_regret is not None
        else None,
        "mean_utility_regret_total": round(mean_utility_regret_total, 6)
        if mean_utility_regret_total is not None
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
    feature_set: str | None = None,
    decompose_cost_usd: float = SELECTIVE_DECOMPOSE_COST_USD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    outcomes, missing = outcomes_from_baselines(
        dataset, routes["training_id"], baseline_root=baseline_root
    )
    scored, summary = score_routes(
        routes,
        outcomes,
        policy=policy,
        feature_set=feature_set,
        split=resolve_dataset_name(dataset),
        decompose_cost_usd=decompose_cost_usd,
    )
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
    feature_set: str | None = None,
    decompose_cost_usd: float = SELECTIVE_DECOMPOSE_COST_USD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    outcomes = outcomes_from_oracle(routes["training_id"], oracle_path=oracle_path)
    scored, summary = score_routes(
        routes,
        outcomes,
        policy=policy,
        feature_set=feature_set,
        split=split,
        decompose_cost_usd=decompose_cost_usd,
    )
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
    feature_set: str | None = None,
    decompose_cost_usd: float = SELECTIVE_DECOMPOSE_COST_USD,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Dispatch: ``split`` → oracle; else ``dataset`` → live baselines."""
    if split is not None:
        return score_routed_split(
            routes,
            split,
            oracle_path=oracle_path,
            policy=policy,
            feature_set=feature_set,
            decompose_cost_usd=decompose_cost_usd,
        )
    if dataset is None:
        raise ValueError("provide dataset (live eval) or split (internal QCE)")
    return score_routed_eval(
        routes,
        dataset,
        baseline_root=baseline_root,
        policy=policy,
        feature_set=feature_set,
        decompose_cost_usd=decompose_cost_usd,
    )


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
        f"n={summary['n_scored']} EM={summary['em_pct']}% "
        f"agent_mUSD={summary['musd']} total_mUSD={summary.get('total_musd')} "
        f"regret={summary.get('mean_utility_regret')} source={summary.get('outcome_source')}"
    )
    print("distribution:", summary.get("router_distribution"))
    if args.out:
        write_score_summary(summary, args.out)
        scored.to_parquet(args.out.with_suffix(".scored.parquet"), index=False)
        print(f"Wrote {args.out} and {args.out.with_suffix('.scored.parquet')}")
