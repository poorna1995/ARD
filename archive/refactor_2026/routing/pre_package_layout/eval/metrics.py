"""
Attach routed predictions to outcome matrices; compute strategy metrics.

IN:  routes DataFrame + outcomes wide matrix (from eval.outcomes)
MID: merge · regret · accuracy · cost
OUT: StrategyMetrics · scored DataFrame with exec_* columns
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from routing.config import AGENTS, TARGET


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


def attach_outcomes(df: pd.DataFrame, outcomes: pd.DataFrame) -> pd.DataFrame:
    """Merge outcomes; exec metrics and utility regret for ``router_pred``."""
    base = _drop_stale_outcome_cols(df)
    out = base.merge(outcomes, left_on="training_id", right_index=True, how="left")
    exec_correct = []
    exec_cost = []
    utility_regret = []
    for _, row in out.iterrows():
        pred = str(row["router_pred"])
        exec_correct.append(int(row.get(f"correct_{pred}", 0)))
        exec_cost.append(float(row.get(f"cost_{pred}", np.nan)))
        u_pred = float(row.get(f"utility_{pred}", np.nan))
        u_max = float(row.get("max_utility", np.nan))
        if pd.isna(u_pred) or pd.isna(u_max):
            utility_regret.append(np.nan)
        else:
            utility_regret.append(max(0.0, u_max - u_pred))
    out["exec_correct"] = exec_correct
    out["exec_cost_usd"] = exec_cost
    out["utility_regret"] = utility_regret
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
    mean_regret = (
        float(np.nanmean(regrets)) if regrets and not all(pd.isna(regrets) for reg in regrets) else None
    )
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
        accs.append(int(row.get(f"correct_{agent}", 0)))
        costs.append(float(row.get(f"cost_{agent}", np.nan)))
        u = float(row.get(f"utility_{agent}", np.nan))
        u_max = float(row.get("max_utility", np.nan))
        regrets.append(max(0.0, u_max - u) if pd.notna(u) and pd.notna(u_max) else np.nan)
        oracle_match.append(int(agent == row[TARGET]))
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
        agent = str(row[agent_col])
        accs.append(int(row.get(f"correct_{agent}", 0)))
        costs.append(float(row.get(f"cost_{agent}", np.nan)))
        u = float(row.get(f"utility_{agent}", np.nan))
        u_max = float(row.get("max_utility", np.nan))
        regrets.append(max(0.0, u_max - u) if pd.notna(u) and pd.notna(u_max) else np.nan)
        if has_oracle_label:
            oracle_match.append(int(agent == row[TARGET]))
    return _strategy_metrics(name, accs, costs, regrets, oracle_match if has_oracle_label else None, len(df))


def baseline_strategies(df: pd.DataFrame) -> list[StrategyMetrics]:
    rows = [metrics_for_agent_column(df, a, name=f"always_{a}") for a in AGENTS]
    rows.append(metrics_for_routed(df, "router_pred", name="router_argmax"))
    return rows
