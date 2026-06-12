"""Gate 3 — cost-efficient adaptive routing.

Deployment problem (NOT "choose the best agent")::

    Choose the cheapest agent that is sufficiently likely to succeed.

Formal objective::

    minimize mean routed cost
    subject to success_rate ≥ α

Canonical Gate 3 policy (**success_threshold**)::

    A(q) = { a : ŝ_a(q) ≥ τ_success }     # sufficiently likely
    â(q) = argmin_{a ∈ A} ĉ_a(q)          # cheapest capable solver

Architecture unchanged (Gates 1a/1b/2 frozen). Alternate Gate 3 encodings:

- **success_threshold** (primary): argmin ĉ among {ŝ ≥ τ}
- **utility** (soft): argmax(ŝ − λĉ); λ tuned on cost, not regret
- **budget**: argmax ŝ among {ĉ ≤ B}

**rank-only** (λ=0 utility) = legacy "pick highest ŝ" ablation — not cost-efficient routing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

from config.global_config.paths import daar_routing_dir
from config.local.constants.agents import ROUTER_AGENTS

Gate3Policy = Literal["utility", "success_threshold", "budget"]
EmptyFeasiblePolicy = Literal["argmax_s_hat", "abstain"]

# Exclude τ=0 (all agents feasible → degenerate always-pick-raw). Fine grid at low τ for coverage.
SUCCESS_THRESHOLD_GRID: tuple[float, ...] = (
    0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95
)
DEFAULT_SUCCESS_THRESHOLD = 0.85
DEFAULT_MIN_SUCCESS_FRACTION = 0.95
DEFAULT_MAX_SUCCESS_LOSS_FRACTION = 0.05  # Option 2: success(λ) ≥ (1−ε)×success(λ=0)
GATE3_OBJECTIVES_JSON = "gate3_cost_objectives_val.json"

COST_EFFICIENT_OBJECTIVE = (
    "minimize mean route_cost_hat (Gate 2 estimate) subject to success ≥ α "
    "(cheapest agent sufficiently likely to succeed)"
)
PRIMARY_COST_METRIC = "mean_route_cost_hat"
CALIBRATION_COST_METRIC = "mean_oracle_cost_usd"
DEFAULT_DEPLOYMENT_POLICY: Gate3Policy = "success_threshold"
DEFAULT_EMPTY_FEASIBLE: EmptyFeasiblePolicy = "abstain"
COST_AWARE_ROUTING_FREEZE_JSON = "cost_aware_routing_freeze.json"
COST_AWARE_ROUTING_FREEZE_VERSION = "v1_cost_efficient"


@dataclass(frozen=True)
class Gate3Decision:
    agent: str | None
    abstained: bool
    policy: Gate3Policy
    meta: dict[str, Any]


def _route_cost(row: pd.Series, agent: str) -> float:
    col = f"route_cost_hat_{agent}"
    if col in row.index and pd.notna(row[col]):
        return float(row[col])
    return float(row[f"c_hat_{agent}"])


def _s_hat(row: pd.Series, agent: str) -> float:
    return float(row[f"s_hat_{agent}"])


def select_utility_agent(row: pd.Series, *, lam: float) -> Gate3Decision:
    """Option 2 routing: â = argmax_a (ŝ_a − λ · ĉ_norm_a)."""
    scores = {
        a: _s_hat(row, a) - lam * float(row[f"route_cost_hat_norm_{a}"])
        for a in ROUTER_AGENTS
    }
    agent = max(scores, key=scores.get)
    return Gate3Decision(
        agent,
        False,
        "utility",
        {
            "lambda": lam,
            "utility_scores": scores,
            "routing_rule": "argmax(s_hat - lambda * route_cost_hat_norm)",
        },
    )


def select_success_threshold_agent(
    row: pd.Series,
    *,
    success_threshold: float,
    empty_feasible: EmptyFeasiblePolicy = DEFAULT_EMPTY_FEASIBLE,
) -> Gate3Decision:
    """Cost-efficient Gate 3: argmin ĉ among {ŝ ≥ τ}; abstain if A is empty."""
    tau = float(success_threshold)
    scores = {a: _s_hat(row, a) for a in ROUTER_AGENTS}
    costs = {a: _route_cost(row, a) for a in ROUTER_AGENTS}
    feasible = [a for a in ROUTER_AGENTS if scores[a] >= tau]
    meta: dict[str, Any] = {
        "tau_success": tau,
        "s_hat": scores,
        "route_cost_hat": costs,
        "feasible_agents": feasible,
        "routing_rule": (
            "if |A|>0: argmin route_cost_hat over A; else: ABSTAIN "
            "(no agent meets minimum confidence τ before spending)"
        ),
        "tau_meaning": (
            "minimum confidence required before paying execution cost "
            "(not raw correctness probability)"
        ),
    }
    if not feasible:
        meta["empty_feasible_policy"] = empty_feasible
        meta["abstained_at"] = "gate3_empty_feasible_set"
        if empty_feasible == "abstain":
            meta["fallback"] = "A_empty: no agent with s_hat >= tau_success"
            return Gate3Decision(None, True, "success_threshold", meta)
        agent = max(ROUTER_AGENTS, key=lambda a: scores[a])
        meta["fallback"] = "A_empty: legacy argmax_s_hat (mixes objectives)"
        return Gate3Decision(agent, False, "success_threshold", meta)

    agent = min(feasible, key=lambda a: (costs[a], -scores[a]))
    meta["chosen_because"] = "min_cost_in_feasible_set"
    return Gate3Decision(agent, False, "success_threshold", meta)


def select_budget_agent(
    row: pd.Series,
    *,
    cost_budget: float,
    empty_feasible: EmptyFeasiblePolicy = "argmax_s_hat",
) -> Gate3Decision:
    """Option 3: â = argmax ŝ_a among {a : ĉ_a ≤ budget}."""
    budget = float(cost_budget)
    scores = {a: _s_hat(row, a) for a in ROUTER_AGENTS}
    costs = {a: _route_cost(row, a) for a in ROUTER_AGENTS}
    feasible = [a for a in ROUTER_AGENTS if costs[a] <= budget]
    meta: dict[str, Any] = {
        "cost_budget": budget,
        "s_hat": scores,
        "route_cost_hat": costs,
        "feasible_agents": feasible,
        "routing_rule": "argmax s_hat among {a : route_cost_hat_a <= budget}",
    }
    if not feasible:
        meta["empty_feasible_policy"] = empty_feasible
        if empty_feasible == "abstain":
            meta["fallback"] = "no_agent_within_budget"
            return Gate3Decision(None, True, "budget", meta)
        agent = max(ROUTER_AGENTS, key=lambda a: scores[a])
        meta["fallback"] = "no_agent_within_budget; used argmax_s_hat"
        return Gate3Decision(agent, False, "budget", meta)

    agent = max(feasible, key=lambda a: scores[a])
    meta["chosen_because"] = "max_s_hat_within_budget"
    return Gate3Decision(agent, False, "budget", meta)


def select_gate3_agent(
    row: pd.Series,
    *,
    policy: Gate3Policy = DEFAULT_DEPLOYMENT_POLICY,
    lam: float = 0.0,
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    cost_budget: float | None = None,
    empty_feasible: EmptyFeasiblePolicy = DEFAULT_EMPTY_FEASIBLE,
) -> Gate3Decision:
    if policy == "success_threshold":
        return select_success_threshold_agent(
            row,
            success_threshold=success_threshold,
            empty_feasible=empty_feasible,
        )
    if policy == "budget":
        if cost_budget is None:
            raise ValueError("cost_budget is required for budget policy")
        return select_budget_agent(row, cost_budget=cost_budget, empty_feasible=empty_feasible)
    return select_utility_agent(row, lam=lam)


def _tuning_cost_column(sweep: pd.DataFrame) -> str:
    """Gate 3 tunes on route_cost_hat; fall back for legacy sweep tables."""
    if PRIMARY_COST_METRIC in sweep.columns:
        return PRIMARY_COST_METRIC
    return CALIBRATION_COST_METRIC


def pick_cost_efficient_threshold(
    sweep: pd.DataFrame,
    *,
    min_success_fraction: float = DEFAULT_MIN_SUCCESS_FRACTION,
    baseline_success: float,
) -> tuple[float, bool]:
    """Option 1 tuning: min route_cost_hat subject to success ≥ α × baseline.

    Returns (tau, meets_success_floor). If no τ meets the floor, pick the τ
    with highest success rate and ``meets_success_floor=False`` (document in freeze).
    """
    cost_col = _tuning_cost_column(sweep)
    floor = float(min_success_fraction) * float(baseline_success)
    eligible = sweep[sweep["success_rate"].astype(float) >= floor]
    if not eligible.empty:
        idx = eligible[cost_col].astype(float).idxmin()
        return float(sweep.loc[idx, "tau_success"]), True
    idx = sweep["success_rate"].astype(float).idxmax()
    return float(sweep.loc[idx, "tau_success"]), False


def pick_cost_efficient_lambda(
    sweep: pd.DataFrame,
    *,
    min_success_fraction: float = DEFAULT_MIN_SUCCESS_FRACTION,
    baseline_success: float,
) -> float:
    """Option 2 tuning: min cost subject to success ≥ α × baseline (NOT min regret)."""
    if "lambda" not in sweep.columns:
        raise ValueError("sweep must include lambda column")
    floor = float(min_success_fraction) * float(baseline_success)
    eligible = sweep[sweep["success_rate"].astype(float) >= floor]
    if eligible.empty:
        idx = sweep["success_rate"].astype(float).idxmax()
        return float(sweep.loc[idx, "lambda"])
    cost_col = _tuning_cost_column(sweep)
    idx = eligible[cost_col].astype(float).idxmin()
    return float(sweep.loc[idx, "lambda"])


def pick_cost_efficient_budget(
    sweep: pd.DataFrame,
    *,
    min_success_fraction: float = DEFAULT_MIN_SUCCESS_FRACTION,
    baseline_success: float,
) -> float:
    """Option 3 tuning: min cost subject to success ≥ α × baseline."""
    if "cost_budget" not in sweep.columns:
        raise ValueError("sweep must include cost_budget column")
    floor = float(min_success_fraction) * float(baseline_success)
    eligible = sweep[sweep["success_rate"].astype(float) >= floor]
    if eligible.empty:
        idx = sweep["success_rate"].astype(float).idxmax()
        return float(sweep.loc[idx, "cost_budget"])
    cost_col = _tuning_cost_column(sweep)
    idx = eligible[cost_col].astype(float).idxmin()
    return float(sweep.loc[idx, "cost_budget"])


def read_gate3_cost_objectives(*, routing_dir=None) -> dict[str, Any]:
    path = (routing_dir or daar_routing_dir()) / GATE3_OBJECTIVES_JSON
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_frozen_gate3_config(*, routing_dir=None) -> dict[str, Any]:
    """Production knobs: cost-efficient success_threshold by default."""
    doc = read_gate3_cost_objectives(routing_dir=routing_dir)
    if not doc:
        return {
            "routing_policy": DEFAULT_DEPLOYMENT_POLICY,
            "deployment_objective": COST_EFFICIENT_OBJECTIVE,
            "lambda": 0.0,
            "success_threshold": DEFAULT_SUCCESS_THRESHOLD,
            "cost_budget": None,
        }
    frozen = doc.get("frozen_production", {})
    return {
        "routing_policy": frozen.get("policy", DEFAULT_DEPLOYMENT_POLICY),
        "deployment_objective": doc.get("deployment_objective", COST_EFFICIENT_OBJECTIVE),
        "lambda": float(frozen.get("lambda", 0.0)),
        "success_threshold": float(
            frozen.get("success_threshold", DEFAULT_SUCCESS_THRESHOLD)
        ),
        "cost_budget": frozen.get("cost_budget"),
    }
