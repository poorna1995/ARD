"""Gate 3 — D-AAR utility routing (Version A decomposed system).

Canonical **online** flow (research-precise)::

    q → QCE → D1: G(q)     φ(q), emb(q)
         → Gate 1a P(solvable)
         → Gate 1a abstain? (high-confidence unsolvable)  (else STOP)
         → Gate 1b ∥ Gate 2
               Gate 2 = D2: G(q)→T(q,a)  then  D3: T(q,a)→ĉ(q,a)
         → Gate 3: cheapest â with ŝ_â ≥ τ (cost-efficient; legacy: argmax ŝ−λĉ)

Gate 1a decides whether routing happens; Gate 1b and Gate 2 run in parallel only
after the abstention check. Batch helpers (``build_query_table``) may precompute
ŝ and ĉ for all queries; ``route_agent`` applies abstention before Gate 3.

At λ=0, U=ŝ: Gate 2 is architecturally active (ĉ available) but empirically
inactive at the val regret-optimal operating point (cost term unused).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from config.global_config.paths import (
    daar_cost_hand_path,
    daar_frame_path,
    daar_models_dir,
    daar_predictions_path,
    daar_routing_dir,
    ensure_daar_dirs,
)
from config.local.constants.agents import ROUTER_AGENTS
from config.local.router.production import PRIMARY_ROUTER_PATH
from daar.pool_solvability import (
    ABSTENTION_THRESHOLD_GRID,
    POOL_SOLVABILITY_MANIFEST,
    POOL_SOLVABILITY_MODEL,
    SOLVABILITY_PROBABILITY_COL,
    predict_solvability_probability,
    read_abstention_threshold,
    should_abstain,
    sweep_abstention_threshold,
)
from daar.query_frames import (
    AGENT_RANK_MANIFEST,
    AGENT_RANK_MODEL_STEM,
    load_query_frame,
    query_level_frame,
)
from daar.gate3_policy import (
    DEFAULT_SUCCESS_THRESHOLD,
    Gate3Decision,
    Gate3Policy,
    SUCCESS_THRESHOLD_GRID,
    pick_cost_efficient_lambda,
    pick_cost_efficient_threshold,
    select_gate3_agent,
)
from router.router import PROBA_COLS, attach_router_predictions, load_router, predict_agent_proba

# Thesis-frozen: decomposed rank head (Gate 1b) + Gate 1a abstain at route time.
Gate1Variant = Literal["decomposed", "gate1b"]
THESIS_VARIANT: Gate1Variant = "decomposed"
AbstentionMode = Literal["max_s_hat", "solvability_probability", "p_solv"]
CostNormMode = Literal["global", "per_agent"]


def _resolve_abstention_threshold(
    *,
    abstention_threshold: float | None,
    theta_s: float | None = None,
) -> float:
    if abstention_threshold is not None:
        return float(abstention_threshold)
    if theta_s is not None:
        return float(theta_s)
    raise TypeError("abstention_threshold is required")


def _normalize_abstention_mode(mode: str) -> AbstentionMode:
    return "solvability_probability" if mode == "p_solv" else mode  # type: ignore[return-value]


def _read_solvability_score(row: pd.Series) -> float:
    if SOLVABILITY_PROBABILITY_COL in row.index and pd.notna(row[SOLVABILITY_PROBABILITY_COL]):
        return float(row[SOLVABILITY_PROBABILITY_COL])
    if "p_solv" in row.index and pd.notna(row["p_solv"]):
        return float(row["p_solv"])
    raise KeyError(
        f"row missing {SOLVABILITY_PROBABILITY_COL!r} (legacy alias: 'p_solv')"
    )

# λ applies to train-normalized route cost (§3.3 dynamic component, unitless after norm).
LAMBDA_GRID = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0)
AGENT_COLS = tuple(f"r_a_{a}" for a in ROUTER_AGENTS)
S_HAT_COLS = tuple(f"s_hat_{a}" for a in ROUTER_AGENTS)
ROUTE_COST_HAT_COLS = tuple(f"route_cost_hat_{a}" for a in ROUTER_AGENTS)
ROUTE_COST_ORACLE_COLS = tuple(f"route_cost_oracle_{a}" for a in ROUTER_AGENTS)
C_HAT_COLS = tuple(f"c_hat_{a}" for a in ROUTER_AGENTS)
COST_COLS = tuple(f"cost_{a}" for a in ROUTER_AGENTS)


def _rank_variant(variant: Gate1Variant) -> Literal["gate1b"]:
    return "gate1b"


def _manifest_path(variant: Gate1Variant, models_dir: Path) -> Path:
    del variant
    return models_dir / AGENT_RANK_MANIFEST


def _pool_solvability_manifest_path(models_dir: Path) -> Path:
    for name in (
        POOL_SOLVABILITY_MANIFEST,
        "solvability_gate1a_manifest.json",
    ):
        path = models_dir / name
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"missing pool solvability manifest under {models_dir} — "
        "run: uv run python scripts/train_pool_solvability.py"
    )


def load_pool_solvability_detector(
    models_dir: Path | None = None,
    *,
    manifest_name: str | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Load frozen pool-solvability detector — P(solvable | φ, emb)."""
    models_dir = models_dir or daar_models_dir()
    manifest_path = (
        models_dir / manifest_name
        if manifest_name
        else _pool_solvability_manifest_path(models_dir)
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = Path(
        manifest.get(
            "model_path",
            models_dir / f"{POOL_SOLVABILITY_MODEL}.joblib",
        )
    )
    if not model_path.is_file():
        for legacy_name in (
            "solvability_gate1a.joblib",
            "solvability_abstain.joblib",
        ):
            legacy = models_dir / legacy_name
            if legacy.is_file():
                model_path = legacy
                break
        else:
            raise FileNotFoundError(model_path)
    obj = joblib.load(model_path)
    feature_cols = list(manifest.get("feature_cols", obj.get("feature_cols", [])))
    model = obj if "platt" in obj else {"pipeline": obj["pipeline"], "platt": None}
    return model, manifest, feature_cols


def load_abstain(
    models_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Deprecated alias for :func:`load_pool_solvability_detector`."""
    return load_pool_solvability_detector(models_dir)


load_gate1a = load_pool_solvability_detector


def load_gate1_manifest(
    variant: Gate1Variant = THESIS_VARIANT,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    models_dir = models_dir or daar_models_dir()
    path = _manifest_path(variant, models_dir)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path} — run: uv run python scripts/train_gate1b_success_soft.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_agent_rank_model(
    variant: Gate1Variant = THESIS_VARIANT,
    models_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    """Return (agent-ranking pipeline, manifest, feature_cols)."""
    models_dir = models_dir or daar_models_dir()
    manifest = load_gate1_manifest(variant, models_dir)
    feature_cols = list(manifest["feature_cols"])
    path = Path(manifest.get("model_path", models_dir / f"{AGENT_RANK_MODEL_STEM}.joblib"))
    return load_router(path)["pipeline"], manifest, feature_cols


load_gate1 = load_agent_rank_model


def load_frame(split: str) -> pd.DataFrame:
    path = daar_frame_path(split)
    if not path.is_file():
        raise FileNotFoundError(path)
    df = pd.read_parquet(path)
    df["training_id"] = df["training_id"].astype(str)
    return df


def load_cost_hand() -> pd.DataFrame:
    """Gate 2 simulator outputs — includes dynamic route-cost primitives."""
    cost = pd.read_parquet(daar_cost_hand_path())
    cost["training_id"] = cost["training_id"].astype(str)
    # §3.3: C_static cancels within-query; route on w1·X1 + w2·X2_dynamic (w1=w2=1).
    cost["route_cost_hat"] = cost["X1_hat"] + cost["X2_dynamic_hat"]
    return cost[
        ["training_id", "agent", "c_hat", "X1_hat", "X2_dynamic_hat", "route_cost_hat"]
    ]


def calibrate_route_cost_norms(*, per_agent: bool = True) -> dict[str, float]:
    """Train-split medians for unitless λ (never val/test)."""
    frame = load_frame("train")
    cost = load_cost_hand()
    m = frame.merge(cost, on=["training_id", "agent"], how="inner")
    m["route_cost_oracle"] = m["X1"] + m["X2_dynamic"]
    hat_med = float(m["route_cost_hat"].median())
    oracle_med = float(m["route_cost_oracle"].median())
    norms: dict[str, float] = {
        "route_cost_hat_median_train": hat_med if hat_med > 0 else 1.0,
        "route_cost_oracle_median_train": oracle_med if oracle_med > 0 else 1.0,
        "cost_norm_mode": 1.0 if per_agent else 0.0,
    }
    if per_agent:
        for agent in ROUTER_AGENTS:
            sub = m[m["agent"] == agent]
            a_hat = float(sub["route_cost_hat"].median()) if not sub.empty else hat_med
            a_oracle = float(sub["route_cost_oracle"].median()) if not sub.empty else oracle_med
            norms[f"route_cost_hat_median_train_{agent}"] = a_hat if a_hat > 0 else 1.0
            norms[f"route_cost_oracle_median_train_{agent}"] = a_oracle if a_oracle > 0 else 1.0
    return norms


def _cost_norm_for_agent(norms: dict[str, float], agent: str, *, kind: str) -> float:
    """``kind`` is ``hat`` or ``oracle``."""
    if float(norms.get("cost_norm_mode", 0.0)) > 0:
        key = f"route_cost_{kind}_median_train_{agent}"
        if key in norms:
            return float(norms[key])
    return float(norms[f"route_cost_{kind}_median_train"])


def predict_gate1_long(
    frame: pd.DataFrame,
    gate1: Any,
    feature_cols: list[str],
    *,
    variant: Gate1Variant = THESIS_VARIANT,
) -> pd.DataFrame:
    del variant
    query = frame.drop_duplicates("training_id").reset_index(drop=True)
    proba = predict_agent_proba(gate1, query, feature_cols)
    score_cols = {f"p_{a}": f"_score_{a}" for a in ROUTER_AGENTS}
    wide = query[["training_id"]].join(proba.rename(columns=score_cols))
    long_df = frame.merge(wide, on="training_id", how="left")
    long_df["s_hat"] = long_df.apply(
        lambda r: float(r[f"_score_{r['agent']}"]), axis=1
    )
    return long_df.drop(columns=list(score_cols.values()))


def build_query_table(
    split: str,
    *,
    variant: Gate1Variant = THESIS_VARIANT,
    gate1: Any | None = None,
    feature_cols: list[str] | None = None,
    solvability_model: Any | None = None,
    solvability_feature_cols: list[str] | None = None,
    cost_hand: pd.DataFrame | None = None,
    norms: dict[str, float] | None = None,
    cost_norm: CostNormMode = "per_agent",
) -> pd.DataFrame:
    """One row per query with wide ŝ, ĉ, oracle r_a, and route costs.

    Batch path: precomputes Gate 1b and Gate 2 for every query. Online routing
    should run Gate 1a first and skip Gate 1b ∥ Gate 2 when abstaining.
    """
    frame = load_frame(split)
    frame["daar_split"] = split
    frame["route_cost_oracle"] = frame["X1"] + frame["X2_dynamic"]
    rank_variant = _rank_variant(variant)
    if gate1 is None or feature_cols is None:
        gate1, _manifest, feature_cols = load_gate1(rank_variant)
    if cost_hand is None:
        cost_hand = load_cost_hand()
    if norms is None:
        norms = calibrate_route_cost_norms(per_agent=(cost_norm == "per_agent"))

    long_df = predict_gate1_long(frame, gate1, feature_cols, variant=rank_variant)
    long_df = long_df.merge(cost_hand, on=["training_id", "agent"], how="inner")
    if long_df["route_cost_hat"].isna().any():
        raise ValueError(f"missing route_cost_hat after merge on split={split!r}")

    solvable = (
        long_df.groupby("training_id", sort=False)["solvable"]
        .first()
        .astype(int)
        .rename("solvable")
    )
    r_wide = long_df.pivot(index="training_id", columns="agent", values="r_a")
    r_wide.columns = [f"r_a_{c}" for c in r_wide.columns]
    s_wide = long_df.pivot(index="training_id", columns="agent", values="s_hat")
    s_wide.columns = [f"s_hat_{c}" for c in s_wide.columns]
    c_wide = long_df.pivot(index="training_id", columns="agent", values="c_hat")
    c_wide.columns = [f"c_hat_{c}" for c in c_wide.columns]
    rc_hat_wide = long_df.pivot(index="training_id", columns="agent", values="route_cost_hat")
    rc_hat_wide.columns = [f"route_cost_hat_{c}" for c in rc_hat_wide.columns]
    rc_oracle_wide = long_df.pivot(index="training_id", columns="agent", values="route_cost_oracle")
    rc_oracle_wide.columns = [f"route_cost_oracle_{c}" for c in rc_oracle_wide.columns]
    cost_wide = long_df.pivot(index="training_id", columns="agent", values="cost_usd")
    cost_wide.columns = [f"cost_{c}" for c in cost_wide.columns]

    meta = long_df.drop_duplicates("training_id").set_index("training_id")
    keep_meta = [c for c in ("dataset", "daar_split") if c in meta.columns]
    meta = meta[keep_meta] if keep_meta else meta.iloc[:, :0]

    out = (
        r_wide.join(s_wide, how="inner")
        .join(c_wide, how="inner")
        .join(rc_hat_wide, how="inner")
        .join(rc_oracle_wide, how="inner")
        .join(cost_wide, how="inner")
        .join(solvable, how="inner")
        .join(meta, how="left")
        .reset_index()
    )
    out["r_best"] = out[list(AGENT_COLS)].max(axis=1)
    out["oracle_best_agent"] = out[list(AGENT_COLS)].idxmax(axis=1).str.removeprefix("r_a_")
    out["s_hat_max"] = out[list(S_HAT_COLS)].max(axis=1)

    for agent in ROUTER_AGENTS:
        hat_norm = _cost_norm_for_agent(norms, agent, kind="hat")
        oracle_norm = _cost_norm_for_agent(norms, agent, kind="oracle")
        out[f"route_cost_hat_norm_{agent}"] = out[f"route_cost_hat_{agent}"] / hat_norm
        out[f"route_cost_oracle_norm_{agent}"] = out[f"route_cost_oracle_{agent}"] / oracle_norm

    if variant == "decomposed":
        if solvability_model is None or solvability_feature_cols is None:
            solvability_model, _manifest, solvability_cols = load_pool_solvability_detector()
        else:
            solvability_cols = solvability_feature_cols
        query = query_level_frame(frame)
        solvability_probability = predict_solvability_probability(
            solvability_model, query, solvability_cols
        )
        out = out.merge(
            pd.DataFrame(
                {
                    "training_id": query["training_id"].astype(str).values,
                    SOLVABILITY_PROBABILITY_COL: solvability_probability,
                }
            ),
            on="training_id",
            how="left",
        )
    return out


def route_agent_decision(
    row: pd.Series,
    *,
    lam: float = 0.0,
    policy: Gate3Policy = "success_threshold",
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    cost_budget: float | None = None,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
) -> Gate3Decision:
    """Pool-solvability abstain? → else Gate 3 policy (cost-efficient or utility)."""
    threshold = _resolve_abstention_threshold(
        abstention_threshold=abstention_threshold, theta_s=theta_s
    )
    mode = _normalize_abstention_mode(abstention)
    if mode == "solvability_probability":
        if should_abstain(_read_solvability_score(row), threshold):
            return Gate3Decision(None, True, policy, {"abstained_at": "gate1a"})
    elif float(row["s_hat_max"]) < threshold:
        return Gate3Decision(None, True, policy, {"abstained_at": "gate1a_max_s_hat"})
    return select_gate3_agent(
        row,
        policy=policy,
        lam=lam,
        success_threshold=success_threshold,
        cost_budget=cost_budget,
    )


def route_agent(
    row: pd.Series,
    *,
    lam: float = 0.0,
    policy: Gate3Policy = "success_threshold",
    success_threshold: float = DEFAULT_SUCCESS_THRESHOLD,
    cost_budget: float | None = None,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
) -> tuple[str | None, bool]:
    """Pool-solvability abstain? → else Gate 3 (default: utility U = ŝ − λĉ)."""
    decision = route_agent_decision(
        row,
        lam=lam,
        policy=policy,
        success_threshold=success_threshold,
        cost_budget=cost_budget,
        abstention_threshold=abstention_threshold,
        theta_s=theta_s,
        abstention=abstention,
    )
    return decision.agent, decision.abstained


def count_route_changes(
    table: pd.DataFrame,
    lam: float,
    *,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "solvability_probability",
    baseline_lam: float = 0.0,
) -> dict[str, float | int]:
    threshold = _resolve_abstention_threshold(
        abstention_threshold=abstention_threshold, theta_s=theta_s
    )
    abstention = _normalize_abstention_mode(abstention)
    """How many routed agents differ vs ``baseline_lam`` (Gate 2 activation diagnostic)."""
    if lam == baseline_lam:
        return {
            "baseline_lambda": baseline_lam,
            "lambda": lam,
            "n": len(table),
            "n_route_changes": 0,
            "pct_route_changes": 0.0,
        }
    changed = 0
    for _, row in table.iterrows():
        base, abst_base = route_agent(
            row,
            lam=baseline_lam,
            policy="utility",
            abstention_threshold=threshold,
            abstention=abstention,
        )
        cur, abst_cur = route_agent(
            row,
            lam=lam,
            policy="utility",
            abstention_threshold=threshold,
            abstention=abstention,
        )
        if abst_base != abst_cur or base != cur:
            changed += 1
    n = len(table)
    return {
        "baseline_lambda": baseline_lam,
        "lambda": lam,
        "n": n,
        "n_route_changes": changed,
        "pct_route_changes": float(changed / n) if n else 0.0,
    }


def _routed_cost_pair(
    row: pd.Series, routed_agent: str | None, abstained: bool
) -> tuple[float, float]:
    """(route_cost_hat, oracle_usd) for one routing decision; 0,0 when abstained."""
    if abstained or routed_agent is None:
        return 0.0, 0.0
    agent = str(routed_agent)
    hat = float(row[f"route_cost_hat_{agent}"])
    usd = float(row[f"cost_{agent}"])
    return hat, usd


def _routing_cost_summary(
    hat_costs: list[float], usd_costs: list[float]
) -> dict[str, float]:
    """Primary = Gate 2 estimate; secondary = post-hoc oracle USD calibration."""
    return {
        "mean_route_cost_hat": float(np.mean(hat_costs)),
        "mean_oracle_cost_usd": float(np.mean(usd_costs)),
    }


def route_cost_calibration_correlation(
    table: pd.DataFrame,
    routed_agents: list[str | None],
    abstained_flags: list[bool],
) -> dict[str, float | int | None]:
    """Pearson ρ between Gate 2 ĉ and oracle USD on routed (non-abstain) queries."""
    hats: list[float] = []
    usds: list[float] = []
    for (_, row), agent, abstained in zip(
        table.iterrows(), routed_agents, abstained_flags, strict=True
    ):
        hat, usd = _routed_cost_pair(row, agent, abstained)
        if abstained:
            continue
        hats.append(hat)
        usds.append(usd)
    n = len(hats)
    if n < 2:
        return {
            "n_routed": n,
            "pearson_r_hat_vs_oracle_usd": None,
            "spearman_r_hat_vs_oracle_usd": None,
        }
    hat_a = np.asarray(hats, dtype=float)
    usd_a = np.asarray(usds, dtype=float)
    if float(np.std(hat_a)) == 0.0 or float(np.std(usd_a)) == 0.0:
        return {
            "n_routed": n,
            "pearson_r_hat_vs_oracle_usd": None,
            "spearman_r_hat_vs_oracle_usd": None,
        }
    pearson = float(np.corrcoef(hat_a, usd_a)[0, 1])
    hat_rank = pd.Series(hat_a).rank().to_numpy()
    usd_rank = pd.Series(usd_a).rank().to_numpy()
    spearman = float(np.corrcoef(hat_rank, usd_rank)[0, 1])
    return {
        "n_routed": n,
        "pearson_r_hat_vs_oracle_usd": pearson,
        "spearman_r_hat_vs_oracle_usd": spearman,
    }


def agent_regret(row: pd.Series, routed_agent: str | None, abstained: bool) -> float:
    """Regret_agent = r_best(q) − r_{â}(q); abstain on unsolvable → 0."""
    r_best = float(row["r_best"])
    if abstained:
        return 0.0 if int(row["solvable"]) == 0 else r_best
    return r_best - float(row[f"r_a_{routed_agent}"])


def utility_regret(
    row: pd.Series,
    routed_agent: str | None,
    abstained: bool,
    *,
    lam: float,
) -> float:
    """Regret_U = U_best − U_â with U_a = r_a − λ·oracle_route_cost (normalized)."""
    utilities = {
        agent: float(row[f"r_a_{agent}"]) - lam * float(row[f"route_cost_oracle_norm_{agent}"])
        for agent in ROUTER_AGENTS
    }
    u_best = max(utilities.values())
    if abstained:
        return 0.0 if int(row["solvable"]) == 0 else max(0.0, u_best)
    u_routed = utilities[str(routed_agent)]
    return max(0.0, u_best - u_routed)


def evaluate_lambda(
    table: pd.DataFrame,
    lam: float,
    *,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
) -> dict[str, float]:
    threshold = _resolve_abstention_threshold(
        abstention_threshold=abstention_threshold, theta_s=theta_s
    )
    abstention = _normalize_abstention_mode(abstention)
    routed_agents: list[str | None] = []
    abstained_flags: list[bool] = []
    agent_regrets: list[float] = []
    utility_regrets: list[float] = []
    success: list[float] = []
    hat_costs: list[float] = []
    usd_costs: list[float] = []

    for _, row in table.iterrows():
        agent, abstained = route_agent(
            row,
            lam=lam,
            policy="utility",
            abstention_threshold=threshold,
            abstention=abstention,
        )
        routed_agents.append(agent)
        abstained_flags.append(abstained)
        agent_regrets.append(agent_regret(row, agent, abstained))
        utility_regrets.append(utility_regret(row, agent, abstained, lam=lam))
        hat, usd = _routed_cost_pair(row, agent, abstained)
        hat_costs.append(hat)
        usd_costs.append(usd)
        if abstained:
            success.append(0.0)
        else:
            success.append(float(row[f"r_a_{agent}"]))

    out = {
        "lambda": lam,
        "abstention_mode": abstention,
        "n": len(table),
        "mean_agent_regret": float(np.mean(agent_regrets)),
        "mean_utility_regret": float(np.mean(utility_regrets)),
        "success_rate": float(np.mean(success)),
        "abstention_rate": float(np.mean(abstained_flags)),
        **_routing_cost_summary(hat_costs, usd_costs),
    }
    out.update(
        route_cost_calibration_correlation(table, routed_agents, abstained_flags)
    )
    return out


def evaluate_success_threshold(
    table: pd.DataFrame,
    success_threshold: float,
    *,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
) -> dict[str, float]:
    """Gate 3 Option 1: cheapest agent among {a : ŝ_a ≥ τ_success}."""
    threshold = _resolve_abstention_threshold(
        abstention_threshold=abstention_threshold, theta_s=theta_s
    )
    abstention = _normalize_abstention_mode(abstention)
    routed_agents: list[str | None] = []
    abstained_flags: list[bool] = []
    agent_regrets: list[float] = []
    success: list[float] = []
    hat_costs: list[float] = []
    usd_costs: list[float] = []

    for _, row in table.iterrows():
        agent, abstained = route_agent(
            row,
            policy="success_threshold",
            success_threshold=success_threshold,
            abstention_threshold=threshold,
            abstention=abstention,
        )
        routed_agents.append(agent)
        abstained_flags.append(abstained)
        agent_regrets.append(agent_regret(row, agent, abstained))
        hat, usd = _routed_cost_pair(row, agent, abstained)
        hat_costs.append(hat)
        usd_costs.append(usd)
        if abstained:
            success.append(0.0)
        else:
            success.append(float(row[f"r_a_{agent}"]))

    out = {
        "tau_success": float(success_threshold),
        "policy": "success_threshold",
        "abstention_mode": abstention,
        "n": len(table),
        "mean_agent_regret": float(np.mean(agent_regrets)),
        "success_rate": float(np.mean(success)),
        "abstention_rate": float(np.mean(abstained_flags)),
        **_routing_cost_summary(hat_costs, usd_costs),
    }
    out.update(
        route_cost_calibration_correlation(table, routed_agents, abstained_flags)
    )
    return out


def sweep_success_thresholds(
    table: pd.DataFrame,
    *,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
    thresholds: tuple[float, ...] = SUCCESS_THRESHOLD_GRID,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            evaluate_success_threshold(
                table,
                tau,
                abstention_threshold=abstention_threshold,
                theta_s=theta_s,
                abstention=abstention,
            )
            for tau in thresholds
        ]
    )


def evaluate_budget(
    table: pd.DataFrame,
    cost_budget: float,
    *,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
) -> dict[str, float]:
    """Option 3: argmax ŝ among agents with route_cost_hat ≤ budget."""
    threshold = _resolve_abstention_threshold(
        abstention_threshold=abstention_threshold, theta_s=theta_s
    )
    abstention = _normalize_abstention_mode(abstention)
    routed_agents: list[str | None] = []
    abstained_flags: list[bool] = []
    agent_regrets: list[float] = []
    success: list[float] = []
    hat_costs: list[float] = []
    usd_costs: list[float] = []

    for _, row in table.iterrows():
        agent, abstained = route_agent(
            row,
            policy="budget",
            cost_budget=cost_budget,
            abstention_threshold=threshold,
            abstention=abstention,
        )
        routed_agents.append(agent)
        abstained_flags.append(abstained)
        agent_regrets.append(agent_regret(row, agent, abstained))
        hat, usd = _routed_cost_pair(row, agent, abstained)
        hat_costs.append(hat)
        usd_costs.append(usd)
        if abstained:
            success.append(0.0)
        else:
            success.append(float(row[f"r_a_{agent}"]))

    out = {
        "cost_budget": float(cost_budget),
        "policy": "budget",
        "abstention_mode": abstention,
        "n": len(table),
        "mean_agent_regret": float(np.mean(agent_regrets)),
        "success_rate": float(np.mean(success)),
        "abstention_rate": float(np.mean(abstained_flags)),
        **_routing_cost_summary(hat_costs, usd_costs),
    }
    out.update(
        route_cost_calibration_correlation(table, routed_agents, abstained_flags)
    )
    return out


def sweep_budgets(
    table: pd.DataFrame,
    *,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
) -> pd.DataFrame:
    """Budget grid from per-query oracle cost quantiles on the split."""
    oracle_costs = []
    for _, row in table.iterrows():
        for agent in ROUTER_AGENTS:
            oracle_costs.append(float(row[f"cost_{agent}"]))
    qs = (0.25, 0.5, 0.75, 0.9, 0.95, 1.0)
    budgets = sorted({float(np.quantile(oracle_costs, q)) for q in qs})
    return pd.DataFrame(
        [
            evaluate_budget(
                table,
                budget,
                abstention_threshold=abstention_threshold,
                theta_s=theta_s,
                abstention=abstention,
            )
            for budget in budgets
        ]
    )


def sweep_joint_regret_threshold(
    table: pd.DataFrame,
    *,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    threshold_grid: tuple[float, ...] = ABSTENTION_THRESHOLD_GRID,
) -> pd.DataFrame:
    """
    Val-only threshold sweep using full decomposed routing regret (archived path).

    Requires ``solvability_probability`` on ``table``.
    """
    if SOLVABILITY_PROBABILITY_COL not in table.columns:
        raise ValueError(
            f"joint threshold sweep requires {SOLVABILITY_PROBABILITY_COL!r} on query table"
        )

    abstain_metrics = sweep_abstention_threshold(
        table[SOLVABILITY_PROBABILITY_COL],
        table["solvable"],
        threshold_grid=threshold_grid,
    ).set_index("abstention_threshold")

    records: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        best: dict[str, Any] | None = None
        for lam in lambdas:
            m = evaluate_lambda(
                table,
                lam,
                abstention_threshold=float(threshold),
                abstention="solvability_probability",
            )
            if best is None or m["mean_agent_regret"] < best["mean_agent_regret"]:
                best = {
                    **m,
                    "abstention_threshold": float(threshold),
                    "lambda_at_threshold": float(lam),
                }
        if best is None:
            continue
        if float(threshold) in abstain_metrics.index:
            row = abstain_metrics.loc[float(threshold)]
            best.update(
                {
                    "f1": float(row["f1"]),
                    "false_abstention_rate_solvable": float(
                        row["false_abstention_rate_solvable"]
                    ),
                    "correct_abstention_rate_unsolvable": float(
                        row["correct_abstention_rate_unsolvable"]
                    ),
                }
            )
        records.append(best)
    return pd.DataFrame(records)


def select_joint_regret_threshold(sweep: pd.DataFrame) -> tuple[float, float]:
    """Return (best_threshold, lambda_at_threshold) from a joint sweep."""
    if sweep.empty:
        raise ValueError("empty joint threshold sweep")
    idx = sweep["mean_agent_regret"].astype(float).idxmin()
    row = sweep.loc[idx]
    threshold_col = (
        "abstention_threshold"
        if "abstention_threshold" in row.index
        else "theta_s"
    )
    lambda_col = (
        "lambda_at_threshold" if "lambda_at_threshold" in row.index else "lambda_at_theta"
    )
    return float(row[threshold_col]), float(row[lambda_col])


sweep_theta_joint_regret = sweep_joint_regret_threshold
pick_theta_joint_regret = select_joint_regret_threshold


def sweep_lambdas(
    table: pd.DataFrame,
    *,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    abstention: AbstentionMode = "max_s_hat",
) -> pd.DataFrame:
    threshold = _resolve_abstention_threshold(
        abstention_threshold=abstention_threshold, theta_s=theta_s
    )
    abstention = _normalize_abstention_mode(abstention)
    return pd.DataFrame(
        [
            evaluate_lambda(table, lam, abstention_threshold=threshold, abstention=abstention)
            for lam in lambdas
        ]
    )


def pick_best_lambda(sweep: pd.DataFrame) -> float:
    best_idx = sweep["mean_agent_regret"].astype(float).idxmin()
    return float(sweep.loc[best_idx, "lambda"])


def build_predictions(
    table: pd.DataFrame,
    *,
    lam: float,
    abstention_threshold: float | None = None,
    theta_s: float | None = None,
    abstention: AbstentionMode = "max_s_hat",
) -> pd.DataFrame:
    threshold = _resolve_abstention_threshold(
        abstention_threshold=abstention_threshold, theta_s=theta_s
    )
    abstention = _normalize_abstention_mode(abstention)
    rows: list[dict[str, Any]] = []
    for _, row in table.iterrows():
        routed, abstained = route_agent(
            row,
            lam=lam,
            policy="utility",
            abstention_threshold=threshold,
            abstention=abstention,
        )
        rec: dict[str, Any] = {
            "training_id": row["training_id"],
            "lambda": lam,
            "abstention_threshold": threshold,
            "abstention_mode": abstention,
            "solvable": int(row["solvable"]),
            "r_best": float(row["r_best"]),
            "oracle_best_agent": row["oracle_best_agent"],
            "routed_agent": routed,
            "abstained": abstained,
            "agent_regret": agent_regret(row, routed, abstained),
            "utility_regret": utility_regret(row, routed, abstained, lam=lam),
        }
        if SOLVABILITY_PROBABILITY_COL in row.index and pd.notna(
            row[SOLVABILITY_PROBABILITY_COL]
        ):
            rec[SOLVABILITY_PROBABILITY_COL] = float(row[SOLVABILITY_PROBABILITY_COL])
        for agent in ROUTER_AGENTS:
            rec[f"s_hat_{agent}"] = float(row[f"s_hat_{agent}"])
            rec[f"c_hat_{agent}"] = float(row[f"c_hat_{agent}"])
            rec[f"route_cost_hat_{agent}"] = float(row[f"route_cost_hat_{agent}"])
            rec[f"r_a_{agent}"] = float(row[f"r_a_{agent}"])
            rec[f"cost_{agent}"] = float(row[f"cost_{agent}"])
        if not abstained and routed is not None:
            rec["routed_success"] = float(row[f"r_a_{routed}"])
            rec["routed_cost_usd"] = float(row[f"cost_{routed}"])
        else:
            rec["routed_success"] = 0.0
            rec["routed_cost_usd"] = 0.0
        rows.append(rec)
    return pd.DataFrame(rows)


def evaluate_baseline(
    table: pd.DataFrame,
    split: str,
    *,
    lam: float = 0.0,
    router_path: Path | None = None,
) -> dict[str, float]:
    """Monolithic hgbm_cvec5_emb_soft_kl — argmax proba, no abstention."""
    router_path = router_path or PRIMARY_ROUTER_PATH
    obj = load_router(router_path)
    feature_cols: list[str] = list(obj["feature_cols"])

    frame = load_frame(split)
    feat = frame.drop_duplicates("training_id").set_index("training_id")
    feat = feat.loc[table["training_id"].astype(str)]
    feat = feat.reset_index()
    scored = attach_router_predictions(
        feat,
        obj["pipeline"],
        feature_cols,
        experiment_id=str(obj.get("experiment_id", router_path.stem)),
    )

    merged = table.merge(
        scored[["training_id", "router_pred"]],
        on="training_id",
        how="left",
    )
    if merged["router_pred"].isna().any():
        raise ValueError("baseline router missing predictions for some queries")

    agent_regrets: list[float] = []
    utility_regrets: list[float] = []
    success: list[float] = []
    costs: list[float] = []

    for _, row in merged.iterrows():
        pred = str(row["router_pred"])
        agent_regrets.append(float(row["r_best"]) - float(row[f"r_a_{pred}"]))
        utilities = {
            a: float(row[f"r_a_{a}"]) - lam * float(row[f"route_cost_oracle_norm_{a}"])
            for a in ROUTER_AGENTS
        }
        u_best = max(utilities.values())
        u_pred = utilities[pred]
        utility_regrets.append(max(0.0, u_best - u_pred))
        success.append(float(row[f"r_a_{pred}"]))
        costs.append(float(row[f"cost_{pred}"]))

    return {
        "baseline": str(router_path.stem),
        "n": len(merged),
        "mean_agent_regret": float(np.mean(agent_regrets)),
        "mean_utility_regret": float(np.mean(utility_regrets)),
        "success_rate": float(np.mean(success)),
        "abstention_rate": 0.0,
        "mean_oracle_cost_usd": float(np.mean(costs)),
    }


def save_predictions(
    df: pd.DataFrame,
    split: str,
    *,
    variant: Gate1Variant = THESIS_VARIANT,
) -> Path:
    ensure_daar_dirs()
    path = daar_predictions_path(variant, split)
    df.to_parquet(path, index=False)
    return path


def route_daar(
    *,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    models_dir: Path | None = None,
    abstention_threshold: float | None = None,
    frozen_lambda: float | None = None,
    eval_test: bool = False,
    cost_norm: CostNormMode = "per_agent",
    theta_from_joint_regret: bool = False,
) -> dict[str, Any]:
    """Run Version A routing (pool solvability → abstain? → rank ∥ cost → utility).

    Abstention threshold defaults to detection-tuned value from the manifest.
    Pass ``theta_from_joint_regret=True`` only to reproduce archived thesis freeze
    (``archive/daar_thesis_freeze/``).
    """
    variant = THESIS_VARIANT
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()
    rank_model, manifest, feature_cols = load_agent_rank_model(variant, models_dir)
    abstention: AbstentionMode = "solvability_probability"
    _solv_model, solvability_manifest, _ = load_pool_solvability_detector(models_dir)
    if theta_from_joint_regret and solvability_manifest.get("theta_s_joint_regret") is not None:
        abstention_threshold = float(
            abstention_threshold
            if abstention_threshold is not None
            else solvability_manifest["theta_s_joint_regret"]
        )
        abstention_rule = "archived thesis: solvability_probability < joint_regret_threshold"
    else:
        abstention_threshold = float(
            abstention_threshold
            if abstention_threshold is not None
            else read_abstention_threshold(solvability_manifest)
        )
        abstention_rule = (
            "detection: abstain iff solvability_probability < abstention_threshold"
        )
    norms = calibrate_route_cost_norms(per_agent=(cost_norm == "per_agent"))

    if frozen_lambda is not None:
        best_lambda = float(frozen_lambda)
    elif theta_from_joint_regret and solvability_manifest.get("lambda_at_theta_joint") is not None:
        best_lambda = float(solvability_manifest["lambda_at_theta_joint"])
    else:
        best_lambda = None

    val_table = build_query_table(
        "val",
        variant=variant,
        gate1=rank_model,
        feature_cols=feature_cols,
        norms=norms,
        cost_norm=cost_norm,
    )
    val_sweep = sweep_lambdas(
        val_table, abstention_threshold=abstention_threshold, lambdas=lambdas, abstention=abstention
    )
    if best_lambda is None:
        best_lambda = float(pick_best_lambda(val_sweep))

    val_metrics = evaluate_lambda(
        val_table, best_lambda, abstention_threshold=abstention_threshold, abstention=abstention
    )
    val_baseline = evaluate_baseline(val_table, "val", lam=best_lambda)
    val_preds = save_predictions(
        build_predictions(
            val_table, lam=best_lambda, abstention_threshold=abstention_threshold, abstention=abstention
        ),
        "val",
        variant=variant,
    )

    report: dict[str, Any] = {
        "gate": 3,
        "variant": variant,
        "gate1_supervision": manifest.get("gate1_supervision", "success_only_soft_kl"),
        "rank_head": "gate1b",
        "abstention_mode": abstention,
        "abstention_rule": abstention_rule,
        "abstention_threshold": abstention_threshold,
        "route_cost_norms": norms,
        "cost_norm_mode": cost_norm,
        "routing_score": "s_hat - lambda * route_cost_hat_norm",
        "lambda_grid": list(lambdas),
        "lambda_frozen": best_lambda,
        "lambda_selection": "min mean_agent_regret on val",
        "primary_metric": "mean_agent_regret",
        "val_lambda_sweep": val_sweep.to_dict(orient="records"),
        f"val_daar_{variant}": val_metrics,
        "val_baseline": val_baseline,
        "predictions": {"val": str(val_preds)},
    }
    report["pool_solvability"] = {
        "manifest": solvability_manifest.get(
            "manifest_path", str(models_dir / POOL_SOLVABILITY_MANIFEST)
        ),
        "abstention_threshold_val_tuned": read_abstention_threshold(solvability_manifest),
        "theta_s_joint_regret": solvability_manifest.get("theta_s_joint_regret"),
        "n_train_queries": solvability_manifest.get("n_train_queries"),
    }
    report["agent_ranking"] = {
        "manifest": manifest.get("manifest_path", str(models_dir / AGENT_RANK_MANIFEST)),
        "gate1_supervision": manifest.get("gate1_supervision"),
        "feature_set": manifest.get("feature_set"),
        "n_train_solvable": manifest.get("n_train_solvable"),
    }
    report["factorization"] = (
        "P(route|q) ≈ P(solvable|q) × 𝟙[U(q,a;λ) optimal]; "
        "U_a = ŝ_a − λ·ĉ_a with ĉ from D2 G→T(q,a) and D3 T→ĉ"
    )
    report["online_flow"] = (
        "QCE → D1 G(q) → Gate 1a → abstain? → [Gate 1b ∥ Gate 2 (D2→D3)] → Gate 3"
    )
    report["decomposition_stages"] = {
        "D1": "QCE → G(q) structure",
        "D2": "G(q) → T(q,a) trajectory",
        "D3": "T(q,a) → ĉ(q,a) cost",
        "gate2": "D2 + D3",
    }
    report["lambda_interpretation"] = (
        "At lambda=0, U=ŝ (rank-only). Gate 2 architecturally integrated; "
        "cost term unused at val regret-optimal operating point."
    )
    report["thesis_freeze"] = "archive/daar_thesis_freeze/"

    if eval_test:
        test_table = build_query_table(
            "test",
            variant=variant,
            gate1=rank_model,
            feature_cols=feature_cols,
            norms=norms,
            cost_norm=cost_norm,
        )
        test_metrics = evaluate_lambda(
            test_table, best_lambda, abstention_threshold=abstention_threshold, abstention=abstention
        )
        test_baseline = evaluate_baseline(test_table, "test", lam=best_lambda)
        test_preds = save_predictions(
            build_predictions(
                test_table, lam=best_lambda, abstention_threshold=abstention_threshold, abstention=abstention
            ),
            "test",
            variant=variant,
        )
        report["test_evaluated_once"] = True
        report[f"test_daar_{variant}"] = test_metrics
        report["test_baseline"] = test_baseline
        report["gate3_pass"] = test_metrics["mean_agent_regret"] <= test_baseline["mean_agent_regret"]
        report["predictions"]["test"] = str(test_preds)

    manifest_path = daar_routing_dir() / f"{variant}_routing_manifest.json"
    manifest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    val_sweep.to_csv(daar_routing_dir() / f"lambda_sweep_val_{variant}.csv", index=False)
    report["manifest_path"] = str(manifest_path)
    return report
