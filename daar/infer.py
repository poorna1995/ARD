"""D-AAR production routing — one query, Version A online order.

Canonical entry::

    from daar.infer import route_query

    result = route_query(training_id="...", eval_tag="gaia")
    agent = result["agent"]          # str | None when abstained
    abstained = result["abstained"]

CLI: ``scripts/infer_daar.py``
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import pandas as pd

from config.global_config.paths import (
    daar_cost_hand_path,
    daar_embeddings_path,
    daar_features_path,
    daar_frames_dir,
    daar_models_dir,
    daar_plans_cache_path,
    decomposer_cache_path,
)
from config.local.constants import DIMS
from config.local.constants.agents import ROUTER_AGENTS
from config.local.constants.datasets import resolve_dataset_name
from config.local.router.paths import PCA_PATH, TRAIN_NORM_JSON
from daar.cost_hand import simulate_cost_hand
from daar.eval_cost import build_eval_cost_hand, load_eval_cost_hand, load_frozen_cost_scales
from daar.pool_solvability import (
    SOLVABILITY_PROBABILITY_COL,
    predict_solvability_probability,
    read_abstention_threshold,
    should_abstain,
)
from daar.gate3_policy import Gate3Policy, read_frozen_gate3_config
from daar.routing import (
    THESIS_VARIANT,
    calibrate_route_cost_norms,
    load_agent_rank_model,
    load_pool_solvability_detector,
    route_agent_decision,
)
from daar.trace_skeleton import build_trace_row
from eval.benchmark import load_routable_eval_frame
from qce.complexity import complexity_dataframe, load_train_norm
from qce.decompose import decompose_batch, load_plans_jsonl
from qce.features.build import ensure_query_heuristics
from router.router import predict_agent_proba
from scripts.build_query_embeddings import DEFAULT_MODEL, encode_queries, matrix_to_frame

PoolSource = Literal["daar", "eval", "ad_hoc"]

PRODUCTION_FLOW = (
    "QCE → D1 → Gate 1a abstain? → Gate 1b ŝ ∥ Gate 2 ĉ → "
    "Gate 3: argmin ĉ among {ŝ ≥ τ} → execute cheapest capable agent"
)


@dataclass(frozen=True)
class RouteQueryResult:
    """Production routing outcome for one query."""

    training_id: str
    agent: str | None
    abstained: bool
    solvability_probability: float
    abstention_threshold: float
    lambda_: float
    source: PoolSource
    details: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        out = dict(self.details)
        out.update(
            {
                "training_id": self.training_id,
                "agent": self.agent,
                "routed_agent": self.agent,
                "abstained": self.abstained,
                SOLVABILITY_PROBABILITY_COL: self.solvability_probability,
                "abstention_threshold": self.abstention_threshold,
                "production_flow": PRODUCTION_FLOW,
            }
        )
        return out


def _load_embeddings_index() -> pd.DataFrame:
    path = daar_embeddings_path()
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path} — run: uv run python scripts/build_daar_embeddings.py"
        )
    emb = pd.read_parquet(path)
    emb["training_id"] = emb["training_id"].astype(str)
    return emb.set_index("training_id")


def find_in_eval_frame(training_id: str, eval_tag: str) -> pd.Series | None:
    frame = load_routable_eval_frame(eval_tag.strip().lower(), build_features=False)
    frame["training_id"] = frame["training_id"].astype(str)
    hit = frame[frame["training_id"] == str(training_id)].drop_duplicates("training_id")
    return None if hit.empty else hit.iloc[0]


def find_in_daar_frames(training_id: str) -> tuple[pd.Series, str] | None:
    tid = str(training_id)
    for split in ("val", "test", "train"):
        path = daar_frames_dir() / f"daar_{split}_frame.parquet"
        if not path.is_file():
            continue
        frame = pd.read_parquet(path)
        frame["training_id"] = frame["training_id"].astype(str)
        hit = frame[frame["training_id"] == tid].drop_duplicates("training_id")
        if not hit.empty:
            return hit.iloc[0], split
    return None


def find_in_features(training_id: str) -> pd.Series | None:
    path = daar_features_path()
    if not path.is_file():
        return None
    feat = pd.read_parquet(path)
    feat["training_id"] = feat["training_id"].astype(str)
    hit = feat[feat["training_id"] == str(training_id)]
    if hit.empty:
        return None
    row = hit.iloc[0].copy()
    emb = _load_embeddings_index()
    if str(training_id) in emb.index:
        for col in emb.columns:
            row[col] = emb.loc[str(training_id), col]
    return row


def load_plan(
    training_id: str,
    *,
    plans_path: Path | None = None,
    eval_tag: str | None = None,
) -> dict[str, Any] | None:
    tid = str(training_id)
    candidates: list[Path] = []
    if plans_path is not None:
        candidates.append(plans_path)
    if eval_tag:
        candidates.append(decomposer_cache_path(eval_tag))
    candidates.append(daar_plans_cache_path())
    for path in candidates:
        if not path.is_file():
            continue
        for rec in load_plans_jsonl(path):
            if str(rec.get("training_id")) == tid:
                return rec
    return None


def qce_build_features(
    *,
    training_id: str,
    query: str,
    dataset: str,
    plan: dict[str, Any],
) -> pd.DataFrame:
    """D1: QCE plan → φ(q), emb(q) feature row."""
    if not TRAIN_NORM_JSON.is_file():
        raise FileNotFoundError(f"missing {TRAIN_NORM_JSON}")
    norm = load_train_norm(TRAIN_NORM_JSON)
    c_row = complexity_dataframe([plan], norm=norm, fit_norm=False).iloc[0]
    base = pd.DataFrame(
        [
            {
                "training_id": str(training_id),
                "query": query,
                "dataset": resolve_dataset_name(dataset),
            }
        ]
    )
    for col in (c for c in DIMS.main if c in c_row.index):
        base[col] = float(c_row[col])
    base = ensure_query_heuristics(base, tag=None, verbose=False)
    if not PCA_PATH.is_file():
        raise FileNotFoundError(f"missing {PCA_PATH}")
    pca = joblib.load(PCA_PATH)
    n_comp = int(getattr(pca, "n_components_", 16))
    raw = encode_queries(
        [query], model_name=DEFAULT_MODEL, batch_size=1, normalize_embeddings=True
    )
    emb = matrix_to_frame(base["training_id"], pca.transform(raw), n_components=n_comp)
    return base.merge(emb, on="training_id", how="left")


def gate2_from_plan(
    plan: dict[str, Any],
    *,
    training_id: str,
    c_static: float = 0.0,
) -> dict[str, dict[str, float]]:
    """Gate 2: D2 G→T(q,a), D3 T→ĉ(q,a) using frozen train scales."""
    rows = [build_trace_row(plan, agent) for agent in ROUTER_AGENTS]
    traces = pd.DataFrame(rows)
    traces["training_id"] = str(training_id)
    scales = load_frozen_cost_scales()
    c_static_s = pd.Series({str(training_id): float(c_static)})
    sim = simulate_cost_hand(traces, c_static_s, scales)
    sim["route_cost_hat"] = sim["X1_hat"] + sim["X2_dynamic_hat"]
    out: dict[str, dict[str, float]] = {}
    for agent in ROUTER_AGENTS:
        row = sim.loc[sim["agent"] == agent].iloc[0]
        out[agent] = {
            "X1_hat": float(row["X1_hat"]),
            "X2_dynamic_hat": float(row["X2_dynamic_hat"]),
            "route_cost_hat": float(row["route_cost_hat"]),
            "c_hat": float(row["c_hat"]),
        }
    return out


def gate2_from_eval_cache(training_id: str, eval_tag: str) -> dict[str, dict[str, float]] | None:
    try:
        cost = load_eval_cost_hand([eval_tag.strip().lower()])
    except FileNotFoundError:
        build_eval_cost_hand(eval_tag.strip().lower())
        cost = load_eval_cost_hand([eval_tag.strip().lower()])
    return _gate2_wide_from_cost_df(cost, training_id)


def gate2_from_daar_cache(training_id: str) -> dict[str, dict[str, float]] | None:
    path = daar_cost_hand_path()
    if not path.is_file():
        return None
    return _gate2_wide_from_cost_df(pd.read_parquet(path), training_id)


def _gate2_wide_from_cost_df(
    cost: pd.DataFrame, training_id: str
) -> dict[str, dict[str, float]] | None:
    cost = cost.copy()
    cost["training_id"] = cost["training_id"].astype(str)
    sub = cost[cost["training_id"] == str(training_id)]
    if sub.empty:
        return None
    if "route_cost_hat" not in sub.columns:
        sub["route_cost_hat"] = sub["X1_hat"] + sub["X2_dynamic_hat"]
    out: dict[str, dict[str, float]] = {}
    for agent in ROUTER_AGENTS:
        row = sub.loc[sub["agent"] == agent].iloc[0]
        out[agent] = {
            "X1_hat": float(row["X1_hat"]),
            "X2_dynamic_hat": float(row["X2_dynamic_hat"]),
            "route_cost_hat": float(row["route_cost_hat"]),
            "c_hat": float(row["c_hat"]),
        }
    return out


def _attach_gate2(route_row: pd.Series, gate2: dict[str, dict[str, float]], norms: dict[str, float]) -> pd.Series:
    out = route_row.copy()
    for agent in ROUTER_AGENTS:
        g2 = gate2[agent]
        out[f"route_cost_hat_{agent}"] = g2["route_cost_hat"]
        key = f"route_cost_hat_median_train_{agent}"
        denom = float(norms.get(key, norms["route_cost_hat_median_train"]))
        out[f"route_cost_hat_norm_{agent}"] = g2["route_cost_hat"] / denom
    return out


def route_query(
    *,
    training_id: str | None = None,
    query: str | None = None,
    dataset: str | None = None,
    eval_tag: str | None = None,
    decompose: bool = False,
    models_dir: Path | None = None,
    abstention_threshold: float | None = None,
    lam: float | None = None,
    routing_policy: Gate3Policy | None = None,
    success_threshold: float | None = None,
    cost_budget: float | None = None,
    # Deprecated keyword for callers that still pass theta_s.
    theta_s: float | None = None,
) -> RouteQueryResult:
    """
    Canonical production D-AAR pipeline for one query.

    Flow::

        QCE → D1 → pool solvability → abstain? → [agent rank ∥ cost] → Gate 3

    **Cost-efficient routing** (default): cheapest agent with ŝ ≥ τ_success.
    Not "choose the best agent" (argmax ŝ).

    Alternate Gate 3 policies: ``utility`` (soft λ), ``budget`` (cost cap).
    Knobs load from ``gate3_cost_objectives_val.json`` when present.

    Provide ``training_id`` (daar pool or eval_samples with ``eval_tag``), or
    ``query`` + ``dataset`` for ad-hoc routing (``decompose=True`` or cached plan).
    """
    if abstention_threshold is None and theta_s is not None:
        abstention_threshold = theta_s
    models_dir = models_dir or daar_models_dir()
    rank_model, rank_manifest, rank_cols = load_agent_rank_model(THESIS_VARIANT, models_dir)
    solvability_model, solvability_manifest, solvability_cols = load_pool_solvability_detector(
        models_dir
    )
    threshold = float(
        abstention_threshold
        if abstention_threshold is not None
        else read_abstention_threshold(solvability_manifest)
    )
    frozen_gate3 = read_frozen_gate3_config()
    policy: Gate3Policy = routing_policy or frozen_gate3["routing_policy"]  # type: ignore[assignment]
    lambda_val = float(lam if lam is not None else frozen_gate3["lambda"])
    tau_success = float(
        success_threshold
        if success_threshold is not None
        else frozen_gate3["success_threshold"]
    )
    budget_val = (
        float(cost_budget)
        if cost_budget is not None
        else (
            float(frozen_gate3["cost_budget"])
            if frozen_gate3.get("cost_budget") is not None
            else None
        )
    )

    source: PoolSource = "ad_hoc"
    plan: dict[str, Any] | None = None
    feature_row: pd.Series | None = None
    split: str | None = None
    tid: str

    if training_id:
        tid = str(training_id)
        hit = find_in_daar_frames(tid)
        if hit is not None:
            feature_row, split = hit
            source = "daar"
        else:
            feat = find_in_features(tid)
            if feat is not None:
                feature_row, source = feat, "daar"
            elif eval_tag:
                source = "eval"
                eval_row = find_in_eval_frame(tid, eval_tag)
                if eval_row is not None:
                    feature_row = eval_row
                    query = query or str(eval_row.get("query", ""))
                    dataset = dataset or str(eval_row.get("dataset", eval_tag))
            if feature_row is not None:
                query = query or str(feature_row.get("query", ""))
                dataset = dataset or str(feature_row.get("dataset", eval_tag or ""))
        plan = load_plan(tid, eval_tag=eval_tag)
    else:
        if not query or not dataset:
            raise ValueError("provide training_id or (query + dataset)")
        tid = f"ad_hoc_{abs(hash(query)) & 0xFFFFFFFF:08x}"

    # ── QCE / D1: resolve G(q), φ(q), emb(q) ──
    if feature_row is None:
        if not query or not dataset:
            raise ValueError(f"no features for {tid!r}; pass --eval-tag or --decompose")
        if plan is None and decompose:
            cache = (
                daar_plans_cache_path()
                if source == "daar"
                else decomposer_cache_path(eval_tag or dataset)
            )
            decompose_batch(
                [{"training_id": tid, "query": query, "dataset": dataset}],
                cache_path=cache,
                force_refresh=False,
            )
            plan = load_plan(tid, plans_path=cache, eval_tag=eval_tag)
        if plan is None:
            raise ValueError(f"no QCE plan for {tid!r} — pass decompose=True or cache plan")
        feature_row = qce_build_features(
            training_id=tid, query=query, dataset=dataset, plan=plan
        ).iloc[0]
    elif plan is None:
        plan = load_plan(tid, eval_tag=eval_tag)

    qdf = pd.DataFrame([feature_row])

    # ── Pool solvability detection ──
    solvability_probability = float(
        predict_solvability_probability(solvability_model, qdf, solvability_cols)[0]
    )
    details: dict[str, Any] = {
        "source": source,
        "daar_split": split,
        "eval_tag": eval_tag,
        "frozen_config": {
            "abstention_threshold": threshold,
            "routing_policy": policy,
            "success_threshold": tau_success,
            "cost_budget": budget_val,
            "lambda": lambda_val,
            "gate3_objective": frozen_gate3.get(
                "deployment_objective",
                "cheapest agent sufficiently likely to succeed",
            ),
            "solvability_features": solvability_manifest.get("feature_set", "cvec5_emb"),
            "agent_rank_features": rank_manifest.get("feature_set", "cvec5"),
        },
        "D1_qce": {"plan_loaded": plan is not None},
        "pool_solvability": {
            SOLVABILITY_PROBABILITY_COL: solvability_probability,
            "abstention_threshold": threshold,
            "abstain": should_abstain(solvability_probability, threshold),
            "rule": (
                "abstain iff solvability_probability < abstention_threshold"
            ),
        },
    }

    if should_abstain(solvability_probability, threshold):
        details["agent_ranking"] = {"skipped": True, "reason": "abstained at gate1a"}
        details["cost_model"] = {"skipped": True, "reason": "abstained at gate1a"}
        details["gate3_routing"] = {
            "skipped": True,
            "abstained_at": "gate1a",
            "routing_rule": "solvability_probability < abstention_threshold",
        }
        return RouteQueryResult(
            training_id=tid,
            agent=None,
            abstained=True,
            solvability_probability=solvability_probability,
            abstention_threshold=threshold,
            lambda_=lambda_val,
            source=source,
            details=details,
        )

    # ── Agent ranking ∥ cost model (only after abstention check) ──
    proba = predict_agent_proba(rank_model, qdf, rank_cols)
    s_hat = {a: float(proba[f"p_{a}"].iloc[0]) for a in ROUTER_AGENTS}

    gate2 = gate2_from_daar_cache(tid)
    gate2_source = "daar_cost_hand.parquet"
    if gate2 is None and eval_tag:
        gate2 = gate2_from_eval_cache(tid, eval_tag)
        gate2_source = f"eval_{eval_tag.strip().lower()}_cost_hand.parquet"
    if gate2 is None and plan is not None:
        c_static = 0.0
        if source == "daar" and "C_static" in feature_row.index:
            c_static = float(feature_row.get("C_static") or 0.0)
        gate2 = gate2_from_plan(plan, training_id=tid, c_static=c_static)
        gate2_source = "simulated D2→D3 (frozen scales)"
    if gate2 is None:
        raise ValueError(
            f"Gate 2 costs missing for {tid!r} — run simulate_daar_cost_hand or build_eval_cost_hand"
        )

    # ── Gate 3: utility routing ──
    norms = calibrate_route_cost_norms(per_agent=True)
    route_row = qdf.iloc[0].copy()
    route_row[SOLVABILITY_PROBABILITY_COL] = solvability_probability
    for agent in ROUTER_AGENTS:
        route_row[f"s_hat_{agent}"] = s_hat[agent]
    route_row = _attach_gate2(route_row, gate2, norms)

    decision = route_agent_decision(
        route_row,
        lam=lambda_val,
        policy=policy,
        success_threshold=tau_success,
        cost_budget=budget_val,
        abstention_threshold=threshold,
        abstention="solvability_probability",
    )
    agent, abstained = decision.agent, decision.abstained

    details["agent_ranking"] = {"s_hat": s_hat}
    details["cost_model"] = {"source": gate2_source, "D2_D3_per_agent": gate2}
    details["gate3_routing"] = {
        "policy": policy,
        **decision.meta,
    }
    if policy == "utility":
        details["gate3_routing"]["utilities"] = {
            a: {
                "s_hat": s_hat[a],
                "c_hat_norm": float(route_row[f"route_cost_hat_norm_{a}"]),
                "U": s_hat[a] - lambda_val * float(route_row[f"route_cost_hat_norm_{a}"]),
            }
            for a in ROUTER_AGENTS
        }

    return RouteQueryResult(
        training_id=tid,
        agent=agent,
        abstained=abstained,
        solvability_probability=solvability_probability,
        abstention_threshold=threshold,
        lambda_=lambda_val,
        source=source,
        details=details,
    )


def infer_daar(**kwargs: Any) -> dict[str, Any]:
    """Backward-compatible alias → :func:`route_query` (returns dict)."""
    return route_query(**kwargs).to_dict()


# Legacy aliases used by tests / older imports
build_features_from_plan = qce_build_features
gate2_wide_from_plan = gate2_from_plan
gate2_wide_from_eval_cache = gate2_from_eval_cache
gate2_wide_from_daar_cache = gate2_from_daar_cache
attach_gate2_to_route_row = _attach_gate2
