"""
Selective QCE routing (Phase 2, opt-in).

Cheap ``emb_only`` router gates full ``cvec5_emb`` + decompose. See ``docs/selective_qce_phase2.md``.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from routing.analysis import (
    DEFAULT_ORACLE,
    attach_outcomes,
    metrics_for_routed,
    oracle_outcome_matrix,
)
from routing.config import (
    DEFAULT_SELECTIVE_TAU,
    PCA_PATH,
    SELECTIVE_CHEAP_ROUTER_PATH,
    SELECTIVE_DECOMPOSE_COST_USD,
    SELECTIVE_FULL_ROUTER_PATH,
    SELECTIVE_TAU_GRID,
    TRAIN_NORM_JSON,
)
from routing.router import (
    PROBA_COLS,
    attach_router_predictions,
    feature_paths,
    load_router,
    load_split_for_feature_set,
    merge_feature_tables,
    overall_complexity,
    resolve_dataset_name,
)

SelectivePath = Literal["cheap", "full"]


@dataclass(frozen=True)
class SelectiveGateConfig:
    """Gate + router paths for selective QCE."""

    tau: float = DEFAULT_SELECTIVE_TAU
    tau_margin: float | None = None
    cheap_router_path: Path = SELECTIVE_CHEAP_ROUTER_PATH
    full_router_path: Path = SELECTIVE_FULL_ROUTER_PATH
    decompose_cost_usd: float = SELECTIVE_DECOMPOSE_COST_USD

    def __post_init__(self) -> None:
        if not 0.0 < self.tau <= 1.0:
            raise ValueError(f"tau must be in (0, 1], got {self.tau}")
        if self.tau_margin is not None and not 0.0 <= self.tau_margin <= 1.0:
            raise ValueError(f"tau_margin must be in [0, 1], got {self.tau_margin}")


def load_selective_routers(
    config: SelectiveGateConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    cheap = load_router(config.cheap_router_path)
    full = load_router(config.full_router_path)
    return cheap, full


def confident_mask(df: pd.DataFrame, config: SelectiveGateConfig) -> pd.Series:
    if "max_prob_emb" not in df.columns:
        raise KeyError("missing max_prob_emb — run attach_cheap_predictions first")
    sure = df["max_prob_emb"].astype(float) >= config.tau
    if config.tau_margin is not None:
        if "margin_top2_emb" not in df.columns:
            raise KeyError("missing margin_top2_emb — required when tau_margin is set")
        sure = sure & (df["margin_top2_emb"].astype(float) >= config.tau_margin)
    return sure


def attach_cheap_predictions(
    df: pd.DataFrame,
    cheap_obj: dict[str, Any],
    *,
    experiment_id: str = "emb_only_gate",
) -> pd.DataFrame:
    pipe = cheap_obj["pipeline"]
    cols: list[str] = list(cheap_obj["feature_cols"])
    out = attach_router_predictions(df, pipe, cols, experiment_id=experiment_id)
    out = out.rename(
        columns={
            "router_pred": "router_pred_cheap",
            "max_prob": "max_prob_emb",
            "second_prob": "second_prob_emb",
            "margin_top2": "margin_top2_emb",
            "router_second": "router_second_cheap",
        }
    )
    for c in PROBA_COLS:
        if c in out.columns:
            out[f"{c}_cheap"] = out[c]
    return out


def attach_full_predictions(
    df: pd.DataFrame,
    full_obj: dict[str, Any],
    *,
    experiment_id: str = "cvec5_emb_full",
    uncertain_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Score full router; if ``uncertain_mask`` given, predict only those rows (no ``dim_*`` elsewhere)."""
    pipe = full_obj["pipeline"]
    cols: list[str] = list(full_obj["feature_cols"])
    out = df.copy()
    mask = (
        uncertain_mask
        if uncertain_mask is not None
        else pd.Series(True, index=out.index)
    )
    for c in PROBA_COLS:
        out[f"{c}_full"] = np.nan
    out["router_pred_full"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    out["router_second_full"] = pd.Series([None] * len(out), index=out.index, dtype=object)
    for col in ("max_prob_full", "second_prob_full", "margin_top2_full"):
        out[col] = np.nan

    if not mask.any():
        return out

    scored = attach_router_predictions(
        out.loc[mask], pipe, cols, experiment_id=experiment_id
    )
    out.loc[mask, "router_pred_full"] = scored["router_pred"].values
    out.loc[mask, "max_prob_full"] = scored["max_prob"].values
    out.loc[mask, "second_prob_full"] = scored["second_prob"].values
    out.loc[mask, "margin_top2_full"] = scored["margin_top2"].values
    out.loc[mask, "router_second_full"] = scored["router_second"].values
    for c in PROBA_COLS:
        if c in scored.columns:
            out.loc[mask, f"{c}_full"] = scored[c].values
    return out


def apply_selective_gate(df: pd.DataFrame, config: SelectiveGateConfig) -> pd.DataFrame:
    """Merge cheap/full router outputs into final ``router_pred`` / ``p_*``."""
    use_cheap = confident_mask(df, config)
    out = df.copy()
    out["selective_tau"] = config.tau
    out["selective_path"] = np.where(use_cheap, "cheap", "full")
    out["used_decompose"] = (~use_cheap).astype(int)

    out["router_pred"] = np.where(use_cheap, out["router_pred_cheap"], out["router_pred_full"])
    for c in PROBA_COLS:
        cheap_c = f"{c}_cheap"
        full_c = f"{c}_full"
        if cheap_c in out.columns and full_c in out.columns:
            out[c] = np.where(use_cheap, out[cheap_c], out[full_c])
        elif c in out.columns:
            pass
    out["max_prob"] = np.where(use_cheap, out["max_prob_emb"], out["max_prob_full"])
    out["second_prob"] = np.where(use_cheap, out["second_prob_emb"], out["second_prob_full"])
    out["margin_top2"] = out["max_prob"] - out["second_prob"]
    out["router_second"] = np.where(use_cheap, out["router_second_cheap"], out["router_second_full"])
    out["router_experiment"] = "selective_qce"
    out["assigned_agent"] = out["router_pred"]
    out["overall"] = out.apply(overall_complexity, axis=1)
    return out


def route_selective_dataframe(
    df: pd.DataFrame,
    config: SelectiveGateConfig,
    *,
    cheap_obj: dict[str, Any] | None = None,
    full_obj: dict[str, Any] | None = None,
    full_on_uncertain_only: bool = False,
) -> pd.DataFrame:
    """Gate on features. Set ``full_on_uncertain_only`` when ``dim_*`` exist only after partial decompose."""
    cheap_obj = cheap_obj or load_router(config.cheap_router_path)
    full_obj = full_obj or load_router(config.full_router_path)
    work = attach_cheap_predictions(df, cheap_obj)
    umask = ~confident_mask(work, config) if full_on_uncertain_only else None
    work = attach_full_predictions(work, full_obj, uncertain_mask=umask)
    return apply_selective_gate(work, config)


def route_selective_split(split: str, config: SelectiveGateConfig) -> pd.DataFrame:
    """Route a QCE v1 split using prebuilt ``complexity_record`` + ``query_embeddings``."""
    df = load_split_for_feature_set(split, "cvec5_emb")
    return route_selective_dataframe(df, config)


def selective_metrics_row(
    df: pd.DataFrame,
    config: SelectiveGateConfig,
    *,
    oracle_path: Path = DEFAULT_ORACLE,
    split: str = "",
) -> dict[str, Any]:
    outcomes = oracle_outcome_matrix(oracle_path, df["training_id"])
    scored = attach_outcomes(df, outcomes)
    m = metrics_for_routed(scored, "router_pred", name="selective_qce")
    agent_cost = pd.to_numeric(scored["exec_cost_usd"], errors="coerce")
    decompose = scored["used_decompose"].astype(float) * config.decompose_cost_usd
    total_cost = agent_cost + decompose
    n = len(scored)
    return {
        "split": split,
        "tau": config.tau,
        "tau_margin": config.tau_margin,
        "n": n,
        "pct_skip_decompose": round(100.0 * float((scored["used_decompose"] == 0).mean()), 1) if n else 0.0,
        "mean_utility_regret": round(m.mean_utility_regret, 6) if m.mean_utility_regret is not None else None,
        "exec_em": round(m.accuracy, 4) if m.accuracy is not None else None,
        "mean_agent_usd": round(float(agent_cost.mean()), 6) if n else None,
        "mean_total_usd": round(float(total_cost.mean()), 6) if n else None,
    }


def sweep_tau(
    split: str,
    config: SelectiveGateConfig,
    *,
    taus: tuple[float, ...] = SELECTIVE_TAU_GRID,
    oracle_path: Path = DEFAULT_ORACLE,
    include_baselines: bool = True,
) -> pd.DataFrame:
    """Sweep τ on one split (features loaded once per baseline; hybrid per τ)."""
    df = load_split_for_feature_set(split, "cvec5_emb")
    cheap_obj, full_obj = load_selective_routers(config)
    work = attach_cheap_predictions(df, cheap_obj)
    work = attach_full_predictions(work, full_obj)
    outcomes = oracle_outcome_matrix(oracle_path, work["training_id"])

    rows: list[dict[str, Any]] = []

    if include_baselines:

        def _baseline(which: str) -> dict[str, Any]:
            tmp = work.copy()
            if which == "cheap":
                tmp["router_pred"] = tmp["router_pred_cheap"]
                for c in PROBA_COLS:
                    cc = f"{c}_cheap"
                    if cc in tmp.columns:
                        tmp[c] = tmp[cc]
                tmp["used_decompose"] = 0
            else:
                tmp["router_pred"] = tmp["router_pred_full"]
                for c in PROBA_COLS:
                    fc = f"{c}_full"
                    if fc in tmp.columns:
                        tmp[c] = tmp[fc]
                tmp["used_decompose"] = 1
            tmp = attach_outcomes(tmp, outcomes)
            m = metrics_for_routed(tmp, "router_pred", name=which)
            agent_cost = pd.to_numeric(tmp["exec_cost_usd"], errors="coerce")
            dec = tmp["used_decompose"].astype(float) * config.decompose_cost_usd
            return {
                "split": split,
                "tau": f"always_{which}",
                "tau_margin": None,
                "n": len(tmp),
                "pct_skip_decompose": 100.0 if which == "cheap" else 0.0,
                "mean_utility_regret": round(m.mean_utility_regret, 6)
                if m.mean_utility_regret is not None
                else None,
                "exec_em": round(m.accuracy, 4) if m.accuracy is not None else None,
                "mean_agent_usd": round(float(agent_cost.mean()), 6),
                "mean_total_usd": round(float((agent_cost + dec).mean()), 6),
            }

        rows.append(_baseline("cheap"))
        rows.append(_baseline("full"))

    for tau in taus:
        cfg = SelectiveGateConfig(
            tau=float(tau),
            tau_margin=config.tau_margin,
            cheap_router_path=config.cheap_router_path,
            full_router_path=config.full_router_path,
            decompose_cost_usd=config.decompose_cost_usd,
        )
        gated = apply_selective_gate(work, cfg)
        gated = attach_outcomes(gated, outcomes)
        m = metrics_for_routed(gated, "router_pred", name="selective")
        agent_cost = pd.to_numeric(gated["exec_cost_usd"], errors="coerce")
        dec = gated["used_decompose"].astype(float) * config.decompose_cost_usd
        rows.append(
            {
                "split": split,
                "tau": float(tau),
                "tau_margin": config.tau_margin,
                "n": len(gated),
                "pct_skip_decompose": round(100.0 * float((gated["used_decompose"] == 0).mean()), 1),
                "mean_utility_regret": round(m.mean_utility_regret, 6)
                if m.mean_utility_regret is not None
                else None,
                "exec_em": round(m.accuracy, 4) if m.accuracy is not None else None,
                "mean_agent_usd": round(float(agent_cost.mean()), 6),
                "mean_total_usd": round(float((agent_cost + dec).mean()), 6),
            }
        )
    return pd.DataFrame(rows)


def recommend_tau(val_sweep: pd.DataFrame, *, full_regret_tolerance: float = 0.01) -> float:
    """
    Pick τ from val sweep.

    Prefer smallest τ with hybrid regret <= always_full + tolerance; else DEFAULT_SELECTIVE_TAU.
    """
    sub = val_sweep[val_sweep["tau"].apply(lambda x: isinstance(x, (int, float)))]
    full_row = val_sweep[val_sweep["tau"] == "always_full"]
    if sub.empty:
        return DEFAULT_SELECTIVE_TAU
    full_regret = float(full_row.iloc[0]["mean_utility_regret"]) if len(full_row) else float("inf")
    cap = full_regret + full_regret_tolerance
    ok = sub[sub["mean_utility_regret"] <= cap].copy()
    if len(ok):
        ok["dist"] = (ok["tau"] - DEFAULT_SELECTIVE_TAU).abs()
        return float(ok.sort_values(["dist", "tau"]).iloc[0]["tau"])
    best = sub.sort_values("mean_utility_regret").iloc[0]
    return float(best["tau"])


def build_eval_embeddings_only(
    base: pd.DataFrame,
    tag: str,
    *,
    verbose: bool = True,
) -> Path:
    """Write ``query_embeddings_{tag}.parquet`` only (no LLM decompose)."""
    import joblib

    from scripts.build_query_embeddings import DEFAULT_MODEL, encode_queries, matrix_to_frame

    _, e_path, _ = feature_paths(tag.strip().lower())
    if not PCA_PATH.is_file():
        raise FileNotFoundError(f"Missing {PCA_PATH}")

    pca = joblib.load(PCA_PATH)
    n_comp = int(getattr(pca, "n_components_", 16))
    if verbose:
        print(f"[selective] Encoding {len(base)} queries ({DEFAULT_MODEL})…")
    raw = encode_queries(
        base["query"].astype(str).tolist(),
        model_name=DEFAULT_MODEL,
        batch_size=32,
        normalize_embeddings=True,
    )
    e_path.parent.mkdir(parents=True, exist_ok=True)
    matrix_to_frame(base["training_id"], pca.transform(raw), n_components=n_comp).to_parquet(
        e_path, index=False
    )
    if verbose:
        print(f"[selective] Wrote {e_path}")
    return e_path


def build_eval_complexity_subset(
    base_subset: pd.DataFrame,
    tag: str,
    *,
    force_refresh: bool = False,
    verbose: bool = True,
) -> Path:
    """Decompose + C(Q) for ``base_subset`` rows; upsert into ``complexity_record_{tag}``."""
    from qce.complexity import complexity_dataframe, load_train_norm, write_complexity_parquet
    from qce.decompose import decompose_batch
    from routing.router import plans_from_cache

    if base_subset.empty:
        c_path, _, _ = feature_paths(tag)
        return c_path

    c_path, _, cache_path = feature_paths(tag.strip().lower())
    rows = base_subset.to_dict(orient="records")
    for r in rows:
        r["dataset"] = resolve_dataset_name(str(r.get("dataset") or tag))

    if verbose:
        print(f"[selective] Decomposing {len(rows)} uncertain queries…")
    decompose_batch(rows, cache_path=cache_path, force_refresh=force_refresh)
    plans = plans_from_cache(rows, cache_path)
    new_c = complexity_dataframe(plans, norm=load_train_norm(TRAIN_NORM_JSON), fit_norm=False)

    if c_path.is_file():
        existing = pd.read_parquet(c_path)
        keep = existing[~existing["training_id"].isin(new_c["training_id"])]
        merged = pd.concat([keep, new_c], ignore_index=True)
    else:
        merged = new_c
    c_path.parent.mkdir(parents=True, exist_ok=True)
    write_complexity_parquet(merged, c_path)
    if verbose:
        print(f"[selective] Updated {c_path} (+{len(new_c)} rows)")
    return c_path


def merge_embeddings_only(base: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Merge base with embedding parquet only (no ``dim_*``)."""
    _, e_path, _ = feature_paths(tag)
    if not e_path.is_file():
        raise FileNotFoundError(f"missing embeddings for {tag!r}: {e_path}")
    emb = pd.read_parquet(e_path)
    drop_e = [x for x in emb.columns if x in base.columns and x != "training_id"]
    merged = base.merge(
        emb.drop(columns=[c for c in drop_e if c in emb.columns], errors="ignore"),
        on="training_id",
        how="left",
        validate="m:1",
    )
    if len(merged) != len(base):
        raise ValueError(f"embedding merge changed row count: {len(base)} → {len(merged)}")
    return merged.reset_index(drop=True)


def selective_build_and_route(
    base: pd.DataFrame,
    tag: str,
    config: SelectiveGateConfig,
    *,
    build: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Eval/GAIA selective path: embeddings for all; decompose only for uncertain rows.

    If ``build=False``, requires existing embedding parquet; runs partial decompose when needed.
    """
    cheap_obj, full_obj = load_selective_routers(config)
    cheap_cols = list(cheap_obj["feature_cols"])

    _, e_path, _ = feature_paths(tag)
    if build or not e_path.is_file():
        build_eval_embeddings_only(base, tag, verbose=verbose)

    emb_df = merge_embeddings_only(base, tag)
    missing_emb = [c for c in cheap_cols if c not in emb_df.columns]
    if missing_emb:
        raise ValueError(f"embedding merge missing columns: {missing_emb[:5]}")

    work = attach_cheap_predictions(emb_df, cheap_obj)
    uncertain_mask = ~confident_mask(work, config)
    n_uncertain = int(uncertain_mask.sum())

    c_path, _, _ = feature_paths(tag)
    if n_uncertain:
        uncertain_ids = set(work.loc[uncertain_mask, "training_id"].astype(str))
        missing_ids: set[str] = uncertain_ids
        if c_path.is_file():
            have = set(pd.read_parquet(c_path, columns=["training_id"])["training_id"].astype(str))
            missing_ids = uncertain_ids - have
        if build or missing_ids:
            sub_base = base[base["training_id"].astype(str).isin(missing_ids or uncertain_ids)].copy()
            build_eval_complexity_subset(
                sub_base, tag, force_refresh=force_refresh, verbose=verbose
            )
        elif verbose:
            print(f"[selective] {n_uncertain} uncertain; complexity parquet already has dim_*")

    if n_uncertain and not c_path.is_file():
        raise FileNotFoundError(f"complexity_record missing for {tag!r}; use --build-features")

  # Merge dim_* + emb for full-router rows only
    if c_path.is_file() and n_uncertain:
        full_feat = merge_feature_tables(base, tag)
        work = attach_cheap_predictions(full_feat, cheap_obj)
        uncertain_mask = ~confident_mask(work, config)
        work = attach_full_predictions(work, full_obj, uncertain_mask=uncertain_mask)
    else:
        work = attach_full_predictions(
            work, full_obj, uncertain_mask=pd.Series(False, index=work.index)
        )

    return apply_selective_gate(work, config)


def save_recommended_tau(
    tau: float,
    val_sweep: pd.DataFrame,
    *,
    path: Path | None = None,
    rule: str = "",
) -> Path:
    from routing.config import SELECTIVE_TAU_JSON

    out = path or SELECTIVE_TAU_JSON
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "recommended_tau": tau,
        "rule": rule or "smallest tau with val regret <= always_full + 0.01",
        "val_sweep_rows": len(val_sweep),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out
