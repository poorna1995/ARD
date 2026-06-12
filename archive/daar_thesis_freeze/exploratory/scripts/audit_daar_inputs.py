"""Audit D-AAR inputs: features, train/infer alignment, eval_samples parity."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.global_config.paths import daar_embeddings_path, daar_features_path
from config.local.router.production import PRIMARY_ROUTER_PATH, PRODUCTION_FEATURE_SET
from daar.gate1 import attach_soft_labels, load_frame
from daar.routing import build_query_table, load_gate1, route_agent
from eval.benchmark import list_eval_sample_datasets, load_routable_eval_frame
from research.soft_train import soft_train_mask
from router.router import load_router, predict_agent_proba, router_feature_cols


def _feature_stats(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    return {
        "n_rows": len(df),
        "missing_cols": [c for c in cols if c not in df.columns],
        "na_frac": {c: float(df[c].isna().mean()) for c in cols if c in df.columns},
        "mean": {c: float(df[c].mean()) for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])},
    }


def audit() -> dict[str, Any]:
    p1_pipe, p1_man, p1_cols = load_gate1("p1")
    base = load_router(PRIMARY_ROUTER_PATH)
    base_cols = list(base["feature_cols"])

    val_long = load_frame("val")
    val_q = attach_soft_labels(val_long, utility_lambda=25.0)
    train_q = attach_soft_labels(load_frame("train"), utility_lambda=25.0)

    agent_mismatch: dict[str, int] = {}
    for c in p1_cols[:5]:
        wide = val_long.pivot_table(index="training_id", columns="agent", values=c, aggfunc="first")
        agent_mismatch[c] = int((wide.nunique(axis=1) > 1).sum())

    p1_pred = predict_agent_proba(p1_pipe, val_q, p1_cols).idxmax(axis=1).str.removeprefix("p_")
    base_pred = predict_agent_proba(base["pipeline"], val_q, base_cols).idxmax(axis=1).str.removeprefix("p_")

    val_table = build_query_table("val", variant="p1", gate1=p1_pipe, feature_cols=p1_cols)
    routed = []
    for _, row in val_table.iterrows():
        a, _ = route_agent(row, lam=0.001, theta_s=0.0)
        routed.append(a)
    rt = pd.Series(routed, index=val_table["training_id"].astype(str))
    pq = pd.Series(p1_pred.values, index=val_q["training_id"].astype(str))
    common = rt.index.intersection(pq.index)

    eval_reports: dict[str, Any] = {}
    for ds in list_eval_sample_datasets():
        hot = load_routable_eval_frame(ds, build_features=False)
        eval_reports[ds] = {
            "stats": _feature_stats(hot, p1_cols),
            "mean_delta_vs_daar_val": {
                c: float(hot[c].mean() - val_q[c].mean())
                for c in p1_cols
                if c in hot.columns and c in val_q.columns
            },
        }

    feat = pd.read_parquet(daar_features_path())
    emb = pd.read_parquet(daar_embeddings_path())

    return {
        "p1_feature_cols": p1_cols,
        "baseline_feature_cols": base_cols,
        "cols_match_p1_baseline": p1_cols == base_cols,
        "daar_val": {
            "query_rows": len(val_q),
            "long_rows": len(val_long),
            "feature_stats": _feature_stats(val_q, p1_cols),
            "agent_phi_mismatch_queries": agent_mismatch,
            "p1_vs_baseline_argmax_agreement": float((p1_pred.values == base_pred.values).mean()),
            "route_agent_vs_argmax_proba": float((rt.loc[common] == pq.loc[common]).mean()),
        },
        "p1_train": {
            "n_queries": len(train_q),
            "soft_kl_queries": int(soft_train_mask(train_q).sum()),
            "excluded_unsolvable": int((~soft_train_mask(train_q)).sum()),
        },
        "daar_feature_artifacts": {
            "features_path": str(daar_features_path()),
            "n_feature_ids": int(feat["training_id"].nunique()),
            "embeddings_path": str(daar_embeddings_path()),
            "n_emb_ids": int(emb["training_id"].nunique()),
        },
        "eval_samples": eval_reports,
    }


def main() -> None:
    report = audit()
    out = Path(__file__).resolve().parents[1] / "datasets/daar/routing/input_audit.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
