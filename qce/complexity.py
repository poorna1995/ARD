"""QCE complexity: C(Q) vector v1.1 (topology + plan semantics) + scalar.

C(Q) groups (``dim_*``) explicitly include relation_type, answer_granularity, and
search_query aggregates — not topology-only. Router R uses the full 5-vector
(``feature_set='cvec'``). ``complexity_graph`` is secondary (calibration/gating).

Normalization: fit ``TrainNorm`` on train plans only; pass the same ``norm`` to
val/test/eval (never refit on eval).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.local.constants import DIMS, TRUST_COLS
from qce.decompose import load_plans_jsonl
from qce.graph import (
    GraphBuildResult,
    TrainNorm,
    build_graphs_from_plans,
    build_task_dag,
    features_dataframe,
    fit_train_norm,
    rescore_dataframe,
)

C_VECTOR_VER = DIMS.ver_main
C_VECTOR_COLS: tuple[str, ...] = DIMS.main
SCALAR_COL = "complexity_graph"
ROUTER_MAIN_FEATURE_SET = "cvec"
TRUST_SCALAR_COLS = TRUST_COLS

_OUTPUT_COLS = (
    "training_id",
    "dataset",
    "c_vector_ver",
    "final_constraint",
    "graph_status",
    "terminal_sink_ok",
    "sink_intermediate_risk",
    *C_VECTOR_COLS,
    *DIMS.legacy,
    SCALAR_COL,
    "plan_trust",
    "verify_fraction",
    "relation_diversity",
    "scoped_retrieve_fraction",
    "weak_search_query_fraction",
)


def c_vector(features: dict[str, Any]) -> np.ndarray:
    missing = [c for c in C_VECTOR_COLS if c not in features]
    if missing:
        raise KeyError(f"C(Q) missing keys {missing}; run rescore_dataframe or complexity_dataframe")
    return np.asarray([float(features[c]) for c in C_VECTOR_COLS], dtype=np.float64)


def complexity_scalar(features: dict[str, Any]) -> float:
    if SCALAR_COL not in features:
        raise KeyError(f"missing {SCALAR_COL!r}")
    return float(features[SCALAR_COL])


def fit_norm_from_plans(plans: list[dict[str, Any]]) -> TrainNorm:
    """Train-split caps only — reuse returned norm for val/test/eval."""
    return fit_train_norm(features_dataframe(build_graphs_from_plans(plans)))


def fit_norm_from_jsonl(path: str | Path) -> TrainNorm:
    return fit_norm_from_plans(load_plans_jsonl(path))


def save_train_norm(norm: TrainNorm, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(norm), indent=2), encoding="utf-8")
    return path


def load_train_norm(path: str | Path) -> TrainNorm:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return TrainNorm(**{k: float(data[k]) for k in asdict(TrainNorm()).keys() if k in data})


def complexity_from_plan(
    plan: dict[str, Any],
    *,
    norm: TrainNorm | None = None,
) -> dict[str, Any]:
    return complexity_from_result(build_task_dag(plan), norm=norm or TrainNorm())


def complexity_from_result(
    result: GraphBuildResult,
    *,
    norm: TrainNorm | None = None,
) -> dict[str, Any]:
    row = rescore_dataframe(
        pd.DataFrame([dict(result.features)]), norm or TrainNorm()
    ).iloc[0].to_dict()
    return _slice_record(row)


def complexity_dataframe(
    plans: list[dict[str, Any]],
    *,
    norm: TrainNorm | None = None,
    fit_norm: bool = False,
) -> pd.DataFrame:
    """
    Build C(Q) for a plan split.

    - **Train:** ``fit_norm=True`` (or pass pre-fit ``norm``).
    - **Val/test/eval:** must pass train ``norm``; ``fit_norm`` must be False.
    """
    if norm is None:
        if not fit_norm:
            raise ValueError(
                "val/test/eval require train-fitted norm=... ; "
                "use fit_norm=True only on the train split."
            )
        norm = fit_norm_from_plans(plans)
    elif fit_norm:
        raise ValueError("Do not set fit_norm=True when norm is already provided.")

    full = rescore_dataframe(features_dataframe(build_graphs_from_plans(plans)), norm)
    out = full[[c for c in _OUTPUT_COLS if c in full.columns]].copy()
    out["c_vector_ver"] = C_VECTOR_VER
    return out


def complexity_splits(
    train_plans: list[dict[str, Any]],
    val_plans: list[dict[str, Any]] | None = None,
    test_plans: list[dict[str, Any]] | None = None,
) -> tuple[TrainNorm, pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None]:
    """Fit norm on train; apply to val/test without refitting."""
    norm = fit_norm_from_plans(train_plans)
    train_df = complexity_dataframe(train_plans, norm=norm)
    val_df = complexity_dataframe(val_plans, norm=norm) if val_plans else None
    test_df = complexity_dataframe(test_plans, norm=norm) if test_plans else None
    return norm, train_df, val_df, test_df


def graph_features_dataframe(
    plans: list[dict[str, Any]],
    *,
    norm: TrainNorm,
) -> pd.DataFrame:
    return rescore_dataframe(features_dataframe(build_graphs_from_plans(plans)), norm)


def router_input_matrix(df: pd.DataFrame) -> np.ndarray:
    cols = list(C_VECTOR_COLS)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"dataframe missing C(Q) columns: {missing}")
    return df[cols].to_numpy(dtype=np.float64)


def router_scalar_vector(df: pd.DataFrame) -> np.ndarray:
    if SCALAR_COL not in df.columns:
        raise KeyError(f"dataframe missing {SCALAR_COL!r}")
    return df[[SCALAR_COL]].to_numpy(dtype=np.float64)


def router_feature_cols() -> tuple[str, ...]:
    return C_VECTOR_COLS


def router_scalar_col() -> str:
    return SCALAR_COL


def calibration_summary(
    df: pd.DataFrame,
    outcome_col: str,
    *,
    score_col: str = SCALAR_COL,
    n_bins: int = 5,
) -> dict[str, Any]:
    if score_col not in df.columns:
        raise KeyError(f"missing score column {score_col!r}")
    if outcome_col not in df.columns:
        raise KeyError(f"missing outcome column {outcome_col!r}")

    sub = df[[score_col, outcome_col]].dropna()
    if sub.empty:
        return {"n": 0, "score_col": score_col, "outcome_col": outcome_col}

    outcome_series = pd.to_numeric(sub[outcome_col], errors="coerce")
    s = sub[score_col]
    out: dict[str, Any] = {
        "n": int(len(sub)),
        "score_col": score_col,
        "outcome_col": outcome_col,
        "score_mean": round(float(s.mean()), 4),
        "outcome_mean": round(float(outcome_series.mean()), 4),
    }
    if len(sub) >= 3 and outcome_series.notna().sum() >= 3:
        out["pearson_r"] = round(float(s.corr(outcome_series, method="pearson")), 4)
        out["spearman_r"] = round(float(s.corr(outcome_series, method="spearman")), 4)

    try:
        sub = sub.assign(_bin=pd.qcut(s, q=min(n_bins, len(sub)), duplicates="drop"))
        bins = (
            sub.groupby("_bin", observed=True)[outcome_col]
            .agg(["mean", "count"])
            .reset_index()
        )
        out["bins"] = [
            {
                "bin": str(row["_bin"]),
                "mean_outcome": round(float(row["mean"]), 4),
                "count": int(row["count"]),
            }
            for _, row in bins.iterrows()
        ]
    except ValueError:
        out["bins"] = []

    return out


def gate_hard(score: float, *, low: float, high: float) -> str:
    if score < low:
        return "low"
    if score > high:
        return "high"
    return "mid"


def _slice_record(features: dict[str, Any]) -> dict[str, Any]:
    row = {c: features[c] for c in C_VECTOR_COLS if c in features}
    for c in _OUTPUT_COLS:
        if c in features and c not in row and c != "c_vector_ver":
            row[c] = features[c]
    row["c_vector_ver"] = C_VECTOR_VER
    return row


def write_complexity_parquet(df: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Build C(Q) parquet (train-fit norm optional)")
    p.add_argument("--plans", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--norm-in", type=Path, default=None, help="train-fitted norm JSON")
    p.add_argument("--norm-out", type=Path, default=None, help="write norm when --fit-norm")
    p.add_argument("--fit-norm", action="store_true", help="fit caps on this file (train only)")
    args = p.parse_args()
    plans = load_plans_jsonl(args.plans)
    norm = load_train_norm(args.norm_in) if args.norm_in else None
    df = complexity_dataframe(plans, norm=norm, fit_norm=args.fit_norm)
    if args.norm_out:
        save_train_norm(
            norm or fit_norm_from_plans(plans),
            args.norm_out,
        )
    write_complexity_parquet(df, args.out)
    print(f"wrote {len(df)} rows → {args.out.resolve()}")
    if len(df) == 0:
        raise SystemExit("no plans loaded — check --plans path (not a placeholder like <gaia_plans>)")
    if "terminal_sink_ok" in df.columns:
        print(f"terminal_sink_ok rate: {df['terminal_sink_ok'].mean():.3f}")
    v1 = [c for c in C_VECTOR_COLS if c in df.columns]
    if v1:
        print(df[v1].describe().round(3))
