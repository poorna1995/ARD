#!/usr/bin/env python3
"""
Evaluate graph_emb router on ``datasets/eval_samples/gaia.parquet``.

Pipeline:
  1. (--build-features) LLM decompose → C(Q) with train norm → query embeddings (train PCA)
  2. Merge features and run saved router (default: G3 tuned)
  3. Compare predictions to ``gaia_routed_results.csv`` heuristic routes (optional)

Usage::

  # First time (needs OPENAI_API_KEY for decompose):
  uv run python routing/eval_gaia.py --build-features

  # Predict only (after features exist):
  uv run python routing/eval_gaia.py

  uv run python routing/eval_gaia.py --router models/router/G3_hgbm_graph_emb_tuned.joblib --plots
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from routing.label_encoding import LabelEncodingClassifier  # noqa: E402
from routing.router_analysis import (  # noqa: E402
    AGENTS,
    attach_router_predictions,
    plot_complexity_region,
    plot_reliability_calibration,
    plot_router_on_complexity,
)
from routing.train_router import (  # noqa: E402
    PROBA_COLS,
    REPO_ROOT,
    load_router,
    predict_agent_proba,
    resolve_feature_cols,
)

GAIA_PARQUET = REPO_ROOT / "datasets/eval_samples/gaia.parquet"
GAIA_ROUTED_CSV = REPO_ROOT / "datasets/eval_samples/gaia_routed_results.csv"
GAIA_PLANS_CACHE = REPO_ROOT / "datasets/decomposer_cache/qce_gaia_plans.jsonl"
GAIA_COMPLEXITY_PARQUET = REPO_ROOT / "datasets/qce_features/complexity_record_gaia.parquet"
GAIA_EMB_PARQUET = REPO_ROOT / "datasets/qce_features/query_embeddings_gaia.parquet"
TRAIN_NORM_JSON = REPO_ROOT / "models/qce_graph/train_norm.json"
PCA_PATH = REPO_ROOT / "models/query_embeddings/pca_16.joblib"
DEFAULT_ROUTER = REPO_ROOT / "models/router/G3_hgbm_graph_emb_tuned.joblib"
OUT_DIR = REPO_ROOT / "results/router_analysis/gaia"

ROUTER_AGENTS = frozenset(AGENTS)


def gaia_base_frame(parquet_path: Path = GAIA_PARQUET) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    if "id" not in df.columns or "query" not in df.columns:
        raise ValueError(f"{parquet_path}: need id, query columns")
    out = pd.DataFrame(
        {
            "training_id": df["id"].astype(str),
            "query": df["query"].astype(str),
            "dataset": "gaia",
            "expected_answer": df["answer"] if "answer" in df.columns else "",
        }
    )
    if "level" in df.columns:
        out["level"] = df["level"]
    if "file_name" in df.columns:
        out["file_name"] = df["file_name"]
    return out


def build_gaia_features(
    *,
    force_refresh: bool = False,
    verbose: bool = True,
) -> None:
    """Decompose GAIA queries, build C(Q) + embeddings (train-fitted norm / PCA)."""
    from qce.complexity import complexity_dataframe, load_train_norm, write_complexity_parquet
    from qce.decompose import decompose_batch, write_plans_jsonl
    from scripts.build_query_embeddings import (
        DEFAULT_MODEL,
        encode_queries,
        matrix_to_frame,
    )

    base = gaia_base_frame()
    rows = base.to_dict(orient="records")
    for r in rows:
        r["dataset"] = "gaia"

    if verbose:
        print(f"Decomposing {len(rows)} GAIA queries (cache: {GAIA_PLANS_CACHE})…")
    plans = decompose_batch(
        rows,
        cache_path=GAIA_PLANS_CACHE,
        force_refresh=force_refresh,
    )
    write_plans_jsonl(GAIA_PLANS_CACHE, plans)

    if not TRAIN_NORM_JSON.is_file():
        raise FileNotFoundError(
            f"Missing train norm {TRAIN_NORM_JSON} — run QCE train complexity first."
        )
    norm = load_train_norm(TRAIN_NORM_JSON)
    c_df = complexity_dataframe(plans, norm=norm, fit_norm=False)
    GAIA_COMPLEXITY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    write_complexity_parquet(c_df, GAIA_COMPLEXITY_PARQUET)
    if verbose:
        print(f"Wrote {GAIA_COMPLEXITY_PARQUET} ({len(c_df)} rows)")

    if not PCA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {PCA_PATH} — run: uv run python scripts/build_query_embeddings.py --split all"
        )
    pca = joblib.load(PCA_PATH)
    n_comp = int(getattr(pca, "n_components_", 16))
    texts = base["query"].astype(str).tolist()
    if verbose:
        print(f"Encoding {len(texts)} queries with {DEFAULT_MODEL}…")
    raw = encode_queries(
        texts,
        model_name=DEFAULT_MODEL,
        batch_size=32,
        normalize_embeddings=True,
    )
    reduced = pca.transform(raw)
    emb_df = matrix_to_frame(base["training_id"], reduced, n_components=n_comp)
    emb_df.to_parquet(GAIA_EMB_PARQUET, index=False)
    if verbose:
        print(f"Wrote {GAIA_EMB_PARQUET} ({len(emb_df)} rows)")


def load_gaia_router_frame(
    *,
    require_features: bool = True,
) -> pd.DataFrame:
    base = gaia_base_frame()
    if require_features:
        if not GAIA_COMPLEXITY_PARQUET.is_file() or not GAIA_EMB_PARQUET.is_file():
            raise FileNotFoundError(
                "GAIA QCE features missing. Run:\n"
                "  uv run python routing/eval_gaia.py --build-features"
            )
        c = pd.read_parquet(GAIA_COMPLEXITY_PARQUET)
        e = pd.read_parquet(GAIA_EMB_PARQUET)
        drop_c = [x for x in c.columns if x in base.columns and x != "training_id"]
        drop_e = [x for x in e.columns if x in base.columns and x != "training_id"]
        merged = base.merge(c.drop(columns=drop_c, errors="ignore"), on="training_id", how="inner")
        merged = merged.merge(e.drop(columns=drop_e, errors="ignore"), on="training_id", how="inner")
        if len(merged) < len(base):
            raise ValueError(
                f"Feature merge dropped rows: {len(base)} → {len(merged)} — rebuild features."
            )
        return merged.reset_index(drop=True)
    return base


def merge_routed_benchmark(df: pd.DataFrame) -> pd.DataFrame:
    """Attach heuristic router + executed agent from ``gaia_routed_results.csv``."""
    if not GAIA_ROUTED_CSV.is_file():
        return df
    routed = pd.read_csv(GAIA_ROUTED_CSV)
    routed = routed.rename(columns={"id": "training_id"})
    keep = [
        "training_id",
        "predicted_agent",
        "agent",
        "is_correct",
        "cost_usd",
        "complexity",
    ]
    keep = [c for c in keep if c in routed.columns]
    out = df.merge(routed[keep], on="training_id", how="left", suffixes=("", "_bench"))
    if "agent" in out.columns:
        out = out.rename(columns={"agent": "bench_executed_agent"})
    if "predicted_agent" in out.columns:
        out = out.rename(columns={"predicted_agent": "bench_router_agent"})
    return out


def run_router_predict(
    df: pd.DataFrame,
    router_path: Path,
) -> pd.DataFrame:
    obj = load_router(router_path)
    pipe = obj["pipeline"]
    feature_cols = list(obj["feature_cols"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Router needs columns missing on GAIA frame: {missing}")
    out = attach_router_predictions(
        df,
        pipe,
        feature_cols,
        experiment_id=str(obj.get("experiment_id", router_path.stem)),
    )
    proba = predict_agent_proba(pipe, df, feature_cols)
    for c in PROBA_COLS:
        out[c] = proba[c].values
    return out


def benchmark_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {"n": len(df)}
    if "bench_router_agent" not in df.columns:
        summary["note"] = "gaia_routed_results.csv not merged"
        return summary

    in_four = df["bench_executed_agent"].isin(ROUTER_AGENTS)
    sub = df[in_four].copy()
    summary["n_bench_in_4_agents"] = int(len(sub))
    summary["bench_executed_distribution"] = (
        sub["bench_executed_agent"].value_counts().to_dict()
    )
    if "router_pred" in sub.columns:
        agree = sub["router_pred"] == sub["bench_router_agent"]
        summary["ml_vs_bench_router_agreement"] = float(agree.mean())
        summary["ml_router_distribution"] = sub["router_pred"].value_counts().to_dict()
        exec_agree = sub["router_pred"] == sub["bench_executed_agent"]
        summary["ml_vs_bench_executed_agreement"] = float(exec_agree.mean())
    if "is_correct" in sub.columns:
        summary["bench_executed_accuracy"] = float(
            pd.to_numeric(sub["is_correct"], errors="coerce").fillna(0).mean()
        )
    return summary


def save_plots(df: pd.DataFrame, out_dir: Path, *, router_id: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if "complexity_graph" in df.columns and "bench_executed_agent" in df.columns:
        work = df.dropna(subset=["complexity_graph", "bench_executed_agent"]).copy()
        work = work[work["bench_executed_agent"].isin(ROUTER_AGENTS)]
        if len(work):
            work["oracle_agent"] = work["bench_executed_agent"]
            plot_complexity_region(
                work,
                out_dir / "complexity_region_executed_agent.png",
                title_suffix=f" (GAIA executed agent, n={len(work)})",
            )
    if "complexity_graph" in df.columns and "router_pred" in df.columns:
        work = df.copy()
        if "bench_executed_agent" in work.columns:
            work["oracle_agent"] = work["bench_executed_agent"]
        else:
            work["oracle_agent"] = work["router_pred"]
        plot_router_on_complexity(
            work,
            out_dir / "complexity_router_vs_executed.png",
            title_suffix=" (GAIA)",
        )
    if "max_prob" in df.columns:
        df = df.copy()
        if "bench_router_agent" in df.columns:
            sub = df[df["bench_executed_agent"].isin(ROUTER_AGENTS)]
            sub["oracle_label_match"] = (sub["router_pred"] == sub["bench_router_agent"]).astype(
                int
            )
            if len(sub):
                plot_reliability_calibration(
                    sub,
                    out_dir / "reliability_vs_bench_router.png",
                    title_suffix=" (GAIA vs bench router)",
                )


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate router on GAIA eval_samples.")
    p.add_argument(
        "--build-features",
        action="store_true",
        help="Run QCE decompose + complexity + embeddings for GAIA (API cost).",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-run LLM decompose even if cache exists.",
    )
    p.add_argument(
        "--router",
        type=Path,
        default=DEFAULT_ROUTER,
        help="Router joblib path.",
    )
    p.add_argument("--plots", action="store_true", help="Write figures under results/router_analysis/gaia/.")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=OUT_DIR,
        help="Output directory for CSV/JSON.",
    )
    return p.parse_args()


def main() -> None:
    import sys as _sys

    _sys.modules.setdefault("__main__", _sys.modules[__name__])
    setattr(_sys.modules["__main__"], "LabelEncodingClassifier", LabelEncodingClassifier)

    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.build_features:
        build_gaia_features(force_refresh=args.force_refresh)

    if not args.router.is_file():
        raise FileNotFoundError(
            f"Router not found: {args.router}\n"
            "Train with: uv run python routing/tune_router.py --save"
        )

    df = load_gaia_router_frame(require_features=True)
    df = run_router_predict(df, args.router)
    df = merge_routed_benchmark(df)

    summary = benchmark_summary(df)
    summary["router"] = str(args.router)
    summary["router_distribution"] = df["router_pred"].value_counts().to_dict()
    summary["mean_max_prob"] = float(df["max_prob"].mean()) if "max_prob" in df.columns else None

    pred_path = args.out_dir / "gaia_router_predictions.csv"
    df.to_csv(pred_path, index=False)
    summary_path = args.out_dir / "gaia_eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\n=== GAIA router eval (n={len(df)}) ===")
    print(f"Router: {args.router.name}")
    print(json.dumps(summary, indent=2))
    print(f"\nWrote {pred_path}")
    print(f"Wrote {summary_path}")

    if args.plots:
        save_plots(df, args.out_dir, router_id=args.router.stem)
        print(f"Wrote plots under {args.out_dir}/")

    if "bench_router_agent" in df.columns:
        sub = df[df["bench_executed_agent"].isin(ROUTER_AGENTS)].copy()
        if len(sub):
            sub["oracle_label_match"] = (
                sub["router_pred"] == sub["bench_router_agent"]
            ).astype(int)
            print(
                f"\nML router agrees with benchmark router on "
                f"{sub['oracle_label_match'].sum()}/{len(sub)} "
                f"({sub['oracle_label_match'].mean():.1%}) four-agent GAIA rows."
            )


if __name__ == "__main__":
    main()
