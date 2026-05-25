#!/usr/bin/env python3
"""
Lightweight hyperparameter tuning for graph_emb routers (paper-facing).

Tunes small grids for HGBM + MLP on train CV only, then fits calibrated models
and reports val/test metrics plus threshold routing (no massive search).

Usage::

  uv run python routing/tune_router.py
  uv run python routing/tune_router.py --save
  uv run python routing/tune_router.py --skip-grid  # calibration + thresholds only
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from routing.label_encoding import LabelEncodingClassifier  # noqa: E402
from routing.router_analysis import (  # noqa: E402
    build_analysis_frame,
    plot_reliability_calibration,
    router_confidence_sweep,
    top2_sweep,
)
from routing.train_router import (  # noqa: E402
    REPO_ROOT,
    TARGET,
    WeightMode,
    compute_sample_weights,
    evaluate,
    fit_router,
    load_split,
    make_pipeline,
    predict_agent_proba,
    resolve_feature_cols,
    save_router,
)

# Principled small grids (graph_emb only).
HGBM_GRID: dict[str, list[Any]] = {
    "max_depth": [3, 5, 7],
    "learning_rate": [0.03, 0.05, 0.1],
    "max_leaf_nodes": [15, 31, 63],
}
MLP_GRID: dict[str, list[Any]] = {
    "hidden_layer_sizes": [(64,), (128,), (128, 64)],
    "alpha": [1e-4, 1e-3],
    "learning_rate_init": [5e-4, 1e-3],
}

FEATURE_SET = "graph_emb"
OUT_DIR = REPO_ROOT / "results/router_tuning"


def _grid_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [
        dict(zip(keys, vals, strict=True))
        for vals in itertools.product(*(grid[k] for k in keys))
    ]


def _cv_score_params(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    model: str,
    weight_mode: WeightMode,
    params: dict[str, Any],
    use_class_weight: bool,
    cv: int = 5,
) -> float:
    class_weight = "balanced" if use_class_weight and weight_mode in ("none", "balanced") else None
    hgbm_p = params if model == "hgbm" else None
    mlp_p = params if model == "mlp" else None
    pipe = make_pipeline(
        feature_cols,
        model=model,
        class_weight=class_weight,
        hgbm_params=hgbm_p,
        mlp_params=mlp_p,
        calibrated=False,
    )
    y = train_df[TARGET].astype(str)
    weights = compute_sample_weights(train_df, y, weight_mode)
    fit_params = {"clf__sample_weight": weights} if weights is not None else {}
    scores = cross_val_score(
        pipe,
        train_df[feature_cols],
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        params=fit_params,
    )
    return float(scores.mean())


def tune_model(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    model: Literal["hgbm", "mlp"],
    weight_mode: WeightMode,
    use_class_weight: bool,
    grid: dict[str, list[Any]],
    cv: int = 5,
    verbose: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    combos = _grid_combos(grid)
    rows: list[dict[str, Any]] = []
    best_score = -1.0
    best_params: dict[str, Any] = combos[0]

    for i, params in enumerate(combos):
        score = _cv_score_params(
            train_df,
            feature_cols,
            model=model,
            weight_mode=weight_mode,
            params=params,
            use_class_weight=use_class_weight,
            cv=cv,
        )
        rows.append({**params, "cv_macro_f1": round(score, 4)})
        if score > best_score:
            best_score = score
            best_params = params
        if verbose and (i + 1) % 9 == 0:
            print(f"  {model}: {i + 1}/{len(combos)} grid points done…")

    table = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    if verbose:
        print(f"\nBest {model} CV macro-F1: {best_score:.4f}")
        print(f"Params: {best_params}")
        print(table.head(5).to_string(index=False))
    return best_params, table


def fit_final_router(
    train_df: pd.DataFrame,
    *,
    model: str,
    weight_mode: WeightMode,
    use_class_weight: bool,
    params: dict[str, Any],
    calibrated: bool,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> tuple[Any, list[str]]:
    feature_cols = resolve_feature_cols(train_df, FEATURE_SET)
    class_weight = "balanced" if use_class_weight and weight_mode in ("none", "balanced") else None
    pipe = make_pipeline(
        feature_cols,
        model=model,
        class_weight=class_weight,
        hgbm_params=params if model == "hgbm" else None,
        mlp_params=params if model == "mlp" else None,
        calibrated=calibrated,
        calibration_method=calibration_method,
    )
    y = train_df[TARGET].astype(str)
    weights = compute_sample_weights(train_df, y, weight_mode)
    fit_router(pipe, train_df, feature_cols, sample_weight=weights)
    return pipe, feature_cols


def _ece_max_prob(df: pd.DataFrame, y_col: str, n_bins: int = 10) -> float:
    work = df.dropna(subset=["max_prob"])
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = 0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (
            (work["max_prob"] >= lo) & (work["max_prob"] < hi)
            if i < n_bins - 1
            else (work["max_prob"] >= lo) & (work["max_prob"] <= hi)
        )
        sub = work[mask]
        if len(sub) == 0:
            continue
        conf = sub["max_prob"].mean()
        acc = sub[y_col].mean()
        ece += len(sub) * abs(acc - conf)
        n += len(sub)
    return float(ece / max(n, 1))


def eval_split(
    pipe: Any,
    feature_cols: list[str],
    df: pd.DataFrame,
    *,
    split: str,
) -> dict[str, Any]:
    res = evaluate(pipe, df, feature_cols)
    proba_df = df.copy()
    proba = predict_agent_proba(pipe, df, feature_cols)
    for c in proba.columns:
        proba_df[c] = proba[c].values
    proba_df["router_pred"] = pipe.predict(df[feature_cols])
    proba_df["max_prob"] = proba.max(axis=1).values
    proba_df["oracle_label_match"] = (
        proba_df["router_pred"] == proba_df[TARGET]
    ).astype(int)

    return {
        "split": split,
        "accuracy": res.accuracy,
        "macro_f1": res.macro_f1,
        "ece_label_match": _ece_max_prob(proba_df, "oracle_label_match"),
        "mean_max_prob": float(proba_df["max_prob"].mean()),
        "report": res.report,
    }


def run_tuning(
    *,
    save: bool = False,
    skip_grid: bool = False,
    cv: int = 5,
    verbose: bool = True,
) -> dict[str, Any]:
    train_df = load_split("train", with_embeddings=True)
    val_df = load_split("val", with_embeddings=True)
    test_df = load_split("test", with_embeddings=True)
    feature_cols = resolve_feature_cols(train_df, FEATURE_SET)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    summary: dict[str, Any] = {
        "feature_set": FEATURE_SET,
        "n_train": len(train_df),
        "cv_folds": cv,
        "grids": {"hgbm": HGBM_GRID, "mlp": MLP_GRID},
    }

    # ── 1. HGBM grid (balanced, like G3) ──
    if skip_grid:
        hgbm_best = {"max_depth": 6, "learning_rate": 0.05, "max_leaf_nodes": 31}
        hgbm_grid_df = pd.DataFrame()
    else:
        if verbose:
            print(f"\n=== HGBM grid ({len(_grid_combos(HGBM_GRID))} combos, train CV) ===")
        hgbm_best, hgbm_grid_df = tune_model(
            train_df,
            feature_cols,
            model="hgbm",
            weight_mode="balanced",
            use_class_weight=True,
            grid=HGBM_GRID,
            cv=cv,
            verbose=verbose,
        )
        hgbm_grid_df.to_csv(OUT_DIR / "hgbm_grid_cv.csv", index=False)

    summary["hgbm_best_params"] = hgbm_best

    # ── 2. MLP grid (balanced_soft, like G5) ──
    if skip_grid:
        mlp_best = {"hidden_layer_sizes": (128, 64), "alpha": 1e-3, "learning_rate_init": 1e-3}
        mlp_grid_df = pd.DataFrame()
    else:
        if verbose:
            print(f"\n=== MLP grid ({len(_grid_combos(MLP_GRID))} combos, train CV) ===")
        mlp_best, mlp_grid_df = tune_model(
            train_df,
            feature_cols,
            model="mlp",
            weight_mode="balanced_soft",
            use_class_weight=False,
            grid=MLP_GRID,
            cv=cv,
            verbose=verbose,
        )
        mlp_grid_df.to_csv(OUT_DIR / "mlp_grid_cv.csv", index=False)

    summary["mlp_best_params"] = mlp_best

    models: dict[str, Any] = {}

    # ── 3. Fit tuned (uncalibrated) + calibrated variants on full train ──
    configs = [
        ("T_hgbm_graph_emb_tuned", "hgbm", "balanced", True, hgbm_best, False),
        ("T_hgbm_graph_emb_calibrated", "hgbm", "balanced", True, hgbm_best, True),
        ("T_mlp_graph_emb_tuned", "mlp", "balanced_soft", False, mlp_best, False),
        ("T_mlp_graph_emb_calibrated", "mlp", "balanced_soft", False, mlp_best, True),
    ]

    for exp_id, model, wmode, use_cw, params, cal in configs:
        pipe, fcols = fit_final_router(
            train_df,
            model=model,
            weight_mode=wmode,
            use_class_weight=use_cw,
            params=params,
            calibrated=cal,
        )
        models[exp_id] = (pipe, fcols)
        row: dict[str, Any] = {
            "experiment_id": exp_id,
            "model": model,
            "weight_mode": wmode,
            "calibrated": cal,
            "params": params,
        }
        for split_name, split_df in [("val", val_df), ("test", test_df)]:
            row[split_name] = eval_split(pipe, fcols, split_df, split=split_name)
        summary[exp_id] = row

        if save:
            save_router(
                pipe,
                feature_cols=fcols,
                experiment_id=exp_id,
                extra_meta={
                    "feature_set": FEATURE_SET,
                    "weight_mode": wmode,
                    "calibrated": cal,
                    "tuned_params": params,
                    "val_macro_f1": row["val"]["macro_f1"],
                    "test_macro_f1": row["test"]["macro_f1"],
                },
            )

    # ── 4. Threshold routing (tuned HGBM: deploy F1; calibrated: uncertainty) ──
    if save:
        tuned_pipe, tuned_fcols = models["T_hgbm_graph_emb_tuned"]
        save_router(
            tuned_pipe,
            feature_cols=tuned_fcols,
            experiment_id="G3_hgbm_graph_emb_tuned",
            path=REPO_ROOT / "models/router/G3_hgbm_graph_emb_tuned.joblib",
            extra_meta={"tuned_params": hgbm_best, "val_macro_f1": summary["T_hgbm_graph_emb_tuned"]["val"]["macro_f1"]},
        )
        save_router(
            models["T_hgbm_graph_emb_calibrated"][0],
            feature_cols=models["T_hgbm_graph_emb_calibrated"][1],
            experiment_id="G3_hgbm_graph_emb_tuned_calibrated",
            path=REPO_ROOT / "models/router/G3_hgbm_graph_emb_tuned_calibrated.joblib",
            extra_meta={"tuned_params": hgbm_best, "calibrated": True},
        )

    threshold_reports: dict[str, Any] = {}
    for route_id in ("T_hgbm_graph_emb_tuned", "T_hgbm_graph_emb_calibrated"):
        pipe, fcols = models[route_id]
        threshold_reports[route_id] = {}
        for split_name in ("val", "test"):
            analysis = build_analysis_frame(
                split_name,
                pipe=pipe,
                feature_cols=fcols,
                experiment_id=route_id,
                with_embeddings=True,
            )
            conf = router_confidence_sweep(analysis)
            top2 = top2_sweep(analysis)
            prefix = f"{route_id}_{split_name}"
            conf.to_csv(OUT_DIR / f"confidence_sweep_{prefix}.csv", index=False)
            top2.to_csv(OUT_DIR / f"top2_sweep_{prefix}.csv", index=False)
            plot_reliability_calibration(
                analysis,
                OUT_DIR / f"calibration_{prefix}.png",
                title_suffix=f" ({route_id}, {split_name})",
            )
            threshold_reports[route_id][split_name] = {
                "best_top2": top2.loc[top2["accuracy"].idxmax()].to_dict() if len(top2) else {},
                "best_confidence_row": conf.loc[conf["accuracy"].idxmax()].to_dict()
                if len(conf)
                else {},
            }

    summary["threshold_routing"] = threshold_reports

    # Compare to uncalibrated G3 baseline on val
    if verbose:
        print("\n=== Tuned model comparison (val) ===")
        rows = []
        for exp_id in models:
            rows.append(
                {
                    "experiment": exp_id,
                    "macro_f1": summary[exp_id]["val"]["macro_f1"],
                    "accuracy": summary[exp_id]["val"]["accuracy"],
                    "ece": summary[exp_id]["val"]["ece_label_match"],
                }
            )
        print(pd.DataFrame(rows).to_string(index=False))

    (OUT_DIR / "tuning_summary.json").write_text(
        json.dumps(summary, indent=2, default=str),
        encoding="utf-8",
    )
    if verbose:
        print(f"\nWrote {OUT_DIR / 'tuning_summary.json'}")
    return summary


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Lightweight graph_emb router tuning.")
    p.add_argument("--save", action="store_true", help="Save tuned joblibs under models/router/.")
    p.add_argument(
        "--skip-grid",
        action="store_true",
        help="Skip CV grid; only fit with default/best-known params + calibration.",
    )
    p.add_argument("--cv", type=int, default=5, help="Train CV folds for grid selection.")
    return p.parse_args()


if __name__ == "__main__":
    import sys as _sys

    _sys.modules.setdefault("__main__", _sys.modules[__name__])
    setattr(_sys.modules["__main__"], "LabelEncodingClassifier", LabelEncodingClassifier)

    args = _parse_args()
    run_tuning(save=args.save, skip_grid=args.skip_grid, cv=args.cv)
