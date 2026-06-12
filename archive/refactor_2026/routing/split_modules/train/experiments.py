"""Router ablations, tuning, seed sweep, LODO, and CLI entrypoints."""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import types
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight

from config.settings import router_model_path
from routing.config import (
    AGENTS,
    CVEC5_ABLATION_CASES,
    CVEC7_ABLATION_CASES,
    DEFAULT_AGENT_MODEL,
    DEFAULT_HGBM_PARAMS,
    EMBEDDING_COL_PREFIX,
    EMBEDDINGS_PARQUET,
    EVAL_SAMPLES_DIR,
    FEATURE_SET_CLI_CHOICES,
    HGBM_GRID,
    HEURISTICS_PARQUET,
    HEURISTIC_COL_PREFIX,
    PROBA_COLS,
    PRODUCTION_FEATURE_SET,
    PRODUCTION_HGBM_PARAMS,
    PRODUCTION_ROUTER_EXPERIMENT_ID,
    PRODUCTION_ROUTER_PATH,
    QCE_DATASETS,
    REPO_ROOT,
    ROUTER_EXPERIMENT_ID,
    ROUTER_MODEL_PATH,
    SEED_STABILITY_DIR,
    SPLIT_CSV,
    SPLIT_PARQUET,
    TARGET,
    TRAIN_NORM_JSON,
    TRUST_ABLATION_CASES,
    TUNED_EXPERIMENT_ID,
    TUNED_HGBM_PARAMS,
    TUNED_ROUTER_EXPERIMENT_ID,
    TUNED_ROUTER_PATH,
    AblationCase,
    FeatureSet,
    TUNE_OUT_DIR,
    experiment_id_for_feature_set,
    feature_set_needs_embeddings,
    feature_set_needs_heuristics,
    feature_set_spec,
    is_production_feature_set,
    model_path_for_feature_set,
    normalize_feature_set,
)
from routing.datasets import resolve_dataset_name
from routing.data.splits import load_split
from routing.features.columns import router_feature_cols, validate_feature_set_data
from routing.infer.predict import load_router, save_router
from routing.train.pipeline import (
    TrainSpec,
    baseline_always_majority,
    baseline_per_dataset_mode,
    compute_sample_weights,
    evaluate,
    fit_router,
    pipeline,
    oversample_minorities,
    predict_agent_proba,
    results_row,
    train_router,
)
def _grid_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]


def _tune_hgbm_cv(train_df: pd.DataFrame, feature_cols: list[str], *, cv: int = 5) -> tuple[dict[str, Any], pd.DataFrame]:
    combos = _grid_combos(HGBM_GRID)
    y = train_df[TARGET].astype(str)
    weights = compute_sample_weights(train_df, y, "balanced")
    fit_params = {"clf__sample_weight": weights} if weights is not None else {}
    best_score, best_params = -1.0, combos[0]
    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        pipe = pipeline(feature_cols, hgbm_params=params, calibrated=False)
        score = float(
            cross_val_score(
                pipe, train_df[feature_cols], y, cv=cv, scoring="f1_macro", n_jobs=-1, params=fit_params
            ).mean()
        )
        rows.append({**params, "cv_macro_f1": round(score, 4)})
        if score > best_score:
            best_score, best_params = score, params
        if i % 9 == 0:
            print(f"  grid {i}/{len(combos)}…")
    return best_params, pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)


def _cli_train(args: argparse.Namespace) -> None:
    fs: FeatureSet = getattr(args, "feature_set", PRODUCTION_FEATURE_SET)
    needs_emb = feature_set_needs_embeddings(fs)
    train_df = load_split("train", with_embeddings=needs_emb)
    val_df = load_split("val", with_embeddings=needs_emb)
    validate_feature_set_data(train_df, fs)
    fs_norm = normalize_feature_set(fs)
    spec = TrainSpec(
        experiment_id=experiment_id_for_feature_set(fs),
        feature_set=fs,
        hgbm_params=PRODUCTION_HGBM_PARAMS if is_production_feature_set(fs) else None,
    )
    pipe, feature_cols, val_res, (cv_mean, cv_std), _ = train_router(
        train_df, val_df, spec=spec, verbose=not args.quiet
    )
    if not args.quiet:
        print(f"feature_set={normalize_feature_set(fs)} n_features={len(feature_cols)}")
        print(predict_agent_proba(pipe, val_df, feature_cols).head(3).to_string())
    if args.save:
        out_path = model_path_for_feature_set(fs)
        from routing.config import LEGACY_HARD_HGBM_EXPERIMENT_ID

        exp_id = (
            LEGACY_HARD_HGBM_EXPERIMENT_ID
            if is_production_feature_set(fs)
            else experiment_id_for_feature_set(fs)
        )
        out = save_router(
            pipe,
            feature_cols=feature_cols,
            experiment_id=exp_id,
            path=out_path,
            extra_meta={
                "val_macro_f1": val_res.macro_f1,
                "train_cv_macro_f1": cv_mean,
                "feature_set": fs_norm,
                "hgbm_params": "default" if spec.hgbm_params is None else spec.hgbm_params,
                "production": is_production_feature_set(fs),
            },
        )
        print("Saved", out)


def _cli_eval(args: argparse.Namespace) -> None:
    if not args.router.is_file():
        raise FileNotFoundError(f"Router not found: {args.router}")
    train_df = load_split("train", with_embeddings=True)
    eval_df = load_split(args.split, with_embeddings=True)
    obj = load_router(args.router)
    meta_path = Path(args.router).with_suffix(".json")
    feature_set_label = obj.get("feature_set")
    if not feature_set_label and meta_path.is_file():
        feature_set_label = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_set")
    result = evaluate(obj["pipeline"], eval_df, list(obj["feature_cols"]))
    rows = [
        results_row("B0_always_raw", baseline_always_majority(train_df, eval_df)),
        results_row("B0_dataset_mode", baseline_per_dataset_mode(train_df, eval_df)),
        results_row(
            str(obj.get("experiment_id", args.router.stem)),
            result,
            feature_set=str(feature_set_label or PRODUCTION_FEATURE_SET),
            n_features=len(obj["feature_cols"]),
            model="hgbm",
        ),
    ]
    table = pd.DataFrame(rows)
    print(f"\n=== Router eval on {args.split!r} (n={len(eval_df)}) ===\n{table.to_string(index=False)}")
    b0 = table[table["experiment"] == "B0_dataset_mode"].iloc[0]
    ml = table.iloc[-1]
    mf = float(ml["macro_f1"])
    print(f"\n{mf:.4f} macro-F1 ({mf - float(b0['macro_f1']):+.4f} vs dataset_mode)\n{result.report}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")


def run_feature_ablation(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Legacy entrypoint — runs the 5-dim (cvec5) ablation table."""
    return run_ablation_cvec5(eval_split=eval_split, out_dir=out_dir, verbose=verbose)


def _run_ablation_table(
    cases: tuple[AblationCase, ...],
    *,
    track: str,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    out_dir = Path(out_dir or REPO_ROOT / f"results/experiments/feature_ablation_{track}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_split("train", with_embeddings=True)
    val_df = load_split("val", with_embeddings=True)
    eval_df = load_split(eval_split, with_embeddings=True)

    rows: list[dict[str, Any]] = []
    for case in cases:
        fs = normalize_feature_set(case.feature_set)
        exp_id = f"hgbm_{track}_{case.label.replace('+', '_').replace(' ', '_').replace('(', '').replace(')', '')}"
        spec = TrainSpec(experiment_id=exp_id, feature_set=case.feature_set)
        if verbose:
            print(f"\n=== [{track}] {case.label} ({fs}) ===")
            if case.description:
                print(f"    {case.description}")
        pipe, fcols, val_res, (cv_mean, cv_std), _ = train_router(
            train_df, val_df, spec=spec, verbose=verbose
        )
        test_res = evaluate(pipe, eval_df, fcols)
        rows.append(
            {
                "track": track,
                "label": case.label,
                "feature_set": fs,
                "description": case.description,
                "n_features": len(fcols),
                "feature_cols": ",".join(fcols),
                "val_macro_f1": round(val_res.macro_f1, 4),
                "val_accuracy": round(val_res.accuracy, 4),
                f"{eval_split}_macro_f1": round(test_res.macro_f1, 4),
                f"{eval_split}_accuracy": round(test_res.accuracy, 4),
                "train_cv_macro_f1": round(cv_mean, 4),
                "train_cv_std": round(cv_std, 4),
            }
        )
    table = pd.DataFrame(rows)
    path = out_dir / f"ablation_{eval_split}.csv"
    table.to_csv(path, index=False)
    if verbose:
        show = table.drop(columns=["feature_cols", "description"], errors="ignore")
        print(f"\n=== Ablation [{track}] ({eval_split}) ===\n{show.to_string(index=False)}")
        print(f"Wrote {path}")
    return table


def run_ablation_cvec5(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    return _run_ablation_table(
        CVEC5_ABLATION_CASES,
        track="cvec5",
        eval_split=eval_split,
        out_dir=out_dir or REPO_ROOT / "results/experiments/feature_ablation_cvec5",
        verbose=verbose,
    )


def run_ablation_cvec7(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    return _run_ablation_table(
        CVEC7_ABLATION_CASES,
        track="cvec7",
        eval_split=eval_split,
        out_dir=out_dir or REPO_ROOT / "results/experiments/feature_ablation_cvec7",
        verbose=verbose,
    )


def run_ablation_trust(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    return _run_ablation_table(
        TRUST_ABLATION_CASES,
        track="trust",
        eval_split=eval_split,
        out_dir=out_dir or REPO_ROOT / "results/experiments/feature_ablation_trust",
        verbose=verbose,
    )


def run_ablation_all(
    *,
    eval_split: str = "test",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    return {
        "cvec5": run_ablation_cvec5(eval_split=eval_split, verbose=verbose),
        "cvec7": run_ablation_cvec7(eval_split=eval_split, verbose=verbose),
        "trust": run_ablation_trust(eval_split=eval_split, verbose=verbose),
    }


def _fit_eval_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    feature_set: FeatureSet,
    hgbm_params: dict[str, Any] | None,
    random_state: int,
) -> tuple[EvalResult, list[str]]:
    """Train on train split only; score ``eval_df`` (val or test)."""
    spec = TrainSpec(
        experiment_id="seed_stability",
        feature_set=feature_set,
        hgbm_params=hgbm_params,
        random_state=random_state,
    )
    feature_cols = router_feature_cols(train_df, spec.feature_set)
    fit_df = oversample_minorities(train_df, spec.target) if spec.oversample else train_df
    y_fit = fit_df[spec.target].astype(str)
    weights = compute_sample_weights(fit_df, y_fit, spec.weight_mode)
    pipe = pipeline(
        feature_cols,
        class_weight=spec.effective_class_weight,
        hgbm_params=spec.hgbm_params,
        random_state=spec.random_state,
    )
    fit_router(pipe, fit_df, feature_cols, target=spec.target, sample_weight=weights)
    return evaluate(pipe, eval_df, feature_cols, target=spec.target), feature_cols


def run_seed_stability(
    *,
    feature_set: FeatureSet = PRODUCTION_FEATURE_SET,
    eval_split: str = "test",
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2024),
    out_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare default vs tuned HGBM on ``feature_set`` across random seeds.

    Trains on the train split only; reports macro-F1 on ``eval_split`` (default test).
    """
    out_dir = Path(out_dir or SEED_STABILITY_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_split("train", with_embeddings=True)
    eval_df = load_split(eval_split, with_embeddings=True)
    feature_cols = router_feature_cols(train_df, feature_set)

    configs: tuple[tuple[str, dict[str, Any] | None], ...] = (
        ("default", DEFAULT_HGBM_PARAMS),
        ("tuned", TUNED_HGBM_PARAMS),
    )

    rows: list[dict[str, Any]] = []
    for config_name, hgbm_params in configs:
        for seed in seeds:
            result, _ = _fit_eval_split(
                train_df,
                eval_df,
                feature_set=feature_set,
                hgbm_params=hgbm_params,
                random_state=seed,
            )
            rows.append(
                {
                    "config": config_name,
                    "seed": seed,
                    "feature_set": normalize_feature_set(feature_set),
                    "n_features": len(feature_cols),
                    f"{eval_split}_macro_f1": round(result.macro_f1, 4),
                    f"{eval_split}_accuracy": round(result.accuracy, 4),
                }
            )
            if verbose:
                print(
                    f"  {config_name} seed={seed} "
                    f"{eval_split} macro-F1={result.macro_f1:.4f} acc={result.accuracy:.4f}"
                )

    per_seed = pd.DataFrame(rows)
    per_seed_path = out_dir / f"per_seed_{eval_split}.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for config_name, _ in configs:
        sub = per_seed[per_seed["config"] == config_name]
        f1_col = f"{eval_split}_macro_f1"
        acc_col = f"{eval_split}_accuracy"
        summary_rows.append(
            {
                "config": config_name,
                "n_seeds": len(sub),
                f"{eval_split}_macro_f1_mean": round(float(sub[f1_col].mean()), 4),
                f"{eval_split}_macro_f1_std": round(float(sub[f1_col].std(ddof=0)), 4),
                f"{eval_split}_accuracy_mean": round(float(sub[acc_col].mean()), 4),
                f"{eval_split}_accuracy_std": round(float(sub[acc_col].std(ddof=0)), 4),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"summary_{eval_split}.csv"
    summary.to_csv(summary_path, index=False)

    if verbose:
        print(f"\n=== Seed stability ({feature_set}, {eval_split}, n={len(seeds)}) ===")
        for _, row in summary.iterrows():
            print(
                f"  {row['config']:8s}  "
                f"macro-F1 {row[f'{eval_split}_macro_f1_mean']:.3f} "
                f"± {row[f'{eval_split}_macro_f1_std']:.3f}  "
                f"(acc {row[f'{eval_split}_accuracy_mean']:.3f} "
                f"± {row[f'{eval_split}_accuracy_std']:.3f})"
            )
        d = summary[summary["config"] == "default"][f"{eval_split}_macro_f1_mean"].iloc[0]
        t = summary[summary["config"] == "tuned"][f"{eval_split}_macro_f1_mean"].iloc[0]
        dd = summary[summary["config"] == "default"][f"{eval_split}_macro_f1_std"].iloc[0]
        td = summary[summary["config"] == "tuned"][f"{eval_split}_macro_f1_std"].iloc[0]
        print(
            f"\n  Δ tuned−default (mean): {t - d:+.4f}  "
            f"(overlap if |Δ| < {dd + td:.4f} ≈ combined spread)"
        )
        print(f"Wrote {per_seed_path}\nWrote {summary_path}")

    return per_seed, summary


def run_leave_one_dataset_out(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    include_dataset_feature: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Hold out one benchmark dataset: train on the other datasets, eval on held-out rows.

    Default ``cvec5_emb`` (no dataset one-hot) tests that C(Q)+emb generalize
    beyond benchmark identity.
    """
    out_dir = Path(out_dir or REPO_ROOT / "results/experiments/lodo")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_all = load_split("train", with_embeddings=True)
    val_all = load_split("val", with_embeddings=True)
    eval_all = load_split(eval_split, with_embeddings=True)
    fs: FeatureSet = "cvec5_emb_ds" if include_dataset_feature else "cvec5_emb"

    rows: list[dict[str, Any]] = []
    for held in QCE_DATASETS:
        train_df = train_all[train_all["dataset"] != held].copy()
        val_df = val_all[val_all["dataset"] != held].copy()
        eval_df = eval_all[eval_all["dataset"] == held].copy()
        if eval_df.empty:
            continue
        spec = TrainSpec(
            experiment_id=f"lodo_hold_{held}_{'ds' if include_dataset_feature else 'nods'}",
            feature_set=fs,
        )
        if verbose:
            print(
                f"\n=== LODO hold-out={held} train_n={len(train_df)} "
                f"eval_n={len(eval_df)} feature_set={fs} ==="
            )
        pipe, fcols, val_res, (cv_mean, _), _ = train_router(
            train_df, val_df, spec=spec, verbose=verbose
        )
        test_res = evaluate(pipe, eval_df, fcols)
        rows.append(
            {
                "held_out_dataset": held,
                "feature_set": fs,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_eval": len(eval_df),
                "val_macro_f1": round(val_res.macro_f1, 4),
                f"{eval_split}_macro_f1": round(test_res.macro_f1, 4),
                f"{eval_split}_accuracy": round(test_res.accuracy, 4),
                "train_cv_macro_f1": round(cv_mean, 4),
            }
        )

    table = pd.DataFrame(rows)
    tag = "with_dataset" if include_dataset_feature else "no_dataset"
    path = out_dir / f"lodo_{tag}_{eval_split}.csv"
    table.to_csv(path, index=False)
    if verbose and len(table):
        print(f"\n=== Leave-one-dataset-out ({eval_split}, {tag}) ===\n{table.to_string(index=False)}")
        print(
            f"Mean {eval_split} macro-F1: {table[f'{eval_split}_macro_f1'].mean():.4f}\n"
            f"Wrote {path}"
        )
    return table

def _cli_tune(args: argparse.Namespace) -> None:
    from routing.analysis import (
        build_analysis_frame,
        plot_reliability_calibration,
        router_confidence_sweep,
        top2_sweep,
    )

    fs: FeatureSet = getattr(args, "feature_set", PRODUCTION_FEATURE_SET)
    needs_emb = feature_set_needs_embeddings(fs)
    train_df = load_split("train", with_embeddings=needs_emb)
    val_df = load_split("val", with_embeddings=needs_emb)
    test_df = load_split("test", with_embeddings=needs_emb)
    validate_feature_set_data(train_df, fs)
    fs_norm = normalize_feature_set(fs)
    feature_cols = router_feature_cols(train_df, fs)
    TUNE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_grid:
        best_params = {"max_depth": 6, "learning_rate": 0.05, "max_leaf_nodes": 31}
        grid_df = pd.DataFrame()
    else:
        print(f"=== HGBM grid ({len(_grid_combos(HGBM_GRID))} combos) ===")
        best_params, grid_df = _tune_hgbm_cv(train_df, feature_cols, cv=args.cv)
        if len(grid_df):
            grid_df.to_csv(TUNE_OUT_DIR / "hgbm_grid_cv.csv", index=False)

    tuned_exp = (
        TUNED_ROUTER_EXPERIMENT_ID
        if is_production_feature_set(fs)
        else f"{TUNED_EXPERIMENT_ID}_{fs_norm}"
    )
    spec = TrainSpec(experiment_id=tuned_exp, feature_set=fs, hgbm_params=best_params)
    pipe, fcols, val_res, (cv_mean, _), _ = train_router(train_df, val_df, spec=spec)
    summary = {
        "experiment_id": tuned_exp,
        "feature_set": fs_norm,
        "comparison_only": is_production_feature_set(fs),
        "hgbm_best_params": best_params,
        "val": {"macro_f1": val_res.macro_f1, "accuracy": val_res.accuracy},
        "test": {
            "macro_f1": evaluate(pipe, test_df, fcols).macro_f1,
            "accuracy": evaluate(pipe, test_df, fcols).accuracy,
        },
        "train_cv_macro_f1": cv_mean,
    }
    if args.save:
        tuned_path = (
            TUNED_ROUTER_PATH
            if is_production_feature_set(fs)
            else model_path_for_feature_set(fs).parent / f"{tuned_exp}.joblib"
        )
        save_router(
            pipe,
            feature_cols=fcols,
            experiment_id=tuned_exp,
            path=tuned_path,
            extra_meta={
                "feature_set": fs_norm,
                "hgbm_params": best_params,
                "val_macro_f1": val_res.macro_f1,
                "train_cv_macro_f1": cv_mean,
                "comparison_only": is_production_feature_set(fs),
            },
        )
        print("Saved", tuned_path)
    if args.calibrated:
        cal_spec = TrainSpec(
            experiment_id=f"{tuned_exp}_calibrated",
            feature_set=fs,
            hgbm_params=best_params,
            calibrated=True,
        )
        cal_pipe, cal_fcols, _, _, _ = train_router(train_df, val_df, spec=cal_spec)
        if args.save:
            cal_path = model_path_for_feature_set(fs).parent / f"{tuned_exp}_calibrated.joblib"
            save_router(
                cal_pipe,
                feature_cols=cal_fcols,
                experiment_id=f"{tuned_exp}_calibrated",
                path=cal_path,
            )

    analysis = build_analysis_frame("val", pipe=pipe, feature_cols=fcols, experiment_id=tuned_exp)
    top2_sweep(analysis).to_csv(TUNE_OUT_DIR / f"top2_sweep_val_{fs}.csv", index=False)
    router_confidence_sweep(analysis).to_csv(TUNE_OUT_DIR / f"confidence_sweep_val_{fs}.csv", index=False)
    plot_reliability_calibration(analysis, TUNE_OUT_DIR / f"calibration_val_{fs}.png", title_suffix=f" ({tuned_exp}, val)")
    (TUNE_OUT_DIR / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QCE router (HGBM)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fs_help = (
        "Feature columns for the router. Prefer cvec5_* / cvec7_* names; "
        "graph_* names are legacy aliases."
    )

    p_train = sub.add_parser("train", help="Train on QCE train/val")
    p_train.add_argument("--save", action="store_true")
    p_train.add_argument("--quiet", action="store_true")
    p_train.add_argument(
        "--feature-set",
        dest="feature_set",
        choices=FEATURE_SET_CLI_CHOICES,
        default=PRODUCTION_FEATURE_SET,
        help=fs_help,
    )

    p_eval = sub.add_parser("eval", help="Evaluate on QCE val/test vs baselines")
    p_eval.add_argument("--split", choices=("val", "test"), default="test")
    p_eval.add_argument("--router", type=Path, default=ROUTER_MODEL_PATH)
    p_eval.add_argument("--json-out", type=Path, default=None)

    p_tune = sub.add_parser(
        "tune",
        help="HGBM grid search (saves comparison model; production uses default HGBM)",
    )
    p_tune.add_argument("--save", action="store_true")
    p_tune.add_argument("--skip-grid", action="store_true")
    p_tune.add_argument("--cv", type=int, default=5)
    p_tune.add_argument("--calibrated", action="store_true")
    p_tune.add_argument(
        "--feature-set",
        dest="feature_set",
        choices=FEATURE_SET_CLI_CHOICES,
        default=PRODUCTION_FEATURE_SET,
        help=f"Feature set to tune (comparison; production={PRODUCTION_FEATURE_SET}).",
    )

    p_abl = sub.add_parser("ablation", help="Feature ablation tables (cvec5 / cvec7 / trust)")
    p_abl.add_argument("--split", choices=("val", "test"), default="test")
    p_abl.add_argument(
        "--track",
        choices=("cvec5", "cvec7", "trust", "all"),
        default="all",
        help="Which ablation table to run (default: all three)",
    )
    p_abl.add_argument("--out-dir", type=Path, default=None)
    p_abl.add_argument("--quiet", action="store_true")

    p_lodo = sub.add_parser("lodo", help="Leave-one-dataset-out routing eval")
    p_lodo.add_argument("--split", choices=("val", "test"), default="test")
    p_lodo.add_argument("--out-dir", type=Path, default=None)
    p_lodo.add_argument(
        "--with-dataset",
        action="store_true",
        help="Include dataset one-hot (default: cvec5+emb only, no dataset)",
    )
    p_lodo.add_argument("--quiet", action="store_true")

    p_seed = sub.add_parser(
        "seed-sweep",
        help="Default vs tuned HGBM across random seeds (freeze check)",
    )
    p_seed.add_argument("--split", choices=("val", "test"), default="test")
    p_seed.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2024],
        help="Random seeds (default: 5 seeds)",
    )
    p_seed.add_argument("--out-dir", type=Path, default=None)
    p_seed.add_argument(
        "--feature-set",
        dest="feature_set",
        default=PRODUCTION_FEATURE_SET,
        choices=FEATURE_SET_CLI_CHOICES,
    )
    p_seed.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "train":
        _cli_train(args)
    elif args.cmd == "eval":
        _cli_eval(args)
    elif args.cmd == "tune":
        _cli_tune(args)
    elif args.cmd == "ablation":
        if args.track == "cvec5":
            run_ablation_cvec5(
                eval_split=args.split,
                out_dir=args.out_dir,
                verbose=not args.quiet,
            )
        elif args.track == "cvec7":
            run_ablation_cvec7(
                eval_split=args.split,
                out_dir=args.out_dir,
                verbose=not args.quiet,
            )
        elif args.track == "trust":
            run_ablation_trust(
                eval_split=args.split,
                out_dir=args.out_dir,
                verbose=not args.quiet,
            )
        else:
            run_ablation_all(eval_split=args.split, verbose=not args.quiet)
    elif args.cmd == "lodo":
        run_leave_one_dataset_out(
            eval_split=args.split,
            out_dir=args.out_dir,
            include_dataset_feature=args.with_dataset,
            verbose=not args.quiet,
        )
    elif args.cmd == "seed-sweep":
        run_seed_stability(
            feature_set=args.feature_set,
            eval_split=args.split,
            seeds=tuple(args.seeds),
            out_dir=args.out_dir,
            verbose=not args.quiet,
        )

