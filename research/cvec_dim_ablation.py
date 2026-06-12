"""
Ablation over QCE five-dimensional complexity vector (with PCA-16 embeddings).

Modes:
  - single: each dim + emb (one complexity signal at a time)
  - loo:    leave-one-out (four dims + emb)
  - full:   all five dims + emb (same as cvec5_emb production)

Reports test utility regret (agent + total) and executed EM under cost-soft HGBM.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from config.global_config.paths import router_production_dir
from config.local.constants import DIMS
from research.baselines import eval_pipeline_on_split
from research.soft_train import (
    _train_frame_with_soft_targets,
    fit_soft_kl_router,
    make_soft_kl_pipeline,
)
from router.config import PRODUCTION_FEATURE_SET, PRODUCTION_HGBM_PARAMS, REPO_ROOT
from router.router import (
    TrainSpec,
    evaluate,
    load_split_for_feature_set,
    router_feature_cols,
    save_router,
    validate_feature_set_data,
)

OUT_ROOT = REPO_ROOT / "results/experiments/cvec5_dim_ablation"
CVEC5_DIMS: tuple[str, ...] = DIMS.main

DIM_LABELS: dict[str, str] = {
    "dim_structural": "z1_structural",
    "dim_reasoning": "z2_reasoning",
    "dim_evidence": "z3_evidence",
    "dim_tool": "z4_tool",
    "dim_coordination_uncertainty": "z5_coordination",
}


@dataclass(frozen=True)
class DimAblationCase:
    case_id: str
    dim_cols: tuple[str, ...]
    description: str
    mode: Literal["full", "single", "loo"]


def _emb_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in router_feature_cols(df, PRODUCTION_FEATURE_SET) if c.startswith("emb_")]


def ablation_cases(*, modes: tuple[str, ...] = ("full", "single", "loo")) -> tuple[DimAblationCase, ...]:
    cases: list[DimAblationCase] = []
    if "full" in modes:
        cases.append(
            DimAblationCase(
                "full",
                CVEC5_DIMS,
                "all five complexity dims + emb (cvec5_emb)",
                "full",
            )
        )
    if "single" in modes:
        for d in CVEC5_DIMS:
            cases.append(
                DimAblationCase(
                    f"only_{DIM_LABELS[d]}",
                    (d,),
                    f"only {DIM_LABELS[d]} + emb",
                    "single",
                )
            )
    if "loo" in modes:
        for d in CVEC5_DIMS:
            rest = tuple(x for x in CVEC5_DIMS if x != d)
            cases.append(
                DimAblationCase(
                    f"loo_{DIM_LABELS[d]}",
                    rest,
                    f"all dims except {DIM_LABELS[d]} + emb",
                    "loo",
                )
            )
    return tuple(cases)


def train_case(
    case: DimAblationCase,
    *,
    seed: int = 42,
    save: bool = False,
    skip_train: bool = False,
) -> tuple[Any, list[str], dict[str, Any]]:
    """Train cost-soft HGBM for one dim subset (+ embeddings)."""
    fs = PRODUCTION_FEATURE_SET
    train_df = _train_frame_with_soft_targets("train", fs)
    validate_feature_set_data(train_df, fs)
    feature_cols = list(case.dim_cols) + _emb_cols(train_df)
    experiment_id = f"hgbm_dimab_{case.case_id}_soft_kl"

    if skip_train and case.mode == "full":
        from research.soft_train import soft_kl_model_path
        from router.router import load_router

        path = soft_kl_model_path("hgbm", fs)
        obj = load_router(path)
        return obj["pipeline"], list(obj["feature_cols"]), {"experiment_id": experiment_id, "reused": True}

    spec = TrainSpec(
        experiment_id=experiment_id,
        feature_set=fs,
        hgbm_params=PRODUCTION_HGBM_PARAMS,
        weight_mode="none",
        use_class_weight=False,
        random_state=seed,
    )
    pipe = make_soft_kl_pipeline(feature_cols, "hgbm", random_state=seed, hgbm_params=spec.hgbm_params)
    pipe, n_queries, n_expanded = fit_soft_kl_router(pipe, train_df, feature_cols)
    val_df = load_split_for_feature_set("val", fs)
    val_label = evaluate(pipe, val_df, feature_cols)
    meta: dict[str, Any] = {
        "experiment_id": experiment_id,
        "case_id": case.case_id,
        "mode": case.mode,
        "description": case.description,
        "dim_cols": list(case.dim_cols),
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "n_train_queries": n_queries,
        "n_train_expanded_rows": n_expanded,
        "val_macro_f1_hard_label": val_label.macro_f1,
        "val_accuracy_hard_label": val_label.accuracy,
        "soft_target": "cost_soft",
    }
    if save:
        path = router_production_dir() / f"{experiment_id}.joblib"
        save_router(
            pipe,
            feature_cols=feature_cols,
            experiment_id=experiment_id,
            path=path,
            extra_meta={"feature_set": fs, "dim_ablation": case.case_id},
        )
        meta["saved"] = str(path.relative_to(REPO_ROOT))
    return pipe, feature_cols, meta


def run_dim_ablation(
    *,
    eval_split: str = "test",
    modes: tuple[str, ...] = ("full", "single", "loo"),
    seed: int = 42,
    save: bool = False,
    reuse_full: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for case in ablation_cases(modes=modes):
        if verbose:
            print(f"\n=== {case.case_id}: {case.description} ===")
        skip = reuse_full and case.mode == "full"
        pipe, feature_cols, meta = train_case(
            case, seed=seed, save=save, skip_train=skip
        )
        if skip and verbose:
            print("  (reused production cvec5_emb checkpoint for full)")
        test_row = eval_pipeline_on_split(
            eval_split,
            pipe,
            feature_cols,
            classifier="hgbm",
            hp_policy="soft_kl",
            experiment_id=meta["experiment_id"],
            feature_set=PRODUCTION_FEATURE_SET,
        )
        row = {
            "case_id": case.case_id,
            "mode": case.mode,
            "description": case.description,
            "n_dim": len(case.dim_cols),
            "dim_cols": ",".join(case.dim_cols),
            "n_features": len(feature_cols),
            f"{eval_split}_exec_em": test_row["exec_em"],
            f"{eval_split}_mean_utility_regret": test_row["mean_utility_regret"],
            f"{eval_split}_mean_utility_regret_total": test_row["mean_utility_regret_total"],
            "val_macro_f1_hard": meta.get("val_macro_f1_hard_label"),
        }
        rows.append(row)
        if verbose:
            print(
                f"  test regret agent={row[f'{eval_split}_mean_utility_regret']:.4f} "
                f"total={row[f'{eval_split}_mean_utility_regret_total']:.4f} "
                f"EM={row[f'{eval_split}_exec_em']:.2%}"
            )

    table = pd.DataFrame(rows)
    out_path = OUT_ROOT / f"dim_ablation_{eval_split}.csv"
    table.to_csv(out_path, index=False)
    summary = {
        "eval_split": eval_split,
        "modes": list(modes),
        "seed": seed,
        "rows": rows,
    }
    (OUT_ROOT / f"dim_ablation_{eval_split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    if verbose:
        show = table.sort_values(f"{eval_split}_mean_utility_regret_total")
        print(f"\n=== Ranked by total regret ({eval_split}) ===\n{show.to_string(index=False)}")
        print(f"\nWrote {out_path}")
    return table


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="QCE 5-dim complexity ablation (+ emb, cost-soft HGBM).")
    p.add_argument("--split", default="test", choices=("val", "test", "both"))
    p.add_argument(
        "--modes",
        nargs="+",
        default=["full", "single", "loo"],
        choices=("full", "single", "loo"),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save", action="store_true", help="Write joblib under models/router/graph_main/")
    p.add_argument(
        "--reuse-full",
        action="store_true",
        help="Load production cvec5_emb joblib for full row (may be stale; default retrains).",
    )
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    splits = ("val", "test") if args.split == "both" else (args.split,)
    for sp in splits:
        run_dim_ablation(
            eval_split=sp,
            modes=tuple(args.modes),
            seed=args.seed,
            save=args.save,
            reuse_full=args.reuse_full,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
