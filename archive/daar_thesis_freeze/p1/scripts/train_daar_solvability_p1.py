"""Gate 1 P1 — utility-softmax router on daar_train (ranking-aligned supervision)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score

from config.global_config.paths import daar_models_dir, ensure_daar_dirs
from config.local.constants.agents import ROUTER_AGENTS
from config.local.router.production import PRODUCTION_FEATURE_SET
from daar.gate1 import (
    P1_MANIFEST,
    P1_MODEL_STEM,
    attach_soft_labels,
    eval_abstention,
    load_frame,
    solvable_series,
    tune_theta,
)
from research.soft_train import fit_soft_kl_router, make_soft_kl_pipeline, soft_train_mask
from router.router import PROBA_COLS, predict_agent_proba, router_feature_cols, save_router
from src.utils.soft_labels import DEFAULT_UTILITY_LAMBDA, SOFT_DOMINANT_COL


def query_score_table(
    query_df: pd.DataFrame,
    pipe: Any,
    feature_cols: list[str],
    frame_long: pd.DataFrame,
) -> pd.DataFrame:
    proba = predict_agent_proba(pipe, query_df, feature_cols)
    wide = proba.copy()
    wide["training_id"] = query_df["training_id"].astype(str).values
    wide = wide[["training_id", *PROBA_COLS]]
    wide.columns = ["training_id", *[f"s_hat_{a}" for a in ROUTER_AGENTS]]
    wide["s_hat_max"] = wide[[f"s_hat_{a}" for a in ROUTER_AGENTS]].max(axis=1)
    wide = wide.merge(solvable_series(frame_long).reset_index(), on="training_id", how="left")
    return wide


def train_daar_solvability_p1(
    *,
    feature_set: str = PRODUCTION_FEATURE_SET,
    random_state: int = 42,
    utility_lambda: float = DEFAULT_UTILITY_LAMBDA,
    softmax_temperature: float = 1.0,
    models_dir: Path | None = None,
    frames_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    train_long = load_frame("train", frames_dir=frames_dir)
    val_long = load_frame("val", frames_dir=frames_dir)
    train_df = attach_soft_labels(
        train_long,
        utility_lambda=utility_lambda,
        softmax_temperature=softmax_temperature,
    )
    n_queries_expected = train_long["training_id"].nunique()
    assert len(train_df) == n_queries_expected, (
        f"expected query-level rows after attach_soft_labels: {n_queries_expected}, "
        f"got {len(train_df)}"
    )
    feature_cols = router_feature_cols(train_df, feature_set)

    pipe = make_soft_kl_pipeline(feature_cols, "hgbm", random_state=random_state)
    pipe, n_queries, n_expanded = fit_soft_kl_router(pipe, train_df, feature_cols)

    model_path = models_dir / f"{P1_MODEL_STEM}.joblib"
    save_router(
        pipe,
        feature_cols=feature_cols,
        experiment_id=P1_MODEL_STEM,
        path=model_path,
        extra_meta={
            "feature_set": feature_set,
            "gate1_variant": "p1",
            "gate1_supervision": "utility_softmax",
            "utility_lambda": utility_lambda,
            "softmax_temperature": softmax_temperature,
        },
    )

    val_query = attach_soft_labels(
        val_long,
        utility_lambda=utility_lambda,
        softmax_temperature=softmax_temperature,
    )
    val_scores = query_score_table(val_query, pipe, feature_cols, val_long)
    theta_s, theta_sweep = tune_theta(val_scores["s_hat_max"], val_scores["solvable"])

    val_hard = val_query.loc[soft_train_mask(val_query)].copy()
    if not val_hard.empty:
        pred = predict_agent_proba(pipe, val_hard, feature_cols).idxmax(axis=1).str.removeprefix("p_")
        hard_acc = float(
            accuracy_score(val_hard[SOFT_DOMINANT_COL].astype(str), pred, normalize=True)
        )
    else:
        hard_acc = None

    manifest: dict[str, Any] = {
        "variant": "p1",
        "gate1_supervision": "utility_softmax",
        "utility_lambda": utility_lambda,
        "softmax_temperature": softmax_temperature,
        "feature_set": feature_set,
        "feature_cols": feature_cols,
        "agents": list(ROUTER_AGENTS),
        "theta_s": theta_s,
        "n_train_queries": n_queries,
        "n_train_expanded_rows": n_expanded,
        "val_hard_dominant_accuracy": hard_acc,
        "val_abstention": eval_abstention(val_scores["s_hat_max"], val_scores["solvable"], theta_s),
        "models_dir": str(models_dir),
        "model_path": str(model_path),
    }

    manifest_path = models_dir / P1_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    theta_sweep.to_csv(models_dir / "theta_s_sweep_p1.csv", index=False)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-set", default=PRODUCTION_FEATURE_SET)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--utility-lambda", type=float, default=DEFAULT_UTILITY_LAMBDA)
    p.add_argument("--softmax-temperature", type=float, default=1.0)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--frames-dir", type=Path, default=None)
    args = p.parse_args()
    print(
        json.dumps(
            train_daar_solvability_p1(
                feature_set=args.feature_set,
                random_state=args.random_state,
                utility_lambda=args.utility_lambda,
                softmax_temperature=args.softmax_temperature,
                models_dir=args.models_dir,
                frames_dir=args.frames_dir,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
