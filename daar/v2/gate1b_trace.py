"""Trajectory-aware Gate 1b for D-AAR 2.0."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config.global_config.paths import daar_models_dir, daar_trace_skeleton_path, ensure_daar_dirs
from daar.gate1 import attach_success_only_soft_labels, load_frame, solvable_series
from research.soft_train import soft_train_mask
from daar.v2.features import attach_query_trace_features, gate1b_trace_feature_cols
from daar.v2.constants import GATE1B_TRACE_MANIFEST, GATE1B_TRACE_STEM
from research.soft_train import fit_soft_kl_router, make_soft_kl_pipeline
from router.router import predict_agent_proba, save_router
from src.utils.soft_labels import SOFT_DOMINANT_COL


def train_gate1b_trace(
    *,
    random_state: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()
    traces = pd.read_parquet(daar_trace_skeleton_path())

    train_long = load_frame("train")
    val_long = load_frame("val")
    train_df = attach_query_trace_features(
        attach_success_only_soft_labels(train_long), traces
    )
    val_df = attach_query_trace_features(
        attach_success_only_soft_labels(val_long), traces
    )

    feature_cols = gate1b_trace_feature_cols(train_df)
    train_mask = soft_train_mask(train_df)

    pipe = make_soft_kl_pipeline(feature_cols, "hgbm", random_state=random_state)
    pipe, _n_queries, _ = fit_soft_kl_router(pipe, train_df, feature_cols)

    model_path = models_dir / f"{GATE1B_TRACE_STEM}.joblib"
    save_router(
        pipe,
        feature_cols=feature_cols,
        experiment_id=GATE1B_TRACE_STEM,
        path=model_path,
        extra_meta={
            "feature_set": "cvec5_trace",
            "gate": "1b",
            "gate1_supervision": "success_only_soft_kl",
            "daar_version": "2.0",
        },
    )

    solvable_ids = set(
        solvable_series(val_long)[solvable_series(val_long) == 1].index.astype(str)
    )
    val_sol = val_df[val_df["training_id"].astype(str).isin(solvable_ids)].copy()
    dominant_acc = 0.0
    if len(val_sol):
        pred = (
            predict_agent_proba(pipe, val_sol, feature_cols)
            .idxmax(axis=1)
            .str.removeprefix("p_")
        )
        dominant_acc = float(
            np.mean(pred.values == val_sol[SOFT_DOMINANT_COL].astype(str).values)
        )

    manifest: dict[str, Any] = {
        "gate": "1b",
        "variant": "gate1b_trace",
        "daar_version": "2.0",
        "feature_set": "cvec5_trace",
        "feature_cols": feature_cols,
        "model_path": str(model_path),
        "n_train_solvable": int(train_mask.sum()),
        "val_dominant_accuracy_solvable": dominant_acc,
    }
    manifest_path = models_dir / GATE1B_TRACE_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest
