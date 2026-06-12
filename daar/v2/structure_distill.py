"""Distill QCE teacher φ(q) from query embeddings; hybrid confidence gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from config.global_config.paths import daar_models_dir, ensure_daar_dirs
from daar.gate1 import load_frame, query_level_frame
from daar.v2.features import CVEC_COLS
from router.router import embedding_feature_cols

STRUCTURE_MODEL_STEM = "structure_distill"
STRUCTURE_MANIFEST = "structure_distill_manifest.json"


def _clip_phi(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr, 0.0, 1.0)


def train_structure_distill(
    *,
    random_state: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Train emb(q) → φ(q) on train; calibrate confidence on val."""
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    train_q = query_level_frame(load_frame("train"))
    val_q = query_level_frame(load_frame("val"))
    emb_cols = embedding_feature_cols(train_q)
    for col in CVEC_COLS:
        if col not in train_q.columns:
            raise ValueError(f"missing teacher column {col!r}")

    x_train = train_q[emb_cols].astype(float).to_numpy()
    y_train = train_q[list(CVEC_COLS)].astype(float).to_numpy()
    x_val = val_q[emb_cols].astype(float).to_numpy()
    y_val = val_q[list(CVEC_COLS)].astype(float).to_numpy()

    base = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.08,
        max_iter=200,
        random_state=random_state,
    )
    pipe = MultiOutputRegressor(base)
    pipe.fit(x_train, y_train)

    y_hat_val = _clip_phi(pipe.predict(x_val))
    per_dim_mae = np.mean(np.abs(y_hat_val - y_val), axis=0)
    per_query_mae = np.mean(np.abs(y_hat_val - y_val), axis=1)
    mae_p90 = float(np.percentile(per_query_mae, 90))
    if mae_p90 <= 0:
        mae_p90 = 1.0

    model_path = models_dir / f"{STRUCTURE_MODEL_STEM}.joblib"
    joblib.dump(
        {
            "pipeline": pipe,
            "emb_cols": emb_cols,
            "target_cols": list(CVEC_COLS),
            "mae_p90": mae_p90,
        },
        model_path,
    )

    manifest: dict[str, Any] = {
        "model": STRUCTURE_MODEL_STEM,
        "teacher": "QCE cvec5 from daar frames",
        "input": "emb_0..emb_15",
        "output": list(CVEC_COLS),
        "val_mean_mae": float(np.mean(per_query_mae)),
        "val_per_dim_mae": {c: float(v) for c, v in zip(CVEC_COLS, per_dim_mae, strict=True)},
        "mae_p90": mae_p90,
        "confidence_formula": "conf = max(0, 1 - query_mae / mae_p90)",
        "default_confidence_threshold": 0.35,
        "model_path": str(model_path),
    }
    manifest_path = models_dir / STRUCTURE_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_structure_distill(
    models_dir: Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    models_dir = models_dir or daar_models_dir()
    manifest_path = models_dir / STRUCTURE_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path} — run: uv run python scripts/train_daar2_structure_distill.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    obj = joblib.load(manifest.get("model_path", models_dir / f"{STRUCTURE_MODEL_STEM}.joblib"))
    return obj, manifest


def predict_phi_hat(
    query: pd.DataFrame,
    obj: dict[str, Any],
) -> pd.DataFrame:
    """Return query frame with ``phi_hat_*`` and ``structure_confidence``."""
    out = query.copy()
    emb_cols = list(obj["emb_cols"])
    target_cols = list(obj["target_cols"])
    x = out[emb_cols].astype(float).to_numpy()
    y_hat = _clip_phi(obj["pipeline"].predict(x))
    mae_p90 = float(obj["mae_p90"])
    for i, col in enumerate(target_cols):
        out[f"phi_hat_{col}"] = y_hat[:, i]
    # confidence vs teacher scale (deploy: vs mae_p90 only using 0 error prior — use kNN later)
    if all(c in out.columns for c in target_cols):
        teacher = out[target_cols].astype(float).to_numpy()
        query_mae = np.mean(np.abs(y_hat - teacher), axis=1)
    else:
        query_mae = np.zeros(len(out))
    out["structure_query_mae"] = query_mae
    out["structure_confidence"] = np.clip(1.0 - query_mae / mae_p90, 0.0, 1.0)
    return out


def apply_hybrid_phi(
    query: pd.DataFrame,
    *,
    confidence_threshold: float,
    obj: dict[str, Any] | None = None,
    manifest: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Replace dim_* with φ̂ when confident; keep QCE φ when not (simulated fallback)."""
    if obj is None or manifest is None:
        obj, manifest = load_structure_distill()
    out = predict_phi_hat(query, obj)
    target_cols = list(obj["target_cols"])
    use_hat = out["structure_confidence"] >= float(confidence_threshold)
    out["used_qce_fallback"] = ~use_hat
    for col in target_cols:
        hat_col = f"phi_hat_{col}"
        out[col] = np.where(use_hat, out[hat_col], out[col])
    out["structure_source"] = np.where(use_hat, "distilled", "qce_fallback")
    return out
