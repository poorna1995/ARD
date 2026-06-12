"""Learned residual Δ̂ on Gate 2 route_cost_hat (P2)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config.global_config.paths import (
    daar_models_dir,
    daar_trace_skeleton_path,
    ensure_daar_dirs,
)
from config.local.constants.agents import ROUTER_AGENTS
from daar.gate1 import load_frame
from daar.routing import load_cost_hand
from daar.v2.features import CVEC_COLS, attach_per_agent_trace_features

RESIDUAL_MODEL_STEM = "cost_residual"
RESIDUAL_MANIFEST = "cost_residual_manifest.json"


def _residual_targets(frame: pd.DataFrame, cost: pd.DataFrame) -> pd.DataFrame:
    m = frame.merge(cost, on=["training_id", "agent"], how="inner")
    m["route_cost_oracle"] = m["X1"] + m["X2_dynamic"]
    if "route_cost_hat" not in m.columns:
        m["route_cost_hat"] = m["X1_hat"] + m["X2_dynamic_hat"]
    m["delta_route_cost"] = m["route_cost_oracle"] - m["route_cost_hat"]
    return m


def train_cost_residual(
    *,
    random_state: int = 42,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    train = load_frame("train")
    cost = load_cost_hand()
    traces = pd.read_parquet(daar_trace_skeleton_path())
    m = _residual_targets(train, cost)
    m = attach_per_agent_trace_features(m, traces)

    models: dict[str, Any] = {}
    metrics: dict[str, Any] = {}
    for agent in ROUTER_AGENTS:
        sub = m[m["agent"] == agent].copy()
        feat = [c for c in CVEC_COLS if c in sub.columns]
        feat += [c for c in sub.columns if c.startswith(f"trace_{agent}_")]
        x = sub[feat].astype(float).fillna(0.0).to_numpy()
        y = sub["delta_route_cost"].astype(float).to_numpy()
        pipe = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "reg",
                    HistGradientBoostingRegressor(
                        max_depth=4,
                        learning_rate=0.05,
                        max_iter=150,
                        random_state=random_state,
                    ),
                ),
            ]
        )
        pipe.fit(x, y)
        pred = pipe.predict(x)
        models[agent] = {"pipeline": pipe, "feature_cols": feat}
        metrics[agent] = {
            "n": len(sub),
            "train_mae": float(np.mean(np.abs(pred - y))),
            "train_rmse": float(np.sqrt(np.mean((pred - y) ** 2))),
        }

    model_path = models_dir / f"{RESIDUAL_MODEL_STEM}.joblib"
    joblib.dump(models, model_path)

    manifest: dict[str, Any] = {
        "model": RESIDUAL_MODEL_STEM,
        "target": "delta_route_cost = route_cost_oracle - route_cost_hat",
        "per_agent": metrics,
        "model_path": str(model_path),
        "routing": "route_cost_hat_final = route_cost_hat + delta_hat",
    }
    manifest_path = models_dir / RESIDUAL_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def load_cost_residual(models_dir: Path | None = None) -> dict[str, Any]:
    models_dir = models_dir or daar_models_dir()
    path = models_dir / f"{RESIDUAL_MODEL_STEM}.joblib"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path} — run: uv run python scripts/train_daar2_cost_residual.py"
        )
    return joblib.load(path)


def apply_cost_residual(
    query_table: pd.DataFrame,
    *,
    residual_models: dict[str, Any] | None = None,
    traces: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Add ``delta_hat_*`` and update ``route_cost_hat_*`` wide columns on query table."""
    if residual_models is None:
        residual_models = load_cost_residual()
    out = query_table.copy()
    out["training_id"] = out["training_id"].astype(str)

    if traces is not None:
        long_rows: list[dict[str, Any]] = []
        for agent in ROUTER_AGENTS:
            for _, row in out.iterrows():
                long_rows.append({"training_id": row["training_id"], "agent": agent})
        long_df = attach_per_agent_trace_features(pd.DataFrame(long_rows), traces)
        long_df = long_df.merge(
            out[["training_id", *[c for c in CVEC_COLS if c in out.columns]]],
            on="training_id",
            how="left",
        )
        for agent in ROUTER_AGENTS:
            spec = residual_models[agent]
            feat = spec["feature_cols"]
            sub = long_df[long_df["agent"] == agent]
            if sub.empty:
                continue
            delta = spec["pipeline"].predict(sub[feat].astype(float).fillna(0.0).to_numpy())
            wide = pd.DataFrame(
                {"training_id": sub["training_id"].values, f"delta_hat_{agent}": delta}
            )
            out = out.merge(wide, on="training_id", how="left")
    else:
        for agent in ROUTER_AGENTS:
            spec = residual_models[agent]
            feat = [c for c in spec["feature_cols"] if c in out.columns]
            if not feat:
                out[f"delta_hat_{agent}"] = 0.0
                continue
            delta = spec["pipeline"].predict(out[feat].astype(float).fillna(0.0).to_numpy())
            out[f"delta_hat_{agent}"] = delta

    for agent in ROUTER_AGENTS:
        dcol = f"delta_hat_{agent}"
        hcol = f"route_cost_hat_{agent}"
        if dcol not in out.columns:
            out[dcol] = 0.0
        if hcol in out.columns:
            out[hcol] = out[hcol].astype(float) + out[dcol].astype(float)
        norm_key = f"route_cost_hat_norm_{agent}"
        if norm_key in out.columns and hcol in out.columns:
            # re-normalize using same denominator stored implicitly in norm * old_hat
            denom = out[hcol].astype(float) / out[norm_key].astype(float).replace(0, np.nan)
            med = float(denom.median(skipna=True)) or 1.0
            out[norm_key] = out[hcol].astype(float) / med
    return out
