"""
Router artifact load/save and batch inference.

IN:  feature DataFrame · trained ``.joblib``
MID: sklearn pipeline predict · proba ranking · complexity scalar
OUT: routed frame (`router_pred`, `p_*`, `max_prob`, `assigned_agent`)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from config.settings import router_model_path
from routing.config import AGENTS, PROBA_COLS, REPO_ROOT, ROUTER_MODEL_PATH, TARGET
from routing.data.aliases import normalize_loader_frame
from routing.features.columns import overall_complexity
from routing.train.pipeline import (
    LabelEncodingClassifier,
    _register_label_encoding_for_unpickle,
    predict_agent_proba,
    top_k_agents,
    top_k_from_row,
)

__all__ = [
    "LabelEncodingClassifier",
    "attach_router_predictions",
    "load_router",
    "load_router_frame",
    "save_router",
    "top_k_agents",
    "top_k_from_row",
]


def load_router(path: str | Path) -> dict[str, Any]:
    _register_label_encoding_for_unpickle()
    obj = joblib.load(path)
    if not isinstance(obj, dict) or "pipeline" not in obj:
        raise TypeError(f"not a router artifact: {path}")
    return obj


def save_router(
    pipe: Pipeline,
    *,
    feature_cols: list[str],
    experiment_id: str,
    path: Path | None = None,
    extra_meta: dict[str, Any] | None = None,
) -> Path:
    path = Path(path or REPO_ROOT / "models/router" / f"{experiment_id}.joblib")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(extra_meta or {})
    joblib.dump(
        {
            "pipeline": pipe,
            "feature_cols": feature_cols,
            "target": TARGET,
            "agents": list(AGENTS),
            "experiment_id": experiment_id,
            "feature_set": meta.get("feature_set"),
            "hgbm_params": meta.get("hgbm_params"),
            "production": meta.get("production", False),
        },
        path,
    )
    sidecar = {"experiment_id": experiment_id, "n_features": len(feature_cols), "feature_cols": feature_cols}
    if extra_meta:
        sidecar.update(extra_meta)
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")
    return path


def attach_router_predictions(
    df: pd.DataFrame,
    pipe: Any,
    feature_cols: list[str],
    *,
    experiment_id: str = "router",
) -> pd.DataFrame:
    proba = predict_agent_proba(pipe, df, feature_cols)
    out = df.copy()
    for c in proba.columns:
        out[c] = proba[c].values
    agents = list(AGENTS)
    p = out[PROBA_COLS].to_numpy()
    order = np.argsort(-p, axis=1)
    out["router_pred"] = [agents[i] for i in order[:, 0]]
    out["max_prob"] = p[np.arange(len(p)), order[:, 0]]
    out["second_prob"] = p[np.arange(len(p)), order[:, 1]]
    out["margin_top2"] = out["max_prob"] - out["second_prob"]
    out["router_second"] = [agents[i] for i in order[:, 1]]
    if p.shape[1] > 2:
        out["router_third"] = [agents[i] for i in order[:, 2]]
    out["router_experiment"] = experiment_id
    return out


def load_router_frame(df: pd.DataFrame, router_path: Path | str | None = None) -> pd.DataFrame:
    df = normalize_loader_frame(df)
    path = router_model_path(ROUTER_MODEL_PATH) if router_path is None else Path(router_path)
    obj = load_router(path)
    feature_cols: list[str] = list(obj["feature_cols"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Router needs missing columns: {missing}")
    out = attach_router_predictions(
        df, obj["pipeline"], feature_cols, experiment_id=str(obj.get("experiment_id", path.stem))
    )
    out["overall"] = out.apply(overall_complexity, axis=1)
    out["assigned_agent"] = out["router_pred"]
    return out
