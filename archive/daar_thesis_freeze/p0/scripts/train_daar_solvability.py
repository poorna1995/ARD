"""Gate 1 — per-agent solvability HGBM classifiers + abstention threshold on daar_val."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from config.global_config.paths import daar_frames_dir, daar_models_dir, ensure_daar_dirs
from config.local.constants.agents import ROUTER_AGENTS
from config.local.router.production import PRODUCTION_FEATURE_SET
from router.router import fit_router, make_pipeline, router_feature_cols

TARGET = "r_a"
THETA_GRID = tuple(round(x, 2) for x in np.arange(0.05, 0.96, 0.05))


def load_frame(split: str, *, frames_dir: Path | None = None) -> pd.DataFrame:
    frames_dir = frames_dir or daar_frames_dir()
    path = frames_dir / f"daar_{split}_frame.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path} — run: uv run python scripts/build_daar_train_frame.py"
        )
    return pd.read_parquet(path)


def train_agent_model(
    train_df: pd.DataFrame,
    agent: str,
    feature_cols: list[str],
    *,
    random_state: int,
) -> Any:
    sub = train_df[train_df["agent"] == agent].copy()
    if sub.empty:
        raise ValueError(f"no train rows for agent {agent!r}")
    pipe = make_pipeline(feature_cols, random_state=random_state, calibrated=True)
    return fit_router(pipe, sub, feature_cols, target=TARGET)


def query_proba_table(
    df: pd.DataFrame,
    models: dict[str, Any],
    feature_cols: list[str],
) -> pd.DataFrame:
    solvable = (
        df.groupby("training_id", sort=False)[TARGET]
        .max()
        .astype(int)
        .rename("solvable")
    )
    rows: list[dict[str, Any]] = []
    for agent, model in models.items():
        sub = df[df["agent"] == agent].drop_duplicates("training_id").copy()
        if sub.empty:
            continue
        proba = model.predict_proba(sub[feature_cols])[:, 1]
        for tid, p in zip(sub["training_id"].astype(str), proba, strict=True):
            rows.append({"training_id": tid, "agent": agent, "s_hat": float(p)})

    wide = pd.DataFrame(rows).pivot(index="training_id", columns="agent", values="s_hat")
    wide.columns = [f"s_hat_{c}" for c in wide.columns]
    wide = wide.reset_index().merge(solvable.reset_index(), on="training_id", how="left")
    wide["s_hat_max"] = wide[[c for c in wide.columns if c.startswith("s_hat_")]].max(axis=1)
    return wide


def tune_theta(val_table: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    y_true = val_table["solvable"].astype(int).to_numpy()
    scores = val_table["s_hat_max"].to_numpy()
    records: list[dict[str, Any]] = []
    best_theta = 0.5
    best_f1 = -1.0
    for theta in THETA_GRID:
        pred = (scores >= theta).astype(int)
        rec = {
            "theta_s": theta,
            "precision": float(precision_score(y_true, pred, zero_division=0)),
            "recall": float(recall_score(y_true, pred, zero_division=0)),
            "f1": float(f1_score(y_true, pred, zero_division=0)),
        }
        records.append(rec)
        if rec["f1"] > best_f1:
            best_f1 = rec["f1"]
            best_theta = theta
    return best_theta, pd.DataFrame(records)


def eval_agent(
    df: pd.DataFrame,
    model: Any,
    agent: str,
    feature_cols: list[str],
) -> dict[str, Any]:
    sub = df[df["agent"] == agent].copy()
    y = sub[TARGET].astype(int)
    proba = model.predict_proba(sub[feature_cols])[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "agent": agent,
        "n": len(sub),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)) if y.nunique() > 1 else None,
        "avg_precision": float(average_precision_score(y, proba)) if y.nunique() > 1 else None,
    }


def eval_abstention(table: pd.DataFrame, theta_s: float) -> dict[str, Any]:
    y_true = table["solvable"].astype(int).to_numpy()
    route = (table["s_hat_max"].to_numpy() >= theta_s).astype(int)
    return {
        "theta_s": theta_s,
        "abstention_rate": float(1.0 - route.mean()),
        "precision": float(precision_score(y_true, route, zero_division=0)),
        "recall": float(recall_score(y_true, route, zero_division=0)),
        "f1": float(f1_score(y_true, route, zero_division=0)),
    }


def train_daar_solvability(
    *,
    feature_set: str = PRODUCTION_FEATURE_SET,
    random_state: int = 42,
    models_dir: Path | None = None,
    frames_dir: Path | None = None,
) -> dict[str, Any]:
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()
    models_dir.mkdir(parents=True, exist_ok=True)

    train_df = load_frame("train", frames_dir=frames_dir)
    val_df = load_frame("val", frames_dir=frames_dir)
    test_df = load_frame("test", frames_dir=frames_dir)
    feature_cols = router_feature_cols(train_df, feature_set)

    models: dict[str, Any] = {}
    train_metrics: list[dict[str, Any]] = []
    for agent in ROUTER_AGENTS:
        models[agent] = train_agent_model(
            train_df, agent, feature_cols, random_state=random_state
        )
        joblib.dump(models[agent], models_dir / f"solvability_{agent}.joblib")
        train_metrics.append(eval_agent(train_df, models[agent], agent, feature_cols))

    val_table = query_proba_table(val_df, models, feature_cols)
    theta_s, theta_sweep = tune_theta(val_table)

    manifest: dict[str, Any] = {
        "feature_set": feature_set,
        "feature_cols": feature_cols,
        "agents": list(ROUTER_AGENTS),
        "theta_s": theta_s,
        "train_agent_metrics": train_metrics,
        "val_agent_metrics": [eval_agent(val_df, models[a], a, feature_cols) for a in ROUTER_AGENTS],
        "val_abstention": eval_abstention(val_table, theta_s),
        "test_agent_metrics": [
            eval_agent(test_df, models[a], a, feature_cols) for a in ROUTER_AGENTS
        ],
        "test_abstention": eval_abstention(
            query_proba_table(test_df, models, feature_cols), theta_s
        ),
        "models_dir": str(models_dir),
    }

    manifest_path = models_dir / "solvability_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    theta_sweep.to_csv(models_dir / "theta_s_sweep.csv", index=False)
    return manifest


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--feature-set", default=PRODUCTION_FEATURE_SET)
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--frames-dir", type=Path, default=None)
    args = p.parse_args()
    print(json.dumps(train_daar_solvability(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
