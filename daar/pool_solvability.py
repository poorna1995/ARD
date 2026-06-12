"""Pool-solvability detection — pre-execution abstention from φ(q) and emb(q)."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from daar.query_frames import SOLVABLE_LABEL

POOL_SOLVABILITY_MANIFEST = "pool_solvability_manifest.json"
POOL_SOLVABILITY_MODEL = "pool_solvability_model"
POOL_SOLVABILITY_FREEZE_VERSION = "v1_final"
POOL_SOLVABILITY_FREEZE_JSON = "pool_solvability_freeze.json"
ABSTENTION_THRESHOLD_GRID = tuple(round(x, 2) for x in np.arange(0.0, 1.01, 0.05))
MAX_FALSE_ABSTAIN_RATE = 0.05
ThresholdStrategy = Literal[
    "max_recall_unsolvable", "max_f1", "min_false_abstention", "max_threshold_under_cap"
]
SOLVABILITY_PROBABILITY_COL = "solvability_probability"


def read_abstention_threshold(manifest: dict[str, Any]) -> float:
    """Read frozen abstention threshold from a model manifest."""
    return float(
        manifest.get(
            "abstention_threshold",
            manifest.get("theta_s", manifest.get("theta_abstain", 0.5)),
        )
    )


def should_abstain(solvability_probability: float, abstention_threshold: float) -> bool:
    """Abstain when calibrated pool-solvability probability is below the threshold."""
    return float(solvability_probability) < float(abstention_threshold)


def abstention_mask(
    solvability_probability: np.ndarray | pd.Series,
    abstention_threshold: float,
) -> np.ndarray:
    return np.asarray(solvability_probability, dtype=float) < float(abstention_threshold)


def sweep_abstention_threshold(
    solvability_probability: pd.Series | np.ndarray,
    pool_solvable: pd.Series | np.ndarray,
    *,
    threshold_grid: tuple[float, ...] = ABSTENTION_THRESHOLD_GRID,
) -> pd.DataFrame:
    """Validation-only threshold sweep: abstain iff solvability_probability < threshold."""
    y = np.asarray(pool_solvable, dtype=int)
    scores = np.asarray(solvability_probability, dtype=float)
    n_solvable = int((y == 1).sum())
    n_unsolvable = int((y == 0).sum())
    records: list[dict[str, Any]] = []
    for threshold in threshold_grid:
        abstain = scores < float(threshold)
        route = ~abstain
        false_abs_solvable = int((abstain & (y == 1)).sum())
        true_abs_unsolvable = int((abstain & (y == 0)).sum())
        y_unsolv = (y == 0).astype(int)
        pred_unsolv = abstain.astype(int)
        records.append(
            {
                "abstention_threshold": float(threshold),
                "precision_unsolvable": float(
                    precision_score(y_unsolv, pred_unsolv, zero_division=0)
                ),
                "recall_unsolvable": float(
                    recall_score(y_unsolv, pred_unsolv, zero_division=0)
                ),
                "f1_unsolvable": float(f1_score(y_unsolv, pred_unsolv, zero_division=0)),
                "abstention_rate": float(abstain.mean()),
                "coverage": float(route.mean()),
                "false_abstention_rate_solvable": (
                    float(false_abs_solvable / n_solvable) if n_solvable else 0.0
                ),
                "correct_abstention_rate_unsolvable": (
                    float(true_abs_unsolvable / n_unsolvable) if n_unsolvable else 0.0
                ),
                "n_false_abstain_solvable": false_abs_solvable,
                "n_correct_abstain_unsolvable": true_abs_unsolvable,
                "n_abstain": int(abstain.sum()),
            }
        )
    return pd.DataFrame(records)


def select_abstention_threshold(
    sweep: pd.DataFrame,
    *,
    strategy: ThresholdStrategy = "max_recall_unsolvable",
    max_false_abstention_solvable: float | None = MAX_FALSE_ABSTAIN_RATE,
) -> float:
    """Choose abstention threshold from a validation sweep (never use test)."""
    if sweep.empty:
        raise ValueError("empty abstention threshold sweep")
    sub = sweep
    if max_false_abstention_solvable is not None:
        sub = sub[
            sub["false_abstention_rate_solvable"].astype(float)
            <= float(max_false_abstention_solvable)
        ]
        if sub.empty:
            raise ValueError(
                "no threshold with false_abstention_rate_solvable "
                f"<= {max_false_abstention_solvable}"
            )
    if strategy == "max_recall_unsolvable":
        idx = sub["recall_unsolvable"].astype(float).idxmax()
    elif strategy == "max_f1":
        idx = sub["f1_unsolvable"].astype(float).idxmax()
    elif strategy == "max_threshold_under_cap":
        idx = sub["abstention_threshold"].astype(float).idxmax()
    elif strategy == "min_false_abstention":
        idx = sub["false_abstention_rate_solvable"].astype(float).idxmin()
    else:
        raise ValueError(f"unknown threshold strategy {strategy!r}")
    return float(sweep.loc[idx, "abstention_threshold"])


def tune_abstention_threshold(
    solvability_probability: pd.Series | np.ndarray,
    pool_solvable: pd.Series | np.ndarray,
    *,
    strategy: ThresholdStrategy = "max_recall_unsolvable",
    max_false_abstention_solvable: float | None = MAX_FALSE_ABSTAIN_RATE,
) -> tuple[float, pd.DataFrame]:
    sweep = sweep_abstention_threshold(solvability_probability, pool_solvable)
    threshold = select_abstention_threshold(
        sweep,
        strategy=strategy,
        max_false_abstention_solvable=max_false_abstention_solvable,
    )
    return threshold, sweep


def expected_calibration_error(
    solvability_probability: np.ndarray,
    pool_solvable: np.ndarray,
    *,
    n_bins: int = 10,
) -> float:
    """Expected calibration error with uniform bins in [0, 1]."""
    scores = np.asarray(solvability_probability, dtype=float)
    y = np.asarray(pool_solvable, dtype=int)
    if len(scores) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (scores >= lo) & (scores < hi if hi < 1.0 else scores <= hi)
        if not mask.any():
            continue
        acc = float(y[mask].mean())
        conf = float(scores[mask].mean())
        ece += mask.mean() * abs(acc - conf)
    return float(ece)


def evaluate_solvability_detection(
    solvability_probability: pd.Series | np.ndarray,
    pool_solvable: pd.Series | np.ndarray,
    *,
    abstention_threshold: float | None = None,
) -> dict[str, float]:
    """Discrimination metrics and optional abstention operating point."""
    y = np.asarray(pool_solvable, dtype=int)
    scores = np.asarray(solvability_probability, dtype=float)
    y_unsolv = (y == 0).astype(int)
    out: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) > 1 else float("nan"),
        "pr_auc": (
            float(average_precision_score(y, scores)) if len(np.unique(y)) > 1 else float("nan")
        ),
        "brier": float(brier_score_loss(y, scores)),
        "ece": expected_calibration_error(scores, y),
        "n": float(len(y)),
        "n_solvable": float((y == 1).sum()),
        "n_unsolvable": float((y == 0).sum()),
    }
    if abstention_threshold is None:
        return out

    abstain = abstention_mask(scores, abstention_threshold)
    route = ~abstain
    routed_unsolv = route & (y == 0)
    selective_risk = (
        float(routed_unsolv.sum() / route.sum()) if route.any() else float("nan")
    )

    out.update(
        {
            "abstention_threshold": float(abstention_threshold),
            "abstention_rate": float(abstain.mean()),
            "coverage": float(route.mean()),
            "selective_risk_unsolvable_among_routed": selective_risk,
            "n_abstain": float(abstain.sum()),
            "precision_unsolvable": float(
                precision_score(y_unsolv, abstain.astype(int), zero_division=0)
            ),
            "recall_unsolvable": float(
                recall_score(y_unsolv, abstain.astype(int), zero_division=0)
            ),
            "f1_unsolvable": float(
                f1_score(y_unsolv, abstain.astype(int), zero_division=0)
            ),
            "false_abstention_rate_solvable": float(
                (abstain & (y == 1)).sum() / max((y == 1).sum(), 1)
            ),
        }
    )
    return out


def calibrate_solvability_on_validation(
    base_pipe: Any,
    validation_features: pd.DataFrame,
    validation_labels: pd.Series,
    feature_columns: list[str],
) -> dict[str, Any]:
    """Platt scaling on validation raw scores — training split is not used."""
    raw = predict_solvability_probability(base_pipe, validation_features, feature_columns)
    platt = LogisticRegression(max_iter=1000)
    platt.fit(
        np.asarray(raw, dtype=float).reshape(-1, 1),
        np.asarray(validation_labels, dtype=int),
    )
    return {"pipeline": base_pipe, "platt": platt}


def _classifier_pipeline(model: Any) -> Any:
    return model["pipeline"] if isinstance(model, dict) and "pipeline" in model else model


def positive_class_index(model: Any) -> int:
    """Index of the pool-solvable positive class in a fitted classifier."""
    base = _classifier_pipeline(model)
    clf = base.named_steps["clf"] if hasattr(base, "named_steps") else base
    est = getattr(clf, "estimator", clf)
    classes = list(getattr(clf, "classes_", getattr(est, "classes_", [])))
    if 1 in classes:
        return classes.index(1)
    if "1" in classes:
        return classes.index("1")
    raise ValueError(f"no positive solvable class in {classes!r}")


def predict_solvability_probability(
    model: Any,
    query_frame: pd.DataFrame,
    feature_columns: list[str],
) -> np.ndarray:
    """Calibrated P(pool-solvable | φ, emb) for query-level rows."""
    base = _classifier_pipeline(model)
    idx = positive_class_index(model)
    raw = base.predict_proba(query_frame[feature_columns])[:, idx]
    if isinstance(model, dict) and model.get("platt") is not None:
        return model["platt"].predict_proba(np.asarray(raw, dtype=float).reshape(-1, 1))[:, 1]
    return raw


def build_pool_solvability_freeze_document(
    *,
    feature_set: str,
    abstention_threshold: float,
    split_query_counts: dict[str, int],
    val_discrimination: dict[str, float],
    val_at_threshold: dict[str, float],
    expected_test: dict[str, float] | None = None,
    test_eval_run: bool = False,
) -> dict[str, Any]:
    """Canonical frozen spec for pool-solvability detection (φ + emb baseline)."""
    recall_pct = round(float(val_at_threshold["recall_unsolvable"]) * 100, 1)
    doc: dict[str, Any] = {
        "freeze_version": POOL_SOLVABILITY_FREEZE_VERSION,
        "status": "frozen",
        "implementation_id": "pool_solvability_detector",
        "target": "y = max_a r_a",
        "target_binary": "y = 1 iff max_a r_a = 1 (any agent correct)",
        "features": "phi(q) + emb(q)",
        "feature_set": feature_set,
        "feature_dims": {
            "phi": (
                "cvec5 — dim_structural, dim_reasoning, dim_evidence, "
                "dim_tool, dim_coordination_uncertainty"
            ),
            "emb": "PCA-16 sentence embedding (emb_0 … emb_15)",
        },
        "classifier": "HGBM",
        "class_weight": "balanced",
        "calibration": "Platt",
        "calibration_split": "val",
        "threshold_objective": (
            "maximize Recall_unsolvable subject to FalseAbstain_solvable <= 5%"
        ),
        "threshold_selection": (
            f"max_recall_unsolvable @ {MAX_FALSE_ABSTAIN_RATE:.0%} false_abstain_solvable"
        ),
        "abstention_threshold": float(abstention_threshold),
        "max_false_abstain_rate": MAX_FALSE_ABSTAIN_RATE,
        "split_query_counts": split_query_counts,
        "headline_result": {
            "roc_auc_range": "0.68–0.72",
            "recall_at_five_percent_false_abstain_percent": "14–15%",
            "interpretation": (
                "pre-execution pool-solvability ceiling — discrimination without "
                "operational abstention gain"
            ),
        },
        "expected_val": {
            "roc_auc": val_discrimination["roc_auc"],
            "pr_auc": val_discrimination["pr_auc"],
            "ece": val_discrimination["ece"],
            "recall_unsolvable": val_at_threshold["recall_unsolvable"],
            "recall_unsolvable_percent": recall_pct,
            "false_abstention_rate_solvable": val_at_threshold[
                "false_abstention_rate_solvable"
            ],
        },
        "test_eval_run": test_eval_run,
        "canonical_scripts": [
            "scripts/train_pool_solvability.py",
            "scripts/eval_solvability_frozen.py",
            "scripts/verify_solvability_freeze.py",
        ],
        "experiment_artifacts": [
            "scripts/audit_solvability_features.py",
            "scripts/plot_solvability_overlap.py",
        ],
        "model_artifacts": [
            "models/daar/pool_solvability_model.joblib",
            "models/daar/pool_solvability_manifest.json",
            "models/daar/abstention_threshold_sweep_val.csv",
        ],
        "eval_artifacts": [
            "datasets/daar/routing/abstention_threshold_sweep_val.json",
            "datasets/daar/routing/solvability_frozen_eval_val.json",
            "datasets/daar/routing/solvability_frozen_eval_test.json",
            "datasets/daar/routing/solvability_feature_audit.json",
            "datasets/daar/routing/solvability_feature_ablation.json",
            "datasets/daar/routing/solvability_trace_pair.json",
        ],
    }
    if expected_test is not None:
        doc["expected_test"] = expected_test
    return doc


# Deprecated aliases (internal thesis freeze names).
GATE1A_MANIFEST = POOL_SOLVABILITY_MANIFEST
GATE1A_MODEL_STEM = POOL_SOLVABILITY_MODEL
GATE1A_FREEZE_VERSION = POOL_SOLVABILITY_FREEZE_VERSION
GATE1A_FREEZE_JSON = POOL_SOLVABILITY_FREEZE_JSON
FALSE_ABSTAIN_CAP = MAX_FALSE_ABSTAIN_RATE
THETA_GRID = ABSTENTION_THRESHOLD_GRID
ThetaSelection = ThresholdStrategy
SOLVABLE_TARGET = SOLVABLE_LABEL
sweep_theta_abstention = sweep_abstention_threshold
pick_theta_abstain = select_abstention_threshold
tune_theta = tune_abstention_threshold
def eval_detection(
    solvability_probability: pd.Series | np.ndarray,
    pool_solvable: pd.Series | np.ndarray,
    *,
    theta_s: float | None = None,
    abstention_threshold: float | None = None,
) -> dict[str, float]:
    threshold = (
        abstention_threshold if abstention_threshold is not None else theta_s
    )
    return evaluate_solvability_detection(
        solvability_probability, pool_solvable, abstention_threshold=threshold
    )


eval_abstention = eval_detection
calibrate_gate1a_on_val = calibrate_solvability_on_validation
predict_solvable_proba = predict_solvability_probability
abstain_mask = abstention_mask
