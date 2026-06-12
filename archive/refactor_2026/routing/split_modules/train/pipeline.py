"""Sklearn router training pipeline (hard-label HGBM)."""

from __future__ import annotations

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
    HEURISTICS_PARQUET,
    HEURISTIC_COL_PREFIX,
    PROBA_COLS,
    PRODUCTION_FEATURE_SET,
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

WeightMode = Literal["none", "balanced", "soft", "balanced_soft"]
class LabelEncodingClassifier(BaseEstimator, ClassifierMixin):
    """Encode string labels for estimators that need integer ``y``."""

    _estimator_type = "classifier"

    def __init__(self, estimator: Any | None = None):
        self.estimator = estimator

    def __sklearn_tags__(self):
        from sklearn.utils._tags import ClassifierTags

        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags(multi_class=True)
        return tags

    def fit(self, X: Any, y: Any, sample_weight: np.ndarray | None = None) -> LabelEncodingClassifier:
        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(np.asarray(y, dtype=str))
        from sklearn.base import clone

        est = clone(self.estimator)
        if sample_weight is not None:
            est.fit(X, y_enc, sample_weight=sample_weight)
        else:
            est.fit(X, y_enc)
        self.estimator_ = est
        self.classes_ = self.le_.classes_
        return self

    def predict(self, X: Any) -> np.ndarray:
        return self.le_.inverse_transform(self.estimator_.predict(X))

    def predict_proba(self, X: Any) -> np.ndarray:
        return self.estimator_.predict_proba(X)


def _register_label_encoding_for_unpickle() -> None:
    cls = LabelEncodingClassifier
    for name in (
        "__main__",
        "train_router",
        "routing.train_router",
        "routing.router",
        "label_encoding",
        "routing.label_encoding",
    ):
        mod = sys.modules.get(name)
        if mod is None and name != "__main__":
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        if mod is not None:
            setattr(mod, "LabelEncodingClassifier", cls)

def pipeline(
    feature_cols: list[str],
    *,
    class_weight: str | dict[str, float] | None = "balanced",
    random_state: int = 42,
    hgbm_params: dict[str, Any] | None = None,
    calibrated: bool = False,
    calibration_cv: int = 3,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> Pipeline:
    num_cols = [c for c in feature_cols if c != "dataset"]
    cat_cols = ["dataset"] if "dataset" in feature_cols else []
    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", SimpleImputer(strategy="constant", fill_value=0), num_cols),
    ]
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    hgbm_kw: dict[str, Any] = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "l2_regularization": 0.0,
        "class_weight": class_weight,
        "random_state": random_state,
    }
    if hgbm_params:
        hgbm_kw.update(hgbm_params)
    clf: Any = HistGradientBoostingClassifier(**hgbm_kw)
    if calibrated:
        clf = CalibratedClassifierCV(clf, cv=calibration_cv, method=calibration_method)
    return Pipeline([("prep", ColumnTransformer(transformers)), ("clf", clf)])


def oracle_soft_probability(df: pd.DataFrame, y: pd.Series) -> np.ndarray:
    weights = np.ones(len(df), dtype=float)
    for i, (idx, agent) in enumerate(y.items()):
        col = f"p_{agent}"
        if col in df.columns:
            weights[i] = float(df.at[idx, col])
    return weights


def compute_sample_weights(df: pd.DataFrame, y: pd.Series, mode: WeightMode) -> np.ndarray | None:
    if mode == "none":
        return None
    if mode == "soft":
        w = oracle_soft_probability(df, y)
    elif mode == "balanced":
        w = compute_sample_weight("balanced", y)
    elif mode == "balanced_soft":
        w = compute_sample_weight("balanced", y) * oracle_soft_probability(df, y)
    else:
        raise ValueError(f"unknown weight mode {mode!r}")
    w = np.asarray(w, dtype=float)
    return w / max(w.mean(), 1e-12)


def oversample_minorities(
    df: pd.DataFrame, target: str, *, min_fraction: float = 0.35, random_state: int = 42
) -> pd.DataFrame:
    counts = df[target].value_counts()
    floor_n = max(1, int(int(counts.max()) * min_fraction))
    rng = np.random.default_rng(random_state)
    parts = [df]
    for label, n in counts.items():
        if n >= floor_n:
            continue
        subset = df[df[target] == label]
        idx = rng.choice(subset.index.to_numpy(), size=floor_n - n, replace=True)
        parts.append(df.loc[idx])
    return pd.concat(parts, ignore_index=True)


def fit_router(
    pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = TARGET,
    sample_weight: np.ndarray | None = None,
) -> Pipeline:
    X, y = df[feature_cols], df[target].astype(str)
    if sample_weight is not None:
        pipe.fit(X, y, clf__sample_weight=sample_weight)
    else:
        pipe.fit(X, y)
    return pipe


@dataclass(frozen=True)
class TrainSpec:
    experiment_id: str = PRODUCTION_ROUTER_EXPERIMENT_ID
    feature_set: FeatureSet = PRODUCTION_FEATURE_SET
    weight_mode: WeightMode = "balanced"
    oversample: bool = False
    target: str = TARGET
    use_class_weight: bool = True
    hgbm_params: dict[str, Any] | None = None
    calibrated: bool = False
    random_state: int = 42

    @property
    def effective_class_weight(self) -> str | None:
        if self.use_class_weight and self.weight_mode in ("none", "balanced"):
            return "balanced"
        return None


ROUTER_SPEC = TrainSpec()


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    macro_f1: float
    report: str
    predictions: np.ndarray


def predict_agent_proba(pipe: Pipeline, df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    clf = pipe.named_steps["clf"]
    proba = pipe.predict_proba(df[feature_cols])
    classes = [str(c) for c in clf.classes_]
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=df.index)
    for a in AGENTS:
        if f"p_{a}" not in out.columns:
            out[f"p_{a}"] = 0.0
    return out[PROBA_COLS]


def top_k_agents(proba_row: pd.Series, k: int = 3) -> list[tuple[str, float]]:
    if k < 1:
        raise ValueError("k must be >= 1")
    order = np.argsort(-proba_row.to_numpy(dtype=float))
    agents = [str(c).replace("p_", "") for c in proba_row.index]
    return [(agents[i], float(proba_row.iloc[i])) for i in order[:k]]


def top_k_from_row(row: pd.Series | dict[str, Any], k: int = 3) -> list[tuple[str, float]]:
    """Top-``k`` ``(agent, probability)`` from a routed dataframe row."""
    r = dict(row)
    series = pd.Series({c: r[c] for c in PROBA_COLS if c in r})
    if series.empty:
        return []
    return top_k_agents(series, k=k)


def agent_run_failed(response: dict[str, Any]) -> bool:
    """True if an agent run should trigger cascade (error flag or empty answer)."""
    if response.get("is_failed"):
        return True
    ans = str(response.get("predicted_answer") or response.get("answer") or "").strip()
    return not ans


def run_agent_cascade(
    row: pd.Series | dict[str, Any],
    *,
    query: str,
    dataset: str,
    model: str = DEFAULT_AGENT_MODEL,
    k: int = 3,
    expected_answer: Any = None,
    **run_kw: Any,
) -> dict[str, Any]:
    """
  Try up to ``k`` router-ranked agents until one succeeds.

    Returns ``executed_agent``, ``cascade_rank`` (1-based), ``cascade_attempts``, and final ``response``.
    """
    from agent.registry import STRATEGY_REGISTRY, run_strategy

    ranked = top_k_from_row(row, k=k)
    ds = resolve_dataset_name(dataset)
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for rank, (agent, prob) in enumerate(ranked, start=1):
        if agent not in STRATEGY_REGISTRY:
            attempts.append({"rank": rank, "agent": agent, "prob": prob, "skipped": True})
            continue
        try:
            response = run_strategy(
                agent,
                query,
                model=model,
                dataset=ds,
                expected_answer=expected_answer,
                **run_kw,
            )
            payload = asdict(response) if hasattr(response, "__dataclass_fields__") else dict(response)
            failed = agent_run_failed(payload)
            attempts.append(
                {
                    "rank": rank,
                    "agent": agent,
                    "prob": prob,
                    "failed": failed,
                    "is_failed": bool(payload.get("is_failed")),
                }
            )
            final = {
                "agent": agent,
                "executed_agent": agent,
                "cascade_rank": rank,
                "model": model,
                "response": payload,
            }
            if not failed:
                break
        except Exception as exc:
            attempts.append(
                {"rank": rank, "agent": agent, "prob": prob, "failed": True, "error": str(exc)}
            )
            final = {
                "agent": agent,
                "executed_agent": agent,
                "cascade_rank": rank,
                "model": model,
                "response": {"is_failed": True, "error": str(exc), "predicted_answer": ""},
            }

    if final is None:
        final = {
            "agent": "",
            "executed_agent": "",
            "cascade_rank": 0,
            "model": model,
            "response": {"is_failed": True, "predicted_answer": ""},
        }

    final["top3_agents"] = ranked
    final["cascade_attempts"] = attempts
    final["router_pred"] = str(dict(row).get("router_pred") or dict(row).get("assigned_agent") or "")
    return final


def evaluate(
    pipe: Pipeline, df: pd.DataFrame, feature_cols: list[str], *, target: str = TARGET
) -> EvalResult:
    y = df[target].astype(str)
    pred = pipe.predict(df[feature_cols])
    labels = list(AGENTS)
    return EvalResult(
        float(accuracy_score(y, pred)),
        float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        classification_report(y, pred, labels=labels, zero_division=0),
        pred,
    )


def baseline_always_majority(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> EvalResult:
    majority = train_df[TARGET].mode()[0]
    pred = np.array([majority] * len(eval_df))
    y = eval_df[TARGET].astype(str)
    labels = list(AGENTS)
    return EvalResult(
        float(accuracy_score(y, pred)),
        float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        classification_report(y, pred, labels=labels, zero_division=0),
        pred,
    )


def baseline_per_dataset_mode(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> EvalResult:
    mode = train_df.groupby("dataset")[TARGET].agg(lambda s: s.mode()[0])
    fallback = train_df[TARGET].mode()[0]
    pred = eval_df["dataset"].map(mode).fillna(fallback).astype(str).values
    y = eval_df[TARGET].astype(str)
    labels = list(AGENTS)
    return EvalResult(
        float(accuracy_score(y, pred)),
        float(f1_score(y, pred, average="macro", labels=labels, zero_division=0)),
        classification_report(y, pred, labels=labels, zero_division=0),
        pred,
    )


def cross_val_macro_f1(
    pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = TARGET,
    sample_weight: np.ndarray | None = None,
    cv: int = 5,
) -> tuple[float, float]:
    params = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    scores = cross_val_score(
        pipe, df[feature_cols], df[target].astype(str), cv=cv, scoring="f1_macro", n_jobs=-1, params=params
    )
    return float(scores.mean()), float(scores.std())


def train_and_evaluate(
    spec: TrainSpec, train_df: pd.DataFrame, val_df: pd.DataFrame
) -> tuple[Pipeline, list[str], EvalResult, tuple[float, float]]:
    feature_cols = router_feature_cols(train_df, spec.feature_set)
    fit_df = oversample_minorities(train_df, spec.target) if spec.oversample else train_df
    y_fit = fit_df[spec.target].astype(str)
    weights = compute_sample_weights(fit_df, y_fit, spec.weight_mode)
    pipe = pipeline(
        feature_cols,
        class_weight=spec.effective_class_weight,
        hgbm_params=spec.hgbm_params,
        calibrated=spec.calibrated,
        random_state=spec.random_state,
    )
    fit_router(pipe, fit_df, feature_cols, target=spec.target, sample_weight=weights)
    val_result = evaluate(pipe, val_df, feature_cols, target=spec.target)
    cv_mean, cv_std = cross_val_macro_f1(
        pipeline(
            feature_cols,
            class_weight=spec.effective_class_weight,
            hgbm_params=spec.hgbm_params,
            calibrated=False,
            random_state=spec.random_state,
        ),
        train_df,
        feature_cols,
        target=spec.target,
        sample_weight=compute_sample_weights(train_df, train_df[spec.target].astype(str), spec.weight_mode),
    )
    return pipe, feature_cols, val_result, (cv_mean, cv_std)


def results_row(name: str, result: EvalResult, **extra: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": name,
        "accuracy": round(result.accuracy, 4),
        "macro_f1": round(result.macro_f1, 4),
    }
    for k, v in extra.items():
        if v is not None and v is not False:
            row[k] = round(v, 4) if isinstance(v, float) else v
    return row


def learnability_summary(
    train_df: pd.DataFrame, val_df: pd.DataFrame, experiment_rows: list[dict[str, Any]]
) -> dict[str, Any]:
    exp_df = pd.DataFrame(experiment_rows)
    ml = exp_df[exp_df["experiment"] == ROUTER_EXPERIMENT_ID]
    best = ml.loc[ml["macro_f1"].idxmax()] if len(ml) else None
    b0_mode = exp_df[exp_df["experiment"] == "B0_dataset_mode"].iloc[0]
    b0_raw = exp_df[exp_df["experiment"] == "B0_always_raw"].iloc[0]
    soft = oracle_soft_probability(train_df, train_df[TARGET].astype(str))
    summary: dict[str, Any] = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "train_class_counts": train_df[TARGET].value_counts().to_dict(),
        "val_class_counts": val_df[TARGET].value_counts().to_dict(),
        "soft_p_oracle_mean": float(np.mean(soft)),
        "baseline_val_macro_f1_always_raw": float(b0_raw["macro_f1"]),
        "baseline_val_macro_f1_dataset_mode": float(b0_mode["macro_f1"]),
    }
    if best is not None:
        summary["best_ml_val_macro_f1"] = float(best["macro_f1"])
        summary["lift_macro_f1_vs_dataset_mode"] = round(float(best["macro_f1"]) - float(b0_mode["macro_f1"]), 4)
    return summary


def train_router(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    spec: TrainSpec = ROUTER_SPEC,
    *,
    verbose: bool = True,
) -> tuple[Pipeline, list[str], EvalResult, tuple[float, float], pd.DataFrame]:
    rows: list[dict[str, Any]] = [
        results_row("B0_always_raw", baseline_always_majority(train_df, val_df)),
        results_row("B0_dataset_mode", baseline_per_dataset_mode(train_df, val_df)),
    ]
    pipe, feature_cols, val_res, (cv_mean, cv_std) = train_and_evaluate(spec, train_df, val_df)
    rows.append(
        results_row(
            spec.experiment_id,
            val_res,
            feature_set=normalize_feature_set(spec.feature_set),
            n_features=len(feature_cols),
            model="hgbm",
            weight_mode=spec.weight_mode,
            oversample=spec.oversample,
            train_cv_macro_f1=cv_mean,
            train_cv_std=cv_std,
        )
    )
    if verbose:
        print(
            f"\n=== {spec.experiment_id} ===\n"
            f"features: {len(feature_cols)} | weights={spec.weight_mode}\n"
            f"Train CV macro-F1: {cv_mean:.4f} ± {cv_std:.4f}\n"
            f"Val accuracy: {val_res.accuracy:.4f} | macro-F1: {val_res.macro_f1:.4f}\n{val_res.report}"
        )
        print("\n--- Val comparison ---\n", pd.DataFrame(rows).to_string(index=False))
        print("\n--- Learnability ---\n", json.dumps(learnability_summary(train_df, val_df, rows), indent=2))
    return pipe, feature_cols, val_res, (cv_mean, cv_std), pd.DataFrame(rows)
