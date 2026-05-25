"""Train and evaluate the agent router on QCE fused features."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.utils.class_weight import compute_sample_weight

from routing.label_encoding import LabelEncodingClassifier

REPO_ROOT = Path(__file__).resolve().parents[1]

TARGET = "oracle_agent"
SOFT_DOMINANT = "soft_dominant_agent"
AGENTS = ("raw", "cot", "react", "multiagent")
SOFT_COLS = [f"p_{a}" for a in AGENTS]
PROBA_COLS = SOFT_COLS

_LABELS_ROOT = Path("datasets/train_samples/v1")
_SPLIT_CSV = {
    "train": _LABELS_ROOT / "qce_train.csv",
    "val": _LABELS_ROOT / "qce_val.csv",
    "test": _LABELS_ROOT / "qce_internal_test.csv",
}
_SPLIT_PARQUET = {"train": "train", "val": "val", "test": "test"}
_EMBEDDINGS_PARQUET = "datasets/qce_features/query_embeddings_{split}.parquet"
EMBEDDING_COL_PREFIX = "emb_"

SKIP_Q_COLS = frozenset({"q_schema_ver", "q_status", "q_err"})

GRAPH_COLS_DEFAULT = (
    "complexity_graph",
    "n_nodes",
    "n_edges",
    "max_depth",
    "width",
    "n_tool_nodes",
    "tool_fraction",
    "critical_path_len",
    "critical_path_tool_steps",
    "has_web",
    "has_code",
    "has_file",
    "plan_trust",
    "n_repair_ops",
    "used_chain_fallback",
    "heavily_repaired",
)

try:
    from qce.complexity import C_VECTOR_COLS as _C_VECTOR_COLS
except ImportError:
    _C_VECTOR_COLS = (
        "dim_structural",
        "dim_compositional",
        "dim_retrieval",
        "dim_verification",
        "dim_uncertainty",
    )

FeatureSetId = Literal["m0", "m2", "cvec", "graph", "graph_emb", "scalar"]
ROUTER_MAIN_FEATURE_SET: FeatureSetId = "graph"
WeightMode = Literal["none", "balanced", "soft", "balanced_soft"]


def query_embeddings_path(split: str, *, root: Path | None = None) -> Path:
    if split not in _SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(_SPLIT_PARQUET)}")
    return Path(root or REPO_ROOT) / _EMBEDDINGS_PARQUET.format(split=_SPLIT_PARQUET[split])


def embedding_feature_cols(df: pd.DataFrame) -> list[str]:
    """Sorted ``emb_*`` columns from ``query_embeddings_*.parquet``."""
    cols = [
        c
        for c in df.columns
        if c.startswith(EMBEDDING_COL_PREFIX)
        and pd.api.types.is_numeric_dtype(df[c])
    ]

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len(EMBEDDING_COL_PREFIX) :]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(cols, key=_key)


def load_query_embeddings(split: str, *, root: Path | None = None) -> pd.DataFrame:
    path = query_embeddings_path(split, root=root)
    if not path.is_file():
        raise FileNotFoundError(
            f"missing query embeddings for {split!r}: {path} "
            f"(expected columns training_id, {EMBEDDING_COL_PREFIX}0, …)"
        )
    emb = pd.read_parquet(path)
    if "training_id" not in emb.columns:
        raise ValueError(f"{path}: missing training_id")
    keep = ["training_id", *embedding_feature_cols(emb)]
    return emb[keep].copy()


def load_split(
    split: str,
    *,
    root: Path | None = None,
    with_embeddings: bool | None = None,
) -> pd.DataFrame:
    """Merge v1 labels + C(Q) parquet; optionally query sentence embeddings."""
    if split not in _SPLIT_CSV:
        raise ValueError(f"split must be one of {list(_SPLIT_CSV)}")
    root = Path(root or REPO_ROOT)
    labels = pd.read_csv(root / _SPLIT_CSV[split])
    feats = pd.read_parquet(
        root / "datasets/qce_features" / f"complexity_record_{_SPLIT_PARQUET[split]}.parquet"
    )
    drop = [c for c in feats.columns if c in labels.columns and c != "training_id"]
    feats = feats.drop(columns=[c for c in drop if c in feats.columns], errors="ignore")
    merged = labels.merge(feats, on="training_id", how="inner")

    emb_path = query_embeddings_path(split, root=root)
    if with_embeddings is None:
        with_embeddings = emb_path.is_file()
    if with_embeddings:
        emb = load_query_embeddings(split, root=root)
        drop_emb = [c for c in emb.columns if c in merged.columns and c != "training_id"]
        emb = emb.drop(columns=[c for c in drop_emb if c in emb.columns], errors="ignore")
        merged = merged.merge(emb, on="training_id", how="inner")

    merged = merged[merged[TARGET].isin(AGENTS)].copy()
    if len(merged) == 0:
        raise ValueError(f"{split}: no rows after filtering to {AGENTS}")
    return merged.reset_index(drop=True)


def text_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in df.columns
        if c.startswith("q_")
        and c not in SKIP_Q_COLS
        and pd.api.types.is_numeric_dtype(df[c])
    ]


def graph_feature_cols(df: pd.DataFrame) -> list[str]:
    out: list[str] = []
    for c in GRAPH_COLS_DEFAULT:
        if c not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(c)
    return out


def c_vector_feature_cols(df: pd.DataFrame) -> list[str]:
    """C(Q) — full five-dim interpretable vector (primary graph representation)."""
    return [c for c in _C_VECTOR_COLS if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


def feature_set_m0(df: pd.DataFrame) -> list[str]:
    return text_feature_cols(df) + ["dataset"]


def feature_set_m2(df: pd.DataFrame) -> list[str]:
    return text_feature_cols(df) + graph_feature_cols(df) + ["dataset"]


def feature_set_cvec(df: pd.DataFrame) -> list[str]:
    """Text QCE + full C(Q) — primary graph representation for router R."""
    return text_feature_cols(df) + c_vector_feature_cols(df) + ["dataset"]


def feature_set_graph(df: pd.DataFrame) -> list[str]:
    """C(Q) graph vector + dataset (no text q_*)."""
    cols = c_vector_feature_cols(df)
    if not cols:
        raise ValueError("missing dim_* columns — build complexity_record_*.parquet first")
    return cols + ["dataset"]


def feature_set_graph_emb(df: pd.DataFrame) -> list[str]:
    """C(Q) five-dim vector + query sentence embedding (``emb_*``) + dataset."""
    cols = c_vector_feature_cols(df)
    if not cols:
        raise ValueError("missing dim_* columns — build complexity_record_*.parquet first")
    emb = embedding_feature_cols(df)
    if not emb:
        raise ValueError(
            "missing emb_* columns — add query_embeddings_{split}.parquet "
            "or call load_split(..., with_embeddings=True)"
        )
    return cols + emb + ["dataset"]


def feature_set_scalar(df: pd.DataFrame) -> list[str]:
    """Ablation only: text + ``complexity_graph`` scalar (not for main experiments)."""
    cols = text_feature_cols(df)
    if "complexity_graph" in df.columns and pd.api.types.is_numeric_dtype(df["complexity_graph"]):
        cols.append("complexity_graph")
    return cols + ["dataset"]


def resolve_feature_cols(df: pd.DataFrame, feature_set: FeatureSetId) -> list[str]:
    if feature_set == "m0":
        return feature_set_m0(df)
    if feature_set == "m2":
        return feature_set_m2(df)
    if feature_set == "cvec":
        return feature_set_cvec(df)
    if feature_set == "graph":
        return feature_set_graph(df)
    if feature_set == "graph_emb":
        return feature_set_graph_emb(df)
    if feature_set == "scalar":
        return feature_set_scalar(df)
    raise ValueError(f"unknown feature_set {feature_set!r}")


def make_pipeline(
    feature_cols: list[str],
    *,
    model: str = "hgbm",
    class_weight: str | dict[str, float] | None = "balanced",
    random_state: int = 42,
    hgbm_params: dict[str, Any] | None = None,
    mlp_params: dict[str, Any] | None = None,
    calibrated: bool = False,
    calibration_cv: int = 3,
    calibration_method: Literal["sigmoid", "isotonic"] = "sigmoid",
) -> Pipeline:
    """Sklearn pipeline: impute + one-hot dataset + classifier."""
    num_cols = [c for c in feature_cols if c != "dataset"]
    cat_cols = ["dataset"] if "dataset" in feature_cols else []

    if model == "mlp":
        num_steps: Any = Pipeline(
            [
                ("impute", SimpleImputer(strategy="constant", fill_value=0)),
                ("scale", StandardScaler()),
            ]
        )
    else:
        num_steps = SimpleImputer(strategy="constant", fill_value=0)

    transformers: list[tuple[str, Any, list[str]]] = [
        ("num", num_steps, num_cols),
    ]
    if cat_cols:
        transformers.append(
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        )

    prep = ColumnTransformer(transformers)

    if model == "hgbm":
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
        clf = HistGradientBoostingClassifier(**hgbm_kw)
    elif model == "logistic":
        clf = LogisticRegression(
            max_iter=2000,
            class_weight=class_weight,
            random_state=random_state,
        )
    elif model == "mlp":
        # MLPClassifier has no class_weight; use sample_weight from fit_router instead.
        mlp_kw: dict[str, Any] = {
            "hidden_layer_sizes": (32, 16),
            "activation": "relu",
            "solver": "adam",
            "alpha": 1e-3,
            "batch_size": 64,
            "learning_rate_init": 1e-3,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 20,
            "random_state": random_state,
        }
        if mlp_params:
            mlp_kw.update(mlp_params)
        mlp = MLPClassifier(**mlp_kw)
        clf = LabelEncodingClassifier(mlp)
    else:
        raise ValueError(f"unknown model {model!r}")

    if calibrated:
        clf = CalibratedClassifierCV(
            clf,
            cv=calibration_cv,
            method=calibration_method,
        )

    return Pipeline([("prep", prep), ("clf", clf)])


def oracle_soft_probability(df: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Per-row soft mass on the hard label agent (``p_{oracle_agent}``)."""
    weights = np.ones(len(df), dtype=float)
    for i, (idx, agent) in enumerate(y.items()):
        col = f"p_{agent}"
        if col in df.columns:
            weights[i] = float(df.at[idx, col])
    return weights


def compute_sample_weights(
    df: pd.DataFrame,
    y: pd.Series,
    mode: WeightMode,
) -> np.ndarray | None:
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
    w = w / max(w.mean(), 1e-12)
    return w


def oversample_minorities(
    df: pd.DataFrame,
    target: str,
    *,
    min_fraction: float = 0.35,
    random_state: int = 42,
) -> pd.DataFrame:
    """Random oversample rare ``target`` classes up to ``min_fraction`` of majority count."""
    counts = df[target].value_counts()
    majority = int(counts.max())
    floor_n = max(1, int(majority * min_fraction))
    rng = np.random.default_rng(random_state)
    parts = [df]
    for label, n in counts.items():
        if n >= floor_n:
            continue
        subset = df[df[target] == label]
        need = floor_n - n
        idx = rng.choice(subset.index.to_numpy(), size=need, replace=True)
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
    X = df[feature_cols]
    y = df[target].astype(str)
    if sample_weight is not None:
        pipe.fit(X, y, clf__sample_weight=sample_weight)
    else:
        pipe.fit(X, y)
    return pipe


@dataclass(frozen=True)
class TrainSpec:
    experiment_id: str
    feature_set: FeatureSetId = "m0"
    model: str = "hgbm"
    weight_mode: WeightMode = "balanced"
    oversample: bool = False
    target: str = TARGET
    # When using explicit sample weights, disable sklearn class_weight to avoid double correction.
    use_class_weight: bool = True

    @property
    def effective_class_weight(self) -> str | None:
        if self.use_class_weight and self.weight_mode in ("none", "balanced"):
            return "balanced"
        return None


@dataclass(frozen=True)
class EvalResult:
    accuracy: float
    macro_f1: float
    report: str
    predictions: np.ndarray


def predict_agent_proba(
    pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
) -> pd.DataFrame:
    """(n, 4) probabilities aligned to ``p_raw`` … ``p_multiagent`` column order."""
    clf = pipe.named_steps["clf"]
    proba = pipe.predict_proba(df[feature_cols])
    classes = [str(c) for c in clf.classes_]
    out = pd.DataFrame(proba, columns=[f"p_{c}" for c in classes], index=df.index)
    for a in AGENTS:
        col = f"p_{a}"
        if col not in out.columns:
            out[col] = 0.0
    return out[PROBA_COLS]


def evaluate(
    pipe: Pipeline,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = TARGET,
) -> EvalResult:
    y = df[target].astype(str)
    pred = pipe.predict(df[feature_cols])
    labels = list(AGENTS)
    acc = float(accuracy_score(y, pred))
    f1 = float(
        f1_score(y, pred, average="macro", labels=labels, zero_division=0)
    )
    rep = classification_report(
        y, pred, labels=labels, zero_division=0
    )
    return EvalResult(acc, f1, rep, pred)


def baseline_always_majority(train_df: pd.DataFrame, eval_df: pd.DataFrame) -> EvalResult:
    majority = train_df[TARGET].mode()[0]
    pred = np.array([majority] * len(eval_df))
    y = eval_df[TARGET].astype(str)
    labels = list(AGENTS)
    return EvalResult(
        float(accuracy_score(y, pred)),
        float(
            f1_score(y, pred, average="macro", labels=labels, zero_division=0)
        ),
        classification_report(y, pred, labels=labels, zero_division=0),
        pred,
    )


def baseline_per_dataset_mode(
    train_df: pd.DataFrame, eval_df: pd.DataFrame
) -> EvalResult:
    mode = train_df.groupby("dataset")[TARGET].agg(lambda s: s.mode()[0])
    fallback = train_df[TARGET].mode()[0]
    pred = eval_df["dataset"].map(mode).fillna(fallback).astype(str).values
    y = eval_df[TARGET].astype(str)
    labels = list(AGENTS)
    return EvalResult(
        float(accuracy_score(y, pred)),
        float(
            f1_score(y, pred, average="macro", labels=labels, zero_division=0)
        ),
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
    X = df[feature_cols]
    y = df[target].astype(str)
    params = {}
    if sample_weight is not None:
        params = {"clf__sample_weight": sample_weight}
    scores = cross_val_score(
        pipe,
        X,
        y,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        params=params,
    )
    return float(scores.mean()), float(scores.std())


def train_and_evaluate(
    spec: TrainSpec,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> tuple[Pipeline, list[str], EvalResult, tuple[float, float]]:
    feature_cols = resolve_feature_cols(train_df, spec.feature_set)
    fit_df = (
        oversample_minorities(train_df, spec.target)
        if spec.oversample
        else train_df
    )
    y_fit = fit_df[spec.target].astype(str)
    weights = compute_sample_weights(fit_df, y_fit, spec.weight_mode)
    pipe = make_pipeline(
        feature_cols,
        model=spec.model,
        class_weight=spec.effective_class_weight,
    )
    fit_router(
        pipe,
        fit_df,
        feature_cols,
        target=spec.target,
        sample_weight=weights,
    )
    val_result = evaluate(pipe, val_df, feature_cols, target=spec.target)
    cv_mean, cv_std = cross_val_macro_f1(
        make_pipeline(
            feature_cols,
            model=spec.model,
            class_weight=spec.effective_class_weight,
        ),
        train_df,
        feature_cols,
        target=spec.target,
        sample_weight=compute_sample_weights(
            train_df, train_df[spec.target].astype(str), spec.weight_mode
        ),
    )
    return pipe, feature_cols, val_result, (cv_mean, cv_std)


def results_row(
    name: str,
    result: EvalResult,
    *,
    feature_set: str | None = None,
    n_features: int | None = None,
    weight_mode: str | None = None,
    model: str | None = None,
    oversample: bool = False,
    train_cv_macro_f1: float | None = None,
    train_cv_std: float | None = None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "experiment": name,
        "accuracy": round(result.accuracy, 4),
        "macro_f1": round(result.macro_f1, 4),
    }
    if model is not None:
        row["model"] = model
    if feature_set is not None:
        row["feature_set"] = feature_set
    if n_features is not None:
        row["n_features"] = n_features
    if weight_mode is not None:
        row["weight_mode"] = weight_mode
    if oversample:
        row["oversample"] = True
    if train_cv_macro_f1 is not None:
        row["train_cv_macro_f1"] = round(train_cv_macro_f1, 4)
    if train_cv_std is not None:
        row["train_cv_std"] = round(train_cv_std, 4)
    return row


def mutual_info_report(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = TARGET,
    top_k: int = 12,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rank numeric features by mutual information with ``target`` (learnability probe)."""
    num_cols = [c for c in feature_cols if c != "dataset"]
    X = df[num_cols].fillna(0).to_numpy()
    y = LabelEncoder().fit_transform(df[target].astype(str))
    mi = mutual_info_classif(X, y, random_state=random_state)
    out = pd.DataFrame({"feature": num_cols, "mutual_info": mi})
    return out.sort_values("mutual_info", ascending=False).head(top_k)


def learnability_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    experiment_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize whether ``oracle_agent`` looks learnable from current features."""
    exp_df = pd.DataFrame(experiment_rows)
    ml = exp_df[exp_df["experiment"].str.match(r"^(B[123]|G\d+)_")]
    best = ml.loc[ml["macro_f1"].idxmax()] if len(ml) else None
    b0_mode = exp_df[exp_df["experiment"] == "B0_dataset_mode"].iloc[0]
    b0_raw = exp_df[exp_df["experiment"] == "B0_always_raw"].iloc[0]

    soft = oracle_soft_probability(train_df, train_df[TARGET].astype(str))
    oracle_eq_soft = float(
        (train_df[TARGET] == train_df[SOFT_DOMINANT]).mean()
    ) if SOFT_DOMINANT in train_df.columns else None

    summary: dict[str, Any] = {
        "n_train": len(train_df),
        "n_val": len(val_df),
        "train_class_counts": train_df[TARGET].value_counts().to_dict(),
        "val_class_counts": val_df[TARGET].value_counts().to_dict(),
        "oracle_equals_soft_dominant_frac": oracle_eq_soft,
        "soft_p_oracle_mean": float(np.mean(soft)),
        "soft_p_oracle_lt_0_25_frac": float(np.mean(soft < 0.25)),
        "baseline_val_macro_f1_always_raw": float(b0_raw["macro_f1"]),
        "baseline_val_macro_f1_dataset_mode": float(b0_mode["macro_f1"]),
        "baseline_val_accuracy_dataset_mode": float(b0_mode["accuracy"]),
    }
    if best is not None:
        summary["best_ml_experiment"] = best["experiment"]
        summary["best_ml_val_macro_f1"] = float(best["macro_f1"])
        summary["best_ml_val_accuracy"] = float(best["accuracy"])
        summary["lift_macro_f1_vs_dataset_mode"] = round(
            float(best["macro_f1"]) - float(b0_mode["macro_f1"]), 4
        )
        summary["beats_dataset_mode_macro_f1"] = bool(
            best["macro_f1"] > b0_mode["macro_f1"]
        )
    return summary


GRAPH_ROUTER_SPECS: tuple[TrainSpec, ...] = (
    TrainSpec(
        "G1_hgbm_graph_balanced",
        feature_set="graph",
        model="hgbm",
        weight_mode="balanced",
    ),
    TrainSpec(
        "G2_hgbm_graph_balanced_soft",
        feature_set="graph",
        model="hgbm",
        weight_mode="balanced_soft",
    ),
    TrainSpec(
        "G2_mlp_graph_balanced_soft",
        feature_set="graph",
        model="mlp",
        weight_mode="balanced_soft",
        use_class_weight=False,
    ),
)

GRAPH_EMB_ROUTER_SPECS: tuple[TrainSpec, ...] = (
    TrainSpec(
        "G3_hgbm_graph_emb_balanced",
        feature_set="graph_emb",
        model="hgbm",
        weight_mode="balanced",
    ),
    TrainSpec(
        "G4_hgbm_graph_emb_balanced_soft",
        feature_set="graph_emb",
        model="hgbm",
        weight_mode="balanced_soft",
    ),
    TrainSpec(
        "G5_mlp_graph_emb_balanced_soft",
        feature_set="graph_emb",
        model="mlp",
        weight_mode="balanced_soft",
        use_class_weight=False,
    ),
)


def graph_router_specs(*, with_embeddings: bool) -> tuple[TrainSpec, ...]:
    if with_embeddings:
        return GRAPH_ROUTER_SPECS + GRAPH_EMB_ROUTER_SPECS
    return GRAPH_ROUTER_SPECS

DEFAULT_SPECS: tuple[TrainSpec, ...] = (
    *GRAPH_ROUTER_SPECS,
    TrainSpec("B1_hgbm_text", feature_set="m0", weight_mode="balanced"),
    TrainSpec("B4_hgbm_cvec", feature_set="cvec", weight_mode="balanced"),
    TrainSpec("B2_hgbm_fusion", feature_set="m2", weight_mode="balanced"),
    TrainSpec(
        "B1_hgbm_text_soft_w",
        feature_set="m0",
        weight_mode="balanced_soft",
        use_class_weight=False,
    ),
    TrainSpec(
        "B2_hgbm_fusion_soft_w",
        feature_set="m2",
        weight_mode="balanced_soft",
        use_class_weight=False,
    ),
    TrainSpec(
        "B2_hgbm_fusion_oversample",
        feature_set="m2",
        weight_mode="balanced",
        oversample=True,
    ),
    TrainSpec(
        "B2_hgbm_fusion_full",
        feature_set="m2",
        weight_mode="balanced_soft",
        oversample=True,
        use_class_weight=False,
    ),
    TrainSpec(
        "B3_logistic_fusion",
        feature_set="m2",
        model="logistic",
        weight_mode="balanced_soft",
        use_class_weight=False,
    ),
)


def run_experiments(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    specs: tuple[TrainSpec, ...] = DEFAULT_SPECS,
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, Pipeline], dict[str, list[str]], EvalResult | None]:
    rows: list[dict[str, Any]] = [
        results_row("B0_always_raw", baseline_always_majority(train_df, val_df)),
        results_row("B0_dataset_mode", baseline_per_dataset_mode(train_df, val_df)),
    ]
    fitted: dict[str, Pipeline] = {}
    feature_map: dict[str, list[str]] = {}
    best_name: str | None = None
    best_f1 = -1.0
    best_detail: EvalResult | None = None

    for spec in specs:
        pipe, feature_cols, val_res, (cv_mean, cv_std) = train_and_evaluate(
            spec, train_df, val_df
        )
        fitted[spec.experiment_id] = pipe
        feature_map[spec.experiment_id] = feature_cols
        rows.append(
            results_row(
                spec.experiment_id,
                val_res,
                feature_set=spec.feature_set,
                n_features=len(feature_cols),
                model=spec.model,
                weight_mode=spec.weight_mode,
                oversample=spec.oversample,
                train_cv_macro_f1=cv_mean,
                train_cv_std=cv_std,
            )
        )
        if val_res.macro_f1 > best_f1:
            best_f1 = val_res.macro_f1
            best_name = spec.experiment_id
            best_detail = val_res

        if verbose:
            print(
                f"\n=== {spec.experiment_id} ===\n"
                f"features ({spec.feature_set}): {len(feature_cols)} cols | "
                f"model={spec.model} weights={spec.weight_mode} oversample={spec.oversample}\n"
                f"Train 5-fold macro-F1: {cv_mean:.4f} +/- {cv_std:.4f}\n"
                f"Val accuracy: {val_res.accuracy:.4f} | Val macro-F1: {val_res.macro_f1:.4f}\n"
                f"{val_res.report}"
            )

    table = pd.DataFrame(rows)
    if verbose:
        print("\n--- Val comparison ---")
        print(table.to_string(index=False))
        summary = learnability_summary(train_df, val_df, rows)
        print("\n--- Learnability summary ---")
        print(json.dumps(summary, indent=2))
        mi = mutual_info_report(train_df, resolve_feature_cols(train_df, "m2"))
        print("\n--- Top mutual information (M2 numeric features) ---")
        print(mi.to_string(index=False))

    return table, fitted, feature_map, best_detail if best_name else None


def _register_label_encoding_for_unpickle() -> None:
    """Artifacts trained via ``python routing/train_router.py`` pickle ``__main__.LabelEncodingClassifier``."""
    import sys
    import types

    cls = LabelEncodingClassifier
    targets = ("__main__", "train_router", "routing.train_router")
    for name in targets:
        mod = sys.modules.get(name)
        if mod is None and name != "__main__":
            mod = types.ModuleType(name)
            sys.modules[name] = mod
        if mod is not None:
            setattr(mod, "LabelEncodingClassifier", cls)


def load_router(path: str | Path) -> dict[str, Any]:
    """Load artifact written by ``save_router``."""
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
    joblib.dump(
        {
            "pipeline": pipe,
            "feature_cols": feature_cols,
            "target": TARGET,
            "agents": list(AGENTS),
            "experiment_id": experiment_id,
        },
        path,
    )
    meta_body: dict[str, Any] = {
        "experiment_id": experiment_id,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
    }
    if extra_meta:
        meta_body.update(extra_meta)
    meta = path.with_suffix(".json")
    meta.write_text(json.dumps(meta_body, indent=2), encoding="utf-8")
    return path


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train agent router experiments.")
    p.add_argument(
        "--graph-only",
        action="store_true",
        help="Train G* graph routers (HGBM + MLP): C(Q) (+ emb_* if parquet present).",
    )
    p.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Do not merge query_embeddings_*.parquet even if files exist.",
    )
    p.add_argument(
        "--save-best",
        action="store_true",
        help="Persist val macro-F1 best ML model under models/router/.",
    )
    p.add_argument(
        "--save-all",
        action="store_true",
        help="Persist every trained experiment under models/router/.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Only print summary table and learnability JSON.",
    )
    return p.parse_args()


if __name__ == "__main__":
    import sys

    # Allow ``python routing/train_router.py`` and stable pickles for MLP wrappers.
    sys.modules.setdefault("__main__", sys.modules[__name__])
    setattr(sys.modules["__main__"], "LabelEncodingClassifier", LabelEncodingClassifier)

    args = _parse_args()
    use_emb = not args.no_embeddings
    train_df = load_split("train", with_embeddings=use_emb)
    val_df = load_split("val", with_embeddings=use_emb)
    if args.graph_only:
        specs = graph_router_specs(with_embeddings=bool(embedding_feature_cols(train_df)))
    else:
        specs = DEFAULT_SPECS

    table, fitted, feature_map, _best = run_experiments(
        train_df,
        val_df,
        specs=specs,
        verbose=not args.quiet,
    )

    if not args.quiet and args.graph_only:
        for exp_id in fitted:
            proba = predict_agent_proba(fitted[exp_id], val_df, feature_map[exp_id])
            print(f"\n--- {exp_id} val predict_proba (head) ---")
            print(proba.head(3).to_string())

    to_save = table
    if args.graph_only:
        to_save = table[table["experiment"].str.match(r"^G\d+_")]

    if args.save_all:
        for _, row in to_save.iterrows():
            exp_id = str(row["experiment"])
            save_router(
                fitted[exp_id],
                feature_cols=feature_map[exp_id],
                experiment_id=exp_id,
                extra_meta={
                    "feature_set": row.get("feature_set"),
                    "weight_mode": row.get("weight_mode"),
                    "val_macro_f1": float(row["macro_f1"]),
                    "val_accuracy": float(row["accuracy"]),
                    "output": "predict_proba",
                    "proba_cols": list(PROBA_COLS),
                },
            )

    if args.save_best and len(to_save):
        best_row = to_save.loc[to_save["macro_f1"].idxmax()]
        exp_id = str(best_row["experiment"])
        out = save_router(
            fitted[exp_id],
            feature_cols=feature_map[exp_id],
            experiment_id=exp_id,
            extra_meta={
                "feature_set": best_row.get("feature_set"),
                "weight_mode": best_row.get("weight_mode"),
                "val_macro_f1": float(best_row["macro_f1"]),
                "val_accuracy": float(best_row["accuracy"]),
                "output": "predict_proba",
                "proba_cols": list(PROBA_COLS),
            },
        )
        print("Saved best", out)
