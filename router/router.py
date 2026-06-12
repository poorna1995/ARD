"""
QCE router — feature load, HGBM train/infer, runtime execution.

Single module (consolidated). CLI/ablations: ``routing.experiments``.
"""

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

from config.common import REPO_ROOT
from config.global_config.paths import (
    complexity_record_path,
    embeddings_parquet_path,
    heuristics_parquet_path,
)
from config.global_config.runtime import resolve_router_path
from router.config import (
    AGENTS,
    DEFAULT_AGENT_MODEL,
    DEFAULT_HGBM_PARAMS,
    EMBEDDING_COL_PREFIX,
    EVAL_SAMPLES_DIR,
    FeatureSet,
    HEURISTIC_COL_PREFIX,
    PROBA_COLS,
    PRODUCTION_FEATURE_SET,
    PRODUCTION_ROUTER_EXPERIMENT_ID,
    REPO_ROOT,
    ROUTER_EXPERIMENT_ID,
    ROUTER_MODEL_PATH,
    SPLIT_CSV,
    SPLIT_PARQUET,
    TARGET,
    feature_set_needs_embeddings,
    feature_set_needs_heuristics,
    feature_set_spec,
    normalize_feature_set,
)
from config.local.constants.datasets import eval_parquet_stem
from router.config import resolve_dataset_name
from qce.features import (
    build_eval_features,
    decompose_cache_status,
    ensure_eval_features,
    feature_paths,
    load_decompose_cache,
    merge_feature_tables,
    plans_from_cache,
)

from config.local.constants import DIMS, TRUST_COLS

try:
    from qce.complexity import C_VECTOR_COLS as _C_VECTOR_COLS
    from qce.complexity import TRUST_SCALAR_COLS as _TRUST_SCALAR_COLS
except ImportError:
    _C_VECTOR_COLS = DIMS.main
    _TRUST_SCALAR_COLS = TRUST_COLS

WeightMode = Literal["none", "balanced", "soft", "balanced_soft"]


# --- column aliases ---
def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "training_id" in df.columns:
        out = df.copy()
        out["training_id"] = out["training_id"].astype(str)
        return out
    for col in ("query_id", "id"):
        if col in df.columns:
            out = df.copy()
            out["training_id"] = out[col].astype(str)
            return out
    return df


def normalize_router_proba_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for agent in AGENTS:
        p_col = f"p_{agent}"
        prob_col = f"prob_{agent}"
        if p_col not in out.columns and prob_col in out.columns:
            out[p_col] = out[prob_col]
    return out


def normalize_loader_frame(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_router_proba_columns(normalize_id_column(df))


# --- feature columns ---
def query_embeddings_path(split: str, *, root: Path | None = None) -> Path:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    if root is not None:
        rel = embeddings_parquet_path(split).relative_to(REPO_ROOT)
        return Path(root) / rel
    return embeddings_parquet_path(split)


def query_heuristics_path(split: str, *, root: Path | None = None) -> Path:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    if root is not None:
        rel = heuristics_parquet_path(split).relative_to(REPO_ROOT)
        return Path(root) / rel
    return heuristics_parquet_path(split)


def embedding_feature_cols(df: pd.DataFrame) -> list[str]:
    cols = [
        c
        for c in df.columns
        if c.startswith(EMBEDDING_COL_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
    ]

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len(EMBEDDING_COL_PREFIX) :]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(cols, key=_key)


def load_query_embeddings(split: str, *, root: Path | None = None) -> pd.DataFrame:
    path = query_embeddings_path(split, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"missing query embeddings: {path}")
    emb = pd.read_parquet(path)
    if "training_id" not in emb.columns:
        raise ValueError(f"{path}: missing training_id")
    return emb[["training_id", *embedding_feature_cols(emb)]].copy()


def heuristic_feature_cols(df: pd.DataFrame) -> list[str]:
    """Return ``heur_*`` columns in canonical build order when complete."""
    present = [
        c
        for c in df.columns
        if c.startswith(HEURISTIC_COL_PREFIX) and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not present:
        return []

    try:
        from router.config import HEUR_ROUTER_COLS

        ordered = [c for c in HEUR_ROUTER_COLS if c in df.columns]
        if len(ordered) == len(HEUR_ROUTER_COLS):
            return ordered
        missing = [c for c in HEUR_ROUTER_COLS if c not in df.columns]
        raise ValueError(
            f"heuristic parquet missing {len(missing)} columns (e.g. {missing[:3]}); "
            "re-run scripts/build_query_heuristics.py --split all"
        )
    except ValueError:
        raise
    except Exception:
        pass

    def _key(name: str) -> tuple[int, str]:
        suffix = name[len(HEURISTIC_COL_PREFIX) :]
        return (int(suffix), name) if suffix.isdigit() else (10**9, name)

    return sorted(present, key=_key)


def load_query_heuristics(split: str, *, root: Path | None = None) -> pd.DataFrame:
    path = query_heuristics_path(split, root=root)
    if not path.is_file():
        raise FileNotFoundError(f"missing query heuristics: {path}")
    heur = pd.read_parquet(path)
    if "training_id" not in heur.columns:
        raise ValueError(f"{path}: missing training_id")
    return heur[["training_id", *heuristic_feature_cols(heur)]].copy()


def c_vector_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in _C_VECTOR_COLS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]


def trust_scalar_feature_cols(df: pd.DataFrame) -> list[str]:
    return [
        c
        for c in _TRUST_SCALAR_COLS
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]


def router_feature_cols(
    df: pd.DataFrame, feature_set: FeatureSet = PRODUCTION_FEATURE_SET
) -> list[str]:
    """Resolve HGBM input columns for a feature-set id (see ``FEATURE_SET_SPECS``)."""
    spec = feature_set_spec(normalize_feature_set(feature_set))
    cols: list[str] = []

    if spec.cvec_version is not None:
        graph = c_vector_feature_cols(df)
        if not graph:
            raise ValueError("missing dim_* — rebuild complexity_record_*.parquet")
        cols.extend(graph)

    if spec.use_emb:
        emb = embedding_feature_cols(df)
        if not emb:
            raise ValueError("missing emb_* — run scripts/build_query_embeddings.py --split all")
        cols.extend(emb)

    if spec.use_trust:
        trust = trust_scalar_feature_cols(df)
        if not trust:
            raise ValueError(
                f"missing trust scalars {_TRUST_SCALAR_COLS} — rebuild complexity_record_*.parquet"
            )
        cols.extend(trust)

    if spec.use_heur:
        heur = heuristic_feature_cols(df)
        if not heur:
            raise ValueError(
                "missing heur_* — run: uv run python scripts/build_query_heuristics.py --split all"
            )
        cols.extend(heur)

    if spec.use_dataset:
        if "dataset" not in df.columns:
            raise ValueError("missing dataset column")
        cols.append("dataset")

    return cols


def validate_feature_set_data(df: pd.DataFrame, feature_set: str) -> None:
    """Ensure parquet columns exist for ``feature_set`` (call after ``load_split``)."""
    router_feature_cols(df, normalize_feature_set(feature_set))


resolve_feature_cols = router_feature_cols


def overall_complexity(row: pd.Series | dict[str, Any]) -> float | None:
    vals: list[float] = []
    for c in _C_VECTOR_COLS:
        v = row.get(c) if isinstance(row, dict) else row.get(c, None)
        if v is not None and pd.notna(v):
            try:
                vals.append(float(v))
            except (TypeError, ValueError):
                pass
    return max(vals) if vals else None
# --- data loaders ---
def _path_with_root(path: Path, root: Path | None) -> Path:
    if root is None or Path(root) == REPO_ROOT:
        return path
    return Path(root) / path.relative_to(REPO_ROOT)


def load_split(
    split: str,
    *,
    root: Path | None = None,
    with_embeddings: bool | None = None,
    with_heuristics: bool | None = None,
    embeddings_dir: Path | None = None,
) -> pd.DataFrame:
    if split not in SPLIT_CSV:
        raise ValueError(f"split must be one of {list(SPLIT_CSV)}")
    root = Path(root or REPO_ROOT)
    labels = pd.read_csv(_path_with_root(SPLIT_CSV[split], root))
    feats = pd.read_parquet(_path_with_root(complexity_record_path(split), root))
    drop = [c for c in feats.columns if c in labels.columns and c != "training_id"]
    merged = labels.merge(
        feats.drop(columns=[c for c in drop if c in feats.columns], errors="ignore"),
        on="training_id",
        how="inner",
    )
    if embeddings_dir is not None:
        emb_path = Path(embeddings_dir) / embeddings_parquet_path(split).name
    else:
        emb_path = query_embeddings_path(split, root=root)
    if with_embeddings is None:
        with_embeddings = emb_path.is_file()
    if with_embeddings:
        if embeddings_dir is not None:
            if not emb_path.is_file():
                raise FileNotFoundError(f"missing query embeddings: {emb_path}")
            emb = pd.read_parquet(emb_path)
            if "training_id" not in emb.columns:
                raise ValueError(f"{emb_path}: missing training_id")
            emb = emb[["training_id", *embedding_feature_cols(emb)]].copy()
        else:
            emb = load_query_embeddings(split, root=root)
        drop_emb = [c for c in emb.columns if c in merged.columns and c != "training_id"]
        merged = merged.merge(
            emb.drop(columns=[c for c in drop_emb if c in emb.columns], errors="ignore"),
            on="training_id",
            how="inner",
        )
    heur_path = query_heuristics_path(split, root=root)
    if with_heuristics is None:
        with_heuristics = heur_path.is_file()
    if with_heuristics:
        heur = load_query_heuristics(split, root=root)
        drop_h = [c for c in heur.columns if c in merged.columns and c != "training_id"]
        merged = merged.merge(
            heur.drop(columns=[c for c in drop_h if c in heur.columns], errors="ignore"),
            on="training_id",
            how="inner",
        )
    merged = merged[merged[TARGET].isin(AGENTS)].copy()
    if merged.empty:
        raise ValueError(f"{split}: no rows after filtering to {AGENTS}")
    return normalize_loader_frame(merged.reset_index(drop=True))


def load_split_for_feature_set(
    split: str,
    feature_set: str,
    *,
    root: Path | None = None,
    embeddings_dir: Path | None = None,
) -> pd.DataFrame:
    """Load labels + complexity + optional emb/heur columns required by ``feature_set``."""
    fs = normalize_feature_set(feature_set)
    return load_split(
        split,
        root=root,
        with_embeddings=feature_set_needs_embeddings(fs),
        with_heuristics=feature_set_needs_heuristics(fs),
        embeddings_dir=embeddings_dir,
    )


def eval_base_frame(parquet_path: Path, dataset: str | None = None) -> pd.DataFrame:
    path = Path(parquet_path)
    df = normalize_loader_frame(pd.read_parquet(path))
    if "query" not in df.columns:
        raise ValueError(f"{path}: missing 'query'")
    id_col = "training_id" if "training_id" in df.columns else "id"
    if id_col not in df.columns:
        raise ValueError(f"{path}: need id or training_id")
    ds = resolve_dataset_name(dataset or path.stem)
    out = pd.DataFrame(
        {
            "training_id": df[id_col].astype(str),
            "query": df["query"].astype(str),
            "dataset": ds,
            "expected_answer": df["answer"].astype(str) if "answer" in df.columns else "",
        }
    )
    for col in ("level", "file_name", "split", "n_hops", "hop_name", "type", "context", "metadata", "paragraphs"):
        if col in df.columns:
            out[col] = df[col]
    return out


def load_eval_parquet(dataset: str, *, data_path: Path | str | None = None) -> pd.DataFrame:
    canonical = resolve_dataset_name(dataset)
    if data_path:
        path = Path(data_path)
    else:
        stem = eval_parquet_stem(canonical)
        path = EVAL_SAMPLES_DIR / f"{stem}.parquet"
        if not path.is_file() and stem != canonical:
            path = EVAL_SAMPLES_DIR / f"{canonical}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Eval parquet not found: {path}")
    return eval_base_frame(path, dataset=canonical)
# --- training ---
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
# --- inference ---
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


def load_router_frame(
    df: pd.DataFrame,
    router_path: Path | str | None = None,
    *,
    feature_tag: str | None = None,
) -> pd.DataFrame:
    df = normalize_loader_frame(df)
    path = resolve_router_path(ROUTER_MODEL_PATH) if router_path is None else Path(router_path)
    obj = load_router(path)
    feature_cols: list[str] = list(obj["feature_cols"])
    missing = [c for c in feature_cols if c not in df.columns]
    if missing and any(c.startswith("heur_") for c in missing):
        from qce.features.build import ensure_query_heuristics

        df = ensure_query_heuristics(df, feature_tag)
        missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Router needs missing columns: {missing}")
    out = attach_router_predictions(
        df, obj["pipeline"], feature_cols, experiment_id=str(obj.get("experiment_id", path.stem))
    )
    out["overall"] = out.apply(overall_complexity, axis=1)
    out["assigned_agent"] = out["router_pred"]
    return out
# --- runtime ---
class RuntimeRouter:
    """Per-query router + optional agent execution."""

    def __init__(
        self,
        query: str,
        *,
        feature_row: pd.Series | dict[str, Any],
        dataset: str,
        model: str = DEFAULT_AGENT_MODEL,
        **run_kw: Any,
    ) -> None:
        from agent.registry import run_strategy

        self._run_strategy = run_strategy
        self.query = query.strip()
        self.dataset = resolve_dataset_name(dataset)
        self.model = model
        self.run_kw = dict(run_kw)
        row = dict(feature_row)
        self._row = row
        self.assigned_agent = str(row.get("assigned_agent") or row.get("router_pred") or "")
        self.overall = row.get("overall") if row.get("overall") is not None else overall_complexity(row)

    @classmethod
    def from_dataframe_row(
        cls, row: pd.Series | dict[str, Any], *, model: str = DEFAULT_AGENT_MODEL, **run_kw: Any
    ) -> RuntimeRouter:
        return cls(
            str(row.get("query") or "").strip(),
            feature_row=row,
            dataset=str(row.get("dataset") or "gaia"),
            model=model,
            **run_kw,
        )

    def inspect(self) -> dict[str, Any]:
        row = self._row
        out: dict[str, Any] = {
            "assigned_agent": self.assigned_agent,
            "router_pred": self.assigned_agent,
            "overall": self.overall,
            "model": self.model,
            "max_prob": row.get("max_prob"),
            "margin_top2": row.get("margin_top2"),
            "router_second": row.get("router_second"),
        }
        for key in PROBA_COLS:
            if key in row:
                out[key] = row[key]
        if "p_raw" in row:
            out["top3_agents"] = top_k_agents(pd.Series({k: row[k] for k in PROBA_COLS if k in row}), k=3)
        return out

    def run(self, *, expected_answer: Any = None) -> dict[str, Any]:
        if not self.assigned_agent:
            raise ValueError("No assigned_agent; run batch routing first.")
        response = self._run_strategy(
            self.assigned_agent,
            self.query,
            model=self.model,
            dataset=self.dataset,
            expected_answer=expected_answer,
            **self.run_kw,
        )
        payload = asdict(response) if hasattr(response, "__dataclass_fields__") else response
        return {"agent": self.assigned_agent, "model": self.model, "response": payload}

    def run_cascade(self, *, expected_answer: Any = None, k: int = 3) -> dict[str, Any]:
        """Run top-``k`` agents in order until one does not fail."""
        return run_agent_cascade(
            self._row,
            query=self.query,
            dataset=self.dataset,
            model=self.model,
            k=k,
            expected_answer=expected_answer,
            **self.run_kw,
        )
__all__ = [
    "AGENTS",
    "PROBA_COLS",
    "PRODUCTION_FEATURE_SET",
    "ROUTER_MODEL_PATH",
    "TARGET",
    "RuntimeRouter",
    "TrainSpec",
    "attach_router_predictions",
    "build_eval_features",
    "ensure_eval_features",
    "evaluate",
    "load_eval_parquet",
    "load_router",
    "load_router_frame",
    "load_split",
    "load_split_for_feature_set",
    "normalize_feature_set",
    "normalize_loader_frame",
    "plans_from_cache",
    "predict_agent_proba",
    "resolve_dataset_name",
    "router_feature_cols",
    "run_agent_cascade",
    "save_router",
    "top_k_from_row",
    "train_router",
    "validate_feature_set_data",
]
