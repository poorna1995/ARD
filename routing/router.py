"""
QCE router — HGBM over complexity vectors + query embeddings.

CLI (``python -m routing``)::

  uv run python -m routing train --feature-set cvec5_emb --save   # legacy hard HGBM only
  uv run python -m routing soft-train --classifier hgbm --save   # primary (soft KL)
  uv run python -m routing soft-train --classifier logreg --save # secondary (soft KL)
  uv run python -m routing tune --save                              # CV-tuned comparison
  uv run python -m routing seed-sweep                             # default vs tuned check
  uv run python -m routing eval --split test

Configuration lives in ``routing.config`` (paths, feature sets, production freeze).
"""

from __future__ import annotations

import argparse
import itertools
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

from routing.config import (
    AGENTS,
    CVEC5_ABLATION_CASES,
    CVEC7_ABLATION_CASES,
    DATASET_ALIASES,
    DECOMPOSER_CACHE_DIR,
    DEFAULT_AGENT_MODEL,
    DEFAULT_HGBM_PARAMS,
    EMBEDDING_COL_PREFIX,
    EMBEDDINGS_PARQUET,
    EVAL_SAMPLES_DIR,
    FEATURE_SET_CLI_CHOICES,
    GRAPH_MAIN_DIM_EMB_MODEL_PATH,
    GRAPH_MAIN_EXPERIMENT_ID,
    GRAPH_MAIN_MODEL_PATH,
    HGBM_GRID,
    PCA_PATH,
    PROBA_COLS,
    PRODUCTION_FEATURE_SET,
    PRODUCTION_HGBM_PARAMS,
    PRODUCTION_ROUTER_EXPERIMENT_ID,
    PRODUCTION_ROUTER_PATH,
    QCE_DATASETS,
    QCE_FEATURES_DIR,
    REPO_ROOT,
    ROUTER_EXPERIMENT_ID,
    ROUTER_MODEL_PATH,
    ROUTER_V2_EXPERIMENT_ID,
    SEED_STABILITY_DIR,
    SOFT_DOMINANT,
    HEURISTICS_PARQUET,
    HEURISTIC_COL_PREFIX,
    SPLIT_CSV,
    SPLIT_PARQUET,
    feature_set_needs_embeddings,
    feature_set_needs_heuristics,
    TARGET,
    TRAIN_NORM_JSON,
    TRUST_ABLATION_CASES,
    TUNE_OUT_DIR,
    TUNED_EXPERIMENT_ID,
    TUNED_HGBM_PARAMS,
    TUNED_MODEL_PATH,
    TUNED_ROUTER_EXPERIMENT_ID,
    TUNED_ROUTER_PATH,
    AblationCase,
    FeatureSet,
    experiment_id_for_feature_set,
    feature_set_cvec_version,
    feature_set_needs_embeddings,
    feature_set_spec,
    is_production_feature_set,
    model_path_for_feature_set,
    normalize_feature_set,
)

# Re-export config symbols for ``from routing.router import …`` callers.
__all__ = [
    "AGENTS",
    "PROBA_COLS",
    "PRODUCTION_FEATURE_SET",
    "PRODUCTION_ROUTER_PATH",
    "ROUTER_MODEL_PATH",
    "TARGET",
    "TUNED_ROUTER_PATH",
    "load_router",
    "load_router_frame",
    "train_router",
]

try:
    from qce.complexity import C_VECTOR_COLS as _C_VECTOR_COLS
    from qce.complexity import TRUST_SCALAR_COLS as _TRUST_SCALAR_COLS
except ImportError:
    _C_VECTOR_COLS = (
        "dim_structural",
        "dim_reasoning",
        "dim_evidence",
        "dim_tool",
        "dim_coordination_uncertainty",
    )
    _TRUST_SCALAR_COLS = (
        "plan_trust",
        "verify_fraction",
        "terminal_sink_ok",
        "sink_intermediate_risk",
    )
_C_VECTOR_COLS_V7 = (
    "dim7_structural",
    "dim7_compositional",
    "dim7_retrieval",
    "dim7_execution",
    "dim7_coordination",
    "dim7_verification",
    "dim7_uncertainty",
)

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


# ── QCE split data ───────────────────────────────────────────────────────────


def query_embeddings_path(split: str, *, root: Path | None = None) -> Path:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    return Path(root or REPO_ROOT) / EMBEDDINGS_PARQUET.format(split=SPLIT_PARQUET[split])


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


def query_heuristics_path(split: str, *, root: Path | None = None) -> Path:
    if split not in SPLIT_PARQUET:
        raise ValueError(f"split must be one of {list(SPLIT_PARQUET)}")
    return Path(root or REPO_ROOT) / HEURISTICS_PARQUET.format(split=SPLIT_PARQUET[split])


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
        import sys

        old = REPO_ROOT / "old"
        if str(old) not in sys.path:
            sys.path.insert(0, str(old))
        from difficulty.feature_measure import HEUR_ROUTER_COLS

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


def load_split(
    split: str,
    *,
    root: Path | None = None,
    with_embeddings: bool | None = None,
    with_heuristics: bool | None = None,
) -> pd.DataFrame:
    if split not in SPLIT_CSV:
        raise ValueError(f"split must be one of {list(SPLIT_CSV)}")
    root = Path(root or REPO_ROOT)
    labels = pd.read_csv(root / SPLIT_CSV[split])
    feats = pd.read_parquet(
        root / "datasets/qce_features" / f"complexity_record_{SPLIT_PARQUET[split]}.parquet"
    )
    drop = [c for c in feats.columns if c in labels.columns and c != "training_id"]
    merged = labels.merge(
        feats.drop(columns=[c for c in drop if c in feats.columns], errors="ignore"),
        on="training_id",
        how="inner",
    )
    emb_path = query_embeddings_path(split, root=root)
    if with_embeddings is None:
        with_embeddings = emb_path.is_file()
    if with_embeddings:
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
    return merged.reset_index(drop=True)


def load_split_for_feature_set(
    split: str,
    feature_set: str,
    *,
    root: Path | None = None,
) -> pd.DataFrame:
    """Load labels + complexity + optional emb/heur columns required by ``feature_set``."""
    fs = normalize_feature_set(feature_set)
    return load_split(
        split,
        root=root,
        with_embeddings=feature_set_needs_embeddings(fs),
        with_heuristics=feature_set_needs_heuristics(fs),
    )


def c_vector_feature_cols(df: pd.DataFrame, *, version: int = 1) -> list[str]:
    cols = _C_VECTOR_COLS_V7 if version == 2 else _C_VECTOR_COLS
    return [c for c in cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]


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
        graph = c_vector_feature_cols(df, version=spec.cvec_version)
        label = "dim7_*" if spec.cvec_version == 2 else "dim_*"
        if not graph:
            raise ValueError(f"missing {label} — rebuild complexity_record_*.parquet")
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


# ── Eval-sample features (GAIA, math, …) ─────────────────────────────────────


def resolve_dataset_name(name: str) -> str:
    return DATASET_ALIASES.get((name or "").strip().lower(), (name or "").strip().lower())


def feature_paths(tag: str) -> tuple[Path, Path, Path]:
    t = tag.strip().lower()
    return (
        QCE_FEATURES_DIR / f"complexity_record_{t}.parquet",
        QCE_FEATURES_DIR / f"query_embeddings_{t}.parquet",
        DECOMPOSER_CACHE_DIR / f"qce_{t}_plans.jsonl",
    )


def eval_base_frame(parquet_path: Path, dataset: str | None = None) -> pd.DataFrame:
    path = Path(parquet_path)
    df = pd.read_parquet(path)
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
    path = Path(data_path) if data_path else EVAL_SAMPLES_DIR / f"{dataset.strip().lower()}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Eval parquet not found: {path}")
    return eval_base_frame(path, dataset=dataset)


def load_decompose_cache(cache_path: Path, *, id_key: str = "training_id") -> dict[str, dict[str, Any]]:
    from qce.decompose import load_plans_jsonl

    return {
        str(rec.get(id_key, "") or ""): rec
        for rec in load_plans_jsonl(cache_path)
        if rec.get(id_key)
    }


def decompose_cache_status(
    rows: list[dict[str, Any]], cache_path: Path, *, id_key: str = "training_id"
) -> tuple[int, int, list[str]]:
    if not cache_path.is_file():
        return 0, len(rows), [str(r[id_key]) for r in rows]
    cached = load_decompose_cache(cache_path, id_key=id_key)
    missing = [str(r[id_key]) for r in rows if str(r[id_key]) not in cached]
    return len(rows) - len(missing), len(rows), missing


def plans_from_cache(
    rows: list[dict[str, Any]], cache_path: Path, *, id_key: str = "training_id"
) -> list[dict[str, Any]]:
    cached = load_decompose_cache(cache_path, id_key=id_key)
    plans, missing = [], []
    for row in rows:
        tid = str(row[id_key])
        if tid in cached:
            plans.append(cached[tid])
        else:
            missing.append(tid)
    if missing:
        raise KeyError(f"{len(missing)} ids missing from {cache_path}")
    return plans


def merge_feature_tables(base: pd.DataFrame, tag: str) -> pd.DataFrame:
    c_path, e_path, _ = feature_paths(tag)
    if not c_path.is_file() or not e_path.is_file():
        raise FileNotFoundError(f"QCE features missing for {tag!r}: {c_path}, {e_path}")
    c, e = pd.read_parquet(c_path), pd.read_parquet(e_path)
    if "training_id" not in c.columns or "training_id" not in e.columns:
        raise KeyError("QCE feature tables must include training_id")
    if c["training_id"].duplicated().any():
        c = c.drop_duplicates(subset=["training_id"], keep="first")
    if e["training_id"].duplicated().any():
        e = e.drop_duplicates(subset=["training_id"], keep="first")
    drop_c = [x for x in c.columns if x in base.columns and x != "training_id"]
    drop_e = [x for x in e.columns if x in base.columns and x != "training_id"]
    merged = base.merge(c.drop(columns=drop_c, errors="ignore"), on="training_id", how="left", validate="m:1")
    merged = merged.merge(e.drop(columns=drop_e, errors="ignore"), on="training_id", how="left", validate="m:1")
    if len(merged) != len(base):
        raise ValueError(f"feature merge changed rows unexpectedly: {len(base)} → {len(merged)}")
    return merged.reset_index(drop=True)


def build_eval_features(
    base: pd.DataFrame, tag: str, *, force_refresh: bool = False, verbose: bool = True
) -> None:
    from qce.complexity import complexity_dataframe, load_train_norm, write_complexity_parquet
    from qce.decompose import decompose_batch
    from scripts.build_query_embeddings import DEFAULT_MODEL, encode_queries, matrix_to_frame

    c_path, e_path, cache_path = feature_paths(tag.strip().lower())
    rows = base.to_dict(orient="records")
    for r in rows:
        r["dataset"] = resolve_dataset_name(str(r.get("dataset") or tag))

    n_hit, n_total, missing_ids = decompose_cache_status(rows, cache_path)
    if verbose:
        print(
            f"Decompose cache {cache_path.name}: {n_hit}/{n_total}"
            + (f" ({len(missing_ids)} missing)" if missing_ids else " (full hit)")
        )

    if force_refresh:
        if verbose:
            print(f"Decomposing {n_total} queries (force_refresh)…")
        plans = decompose_batch(rows, cache_path=cache_path, force_refresh=True)
    elif not missing_ids:
        if verbose:
            print("Skipping LLM decompose — all ids in cache.")
        plans = plans_from_cache(rows, cache_path)
    else:
        if verbose:
            print(f"Decomposing {len(missing_ids)} missing ({n_hit} cached)…")
        plans = decompose_batch(rows, cache_path=cache_path, force_refresh=False)
        _, _, still_missing = decompose_cache_status(rows, cache_path)
        if still_missing:
            raise RuntimeError(f"{len(still_missing)} ids still missing after decompose")

    if not TRAIN_NORM_JSON.is_file():
        raise FileNotFoundError(f"Missing {TRAIN_NORM_JSON}")
    c_df = complexity_dataframe(plans, norm=load_train_norm(TRAIN_NORM_JSON), fit_norm=False)
    c_path.parent.mkdir(parents=True, exist_ok=True)
    write_complexity_parquet(c_df, c_path)
    if verbose:
        print(f"Wrote {c_path} ({len(c_df)} rows)")

    if not PCA_PATH.is_file():
        raise FileNotFoundError(f"Missing {PCA_PATH}")
    pca = joblib.load(PCA_PATH)
    n_comp = int(getattr(pca, "n_components_", 16))
    if verbose:
        print(f"Encoding {len(base)} queries ({DEFAULT_MODEL})…")
    raw = encode_queries(base["query"].astype(str).tolist(), model_name=DEFAULT_MODEL, batch_size=32, normalize_embeddings=True)
    matrix_to_frame(base["training_id"], pca.transform(raw), n_components=n_comp).to_parquet(e_path, index=False)
    if verbose:
        print(f"Wrote {e_path}")


def ensure_eval_features(
    base: pd.DataFrame,
    tag: str,
    *,
    build: bool = False,
    force_refresh: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    c_path, e_path, _ = feature_paths(tag)
    if not c_path.is_file() or not e_path.is_file():
        if not build:
            raise FileNotFoundError(f"QCE features missing for {tag!r}; use --build-features")
        build_eval_features(base, tag, force_refresh=force_refresh, verbose=verbose)
    return merge_feature_tables(base, tag)


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


# ── Sklearn pipeline & training ──────────────────────────────────────────────


def make_pipeline(
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
    pipe = make_pipeline(
        feature_cols,
        class_weight=spec.effective_class_weight,
        hgbm_params=spec.hgbm_params,
        calibrated=spec.calibrated,
        random_state=spec.random_state,
    )
    fit_router(pipe, fit_df, feature_cols, target=spec.target, sample_weight=weights)
    val_result = evaluate(pipe, val_df, feature_cols, target=spec.target)
    cv_mean, cv_std = cross_val_macro_f1(
        make_pipeline(
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
    meta = {"experiment_id": experiment_id, "n_features": len(feature_cols), "feature_cols": feature_cols}
    if extra_meta:
        meta.update(extra_meta)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


# ── Inference ─────────────────────────────────────────────────────────────────


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
    path = Path(router_path or ROUTER_MODEL_PATH)
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


# ── CLI ───────────────────────────────────────────────────────────────────────


def _grid_combos(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = list(grid.keys())
    return [dict(zip(keys, vals, strict=True)) for vals in itertools.product(*(grid[k] for k in keys))]


def _tune_hgbm_cv(train_df: pd.DataFrame, feature_cols: list[str], *, cv: int = 5) -> tuple[dict[str, Any], pd.DataFrame]:
    combos = _grid_combos(HGBM_GRID)
    y = train_df[TARGET].astype(str)
    weights = compute_sample_weights(train_df, y, "balanced")
    fit_params = {"clf__sample_weight": weights} if weights is not None else {}
    best_score, best_params = -1.0, combos[0]
    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        pipe = make_pipeline(feature_cols, hgbm_params=params, calibrated=False)
        score = float(
            cross_val_score(
                pipe, train_df[feature_cols], y, cv=cv, scoring="f1_macro", n_jobs=-1, params=fit_params
            ).mean()
        )
        rows.append({**params, "cv_macro_f1": round(score, 4)})
        if score > best_score:
            best_score, best_params = score, params
        if i % 9 == 0:
            print(f"  grid {i}/{len(combos)}…")
    return best_params, pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)


def _cli_train(args: argparse.Namespace) -> None:
    fs: FeatureSet = getattr(args, "feature_set", PRODUCTION_FEATURE_SET)
    needs_emb = feature_set_needs_embeddings(fs)
    train_df = load_split("train", with_embeddings=needs_emb)
    val_df = load_split("val", with_embeddings=needs_emb)
    validate_feature_set_data(train_df, fs)
    fs_norm = normalize_feature_set(fs)
    spec = TrainSpec(
        experiment_id=experiment_id_for_feature_set(fs),
        feature_set=fs,
        hgbm_params=PRODUCTION_HGBM_PARAMS if is_production_feature_set(fs) else None,
    )
    pipe, feature_cols, val_res, (cv_mean, cv_std), _ = train_router(
        train_df, val_df, spec=spec, verbose=not args.quiet
    )
    if not args.quiet:
        print(f"feature_set={normalize_feature_set(fs)} n_features={len(feature_cols)}")
        print(predict_agent_proba(pipe, val_df, feature_cols).head(3).to_string())
    if args.save:
        out_path = model_path_for_feature_set(fs)
        from routing.config import LEGACY_HARD_HGBM_EXPERIMENT_ID

        exp_id = (
            LEGACY_HARD_HGBM_EXPERIMENT_ID
            if is_production_feature_set(fs)
            else experiment_id_for_feature_set(fs)
        )
        out = save_router(
            pipe,
            feature_cols=feature_cols,
            experiment_id=exp_id,
            path=out_path,
            extra_meta={
                "val_macro_f1": val_res.macro_f1,
                "train_cv_macro_f1": cv_mean,
                "feature_set": fs_norm,
                "hgbm_params": "default" if spec.hgbm_params is None else spec.hgbm_params,
                "production": is_production_feature_set(fs),
            },
        )
        print("Saved", out)


def _cli_eval(args: argparse.Namespace) -> None:
    if not args.router.is_file():
        raise FileNotFoundError(f"Router not found: {args.router}")
    train_df = load_split("train", with_embeddings=True)
    eval_df = load_split(args.split, with_embeddings=True)
    obj = load_router(args.router)
    meta_path = Path(args.router).with_suffix(".json")
    feature_set_label = obj.get("feature_set")
    if not feature_set_label and meta_path.is_file():
        feature_set_label = json.loads(meta_path.read_text(encoding="utf-8")).get("feature_set")
    result = evaluate(obj["pipeline"], eval_df, list(obj["feature_cols"]))
    rows = [
        results_row("B0_always_raw", baseline_always_majority(train_df, eval_df)),
        results_row("B0_dataset_mode", baseline_per_dataset_mode(train_df, eval_df)),
        results_row(
            str(obj.get("experiment_id", args.router.stem)),
            result,
            feature_set=str(feature_set_label or PRODUCTION_FEATURE_SET),
            n_features=len(obj["feature_cols"]),
            model="hgbm",
        ),
    ]
    table = pd.DataFrame(rows)
    print(f"\n=== Router eval on {args.split!r} (n={len(eval_df)}) ===\n{table.to_string(index=False)}")
    b0 = table[table["experiment"] == "B0_dataset_mode"].iloc[0]
    ml = table.iloc[-1]
    mf = float(ml["macro_f1"])
    print(f"\n{mf:.4f} macro-F1 ({mf - float(b0['macro_f1']):+.4f} vs dataset_mode)\n{result.report}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(table.to_json(orient="records", indent=2), encoding="utf-8")


def run_feature_ablation(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Legacy entrypoint — runs the 5-dim (cvec5) ablation table."""
    return run_ablation_cvec5(eval_split=eval_split, out_dir=out_dir, verbose=verbose)


def _run_ablation_table(
    cases: tuple[AblationCase, ...],
    *,
    track: str,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    out_dir = Path(out_dir or REPO_ROOT / f"results/experiments/feature_ablation_{track}")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_split("train", with_embeddings=True)
    val_df = load_split("val", with_embeddings=True)
    eval_df = load_split(eval_split, with_embeddings=True)

    rows: list[dict[str, Any]] = []
    for case in cases:
        fs = normalize_feature_set(case.feature_set)
        exp_id = f"hgbm_{track}_{case.label.replace('+', '_').replace(' ', '_').replace('(', '').replace(')', '')}"
        spec = TrainSpec(experiment_id=exp_id, feature_set=case.feature_set)
        if verbose:
            print(f"\n=== [{track}] {case.label} ({fs}) ===")
            if case.description:
                print(f"    {case.description}")
        pipe, fcols, val_res, (cv_mean, cv_std), _ = train_router(
            train_df, val_df, spec=spec, verbose=verbose
        )
        test_res = evaluate(pipe, eval_df, fcols)
        rows.append(
            {
                "track": track,
                "label": case.label,
                "feature_set": fs,
                "description": case.description,
                "n_features": len(fcols),
                "feature_cols": ",".join(fcols),
                "val_macro_f1": round(val_res.macro_f1, 4),
                "val_accuracy": round(val_res.accuracy, 4),
                f"{eval_split}_macro_f1": round(test_res.macro_f1, 4),
                f"{eval_split}_accuracy": round(test_res.accuracy, 4),
                "train_cv_macro_f1": round(cv_mean, 4),
                "train_cv_std": round(cv_std, 4),
            }
        )
    table = pd.DataFrame(rows)
    path = out_dir / f"ablation_{eval_split}.csv"
    table.to_csv(path, index=False)
    if verbose:
        show = table.drop(columns=["feature_cols", "description"], errors="ignore")
        print(f"\n=== Ablation [{track}] ({eval_split}) ===\n{show.to_string(index=False)}")
        print(f"Wrote {path}")
    return table


def run_ablation_cvec5(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    return _run_ablation_table(
        CVEC5_ABLATION_CASES,
        track="cvec5",
        eval_split=eval_split,
        out_dir=out_dir or REPO_ROOT / "results/experiments/feature_ablation_cvec5",
        verbose=verbose,
    )


def run_ablation_cvec7(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    return _run_ablation_table(
        CVEC7_ABLATION_CASES,
        track="cvec7",
        eval_split=eval_split,
        out_dir=out_dir or REPO_ROOT / "results/experiments/feature_ablation_cvec7",
        verbose=verbose,
    )


def run_ablation_trust(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    return _run_ablation_table(
        TRUST_ABLATION_CASES,
        track="trust",
        eval_split=eval_split,
        out_dir=out_dir or REPO_ROOT / "results/experiments/feature_ablation_trust",
        verbose=verbose,
    )


def run_ablation_all(
    *,
    eval_split: str = "test",
    verbose: bool = True,
) -> dict[str, pd.DataFrame]:
    return {
        "cvec5": run_ablation_cvec5(eval_split=eval_split, verbose=verbose),
        "cvec7": run_ablation_cvec7(eval_split=eval_split, verbose=verbose),
        "trust": run_ablation_trust(eval_split=eval_split, verbose=verbose),
    }


def _fit_eval_split(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    *,
    feature_set: FeatureSet,
    hgbm_params: dict[str, Any] | None,
    random_state: int,
) -> tuple[EvalResult, list[str]]:
    """Train on train split only; score ``eval_df`` (val or test)."""
    spec = TrainSpec(
        experiment_id="seed_stability",
        feature_set=feature_set,
        hgbm_params=hgbm_params,
        random_state=random_state,
    )
    feature_cols = router_feature_cols(train_df, spec.feature_set)
    fit_df = oversample_minorities(train_df, spec.target) if spec.oversample else train_df
    y_fit = fit_df[spec.target].astype(str)
    weights = compute_sample_weights(fit_df, y_fit, spec.weight_mode)
    pipe = make_pipeline(
        feature_cols,
        class_weight=spec.effective_class_weight,
        hgbm_params=spec.hgbm_params,
        random_state=spec.random_state,
    )
    fit_router(pipe, fit_df, feature_cols, target=spec.target, sample_weight=weights)
    return evaluate(pipe, eval_df, feature_cols, target=spec.target), feature_cols


def run_seed_stability(
    *,
    feature_set: FeatureSet = PRODUCTION_FEATURE_SET,
    eval_split: str = "test",
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2024),
    out_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compare default vs tuned HGBM on ``feature_set`` across random seeds.

    Trains on the train split only; reports macro-F1 on ``eval_split`` (default test).
    """
    out_dir = Path(out_dir or SEED_STABILITY_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df = load_split("train", with_embeddings=True)
    eval_df = load_split(eval_split, with_embeddings=True)
    feature_cols = router_feature_cols(train_df, feature_set)

    configs: tuple[tuple[str, dict[str, Any] | None], ...] = (
        ("default", DEFAULT_HGBM_PARAMS),
        ("tuned", TUNED_HGBM_PARAMS),
    )

    rows: list[dict[str, Any]] = []
    for config_name, hgbm_params in configs:
        for seed in seeds:
            result, _ = _fit_eval_split(
                train_df,
                eval_df,
                feature_set=feature_set,
                hgbm_params=hgbm_params,
                random_state=seed,
            )
            rows.append(
                {
                    "config": config_name,
                    "seed": seed,
                    "feature_set": normalize_feature_set(feature_set),
                    "n_features": len(feature_cols),
                    f"{eval_split}_macro_f1": round(result.macro_f1, 4),
                    f"{eval_split}_accuracy": round(result.accuracy, 4),
                }
            )
            if verbose:
                print(
                    f"  {config_name} seed={seed} "
                    f"{eval_split} macro-F1={result.macro_f1:.4f} acc={result.accuracy:.4f}"
                )

    per_seed = pd.DataFrame(rows)
    per_seed_path = out_dir / f"per_seed_{eval_split}.csv"
    per_seed.to_csv(per_seed_path, index=False)

    summary_rows: list[dict[str, Any]] = []
    for config_name, _ in configs:
        sub = per_seed[per_seed["config"] == config_name]
        f1_col = f"{eval_split}_macro_f1"
        acc_col = f"{eval_split}_accuracy"
        summary_rows.append(
            {
                "config": config_name,
                "n_seeds": len(sub),
                f"{eval_split}_macro_f1_mean": round(float(sub[f1_col].mean()), 4),
                f"{eval_split}_macro_f1_std": round(float(sub[f1_col].std(ddof=0)), 4),
                f"{eval_split}_accuracy_mean": round(float(sub[acc_col].mean()), 4),
                f"{eval_split}_accuracy_std": round(float(sub[acc_col].std(ddof=0)), 4),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / f"summary_{eval_split}.csv"
    summary.to_csv(summary_path, index=False)

    if verbose:
        print(f"\n=== Seed stability ({feature_set}, {eval_split}, n={len(seeds)}) ===")
        for _, row in summary.iterrows():
            print(
                f"  {row['config']:8s}  "
                f"macro-F1 {row[f'{eval_split}_macro_f1_mean']:.3f} "
                f"± {row[f'{eval_split}_macro_f1_std']:.3f}  "
                f"(acc {row[f'{eval_split}_accuracy_mean']:.3f} "
                f"± {row[f'{eval_split}_accuracy_std']:.3f})"
            )
        d = summary[summary["config"] == "default"][f"{eval_split}_macro_f1_mean"].iloc[0]
        t = summary[summary["config"] == "tuned"][f"{eval_split}_macro_f1_mean"].iloc[0]
        dd = summary[summary["config"] == "default"][f"{eval_split}_macro_f1_std"].iloc[0]
        td = summary[summary["config"] == "tuned"][f"{eval_split}_macro_f1_std"].iloc[0]
        print(
            f"\n  Δ tuned−default (mean): {t - d:+.4f}  "
            f"(overlap if |Δ| < {dd + td:.4f} ≈ combined spread)"
        )
        print(f"Wrote {per_seed_path}\nWrote {summary_path}")

    return per_seed, summary


def run_leave_one_dataset_out(
    *,
    eval_split: str = "test",
    out_dir: Path | None = None,
    include_dataset_feature: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Hold out one benchmark dataset: train on the other datasets, eval on held-out rows.

    Default ``cvec5_emb`` (no dataset one-hot) tests that C(Q)+emb generalize
    beyond benchmark identity.
    """
    out_dir = Path(out_dir or REPO_ROOT / "results/experiments/lodo")
    out_dir.mkdir(parents=True, exist_ok=True)
    train_all = load_split("train", with_embeddings=True)
    val_all = load_split("val", with_embeddings=True)
    eval_all = load_split(eval_split, with_embeddings=True)
    fs: FeatureSet = "cvec5_emb_ds" if include_dataset_feature else "cvec5_emb"

    rows: list[dict[str, Any]] = []
    for held in QCE_DATASETS:
        train_df = train_all[train_all["dataset"] != held].copy()
        val_df = val_all[val_all["dataset"] != held].copy()
        eval_df = eval_all[eval_all["dataset"] == held].copy()
        if eval_df.empty:
            continue
        spec = TrainSpec(
            experiment_id=f"lodo_hold_{held}_{'ds' if include_dataset_feature else 'nods'}",
            feature_set=fs,
        )
        if verbose:
            print(
                f"\n=== LODO hold-out={held} train_n={len(train_df)} "
                f"eval_n={len(eval_df)} feature_set={fs} ==="
            )
        pipe, fcols, val_res, (cv_mean, _), _ = train_router(
            train_df, val_df, spec=spec, verbose=verbose
        )
        test_res = evaluate(pipe, eval_df, fcols)
        rows.append(
            {
                "held_out_dataset": held,
                "feature_set": fs,
                "n_train": len(train_df),
                "n_val": len(val_df),
                "n_eval": len(eval_df),
                "val_macro_f1": round(val_res.macro_f1, 4),
                f"{eval_split}_macro_f1": round(test_res.macro_f1, 4),
                f"{eval_split}_accuracy": round(test_res.accuracy, 4),
                "train_cv_macro_f1": round(cv_mean, 4),
            }
        )

    table = pd.DataFrame(rows)
    tag = "with_dataset" if include_dataset_feature else "no_dataset"
    path = out_dir / f"lodo_{tag}_{eval_split}.csv"
    table.to_csv(path, index=False)
    if verbose and len(table):
        print(f"\n=== Leave-one-dataset-out ({eval_split}, {tag}) ===\n{table.to_string(index=False)}")
        print(
            f"Mean {eval_split} macro-F1: {table[f'{eval_split}_macro_f1'].mean():.4f}\n"
            f"Wrote {path}"
        )
    return table


def _cli_tune(args: argparse.Namespace) -> None:
    from routing.analysis import (
        build_analysis_frame,
        plot_reliability_calibration,
        router_confidence_sweep,
        top2_sweep,
    )

    fs: FeatureSet = getattr(args, "feature_set", PRODUCTION_FEATURE_SET)
    needs_emb = feature_set_needs_embeddings(fs)
    train_df = load_split("train", with_embeddings=needs_emb)
    val_df = load_split("val", with_embeddings=needs_emb)
    test_df = load_split("test", with_embeddings=needs_emb)
    validate_feature_set_data(train_df, fs)
    fs_norm = normalize_feature_set(fs)
    feature_cols = router_feature_cols(train_df, fs)
    TUNE_OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.skip_grid:
        best_params = {"max_depth": 6, "learning_rate": 0.05, "max_leaf_nodes": 31}
        grid_df = pd.DataFrame()
    else:
        print(f"=== HGBM grid ({len(_grid_combos(HGBM_GRID))} combos) ===")
        best_params, grid_df = _tune_hgbm_cv(train_df, feature_cols, cv=args.cv)
        if len(grid_df):
            grid_df.to_csv(TUNE_OUT_DIR / "hgbm_grid_cv.csv", index=False)

    tuned_exp = (
        TUNED_ROUTER_EXPERIMENT_ID
        if is_production_feature_set(fs)
        else f"{TUNED_EXPERIMENT_ID}_{fs_norm}"
    )
    spec = TrainSpec(experiment_id=tuned_exp, feature_set=fs, hgbm_params=best_params)
    pipe, fcols, val_res, (cv_mean, _), _ = train_router(train_df, val_df, spec=spec)
    summary = {
        "experiment_id": tuned_exp,
        "feature_set": fs_norm,
        "comparison_only": is_production_feature_set(fs),
        "hgbm_best_params": best_params,
        "val": {"macro_f1": val_res.macro_f1, "accuracy": val_res.accuracy},
        "test": {
            "macro_f1": evaluate(pipe, test_df, fcols).macro_f1,
            "accuracy": evaluate(pipe, test_df, fcols).accuracy,
        },
        "train_cv_macro_f1": cv_mean,
    }
    if args.save:
        tuned_path = (
            TUNED_ROUTER_PATH
            if is_production_feature_set(fs)
            else model_path_for_feature_set(fs).parent / f"{tuned_exp}.joblib"
        )
        save_router(
            pipe,
            feature_cols=fcols,
            experiment_id=tuned_exp,
            path=tuned_path,
            extra_meta={
                "feature_set": fs_norm,
                "hgbm_params": best_params,
                "val_macro_f1": val_res.macro_f1,
                "train_cv_macro_f1": cv_mean,
                "comparison_only": is_production_feature_set(fs),
            },
        )
        print("Saved", tuned_path)
    if args.calibrated:
        cal_spec = TrainSpec(
            experiment_id=f"{tuned_exp}_calibrated",
            feature_set=fs,
            hgbm_params=best_params,
            calibrated=True,
        )
        cal_pipe, cal_fcols, _, _, _ = train_router(train_df, val_df, spec=cal_spec)
        if args.save:
            cal_path = model_path_for_feature_set(fs).parent / f"{tuned_exp}_calibrated.joblib"
            save_router(
                cal_pipe,
                feature_cols=cal_fcols,
                experiment_id=f"{tuned_exp}_calibrated",
                path=cal_path,
            )

    analysis = build_analysis_frame("val", pipe=pipe, feature_cols=fcols, experiment_id=tuned_exp)
    top2_sweep(analysis).to_csv(TUNE_OUT_DIR / f"top2_sweep_val_{fs}.csv", index=False)
    router_confidence_sweep(analysis).to_csv(TUNE_OUT_DIR / f"confidence_sweep_val_{fs}.csv", index=False)
    plot_reliability_calibration(analysis, TUNE_OUT_DIR / f"calibration_val_{fs}.png", title_suffix=f" ({tuned_exp}, val)")
    (TUNE_OUT_DIR / "tuning_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QCE router (HGBM)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    fs_help = (
        "Feature columns for the router. Prefer cvec5_* / cvec7_* names; "
        "graph_* names are legacy aliases."
    )

    p_train = sub.add_parser("train", help="Train on QCE train/val")
    p_train.add_argument("--save", action="store_true")
    p_train.add_argument("--quiet", action="store_true")
    p_train.add_argument(
        "--feature-set",
        dest="feature_set",
        choices=FEATURE_SET_CLI_CHOICES,
        default=PRODUCTION_FEATURE_SET,
        help=fs_help,
    )

    p_eval = sub.add_parser("eval", help="Evaluate on QCE val/test vs baselines")
    p_eval.add_argument("--split", choices=("val", "test"), default="test")
    p_eval.add_argument("--router", type=Path, default=ROUTER_MODEL_PATH)
    p_eval.add_argument("--json-out", type=Path, default=None)

    p_tune = sub.add_parser(
        "tune",
        help="HGBM grid search (saves comparison model; production uses default HGBM)",
    )
    p_tune.add_argument("--save", action="store_true")
    p_tune.add_argument("--skip-grid", action="store_true")
    p_tune.add_argument("--cv", type=int, default=5)
    p_tune.add_argument("--calibrated", action="store_true")
    p_tune.add_argument(
        "--feature-set",
        dest="feature_set",
        choices=FEATURE_SET_CLI_CHOICES,
        default=PRODUCTION_FEATURE_SET,
        help=f"Feature set to tune (comparison; production={PRODUCTION_FEATURE_SET}).",
    )

    p_abl = sub.add_parser("ablation", help="Feature ablation tables (cvec5 / cvec7 / trust)")
    p_abl.add_argument("--split", choices=("val", "test"), default="test")
    p_abl.add_argument(
        "--track",
        choices=("cvec5", "cvec7", "trust", "all"),
        default="all",
        help="Which ablation table to run (default: all three)",
    )
    p_abl.add_argument("--out-dir", type=Path, default=None)
    p_abl.add_argument("--quiet", action="store_true")

    p_lodo = sub.add_parser("lodo", help="Leave-one-dataset-out routing eval")
    p_lodo.add_argument("--split", choices=("val", "test"), default="test")
    p_lodo.add_argument("--out-dir", type=Path, default=None)
    p_lodo.add_argument(
        "--with-dataset",
        action="store_true",
        help="Include dataset one-hot (default: cvec5+emb only, no dataset)",
    )
    p_lodo.add_argument("--quiet", action="store_true")

    p_seed = sub.add_parser(
        "seed-sweep",
        help="Default vs tuned HGBM across random seeds (freeze check)",
    )
    p_seed.add_argument("--split", choices=("val", "test"), default="test")
    p_seed.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2024],
        help="Random seeds (default: 5 seeds)",
    )
    p_seed.add_argument("--out-dir", type=Path, default=None)
    p_seed.add_argument(
        "--feature-set",
        dest="feature_set",
        default=PRODUCTION_FEATURE_SET,
        choices=FEATURE_SET_CLI_CHOICES,
    )
    p_seed.add_argument("--quiet", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd == "train":
        _cli_train(args)
    elif args.cmd == "eval":
        _cli_eval(args)
    elif args.cmd == "tune":
        _cli_tune(args)
    elif args.cmd == "ablation":
        if args.track == "cvec5":
            run_ablation_cvec5(
                eval_split=args.split,
                out_dir=args.out_dir,
                verbose=not args.quiet,
            )
        elif args.track == "cvec7":
            run_ablation_cvec7(
                eval_split=args.split,
                out_dir=args.out_dir,
                verbose=not args.quiet,
            )
        elif args.track == "trust":
            run_ablation_trust(
                eval_split=args.split,
                out_dir=args.out_dir,
                verbose=not args.quiet,
            )
        else:
            run_ablation_all(eval_split=args.split, verbose=not args.quiet)
    elif args.cmd == "lodo":
        run_leave_one_dataset_out(
            eval_split=args.split,
            out_dir=args.out_dir,
            include_dataset_feature=args.with_dataset,
            verbose=not args.quiet,
        )
    elif args.cmd == "seed-sweep":
        run_seed_stability(
            feature_set=args.feature_set,
            eval_split=args.split,
            seeds=tuple(args.seeds),
            out_dir=args.out_dir,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
