"""
Classifier baselines on frozen ``cvec5_emb`` features (Step 2).

Trains logreg / LightGBM with the same label and imbalance recipe as production HGBM,
evaluates **test utility regret** via ``routing.analysis`` (execution oracle).

Primary decision metric: ``mean_utility_regret`` on test.
Refreeze production only if test regret <= ``REFREEZE_REGRET_MAX`` (0.341 = 0.361 − 0.02).
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from routing.analysis import (
    DEFAULT_ORACLE,
    attach_outcomes,
    metrics_for_routed,
    oracle_outcome_matrix,
)
from routing.config import (
    HGBM_GRID,
    PRODUCTION_FEATURE_SET,
    LEGACY_HARD_HGBM_PATH,
    REPO_ROOT,
)
from routing.router import (
    TARGET,
    TrainSpec,
    attach_router_predictions,
    compute_sample_weights,
    evaluate,
    load_router,
    load_split,
    router_feature_cols,
    validate_feature_set_data,
)
from src.utils.soft_labels import DEFAULT_UTILITY_LAMBDA
from sklearn.ensemble import HistGradientBoostingClassifier

# Step 1 frozen HGBM test regret ≈ 0.361; refreeze only if new model beats this by ≥ 0.02.
REFREEZE_REGRET_MAX = 0.341
BASELINES_OUT_DIR = REPO_ROOT / "results/experiments/classifier_baselines_cvec5_emb"

ClassifierName = Literal["hgbm", "logreg", "lgbm"]
HpPolicy = Literal["frozen", "default", "cv_tuned", "soft_kl"]

# LightGBM defaults aligned with production HGBM (``make_pipeline`` / ``max_iter=300``).
LGBM_DEFAULT_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "learning_rate": 0.05,
    "max_depth": 6,
    "num_leaves": 31,
    "min_child_samples": 10,
    "class_weight": "balanced",
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}

# Same grid axes as ``HGBM_GRID``; ``num_leaves`` is the LGBM analogue of ``max_leaf_nodes``.
LGBM_GRID: dict[str, list[Any]] = {
    "max_depth": HGBM_GRID["max_depth"],
    "learning_rate": HGBM_GRID["learning_rate"],
    "num_leaves": HGBM_GRID["max_leaf_nodes"],
}

TreeBackend = Literal["lightgbm", "sklearn_hgb"]


def _resolve_tree_backend() -> TreeBackend:
    """Prefer LightGBM; fall back to sklearn HGB when the native lib is unavailable (e.g. missing libomp)."""
    try:
        from lightgbm import LGBMClassifier  # noqa: F401
    except (ImportError, OSError) as exc:
        print(
            "WARNING: lightgbm unavailable "
            f"({exc!r}); using sklearn HistGradientBoostingClassifier for tree baselines. "
            "On macOS: brew install libomp"
        )
        return "sklearn_hgb"
    return "lightgbm"


def _tree_classifier(
    *,
    backend: TreeBackend,
    lgbm_params: dict[str, Any] | None,
    random_state: int,
) -> Any:
    if backend == "lightgbm":
        from lightgbm import LGBMClassifier

        kw = {**LGBM_DEFAULT_PARAMS, "random_state": random_state}
        if lgbm_params:
            kw.update(lgbm_params)
        return LGBMClassifier(**kw)

    # Map LGBM-style grid params to sklearn HGB (same role in Step 2 comparison).
    hgbm_kw: dict[str, Any] = {
        "max_iter": 300,
        "learning_rate": 0.05,
        "max_depth": 6,
        "max_leaf_nodes": 31,
        "min_samples_leaf": 10,
        "class_weight": "balanced",
        "random_state": random_state,
    }
    if lgbm_params:
        if "max_depth" in lgbm_params:
            hgbm_kw["max_depth"] = lgbm_params["max_depth"]
        if "learning_rate" in lgbm_params:
            hgbm_kw["learning_rate"] = lgbm_params["learning_rate"]
        if "num_leaves" in lgbm_params:
            hgbm_kw["max_leaf_nodes"] = lgbm_params["num_leaves"]
    return HistGradientBoostingClassifier(**hgbm_kw)


def _numeric_feature_cols(feature_cols: list[str]) -> list[str]:
    return [c for c in feature_cols if c != "dataset"]


def _prep_step(feature_cols: list[str]) -> ColumnTransformer:
    """Impute numeric features — same rule as ``router.make_pipeline``."""
    num_cols = _numeric_feature_cols(feature_cols)
    prep = ColumnTransformer(
        [("num", SimpleImputer(strategy="constant", fill_value=0), num_cols)],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    # Keep column names through to LGBM (avoids "X does not have valid feature names" warnings).
    prep.set_output(transform="pandas")
    return prep


def _balanced_fit_params(df: pd.DataFrame, target: str = TARGET) -> dict[str, Any]:
    """Match production HGBM: ``compute_sample_weight('balanced')`` passed as ``clf__sample_weight``."""
    weights = compute_sample_weights(df, df[target].astype(str), "balanced")
    if weights is None:
        return {}
    return {"clf__sample_weight": weights}


def make_logreg_pipeline(
    feature_cols: list[str],
    *,
    random_state: int = 42,
    class_weight: str | dict[str, float] | None = "balanced",
) -> Pipeline:
    """Linear baseline: impute → scale → multinomial logreg."""
    return Pipeline(
        [
            ("prep", _prep_step(feature_cols)),
            ("scale", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    C=1.0,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def make_lgbm_pipeline(
    feature_cols: list[str],
    *,
    lgbm_params: dict[str, Any] | None = None,
    random_state: int = 42,
    backend: TreeBackend | None = None,
) -> Pipeline:
    """Tree baseline: impute → LightGBM or sklearn HGB fallback (balanced + sample weights at fit)."""
    resolved = backend or _resolve_tree_backend()
    clf = _tree_classifier(backend=resolved, lgbm_params=lgbm_params, random_state=random_state)
    return Pipeline([("prep", _prep_step(feature_cols)), ("clf", clf)])


def fit_baseline(
    pipe: Pipeline,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    target: str = TARGET,
) -> Pipeline:
    """Fit with the same balanced sample weights as ``router.fit_router``."""
    fit_params = _balanced_fit_params(train_df, target)
    pipe.fit(train_df[feature_cols], train_df[target].astype(str), **fit_params)
    return pipe


def tune_lgbm_cv(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    *,
    cv: int = 5,
    random_state: int = 42,
    backend: TreeBackend | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Select tree hyperparameters by train CV macro-F1 (same objective as HGBM tune)."""
    resolved = backend or _resolve_tree_backend()
    combos = [
        dict(zip(LGBM_GRID.keys(), vals, strict=True))
        for vals in itertools.product(*(LGBM_GRID[k] for k in LGBM_GRID))
    ]
    y = train_df[TARGET].astype(str)
    fit_params = _balanced_fit_params(train_df, TARGET)
    best_score, best_params = -1.0, combos[0]
    rows: list[dict[str, Any]] = []
    for i, params in enumerate(combos, start=1):
        pipe = make_lgbm_pipeline(
            feature_cols, lgbm_params=params, random_state=random_state, backend=resolved
        )
        score = float(
            cross_val_score(
                pipe,
                train_df[feature_cols],
                y,
                cv=cv,
                scoring="f1_macro",
                n_jobs=-1,
                params=fit_params,
            ).mean()
        )
        rows.append({**params, "cv_macro_f1": round(score, 4)})
        if score > best_score:
            best_score, best_params = score, params
        if i % 9 == 0:
            print(f"  LGBM grid {i}/{len(combos)}…")
    grid_df = pd.DataFrame(rows).sort_values("cv_macro_f1", ascending=False)
    return best_params, grid_df


def _exec_metrics_row(
    *,
    classifier: ClassifierName,
    hp_policy: HpPolicy,
    split: str,
    experiment_id: str,
    feature_cols: list[str],
    label_eval: Any,
    exec_metrics: Any,
    tree_backend: str | None = None,
) -> dict[str, Any]:
    """One comparison-table row: label diagnostics + execution / regret from oracle."""
    return {
        "classifier": classifier,
        "hp_policy": hp_policy,
        "split": split,
        "experiment_id": experiment_id,
        "feature_set": PRODUCTION_FEATURE_SET,
        "n_features": len(feature_cols),
        "n": exec_metrics.n,
        "utility_lambda": DEFAULT_UTILITY_LAMBDA,
        "macro_f1": round(label_eval.macro_f1, 4),
        "accuracy_label": round(label_eval.accuracy, 4),
        "exec_em": round(exec_metrics.accuracy, 4),
        "mean_cost_usd": round(exec_metrics.mean_cost_usd, 8),
        "mean_utility_regret": (
            round(exec_metrics.mean_utility_regret, 6)
            if exec_metrics.mean_utility_regret is not None
            else None
        ),
        "oracle_label_match": (
            round(exec_metrics.oracle_match_rate, 4)
            if exec_metrics.oracle_match_rate is not None
            else None
        ),
        "em_per_usd": (
            round(exec_metrics.em_per_usd, 2) if exec_metrics.em_per_usd is not None else None
        ),
        "tree_backend": tree_backend,
    }


def eval_pipeline_on_split(
    split: str,
    pipe: Pipeline,
    feature_cols: list[str],
    *,
    classifier: ClassifierName,
    hp_policy: HpPolicy,
    experiment_id: str,
    oracle_path: Path = DEFAULT_ORACLE,
    tree_backend: str | None = None,
) -> dict[str, Any]:
    """
    Score a fitted sklearn pipeline on a split.

    Flow: load split → predict agent → merge oracle execution → label F1 + regret.
    """
    df = load_split(split, with_embeddings=True)
    outcomes = oracle_outcome_matrix(oracle_path, df["training_id"])
    df = attach_router_predictions(df, pipe, feature_cols, experiment_id=experiment_id)
    df = attach_outcomes(df, outcomes)
    label_eval = evaluate(pipe, df, feature_cols, target=TARGET)
    exec_metrics = metrics_for_routed(df, "router_pred", name=experiment_id)
    return _exec_metrics_row(
        classifier=classifier,
        hp_policy=hp_policy,
        split=split,
        experiment_id=experiment_id,
        feature_cols=feature_cols,
        label_eval=label_eval,
        exec_metrics=exec_metrics,
        tree_backend=tree_backend,
    )


def eval_frozen_hgbm(
    split: str,
    *,
    router_path: Path = LEGACY_HARD_HGBM_PATH,
    oracle_path: Path = DEFAULT_ORACLE,
) -> dict[str, Any]:
    """Reference row: production artifact, no retraining."""
    obj = load_router(router_path)
    pipe = obj["pipeline"]
    feature_cols: list[str] = list(obj["feature_cols"])
    experiment_id = str(obj.get("experiment_id", router_path.stem))
    return eval_pipeline_on_split(
        split,
        pipe,
        feature_cols,
        classifier="hgbm",
        hp_policy="frozen",
        experiment_id=experiment_id,
        oracle_path=oracle_path,
    )


def run_classifier_baselines(
    *,
    splits: tuple[str, ...] = ("test",),
    out_dir: Path = BASELINES_OUT_DIR,
    oracle_path: Path = DEFAULT_ORACLE,
    cv: int = 5,
    random_state: int = 42,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Train/evaluate all Step 2 rows and write ``baselines_{split}.csv`` plus summary JSON.

    Rows (per split):
      1. HGBM frozen (production joblib)
      2. Logistic regression (default)
      3. LightGBM (default)
      4. LightGBM (CV-tuned on train macro-F1)
    """
    train_spec = TrainSpec(random_state=random_state)
    train_df = load_split("train", with_embeddings=True)
    validate_feature_set_data(train_df, train_spec.feature_set)
    feature_cols = router_feature_cols(train_df, train_spec.feature_set)

    rows: list[dict[str, Any]] = []
    tree_backend = _resolve_tree_backend()

    if verbose:
        print(
            f"feature_set={PRODUCTION_FEATURE_SET} n_features={len(feature_cols)} "
            f"λ={DEFAULT_UTILITY_LAMBDA} tree_backend={tree_backend}"
        )

    for split in splits:
        if verbose:
            print(f"\n--- split={split!r} ---")

        if verbose:
            print("  [1/4] HGBM frozen (production)")
        rows.append(eval_frozen_hgbm(split, oracle_path=oracle_path))

        if verbose:
            print("  [2/4] Logistic regression (default)")
        logreg = make_logreg_pipeline(feature_cols, random_state=random_state)
        fit_baseline(logreg, train_df, feature_cols)
        rows.append(
            eval_pipeline_on_split(
                split,
                logreg,
                feature_cols,
                classifier="logreg",
                hp_policy="default",
                experiment_id="logreg_cvec5_emb_default",
                oracle_path=oracle_path,
            )
        )

        if verbose:
            print(f"  [3/4] LightGBM / tree (default, backend={tree_backend})")
        lgbm = make_lgbm_pipeline(feature_cols, random_state=random_state, backend=tree_backend)
        fit_baseline(lgbm, train_df, feature_cols)
        rows.append(
            eval_pipeline_on_split(
                split,
                lgbm,
                feature_cols,
                classifier="lgbm",
                hp_policy="default",
                experiment_id="lgbm_cvec5_emb_default",
                oracle_path=oracle_path,
                tree_backend=tree_backend,
            )
        )

        if verbose:
            print(f"  [4/4] LightGBM / tree (CV-tuned, cv={cv}, backend={tree_backend})")
        best_params, grid_df = tune_lgbm_cv(
            train_df, feature_cols, cv=cv, random_state=random_state, backend=tree_backend
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        grid_df.to_csv(out_dir / "lgbm_grid_cv.csv", index=False)
        lgbm_tuned = make_lgbm_pipeline(
            feature_cols, lgbm_params=best_params, random_state=random_state, backend=tree_backend
        )
        fit_baseline(lgbm_tuned, train_df, feature_cols)
        rows.append(
            eval_pipeline_on_split(
                split,
                lgbm_tuned,
                feature_cols,
                classifier="lgbm",
                hp_policy="cv_tuned",
                experiment_id="lgbm_cvec5_emb_tuned",
                oracle_path=oracle_path,
                tree_backend=tree_backend,
            )
        )
        if verbose:
            print(f"    best LGBM params: {best_params}")

    table = pd.DataFrame(rows)
    if "tree_backend" in table.columns:
        table["tree_backend"] = table["tree_backend"].where(table["tree_backend"].notna(), None)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in splits:
        sub = table[table["split"] == split]
        sub.to_csv(out_dir / f"baselines_{split}.csv", index=False)
    table.to_csv(out_dir / "baselines_all.csv", index=False)

    summary = _baseline_summary(table)
    summary["tree_backend"] = tree_backend
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if verbose:
        print(f"\nWrote {out_dir / 'baselines_all.csv'}")
        print(json.dumps(summary, indent=2))

    return table


def _baseline_summary(table: pd.DataFrame) -> dict[str, Any]:
    """Refreeze recommendation from test regret vs frozen HGBM."""
    test = table[table["split"] == "test"].copy()
    summary: dict[str, Any] = {
        "utility_lambda": DEFAULT_UTILITY_LAMBDA,
        "refreeze_regret_max": REFREEZE_REGRET_MAX,
        "rows": table.to_dict(orient="records"),
    }
    if test.empty:
        summary["note"] = "no test split in table"
        return summary

    hgbm_rows = test[(test["classifier"] == "hgbm") & (test["hp_policy"] == "frozen")]
    if len(hgbm_rows):
        hgbm_regret = float(hgbm_rows.iloc[0]["mean_utility_regret"])
        summary["hgbm_frozen_test_regret"] = hgbm_regret

    trainable = test[~((test["classifier"] == "hgbm") & (test["hp_policy"] == "frozen"))]
    if len(trainable):
        best_idx = trainable["mean_utility_regret"].astype(float).idxmin()
        best = trainable.loc[best_idx]
        summary["best_trainable"] = {
            "classifier": best["classifier"],
            "hp_policy": best["hp_policy"],
            "experiment_id": best["experiment_id"],
            "mean_utility_regret": float(best["mean_utility_regret"]),
            "exec_em": float(best["exec_em"]),
            "macro_f1": float(best["macro_f1"]),
        }
        regret = float(best["mean_utility_regret"])
        summary["refreeze_recommended"] = regret <= REFREEZE_REGRET_MAX
        summary["regret_gap_vs_threshold"] = round(regret - REFREEZE_REGRET_MAX, 6)
        summary["refreeze_note"] = (
            "Regret threshold met for best trainable model. Before refreezing production, "
            "confirm val regret, mean_cost_usd, and exec_em (not regret alone)."
        )
        below = trainable[trainable["mean_utility_regret"].astype(float) <= REFREEZE_REGRET_MAX]
        if len(below):
            summary["models_below_regret_threshold"] = below[
                ["classifier", "hp_policy", "mean_utility_regret", "exec_em", "mean_cost_usd"]
            ].to_dict(orient="records")

    regrets = test.set_index(["classifier", "hp_policy"])["mean_utility_regret"].astype(float).to_dict()
    summary["test_regret_by_model"] = {f"{k[0]}_{k[1]}": v for k, v in regrets.items()}

    if len(trainable) >= 2 and len(hgbm_rows):
        spread = float(trainable["mean_utility_regret"].max() - trainable["mean_utility_regret"].min())
        summary["trainable_regret_spread"] = round(spread, 6)
        if spread < 0.02:
            summary["interpretation"] = (
                "Classifier families yield similar test regret; bottleneck is likely "
                "features/labels/oracle signal, not HGBM vs logreg vs LGBM."
            )

    return summary


def main_baselines(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Step 2: logreg / LightGBM baselines on cvec5_emb (test regret primary)."
    )
    p.add_argument("--split", choices=("val", "test", "both"), default="test")
    p.add_argument("--out-dir", type=Path, default=BASELINES_OUT_DIR)
    p.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    p.add_argument("--cv", type=int, default=5, help="CV folds for LGBM tune (train only)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    splits = ("val", "test") if args.split == "both" else (args.split,)
    run_classifier_baselines(
        splits=splits,
        out_dir=args.out_dir,
        oracle_path=args.oracle,
        cv=args.cv,
        random_state=args.seed,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main_baselines()
