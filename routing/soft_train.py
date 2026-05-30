"""
Soft-target router training (Experiment C).

Target: ``[p_raw, p_cot, p_react, p_multiagent]`` from split CSVs.
Loss: cross-entropy / KL — expand each query to four weighted rows ``(x, agent=k, weight=p_k)``.

Models: ``--classifier hgbm`` (default) or ``--classifier logreg``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from routing.analysis import DEFAULT_ORACLE
from routing.baselines import eval_frozen_hgbm, eval_pipeline_on_split, make_logreg_pipeline
from routing.config import (
    PRIMARY_ROUTER_EXPERIMENT_ID,
    PRODUCTION_FEATURE_SET,
    PRODUCTION_HGBM_PARAMS,
    PRODUCTION_ROUTER_EXPERIMENT_ID,
    REPO_ROOT,
    ROUTER_ROLE_BY_CLASSIFIER,
    SECONDARY_ROUTER_EXPERIMENT_ID,
    SOFT_KL_SEED_STABILITY_DIR,
    normalize_feature_set,
    soft_kl_experiment_id,
    soft_kl_out_dir,
)
from routing.router import (
    AGENTS,
    PROBA_COLS,
    ROUTER_SPEC,
    TrainSpec,
    evaluate,
    load_split_for_feature_set,
    make_pipeline,
    router_feature_cols,
    save_router,
    validate_feature_set_data,
)
from src.utils.soft_labels import SOFT_SUM_COL

ClassifierName = Literal["hgbm", "logreg"]

SOFT_KL_EXPERIMENT_IDS: dict[ClassifierName, str] = {
    "hgbm": "hgbm_cvec5_emb_soft_kl",
    "logreg": "logreg_cvec5_emb_soft_kl",
}
SOFT_KL_OUT_DIR = REPO_ROOT / "results/experiments/soft_kl_cvec5_emb"
BASELINES_CSV = REPO_ROOT / "results/experiments/classifier_baselines_cvec5_emb/baselines_all.csv"


def soft_kl_model_path(classifier: ClassifierName, feature_set: str = PRODUCTION_FEATURE_SET) -> Path:
    exp = soft_kl_experiment_id(classifier, feature_set)
    return REPO_ROOT / "models/router/graph_main" / f"{exp}.joblib"


def soft_target_matrix(df: pd.DataFrame) -> np.ndarray:
    """Row-wise soft targets in ``AGENTS`` / ``PROBA_COLS`` order; rows sum to 1 when mass > 0."""
    missing = [c for c in PROBA_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"missing soft label columns: {missing}")
    p = df[PROBA_COLS].astype(float).to_numpy(copy=True)
    row_sum = p.sum(axis=1, keepdims=True)
    np.divide(p, row_sum, out=p, where=row_sum > 0)
    return p


def soft_train_mask(df: pd.DataFrame) -> np.ndarray:
    """Rows with positive soft mass (at least one correct agent in oracle)."""
    if SOFT_SUM_COL in df.columns:
        return df[SOFT_SUM_COL].astype(float).to_numpy() > 0
    return soft_target_matrix(df).sum(axis=1) > 0


def expand_soft_multiclass_rows(
    df: pd.DataFrame,
    feature_cols: list[str],
    p: np.ndarray,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """
    Expand each query into ``len(AGENTS)`` rows for weighted multiclass CE.

    Row ``(x_i, class=k)`` with weight ``p_ik`` → ``KL(p_i || q_i)`` up to a constant.
    """
    parts_x: list[pd.DataFrame] = []
    labels: list[str] = []
    weights: list[float] = []
    for i in range(len(df)):
        row = df.iloc[[i]]
        pi = p[i]
        for k, agent in enumerate(AGENTS):
            parts_x.append(row)
            labels.append(agent)
            weights.append(float(pi[k]))
    x_exp = pd.concat(parts_x, ignore_index=True)
    return x_exp, pd.Series(labels, dtype=str), np.asarray(weights, dtype=float)


def make_soft_kl_pipeline(
    feature_cols: list[str],
    classifier: ClassifierName,
    *,
    random_state: int = 42,
    hgbm_params: dict[str, Any] | None = None,
) -> Pipeline:
    """Prep + classifier; soft mass enters via expanded-row weights (no hard ``class_weight``)."""
    if classifier == "logreg":
        return make_logreg_pipeline(feature_cols, random_state=random_state, class_weight=None)
    return make_pipeline(
        feature_cols,
        class_weight=None,
        random_state=random_state,
        hgbm_params=hgbm_params,
        calibrated=False,
    )


def fit_soft_kl_router(
    pipe: Pipeline,
    train_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Pipeline, int, int]:
    """Fit on expanded soft rows; return (pipe, n_queries, n_expanded_rows)."""
    mask = soft_train_mask(train_df)
    sub = train_df.loc[mask]
    if sub.empty:
        raise ValueError("no training rows with positive soft label mass")
    p = soft_target_matrix(sub)
    x_exp, y_exp, w_exp = expand_soft_multiclass_rows(sub, feature_cols, p)
    pipe.fit(x_exp[feature_cols], y_exp, clf__sample_weight=w_exp)
    return pipe, len(sub), len(x_exp)


def train_soft_kl_router(
    *,
    classifier: ClassifierName = "hgbm",
    spec: TrainSpec | None = None,
    random_state: int = 42,
) -> tuple[Pipeline, list[str], dict[str, Any]]:
    """Train on train split; return pipeline, feature columns, and training meta."""
    spec = spec or ROUTER_SPEC
    fs = normalize_feature_set(spec.feature_set)
    experiment_id = soft_kl_experiment_id(classifier, fs)
    train_df = load_split_for_feature_set("train", fs)
    validate_feature_set_data(train_df, fs)
    feature_cols = router_feature_cols(train_df, fs)
    hgbm_params = spec.hgbm_params if spec.hgbm_params is not None else PRODUCTION_HGBM_PARAMS
    pipe = make_soft_kl_pipeline(
        feature_cols, classifier, random_state=random_state, hgbm_params=hgbm_params
    )
    pipe, n_queries, n_expanded = fit_soft_kl_router(pipe, train_df, feature_cols)
    val_df = load_split_for_feature_set("val", fs)
    val_label = evaluate(pipe, val_df, feature_cols)
    meta: dict[str, Any] = {
        "classifier": classifier,
        "experiment_id": experiment_id,
        "feature_set": fs,
        "target": "soft_p",
        "loss": "kl_soft_ce",
        "n_train_queries": n_queries,
        "n_train_expanded_rows": n_expanded,
        "val_macro_f1_hard_label": val_label.macro_f1,
        "val_accuracy_hard_label": val_label.accuracy,
    }
    if classifier == "hgbm":
        meta["hgbm_params"] = "default" if spec.hgbm_params is None else spec.hgbm_params
    return pipe, feature_cols, meta


def eval_soft_kl_splits(
    pipe: Pipeline,
    feature_cols: list[str],
    classifier: ClassifierName,
    splits: tuple[str, ...] = ("val", "test"),
    *,
    feature_set: str = PRODUCTION_FEATURE_SET,
    oracle_path: Path = DEFAULT_ORACLE,
) -> pd.DataFrame:
    """Utility regret + EM (same harness as Step 2 baselines)."""
    fs = normalize_feature_set(feature_set)
    experiment_id = soft_kl_experiment_id(classifier, fs)
    rows = []
    for split in splits:
        rows.append(
            eval_pipeline_on_split(
                split,
                pipe,
                feature_cols,
                classifier=classifier,
                hp_policy="soft_kl",
                experiment_id=experiment_id,
                oracle_path=oracle_path,
                feature_set=fs,
            )
        )
    return pd.DataFrame(rows)


def _test_regret_from_baselines(clf: str, hp_policy: str) -> float | None:
    if not BASELINES_CSV.is_file():
        return None
    t = pd.read_csv(BASELINES_CSV)
    sub = t[(t["split"] == "test") & (t["classifier"] == clf) & (t["hp_policy"] == hp_policy)]
    if len(sub):
        return float(sub.iloc[0]["mean_utility_regret"])
    return None


def _test_regret_from_soft_summary(clf: ClassifierName) -> float | None:
    for path in (
        SOFT_KL_OUT_DIR / f"soft_kl_metrics_{clf}.csv",
        SOFT_KL_OUT_DIR / f"{clf}_soft_kl_summary.json",
    ):
        if not path.is_file():
            continue
        if path.suffix == ".csv":
            sub = pd.read_csv(path)
            test = sub[sub["split"] == "test"]
            if len(test):
                return float(test.iloc[0]["mean_utility_regret"])
        else:
            data = json.loads(path.read_text(encoding="utf-8"))
            for m in data.get("metrics", []):
                if m.get("split") == "test":
                    return float(m["mean_utility_regret"])
    return None


def run_soft_kl_seed_sweep(
    *,
    classifier: ClassifierName = "logreg",
    feature_set: str = PRODUCTION_FEATURE_SET,
    splits: tuple[str, ...] = ("val", "test"),
    seeds: tuple[int, ...] = (42, 123, 456, 789, 2024),
    out_dir: Path | None = None,
    verbose: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Retrain soft-KL model per seed; report utility regret (primary) and hard-label macro-F1."""
    fs = normalize_feature_set(feature_set)
    out_dir = Path(out_dir or soft_kl_out_dir(fs))
    out_dir.mkdir(parents=True, exist_ok=True)
    experiment_id = soft_kl_experiment_id(classifier, fs)

    rows: list[dict[str, Any]] = []
    for seed in seeds:
        spec = TrainSpec(
            experiment_id=experiment_id,
            feature_set=fs,
            hgbm_params=PRODUCTION_HGBM_PARAMS,
            weight_mode="none",
            use_class_weight=False,
            random_state=seed,
        )
        pipe, feature_cols, _ = train_soft_kl_router(
            classifier=classifier, spec=spec, random_state=seed
        )
        for split in splits:
            m = eval_pipeline_on_split(
                split,
                pipe,
                feature_cols,
                classifier=classifier,
                hp_policy="soft_kl",
                experiment_id=experiment_id,
                feature_set=fs,
            )
            label = evaluate(pipe, load_split_for_feature_set(split, fs), feature_cols)
            row = {
                "classifier": classifier,
                "seed": seed,
                "split": split,
                "mean_utility_regret": m["mean_utility_regret"],
                "exec_em": m["exec_em"],
                "mean_cost_usd": m["mean_cost_usd"],
                "macro_f1_hard_label": label.macro_f1,
            }
            rows.append(row)
            if verbose:
                print(
                    f"  {classifier} seed={seed} {split} "
                    f"regret={m['mean_utility_regret']:.4f} em={m['exec_em']:.2f}"
                )

    per_seed = pd.DataFrame(rows)
    per_seed.to_csv(out_dir / f"per_seed_{classifier}.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for split in splits:
        sub = per_seed[per_seed["split"] == split]
        summary_rows.append(
            {
                "classifier": classifier,
                "split": split,
                "regret_mean": float(sub["mean_utility_regret"].mean()),
                "regret_std": float(sub["mean_utility_regret"].std(ddof=0)),
                "regret_min": float(sub["mean_utility_regret"].min()),
                "regret_max": float(sub["mean_utility_regret"].max()),
                "exec_em_mean": float(sub["exec_em"].mean()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / f"summary_{classifier}.csv", index=False)
    if verbose:
        print(summary.to_string(index=False))
    return per_seed, summary


def comparison_table(soft_logreg_test: float | None = None) -> list[dict[str, Any]]:
    """Four-way test regret: hard/soft × hgbm/logreg."""
    soft_hgbm = _test_regret_from_soft_summary("hgbm")
    soft_logreg = soft_logreg_test if soft_logreg_test is not None else _test_regret_from_soft_summary("logreg")
    return [
        {
            "model": "Hard HGBM",
            "test_regret": _test_regret_from_baselines("hgbm", "frozen"),
        },
        {
            "model": "Hard Logistic",
            "test_regret": _test_regret_from_baselines("logreg", "default"),
        },
        {"model": "Soft HGBM (p_*)", "test_regret": soft_hgbm},
        {"model": "Soft Logistic (p_*)", "test_regret": soft_logreg},
    ]


def main_soft_train(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Train with soft p_* targets (KL / soft CE).")
    p.add_argument(
        "--classifier",
        choices=("hgbm", "logreg"),
        default="hgbm",
        help="Model family (default: hgbm)",
    )
    p.add_argument("--save", action="store_true", help="Write joblib under models/router/graph_main/")
    p.add_argument("--split", choices=("val", "test", "both"), default="both")
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Metrics output dir (default: soft_kl_cvec5_emb or feature_ablation_soft_kl/<fs>)",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    p.add_argument(
        "--seed-sweep",
        action="store_true",
        help="Run multi-seed stability on val+test regret (no save unless --save).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456, 789, 2024],
    )
    p.add_argument(
        "--feature-set",
        default=PRODUCTION_FEATURE_SET,
        help="Feature set id (emb_only | heur_emb | cvec5_emb, …)",
    )
    args = p.parse_args(argv)

    clf: ClassifierName = args.classifier
    fs = normalize_feature_set(args.feature_set)

    if args.seed_sweep:
        run_soft_kl_seed_sweep(
            classifier=clf,
            feature_set=fs,
            splits=("val", "test"),
            seeds=tuple(args.seeds),
            out_dir=args.out_dir or soft_kl_out_dir(fs),
            verbose=not args.quiet,
        )
        return

    experiment_id = soft_kl_experiment_id(clf, fs)
    out_dir = args.out_dir or soft_kl_out_dir(fs)

    spec = TrainSpec(
        experiment_id=experiment_id,
        feature_set=fs,
        hgbm_params=PRODUCTION_HGBM_PARAMS,
        weight_mode="none",
        use_class_weight=False,
        random_state=args.seed,
    )
    pipe, feature_cols, meta = train_soft_kl_router(
        classifier=clf, spec=spec, random_state=args.seed
    )

    splits = ("val", "test") if args.split == "both" else (args.split,)
    table = eval_soft_kl_splits(pipe, feature_cols, clf, splits=splits, feature_set=fs)
    out_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir = out_dir
    table.to_csv(args.out_dir / f"soft_kl_metrics_{clf}.csv", index=False)

    if "test" in splits:
        soft_test = float(table.loc[table["split"] == "test", "mean_utility_regret"].iloc[0])
        meta["comparison"] = {"soft_kl_test_regret": soft_test}
        if fs == PRODUCTION_FEATURE_SET:
            hard_hgbm = eval_frozen_hgbm("test")
            meta["comparison"]["hard_hgbm_test_regret"] = float(hard_hgbm["mean_utility_regret"])
            meta["comparison"]["delta_vs_hard_hgbm"] = soft_test - float(
                hard_hgbm["mean_utility_regret"]
            )
            meta["four_way_test_regret"] = comparison_table(
                soft_logreg_test=soft_test if clf == "logreg" else None
            )

    summary = {"training": meta, "metrics": table.to_dict(orient="records")}
    summary_path = args.out_dir / f"{clf}_soft_kl_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (args.out_dir / "soft_kl_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if args.save:
        path = save_router(
            pipe,
            feature_cols=feature_cols,
            experiment_id=experiment_id,
            path=soft_kl_model_path(clf, fs),
            extra_meta={
                **meta,
                "target": "soft_p",
                "loss": "kl_soft_ce",
                "router_role": ROUTER_ROLE_BY_CLASSIFIER[clf],
                "production": experiment_id == PRODUCTION_ROUTER_EXPERIMENT_ID,
                "primary_router": PRIMARY_ROUTER_EXPERIMENT_ID,
                "secondary_router": SECONDARY_ROUTER_EXPERIMENT_ID,
            },
        )
        if not args.quiet:
            print("Saved", path)

    if not args.quiet:
        print(json.dumps(summary, indent=2))
        if meta.get("four_way_test_regret"):
            comp = pd.DataFrame(meta["four_way_test_regret"])
            comp.to_csv(args.out_dir / "four_way_test_regret.csv", index=False)
            print("\n--- Test regret comparison ---")
            for row in meta["four_way_test_regret"]:
                r = row["test_regret"]
                rs = f"{r:.6f}" if r is not None else "n/a"
                print(f"  {row['model']:<22} {rs}")


if __name__ == "__main__":
    main_soft_train()
