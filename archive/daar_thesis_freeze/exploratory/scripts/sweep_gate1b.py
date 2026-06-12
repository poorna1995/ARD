"""Gate 1b sweep — HGBM hyperparameters and feature ablation (success-only soft-KL)."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import pandas as pd

from config.global_config.paths import daar_models_dir, daar_routing_dir
from config.local.router.production import PRODUCTION_FEATURE_SET
from daar.routing import build_query_table, evaluate_lambda, load_gate1a


def _train_gate1b_module():
    path = Path(__file__).with_name("train_gate1b_success_soft.py")
    spec = importlib.util.spec_from_file_location("train_gate1b_success_soft", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# Default router.pipeline HGBM baseline
DEFAULT_HGBM: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.05,
    "min_samples_leaf": 10,
    "max_leaf_nodes": 31,
}

HGBM_PARAM_GRID: list[dict[str, Any]] = [
    DEFAULT_HGBM,
    {"max_depth": 4, "learning_rate": 0.05, "min_samples_leaf": 20, "max_leaf_nodes": 15},
    {"max_depth": 4, "learning_rate": 0.1, "min_samples_leaf": 10, "max_leaf_nodes": 31},
    {"max_depth": 6, "learning_rate": 0.03, "min_samples_leaf": 10, "max_leaf_nodes": 31},
    {"max_depth": 6, "learning_rate": 0.05, "min_samples_leaf": 5, "max_leaf_nodes": 63},
    {"max_depth": 6, "learning_rate": 0.1, "min_samples_leaf": 15, "max_leaf_nodes": 31},
    {"max_depth": 8, "learning_rate": 0.05, "min_samples_leaf": 10, "max_leaf_nodes": 63},
    {"max_depth": 8, "learning_rate": 0.1, "min_samples_leaf": 5, "max_leaf_nodes": 127},
    {"max_depth": 10, "learning_rate": 0.05, "min_samples_leaf": 5, "max_leaf_nodes": 127},
]

FEATURE_ABLATION_SETS: list[str] = [
    "cvec5",  # dim_* only
    "emb_only",
    "cvec5_emb",  # dim + emb (production)
]


def _decomposed_val_regret(
    pipe: Any,
    feature_cols: list[str],
    *,
    models_dir: Path,
) -> dict[str, float]:
    """Full pipeline regret with frozen Gate 1a θ/λ from manifest."""
    _pipe, gate1a_manifest, _ = load_gate1a(models_dir)
    theta_s = float(
        gate1a_manifest.get(
            "theta_s_joint_regret",
            gate1a_manifest.get("theta_s", 0.5),
        )
    )
    lam = float(
        gate1a_manifest.get(
            "lambda_at_theta_joint",
            0.0,
        )
    )
    table = build_query_table(
        "val",
        variant="decomposed",
        gate1=pipe,
        feature_cols=feature_cols,
        cost_norm="per_agent",
    )
    m = evaluate_lambda(table, lam, theta_s=theta_s, abstention="p_solv")
    return {
        "decomposed_val_regret": float(m["mean_agent_regret"]),
        "decomposed_val_success": float(m["success_rate"]),
        "decomposed_theta_s": theta_s,
        "decomposed_lambda": lam,
    }


def _run_trial(
    *,
    feature_set: str,
    hgbm_params: dict[str, Any],
    random_state: int,
    models_dir: Path,
    trial_id: str,
) -> dict[str, Any]:
    train_gate1b = _train_gate1b_module().train_gate1b_success_soft
    result = train_gate1b(
        feature_set=feature_set,
        random_state=random_state,
        hgbm_params=hgbm_params,
        models_dir=models_dir,
        model_stem=f"gate1b_sweep_{trial_id}",
        write_model=False,
    )
    pipe = result.pop("pipeline")
    feature_cols = list(result["feature_cols"])
    result.update(
        _decomposed_val_regret(pipe, feature_cols, models_dir=models_dir)
    )
    result["trial_id"] = trial_id
    return result


def sweep_gate1b(
    *,
    mode: str = "all",
    random_state: int = 42,
    models_dir: Path | None = None,
    promote_best: bool = False,
) -> dict[str, Any]:
    models_dir = models_dir or daar_models_dir()
    rows: list[dict[str, Any]] = []

    if mode in ("all", "hgbm"):
        for i, params in enumerate(HGBM_PARAM_GRID):
            rows.append(
                _run_trial(
                    feature_set=PRODUCTION_FEATURE_SET,
                    hgbm_params=params,
                    random_state=random_state,
                    models_dir=models_dir,
                    trial_id=f"hgbm_{i:02d}",
                )
            )

    if mode in ("all", "features"):
        for fs in FEATURE_ABLATION_SETS:
            if fs == PRODUCTION_FEATURE_SET and mode == "all":
                continue  # already covered by hgbm grid with default params
            rows.append(
                _run_trial(
                    feature_set=fs,
                    hgbm_params=DEFAULT_HGBM,
                    random_state=random_state,
                    models_dir=models_dir,
                    trial_id=f"feat_{fs}",
                )
            )

    df = pd.DataFrame(rows).sort_values(
        "decomposed_val_regret", ascending=True, kind="mergesort"
    )
    best = df.iloc[0].to_dict()

    report: dict[str, Any] = {
        "gate": "1b",
        "supervision": "success_only_soft_kl",
        "mode": mode,
        "n_trials": len(rows),
        "best_trial": best,
        "baseline_reference": {
            "gate1b_default_cvec5_emb": {
                "note": "pre-sweep production gate1b",
                "decomposed_val_regret_approx": 0.129,
            },
        },
        "trials": df.to_dict(orient="records"),
    }

    out_json = daar_routing_dir() / "gate1b_sweep_results.json"
    out_csv = daar_routing_dir() / "gate1b_sweep_results.csv"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    df.to_csv(out_csv, index=False)
    report["report_path"] = str(out_json)
    report["csv_path"] = str(out_csv)

    if promote_best:
        train_gate1b = _train_gate1b_module().train_gate1b_success_soft
        promoted = train_gate1b(
            feature_set=str(best["feature_set"]),
            random_state=random_state,
            hgbm_params=best.get("hgbm_params") if isinstance(best.get("hgbm_params"), dict) else None,
            models_dir=models_dir,
            write_model=True,
        )
        report["promoted_model"] = {
            "model_path": promoted.get("model_path"),
            "val_mean_agent_regret": promoted.get("val_mean_agent_regret"),
            "decomposed_val_regret": best.get("decomposed_val_regret"),
        }

    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["all", "hgbm", "features"], default="all")
    p.add_argument("--random-state", type=int, default=42)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument(
        "--promote-best",
        action="store_true",
        help="retrain and save best trial as production gate1b",
    )
    args = p.parse_args()
    print(
        json.dumps(
            sweep_gate1b(
                mode=args.mode,
                random_state=args.random_state,
                models_dir=args.models_dir,
                promote_best=args.promote_best,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
