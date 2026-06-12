"""Gate 1a ablation: P(solvable|cvec5) vs frozen P(solvable|cvec5_emb).

Gate 1b, Gate 2, Gate 3 frozen. Only Gate 1a feature set changes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config.common import REPO_ROOT
from config.global_config.paths import daar_models_dir, daar_routing_dir, ensure_daar_dirs
from daar.routing import (
    LAMBDA_GRID,
    build_query_table,
    evaluate_lambda,
    load_gate1,
    load_gate1a,
    pick_theta_joint_regret,
    sweep_theta_joint_regret,
)
import importlib.util


def _train_gate1a_module():
    path = Path(__file__).with_name("train_gate1a_solvability.py")
    spec = importlib.util.spec_from_file_location("train_gate1a_solvability", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

GATE1A_CVEC5_STEM = "solvability_gate1a_cvec5"
GATE1A_CVEC5_MANIFEST = "solvability_gate1a_cvec5_manifest.json"
ABLATION_MODELS_DIR = (
    REPO_ROOT / "archive" / "daar_thesis_freeze" / "exploratory" / "models"
)


def _decomposed_metrics(
    *,
    gate1a_pipe: Any,
    gate1a_cols: list[str],
    theta_s: float,
    lam: float,
    models_dir: Path,
) -> dict[str, float]:
    gate1b, _man, gate1b_cols = load_gate1("decomposed", models_dir)
    table = build_query_table(
        "val",
        variant="decomposed",
        gate1=gate1b,
        feature_cols=gate1b_cols,
        gate1a_pipe=gate1a_pipe,
        gate1a_feature_cols=gate1a_cols,
        cost_norm="per_agent",
    )
    m = evaluate_lambda(table, lam, theta_s=theta_s, abstention="p_solv")
    return {
        "decomposed_val_regret": float(m["mean_agent_regret"]),
        "decomposed_val_success": float(m["success_rate"]),
        "decomposed_abstention_rate": float(m["abstention_rate"]),
    }


def _joint_tune(
    *,
    gate1a_pipe: Any,
    gate1a_cols: list[str],
    models_dir: Path,
) -> tuple[float, float, dict[str, Any]]:
    gate1b, _man, gate1b_cols = load_gate1("decomposed", models_dir)
    table = build_query_table(
        "val",
        variant="decomposed",
        gate1=gate1b,
        feature_cols=gate1b_cols,
        gate1a_pipe=gate1a_pipe,
        gate1a_feature_cols=gate1a_cols,
        cost_norm="per_agent",
    )
    joint_sweep = sweep_theta_joint_regret(table, lambdas=LAMBDA_GRID)
    theta_joint, lambda_joint = pick_theta_joint_regret(joint_sweep)
    return theta_joint, lambda_joint, {
        "joint_sweep_rows": len(joint_sweep),
        "best_theta_s": theta_joint,
        "best_lambda": lambda_joint,
    }


def run_ablation(
    *,
    models_dir: Path | None = None,
    ablation_models_dir: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    ensure_daar_dirs()
    models_dir = models_dir or daar_models_dir()
    ablation_models_dir = ablation_models_dir or ABLATION_MODELS_DIR
    ablation_models_dir.mkdir(parents=True, exist_ok=True)

    # Frozen reference (cvec5_emb Gate 1a)
    emb_pipe, emb_manifest, emb_cols = load_gate1a(models_dir)
    emb_theta_joint = float(emb_manifest.get("theta_s_joint_regret", 0.05))
    emb_lam = float(emb_manifest.get("lambda_at_theta_joint", 0.0))
    emb_theta_f1 = float(emb_manifest.get("theta_s", 0.4))
    emb_frozen = {
        "feature_set": emb_manifest.get("feature_set", "cvec5_emb"),
        "val_roc_auc": emb_manifest.get("val_roc_auc"),
        "theta_s_f1": emb_theta_f1,
        "theta_s_joint_regret": emb_theta_joint,
        "lambda_at_theta_joint": emb_lam,
        **_decomposed_metrics(
            gate1a_pipe=emb_pipe,
            gate1a_cols=emb_cols,
            theta_s=emb_theta_joint,
            lam=emb_lam,
            models_dir=models_dir,
        ),
    }
    emb_f1_route = _decomposed_metrics(
        gate1a_pipe=emb_pipe,
        gate1a_cols=emb_cols,
        theta_s=emb_theta_f1,
        lam=0.0,
        models_dir=models_dir,
    )

    # Train Gate 1a on cvec5 only (ablation artifact — does not overwrite frozen model)
    cvec5_train = _train_gate1a_module().train_gate1a_solvability(
        feature_set="cvec5",
        models_dir=ablation_models_dir,
        model_stem=GATE1A_CVEC5_STEM,
        manifest_name=GATE1A_CVEC5_MANIFEST,
        write_artifacts=True,
    )
    cvec5_pipe = cvec5_train["pipeline"]
    cvec5_cols = list(cvec5_train["feature_cols"])
    cvec5_theta_f1 = float(cvec5_train["theta_s"])

    theta_joint, lambda_joint, joint_info = _joint_tune(
        gate1a_pipe=cvec5_pipe,
        gate1a_cols=cvec5_cols,
        models_dir=models_dir,
    )

    # Update ablation manifest with joint tune (not the frozen gate1a manifest)
    cvec5_manifest_path = ablation_models_dir / GATE1A_CVEC5_MANIFEST
    cvec5_manifest = json.loads(cvec5_manifest_path.read_text(encoding="utf-8"))
    cvec5_manifest["theta_s_joint_regret"] = theta_joint
    cvec5_manifest["lambda_at_theta_joint"] = lambda_joint
    cvec5_manifest_path.write_text(json.dumps(cvec5_manifest, indent=2) + "\n", encoding="utf-8")

    cvec5_joint = _decomposed_metrics(
        gate1a_pipe=cvec5_pipe,
        gate1a_cols=cvec5_cols,
        theta_s=theta_joint,
        lam=lambda_joint,
        models_dir=models_dir,
    )
    cvec5_f1_route = _decomposed_metrics(
        gate1a_pipe=cvec5_pipe,
        gate1a_cols=cvec5_cols,
        theta_s=cvec5_theta_f1,
        lam=0.0,
        models_dir=models_dir,
    )

    report: dict[str, Any] = {
        "experiment": "gate1a_cvec5_ablation",
        "question": "Does Gate 1a need embeddings, or is cvec5 enough for abstention?",
        "protocol": {
            "gate1b": "frozen cvec5",
            "gate2": "frozen cost_hand",
            "gate3": "lambda from joint tune per Gate1a variant",
            "gate1a_train": "all 1050 queries, binary solvable",
        },
        "gate1a_cvec5_emb_frozen": emb_frozen,
        "gate1a_cvec5_emb_f1_theta_route": {
            "theta_s": emb_theta_f1,
            "lambda": 0.0,
            **emb_f1_route,
        },
        "gate1a_cvec5_ablation": {
            "feature_set": "cvec5",
            "model_path": str(ablation_models_dir / f"{GATE1A_CVEC5_STEM}.joblib"),
            "val_roc_auc": cvec5_train.get("val_roc_auc"),
            "val_accuracy": cvec5_train.get("val_accuracy"),
            "theta_s_f1": cvec5_theta_f1,
            "theta_s_joint_regret": theta_joint,
            "lambda_at_theta_joint": lambda_joint,
            "joint_tune": joint_info,
            **cvec5_joint,
        },
        "gate1a_cvec5_f1_theta_route": {
            "theta_s": cvec5_theta_f1,
            "lambda": 0.0,
            **cvec5_f1_route,
        },
        "comparison_joint_theta": [
            {
                "gate1a_features": "cvec5_emb",
                "decomposed_val_regret": emb_frozen["decomposed_val_regret"],
                "decomposed_val_success": emb_frozen["decomposed_val_success"],
                "abstention_rate": emb_frozen["decomposed_abstention_rate"],
            },
            {
                "gate1a_features": "cvec5",
                "decomposed_val_regret": cvec5_joint["decomposed_val_regret"],
                "decomposed_val_success": cvec5_joint["decomposed_val_success"],
                "abstention_rate": cvec5_joint["decomposed_abstention_rate"],
            },
        ],
    }

    out_path = out_path or daar_routing_dir() / "gate1a_cvec5_ablation.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["report_path"] = str(out_path)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--models-dir", type=Path, default=None, help="frozen Gate 1a/1b dir")
    p.add_argument(
        "--ablation-models-dir",
        type=Path,
        default=None,
        help="Gate 1a cvec5 ablation artifacts (default: archive/exploratory/models)",
    )
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    print(
        json.dumps(
            run_ablation(
                models_dir=args.models_dir,
                ablation_models_dir=args.ablation_models_dir,
                out_path=args.out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
