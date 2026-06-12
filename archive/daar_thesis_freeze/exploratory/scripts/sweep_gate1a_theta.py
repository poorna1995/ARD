"""Val θ sweep for a trained Gate 1a solvability head (no retrain)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from config.global_config.paths import daar_models_dir, daar_routing_dir
from daar.gate1 import (
    GATE1A_MANIFEST,
    GATE1A_MODEL_STEM,
    ThetaSelection,
    eval_abstention,
    load_frame,
    predict_solvable_proba,
    query_labels_with_solvable,
    solvable_series,
    sweep_theta_abstention,
    pick_theta_abstain,
)


def load_gate1a_model(
    models_dir: Path | None = None,
) -> tuple[Any, dict[str, Any], list[str]]:
    models_dir = models_dir or daar_models_dir()
    manifest_path = models_dir / GATE1A_MANIFEST
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"missing {manifest_path} — run: uv run python scripts/train_gate1a_solvability.py"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    model_path = Path(manifest.get("model_path", models_dir / f"{GATE1A_MODEL_STEM}.joblib"))
    obj = joblib.load(model_path)
    feature_cols = list(manifest.get("feature_cols", obj.get("feature_cols", [])))
    return obj["pipeline"], manifest, feature_cols


def sweep_gate1a_theta(
    split: str = "val",
    *,
    theta_strategy: ThetaSelection = "max_f1",
    max_false_abstention_solvable: float | None = None,
    models_dir: Path | None = None,
    frames_dir: Path | None = None,
    out_path: Path | None = None,
) -> dict[str, Any]:
    pipe, manifest, feature_cols = load_gate1a_model(models_dir)
    frame_long = load_frame(split, frames_dir=frames_dir)
    query = query_labels_with_solvable(frame_long)
    scores = predict_solvable_proba(pipe, query, feature_cols)
    solvable = solvable_series(frame_long).astype(int)

    sweep = sweep_theta_abstention(pd.Series(scores), solvable)
    theta_s = pick_theta_abstain(
        sweep,
        strategy=theta_strategy,
        max_false_abstention_solvable=max_false_abstention_solvable,
    )

    report: dict[str, Any] = {
        "split": split,
        "n_queries": len(query),
        "n_solvable": int(solvable.sum()),
        "n_unsolvable": int((solvable == 0).sum()),
        "theta_selected": theta_s,
        "theta_selection": theta_strategy,
        "max_false_abstention_solvable": max_false_abstention_solvable,
        "metrics_at_theta": eval_abstention(pd.Series(scores), solvable, theta_s),
        "sweep": sweep.to_dict(orient="records"),
        "model_manifest": manifest.get("manifest_path", str((models_dir or daar_models_dir()) / GATE1A_MANIFEST)),
    }

    out_path = out_path or daar_routing_dir() / f"gate1a_theta_sweep_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    sweep.to_csv(csv_path, index=False)
    report["report_path"] = str(out_path)
    report["sweep_csv"] = str(csv_path)
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument(
        "--theta-strategy",
        choices=["max_f1", "min_false_abstention"],
        default="max_f1",
    )
    p.add_argument("--max-false-abstention-solvable", type=float, default=None)
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--frames-dir", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    print(
        json.dumps(
            sweep_gate1a_theta(
                args.split,
                theta_strategy=args.theta_strategy,
                max_false_abstention_solvable=args.max_false_abstention_solvable,
                models_dir=args.models_dir,
                frames_dir=args.frames_dir,
                out_path=args.out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
