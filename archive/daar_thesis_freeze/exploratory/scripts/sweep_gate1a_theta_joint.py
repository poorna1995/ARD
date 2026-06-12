"""Tune Gate 1a θ on val using decomposed routing regret (not F1 alone)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from config.global_config.paths import daar_models_dir, daar_routing_dir
from daar.gate1 import GATE1A_MANIFEST
from daar.routing import (
    LAMBDA_GRID,
    build_query_table,
    load_gate1,
    load_gate1a,
    pick_theta_joint_regret,
    sweep_theta_joint_regret,
)


def sweep_and_update_gate1a_manifest(
    split: str = "val",
    *,
    models_dir: Path | None = None,
    lambdas: tuple[float, ...] = LAMBDA_GRID,
    write_manifest: bool = True,
    out_path: Path | None = None,
) -> dict[str, Any]:
    models_dir = models_dir or daar_models_dir()
    gate1b, _gate1b_manifest, feature_cols = load_gate1("decomposed", models_dir)
    table = build_query_table(
        split,
        variant="decomposed",
        gate1=gate1b,
        feature_cols=feature_cols,
        cost_norm="per_agent",
    )

    joint_sweep = sweep_theta_joint_regret(table, lambdas=lambdas)
    theta_joint, lambda_joint = pick_theta_joint_regret(joint_sweep)

    _pipe, gate1a_manifest, _cols = load_gate1a(models_dir)
    f1_theta = float(gate1a_manifest.get("theta_s", 0.5))

    report: dict[str, Any] = {
        "split": split,
        "n_queries": len(table),
        "theta_selection": "min_joint_regret",
        "cost_norm_mode": "per_agent",
        "theta_s_f1_reference": f1_theta,
        "theta_s_joint_regret": theta_joint,
        "lambda_at_theta_joint": lambda_joint,
        "joint_sweep": joint_sweep.to_dict(orient="records"),
    }

    if write_manifest:
        gate1a_manifest["theta_s_joint_regret"] = theta_joint
        gate1a_manifest["lambda_at_theta_joint"] = lambda_joint
        gate1a_manifest["theta_joint_selection"] = "min_joint_regret"
        gate1a_manifest["theta_joint_split"] = split
        gate1a_manifest["cost_norm_at_joint_tune"] = "per_agent"
        manifest_path = Path(
            gate1a_manifest.get("manifest_path", models_dir / GATE1A_MANIFEST)
        )
        manifest_path.write_text(json.dumps(gate1a_manifest, indent=2) + "\n", encoding="utf-8")
        report["gate1a_manifest_updated"] = str(manifest_path)

    out_path = out_path or daar_routing_dir() / f"gate1a_theta_joint_{split}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    csv_path = out_path.with_suffix(".csv")
    joint_sweep.to_csv(csv_path, index=False)
    report["report_path"] = str(out_path)
    report["sweep_csv"] = str(csv_path)

    # F1-optimal row for comparison
    if "mean_agent_regret" in joint_sweep.columns:
        f1_row = joint_sweep.loc[joint_sweep["theta_s"].astype(float) == f1_theta]
        if not f1_row.empty:
            report["regret_at_f1_theta"] = float(f1_row.iloc[0]["mean_agent_regret"])
        best_row = joint_sweep.loc[joint_sweep["mean_agent_regret"].idxmin()]
        report["regret_at_joint_theta"] = float(best_row["mean_agent_regret"])

    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--split", default="val", choices=["val"])
    p.add_argument("--models-dir", type=Path, default=None)
    p.add_argument("--no-write-manifest", action="store_true")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    print(
        json.dumps(
            sweep_and_update_gate1a_manifest(
                args.split,
                models_dir=args.models_dir,
                write_manifest=not args.no_write_manifest,
                out_path=args.out,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
