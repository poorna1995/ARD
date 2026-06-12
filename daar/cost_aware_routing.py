"""Cost-aware adaptive routing freeze spec and helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from config.global_config.paths import daar_routing_dir
from daar.gate3_policy import (
    COST_AWARE_ROUTING_FREEZE_JSON,
    COST_AWARE_ROUTING_FREEZE_VERSION,
    COST_EFFICIENT_OBJECTIVE,
    DEFAULT_DEPLOYMENT_POLICY,
    DEFAULT_MIN_SUCCESS_FRACTION,
    GATE3_OBJECTIVES_JSON,
)

FREEZE_SPEC = {
    "framework": "D-AAR cost-aware adaptive routing",
    "deployment_problem": (
        "Choose the cheapest agent sufficiently likely to succeed — "
        "not the highest-confidence agent."
    ),
    "gate1a": {
        "model": "P(pool-solvable | φ, emb)",
        "decision": "abstain if solvability_probability < θ",
        "theta_source": "models/daar/pool_solvability_manifest.json",
    },
    "gate1b": {
        "model": "ŝ_a = P(agent a succeeds | q, solvable)",
        "supervision": "success-only soft-KL on solvable train queries",
        "features": "cvec5",
    },
    "gate2": {
        "model": "ĉ_a = predicted execution cost(q, a)",
        "implementation": "D2 trace skeleton + D3 rule simulation (frozen train scales)",
    },
    "gate3": {
        "policy": "success_threshold",
        "feasible_set": "A(q) = { a : ŝ_a(q) ≥ τ_success }",
        "decision": (
            "â = argmin_{a∈A} ĉ_a(q) if |A|>0; else ABSTAIN"
        ),
        "empty_feasible_fallback": "abstain",
        "empty_feasible_rationale": (
            "Objective is minimize cost while maintaining success; "
            "if no agent meets minimum confidence τ, do not spend."
        ),
        "tau_meaning": (
            "Minimum confidence required before paying execution cost "
            "(operational threshold, not oracle correctness probability)."
        ),
        "tau_selection": (
            "On val: minimize mean route_cost_hat (Gate 2 estimate) subject to "
            "success_rate ≥ α × baseline_success (α=0.95); NOT regret or F1."
        ),
        "cost_metrics": {
            "primary": "mean_route_cost_hat (Gate 2; what Gate 3 minimizes)",
            "calibration": "mean_oracle_cost_usd (post-hoc baseline execution; not at route time)",
        },
    },
    "legacy_ablation": {
        "policy": "utility rank-only",
        "rule": "argmax ŝ (λ=0) or regret-tuned λ",
        "note": "accuracy baseline only — not deployment objective",
    },
}


def build_cost_aware_routing_freeze(
    objectives_doc: dict[str, Any] | None = None,
    *,
    routing_dir: Path | None = None,
) -> dict[str, Any]:
    routing_dir = routing_dir or daar_routing_dir()
    if objectives_doc is None:
        path = routing_dir / GATE3_OBJECTIVES_JSON
        if not path.is_file():
            raise FileNotFoundError(
                f"missing {path} — run: uv run python scripts/eval_cost_efficient_routing.py"
            )
        objectives_doc = json.loads(path.read_text(encoding="utf-8"))

    frozen = objectives_doc.get("frozen_production", {})
    tau = float(frozen.get("success_threshold", 0.5))
    alpha = float(objectives_doc.get("min_success_fraction", DEFAULT_MIN_SUCCESS_FRACTION))
    meets_floor = bool(
        frozen.get("meets_success_floor", objectives_doc.get("tau_meets_success_floor", False))
    )
    metrics = objectives_doc.get("options", {}).get("option1_success_threshold", {}).get(
        "metrics", {}
    )

    return {
        "freeze_version": COST_AWARE_ROUTING_FREEZE_VERSION,
        "status": "frozen",
        "objective": COST_EFFICIENT_OBJECTIVE,
        **FREEZE_SPEC,
        "frozen_operating_point": {
            "gate3_policy": frozen.get("policy", DEFAULT_DEPLOYMENT_POLICY),
            "tau_success": tau,
            "min_success_fraction_alpha": alpha,
            "success_floor": objectives_doc.get("success_floor"),
            "meets_success_floor": meets_floor,
            "val_metrics_at_tau": metrics,
            "limitation_if_not_meets_floor": (
                None
                if meets_floor
                else (
                    "No τ in validation grid satisfies success ≥ α with Gate 3 abstain-on-empty-A; "
                    "τ frozen at best-coverage point. Improve Gate 1b ŝ or relax α."
                )
            ),
        },
        "canonical_scripts": [
            "scripts/eval_cost_efficient_routing.py",
            "scripts/eval_cost_efficient_on_eval_samples.py",
            "scripts/verify_cost_aware_routing_freeze.py",
            "scripts/infer_daar.py",
        ],
        "objectives_artifact": str(routing_dir / GATE3_OBJECTIVES_JSON),
    }


def write_cost_aware_routing_freeze(
    *,
    routing_dir: Path | None = None,
    objectives_doc: dict[str, Any] | None = None,
) -> Path:
    routing_dir = routing_dir or daar_routing_dir()
    routing_dir.mkdir(parents=True, exist_ok=True)
    doc = build_cost_aware_routing_freeze(objectives_doc, routing_dir=routing_dir)
    path = routing_dir / COST_AWARE_ROUTING_FREEZE_JSON
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    return path
