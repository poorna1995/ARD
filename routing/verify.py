"""
Production health check — load primary router, predict golden rows, verify paths.

Universe: both (paths for A internal QCE + B live eval)
IN:  config paths, tests/golden/inference_universe_a.parquet
MID: load router · run inference · check critical files exist
OUT: exit 0 + summary text, or exit 1 on failure
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from config.global_config.paths import oracle_results_csv
from config.global_config.runtime import (
    REQUIRED_AGENT_KEYS,
    configure_logging,
    format_env_report,
    verify_env_keys,
)
from config.common import REPO_ROOT
from router.config import (
    EVAL_SAMPLES_DIR,
    LIVE_EVAL_BASELINE_ROOT,
    PCA_PATH,
    PRIMARY_ROUTER_PATH,
    ROUTER_MODEL_PATH,
    SPLIT_CSV,
    TRAIN_NORM_JSON,
)
from router.router import PROBA_COLS, load_router_frame

GOLDEN_DIR = REPO_ROOT / "tests" / "golden"
GOLDEN_MANIFEST = GOLDEN_DIR / "manifest.json"


def _required_paths() -> list[tuple[str, Path, bool]]:
    """(label, path, required). Optional paths warn only."""
    oracle = oracle_results_csv()
    return [
        ("primary_router", PRIMARY_ROUTER_PATH, True),
        ("router_model_alias", ROUTER_MODEL_PATH, True),
        ("qce_test_split", SPLIT_CSV["test"], True),
        ("oracle_results", oracle, True),
        ("eval_samples_dir", EVAL_SAMPLES_DIR, True),
        ("train_norm", TRAIN_NORM_JSON, True),
        ("pca_embeddings", PCA_PATH, True),
        ("live_eval_baseline_root", LIVE_EVAL_BASELINE_ROOT, False),
        ("golden_manifest", GOLDEN_MANIFEST, False),
    ]


def check_paths(*, verbose: bool = True) -> list[str]:
    """Return list of error messages (empty if all required paths ok)."""
    errors: list[str] = []
    for label, path, required in _required_paths():
        ok = path.exists()
        if ok:
            if verbose:
                print(f"  OK  {label}: {path.relative_to(REPO_ROOT)}")
        elif required:
            errors.append(f"missing required {label}: {path}")
            if verbose:
                print(f"  FAIL {label}: {path.relative_to(REPO_ROOT)}")
        elif verbose:
            print(f"  skip {label} (optional): {path.relative_to(REPO_ROOT)}")
    return errors


def run_golden_inference(*, atol: float = 1e-6, verbose: bool = True) -> list[str]:
    """Compare load_router_frame on golden parquet vs expected JSON."""
    errors: list[str] = []
    manifest_path = GOLDEN_MANIFEST
    if not manifest_path.is_file():
        return ["golden manifest missing: tests/golden/manifest.json"]

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    uni = manifest.get("universe_a", {})
    feat_path = REPO_ROOT / uni.get("features_parquet", "tests/golden/inference_universe_a.parquet")
    exp_path = REPO_ROOT / uni.get("expected_json", "tests/golden/expected_predictions_universe_a.json")
    atol = float(uni.get("atol", atol))

    if not feat_path.is_file():
        return [f"golden features missing: {feat_path}"]
    if not exp_path.is_file():
        return [f"golden expected missing: {exp_path}"]

    import pandas as pd

    df = pd.read_parquet(feat_path)
    expected: dict[str, Any] = json.loads(exp_path.read_text(encoding="utf-8"))
    router_path = manifest.get("primary_router", str(PRIMARY_ROUTER_PATH.relative_to(REPO_ROOT)))
    routed = load_router_frame(df, router_path=REPO_ROOT / router_path)

    for tid, exp in expected.items():
        if tid not in routed["training_id"].astype(str).values:
            errors.append(f"golden training_id missing in routed output: {tid}")
            continue
        row = routed[routed["training_id"].astype(str) == tid].iloc[0]
        if str(row["router_pred"]) != str(exp["router_pred"]):
            errors.append(f"{tid}: router_pred {row['router_pred']!r} != {exp['router_pred']!r}")
        for col in PROBA_COLS + ["max_prob"]:
            if col not in exp:
                continue
            got = float(row[col])
            want = float(exp[col])
            if not np.isclose(got, want, rtol=0, atol=atol):
                errors.append(f"{tid}: {col} {got} != {want} (atol={atol})")

    if verbose and not errors:
        print(f"  OK  golden inference ({len(expected)} rows, atol={atol})")
    return errors


def run_verify(
    *,
    skip_golden: bool = False,
    require_secrets: bool = False,
    verbose: bool = True,
) -> int:
    configure_logging()
    if verbose:
        print("routing verify — paths")
    path_errors = check_paths(verbose=verbose)

    secret_errors: list[str] = []
    if verbose:
        print("routing verify — secrets")
    secret = verify_env_keys(require=REQUIRED_AGENT_KEYS if require_secrets else ())
    if verbose:
        print(f"  {format_env_report(secret)}")
    if require_secrets and not secret.ok:
        secret_errors = [f"missing secrets: {', '.join(secret.missing_required)}"]

    golden_errors: list[str] = []
    if not skip_golden:
        if verbose:
            print("routing verify — golden inference (universe A)")
        golden_errors = run_golden_inference(verbose=verbose)

    errors = path_errors + secret_errors + golden_errors
    if errors:
        if verbose:
            print("\nFAILED:")
            for e in errors:
                print(f"  - {e}")
        return 1
    if verbose:
        print("\nOK — routing verify passed")
    return 0


def main_verify(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Health check: paths + golden router inference.")
    p.add_argument("--skip-golden", action="store_true", help="Only check paths (no model inference).")
    p.add_argument(
        "--require-secrets",
        action="store_true",
        help="Fail if OPENAI_API_KEY is unset (needed for agent runs / decompose).",
    )
    p.add_argument("-q", "--quiet", action="store_true")
    args = p.parse_args(argv)
    code = run_verify(
        skip_golden=args.skip_golden,
        require_secrets=args.require_secrets,
        verbose=not args.quiet,
    )
    sys.exit(code)


if __name__ == "__main__":
    main_verify()
