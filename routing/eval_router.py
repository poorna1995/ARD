"""Evaluate saved router artifacts on val/test splits vs baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from routing.train_router import (
    baseline_always_majority,
    baseline_per_dataset_mode,
    evaluate,
    load_router,
    load_split,
    predict_agent_proba,
    results_row,
)

DEFAULT_ROUTER = _REPO_ROOT / "models/router/G3_hgbm_graph_emb_balanced.joblib"


def eval_saved_router(
    artifact_path: Path,
    eval_df: pd.DataFrame,
    *,
    experiment_id: str | None = None,
) -> tuple[dict[str, Any], Any]:
    """Score one ``save_router`` joblib on ``eval_df``; returns (summary row, EvalResult)."""
    obj = load_router(artifact_path)
    pipe = obj["pipeline"]
    feature_cols: list[str] = list(obj["feature_cols"])
    exp_id = experiment_id or str(obj.get("experiment_id") or artifact_path.stem)

    missing = [c for c in feature_cols if c not in eval_df.columns]
    if missing:
        raise ValueError(
            f"{exp_id}: eval split missing feature columns {missing}. "
            "Build complexity_record_*.parquet and query_embeddings_*.parquet first."
        )

    result = evaluate(pipe, eval_df, feature_cols)
    row = results_row(
        exp_id,
        result,
        feature_set=obj.get("feature_set"),
        n_features=len(feature_cols),
        model="saved",
    )
    row["artifact"] = str(artifact_path)
    return row, result


def run_eval(
    *,
    split: str,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    router_paths: list[Path],
    verbose: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        results_row("B0_always_raw", baseline_always_majority(train_df, eval_df)),
        results_row("B0_dataset_mode", baseline_per_dataset_mode(train_df, eval_df)),
    ]

    details: list[tuple[str, Any]] = []
    for path in router_paths:
        row, result = eval_saved_router(path, eval_df)
        rows.append(row)
        details.append((row["experiment"], result))

    table = pd.DataFrame(rows)
    if verbose:
        print(f"\n=== Router eval on {split!r} (n={len(eval_df)}) ===\n")
        print(table.to_string(index=False))
        b0 = table[table["experiment"] == "B0_dataset_mode"].iloc[0]
        for exp_id, result in details:
            row = table[table["experiment"] == exp_id].iloc[0]
            lift = float(row["macro_f1"]) - float(b0["macro_f1"])
            acc_lift = float(row["accuracy"]) - float(b0["accuracy"])
            print(
                f"\n=== {exp_id} ===\n"
                f"macro-F1 {lift:+.4f} vs dataset_mode | accuracy {acc_lift:+.4f}\n"
                f"{result.report}"
            )

    return table


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate saved router joblibs on val or internal test."
    )
    p.add_argument(
        "--split",
        choices=("val", "test"),
        default="test",
        help="Eval split (default: test = qce_internal_test.csv).",
    )
    p.add_argument(
        "--router",
        type=Path,
        action="append",
        default=None,
        help="Path to .joblib artifact (repeatable). Default: G3 graph_emb balanced.",
    )
    p.add_argument(
        "--all-saved",
        action="store_true",
        help="Evaluate every models/router/*.joblib (can be slow).",
    )
    p.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Do not merge query_embeddings_*.parquet.",
    )
    p.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write results table as JSON records.",
    )
    p.add_argument(
        "--proba-head",
        type=int,
        default=0,
        metavar="N",
        help="Print first N rows of predict_proba for the first router (0=off).",
    )
    return p.parse_args()


def _resolve_router_paths(args: argparse.Namespace) -> list[Path]:
    if args.router:
        return [Path(p) for p in args.router]
    if args.all_saved:
        root = _REPO_ROOT / "models/router"
        return sorted(root.glob("*.joblib"))
    return [DEFAULT_ROUTER]


if __name__ == "__main__":
    args = _parse_args()
    use_emb = not args.no_embeddings
    train_df = load_split("train", with_embeddings=use_emb)
    eval_df = load_split(args.split, with_embeddings=use_emb)
    paths = _resolve_router_paths(args)

    missing_files = [p for p in paths if not p.is_file()]
    if missing_files:
        raise FileNotFoundError(
            "Missing router artifact(s): "
            + ", ".join(str(p) for p in missing_files)
            + ". Train with: uv run python routing/train_router.py --graph-only --save-all"
        )

    table = run_eval(
        split=args.split,
        train_df=train_df,
        eval_df=eval_df,
        router_paths=paths,
        verbose=True,
    )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(
            table.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
        print(f"\nWrote {args.json_out}")

    if args.proba_head > 0 and paths:
        obj = load_router(paths[0])
        proba = predict_agent_proba(
            obj["pipeline"], eval_df, obj["feature_cols"]
        )
        print(f"\n--- {paths[0].stem} predict_proba (head) ---")
        print(proba.head(args.proba_head).to_string())
