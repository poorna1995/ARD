#!/usr/bin/env python3
"""
Apply MATH strip_string normalization to gold ``answer`` columns in parquets.

Updates in place:
  - datasets/processed/math/{train,test}/data.parquet
  - datasets/train_samples/math.parquet
  - datasets/eval_samples/math.parquet
  - datasets/train_samples/combined.parquet (math rows only)
  - datasets/train_samples/combined_raw.parquet (math rows only)

Usage:
  python scripts/normalize_math_data.py
  python scripts/normalize_math_data.py --reprocess   # also run download.py --reprocess
  python scripts/normalize_math_data.py --regen-samples  # patch + rebuild train/eval samples
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.math_latex import normalize_math_answer  # noqa: E402


def _normalize_df_answers(df: pd.DataFrame, label: str) -> int:
    if "answer" not in df.columns:
        print(f"  skip {label}: no answer column")
        return 0
    before = df["answer"].copy()
    df["answer"] = df["answer"].apply(normalize_math_answer)
    changed = int((before != df["answer"]).sum())
    print(f"  {label}: {len(df):,} rows, {changed:,} answers updated")
    return changed


def _patch_file(path: Path) -> int:
    if not path.exists():
        print(f"  skip (missing): {path}")
        return 0
    df = pd.read_parquet(path)
    n = _normalize_df_answers(df, path.name)
    df.to_parquet(path, index=False)
    return n


def _patch_combined(path: Path) -> int:
    if not path.exists():
        print(f"  skip (missing): {path}")
        return 0
    df = pd.read_parquet(path)
    if "dataset_source" in df.columns:
        mask = df["dataset_source"] == "math"
    elif "dataset_training_id" in df.columns:
        mask = df["dataset_training_id"].astype(str).str.startswith("math_")
    else:
        print(f"  skip {path.name}: cannot identify math rows")
        return 0
    if not mask.any():
        return 0
    before = df.loc[mask, "answer"].copy()
    df.loc[mask, "answer"] = df.loc[mask, "answer"].apply(normalize_math_answer)
    changed = int((before != df.loc[mask, "answer"]).sum())
    print(f"  {path.name}: {mask.sum():,} math rows, {changed:,} answers updated")
    df.to_parquet(path, index=False)
    return changed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reprocess",
        action="store_true",
        help="Run scripts/download.py --dataset math --reprocess first",
    )
    parser.add_argument(
        "--regen-samples",
        action="store_true",
        help="After patching, run create_train_samples.py and create_eval_samples.py",
    )
    args = parser.parse_args()

    if args.reprocess:
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "download.py"),
            "--dataset",
            "math",
            "--reprocess",
        ]
        print("→", " ".join(cmd))
        subprocess.check_call(cmd, cwd=ROOT)
        if not args.regen_samples:
            print("\nReprocess done (processed/math updated via loader).")
            return

    print("\nPatching answer columns with MATH strip_string…")
    total = 0
    for split in ("train", "test"):
        total += _patch_file(ROOT / "datasets" / "processed" / "math" / split / "data.parquet")

    for name in ("math.parquet",):
        total += _patch_file(ROOT / "datasets" / "train_samples" / name)
        total += _patch_file(ROOT / "datasets" / "eval_samples" / name)

    for name in ("combined.parquet", "combined_raw.parquet"):
        total += _patch_combined(ROOT / "datasets" / "train_samples" / name)

    print(f"\nTotal answer cells updated: {total:,}")

    if args.regen_samples:
        for script in ("create_train_samples.py", "create_eval_samples.py"):
            cmd = [sys.executable, str(ROOT / "scripts" / script)]
            print("→", " ".join(cmd))
            subprocess.check_call(cmd, cwd=ROOT)

    # Spot-check
    sample = ROOT / "datasets" / "train_samples" / "math.parquet"
    if sample.exists():
        df = pd.read_parquet(sample)
        row = df[df["id"] == "math_06441"]
        if len(row):
            print(f"\nSpot check math_06441 answer: {row.iloc[0]['answer']!r}")

    print("\nDone. Gold answers are MATH-normalized.")


if __name__ == "__main__":
    main()
