#!/usr/bin/env python3
"""
Create 500-row train samples (seed=42) from processed train splits.

Sampling (unchanged): MATH level×subject, Hotpot level×type,
MuSiQue 6 hop strata + ~66% unique answers.

Additive for training files only: ``training_id`` (MATH);
``training_id`` + ``dataset_source`` (Hotpot, MuSiQue).

Outputs:
  datasets/train_samples/{math,hotpot,musique}.parquet
  datasets/train_samples/combined.parquet  (1,500 rows; global training_id)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.train_samples import (
    combine_training_samples,
    sample_hotpot,
    sample_math,
    sample_musique,
)  # noqa: E402

SEED = 42
OUT_DIR = ROOT / "datasets" / "train_samples"
TRAIN_N = 500


def _print_dist(name: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n{name} (n={len(df)})")
    print(df.groupby(cols, dropna=False).size().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=TRAIN_N)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    proc = ROOT / "datasets" / "processed"

    math_df = pd.read_parquet(proc / "math" / "train" / "data.parquet")
    math_s = sample_math(math_df, args.n, args.seed, add_training_id=True)
    math_path = OUT_DIR / "math.parquet"
    math_s.to_parquet(math_path, index=False)
    _print_dist("math", math_s, ["level", "type"])
    assert math_s["training_id"].iloc[0] == "math_0"
    print(f"  → {math_path}")

    hot_df = pd.read_parquet(proc / "hotpot" / "train" / "data.parquet")
    hot_s = sample_hotpot(hot_df, args.n, args.seed, add_training_metadata=True)
    hot_path = OUT_DIR / "hotpot.parquet"
    hot_s.to_parquet(hot_path, index=False)
    assert hot_s["dataset_source"].eq("hotpotqa").all()
    assert hot_s["training_id"].iloc[0] == "hotpot_0"
    print(f"  → {hot_path}")

    mus_df = pd.read_parquet(proc / "musique" / "train" / "data.parquet")
    mus_s = sample_musique(mus_df, args.n, args.seed, add_training_metadata=True)
    mus_path = OUT_DIR / "musique.parquet"
    mus_s.to_parquet(mus_path, index=False)
    assert mus_s["dataset_source"].eq("musique").all()
    assert mus_s["training_id"].iloc[0] == "musique_0"
    n_ans = mus_s["answer"].nunique()
    print(f"  unique answers: {n_ans}/{len(mus_s)} ({100 * n_ans / len(mus_s):.1f}%)")
    _print_dist("musique", mus_s, ["n_hops", "hop_name"])
    print(f"  → {mus_path}")

    combined = combine_training_samples(math_s, hot_s, mus_s, expected_total=3 * args.n)
    combined_path = OUT_DIR / "combined.parquet"
    combined.to_parquet(combined_path, index=False)
    assert combined["training_id"].nunique() == len(combined)
    print(f"\ncombined: {len(combined)} rows, unique training_id → {combined_path}")

    print("\nAll train samples saved.")


if __name__ == "__main__":
    main()
