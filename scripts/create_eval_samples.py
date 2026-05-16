#!/usr/bin/env python3
"""
Create benchmark evaluation subsamples (seed=42) from processed test/val splits.

Outputs under datasets/eval_samples/:
  math.parquet      (200, test — level × subject)
  hotpot.parquet    (200, validation — level × type)
  musique.parquet   (200, validation — hop strata, ~66% unique answers)
  mmlu.parquet      (200, test — category)
  gaia.parquet      (~165, validation — full set)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.utils.train_samples import (  # noqa: E402
    EVAL_N,
    sample_gaia,
    sample_hotpot,
    sample_math,
    sample_mmlu,
    sample_musique,
)

SEED = 42
OUT_DIR = ROOT / "datasets" / "eval_samples"
PROC = ROOT / "datasets" / "processed"


def _print_dist(name: str, df: pd.DataFrame, cols: list[str]) -> None:
    print(f"\n{name} (n={len(df)})")
    print(df.groupby(cols, dropna=False).size().sort_index().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=EVAL_N, help="eval size for stratified sets")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    math_df = pd.read_parquet(PROC / "math" / "test" / "data.parquet")
    math_s = sample_math(math_df, args.n, args.seed)
    math_s.to_parquet(OUT_DIR / "math.parquet", index=False)
    _print_dist("math (test)", math_s, ["level", "type"])
    print(f"  → {OUT_DIR / 'math.parquet'}")

    hot_df = pd.read_parquet(PROC / "hotpot" / "validation" / "data.parquet")
    hot_s = sample_hotpot(hot_df, args.n, args.seed)
    hot_s.to_parquet(OUT_DIR / "hotpot.parquet", index=False)
    _print_dist("hotpot (validation)", hot_s, ["level", "type"])
    print(f"  → {OUT_DIR / 'hotpot.parquet'}")

    mus_df = pd.read_parquet(PROC / "musique" / "validation" / "data.parquet")
    mus_s = sample_musique(mus_df, args.n, args.seed)
    mus_s.to_parquet(OUT_DIR / "musique.parquet", index=False)
    _print_dist("musique (validation)", mus_s, ["n_hops", "hop_name"])
    n_ans = mus_s["answer"].nunique()
    print(f"  unique answers: {n_ans}/{len(mus_s)} ({100 * n_ans / len(mus_s):.1f}%)")
    print(f"  → {OUT_DIR / 'musique.parquet'}")

    mmlu_df = pd.read_parquet(PROC / "mmlu_pro" / "test" / "data.parquet")
    mmlu_s = sample_mmlu(mmlu_df, args.n, args.seed)
    mmlu_s.to_parquet(OUT_DIR / "mmlu.parquet", index=False)
    _print_dist("mmlu (test)", mmlu_s, ["category"])
    print(f"  → {OUT_DIR / 'mmlu.parquet'}")

    gaia_df = pd.read_parquet(PROC / "gaia.parquet")
    gaia_s = sample_gaia(gaia_df, n=None, seed=args.seed)
    gaia_s.to_parquet(OUT_DIR / "gaia.parquet", index=False)
    if "level" in gaia_s.columns:
        _print_dist("gaia (validation)", gaia_s, ["level"])
    print(f"  → {OUT_DIR / 'gaia.parquet'}")


if __name__ == "__main__":
    main()
