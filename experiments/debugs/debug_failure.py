#!/usr/bin/env python3
"""
experiments/debug_failures.py
Shows the exact error and column details for every dataset.

Usage:
    python experiments/debug_failures.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import traceback

datasets = ["gaia", "mmlu_pro", "musique"]

for name in datasets:
    parquet = Path(f"datasets/processed/{name}.parquet")
    print(f"\n{'═'*58}")
    print(f"  {name.upper()}")
    print(f"{'═'*58}")

    if not parquet.exists():
        print(f"  ✗  Parquet not found: {parquet}")
        continue

    try:
        df = pd.read_parquet(parquet)
        print(f"  Shape   : {df.shape}")
        print(f"  Columns : {list(df.columns)}")
        print()

        # Print dtype + null count + first raw value for every column
        print(f"  {'Column':<22} {'dtype':<12} {'nulls':>6}  first value")
        print(f"  {'─'*22} {'─'*12} {'─'*6}  {'─'*30}")
        for col in df.columns:
            dtype = str(df[col].dtype)
            nulls = int(df[col].isna().sum())
            first = repr(df[col].iloc[0])[:60]
            print(f"  {col:<22} {dtype:<12} {nulls:>6}  {first}")

        # Dataset-specific checks
        print()
        if name == "gaia":
            if "level" in df.columns:
                print(
                    f"  level values   : {df['level'].value_counts().sort_index().to_dict()}"
                )
            if "complexity_gt" in df.columns:
                print(
                    f"  complexity_gt  : {df['complexity_gt'].value_counts().sort_index().to_dict()}"
                )
            else:
                print("  ✗  complexity_gt column MISSING")
            nulls_q = df["query"].isna().sum()
            empty_q = (df["query"].str.strip() == "").sum()
            print(f"  query nulls    : {nulls_q}")
            print(f"  query empty    : {empty_q}")

        elif name == "mmlu_pro":
            if "options" in df.columns:
                import numpy as np

                sample = df["options"].iloc[0]
                print(f"  options[0] type : {type(sample).__name__}")
                print(f"  options[0] val  : {repr(sample)[:80]}")
                n_opts = df["options"].apply(
                    lambda x: len(x) if isinstance(x, list) else 0
                )
                print(f"  options mean len: {n_opts.mean():.1f}")
            if "cot_content" in df.columns:
                blank = (df["cot_content"].fillna("").str.strip() == "").sum()
                print(f"  cot blank rows  : {blank}/{len(df)}")
                print(f"  cot sample      : {repr(df['cot_content'].iloc[0])[:100]}")

        elif name == "musique":
            if "num_hops" in df.columns:
                print(
                    f"  num_hops dist  : {df['num_hops'].value_counts().sort_index().to_dict()}"
                )
            else:
                print("  ✗  num_hops column MISSING")
            if "complexity_gt" in df.columns:
                print(
                    f"  complexity_gt  : {df['complexity_gt'].value_counts().sort_index().to_dict()}"
                )
            else:
                print("  ✗  complexity_gt column MISSING")
            if "answerable" in df.columns:
                print(f"  answerable dist: {df['answerable'].value_counts().to_dict()}")
            # Check query nulls
            nulls_q = df["query"].isna().sum()
            print(f"  query nulls    : {nulls_q}")

    except Exception:
        print(f"  ✗  Exception reading parquet:")
        traceback.print_exc()
