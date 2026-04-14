#!/usr/bin/env python3
"""
experiments/debug_mmlu.py
Run this to see exactly what's in the options and cot_content columns.

Usage:
    python experiments/debug_mmlu.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

parquet = Path("datasets/processed/mmlu_pro.parquet")
if not parquet.exists():
    print("parquet not found — downloading raw now...")
    from datasets import load_dataset

    ds = load_dataset("TIGER-Lab/MMLU-Pro", split="test", trust_remote_code=True)
    df = ds.to_pandas()
    print("Downloaded raw df, shape:", df.shape)
else:
    df = pd.read_parquet(parquet)
    print("Loaded parquet, shape:", df.shape)

print("\n=== ALL COLUMN NAMES ===")
print(list(df.columns))

print("\n=== options column ===")
print("dtype       :", df["options"].dtype)
print("type of [0] :", type(df["options"].iloc[0]).__name__)
print("value of [0]:", repr(df["options"].iloc[0])[:300])
print("value of [1]:", repr(df["options"].iloc[1])[:300])

print("\n=== cot_content column ===")
print(
    "dtype       :",
    df["cot_content"].dtype if "cot_content" in df.columns else "MISSING",
)
if "cot_content" in df.columns:
    print("type of [0] :", type(df["cot_content"].iloc[0]).__name__)
    print("value of [0]:", repr(str(df["cot_content"].iloc[0]))[:200])

print("\n=== answer column ===")
print("sample values:", df["answer"].head(5).tolist())

print("\n=== Full first row ===")
row = df.iloc[0]
for col in df.columns:
    print(f"  {col:<20} {type(row[col]).__name__:<12} {repr(row[col])[:80]}")
