#!/usr/bin/env python3
"""
Mandatory pre-training data inspection.

Run before any training code:
    python scripts/inspect_training_data.py

Sets random seeds to 42, loads train/eval benchmark parquets, prints
standard checks and dataset-specific validations.
"""

from __future__ import annotations

import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Seeds (must be first) ────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
try:
    import torch

    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    _TORCH = f"torch {torch.__version__}"
except ImportError:
    _TORCH = "torch not installed"

print(f"Random seeds set to {SEED} (numpy, random, {_TORCH})\n")

GAIA_FILES_ROOT = ROOT / "datasets" / "gaia_files" / "2023" / "validation"

DATASETS = {
    "MATH (train sample)": {
        "path": ROOT / "datasets/train_samples/math.parquet",
        "question_col": "query",
        "answer_col": "answer",
        "role": "train",
    },
    "HotpotQA (train sample)": {
        "path": ROOT / "datasets/train_samples/hotpot.parquet",
        "question_col": "query",
        "answer_col": "answer",
        "role": "train",
    },
    "MuSiQue (train sample)": {
        "path": ROOT / "datasets/train_samples/musique.parquet",
        "question_col": "query",
        "answer_col": "answer",
        "role": "train",
    },
    "MMLU-Pro (eval sample)": {
        "path": ROOT / "datasets/eval_samples/mmlu.parquet",
        "question_col": "query",
        "answer_col": "answer",
        "role": "eval",
    },
    "GAIA (eval / full val)": {
        "path": ROOT / "datasets/eval_samples/gaia.parquet",
        "question_col": "query",
        "answer_col": "answer",
        "role": "eval",
    },
}

FULL_POOLS = {
    "MATH processed train": ROOT / "datasets/processed/math/train/data.parquet",
    "Hotpot processed train": ROOT / "datasets/processed/hotpot/train/data.parquet",
    "MuSiQue processed train": ROOT / "datasets/processed/musique/train/data.parquet",
}


def _sep(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def _standard_inspection(name: str, df: pd.DataFrame, question_col: str, answer_col: str) -> None:
    _sep(name)
    print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nNull counts per column:")
    print(df.isna().sum().to_string())
    print("\nFirst 3 rows:")
    with pd.option_context("display.max_colwidth", 80, "display.width", 120):
        print(df.head(3).to_string())

    qcol = question_col if question_col in df.columns else None
    acol = answer_col if answer_col in df.columns else None
    if qcol and acol:
        bad_q = df[qcol].isna() | (df[qcol].astype(str).str.strip() == "")
        bad_a = df[acol].isna() | (df[acol].astype(str).str.strip() == "")
        n_drop = int((bad_q | bad_a).sum())
        print(f"\nCleaning check (null/empty {qcol!r} or {acol!r}): {n_drop} rows would be dropped")
    else:
        print(f"\nWARNING: question/answer columns not found (expected {question_col!r}, {answer_col!r})")


def _is_flat_string_series(s: pd.Series) -> bool:
    for v in s.dropna().head(50):
        if isinstance(v, (dict, list, np.ndarray)):
            return False
    return True


def _check_math(df: pd.DataFrame, full_train: pd.DataFrame | None) -> None:
    print("\n--- MATH-specific ---")
    level_col = "level" if "level" in df.columns else None
    if level_col is None and "difficulty" in df.columns:
        level_col = "difficulty"
    subject_col = "type" if "type" in df.columns else None
    print(f"Difficulty column: {level_col!r} (subject/strand column: {subject_col!r})")
    if level_col:
        print("\nCount per level (train sample):")
        print(df[level_col].value_counts().sort_index().to_string())
        ok = (df[level_col].value_counts() >= 100).all()
        print(f"  ≥100 per level in sample: {'YES' if ok else 'NO (sample is n=500 stratified)'}")
    if full_train is not None and level_col:
        print("\nCount per level (full processed train pool):")
        full_counts = full_train[level_col].value_counts().sort_index()
        print(full_counts.to_string())
        ok_full = (full_counts >= 100).all()
        print(f"  ≥100 per level in full train: {'YES' if ok_full else 'NO'}")

    print("\nLaTeX sanity check (20 random rows from query + solution):")
    _latex_check_sample(df, n=20)


def _latex_check_sample(df: pd.DataFrame, n: int = 20) -> None:
    cols = [c for c in ("query", "solution") if c in df.columns]
    if not cols:
        print("  skip: no query/solution columns")
        return
    sample = df.sample(n=min(n, len(df)), random_state=SEED)
    errors: list[str] = []
    parsed = 0
    for idx, row in sample.iterrows():
        for col in cols:
            text = str(row[col])
            err = _latex_issues(text)
            if err:
                errors.append(f"  id={row.get('id', idx)} col={col}: {err}")
            else:
                parsed += 1
    print(f"  checked {len(sample) * len(cols)} fields; ok={parsed}, issues={len(errors)}")
    for line in errors[:10]:
        print(line)
    if len(errors) > 10:
        print(f"  ... and {len(errors) - 10} more")


def _latex_issues(text: str) -> str | None:
    """Return error message if LaTeX looks broken; None if ok."""
    if not text or not isinstance(text, str):
        return "empty"
    # Brace balance (ignoring escaped braces is approximate)
    depth = 0
    for i, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                return "unbalanced }"
    if depth != 0:
        return f"unbalanced braces (depth={depth})"
    try:
        from sympy.parsing.latex import parse_latex

        # Only try parse on subsample with math delimiters to avoid false positives on plain text
        if "\\" in text or "$" in text:
            parse_latex(text[:500] if len(text) > 500 else text)
    except ImportError:
        return None  # sympy latex parser unavailable; brace check only
    except Exception as exc:
        return f"sympy parse: {type(exc).__name__}"
    return None


def _check_hotpot(df: pd.DataFrame) -> None:
    print("\n--- HotpotQA-specific ---")
    q_ok = "query" in df.columns and _is_flat_string_series(df["query"])
    a_ok = "answer" in df.columns and _is_flat_string_series(df["answer"])
    print(f"query flat string: {q_ok}  |  answer flat string: {a_ok}")
    sup_cols = [c for c in df.columns if "support" in c.lower() or "context" in c.lower()]
    if sup_cols:
        print(f"Supporting-fact columns present: {sup_cols}")
    else:
        print("Supporting facts: NOT in processed parquet (flat query/answer only)")
    if "level" in df.columns and "type" in df.columns:
        print("\nlevel × type counts:")
        print(pd.crosstab(df["level"], df["type"]).to_string())


def _check_musique(df: pd.DataFrame) -> None:
    print("\n--- MuSiQue-specific ---")
    if "n_hops" in df.columns:
        print("Hop count column: 'n_hops' (direct)")
        print(df["n_hops"].value_counts().sort_index().to_string())
    else:
        print("Hop count: deriving from id …")
        from src.data.musique_loader import _extract_hop_info

        derived = df["id"].astype(str).apply(_extract_hop_info).apply(pd.Series)
        print(derived["n_hops"].value_counts().sort_index().to_string())
    if "hop_name" in df.columns:
        print("\nn_hops × hop_name:")
        print(pd.crosstab(df["n_hops"], df["hop_name"]).to_string())
    if "id" in df.columns:
        sample_id = df["id"].iloc[0]
        print(f"\nExample id format: {sample_id!r}")


def _check_mmlu(df: pd.DataFrame) -> None:
    print("\n--- MMLU-Pro-specific ---")
    cat_col = None
    for cand in ("category", "subject", "discipline"):
        if cand in df.columns:
            cat_col = cand
            break
    print(f"Stratification column: {cat_col!r}")
    if cat_col:
        print(df[cat_col].value_counts().sort_index().to_string())
    if "options" in df.columns:
        n_list = sum(isinstance(x, list) for x in df["options"].dropna().head(20))
        print(f"\noptions column: sample of 20 — {n_list} are Python lists")
        empty_opts = df["options"].isna() | df["options"].apply(
            lambda x: x is None or (isinstance(x, list) and len(x) == 0)
        )
        print(f"  rows with missing/empty options: {empty_opts.sum()}")


def _check_gaia(df: pd.DataFrame) -> None:
    print("\n--- GAIA-specific ---")
    level_col = "level" if "level" in df.columns else None
    print(f"Difficulty column: {level_col!r} (values 1=easy, 2=medium, 3=hard)")
    if level_col:
        print(df[level_col].value_counts().sort_index().to_string())
    fname_col = "file_name" if "file_name" in df.columns else None
    if not fname_col:
        print("file_name column: NOT present — attachments not linked in this parquet")
        return
    has_file = df[fname_col].fillna("").astype(str).str.strip() != ""
    print(f"\nRows with file_name set: {has_file.sum()} / {len(df)}")
    missing: list[str] = []
    found = 0
    for _, row in df[has_file].iterrows():
        fn = str(row[fname_col]).strip()
        local = GAIA_FILES_ROOT / fn
        if local.exists():
            found += 1
        else:
            missing.append(fn)
    print(f"Attachments accessible under {GAIA_FILES_ROOT}:")
    print(f"  found: {found}  |  missing: {len(missing)}")
    if missing[:5]:
        print("  example missing:", missing[:5])


def main() -> None:
    math_full = None
    if FULL_POOLS["MATH processed train"].exists():
        math_full = pd.read_parquet(FULL_POOLS["MATH processed train"])

    for name, cfg in DATASETS.items():
        path: Path = cfg["path"]
        if not path.exists():
            print(f"SKIP {name}: not found at {path}")
            continue
        df = pd.read_parquet(path)
        _standard_inspection(name, df, cfg["question_col"], cfg["answer_col"])

        if "MATH" in name:
            _check_math(df, math_full)
        elif "Hotpot" in name:
            _check_hotpot(df)
        elif "MuSiQue" in name:
            _check_musique(df)
        elif "MMLU" in name:
            _check_mmlu(df)
        elif "GAIA" in name:
            _check_gaia(df)

    _sep("Full processed train pools (reference)")
    for label, path in FULL_POOLS.items():
        if path.exists():
            df = pd.read_parquet(path)
            print(f"{label}: {df.shape}")
        else:
            print(f"{label}: MISSING at {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()
