"""
src/loaders/mmlu_pro_loader.py
────────────────────────────────────────────────────────────────
Loader for MMLU-Pro (TIGER-Lab/MMLU-Pro).

Root cause of options being empty:
  HuggingFace's .to_pandas() silently converts Sequence columns
  (like options) into empty numpy ndarrays when the Arrow schema
  has a nested list type. Fix: extract options directly from the
  HF Dataset object BEFORE calling .to_pandas().
"""

from __future__ import annotations

import ast
import logging

import numpy as np
import pandas as pd
from datasets import load_dataset

from .base import BaseLoader

logger = logging.getLogger(__name__)


class MMLUProLoader(BaseLoader):

    DATASET_NAME = "mmlu_pro"

    def _download(self) -> pd.DataFrame:
        hf_repo = self.config.get("hf_repo", "TIGER-Lab/MMLU-Pro")
        split = self.config.get("split", "test")

        logger.info(f"  HF repo: {hf_repo}  split: {split}")
        ds = load_dataset(hf_repo, split=split)

        # ── Convert to pandas normally ────────────────────────────
        df = ds.to_pandas()

        # ── Re-extract options directly from the Arrow dataset ────
        # .to_pandas() collapses nested list columns to empty ndarrays.
        # Pulling from ds["options"] gives us proper Python lists.
        try:
            options_col = [list(row) for row in ds["options"]]
            df["options"] = options_col
            sample_len = len(options_col[0]) if options_col else 0
            logger.info(f"  Re-extracted options: {sample_len} choices per question")
        except Exception as e:
            logger.warning(f"  Could not re-extract options directly: {e}")

        return df

    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # ── Create id if missing ──────────────────────────────────
        if "question_id" not in df.columns:
            df["question_id"] = [f"mmlu_pro_{i}" for i in range(len(df))]

        # ── Rename to canonical columns ───────────────────────────
        df = df.rename(
            columns={
                "question_id": "id",
                "question": "query",
                "answer": "answer",
            }
        )

        # ── Keep useful columns ───────────────────────────────────
        keep = [
            "id",
            "query",
            "answer",
            "answer_index",
            "options",
            "category",
            "cot_content",
            "src",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # ── Normalise options → plain Python list ─────────────────
        # Handles every format we might receive:
        #   list        → use directly
        #   ndarray     → .tolist()
        #   str "[...]" → ast.literal_eval
        #   empty/None  → None (marked missing)
        def to_list(val):
            if isinstance(val, list):
                return val if len(val) > 0 else None
            if isinstance(val, np.ndarray):
                lst = val.tolist()
                return lst if len(lst) > 0 else None
            if isinstance(val, str):
                stripped = val.strip()
                if stripped.startswith("["):
                    try:
                        parsed = ast.literal_eval(stripped)
                        if isinstance(parsed, list) and len(parsed) > 0:
                            return parsed
                    except Exception:
                        pass
            return None

        if "options" in df.columns:
            df["options"] = df["options"].apply(to_list)
            n_valid = df["options"].notna().sum()
            n_missing = df["options"].isna().sum()
            logger.info(f"  options: {n_valid} valid, {n_missing} missing/empty")

        # ── CoT length proxy ──────────────────────────────────────
        if "cot_content" in df.columns:
            df["cot_length"] = df["cot_content"].fillna("").str.len()
            blank = (df["cot_length"] == 0).sum()
            if blank == len(df):
                logger.warning("  cot_content is entirely blank in this HF version")
            logger.info(f"  cot_length: mean={df['cot_length'].mean():.0f} chars")

        # ── Stratified sample across categories ───────────────────
        max_n = self.config.get("max_samples")
        if max_n and len(df) > max_n and "category" in df.columns:
            per_cat = max_n // df["category"].nunique()
            df = (
                df.groupby("category", group_keys=False)
                .apply(lambda g: g.sample(n=min(per_cat, len(g)), random_state=42))
                .reset_index(drop=True)
            )
            logger.info(f"  Stratified sample → {len(df)} rows")

        df = df.reset_index(drop=True)
        return df

    def _extra_verify(self, df: pd.DataFrame) -> bool:
        ok = True

        if "category" in df.columns:
            vc = df["category"].value_counts()
            print(f"  Categories: {df['category'].nunique()} unique")
            print(f"  Top 3    : {vc.head(3).to_dict()}")

        if "options" in df.columns:
            n_opts = df["options"].apply(lambda x: len(x) if isinstance(x, list) else 0)
            print(
                f"  Options/q : mean={n_opts.mean():.1f}  "
                f"min={n_opts.min()}  max={n_opts.max()}"
            )
            if n_opts.mean() < 4:
                print("  ⚠  options still empty — see troubleshooting below")
                print("     Run: python experiments/debug_mmlu.py")
                ok = False
            else:
                print(f"  Options  : OK  ({n_opts.mean():.0f} per question)")

        if "cot_length" in df.columns:
            mean_cot = df["cot_length"].mean()
            if mean_cot == 0:
                print("  CoT len  : 0 (blank in this HF version — not an error)")
            else:
                print(f"  CoT len  : mean={mean_cot:.0f} chars")

        return ok
