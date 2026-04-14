"""
src/loaders/aime_loader.py
────────────────────────────────────────────────────────────────
Loader for AIME (American Invitational Mathematics Examination).

AIME problems are competition math questions with integer answers
in [0, 999]. They are significantly harder than MATH benchmark
and most current LLMs fail on them, making AIME useful for
probing the "very hard" end of complexity estimators.

HuggingFace: Maxwell-Jia/AIME_1983_2024

Canonical columns:
  id       → derived from Year + Part + Problem number
  query    → Problem
  answer   → Answer (integer as string)

Extra columns kept:
  Year, Part (AIME I / II), difficulty_proxy
"""

from __future__ import annotations

import logging

import pandas as pd
from datasets import load_dataset

from .base import BaseLoader

logger = logging.getLogger(__name__)


class AIMELoader(BaseLoader):

    DATASET_NAME = "aime"

    def _download(self) -> pd.DataFrame:
        hf_repo = self.config.get("hf_repo", "Maxwell-Jia/AIME_1983_2024")
        split = self.config.get("split", "train")

        logger.info(f"  HF repo: {hf_repo}  split: {split}")

        try:
            ds = load_dataset(hf_repo, split=split)
            return ds.to_pandas()
        except Exception as e:
            logger.warning(f"  Primary repo failed ({e}). Trying alternate...")
            # Try alternate repo
            try:
                ds = load_dataset("AI-MO/aimo-validation-aime", split="train")
                return ds.to_pandas()
            except Exception as e2:
                raise RuntimeError(
                    f"AIME: Both repos failed.\n"
                    f"Primary error: {e}\nAlternate error: {e2}"
                )

    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # ── Normalize column names (repo schemas vary) ────────────
        # Common variants seen across different HF repos:
        col_aliases = {
            "Problem":  "query",
            "problem":  "query",
            "question": "query",
            "Answer":   "answer",
            "answer":   "answer",
            "solution": "solution",
            "Year":     "year",
            "year":     "year",
            "Part":     "part",
        }
        df = df.rename(columns={k: v for k, v in col_aliases.items()
                                  if k in df.columns and v not in df.columns})

        # ── Create canonical ID ───────────────────────────────────
        if "id" not in df.columns:
            if "year" in df.columns and "part" in df.columns:
                df["id"] = (
                    "aime_"
                    + df["year"].astype(str)
                    + "_"
                    + df["part"].astype(str).str.replace(" ", "_").str.lower()
                    + "_p"
                    + (df.groupby(["year", "part"]).cumcount() + 1).astype(str)
                )
            else:
                df["id"] = [f"aime_{i:04d}" for i in range(len(df))]

        # ── Ensure answer is a string ─────────────────────────────
        df["answer"] = df["answer"].astype(str).str.strip()

        # ── Year filter (recent problems are harder) ──────────────
        year_gte = self.config.get("filter", {}).get("year_gte")
        if year_gte and "year" in df.columns:
            before = len(df)
            df["year"] = pd.to_numeric(df["year"], errors="coerce")
            df = df[df["year"] >= year_gte].copy()
            logger.info(f"  Year filter (>= {year_gte}): {before} → {len(df)} rows")

        # # ── Complexity proxy ──────────────────────────────────────
        # # AIME II is systematically harder than AIME I.
        # # All AIME problems are intrinsically hard, but we can
        # # create a within-AIME difficulty scale.
        # if "part" in df.columns:
        #     df["difficulty_proxy"] = df["part"].apply(
        #         lambda p: 0.85 if "II" in str(p) else 0.75
        #     )
        # else:
        #     df["difficulty_proxy"] = 0.80  # default high difficulty

        # df["complexity_gt"] = df["difficulty_proxy"]

        # ── Keep needed columns ───────────────────────────────────
        keep = ["id", "query", "answer", "year", "part",
                # "complexity_gt",
                
                 "solution"]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # ── max_samples not applied here (dataset is small, ~900) ─
        df = df.reset_index(drop=True)
        return df

    def _extra_verify(self, df: pd.DataFrame) -> bool:
        ok = True
        if "year" in df.columns:
            print(f"  Year range: {df['year'].min()} – {df['year'].max()}")
        if "part" in df.columns:
            print(f"  Parts    : {df['part'].value_counts().to_dict()}")
        # Check answers are integers in [0, 999]
        try:
            ans_int = pd.to_numeric(df["answer"], errors="coerce")
            invalid = ans_int.isna().sum()
            out_of_range = ((ans_int < 0) | (ans_int > 999)).sum()
            if invalid > 0:
                print(f"  ⚠  {invalid} non-integer answers")
                ok = False
            if out_of_range > 0:
                print(f"  ⚠  {out_of_range} answers outside [0, 999]")
        except Exception:
            pass
        return ok
