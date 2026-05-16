"""
src/data/math_loader.py
────────────────────────────────────────────────────────────────
Loader for EleutherAI/hendrycks_math.

Downloads all 7 subjects × 2 splits (train + test), concatenates
them, optionally drops Asymptote figure problems, extracts the
final \\boxed{} answer from each solution, and returns a single
canonical DataFrame.

Canonical columns
─────────────────
  id               str   "math_00000" …
  query            str   Competition math problem (LaTeX)
  answer           str   Last \\boxed{} content, MATH-normalized LaTeX
  level            str   "Level 1" … "Level 5"
  type             str   Subject (Algebra, Geometry, …)
  solution         str   Full step-by-step solution (LaTeX)
  split            str   "train" | "test"
  solution_length  int   len(solution) — within-level complexity proxy

Config keys (all optional)
──────────────────────────
  max_samples      int | null   Cap after all filtering.
  exclude_figures  bool         Drop [asy] blocks (default: true).
  levels           list | null  e.g. ["Level 5"]. null = all.
  subjects         list | null  e.g. ["Algebra"]. null = all.
  normalize_answers bool        Apply MATH strip_string to answer (default: true).
"""

from __future__ import annotations

import logging
import re
from typing import Optional

import pandas as pd

from src.math_latex import extract_last_boxed, normalize_math_answer

from .base import BaseLoader

logger = logging.getLogger(__name__)

_ASY_RE = re.compile(r"\[asy\]", re.IGNORECASE)

_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def _extract_boxed(text: str) -> Optional[str]:
    """Return content of the last \\boxed{...} in text (handles nested braces)."""
    return extract_last_boxed(text)


class MathLoader(BaseLoader):
    NAME = "math"

    # ── Download ──────────────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset  # type: ignore

        hf_repo = self.cfg.get("hf_repo", "EleutherAI/hendrycks_math")
        frames: list[pd.DataFrame] = []

        for subj in _SUBJECTS:
            try:
                ds = load_dataset(
                    hf_repo, subj,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
            except Exception as exc:
                logger.warning(f"[math] skipping subject '{subj}': {exc}")
                continue

            for split in ("train", "test"):
                if split not in ds:
                    continue
                df = ds[split].to_pandas()
                df["split"] = split
                frames.append(df)
                logger.info(f"[math]   {subj}/{split}: {len(df):,} rows")

        if not frames:
            raise RuntimeError("[math] nothing downloaded — check HF access")

        raw = pd.concat(frames, ignore_index=True)
        logger.info(f"[math] total raw rows: {len(raw):,}")
        return raw

    # ── Process ───────────────────────────────────────────────────────────

    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={"problem": "query"}).copy()

        # 1. Drop Asymptote figure problems
        if self.cfg.get("exclude_figures", True):
            before = len(df)
            df = df[~df["query"].str.contains(_ASY_RE, na=False)].reset_index(drop=True)
            logger.info(f"[math] removed {before - len(df):,} [asy] problems")

        # 2. Optional level / subject filters
        if levels := self.cfg.get("levels"):
            df = df[df["level"].isin(levels)].reset_index(drop=True)
            logger.info(f"[math] level filter → {len(df):,} rows")
        if subjects := self.cfg.get("subjects"):
            df = df[df["type"].isin(subjects)].reset_index(drop=True)
            logger.info(f"[math] subject filter → {len(df):,} rows")

        # 3. Extract answer from last \boxed{}
        df["answer"] = df["solution"].apply(_extract_boxed)
        # Drop both NaN (no \boxed found) and empty-string answers
        before = len(df)
        df = df[df["answer"].notna() & (df["answer"].str.strip() != "")].reset_index(drop=True)
        dropped = before - len(df)
        if dropped:
            logger.warning(f"[math] dropped {dropped:,} rows with missing/blank answer")

        if self.cfg.get("normalize_answers", True):
            before_unique = df["answer"].nunique()
            df["answer"] = df["answer"].apply(normalize_math_answer)
            logger.info(
                f"[math] normalized answers (unique {before_unique:,} → {df['answer'].nunique():,})"
            )

        # 3b. Drop malformed level values (e.g. "Level ?" from dirty HF data)
        valid_levels = {"Level 1", "Level 2", "Level 3", "Level 4", "Level 5"}
        if "level" in df.columns:
            before = len(df)
            df = df[df["level"].isin(valid_levels)].reset_index(drop=True)
            if len(df) < before:
                logger.info(f"[math] dropped {before - len(df)} rows with invalid level")

        # 4. Complexity proxy
        df["solution_length"] = df["solution"].str.len().astype(int)

        # 5. Stable IDs
        df["id"] = [f"math_{i:05d}" for i in range(len(df))]

        # 6. Cap
        df = self._cap(df)

        cols = ["id", "query", "answer", "level", "type", "solution",
                "split", "solution_length"]
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    # ── Verify ────────────────────────────────────────────────────────────

    def _verify(self, df: pd.DataFrame) -> bool:
        ok = True
        if "level" in df.columns:
            print(f"  levels   : {df['level'].value_counts().sort_index().to_dict()}")
        if "type" in df.columns:
            print(f"  subjects : {df['type'].value_counts().to_dict()}")
        print(f"  sol_len  : min={df['solution_length'].min()}  "
              f"mean={df['solution_length'].mean():.0f}  "
              f"max={df['solution_length'].max()}")
        return ok