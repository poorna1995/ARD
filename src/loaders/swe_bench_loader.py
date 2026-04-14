"""
src/loaders/swe_bench_loader.py
────────────────────────────────────────────────────────────────
Loader for SWE-Bench Verified (princeton-nlp/SWE-bench_Verified).

Each row = one GitHub issue the agent must fix.
Canonical columns produced:
  id       → instance_id (e.g. "astropy__astropy-12907")
  query    → problem_statement (the issue text)
  answer   → patch (gold diff, for reference only)

Extra columns kept:
  repo, base_commit, version, hints_text
"""

from __future__ import annotations

import logging

import pandas as pd
from datasets import load_dataset

from .base import BaseLoader

logger = logging.getLogger(__name__)


class SWEBenchLoader(BaseLoader):

    DATASET_NAME = "swe_bench"

    def _download(self) -> pd.DataFrame:
        hf_repo = self.config.get("hf_repo", "princeton-nlp/SWE-bench_Verified")
        split = self.config.get("split", "test")

        logger.info(f"  HF repo: {hf_repo}  split: {split}")
        ds = load_dataset(hf_repo, split=split)
        return ds.to_pandas()

    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # ── Rename to canonical columns ───────────────────────────
        df = df.rename(
            columns={
                "instance_id": "id",
                "problem_statement": "query",
                "patch": "answer",
            }
        )

        # ── Keep useful metadata columns ──────────────────────────
        keep = [
            "id",
            "query",
            "answer",
            "repo",
            "base_commit",
            "version",
            "hints_text",
            "difficulty",
            "created_at",
            "FAIL_TO_PASS",
            "PASS_TO_PASS",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        # ── Derived complexity proxy: number of test cases that must flip ──
        # FAIL_TO_PASS is a JSON list of test names that must go from fail→pass
        # More tests = more involved fix = proxy for complexity
        if "FAIL_TO_PASS" in df.columns:
            import ast

            def count_tests(v):
                try:
                    lst = ast.literal_eval(v) if isinstance(v, str) else v
                    return len(lst) if isinstance(lst, list) else 0
                except Exception:
                    return 0

            df["num_fail_to_pass"] = df["FAIL_TO_PASS"].apply(count_tests)
            logger.info(f"  Added 'num_fail_to_pass' (complexity proxy)")

        # ── Drop rows with empty problem statements ────────────────
        before = len(df)
        df = df[df["query"].notna() & (df["query"].str.strip() != "")]
        dropped = before - len(df)
        if dropped:
            logger.warning(f"  Dropped {dropped} rows with empty query")

        df = self._apply_max_samples(df)
        df = df.reset_index(drop=True)
        return df

    def _extra_verify(self, df: pd.DataFrame) -> bool:
        ok = True
        if "repo" in df.columns:
            print(f"  Repos    : {df['repo'].nunique()} unique repos")
            print(f"  Top repos: {df['repo'].value_counts().head(3).to_dict()}")
        if "num_fail_to_pass" in df.columns:
            print(
                f"  Tests/task: mean={df['num_fail_to_pass'].mean():.1f}, "
                f"max={df['num_fail_to_pass'].max()}"
            )
        # Sanity: problem statements should be long (GitHub issues)
        short = (df["query"].str.len() < 50).sum()
        if short > 5:
            print(f"  ⚠  {short} very short problem statements (<50 chars)")
            ok = False
        return ok
