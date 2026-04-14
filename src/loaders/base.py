"""
src/loaders/base.py
────────────────────────────────────────────────────────────────
Abstract base class that every dataset loader must implement.

Rules:
- load()  → returns a pandas DataFrame with canonical columns
- verify() → returns True/False + prints a summary
- All loaders save a processed parquet to datasets/processed/
  so downstream code never calls HuggingFace again.
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# ── Canonical column names every loader must produce ────────────
REQUIRED_COLUMNS = ["id", "query", "answer"]


class BaseLoader(abc.ABC):
    """
    Abstract base for all dataset loaders.

    Subclasses must implement:
        _download()  → raw data as pd.DataFrame
        _process()   → clean, rename to canonical columns
    """

    DATASET_NAME: str = "base"  # override in subclass

    def __init__(self, config: dict, data_root: str = "datasets"):
        self.config = config
        self.data_root = Path(data_root)
        self.raw_dir = self.data_root / "raw" / self.DATASET_NAME
        self.processed_dir = self.data_root / "processed"
        self.cache_dir = self.data_root / "cache"

        # Create dirs if missing
        for d in [self.raw_dir, self.processed_dir, self.cache_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._df: Optional[pd.DataFrame] = None

    # ── Public API ────────────────────────────────────────────────

    def load(self, force_redownload: bool = False) -> pd.DataFrame:
        """
        Main entry point. Returns a DataFrame with canonical columns.
        Uses cached parquet if available (skips HF download).
        """
        parquet_path = self.processed_dir / f"{self.DATASET_NAME}.parquet"

        if parquet_path.exists() and not force_redownload:
            logger.info(f"[{self.DATASET_NAME}] Loading from cache: {parquet_path}")
            self._df = pd.read_parquet(parquet_path)
            return self._df

        logger.info(f"[{self.DATASET_NAME}] Downloading from source...")
        raw_df = self._download()

        logger.info(f"[{self.DATASET_NAME}] Processing {len(raw_df)} raw rows...")
        self._df = self._process(raw_df)

        # Validate
        self._validate_columns()

        # Save processed parquet (canonical format)
        self._df.to_parquet(parquet_path, index=False)
        logger.info(
            f"[{self.DATASET_NAME}] Saved → {parquet_path}  ({len(self._df)} rows)"
        )

        return self._df

    def verify(self) -> bool:
        """
        Quick sanity check: print a summary and return True if OK.
        Call after load() to confirm the dataset is usable.
        """
        if self._df is None:
            logger.warning(f"[{self.DATASET_NAME}] Not loaded yet. Call load() first.")
            return False

        df = self._df
        ok = True

        print(f"\n{'─'*50}")
        print(f"  {self.DATASET_NAME.upper()}  ({len(df)} rows)")
        print(f"{'─'*50}")
        print(f"  Columns : {list(df.columns)}")
        print(f"  ID       : {df['id'].nunique()} unique IDs")
        print(f"  Query    : avg {df['query'].str.len().mean():.0f} chars")

        # Check for nulls in required columns
        for col in REQUIRED_COLUMNS:
            nulls = df[col].isna().sum()
            if nulls > 0:
                print(f"  ⚠  '{col}' has {nulls} nulls")
                ok = False

        # Dataset-specific checks
        extra = self._extra_verify(df)
        ok = ok and extra

        status = "✓ PASS" if ok else "✗ FAIL"
        print(f"  Status   : {status}")
        print(f"{'─'*50}\n")
        return ok

    def get_sample(self, n: int = 3, seed: int = 42) -> pd.DataFrame:
        """Return n random rows for quick inspection."""
        if self._df is None:
            self.load()
        return self._df.sample(n=min(n, len(self._df)), random_state=seed)

    # ── Abstract methods ──────────────────────────────────────────

    @abc.abstractmethod
    def _download(self) -> pd.DataFrame:
        """Download raw data and return as DataFrame."""

    @abc.abstractmethod
    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and normalize raw_df.
        Must return a DataFrame with at minimum: id, query, answer.
        """

    def _extra_verify(self, df: pd.DataFrame) -> bool:
        """
        Optional dataset-specific checks.
        Override in subclasses to add custom assertions.
        Returns True if checks pass.
        """
        return True

    # ── Private helpers ───────────────────────────────────────────

    def _validate_columns(self):
        """Raise if required canonical columns are missing."""
        missing = [c for c in REQUIRED_COLUMNS if c not in self._df.columns]
        if missing:
            raise ValueError(
                f"[{self.DATASET_NAME}] _process() must produce columns: "
                f"{REQUIRED_COLUMNS}. Missing: {missing}"
            )

    def _apply_max_samples(self, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """Subsample if max_samples is set in config."""
        max_n = self.config.get("max_samples")
        if max_n and len(df) > max_n:
            df = df.sample(n=max_n, random_state=seed).reset_index(drop=True)
            logger.info(f"[{self.DATASET_NAME}] Subsampled to {max_n} rows")
        return df
