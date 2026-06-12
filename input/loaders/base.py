"""
src/data/base.py
────────────────────────────────────────────────────────────────
Abstract base class for all dataset loaders.

Contract every subclass must honour
────────────────────────────────────
  _download() → raw pd.DataFrame   (HuggingFace / disk)
  _process(raw) → pd.DataFrame     (canonical columns)

Canonical columns (required in every processed DataFrame)
──────────────────────────────────────────────────────────
  id       str   unique row identifier
  query    str   the question / problem text
  answer   str   expected answer

Optional but standardised
  split    str   "train" | "validation" | "test"  (when multi-split)

BaseLoader provides
───────────────────
  load(force)    cache-aware entry point → DataFrame
  verify()       sanity checks + pretty summary
  sample(n)      random rows for quick inspection
  _cap(df)       apply max_samples from config

Cache behaviour (per-split folders)
───────────────────────────────────
  datasets/
    raw/{name}/{split}/raw.parquet
    processed/{name}/{split}/data.parquet

  load()             → concat of all on-disk splits
  load(split="test") → single split only
  load(force=True)   → re-download from HF + reprocess all splits
  load(reprocess=True) → re-run _process() on existing raw splits
"""

from __future__ import annotations

import abc
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ["id", "query", "answer"]
PROCESSED_FILE = "data.parquet"
RAW_FILE = "raw.parquet"


class BaseLoader(abc.ABC):
    NAME: str = ""  # must override in every subclass

    def __init__(self, cfg: dict, data_root: str | Path = "datasets") -> None:
        self.cfg = cfg
        self.root = Path(data_root)
        self.raw_dir = self.root / "raw" / self.NAME
        self.processed_dataset_dir = self.root / "processed" / self.NAME
        self.cache_dir = self.root / "cache"
        for d in (self.raw_dir, self.processed_dataset_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self._df: Optional[pd.DataFrame] = None

    # ── Paths ─────────────────────────────────────────────────────────────

    def raw_split_dir(self, split: str) -> Path:
        return self.raw_dir / split

    def raw_split_path(self, split: str) -> Path:
        return self.raw_split_dir(split) / RAW_FILE

    def processed_split_dir(self, split: str) -> Path:
        return self.processed_dataset_dir / split

    def processed_split_path(self, split: str) -> Path:
        return self.processed_split_dir(split) / PROCESSED_FILE

    @property
    def legacy_processed_path(self) -> Path:
        """Old single-file layout (migrated automatically on load)."""
        return self.root / "processed" / f"{self.NAME}.parquet"

    @property
    def legacy_raw_path(self) -> Path:
        return self.raw_dir / "raw.parquet"

    # backwards compatibility
    @property
    def processed_path(self) -> Path:
        return self.processed_dataset_dir

    @property
    def parquet_path(self) -> Path:
        return self.processed_dataset_dir

    @property
    def raw_path(self) -> Path:
        return self.legacy_raw_path

    # ── Public API ────────────────────────────────────────────────────────

    def available_splits(self, *, processed: bool = True) -> list[str]:
        """Split folder names that already exist on disk."""
        root = self.processed_dataset_dir if processed else self.raw_dir
        if not root.exists():
            return []
        out: list[str] = []
        for p in sorted(root.iterdir()):
            if not p.is_dir():
                continue
            marker = (
                self.processed_split_path(p.name)
                if processed
                else self.raw_split_path(p.name)
            )
            if marker.exists():
                out.append(p.name)
        return out

    def load(
        self,
        force: bool = False,
        reprocess: bool = False,
        split: str | None = None,
    ) -> pd.DataFrame:
        """
        Per-split cache-aware load.

        split=None      Load all splits, return concatenated DataFrame.
        split="test"    Load only that split folder.
        force=True      Re-download from HuggingFace, reprocess, save all splits.
        reprocess=True  Re-run _process() on existing raw split parquets.
        """
        if split is not None:
            return self._load_one_split(split, force=force, reprocess=reprocess)

        if not force and not reprocess:
            on_disk = self.available_splits(processed=True)
            if on_disk:
                logger.info(
                    f"[{self.NAME}] processed cache hit → "
                    f"{self.processed_dataset_dir}/{{{','.join(on_disk)}}}"
                )
                self._df = self._concat_splits(
                    [pd.read_parquet(self.processed_split_path(s)) for s in on_disk]
                )
                return self._df
            if self._migrate_legacy_processed():
                return self.load(force=False, reprocess=False)

        self._df = self._run_pipeline(force=force, reprocess=reprocess)
        return self._df

    def verify(self) -> bool:
        """Print a summary and return True only if all checks pass."""
        if self._df is None:
            logger.warning(f"[{self.NAME}] call load() first")
            return False

        df, ok = self._df, True
        sep = "─" * 52
        print(f"\n{sep}")
        print(f"  {self.NAME.upper()}  ({len(df):,} rows × {df.shape[1]} cols)")
        print(sep)
        print(f"  storage    : {self.processed_dataset_dir}/<split>/{PROCESSED_FILE}")
        print(f"  on disk    : {self.available_splits(processed=True)}")
        print(f"  columns    : {list(df.columns)}")
        print(f"  unique ids : {df['id'].nunique():,}")
        print(f"  avg query  : {df['query'].str.len().mean():.0f} chars")

        for col in REQUIRED_COLUMNS:
            n = int(df[col].isna().sum())
            if n:
                print(f"  ⚠  '{col}' has {n:,} nulls")
                ok = False

        if "split" in df.columns:
            counts = df["split"].value_counts().sort_index().to_dict()
            print(f"  splits     : {counts}")

        ok = self._verify(df) and ok
        print(f"  status     : {'✓ PASS' if ok else '✗ FAIL'}")
        print(sep + "\n")
        return ok

    def sample(self, n: int = 3) -> pd.DataFrame:
        if self._df is None:
            self.load()
        return self._df.sample(n=min(n, len(self._df)), random_state=42)

    # ── Abstract methods ──────────────────────────────────────────────────

    @abc.abstractmethod
    def _download(self) -> pd.DataFrame:
        """Download raw data from HuggingFace and return as-is."""

    @abc.abstractmethod
    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Clean raw data → canonical columns (id, query, answer, ...)."""

    def _verify(self, df: pd.DataFrame) -> bool:
        """Optional dataset-specific checks. Override to add assertions."""
        return True

    # ── Pipeline ──────────────────────────────────────────────────────────

    def _load_one_split(
        self, split: str, *, force: bool, reprocess: bool
    ) -> pd.DataFrame:
        path = self.processed_split_path(split)
        if path.exists() and not force and not reprocess:
            logger.info(f"[{self.NAME}] processed cache hit → {path}")
            return pd.read_parquet(path)

        self.load(force=force, reprocess=reprocess)
        if not path.exists():
            have = self.available_splits(processed=True)
            raise ValueError(
                f"[{self.NAME}] split '{split}' not found after load. "
                f"Available: {have}"
            )
        return pd.read_parquet(path)

    def _run_pipeline(self, *, force: bool, reprocess: bool) -> pd.DataFrame:
        raw = self._load_raw_for_pipeline(force=force, reprocess=reprocess)
        self._persist_raw_splits(raw)

        frames: list[pd.DataFrame] = []
        for sp in self._splits_in_frame(raw):
            part = raw[raw["split"] == sp].copy() if "split" in raw.columns else raw
            logger.info(
                f"[{self.NAME}] processing split '{sp}' ({len(part):,} rows) …"
            )
            proc = self._process(part)
            self._validate(proc)
            self._save_processed_split(sp, proc)
            frames.append(proc)

        combined = self._concat_splits(frames)
        logger.info(
            f"[{self.NAME}] saved {len(frames)} split(s) under "
            f"{self.processed_dataset_dir}"
        )
        return combined

    def _load_raw_for_pipeline(self, *, force: bool, reprocess: bool) -> pd.DataFrame:
        if reprocess and not force:
            on_disk = self.available_splits(processed=False)
            if on_disk:
                logger.info(f"[{self.NAME}] raw cache hit → {self.raw_dir}/<split>")
                return self._concat_splits(
                    [pd.read_parquet(self.raw_split_path(s)) for s in on_disk]
                )
            if self._migrate_legacy_raw():
                return self._load_raw_for_pipeline(force=False, reprocess=True)

        logger.info(f"[{self.NAME}] downloading from HuggingFace …")
        return self._download()

    def _persist_raw_splits(self, raw: pd.DataFrame) -> None:
        for sp in self._splits_in_frame(raw):
            part = raw[raw["split"] == sp].copy() if "split" in raw.columns else raw
            d = self.raw_split_dir(sp)
            d.mkdir(parents=True, exist_ok=True)
            path = self.raw_split_path(sp)
            part.to_parquet(path, index=False)
            logger.info(f"[{self.NAME}] raw saved  → {path}  ({len(part):,} rows)")

    def _save_processed_split(self, split: str, df: pd.DataFrame) -> None:
        d = self.processed_split_dir(split)
        d.mkdir(parents=True, exist_ok=True)
        path = self.processed_split_path(split)
        df.to_parquet(path, index=False)
        logger.info(f"[{self.NAME}] processed → {path}  ({len(df):,} rows)")

    def _splits_in_frame(self, df: pd.DataFrame) -> list[str]:
        if "split" in df.columns:
            return sorted(df["split"].dropna().astype(str).unique().tolist())
        return [str(self.cfg.get("split", "all"))]

    @staticmethod
    def _concat_splits(frames: list[pd.DataFrame]) -> pd.DataFrame:
        if not frames:
            raise RuntimeError("no splits to concatenate")
        if len(frames) == 1:
            return frames[0].reset_index(drop=True)
        return pd.concat(frames, ignore_index=True)

    def _migrate_legacy_processed(self) -> bool:
        path = self.legacy_processed_path
        if not path.exists():
            return False
        logger.info(f"[{self.NAME}] migrating legacy processed file → split folders")
        df = pd.read_parquet(path)
        if "split" not in df.columns:
            sp = str(self.cfg.get("split", "all"))
            self._save_processed_split(sp, df)
        else:
            for sp in self._splits_in_frame(df):
                self._save_processed_split(sp, df[df["split"] == sp].copy())
        return True

    def _migrate_legacy_raw(self) -> bool:
        path = self.legacy_raw_path
        if not path.exists():
            return False
        logger.info(f"[{self.NAME}] migrating legacy raw file → split folders")
        raw = pd.read_parquet(path)
        self._persist_raw_splits(raw)
        return True

    # ── Helpers ───────────────────────────────────────────────────────────

    def _validate(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(
                f"[{self.NAME}] _process() must produce {REQUIRED_COLUMNS}. "
                f"Missing: {missing}"
            )

    def _cap(self, df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
        """Subsample to max_samples if configured."""
        n = self.cfg.get("max_samples")
        if n and len(df) > n:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
            logger.info(f"[{self.NAME}] capped to {n:,} rows")
        return df
