"""
src/data/hotpot_loader.py
────────────────────────────────────────────────────────────────
Loader for HotpotQA (hotpot_qa, distractor config by default).

Both train and validation splits are downloaded and merged.

Canonical columns
─────────────────
  id        str   original row id
  query     str   the multi-hop question
  answer    str   gold answer
  type      str   "bridge" | "comparison"
  level     str   "easy" | "medium" | "hard"
  split     str   "train" | "validation"

Config keys (all optional)
──────────────────────────
  hf_repo      str          default: "hotpot_qa"
  hf_config    str          default: "distractor" (or "fullwiki")
  max_samples  int | null
"""

from __future__ import annotations

import logging
import pandas as pd

from .base import BaseLoader

logger = logging.getLogger(__name__)


class HotpotLoader(BaseLoader):
    NAME = "hotpot"

    # ── Download ──────────────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset  # type: ignore

        hf_repo   = self.cfg.get("hf_repo",   "hotpot_qa")
        hf_config = self.cfg.get("hf_config", "distractor")
        frames: list[pd.DataFrame] = []

        for split in ("train", "validation"):
            try:
                ds = load_dataset(
                    hf_repo, hf_config, split=split,
                    cache_dir=str(self.cache_dir),
                    trust_remote_code=True,
                )
                df = ds.to_pandas()
                df["split"] = split
                frames.append(df)
                logger.info(f"[hotpot] {split}: {len(df):,} rows")
            except Exception as exc:
                logger.warning(f"[hotpot] split '{split}' failed: {exc}")

        if not frames:
            raise RuntimeError("[hotpot] nothing downloaded")

        raw = pd.concat(frames, ignore_index=True)
        logger.info(f"[hotpot] total raw rows: {len(raw):,}")
        return raw

    # ── Process ───────────────────────────────────────────────────────────

    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={"question": "query"}).copy()

        # Ensure id
        if "id" not in df.columns:
            df["id"] = [f"hotpot_{i:05d}" for i in range(len(df))]
        df["id"] = df["id"].astype(str)

        df = self._cap(df)

        cols = ["id", "query", "answer", "type", "level", "split"]
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    # ── Verify ────────────────────────────────────────────────────────────

    def _verify(self, df: pd.DataFrame) -> bool:
        if "type" in df.columns:
            print(f"  types  : {df['type'].value_counts().to_dict()}")
        if "level" in df.columns:
            print(f"  levels : {df['level'].value_counts().to_dict()}")
        return True