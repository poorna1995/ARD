"""
src/data/musique_loader.py
────────────────────────────────────────────────────────────────
Loader for MuSiQue multi-hop QA dataset.

HuggingFace repo : dgslibisey/MuSiQue  (default)

Both train and validation splits are downloaded and merged.
Unanswerable questions (answerable == False) are dropped by
default since they have no gold answer.

Canonical columns
─────────────────
  id            str   original row id
  query         str   the multi-hop question
  answer        str   gold answer string
  split         str   "train" | "validation"
  n_hops        int   hop count parsed from id (e.g. 3 in "3hop2__...")
  hop_type      int   1=linear, 2=parallel, 3=branching (default 1 if omitted)
  hop_name      str   "linear" | "parallel" | "branching" | "unknown"
  answerable    bool  kept for reference (all True after filtering)
  context       dict  normalized passages (from ``paragraphs``)
  question_decomposition  list[dict]  sub-questions with gold answers (MuSiQue only)

Config keys (all optional)
──────────────────────────
  hf_repo              str          HuggingFace repo id.
  max_samples          int | null   Row cap after all filtering.
  keep_unanswerable    bool         Keep unanswerable rows (default: false).
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from .base import BaseLoader

logger = logging.getLogger(__name__)

_HOP_NAMES = {1: "linear", 2: "parallel", 3: "branching"}
_HOP_ID_RE = re.compile(r"(\d+)hop(\d+)?")


def _normalize_question_decomposition(value: Any) -> list[dict[str, Any]] | None:
    """Coerce HF / parquet ndarray rows to a plain list of dicts."""
    if value is None:
        return None
    from agent.episode_context import _as_list

    items = _as_list(value)
    if not items:
        return None
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(dict(item))
    return out or None


def _extract_hop_info(qid: str) -> dict[str, int | str | None]:
    m = _HOP_ID_RE.match(qid)
    if not m:
        return {"n_hops": None, "hop_type": None, "hop_name": None}
    n_hops = int(m.group(1))
    hop_type = int(m.group(2)) if m.group(2) else 1
    hop_name = _HOP_NAMES.get(hop_type, "unknown")
    return {"n_hops": n_hops, "hop_type": hop_type, "hop_name": hop_name}


class MuSiQueLoader(BaseLoader):
    NAME = "musique"

    # ── Download ──────────────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset  # type: ignore

        hf_repo = self.cfg.get("hf_repo", "dgslibisey/MuSiQue")
        frames: list[pd.DataFrame] = []

        try:
            ds = load_dataset(hf_repo, cache_dir=str(self.cache_dir))
        except Exception as exc:
            raise RuntimeError(f"[musique] download failed: {exc}") from exc

        for split in ("train", "validation"):
            if split not in ds:
                logger.warning(f"[musique] split '{split}' not found, skipping")
                continue
            df = ds[split].to_pandas()
            df["split"] = split
            frames.append(df)
            logger.info(f"[musique] {split}: {len(df):,} rows")

        if not frames:
            raise RuntimeError("[musique] no usable splits found")

        raw = pd.concat(frames, ignore_index=True)
        logger.info(f"[musique] total raw rows: {len(raw):,}")
        return raw

    # ── Process ───────────────────────────────────────────────────────────

    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()

        # Normalise question column name (varies by HF version)
        question_col = next(
            (c for c in ("question", "input", "query") if c in df.columns), None
        )
        if question_col is None:
            raise ValueError(f"[musique] cannot find question column. Got: {list(df.columns)}")
        df = df.rename(columns={question_col: "query"})

        # Ensure id column
        if "id" not in df.columns:
            df["id"] = [f"musique_{i:05d}" for i in range(len(df))]
        df["id"] = df["id"].astype(str)

        # Drop unanswerable rows (no gold answer)
        if "answerable" in df.columns and not self.cfg.get("keep_unanswerable", False):
            before = len(df)
            df = df[df["answerable"].astype(bool)].reset_index(drop=True)
            logger.info(f"[musique] dropped {before - len(df):,} unanswerable rows")

        hop_cols = df["id"].apply(_extract_hop_info).apply(pd.Series)
        df = pd.concat([df, hop_cols], axis=1)

        from agent.episode_context import normalize_episode_context

        if "paragraphs" in df.columns:
            df["context"] = df["paragraphs"].apply(
                lambda p: normalize_episode_context(p, dataset="musique")
            )
        elif "context" in df.columns:
            df["context"] = df["context"].apply(
                lambda c: normalize_episode_context(c, dataset="musique")
            )

        if "question_decomposition" in df.columns:
            df["question_decomposition"] = df["question_decomposition"].apply(
                _normalize_question_decomposition
            )

        df = self._cap(df)

        cols = [
            "id", "query", "answer", "split", "n_hops", "hop_type",
            "hop_name", "answerable", "context", "question_decomposition",
        ]
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    # ── Verify ────────────────────────────────────────────────────────────

    def _verify(self, df: pd.DataFrame) -> bool:
        if "n_hops" in df.columns:
            print(f"  n_hops   : {df['n_hops'].value_counts().sort_index().to_dict()}")
        if "hop_name" in df.columns:
            print(f"  hop_name : {df['hop_name'].value_counts().to_dict()}")
        return True