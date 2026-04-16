# """
# src/loaders/gaia_loader.py
# ────────────────────────────────────────────────────────────────
# Loader for GAIA benchmark (gaia-benchmark/GAIA).

# GAIA has 3 annotated difficulty levels:
#   Level 1 → simple, mostly single-step
#   Level 2 → multi-step, tool use required
#   Level 3 → very hard, multi-hop, complex tool use

# This makes GAIA uniquely valuable: Level is a ground-truth
# complexity label we can use to VALIDATE our estimators.

# Canonical columns produced:
#   id       → task_id
#   query    → Question
#   answer   → Final answer

# Extra columns kept:
#   level (1-3), annotator_steps, tools_required, file_name
# """

# from __future__ import annotations

# import logging

# import pandas as pd
# from datasets import load_dataset

# from .base import BaseLoader

# logger = logging.getLogger(__name__)


# class GAIALoader(BaseLoader):

#     DATASET_NAME = "gaia"

#     def _download(self) -> pd.DataFrame:
#         hf_repo = self.config.get("hf_repo", "gaia-benchmark/GAIA")
#         split = self.config.get("split", "validation")
#         hf_config = self.config.get("hf_config", "2023_all")

#         logger.info(f"  HF repo: {hf_repo}  config: {hf_config}  split: {split}")
#         ds = load_dataset(hf_repo, hf_config, split=split)

#         return ds.to_pandas()

#     def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
#         df = raw_df.copy()

#         # ── Rename to canonical columns ───────────────────────────
#         # GAIA column names vary slightly across versions
#         col_map = {}
#         for src, dst in [
#             ("task_id", "id"),
#             ("Question", "query"),
#             ("Final answer", "answer"),
#             ("Level", "level"),
#         ]:
#             if src in df.columns:
#                 col_map[src] = dst
#         df = df.rename(columns=col_map)

#         # ── Parse annotator metadata ──────────────────────────────
#         if "Annotator Metadata" in df.columns:
#             # Metadata is a dict with keys like "Steps", "Tools"
#             def safe_get(meta, key, default=None):
#                 try:
#                     if isinstance(meta, dict):
#                         return meta.get(key, default)
#                     return default
#                 except Exception:
#                     return default

#             df["annotator_steps"] = df["Annotator Metadata"].apply(
#                 lambda m: safe_get(m, "Steps")
#             )
#             df["annotator_tools"] = df["Annotator Metadata"].apply(
#                 lambda m: safe_get(m, "Tools")
#             )
#             df["steps_num"] = df["Annotator Metadata"].apply(
#                 lambda m: safe_get(m, "Number of steps")
#             )   
#             df["tool_num"] = df["Annotator Metadata"].apply(
#                 lambda m: safe_get(m, "Number of tools")
#             )
#             df["time_taken"] = df["Annotator Metadata"].apply(
#                 lambda m: safe_get(m, "How long did this take?")
#             )
    
#             df = df.drop(columns=["Annotator Metadata"])

#         # ── Normalize level to complexity_gt (0.0 – 1.0) ─────────
#         # Level 1 → 0.2, Level 2 → 0.5, Level 3 → 0.9
#         # level_map = {1: 0.2, 2: 0.5, 3: 0.9}
#         # if "level" in df.columns:
#         #     df["complexity_gt"] = df["level"].map(level_map)
#         #     logger.info("  Added 'complexity_gt' from GAIA levels (ground truth)")

#         # ── Keep only needed columns ──────────────────────────────
#         keep = [
#             "id",
#             "query",
#             "answer",
#             "level",
#             # "complexity_gt",
#             "annotator_steps",
#             "annotator_tools",
#             "file_name",
#             "steps_num",
#             "tool_num",
#             "time_taken"

#         ]
#         keep = [c for c in keep if c in df.columns]
#         df = df[keep].copy()

#         # ── Drop unanswerable rows ────────────────────────────────
#         before = len(df)
#         df = df[df["query"].notna() & (df["query"].str.strip() != "")]
#         logger.info(f"  Dropped {before - len(df)} rows with empty query")

#         df = self._apply_max_samples(df)
#         df = df.reset_index(drop=True)
#         return df

#     def _extra_verify(self, df: pd.DataFrame) -> bool:
#         ok = True
#         if "level" in df.columns:
#             dist = df["level"].value_counts().sort_index().to_dict()
#             print(f"  Levels   : {dist}  (1=easy → 3=hard)")
#         if "complexity_gt" in df.columns:
#             print(
#                 f"  GT range : {df['complexity_gt'].min():.2f} – "
#                 f"{df['complexity_gt'].max():.2f}"
#             )
#         return ok



"""
src/loaders/gaia_loader.py
────────────────────────────────────────────────────────────────
Loader for GAIA benchmark (gaia-benchmark/GAIA).

GAIA has 3 annotated difficulty levels:
  Level 1 → simple, mostly single-step
  Level 2 → multi-step, tool use required
  Level 3 → very hard, multi-hop, complex tool use

This makes GAIA uniquely valuable: Level is a ground-truth
complexity label we can use to VALIDATE our estimators.

Canonical columns produced:
  id       → task_id
  query    → Question
  answer   → Final answer

Extra columns kept:
  level (1-3), annotator_steps, tools_required, file_name

Output structure:
  data/
    raw/     → raw_{split}.parquet   (untouched HF dump)
    golden/  → golden_{split}.parquet (processed, validated)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

from .base import BaseLoader

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

METADATA_KEYS: dict[str, str] = {
    "annotator_steps": "Steps",
    "annotator_tools": "Tools",
    "steps_num":       "Number of steps",
    "tool_num":        "Number of tools",
    "time_taken":      "How long did this take?",
}

LEVEL_COMPLEXITY_MAP: dict[int, float] = {1: 0.2, 2: 0.5, 3: 0.9}

KEEP_COLUMNS: list[str] = [
    "id",
    "query",
    "answer",
    "level",
    # "complexity_gt",
    "annotator_steps",
    "annotator_tools",
    "file_name",
    "steps_num",
    "tool_num",
    "time_taken",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_time_to_minutes(val) -> int | pd.NA:
    """Convert strings like '5 minutes', '10 mins', '2 hours' → int (minutes)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return pd.NA
    match = re.match(r"(\d+)\s*(hours|hour|hr|min)", str(val).strip().lower())
    if match:
        n, unit = int(match.group(1)), match.group(2)
        return n * 60 if unit.startswith("h") else n
    fallback = re.match(r"(\d+)", str(val))
    return int(fallback.group(1)) if fallback else pd.NA


def _expand_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract all annotator-metadata fields in a single .apply() pass,
    then cast numeric columns to nullable Int64.
    """
    df[list(METADATA_KEYS.keys())] = df["Annotator Metadata"].apply(
        lambda m: pd.Series({
            col: (m.get(key) if isinstance(m, dict) else None)
            for col, key in METADATA_KEYS.items()
        })
    )

    df[["steps_num", "tool_num"]] = (
        df[["steps_num", "tool_num"]]
        .apply(pd.to_numeric, errors="coerce")
        .astype("Int64")
    )

    df["time_taken"] = df["time_taken"].apply(_parse_time_to_minutes).astype("Int64")

    return df.drop(columns=["Annotator Metadata"])


# ── Loader ────────────────────────────────────────────────────────────────────

class GAIALoader(BaseLoader):

    DATASET_NAME = "gaia"

    # ── Paths ──────────────────────────────────────────────────────────────
    @property
    def _split(self) -> str:
        return self.config.get("split", "validation")

    @property
    def _raw_path(self) -> Path:
        root = Path(self.config.get("data_dir", "data"))
        path = root / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"raw_{self._split}.parquet"

    @property
    def _golden_path(self) -> Path:
        root = Path(self.config.get("data_dir", "data"))
        path = root / "golden"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"golden_{self._split}.parquet"

    # ── Download ───────────────────────────────────────────────────────────
    def _download(self) -> pd.DataFrame:
        hf_repo   = self.config.get("hf_repo",   "gaia-benchmark/GAIA")
        hf_config = self.config.get("hf_config", "2023_all")

        logger.info(f"  HF repo: {hf_repo}  config: {hf_config}  split: {self._split}")
        raw_df = load_dataset(hf_repo, hf_config, split=self._split).to_pandas()

        # ── Persist raw snapshot ──────────────────────────────────────────
        raw_df.to_parquet(self._raw_path, index=False)
        logger.info(f"  Raw data saved → {self._raw_path}  ({len(raw_df):,} rows)")

        return raw_df

    # ── Process ────────────────────────────────────────────────────────────
    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # ── Rename to canonical columns ───────────────────────────────────
        col_map = {
            src: dst
            for src, dst in [
                ("task_id",      "id"),
                ("Question",     "query"),
                ("Final answer", "answer"),
                ("Level",        "level"),
            ]
            if src in df.columns
        }
        df = df.rename(columns=col_map)

        # ── Parse annotator metadata ──────────────────────────────────────
        if "Annotator Metadata" in df.columns:
            df = _expand_metadata(df)

        # ── Normalize level to complexity_gt (0.0 – 1.0) ─────────────────
        # if "level" in df.columns:
        #     df["complexity_gt"] = df["level"].map(LEVEL_COMPLEXITY_MAP)
        #     logger.info("  Added 'complexity_gt' from GAIA levels (ground truth)")

        # ── Keep only needed columns ──────────────────────────────────────
        keep = [c for c in KEEP_COLUMNS if c in df.columns]
        df = df[keep].copy()

        # ── Drop unanswerable rows ────────────────────────────────────────
        before = len(df)
        df = df[df["query"].notna() & (df["query"].str.strip() != "")]
        logger.info(f"  Dropped {before - len(df)} rows with empty query")

        df = self._apply_max_samples(df)
        df = df.reset_index(drop=True)

        # ── Persist golden snapshot ───────────────────────────────────────
        df.to_parquet(self._golden_path, index=False)
        logger.info(f"  Golden data saved → {self._golden_path}  ({len(df):,} rows)")

        return df

    # ── Verify ─────────────────────────────────────────────────────────────
    def _extra_verify(self, df: pd.DataFrame) -> bool:
        if "level" in df.columns:
            dist = df["level"].value_counts().sort_index().to_dict()
            logger.info(f"  Levels   : {dist}  (1=easy → 3=hard)")

        if "complexity_gt" in df.columns:
            logger.info(
                f"  GT range : {df['complexity_gt'].min():.2f} – "
                f"{df['complexity_gt'].max():.2f}"
            )

        if "time_taken" in df.columns:
            valid = df["time_taken"].dropna()
            if len(valid):
                logger.info(
                    f"  Time     : min={valid.min()} min  "
                    f"max={valid.max()} min  "
                    f"median={valid.median():.0f} min"
                )
        return True