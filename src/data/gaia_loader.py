"""
src/data/gaia_loader.py
────────────────────────────────────────────────────────────────
Loader for the GAIA benchmark (gaia-benchmark/GAIA).

Only the validation split has gold answers and is publicly
accessible (test answers are withheld by the benchmark authors).

Requires HuggingFace authentication:
    huggingface-cli login
    # or set HF_TOKEN in your .env file

Canonical columns
─────────────────
  id            str    task_id (UUID)
  query         str    the question
  answer        str    gold final answer
  level         int    1 (easy) → 3 (hard)  ← ground-truth difficulty label
  split         str    "validation"
  file_name     str    attachment filename (empty string if none)

Annotator metadata (expanded from nested dict)
  steps         str    annotator step description
  tools         str    tools used
  steps_num     Int64  number of steps
  tool_num      Int64  number of tools
  time_min      Int64  time taken (minutes)

Config keys (all optional)
──────────────────────────
  hf_repo      str          default: "gaia-benchmark/GAIA"
  hf_config    str          default: "2023_all"
  split        str          default: "validation"
  max_samples  int | null
"""

from __future__ import annotations

import logging
import re

import pandas as pd

from .base import BaseLoader

logger = logging.getLogger(__name__)


def _to_minutes(val) -> "int | pd.NA":
    """'5 minutes' / '2 hours' / '10' → int minutes."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return pd.NA
    s = str(val).strip().lower()
    m = re.match(r"(\d+)\s*(h|min)", s)
    if m:
        n, u = int(m.group(1)), m.group(2)
        return n * 60 if u.startswith("h") else n
    m2 = re.match(r"(\d+)", s)
    return int(m2.group(1)) if m2 else pd.NA


def _expand_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """Unpack 'Annotator Metadata' dict column into flat columns."""
    key_map = {
        "steps":     "Steps",
        "tools":     "Tools",
        "steps_num": "Number of steps",
        "tool_num":  "Number of tools",
        "time_min":  "How long did this take?",
    }
    meta = df["Annotator Metadata"].apply(
        lambda m: pd.Series({
            col: (m.get(hf_key) if isinstance(m, dict) else None)
            for col, hf_key in key_map.items()
        })
    )
    df = pd.concat([df.drop(columns=["Annotator Metadata"]), meta], axis=1)

    for col in ("steps_num", "tool_num"):
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    df["time_min"] = df["time_min"].apply(_to_minutes).astype("Int64")
    return df


def append_attachment_to_query(query: str, file_name: str | None) -> str:
    """Append attachment basename to the question for tool-using agents (read_file)."""
    q = (query or "").strip()
    fn = "" if file_name is None or (isinstance(file_name, float) and pd.isna(file_name)) else str(file_name).strip()
    if not fn:
        return q
    marker = f"Attached file: {fn}"
    if marker.lower() in q.lower():
        return q
    return f"{q}\n\n{marker}"


class GAIALoader(BaseLoader):
    NAME = "gaia"

    # ── Download ──────────────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset  # type: ignore

        hf_repo   = self.cfg.get("hf_repo",   "gaia-benchmark/GAIA")
        hf_config = self.cfg.get("hf_config", "2023_all")
        split     = self.cfg.get("split",     "validation")

        logger.info(f"[gaia] {hf_repo} / {hf_config} / {split}")
        df = load_dataset(
            hf_repo, hf_config, split=split,
            cache_dir=str(self.cache_dir),
        ).to_pandas()
        df["split"] = split
        return df

    # ── Process ───────────────────────────────────────────────────────────

    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={
            "task_id":      "id",
            "Question":     "query",
            "Final answer": "answer",
            "Level":        "level",
        }).copy()

        if "Annotator Metadata" in df.columns:
            df = _expand_metadata(df)

        # Drop rows with no question
        before = len(df)
        df = df[df["query"].notna() & (df["query"].str.strip() != "")].reset_index(drop=True)
        if before > len(df):
            logger.info(f"[gaia] dropped {before - len(df)} empty-query rows")

        if "file_name" in df.columns:
            df["query"] = [
                append_attachment_to_query(q, fn)
                for q, fn in zip(df["query"], df["file_name"], strict=True)
            ]
            n = int((df["file_name"].fillna("").astype(str).str.strip() != "").sum())
            if n:
                logger.info(f"[gaia] appended 'Attached file: <name>' to query for {n:,} rows")

        df = self._cap(df)

        cols = ["id", "query", "answer", "level", "split", "file_name",
                "steps", "tools", "steps_num", "tool_num", "time_min"]
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    # ── Verify ────────────────────────────────────────────────────────────

    def _verify(self, df: pd.DataFrame) -> bool:
        if "level" in df.columns:
            dist = df["level"].value_counts().sort_index().to_dict()
            print(f"  levels   : {dist}  (1=easy, 3=hard)")
        if "file_name" in df.columns:
            n_attach = (df["file_name"].fillna("") != "").sum()
            print(f"  w/ files : {n_attach:,} rows have attachments")
        if "time_min" in df.columns:
            valid = df["time_min"].dropna()
            if len(valid):
                print(f"  time     : median={valid.median():.0f} min  max={valid.max()} min")
        return True