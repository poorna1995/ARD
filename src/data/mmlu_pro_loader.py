"""
src/data/mmlu_pro_loader.py
────────────────────────────────────────────────────────────────
Loader for MMLU-Pro (TIGER-Lab/MMLU-Pro).

Both validation and test splits are downloaded and merged.

Known HuggingFace issue: .to_pandas() collapses nested list
columns (like `options`) into empty numpy arrays when the Arrow
schema uses a list type.  Fix: pull the column directly from the
HF Dataset object before converting to pandas.

Canonical columns
─────────────────
  id            str    question_id (or synthetic "mmlu_00000")
  query         str    the question text
  answer        str    letter answer (A–J)
  answer_index  int    0-based index into options
  options       list   10-choice list
  category      str    subject category (14 domains)
  cot_length    int    len(cot_content) — 0 if blank in this HF version
  split         str    "validation" | "test"

Config keys (all optional)
──────────────────────────
  hf_repo       str          default: "TIGER-Lab/MMLU-Pro"
  max_samples   int | null   If set, stratified sample across categories.
"""

from __future__ import annotations

import ast
import logging

import numpy as np
import pandas as pd

from .base import BaseLoader

logger = logging.getLogger(__name__)


def _coerce_list(val) -> list | None:
    """Normalise options values regardless of how HF serialised them."""
    if isinstance(val, list):
        return val or None
    if isinstance(val, np.ndarray):
        lst = val.tolist()
        return lst or None
    if isinstance(val, str):
        try:
            parsed = ast.literal_eval(val.strip())
            if isinstance(parsed, list) and parsed:
                return parsed
        except Exception:
            pass
    return None


def normalize_options(options: object) -> list[str]:
    """Return option texts as a flat list of non-empty strings."""
    raw = _coerce_list(options)
    if raw is None and isinstance(options, np.ndarray):
        raw = options.tolist()
    if not raw:
        return []
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    out: list[str] = []
    for item in raw:
        if item is None or (isinstance(item, float) and pd.isna(item)):
            continue
        text = str(item).strip()
        if text:
            out.append(text)
    return out


def format_options_block(options: object) -> str:
    """Format choices as 'a. opt1, b. opt2, ...' for agent prompts."""
    opts = normalize_options(options)
    if not opts:
        return ""
    parts = [f"{chr(ord('a') + i)}. {text}" for i, text in enumerate(opts)]
    return ", ".join(parts)


def append_options_to_query(query: str, options: object) -> str:
    """Append multiple-choice options to the question text."""
    q = (query or "").strip()
    block = format_options_block(options)
    if not block:
        return q
    if block in q:
        return q
    return f"{q}\n\n{block}"


class MMLUProLoader(BaseLoader):
    NAME = "mmlu_pro"

    # ── Download ──────────────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        from datasets import load_dataset  # type: ignore

        hf_repo = self.cfg.get("hf_repo", "TIGER-Lab/MMLU-Pro")
        frames: list[pd.DataFrame] = []

        for split in ("validation", "test"):
            try:
                ds = load_dataset(hf_repo, split=split, cache_dir=str(self.cache_dir))
                df = ds.to_pandas()
                # Fix collapsed options column — pull directly from Arrow
                try:
                    df["options"] = [list(r) for r in ds["options"]]
                    logger.info(f"[mmlu_pro] re-extracted options for {split}")
                except Exception as exc:
                    logger.warning(f"[mmlu_pro] could not fix options for {split}: {exc}")
                df["split"] = split
                frames.append(df)
                logger.info(f"[mmlu_pro] {split}: {len(df):,} rows")
            except Exception as exc:
                logger.warning(f"[mmlu_pro] split '{split}' failed: {exc}")

        if not frames:
            raise RuntimeError("[mmlu_pro] nothing downloaded")

        raw = pd.concat(frames, ignore_index=True)
        logger.info(f"[mmlu_pro] total raw rows: {len(raw):,}")
        return raw

    # ── Process ───────────────────────────────────────────────────────────

    def _process(self, raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.rename(columns={"question": "query", "question_id": "id"}).copy()

        if "id" not in df.columns:
            df["id"] = [f"mmlu_{i:05d}" for i in range(len(df))]
        df["id"] = df["id"].astype(str)

        # Normalise options
        if "options" in df.columns:
            df["options"] = df["options"].apply(_coerce_list)
            n_valid = int(df["options"].notna().sum())
            n_bad = len(df) - n_valid
            if n_bad:
                logger.warning(f"[mmlu_pro] {n_bad:,} rows with empty options")
            logger.info(f"[mmlu_pro] valid options: {n_valid:,}/{len(df):,}")

        # CoT length (often blank depending on HF snapshot)
        if "cot_content" in df.columns:
            df["cot_length"] = df["cot_content"].fillna("").str.len().astype(int)
            if df["cot_length"].sum() == 0:
                logger.info("[mmlu_pro] cot_content is blank in this HF version (expected)")

        # Stratified cap across categories
        max_n = self.cfg.get("max_samples")
        if max_n and len(df) > max_n and "category" in df.columns:
            per_cat = max(1, max_n // df["category"].nunique())
            df = (
                df.groupby("category", group_keys=False)
                  .apply(lambda g: g.sample(n=min(per_cat, len(g)), random_state=42))
                  .reset_index(drop=True)
            )
            logger.info(f"[mmlu_pro] stratified cap → {len(df):,} rows")

        if "options" in df.columns:
            df["query"] = [
                append_options_to_query(q, opts)
                for q, opts in zip(df["query"], df["options"], strict=True)
            ]
            n = int(df["options"].apply(lambda o: bool(normalize_options(o))).sum())
            if n:
                logger.info(f"[mmlu_pro] appended option list to query for {n:,} rows")

        cols = ["id", "query", "answer", "answer_index", "options",
                "category", "cot_length", "split"]
        return df[[c for c in cols if c in df.columns]].reset_index(drop=True)

    # ── Verify ────────────────────────────────────────────────────────────

    def _verify(self, df: pd.DataFrame) -> bool:
        ok = True
        if "options" in df.columns:
            avg = df["options"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            ).mean()
            if avg < 4:
                print(f"  ⚠  avg options/q = {avg:.1f} (expected ≥ 4)")
                ok = False
            else:
                print(f"  options/q  : {avg:.1f} avg")
        if "category" in df.columns:
            print(f"  categories : {df['category'].nunique()} unique")
            print(f"  top 3      : {df['category'].value_counts().head(3).to_dict()}")
        return ok