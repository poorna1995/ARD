"""
src/loaders/musique_loader.py
────────────────────────────────────────────────────────────────
Loader for MuSiQue multi-hop QA dataset.

MuSiQue is a multi-hop QA benchmark where each question requires
chaining 2, 3, or 4 reasoning steps (hops). The number of hops
is an excellent built-in complexity proxy:
  2-hop → moderate
  3-hop → hard
  4-hop → very hard

HuggingFace ID: "dgslibs/musique" or manual JSON download.
The official source is: https://github.com/StonyBrookNLP/musique
We use the HF mirror that is most commonly available.

Canonical columns:
  id           → id
  query        → question
  answer       → answer

Extra columns kept:
  answerable, decomposition (sub-questions), paragraphs,
  num_hops (derived), complexity_gt (derived from hops)
"""

from __future__ import annotations

import logging

import pandas as pd

from .base import BaseLoader

logger = logging.getLogger(__name__)


class MuSiQueLoader(BaseLoader):

    DATASET_NAME = "musique"

    # Hop count → normalized complexity ground truth
    HOP_COMPLEXITY_MAP = {2: 0.35, 3: 0.65, 4: 0.90}

    def _download(self) -> pd.DataFrame:
        # Try config-provided HF repo first, then known fallbacks
        configured_repo = self.config.get("hf_repo")
        repos_to_try = []
        if configured_repo:
            repos_to_try.append((configured_repo, None))
        repos_to_try.extend(
            [
                ("dgslibisey/MuSiQue", None),
                ("dgslibisey/musique", None),
                ("dgslibs/musique", None),
                ("Jeko-2003/MuSiQue", None),
                ("musique", None),
            ]
        )

        # Deduplicate while preserving order
        seen = set()
        repos_to_try = [r for r in repos_to_try if not (r[0] in seen or seen.add(r[0]))]

        for repo, config in repos_to_try:
            try:
                from datasets import load_dataset

                split = self.config.get("split", "validation")
                logger.info(f"  Trying HF repo: {repo}")
                if config:
                    ds = load_dataset(repo, config, split=split)
                else:
                    ds = load_dataset(repo, split=split)
                df = ds.to_pandas()
                logger.info(f"  ✓ Loaded {len(df)} rows from {repo}")
                return df
            except Exception as e:
                logger.warning(f"  Repo {repo} failed: {e}")

        # Fallback: try to load from local raw dir
        raw_path = self.raw_dir / "musique_ans_v1.0_dev.jsonl"
        if raw_path.exists():
            logger.info(f"  Falling back to local file: {raw_path}")
            return pd.read_json(raw_path, lines=True)

        raise RuntimeError(
            "MuSiQue: Could not load from any HF repo or local file.\n"
            "Download manually from: https://github.com/StonyBrookNLP/musique\n"
            f"Place musique_ans_v1.0_dev.jsonl in: {self.raw_dir}"
        )

    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # ── Rename to canonical columns ───────────────────────────
        df = df.rename(
            columns={
                "question": "query",
            }
        )
        # 'id' and 'answer' usually already named correctly

        # ── Filter to answerable questions only ───────────────────
        if "answerable" in df.columns:
            before = len(df)
            df = df[df["answerable"] == True].copy()
            logger.info(f"  Filtered to answerable: {before} → {len(df)} rows")

        # ── Derive num_hops from decomposition ────────────────────
        # decomposition is a list of sub-questions
        if "decomposition" in df.columns:

            def count_hops(decomp):
                if isinstance(decomp, list):
                    return len(decomp)
                return 0

            df["num_hops"] = df["decomposition"].apply(count_hops)
        elif "question_decomposition" in df.columns:
            df["num_hops"] = df["question_decomposition"].apply(
                lambda x: len(x) if isinstance(x, list) else 0
            )
            df = df.rename(columns={"question_decomposition": "decomposition"})
        else:
            # Fallback: estimate hops from question id suffix (_2hop, _3hop, _4hop)
            def infer_hops_from_id(qid):
                for k in [4, 3, 2]:
                    if f"_{k}hop" in str(qid) or f"_{k}_" in str(qid):
                        return k
                return 2  # default

            df["num_hops"] = df["id"].apply(infer_hops_from_id)
            logger.warning(
                "  'decomposition' column not found; "
                "inferred num_hops from ID strings"
            )

        # # ── Map hops → normalized complexity ground truth ─────────
        # df["complexity_gt"] = df["num_hops"].map(self.HOP_COMPLEXITY_MAP)
        # # Fill any unmapped hop counts with max complexity
        # df["complexity_gt"] = df["complexity_gt"].fillna(0.90)
        # logger.info("  Added 'complexity_gt' from hop count (ground truth)")

        # ── Keep only needed columns ──────────────────────────────
        keep = [
            "id",
            "query",
            "answer",
            "num_hops",
            # "complexity_gt",
            "decomposition",
            "paragraphs",
        ]
        keep = [c for c in keep if c in df.columns]
        df = df[keep].copy()

        df = self._apply_max_samples(df)
        df = df.reset_index(drop=True)
        return df

    def _extra_verify(self, df: pd.DataFrame) -> bool:
        ok = True
        if "num_hops" in df.columns:
            hop_dist = df["num_hops"].value_counts().sort_index().to_dict()
            print(f"  Hop dist : {hop_dist}")
            if max(hop_dist.keys(), default=0) < 3:
                print("  ⚠  No 3-hop or 4-hop questions found")
                ok = False
        if "complexity_gt" in df.columns:
            print(
                f"  GT range : {df['complexity_gt'].min():.2f} – "
                f"{df['complexity_gt'].max():.2f}"
            )
        return ok
