#!/usr/bin/env python3
"""
Create Phase 5 directory layout with symlinks to legacy paths (no data copies).

Safe to re-run; skips links that already exist and match.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

SPLIT_FILES = ("train", "val", "test")
EVAL_TAGS = ("gaia", "hotpot", "math", "mmlu", "musique", "mmlu_pro")


def _link(link: Path, target: Path, *, relative: bool = True) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        print(f"skip (not a symlink): {link}", file=sys.stderr)
        return
    dest = os.path.relpath(target, link.parent) if relative else str(target)
    link.symlink_to(dest)
    print(f"linked {link} -> {dest}")


def main() -> None:
    legacy_eval = REPO / "datasets/eval_samples"
    legacy_labels = REPO / "datasets/train_samples/v1"
    legacy_qce = REPO / "datasets/qce_features"
    legacy_cache = REPO / "datasets/decomposer_cache"
    legacy_graph = REPO / "models/router/graph_main"

    new_eval = REPO / "datasets/input/samples/eval"
    new_labels = REPO / "datasets/input/labels/v1"
    split_dir = REPO / "datasets/intermediate/qce_features/by_split"
    by_dataset = REPO / "datasets/intermediate/qce_features/by_dataset"
    new_cache = REPO / "datasets/intermediate/decomposer_cache"
    production = REPO / "models/router/production"
    selective = REPO / "models/router/selective"

    if legacy_eval.is_dir():
        for p in legacy_eval.glob("*.parquet"):
            _link(new_eval / p.name, p)

    if legacy_labels.is_dir():
        for p in legacy_labels.iterdir():
            if p.is_file():
                _link(new_labels / p.name, p)

    if legacy_qce.is_dir():
        for sp in SPLIT_FILES:
            for prefix in ("complexity_record", "query_embeddings", "query_heuristics"):
                src = legacy_qce / f"{prefix}_{sp}.parquet"
                if src.is_file():
                    _link(split_dir / src.name, src)
        for tag in EVAL_TAGS:
            tag_dir = by_dataset / tag
            for prefix in ("complexity_record", "query_embeddings"):
                src = legacy_qce / f"{prefix}_{tag}.parquet"
                if src.is_file():
                    _link(tag_dir / src.name, src)

    if legacy_cache.is_dir():
        for p in legacy_cache.glob("*.jsonl"):
            _link(new_cache / p.name, p)

    if legacy_graph.is_dir() and not production.exists():
        _link(production, legacy_graph)

    cheap = legacy_graph / "hgbm_emb_only_soft_kl.joblib"
    if cheap.is_file():
        _link(selective / cheap.name, cheap)
        meta = cheap.with_suffix(".json")
        if meta.is_file():
            _link(selective / meta.name, meta)

    runs = REPO / "results/runs"
    runs.mkdir(parents=True, exist_ok=True)
    print(f"ready: {runs}")


if __name__ == "__main__":
    main()
