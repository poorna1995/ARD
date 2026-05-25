#!/usr/bin/env python3
"""Paper figures: complexity regions, calibration, top-2 routing, cost–accuracy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from routing.router_analysis import (  # noqa: E402
    DEFAULT_ORACLE,
    DEFAULT_ROUTER,
    run_full_analysis,
)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Router paper analyses and figures.")
    p.add_argument(
        "--split",
        choices=("val", "test", "both"),
        default="both",
        help="Eval split(s) to analyze (default: val + test).",
    )
    p.add_argument(
        "--router",
        type=Path,
        default=DEFAULT_ROUTER,
        help="Saved router .joblib (default: G3 graph_emb balanced).",
    )
    p.add_argument(
        "--oracle",
        type=Path,
        default=DEFAULT_ORACLE,
        help="Long-form oracle CSV with per-agent cost/correctness.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output root (default: results/router_analysis/<split>/).",
    )
    p.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip query embeddings merge (graph-only routers only).",
    )
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    splits = ["val", "test"] if args.split == "both" else [args.split]
    root = args.out_dir or (_REPO_ROOT / "results/router_analysis")

    all_summaries: dict[str, dict] = {}
    for split in splits:
        out = root / split
        print(f"\n=== Analyzing {split!r} → {out} ===")
        summary = run_full_analysis(
            split,
            out_dir=out,
            router_path=args.router,
            oracle_path=args.oracle,
            with_embeddings=not args.no_embeddings,
        )
        all_summaries[split] = summary
        print(json.dumps(summary, indent=2))
        print(f"Wrote figures and tables under {out}/")

    if len(all_summaries) > 1:
        combined_path = root / "summary_all_splits.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        combined_path.write_text(
            json.dumps(all_summaries, indent=2),
            encoding="utf-8",
        )
        print(f"\nCombined summary: {combined_path}")


if __name__ == "__main__":
    main()
