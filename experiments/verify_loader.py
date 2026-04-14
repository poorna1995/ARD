#!/usr/bin/env python3
"""
experiments/verify_loaders.py
────────────────────────────────────────────────────────────────
Verify all downloaded datasets are correctly cached and
pass quality checks.

Run from project root AFTER download_all.py:
    python experiments/verify_loaders.py

What it checks per dataset:
  - Parquet file exists in datasets/processed/
  - Required canonical columns present (id, query, answer)
  - No nulls in required columns
  - Dataset-specific checks (hop count, level distribution, etc.)
  - Prints a human-readable summary table
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.loaders import get_loader, LOADER_REGISTRY
from src.utils import setup_logging, load_config

import logging

logger = logging.getLogger(__name__)


def main():
    setup_logging("INFO")

    config_path = Path("configs/datasets.yaml")
    if not config_path.exists():
        print(f"ERROR: Config not found at {config_path}")
        sys.exit(1)

    all_configs = load_config(config_path)

    print("\n" + "═" * 60)
    print("  Task Complexity Benchmark — Loader Verification")
    print("═" * 60 + "\n")

    results = {}

    for name, LoaderCls in LOADER_REGISTRY.items():
        cfg = all_configs.get(name, {})
        parquet = Path("datasets/processed") / f"{name}.parquet"

        if not parquet.exists():
            print(f"\n  [{name}]  ⚠  Parquet not found at {parquet}")
            print(
                f"          Run: python experiments/download_all.py --datasets {name}"
            )
            results[name] = False
            continue

        try:
            loader = get_loader(name, cfg)
            df = loader.load()  # loads from cache (fast)
            passed = loader.verify()  # prints detailed summary
            results[name] = passed
        except Exception as e:
            print(f"\n  [{name}]  ✗  Exception during verify: {e}")
            logger.error(f"{name} verify failed", exc_info=True)
            results[name] = False

    # ── Final report ──────────────────────────────────────────────
    print("═" * 60)
    print("  VERIFICATION RESULTS")
    print("═" * 60)
    for name, passed in results.items():
        icon = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name:<20}  {icon}")

    total_pass = sum(results.values())
    print(f"\n  {total_pass}/{len(results)} datasets verified.")
    print("═" * 60 + "\n")

    if total_pass < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
