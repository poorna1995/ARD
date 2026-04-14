#!/usr/bin/env python3
"""
experiments/download_all.py
────────────────────────────────────────────────────────────────
Download and cache all benchmark datasets.

Run from the project root:
    python experiments/download_all.py

Options:
    --datasets  swe_bench gaia mmlu_pro   (space-separated, default: all)
    --force                               (re-download even if cached)
    --dry-run                             (print what would run, don't download)

What it does:
    1. Reads configs/datasets.yaml
    2. For each active dataset, calls loader.load()
    3. Saves processed parquet to datasets/processed/<name>.parquet
    4. Prints a summary table at the end
"""

import argparse
import os
import sys
import time
from pathlib import Path

# ── Make src importable from project root ─────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.loaders import get_loader, LOADER_REGISTRY
from src.utils import setup_logging, load_config

import logging

logger = logging.getLogger(__name__)


def _load_env_file():
    """Load .env if python-dotenv exists; otherwise parse basic KEY=VALUE lines."""
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except Exception:
        pass

    # Minimal fallback parser for simple .env files.
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def parse_args():
    parser = argparse.ArgumentParser(description="Download all benchmark datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(LOADER_REGISTRY.keys()),
        choices=list(LOADER_REGISTRY.keys()),
        help="Which datasets to download (default: all)",
    )
    parser.add_argument(
        "--force", action="store_true", help="Re-download even if cached parquet exists"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print plan without downloading"
    )
    parser.add_argument(
        "--config", default="configs/datasets.yaml", help="Path to datasets.yaml"
    )
    parser.add_argument(
        "--data-root", default="datasets", help="Root directory for dataset storage"
    )
    parser.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser.parse_args()


def main():
    # ── Load .env and normalize HF token variable names ─────────
    _load_env_file()
    if not os.getenv("HF_TOKEN"):
        legacy_token = os.getenv("HUGGINGFACE_TOKEN")
        if legacy_token:
            os.environ["HF_TOKEN"] = legacy_token
            # Keep backward-compat naming too.
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", legacy_token)

    args = parse_args()
    setup_logging(
        level=args.log_level, log_file=f"logs/download_{int(time.time())}.log"
    )

    # ── Load config ───────────────────────────────────────────────
    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)
    all_configs = load_config(config_path)

    print("\n" + "═" * 60)
    print("  Task Complexity Benchmark — Dataset Download")
    print("═" * 60)
    print(f"  Datasets : {args.datasets}")
    print(f"  Force    : {args.force}")
    print(f"  Data root: {args.data_root}")
    print("═" * 60 + "\n")

    if args.dry_run:
        print("DRY RUN — no data will be downloaded.\n")
        for name in args.datasets:
            cfg = all_configs.get(name, {})
            print(f"  Would download: {name}")
            print(f"    repo    : {cfg.get('hf_repo', 'N/A')}")
            print(f"    split   : {cfg.get('split', 'N/A')}")
            print(f"    max     : {cfg.get('max_samples', 'all')}")
            print()
        return

    # ── Download loop ─────────────────────────────────────────────
    results = {}

    for name in args.datasets:
        if name not in all_configs:
            logger.warning(f"No config found for '{name}' in {config_path}. Skipping.")
            results[name] = {"status": "NO_CONFIG", "rows": 0, "time_s": 0}
            continue

        cfg = all_configs[name]
        print(f"\n{'─'*50}")
        print(f"  Downloading: {cfg.get('display_name', name)}")
        print(f"{'─'*50}")

        t0 = time.time()
        try:
            loader = get_loader(name, cfg, data_root=args.data_root)
            df = loader.load(force_redownload=args.force)
            elapsed = time.time() - t0

            results[name] = {
                "status": "OK",
                "rows": len(df),
                "columns": list(df.columns),
                "time_s": round(elapsed, 1),
            }
            print(f"  ✓  {len(df)} rows  |  {elapsed:.1f}s")

        except Exception as e:
            elapsed = time.time() - t0
            logger.error(f"  ✗  {name} FAILED: {e}", exc_info=True)
            results[name] = {
                "status": "FAILED",
                "rows": 0,
                "time_s": round(elapsed, 1),
                "error": str(e),
            }

    # ── Summary table ─────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  SUMMARY")
    print("═" * 60)
    print(f"  {'Dataset':<18} {'Status':<10} {'Rows':>8} {'Time':>8}")
    print(f"  {'─'*18} {'─'*10} {'─'*8} {'─'*8}")
    for name, r in results.items():
        status_sym = "✓" if r["status"] == "OK" else "✗"
        print(
            f"  {name:<18} {status_sym} {r['status']:<8} "
            f"{r['rows']:>8,} {r['time_s']:>7.1f}s"
        )

    total_ok = sum(1 for r in results.values() if r["status"] == "OK")
    total = len(results)
    print(f"\n  {total_ok}/{total} datasets downloaded successfully.")
    print("═" * 60 + "\n")

    if total_ok < total:
        sys.exit(1)  # non-zero exit so CI can detect failures


if __name__ == "__main__":
    main()
