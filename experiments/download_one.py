#!/usr/bin/env python3
"""
experiments/download_one.py
────────────────────────────────────────────────────────────────
Download a SINGLE dataset, inspect it, and save to parquet.

Usage (from project root):
    python experiments/download_one.py --dataset gaia
    python experiments/download_one.py --dataset musique
    python experiments/download_one.py --dataset mmlu_pro
    python experiments/download_one.py --dataset swe_bench
    python experiments/download_one.py --dataset aime

What it does:
    1. Downloads the dataset from HuggingFace
    2. Processes it into canonical columns (id, query, answer, ...)
    3. Saves to datasets/processed/<name>.parquet
    4. Prints a full inspection report:
         - shape, columns, null counts
         - sample rows
         - complexity_gt distribution (if available)
         - query length stats
"""

import argparse
import os
import sys
from pathlib import Path

# Works regardless of what the root folder is named
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
print(f"[debug] project root: {project_root}")  # remove after confirming

from src.loaders import get_loader, LOADER_REGISTRY
from src.utils import setup_logging, load_config

import logging

logger = logging.getLogger(__name__)


def _load_env_file():
    """Load .env if python-dotenv exists; otherwise parse basic KEY=VALUE lines."""
    env_path = project_root / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_path)
        return
    except Exception:
        pass

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        os.environ.setdefault(key, val)


def inspect(df, name: str):
    """Print a thorough inspection of the loaded DataFrame."""
    sep = "═" * 58

    print(f"\n{sep}")
    print(f"  DATASET : {name.upper()}")
    print(f"  Shape   : {df.shape[0]:,} rows  ×  {df.shape[1]} columns")
    print(sep)

    # ── Columns & types ───────────────────────────────────────
    print("\n── Columns ──────────────────────────────────────────")
    for col in df.columns:
        dtype = str(df[col].dtype)
        nulls = df[col].isna().sum()
        null_str = f"  ({nulls} nulls ⚠)" if nulls else ""
        print(f"  {col:<25} {dtype:<12}{null_str}")

    # ── Query length stats ────────────────────────────────────
    print("\n── Query length (chars) ─────────────────────────────")
    ql = df["query"].str.len()
    print(f"  min    : {ql.min():>8,}")
    print(f"  median : {ql.median():>8,.0f}")
    print(f"  mean   : {ql.mean():>8,.0f}")
    print(f"  max    : {ql.max():>8,}")

    # ── Complexity ground truth (if available) ────────────────
    # if "complexity_gt" in df.columns:
    #     print("\n── Complexity GT distribution ───────────────────────")
    #     gt = df["complexity_gt"].dropna()
    #     print(f"  range  : {gt.min():.2f} → {gt.max():.2f}")
    #     print(f"  mean   : {gt.mean():.3f}")
    #     # Value counts
    #     vc = gt.value_counts().sort_index()
    #     for val, cnt in vc.items():
    #         bar = "█" * int(cnt / vc.max() * 20)
    #         print(f"  {val:.2f}  {bar}  {cnt}")

    if "level" in df.columns:
        print("\n── GAIA Level distribution ──────────────────────────")
        vc = df["level"].value_counts().sort_index()
        labels = {1: "Level 1 (easy)  ", 2: "Level 2 (medium)", 3: "Level 3 (hard)  "}
        for lvl, cnt in vc.items():
            bar = "█" * int(cnt / vc.max() * 25)
            print(f"  {labels.get(lvl, str(lvl))}  {bar}  {cnt}")

    if "num_hops" in df.columns:
        print("\n── MuSiQue hop distribution ─────────────────────────")
        vc = df["num_hops"].value_counts().sort_index()
        for hops, cnt in vc.items():
            bar = "█" * int(cnt / vc.max() * 25)
            print(f"  {hops}-hop  {bar}  {cnt}")

    if "category" in df.columns:
        print("\n── MMLU-Pro categories ──────────────────────────────")
        vc = df["category"].value_counts().head(8)
        for cat, cnt in vc.items():
            bar = "█" * int(cnt / vc.max() * 20)
            print(f"  {cat:<20} {bar}  {cnt}")

    if "repo" in df.columns:
        print("\n── SWE-Bench repos (top 5) ──────────────────────────")
        vc = df["repo"].value_counts().head(5)
        for repo, cnt in vc.items():
            print(f"  {repo:<40}  {cnt}")

    # ── Sample rows ───────────────────────────────────────────
    print("\n── 3 random sample rows ─────────────────────────────")
    samples = df.sample(n=min(3, len(df)), random_state=42)
    for i, (_, row) in enumerate(samples.iterrows(), 1):
        print(f"\n  ── Sample {i} ──")
        print(f"  ID     : {row['id']}")
        query_preview = str(row["query"])[:200].replace("\n", " ")
        print(
            f"  Query  : {query_preview}{'...' if len(str(row['query'])) > 200 else ''}"
        )
        answer_preview = str(row["answer"])[:80].replace("\n", " ")
        print(f"  Answer : {answer_preview}")
        # if "complexity_gt" in row:
        #     print(f"  GT     : {row['complexity_gt']}")

    print(f"\n{sep}")
    print(f"  Saved → datasets/processed/{name}.parquet")
    print(sep + "\n")


def main():
    # ── Load .env and normalize HF token variable names ─────────
    _load_env_file()
    if not os.getenv("HF_TOKEN"):
        legacy_token = os.getenv("HUGGINGFACE_TOKEN")
        if legacy_token:
            os.environ["HF_TOKEN"] = legacy_token
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", legacy_token)

    parser = argparse.ArgumentParser(
        description="Download and inspect a single benchmark dataset"
    )
    parser.add_argument(
        "--dataset",
        "-d",
        required=True,
        choices=list(LOADER_REGISTRY.keys()),
        help=f"Dataset to download. Options: {list(LOADER_REGISTRY.keys())}",
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Re-download even if cached parquet already exists",
    )
    parser.add_argument(
        "--config", default="configs/datasets.yaml", help="Path to datasets.yaml"
    )
    parser.add_argument(
        "--data-root",
        default="datasets",
        help="Root directory for dataset storage (default: datasets/)",
    )
    args = parser.parse_args()

    setup_logging("INFO", log_file=f"logs/download_{args.dataset}.log")

    # ── Load config ───────────────────────────────────────────
    all_configs = load_config(args.config)
    if args.dataset not in all_configs:
        print(f"ERROR: No config for '{args.dataset}' in {args.config}")
        sys.exit(1)

    cfg = all_configs[args.dataset]
    print(f"\nDownloading: {cfg.get('display_name', args.dataset)}")
    print(f"  HF repo : {cfg.get('hf_repo', 'N/A')}")
    print(f"  split   : {cfg.get('split', 'N/A')}")
    print(f"  max     : {cfg.get('max_samples', 'all')}")

    # ── Download & process ────────────────────────────────────
    loader = get_loader(args.dataset, cfg, data_root=args.data_root)

    try:
        df = loader.load(force_redownload=args.force)
    except Exception as e:
        print(f"\nERROR during download: {e}")
        print("\nTroubleshooting tips:")
        if args.dataset == "gaia":
            print("  • GAIA requires HuggingFace login.")
            print("    Run: huggingface-cli login")
            print("    Or:  export HF_TOKEN=hf_your_token_here")
            print("    Then request access at: huggingface.co/gaia-benchmark/GAIA")
        elif args.dataset == "musique":
            print("  • Try: pip install datasets --upgrade")
            print("  • Or download manually from:")
            print("    https://github.com/StonyBrookNLP/musique")
            print(f"    Place musique_ans_v1.0_dev.jsonl in: datasets/raw/musique/")
        else:
            print("  • Check your internet connection")
            print("  • Try: pip install datasets --upgrade")
        sys.exit(1)

    # ── Inspect ───────────────────────────────────────────────
    inspect(df, args.dataset)

    # ── Verify ───────────────────────────────────────────────
    loader.verify()


if __name__ == "__main__":
    main()
