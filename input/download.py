"""Download, preprocess, and verify benchmark datasets (``python -m input.download``)."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── Ensure project root is on sys.path ────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from input.loaders.registry import REGISTRY, get_loader  # noqa: E402
from config.global_config.paths import CONFIG_DATASETS

# ── Logging ───────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────

def _load_env() -> None:
    """Load .env (python-dotenv if available, else manual parse)."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    try:
        from dotenv import load_dotenv
        load_dotenv(env_path)
        return
    except ImportError:
        pass
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


def _load_config(path: str) -> dict:
    from config.common import load_yaml

    return load_yaml(Path(path))


def _print_summary(df, name: str) -> None:
    """Pretty-print a quick dataset summary after loading."""
    sep = "═" * 56
    print(f"\n{sep}")
    print(f"  {name.upper():20s}  {len(df):>7,} rows × {df.shape[1]} cols")
    print(sep)

    # Query length
    ql = df["query"].str.len()
    print(f"  query len  min={ql.min():,}  median={ql.median():.0f}  max={ql.max():,}")

    # Split breakdown
    if "split" in df.columns:
        for sp, cnt in df["split"].value_counts().sort_index().items():
            bar = "█" * int(cnt / len(df) * 30)
            print(f"  {sp:<12} {bar}  {cnt:,}")

    # Sample rows
    print(f"\n  ── 2 sample rows ──")
    for _, row in df.sample(n=min(2, len(df)), random_state=0).iterrows():
        q = str(row["query"])[:120].replace("\n", " ")
        a = str(row["answer"])[:60].replace("\n", " ")
        print(f"  id    : {row['id']}")
        print(f"  query : {q}{'…' if len(str(row['query'])) > 120 else ''}")
        print(f"  answer: {a}")
        print()
    print(sep + "\n")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    _load_env()

    # Normalise HF token env var names
    if not os.getenv("HF_TOKEN"):
        if tok := os.getenv("HUGGINGFACE_TOKEN"):
            os.environ["HF_TOKEN"] = tok
            os.environ.setdefault("HUGGINGFACE_HUB_TOKEN", tok)

    parser = argparse.ArgumentParser(
        description="Download and preprocess benchmark datasets.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available datasets: {sorted(REGISTRY)}",
    )
    parser.add_argument(
        "--dataset", "-d", nargs="+", default=["all"],
        metavar="NAME",
        help="Dataset name(s) or 'all'",
    )
    parser.add_argument(
        "--force", "-f", action="store_true",
        help="Re-download even if cached parquet exists",
    )
    parser.add_argument(
        "--config", default=str(CONFIG_DATASETS),
        help="Path to datasets.yaml",
    )
    parser.add_argument(
        "--data-root", default=str(ROOT / "datasets"),  # legacy root; loaders use raw/processed subdirs
        help="Root directory for dataset storage",
    )
    parser.add_argument(
        "--split", default=None,
        help="Load only this split (e.g. train, test, validation)",
    )
    parser.add_argument(
        "--reprocess", action="store_true",
        help="Re-run processing on cached raw splits (no HF download)",
    )
    args = parser.parse_args()

    cfg_all = _load_config(args.config)
    targets = sorted(REGISTRY) if args.dataset == ["all"] else args.dataset

    results: dict[str, str] = {}

    for name in targets:
        print(f"\n{'─'*56}")
        logger.info(f"Starting: {name}")

        if name not in REGISTRY:
            logger.error(f"Unknown dataset '{name}'. Available: {sorted(REGISTRY)}")
            results[name] = "UNKNOWN"
            continue

        if name not in cfg_all:
            logger.error(f"No config entry for '{name}' in {args.config}")
            results[name] = "NO CONFIG"
            continue

        try:
            loader = get_loader(name, cfg_all[name], data_root=args.data_root)
            df = loader.load(
                force=args.force,
                reprocess=args.reprocess,
                split=args.split,
            )
            _print_summary(df, name)
            ok = loader.verify()
            results[name] = "✓ PASS" if ok else "✗ FAIL"

        except Exception as exc:
            logger.error(f"[{name}] FAILED: {exc}", exc_info=True)
            results[name] = f"ERROR: {exc}"

            # Dataset-specific hints
            if name == "gaia":
                print("  Tip: GAIA needs HF auth → huggingface-cli login")
                print("       or set HF_TOKEN=hf_... in your .env file")
            elif name == "math":
                print("  Tip: pip install datasets --upgrade")

    # ── Final summary table ───────────────────────────────────────────
    print(f"\n{'═'*40}")
    print("  RESULTS")
    print(f"{'═'*40}")
    for name, status in results.items():
        print(f"  {name:<12} {status}")
    print(f"{'═'*40}\n")


if __name__ == "__main__":
    main()