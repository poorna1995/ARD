from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download, list_repo_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GAIA attachment files listed in raw parquet."
    )
    parser.add_argument(
        "--parquet",
        default="datasets/raw/raw_gaia.parquet",
        help="Path to raw GAIA parquet containing file_path/file_name.",
    )
    parser.add_argument(
        "--out",
        default="datasets/gaia_files",
        help="Local output root to store attachment files.",
    )
    parser.add_argument(
        "--repo-id",
        default="gaia-benchmark/GAIA",
        help="Hugging Face GAIA dataset repo id.",
    )
    parser.add_argument(
        "--repo-type",
        default="dataset",
        help="Hugging Face repo type.",
    )
    parser.add_argument(
        "--index-out",
        default="datasets/processed/gaia_attachment_index.parquet",
        help="Where to save task_id -> local path mapping for retrieval.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print actions without downloading files.",
    )
    return parser.parse_args()


def _required_attachments(df: pd.DataFrame) -> pd.DataFrame:
    if "task_id" not in df.columns:
        raise ValueError("Expected 'task_id' in parquet.")
    if "file_path" not in df.columns:
        raise ValueError("Expected 'file_path' in parquet.")

    subset = df[["task_id", "file_name", "file_path"]].copy()
    subset["file_path"] = subset["file_path"].fillna("").astype(str).str.strip()
    subset["file_name"] = subset["file_name"].fillna("").astype(str).str.strip()
    subset = subset[subset["file_path"] != ""].drop_duplicates(subset=["task_id", "file_path"])
    subset = subset.reset_index(drop=True)
    return subset


def _build_index(subset: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    index_df = subset.copy()
    index_df["relative_attachment_path"] = index_df["file_path"]
    index_df["local_attachment_path"] = index_df["file_path"].map(
        lambda p: str((out_root / p).resolve())
    )
    index_df["local_exists"] = index_df["file_path"].map(lambda p: (out_root / p).exists())
    return index_df


def main() -> None:
    args = parse_args()

    parquet_path = Path(args.parquet)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(parquet_path)
    needed = _required_attachments(df)
    required_paths = sorted(set(needed["file_path"].tolist()))

    print(f"Rows in parquet: {len(df)}")
    print(f"Rows requiring attachments: {len(needed)}")
    print(f"Unique attachment paths required: {len(required_paths)}")

    repo_files = set(list_repo_files(repo_id=args.repo_id, repo_type=args.repo_type))
    missing_on_hf = [path for path in required_paths if path not in repo_files]
    if missing_on_hf:
        print(f"ERROR: {len(missing_on_hf)} attachment paths not found on HF.")
        for path in missing_on_hf[:20]:
            print(f"  - {path}")
        raise SystemExit(1)

    print("Verified: all required file_path values exist on Hugging Face.")

    downloaded = 0
    already_present = 0
    for rel_path in required_paths:
        target = out_root / rel_path
        if target.exists():
            already_present += 1
            continue

        if args.dry_run:
            print(f"[DRY RUN] would download {rel_path} -> {target}")
            continue

        target.parent.mkdir(parents=True, exist_ok=True)
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type=args.repo_type,
            filename=rel_path,
            local_dir=str(out_root),
            local_dir_use_symlinks=False,
        )
        downloaded += 1
        print(f"Downloaded: {rel_path}")

    index_path = Path(args.index_out)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    index_df = _build_index(needed, out_root)
    index_df.to_parquet(index_path, index=False)

    missing_local = int((~index_df["local_exists"]).sum())
    print("\nDone.")
    print(f"Downloaded now: {downloaded}")
    print(f"Already present: {already_present}")
    print(f"Index written: {index_path.resolve()}")
    print(f"Attachment rows still missing locally: {missing_local}")
    print(f"Attachment root: {out_root.resolve()}")


if __name__ == "__main__":
    main()
