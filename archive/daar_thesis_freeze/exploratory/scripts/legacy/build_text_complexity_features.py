#!/usr/bin/env python3
"""Build QCE text complexity feature parquets from a stored query corpus."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from qce.text_complexity_pipeline import (  # noqa: E402
    DEFAULT_CORPUS_PATH,
    DEFAULT_TEXT_ESTIMATOR_DIR,
    DEFAULT_TEXT_FEATURES_DIR,
    build_text_complexity_features,
)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_CORPUS_PATH,
        help="Corpus CSV with training_id, query, dataset",
    )
    parser.add_argument("--train-csv", type=Path, default=None)
    parser.add_argument("--val-csv", type=Path, default=None)
    parser.add_argument("--test-csv", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_TEXT_FEATURES_DIR)
    parser.add_argument(
        "--calibrator-dir", type=Path, default=DEFAULT_TEXT_ESTIMATOR_DIR
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="Include discarded rows (default: labeled only)",
    )
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    result = build_text_complexity_features(
        source_corpus_path=args.source,
        train_corpus_path=args.train_csv,
        validation_corpus_path=args.val_csv,
        test_corpus_path=args.test_csv,
        output_directory=args.out_dir,
        estimator_directory=args.calibrator_dir,
        labeled_only=not args.all_rows,
        train_fraction=args.train_frac,
        validation_fraction=args.val_frac,
        random_state=args.seed,
    )
    print(
        f"Text complexity features built: "
        f"train={result.train_row_count} "
        f"val={result.validation_row_count} "
        f"test={result.test_row_count}\n"
        f"  {result.train_artifact_path}\n"
        f"  {result.validation_artifact_path}\n"
        f"  {result.test_artifact_path}\n"
        f"  estimator: {result.estimator_directory}"
    )


if __name__ == "__main__":
    main()
