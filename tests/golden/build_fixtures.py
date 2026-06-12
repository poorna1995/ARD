#!/usr/bin/env python3
"""
Regenerate universe-A golden inference fixtures (internal QCE test split sample).

IN:  datasets/qce_features + qce_internal_test.csv (via load_split)
OUT: tests/golden/inference_universe_a.parquet, expected_predictions_universe_a.json

Run after intentional router retrain only — then review diff and update manifest checksums.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from routing.config import PRIMARY_ROUTER_PATH, REPO_ROOT  # noqa: E402
from routing.router import PROBA_COLS, load_router_frame, load_split  # noqa: E402

GOLDEN = Path(__file__).resolve().parent
N_ROWS = 20
SEED = 42


def main() -> None:
    df = load_split("test", with_embeddings=True)
    sample = df.sample(n=min(N_ROWS, len(df)), random_state=SEED).copy()
    routed = load_router_frame(sample, router_path=PRIMARY_ROUTER_PATH)

    feat_cols = [c for c in routed.columns if c.startswith(("dim_", "emb_")) or c in ("dataset", "training_id")]
    keep = list(dict.fromkeys(["training_id"] + feat_cols))
    routed[keep].to_parquet(GOLDEN / "inference_universe_a.parquet", index=False)

    expected: dict[str, dict] = {}
    for _, row in routed.iterrows():
        tid = str(row["training_id"])
        expected[tid] = {
            "router_pred": str(row["router_pred"]),
            **{c: float(row[c]) for c in PROBA_COLS},
            "max_prob": float(row["max_prob"]),
        }
    (GOLDEN / "expected_predictions_universe_a.json").write_text(
        json.dumps(expected, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(expected)} rows to {GOLDEN}")
    print(f"Router: {PRIMARY_ROUTER_PATH.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
