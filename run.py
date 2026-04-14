"""
pipeline.py
===========
End-to-end pipeline: datasets → TaskComplexityEstimator → scored parquet files.

Usage
-----
    from complexity.pipeline import run_pipeline

    results = run_pipeline(
        data_paths   = {"gaia": Path("datasets/processed/gaia.parquet"), ...},
        output_dir   = Path("data/processed"),
        query_col_map = {"gaia": "query", ...},   # optional, uses defaults
    )

The function returns a dict with:
  scored_datasets   : dict[name → scored DataFrame]
  summary_df        : cross-dataset summary table (also saved to disk)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from tqdm import tqdm
from pathlib import Path
from src.complexity.pipeline import run_pipeline
from src.config import EXPECTED_DIFFICULTY_RANK, QUERY_COL_MAP
from src.complexity.estimator import TaskComplexityEstimator

logger = logging.getLogger(__name__)

# Columns produced by the estimator that get merged back into each dataset
_COMPLEXITY_COLS: List[str] = [
    "C", "S", "R", "T", "D", "TT",
    "band", "task_type", "bloom_level",
    "tool_categories", "domains",
]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_datasets(
    data_paths: Dict[str, Path],
    query_col_map: Optional[Dict[str, str]] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Load all datasets from parquet files and validate the query column.

    Parameters
    ----------
    data_paths    : dict mapping dataset name → Path to .parquet file
    query_col_map : dict mapping dataset name → column name for the query text.
                    Falls back to ``config.QUERY_COL_MAP``, then to "query".

    Returns
    -------
    dict[name → DataFrame]
    """
    col_map  = {**QUERY_COL_MAP, **(query_col_map or {})}
    datasets: Dict[str, pd.DataFrame] = {}

    for name, path in data_paths.items():
        path = Path(path)
        if not path.exists():
            logger.warning("Dataset '%s' not found at %s — skipping.", name, path)
            continue

        df = pd.read_parquet(path)
        q_col = col_map.get(name, "query")

        if q_col not in df.columns:
            candidates = [c for c in df.columns if "query" in c.lower() or "question" in c.lower()]
            raise KeyError(
                f"[{name}] Column '{q_col}' not found. "
                f"Possible candidates: {candidates or list(df.columns)}"
            )

        logger.info(
            "Loaded %s — %d rows × %d cols  (query col: '%s')",
            name.upper(), len(df), df.shape[1], q_col,
        )
        datasets[name] = df

    return datasets


# ─────────────────────────────────────────────────────────────────────────────
# Scoring a single dataset
# ─────────────────────────────────────────────────────────────────────────────

def score_dataset(
    name: str,
    df: pd.DataFrame,
    estimator: TaskComplexityEstimator,
    query_col: str = "query",
) -> pd.DataFrame:
    """
    Score every query in *df* and merge complexity columns back.

    Parameters
    ----------
    name      : dataset name (for logging)
    df        : raw dataset DataFrame
    estimator : a ready-to-use TaskComplexityEstimator
    query_col : name of the column containing query strings

    Returns
    -------
    DataFrame = original df columns + complexity columns side-by-side
    """
    queries: List[str] = df[query_col].fillna("").astype(str).tolist()

    sep = "─" * 60
    print(f"\n{sep}")
    print(f"⚙  Scoring : {name.upper()}  ({len(queries):,} queries)")

    scored_df = estimator.score_batch(queries)

    merged_df = pd.concat(
        [df.reset_index(drop=True), scored_df[_COMPLEXITY_COLS].reset_index(drop=True)],
        axis=1,
    )

    # Per-dataset summary
    band_counts = merged_df["band"].value_counts()
    n           = len(merged_df)
    print(f"   Avg C  : {merged_df['C'].mean():.3f}  "
          f"(min={merged_df['C'].min():.3f}, max={merged_df['C'].max():.3f})")
    print("   Bands  :")
    for band, count in band_counts.items():
        print(f"     {band:<12} {count:>6,}  ({count / n * 100:5.1f}%)")

    return merged_df


# ─────────────────────────────────────────────────────────────────────────────
# Cross-dataset summary
# ─────────────────────────────────────────────────────────────────────────────

def build_summary(scored_datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a cross-dataset summary table with mean sub-scores and band
    distribution percentages.

    Also attaches the expected human difficulty rank from ``config``
    (useful for Spearman rank-correlation checks).

    Parameters
    ----------
    scored_datasets : dict[name → merged DataFrame from score_dataset()]

    Returns
    -------
    pd.DataFrame indexed by dataset name
    """
    rows = []
    for name, df in scored_datasets.items():
        band_clean = df["band"].str.extract(r"(LOW|MEDIUM|HIGH|VERY HIGH)")[0]
        rows.append({
            "dataset":   name,
            "n_queries": len(df),
            "mean_C":    round(df["C"].mean(),  3),
            "mean_S":    round(df["S"].mean(),  3),
            "mean_R":    round(df["R"].mean(),  3),
            "mean_T":    round(df["T"].mean(),  3),
            "mean_D":    round(df["D"].mean(),  3),
            "mean_TT":   round(df["TT"].mean(), 3),
            "pct_LOW":       round((band_clean == "LOW").sum()       / len(df) * 100, 1),
            "pct_MEDIUM":    round((band_clean == "MEDIUM").sum()    / len(df) * 100, 1),
            "pct_HIGH":      round((band_clean == "HIGH").sum()      / len(df) * 100, 1),
            "pct_VERY_HIGH": round((band_clean == "VERY HIGH").sum() / len(df) * 100, 1),
        })

    summary_df = pd.DataFrame(rows).set_index("dataset")
    summary_df["expected_rank"] = summary_df.index.map(
        lambda n: EXPECTED_DIFFICULTY_RANK.get(n, -1)
    )
    return summary_df


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    data_paths:    Dict[str, Path],
    output_dir:    Path,
    query_col_map: Optional[Dict[str, str]] = None,
    weights:       Optional[Dict[str, float]] = None,
) -> Dict:
    """
    Full pipeline: load → score → save → summarise.

    Parameters
    ----------
    data_paths    : dict[name → Path] to raw parquet files
    output_dir    : directory where scored parquets and summary are written
    query_col_map : override query column names per dataset
    weights       : custom estimator weights (optional)

    Returns
    -------
    dict with:
      scored_datasets : dict[name → merged DataFrame]
      summary_df      : cross-dataset summary table
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Load ──────────────────────────────────────────────────────────────────
    col_map  = {**QUERY_COL_MAP, **(query_col_map or {})}
    datasets = load_datasets(data_paths, col_map)

    if not datasets:
        raise RuntimeError("No datasets were loaded — check data_paths.")

    # ── Score ─────────────────────────────────────────────────────────────────
    estimator       = TaskComplexityEstimator(weights=weights)
    scored_datasets: Dict[str, pd.DataFrame] = {}

    for name, df in datasets.items():
        q_col     = col_map.get(name, "query")
        merged_df = score_dataset(name, df, estimator, query_col=q_col)
        scored_datasets[name] = merged_df

        out_path = output_dir / f"{name}_complexity.parquet"
        merged_df.to_parquet(out_path, index=False)
        print(f"   💾 Saved → {out_path}")

    # ── Summarise ─────────────────────────────────────────────────────────────
    summary_df = build_summary(scored_datasets)

    sep = "═" * 60
    print(f"\n{sep}")
    print("📊 CROSS-DATASET COMPLEXITY SUMMARY")
    print(sep)
    print(summary_df.drop(columns="expected_rank", errors="ignore").to_string())

    summary_path = output_dir / "all_datasets_complexity_summary.parquet"
    summary_df.to_parquet(summary_path)
    print(f"\n✅ All files saved → {output_dir}/")
    print(f"   Summary → {summary_path}")

    return {
        "scored_datasets": scored_datasets,
        "summary_df":      summary_df,
    }





if __name__ == "__main__":
    results = run_pipeline(
        data_paths={
            "gaia": Path("datasets/processed/gaia.parquet"),
            "aime": Path("datasets/processed/aime.parquet"),
            "mmlu_pro": Path("datasets/processed/mmlu_pro.parquet"),
            "swe_bench": Path("datasets/processed/swe_bench.parquet"),
            "musique": Path("datasets/processed/musique.parquet"),
        },
        output_dir=Path("data/processed"),
    )