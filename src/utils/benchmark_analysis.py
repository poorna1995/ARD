"""Pre-training / pre-benchmark breakdown for stratified subsamples."""

from __future__ import annotations

import pandas as pd

try:
    from IPython.display import display
except ImportError:

    def display(obj):  # type: ignore[misc]
        print(obj)


def _group_keys(key, n_cols: int) -> tuple:
    """Normalize pandas groupby key to a tuple of length ``n_cols``."""
    if isinstance(key, tuple):
        return key if len(key) == n_cols else (key,)
    return (key,) if n_cols == 1 else (key,)


def analyze_benchmark_sample(
    name: str,
    sample: pd.DataFrame,
    full: pd.DataFrame,
    stratum_cols: list[str],
    *,
    role: str = "train",
    pool_label: str = "processed pool",
    answer_unique_note: str | None = None,
) -> None:
    """Print breakdown for a train or eval benchmark subsample."""
    sep = "=" * 72
    print(sep)
    print(f"{name.upper()} {role} sample")
    print(sep)
    print(
        f"rows: {len(sample):,}  |  full {pool_label}: {len(full):,}  "
        f"|  coverage: {100 * len(sample) / len(full):.2f}%"
    )
    print(f"columns: {list(sample.columns)}")
    if "split" in sample.columns:
        print(f"seed: 42  |  split column: {sample['split'].unique().tolist()}")
    else:
        print("seed: 42")

    print("\n--- null / empty checks ---")
    nulls = sample.isna().sum()
    print(nulls[nulls > 0].to_string() if nulls.any() else "  no nulls")
    empty_q = (
        int((sample["query"].str.strip() == "").sum()) if "query" in sample.columns else 0
    )
    empty_a = int((sample["answer"].astype(str).str.strip() == "").sum())
    dup_id = int(sample["id"].duplicated().sum())
    print(f"  empty query: {empty_q}  |  empty answer: {empty_a}  |  duplicate id: {dup_id}")

    print("\n--- text length (chars) ---")
    qlen = sample["query"].str.len()
    alen = sample["answer"].astype(str).str.len()
    print(
        f"  query  — min={qlen.min()}, median={qlen.median():.0f}, "
        f"mean={qlen.mean():.0f}, max={qlen.max()}"
    )
    print(
        f"  answer — min={alen.min()}, median={alen.median():.0f}, "
        f"mean={alen.mean():.0f}, max={alen.max()}"
    )

    print(f"\n--- stratification ({' × '.join(stratum_cols)}) ---")
    if len(stratum_cols) == 1:
        col = stratum_cols[0]
        ct = sample[col].value_counts().sort_index().to_frame("n")
        ct.loc["All"] = len(sample)
    else:
        ct = pd.crosstab(*[sample[c] for c in stratum_cols], margins=True)
    display(ct)

    print("\n--- per-stratum counts (sample vs full pool %) ---")
    rows = []
    for key, g in sample.groupby(stratum_cols, observed=True):
        parts = _group_keys(key, len(stratum_cols))
        mask = pd.Series(True, index=full.index)
        for col, val in zip(stratum_cols, parts):
            mask &= full[col] == val
        n_full = int(mask.sum())
        rows.append(
            {
                **dict(zip(stratum_cols, parts)),
                "n_sample": len(g),
                "n_full": n_full,
                "pct_of_full": round(100 * len(g) / n_full, 3) if n_full else 0,
            }
        )
    display(pd.DataFrame(rows).sort_values(stratum_cols))

    if answer_unique_note or "answer" in sample.columns:
        n_uniq = sample["answer"].nunique()
        print("\n--- answers ---")
        print(f"  unique answers: {n_uniq} / {len(sample)} ({100 * n_uniq / len(sample):.1f}%)")
        if answer_unique_note:
            print(f"  note: {answer_unique_note}")
        vc = sample["answer"].value_counts()
        repeats = vc[vc > 1]
        if len(repeats):
            extra = int((repeats - 1).sum())
            print(f"  answers appearing >1×: {len(repeats)}  |  extra repeat rows: {extra}")
            print("  top repeated:")
            display(repeats.head(8).to_frame("count"))

    if "solution_length" in sample.columns:
        print("\n--- solution_length by level (complexity proxy) ---")
        display(
            sample.groupby("level", observed=True)["solution_length"]
            .agg(["count", "mean", "median", "min", "max"])
            .round(0)
        )

    print()
