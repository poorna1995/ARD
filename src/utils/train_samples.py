"""Benchmark subsamples for train (500) and eval (200 / full GAIA)."""

from __future__ import annotations

import pandas as pd

from .stratified_sample import stratified_sample

MATH_LEVELS = [f"Level {i}" for i in range(1, 6)]
MATH_SUBJECTS = [
    "Algebra",
    "Counting & Probability",
    "Geometry",
    "Intermediate Algebra",
    "Number Theory",
    "Prealgebra",
    "Precalculus",
]

MUSIQUE_STRATA = [
    (2, "linear"),
    (3, "linear"),
    (3, "parallel"),
    (4, "linear"),
    (4, "parallel"),
    (4, "branching"),
]

HOTPOT_LEVELS = ["easy", "medium", "hard"]
HOTPOT_TYPES = ["bridge", "comparison"]

MUSIQUE_UNIQUE_ANSWER_FRACTION = 330 / 500


def _musique_stratum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["stratum"] = df["n_hops"].astype(str) + "hop_" + df["hop_name"].astype(str)
    allowed = {f"{h}hop_{name}" for h, name in MUSIQUE_STRATA}
    missing = allowed - set(df["stratum"].unique())
    if missing:
        raise ValueError(f"musique train missing strata: {sorted(missing)}")
    return df[df["stratum"].isin(allowed)]


def _swap_to_unique_answers(
    sample: pd.DataFrame,
    pool: pd.DataFrame,
    target_unique: int,
    seed: int,
) -> pd.DataFrame:
    work = sample.reset_index(drop=True).copy()
    max_tries = len(work) * 20
    tries = 0

    while work["answer"].nunique() < target_unique and tries < max_tries:
        tries += 1
        counts = work["answer"].value_counts()
        dup_answers = counts[counts > 1].index.tolist()
        if not dup_answers:
            break

        dup_rows = work[work["answer"].isin(dup_answers)]
        row_idx = dup_rows.sample(1, random_state=seed + tries).index[0]
        row = work.loc[row_idx]
        used = set(work["answer"].astype(str))
        candidates = pool[
            (pool["stratum"] == row["stratum"])
            & (~pool["id"].isin(work["id"]))
            & (~pool["answer"].astype(str).isin(used))
        ]
        if candidates.empty:
            continue
        replacement = candidates.sample(1, random_state=seed + tries + 1).iloc[0]
        work.loc[row_idx] = replacement

    return work


def _append_math_training_id(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["training_id"] = [f"math_{i}" for i in range(len(out))]
    return out


def _append_hotpot_training_meta(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["dataset_source"] = "hotpotqa"
    out["training_id"] = [f"hotpot_{i}" for i in range(len(out))]
    return out


def _append_musique_training_meta(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()
    out["dataset_source"] = "musique"
    out["training_id"] = [f"musique_{i}" for i in range(len(out))]
    return out


def sample_math(
    df: pd.DataFrame,
    n: int = 500,
    seed: int = 42,
    *,
    add_training_id: bool = False,
) -> pd.DataFrame:
    """
    Stratified sample across ``level`` × ``type`` (subject).

    Optional: ``training_id`` = ``math_0`` … (additive).
    """
    work = df[df["level"].isin(MATH_LEVELS) & df["type"].isin(MATH_SUBJECTS)].copy()
    out = stratified_sample(work, ["level", "type"], n, seed=seed)
    if add_training_id:
        out = _append_math_training_id(out)
    return out


def sample_hotpot(
    df: pd.DataFrame,
    n: int = 500,
    seed: int = 42,
    *,
    add_training_metadata: bool = False,
) -> pd.DataFrame:
    """
    Stratified sample across ``level`` × ``type`` (bridge / comparison).

    Optional: ``dataset_source`` = ``hotpotqa``, ``training_id`` = ``hotpot_0`` …
    """
    work = df[df["level"].isin(HOTPOT_LEVELS) & df["type"].isin(HOTPOT_TYPES)].copy()
    out = stratified_sample(work, ["level", "type"], n, seed=seed)
    if add_training_metadata:
        out = _append_hotpot_training_meta(out)
    return out


def sample_musique(
    df: pd.DataFrame,
    n: int = 500,
    seed: int = 42,
    *,
    unique_answer_fraction: float = MUSIQUE_UNIQUE_ANSWER_FRACTION,
    add_training_metadata: bool = False,
) -> pd.DataFrame:
    """
    Stratified sample across 6 hop patterns; same-stratum swaps toward
    ``unique_answer_fraction`` distinct answers (~66% for n=500).

    Optional: ``dataset_source`` = ``musique``, ``training_id`` = ``musique_0`` …
    """
    if not 0.0 <= unique_answer_fraction <= 1.0:
        raise ValueError("unique_answer_fraction must be in [0, 1]")

    pool = _musique_stratum(df)
    target_unique = int(round(n * unique_answer_fraction))

    sample = stratified_sample(pool, "stratum", n, seed=seed, shuffle=False)
    sample = _swap_to_unique_answers(sample, pool, target_unique, seed)

    out = sample.drop(columns=["stratum"], errors="ignore").sample(
        frac=1, random_state=seed
    ).reset_index(drop=True)
    if add_training_metadata:
        out = _append_musique_training_meta(out)
    return out


# Back-compat alias (same as ``sample_musique``)
def sample_musique_fine_strata(
    df: pd.DataFrame,
    n: int = 500,
    seed: int = 42,
    *,
    unique_answer_fraction: float = MUSIQUE_UNIQUE_ANSWER_FRACTION,
) -> pd.DataFrame:
    return sample_musique(
        df, n, seed, unique_answer_fraction=unique_answer_fraction, add_training_metadata=False
    )


def sample_mmlu(df: pd.DataFrame, n: int = 200, seed: int = 42) -> pd.DataFrame:
    if "category" not in df.columns:
        raise ValueError("mmlu sample requires 'category' column")
    if len(df) < n:
        raise ValueError(f"cannot sample n={n} from {len(df)} rows")
    return stratified_sample(df, "category", n, seed=seed)


def sample_gaia(df: pd.DataFrame, n: int | None = None, seed: int = 42) -> pd.DataFrame:
    if n is None or n >= len(df):
        out = df.copy()
        return out.sample(frac=1, random_state=seed).reset_index(drop=True)
    if "level" in df.columns:
        return stratified_sample(df, "level", n, seed=seed)
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


def combine_training_samples(
    math_df: pd.DataFrame,
    hotpot_df: pd.DataFrame,
    musique_df: pd.DataFrame,
    *,
    expected_total: int | None = 1500,
) -> pd.DataFrame:
    """
    Stack MATH → HotpotQA → MuSiQue; ``dataset_source`` on MATH if missing;
    copy prior ``training_id`` to ``dataset_training_id``; assign global
    ``training_id`` = ``train_0000`` … ``train_{n-1}``.
    """
    parts: list[pd.DataFrame] = []
    for raw, fallback_src in (
        (math_df.copy(), "math"),
        (hotpot_df.copy(), "hotpotqa"),
        (musique_df.copy(), "musique"),
    ):
        df = raw
        if "dataset_source" not in df.columns:
            df["dataset_source"] = fallback_src
        if "training_id" in df.columns:
            df["dataset_training_id"] = df["training_id"].astype(str)
        parts.append(df)

    out = pd.concat(parts, ignore_index=True, sort=False)
    out["training_id"] = [f"train_{i:04d}" for i in range(len(out))]

    if expected_total is not None and len(out) != expected_total:
        raise ValueError(
            f"combined training rows {len(out)} != expected {expected_total}"
        )
    return out


TRAIN_N = 500
EVAL_N = 200
