"""Equal-count stratified sampling across discrete strata."""

from __future__ import annotations

from typing import Hashable, Sequence, Union

import pandas as pd

StratumCols = Union[str, Sequence[str]]


def _stratum_series(df: pd.DataFrame, cols: StratumCols) -> pd.Series:
    if isinstance(cols, str):
        return df[cols].astype(str)
    parts = [df[c].astype(str) for c in cols]
    out = parts[0]
    for p in parts[1:]:
        out = out + "\0" + p
    return out


def _eligible_pool(
    work: pd.DataFrame,
    pool: pd.Index,
    *,
    exclude_index: set[Hashable],
    unique_answer_col: str | None,
    used_answers: set[str],
) -> pd.Index:
    mask = ~pool.isin(exclude_index)
    if unique_answer_col is not None:
        answers = work.loc[pool, unique_answer_col].astype(str)
        mask = mask & ~answers.isin(used_answers)
    return pool[mask]


def _greedy_pick(
    work: pd.DataFrame,
    eligible: pd.Index,
    want: int,
    *,
    unique_answer_col: str,
    used_answers: set[str],
    seed: int,
) -> list[Hashable]:
    """Pick up to ``want`` rows; at most one row per distinct answer value."""
    if want <= 0 or len(eligible) == 0:
        return []
    order = work.loc[eligible].sample(frac=1, random_state=seed).index
    chosen: list[Hashable] = []
    for idx in order:
        if len(chosen) >= want:
            break
        ans = str(work.at[idx, unique_answer_col])
        if ans not in used_answers:
            chosen.append(idx)
            used_answers.add(ans)
    return chosen


def stratified_sample(
    df: pd.DataFrame,
    stratum_cols: StratumCols,
    n: int,
    *,
    seed: int = 42,
    shuffle: bool = True,
    exclude_index: set[Hashable] | None = None,
    unique_answer_col: str | None = None,
    used_answers: set[str] | None = None,
) -> pd.DataFrame:
    """
    Sample exactly ``n`` rows with (as equal as possible) counts per stratum.

    Strata with fewer rows than their quota contribute all available rows;
    any shortfall is filled from the remaining pool (deterministic, seed-based).

    If ``unique_answer_col`` is set, only rows whose answer is not already in
    ``used_answers`` are eligible; picked answers are added to ``used_answers``.
    """
    if n <= 0:
        raise ValueError("n must be positive")

    exclude_index = set(exclude_index or ())
    used_answers = used_answers if used_answers is not None else set()

    work = df.copy()
    work["_stratum"] = _stratum_series(work, stratum_cols)

    keys = sorted(work["_stratum"].unique())
    n_groups = len(keys)
    base, extra = divmod(n, n_groups)
    targets = {k: base + (1 if i < extra else 0) for i, k in enumerate(keys)}

    picked_idx: list[Hashable] = []
    for i, key in enumerate(keys):
        pool = work.index[work["_stratum"] == key]
        eligible = _eligible_pool(
            work,
            pool,
            exclude_index=exclude_index | set(picked_idx),
            unique_answer_col=unique_answer_col,
            used_answers=used_answers,
        )
        want = targets[key]
        if want <= 0:
            continue
        if unique_answer_col is not None:
            picked_idx.extend(
                _greedy_pick(
                    work,
                    eligible,
                    want,
                    unique_answer_col=unique_answer_col,
                    used_answers=used_answers,
                    seed=seed + i,
                )
            )
        elif len(eligible):
            take = min(want, len(eligible))
            picked_idx.extend(
                work.loc[eligible]
                .sample(n=take, random_state=seed + i)
                .index.tolist()
            )

    need = n - len(picked_idx)
    if need:
        rest = _eligible_pool(
            work,
            work.index,
            exclude_index=exclude_index | set(picked_idx),
            unique_answer_col=unique_answer_col,
            used_answers=used_answers,
        )
        if unique_answer_col is not None:
            picked_idx.extend(
                _greedy_pick(
                    work,
                    rest,
                    need,
                    unique_answer_col=unique_answer_col,
                    used_answers=used_answers,
                    seed=seed + n_groups,
                )
            )
            need = n - len(picked_idx)
        if need:
            rest = _eligible_pool(
                work,
                work.index,
                exclude_index=exclude_index | set(picked_idx),
                unique_answer_col=None,
                used_answers=used_answers,
            )
            if len(rest) < need:
                raise ValueError(
                    f"stratified shortfall: need {need} more rows but only {len(rest)} eligible"
                )
            picked_idx.extend(
                work.loc[rest]
                .sample(n=need, random_state=seed + n_groups + 1)
                .index.tolist()
            )

    out = work.loc[picked_idx].drop(columns=["_stratum"])
    if shuffle:
        out = out.sample(frac=1, random_state=seed).reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out
