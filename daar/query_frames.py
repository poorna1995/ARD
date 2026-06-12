"""D-AAR query frames, oracle labels, and agent soft-target construction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config.global_config.paths import daar_frames_dir
from config.local.constants.agents import ROUTER_AGENTS
from src.utils.soft_labels import (
    SOFT_DOMINANT_COL,
    SOFT_SUM_COL,
    attach_utility_softmax_labels,
    soft_probability_columns,
)

P0_MANIFEST = "solvability_manifest.json"
P1_MANIFEST = "solvability_p1_manifest.json"
P1_MODEL_STEM = "solvability_p1_soft_kl"
AGENT_RANK_MANIFEST = "solvability_gate1b_manifest.json"
AGENT_RANK_MODEL_STEM = "solvability_gate1b_soft_kl"
SOLVABLE_LABEL = "solvable"
AGENT_SCORE_PREFIX = "s_hat"


def split_training_ids(*, frames_dir: Path | None = None) -> dict[str, set[str]]:
    """Query-level training_id sets per daar split."""
    out: dict[str, set[str]] = {}
    for split in ("train", "val", "test"):
        path = (frames_dir or daar_frames_dir()) / f"daar_{split}_frame.parquet"
        if not path.is_file():
            continue
        df = pd.read_parquet(path, columns=["training_id"])
        out[split] = set(df["training_id"].astype(str).unique())
    return out


def assert_disjoint_splits(*, frames_dir: Path | None = None) -> dict[str, int]:
    """Require train / val / test query ids to be disjoint (no leakage)."""
    ids = split_training_ids(frames_dir=frames_dir)
    required = ("train", "val")
    missing = [s for s in required if s not in ids]
    if missing:
        raise FileNotFoundError(
            f"missing daar frames for splits {missing} — build train/val frames first"
        )
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if left not in ids or right not in ids:
            continue
        overlap = ids[left] & ids[right]
        if overlap:
            sample = sorted(overlap)[:5]
            raise ValueError(
                f"split leakage: {left} ∩ {right} has {len(overlap)} ids (e.g. {sample})"
            )
    return {split: len(id_set) for split, id_set in ids.items()}


def load_query_frame(split: str, *, frames_dir: Path | None = None) -> pd.DataFrame:
    frames_dir = frames_dir or daar_frames_dir()
    path = frames_dir / f"daar_{split}_frame.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {path} — run: uv run python scripts/build_daar_train_frame.py"
        )
    df = pd.read_parquet(path)
    df["training_id"] = df["training_id"].astype(str)
    return df


def query_level_frame(frame_long: pd.DataFrame) -> pd.DataFrame:
    return frame_long.drop_duplicates("training_id").copy()


def oracle_from_frame(frame_long: pd.DataFrame) -> pd.DataFrame:
    """Long oracle compatible with ``attach_utility_softmax_labels`` (r_a → perf)."""
    oracle = frame_long[["training_id", "agent", "r_a", "cost_usd"]].copy()
    oracle["is_correct"] = pd.to_numeric(oracle["r_a"], errors="coerce").fillna(0).astype(int)
    oracle["is_failed"] = 0
    oracle["predicted_answer"] = np.where(oracle["is_correct"] > 0, "1", "")
    return oracle


def attach_soft_labels(
    frame_long: pd.DataFrame,
    *,
    utility_lambda: float,
    softmax_temperature: float = 1.0,
) -> pd.DataFrame:
    labels = query_level_frame(frame_long)
    oracle = oracle_from_frame(frame_long)
    return attach_utility_softmax_labels(
        labels,
        oracle,
        utility_lambda=utility_lambda,
        softmax_temperature=softmax_temperature,
    )


def success_only_distribution(r_by_agent: dict[str, int]) -> dict[str, float]:
    """Uniform soft mass over successful agents only."""
    correct = [a for a in ROUTER_AGENTS if int(r_by_agent.get(a, 0)) > 0]
    probs = {a: 0.0 for a in ROUTER_AGENTS}
    if correct:
        weight = 1.0 / len(correct)
        for agent in correct:
            probs[agent] = weight
    return probs


def attach_success_only_soft_labels(frame_long: pd.DataFrame) -> pd.DataFrame:
    """Agent-rank targets: uniform mass on ``{a : r_a = 1}`` only."""
    labels = query_level_frame(frame_long).copy()
    labels["training_id"] = labels["training_id"].astype(str)
    soft_cols = soft_probability_columns(ROUTER_AGENTS)

    r_wide = frame_long.pivot_table(index="training_id", columns="agent", values="r_a", aggfunc="first")
    rows: list[dict[str, float]] = []
    for tid in labels["training_id"]:
        if tid not in r_wide.index:
            rows.append({col: 0.0 for col in soft_cols})
            continue
        r_row = r_wide.loc[tid]
        dist = success_only_distribution(
            {str(agent): int(pd.to_numeric(r_row.get(agent, 0), errors="coerce") or 0) for agent in ROUTER_AGENTS}
        )
        rows.append({f"p_{agent}": dist[agent] for agent in ROUTER_AGENTS})

    soft = pd.DataFrame(rows, columns=soft_cols)
    soft[SOFT_SUM_COL] = soft[soft_cols].sum(axis=1)
    dominant = pd.Series(pd.NA, index=soft.index, dtype="object")
    has_mass = soft[SOFT_SUM_COL] > 0
    if has_mass.any():
        dominant.loc[has_mass] = (
            soft.loc[has_mass, soft_cols].idxmax(axis=1).str.removeprefix("p_")
        )
    soft[SOFT_DOMINANT_COL] = dominant

    drop_cols = [c for c in (*soft_cols, SOFT_SUM_COL, SOFT_DOMINANT_COL) if c in labels.columns]
    out = labels.drop(columns=drop_cols, errors="ignore")
    out = pd.concat([out.reset_index(drop=True), soft.reset_index(drop=True)], axis=1)
    out[SOLVABLE_LABEL] = out["training_id"].map(pool_solvable_series(frame_long)).astype(int)
    return out


def pool_solvable_series(frame_long: pd.DataFrame) -> pd.Series:
    return (
        frame_long.groupby("training_id", sort=False)[SOLVABLE_LABEL]
        .first()
        .astype(int)
        .rename(SOLVABLE_LABEL)
    )


def query_labels_with_solvable(frame_long: pd.DataFrame) -> pd.DataFrame:
    """Query-level feature rows with binary pool-solvability label."""
    labels = query_level_frame(frame_long).drop(columns=[SOLVABLE_LABEL], errors="ignore")
    labels = labels.copy()
    labels["training_id"] = labels["training_id"].astype(str)
    labels[SOLVABLE_LABEL] = labels["training_id"].map(pool_solvable_series(frame_long)).astype(int)
    return labels


def agent_soft_columns() -> list[str]:
    return soft_probability_columns(ROUTER_AGENTS)


# Backward-compatible aliases (deprecated internal names).
load_frame = load_query_frame
solvable_series = pool_solvable_series
soft_cols = agent_soft_columns
GATE1B_MANIFEST = AGENT_RANK_MANIFEST
GATE1B_MODEL_STEM = AGENT_RANK_MODEL_STEM
SOLVABLE_TARGET = SOLVABLE_LABEL
SCORE_PREFIX = AGENT_SCORE_PREFIX
