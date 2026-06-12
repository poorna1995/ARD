"""Column name aliases for loaders (``query_id`` / ``prob_*`` → canonical names)."""

from __future__ import annotations

import pandas as pd

from routing.config import AGENTS


def normalize_id_column(df: pd.DataFrame) -> pd.DataFrame:
    if "training_id" in df.columns:
        out = df.copy()
        out["training_id"] = out["training_id"].astype(str)
        return out
    for col in ("query_id", "id"):
        if col in df.columns:
            out = df.copy()
            out["training_id"] = out[col].astype(str)
            return out
    return df


def normalize_router_proba_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for agent in AGENTS:
        p_col = f"p_{agent}"
        prob_col = f"prob_{agent}"
        if p_col not in out.columns and prob_col in out.columns:
            out[p_col] = out[prob_col]
    return out


def normalize_loader_frame(df: pd.DataFrame) -> pd.DataFrame:
    return normalize_router_proba_columns(normalize_id_column(df))
