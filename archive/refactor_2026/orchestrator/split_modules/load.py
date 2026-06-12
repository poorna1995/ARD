"""Load eval frames and pre-routed artifacts for the orchestrator."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from agent.episode_context import datasets_requiring_episode_context
from config.paths import orchestrator_default_root
from config.settings import output_root
from prompts.prompts_core import DATASETS, TASK_DESCRIPTION
from routing.data.aliases import normalize_loader_frame
from routing.router import eval_base_frame, load_eval_parquet, load_split, resolve_dataset_name

DEFAULT_OUTPUT_ROOT = output_root(orchestrator_default_root())
QCE_SPLITS = ("train", "val", "test")
ROUTE_AGENT_COLS = ("router_pred", "assigned_agent")
EXECUTION_DONE_COLS = ("predicted_answer", "response_predicted_answer")
_EPISODE_CONTEXT_COLS = (
    "context",
    "paragraphs",
    "passages",
    "episode_context",
    "metadata",
)

def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported routes file type: {path.suffix} (use .parquet or .csv)")


def _routes_has_usable_episode_context(df: pd.DataFrame) -> bool:
    for col in _EPISODE_CONTEXT_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        if series.notna().any():
            return True
    return False


def _ensure_training_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "training_id" not in out.columns:
        if "id" in out.columns:
            out["training_id"] = out["id"].astype(str)
        else:
            raise KeyError("Routes/eval frame needs training_id or id")
    else:
        out["training_id"] = out["training_id"].astype(str)
    return out


def load_routes_frame(
    routes_path: str | Path,
    *,
    dataset: str | None = None,
    data_path: str | Path | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """
    Load a route-only artifact (``router_pred`` / proba columns) for execution.

    Merges eval rows from ``--dataset``, ``--data_path``, or ``--split`` when
    ``query`` is missing **or** when Hotpot/MuSiQue passage ``context`` is missing
    (required for ReAct ``retrieve``).
    """
    path = Path(routes_path)
    if not path.is_file():
        raise FileNotFoundError(f"Routes file not found: {path}")

    df = _ensure_training_id(_read_table(path))
    df = normalize_loader_frame(df)
    if not any(c in df.columns for c in ROUTE_AGENT_COLS):
        raise ValueError(f"Routes file must include one of {ROUTE_AGENT_COLS}: {path}")

    if "router_pred" not in df.columns:
        df["router_pred"] = df["assigned_agent"].astype(str)
    if "assigned_agent" not in df.columns:
        df["assigned_agent"] = df["router_pred"].astype(str)

    needs_query_merge = (
        "query" not in df.columns
        or df["query"].isna().all()
        or df["query"].astype(str).str.strip().eq("").all()
    )
    row_ds = str(df["dataset"].iloc[0]) if len(df) and "dataset" in df.columns else ""
    ds_key = resolve_dataset_name(dataset or row_ds)
    needs_context_merge = (
        ds_key in datasets_requiring_episode_context()
        and not _routes_has_usable_episode_context(df)
    )

    if needs_query_merge or needs_context_merge:
        if split is None and not dataset and data_path is None and needs_query_merge:
            raise ValueError(
                "Routes file has no query column; pass --dataset, --data_path, or --split to merge eval rows."
            )
        if needs_context_merge and split is None and not dataset and not data_path:
            raise ValueError(
                f"Routes for {ds_key!r} lack passage context; pass --dataset {ds_key} "
                "(or --data_path to eval parquet) when executing."
            )
        eval_df, _ = load_eval_frame(
            dataset=dataset or ds_key,
            data_path=data_path,
            split=split,
        )
        eval_df = _ensure_training_id(eval_df)
        key = "training_id"
        if key not in eval_df.columns:
            raise KeyError(f"Cannot merge routes with eval data without {key!r}")

        if needs_query_merge:
            merge_cols = [c for c in eval_df.columns if c not in df.columns or c == key]
        else:
            merge_cols = [key] + [
                c
                for c in eval_df.columns
                if c != key and (c not in df.columns or c in _EPISODE_CONTEXT_COLS)
            ]
        df = df.merge(eval_df[merge_cols], on=key, how="left", suffixes=("", "_eval"))
        if needs_query_merge and "query" not in df.columns:
            raise KeyError("Merged eval frame still missing query")

    if "expected_answer" not in df.columns:
        for alt in ("answer", "gold", "reference"):
            if alt in df.columns:
                df["expected_answer"] = df[alt]
                break

    return df


def load_eval_frame(
    *,
    dataset: str | None,
    data_path: str | Path | None,
    split: str | None,
) -> tuple[pd.DataFrame, str]:
    """
    Return (dataframe, feature_tag).

    ``feature_tag`` names on-disk QCE artifacts (e.g. gaia, test).
    """
    if split is not None:
        if split not in QCE_SPLITS:
            raise ValueError(f"--split must be one of {QCE_SPLITS}")
        df = load_split(split, with_embeddings=True)
        return df, split

    if not dataset:
        raise ValueError("Provide --dataset or --split")

    ds_key = resolve_dataset_name(dataset)
    if ds_key not in TASK_DESCRIPTION and ds_key not in DATASETS:
        known = sorted(set(DATASETS) | set(TASK_DESCRIPTION))
        raise ValueError(f"Unknown dataset {dataset!r}. Known keys: {known}")

    if data_path is not None:
        df = eval_base_frame(Path(data_path), dataset=ds_key)
        tag = Path(data_path).stem
    else:
        df = load_eval_parquet(ds_key)
        tag = dataset.strip().lower()
    return df, tag

