"""Normalize corpus rows for QCE decomposition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from config.local.constants import DS, META_KEYS

DATASETS = DS.names
DS_ALIASES = DS.aliases


@dataclass(frozen=True)
class QueryIn:
    training_id: str | None
    dataset: str
    query_text: str
    metadata: dict[str, str | int | float] = field(default_factory=dict)


def _meta_value(value: Any) -> str | int | float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)) and float(value).is_integer():
        return int(value)
    s = str(value).strip()
    return s if s and s.lower() != "nan" else None


def resolve_ds(dataset: str) -> str:
    return DS.resolve(dataset)


def resolve_row(
    row: dict[str, Any],
    *,
    id_key: str = "training_id",
    query_key: str = "query",
    dataset_key: str = "dataset",
) -> QueryIn:
    training_id = row.get(id_key)
    query = row.get(query_key)
    dataset = row.get(dataset_key)
    if query is None or (isinstance(query, float) and query != query):
        raise ValueError(f"row missing {query_key!r} (training_id={training_id!r})")
    if not str(query).strip():
        raise ValueError(f"empty query (training_id={training_id!r})")

    metadata: dict[str, str | int | float] = {}
    for key in META_KEYS:
        if key not in row:
            continue
        cleaned = _meta_value(row[key])
        if cleaned is not None:
            metadata[key] = cleaned

    return QueryIn(
        training_id=str(training_id) if training_id is not None else None,
        dataset=resolve_ds(str(dataset)),
        query_text=str(query),
        metadata=metadata,
    )
