"""
Normalize Hotpot / MuSiQue passage columns into agent runtime ``context``.

Agents expect::

    {"paragraphs": [{"title": str, "text": str}, ...]}

Parquet may store ``context`` as a nested dict/list or JSON string (after pipeline export).
"""

from __future__ import annotations

import json
from typing import Any, Mapping

from config.local.constants import DS


def datasets_requiring_episode_context() -> frozenset[str]:
    return DS.wiki_qa


def _coerce_raw(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            return value
    return value


def _paragraph(title: str, text: str) -> dict[str, str]:
    return {"title": (title or "").strip(), "text": (text or "").strip()}


def _as_list(value: Any) -> list[Any]:
    """Coerce list/tuple/ndarray-like values; avoid truth tests on arrays."""
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, (str, bytes, dict)):
        return [value]
    tolist = getattr(value, "tolist", None)
    if callable(tolist):
        out = tolist()
        return out if isinstance(out, list) else [out]
    return [value]


def _dict_list_field(raw: dict[str, Any], *keys: str) -> list[Any]:
    for key in keys:
        if key not in raw:
            continue
        seq = _as_list(raw[key])
        if seq:
            return seq
    return []


def _paragraphs_from_hotpot(raw: Any) -> list[dict[str, str]]:
    """HotpotQA ``context``: ``{title: [...], sentences: [[...], ...]}``."""
    if raw is None:
        return []

    if isinstance(raw, dict) and "paragraphs" in raw:
        return _paragraphs_from_list(raw["paragraphs"])

    if isinstance(raw, dict):
        titles = _dict_list_field(raw, "title", "titles")
        sents = _dict_list_field(raw, "sentences", "sentence")
        if titles and sents:
            out: list[dict[str, str]] = []
            for i, title in enumerate(titles):
                title_s = str(title or "").strip()
                block = _as_list(sents[i]) if i < len(sents) else []
                text = " ".join(str(x).strip() for x in block if str(x).strip())
                if text:
                    out.append(_paragraph(title_s, text))
            return out

    if isinstance(raw, list):
        out = []
        for item in raw:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                title = str(item[0] or "")
                body = item[1]
                if isinstance(body, list):
                    text = " ".join(str(x).strip() for x in body if str(x).strip())
                else:
                    text = str(body).strip()
                if text:
                    out.append(_paragraph(title, text))
        return out

    return []


def _paragraphs_from_musique(raw: Any) -> list[dict[str, str]]:
    """MuSiQue ``paragraphs``: list of dicts with ``paragraph_text`` / ``text``."""
    if raw is None:
        return []

    if isinstance(raw, dict) and "paragraphs" in raw:
        return _paragraphs_from_list(raw["paragraphs"])

    return _paragraphs_from_list(raw)


def _paragraphs_from_list(raw: Any) -> list[dict[str, str]]:
    items = raw if isinstance(raw, list) else _as_list(raw)
    if not items:
        return []

    out: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, dict):
            title = str(
                item.get("title")
                or item.get("doc_title")
                or item.get("document_title")
                or ""
            ).strip()
            text = str(
                item.get("text")
                or item.get("paragraph_text")
                or item.get("content")
                or ""
            ).strip()
            if text:
                out.append(_paragraph(title, text))
        elif isinstance(item, str) and item.strip():
            out.append(_paragraph("", item.strip()))
    return out


def normalize_episode_context(
    raw: Any,
    *,
    dataset: str,
) -> dict[str, Any] | None:
    """
    Convert a dataset-specific passage field to canonical agent ``context``.

    Returns ``None`` when no usable paragraphs are found.
    """
    ds = (dataset or "").strip().lower()
    value = _coerce_raw(raw)

    if value is None:
        return None

    if isinstance(value, dict) and "paragraphs" in value:
        paragraphs = _paragraphs_from_list(value["paragraphs"])
    elif ds == "hotpot":
        paragraphs = _paragraphs_from_hotpot(value)
    elif ds == "musique":
        paragraphs = _paragraphs_from_musique(value)
    else:
        paragraphs = _paragraphs_from_list(value)
        if not paragraphs and isinstance(value, dict):
            paragraphs = _paragraphs_from_hotpot(value)

    if not paragraphs:
        return None

    return {"paragraphs": paragraphs}


def episode_context_from_row(
    row: Mapping[str, Any],
    dataset: str,
) -> dict[str, Any] | None:
    """
    Extract runtime context from a parquet / checkpoint row.

    Only Hotpot and MuSiQue carry passage context; other datasets return ``None``.

    Tries, in order: ``context``, ``episode_context``, ``paragraphs``, ``passages``.
    """
    ds = (dataset or str(row.get("dataset_source") or row.get("dataset") or "")).strip().lower()
    if ds not in DS.wiki_qa:
        return None

    for key in ("context", "episode_context", "paragraphs", "passages"):
        if key not in row:
            continue
        raw = row.get(key)
        if raw is None:
            continue
        if key == "paragraphs":
            normalized = normalize_episode_context({"paragraphs": _coerce_raw(raw)}, dataset=ds)
        else:
            normalized = normalize_episode_context(_coerce_raw(raw), dataset=ds)
        if normalized:
            return normalized

    return None
