"""
Dataset-gated defaults for open-Wikipedia QA (hotpot, musique).

Runtime keys ``n_hops`` and ``hop_name`` are passed via ``run(..., **kwargs)``
from the benchmark row / checkpoint metadata — never injected into the user query.
"""

from __future__ import annotations

import json
from typing import Any, Mapping

RUNTIME_HOP_KEYS: frozenset[str] = frozenset({"n_hops", "hop_name"})

# Open-Wikipedia QA datasets (profiles + planner repair + observation fallback).
OPEN_WIKI_QA_DATASETS: frozenset[str] = frozenset({"hotpot", "musique"})

_WIKI_TOOLS = frozenset({"wikipedia_search", "wikipedia"})


def is_open_wiki_qa(dataset: str) -> bool:
    return (dataset or "").strip().lower() in OPEN_WIKI_QA_DATASETS


def coerce_positive_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        n = int(float(value))
        return n if n > 0 else None
    except (TypeError, ValueError):
        return None


def coerce_hop_name(value: Any) -> str | None:
    if value is None:
        return None
    s = str(value).strip().lower()
    return s or None


def extract_hop_metadata(kwargs: Mapping[str, Any] | None) -> tuple[int | None, str | None]:
    raw = kwargs or {}
    return coerce_positive_int(raw.get("n_hops")), coerce_hop_name(raw.get("hop_name"))


def promote_hop_metadata(
    kw: dict[str, Any],
    metadata: Mapping[str, Any] | None,
    *,
    overwrite: bool = False,
) -> None:
    """Copy ``n_hops`` / ``hop_name`` from a metadata dict onto ``kw`` (coerced)."""
    if not metadata:
        return
    nh = coerce_positive_int(metadata.get("n_hops"))
    if nh is not None and (overwrite or "n_hops" not in kw):
        kw["n_hops"] = nh
    hn = coerce_hop_name(metadata.get("hop_name"))
    if hn is not None and (overwrite or "hop_name" not in kw):
        kw["hop_name"] = hn


def agent_kwargs_from_row(row: Mapping[str, Any], **extra: Any) -> dict[str, Any]:
    """Build ``run_strategy`` kwargs from a parquet / checkpoint row."""
    out: dict[str, Any] = dict(extra)
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    promote_hop_metadata(out, row, overwrite=True)
    promote_hop_metadata(out, meta, overwrite=False)
    return out


def react_max_steps_for_dataset(
    dataset: str,
    n_hops: int | None = None,
) -> int | None:
    """
    ReAct step budget for open-Wikipedia QA.

    Each hop is typically search + read (2 steps), plus Finish and occasional
    invalid-Finish retries that still consume the outer step budget.
    """
    profiled = apply_react_profile(dataset, {}, n_hops=n_hops)
    steps = profiled.get("max_steps")
    return int(steps) if steps is not None else None


def apply_react_profile(
    dataset: str,
    agent_params: Mapping[str, Any],
    *,
    n_hops: int | None = None,
) -> dict[str, Any]:
    out = dict(agent_params)
    ds = (dataset or "").strip().lower()
    if ds not in OPEN_WIKI_QA_DATASETS:
        return out
    if ds == "musique":
        hops = n_hops or 4
        # ~2 steps/hop + Finish + retry headroom; floor 16 for 4-hop chains.
        out.setdefault("max_steps", max(16, 2 * hops + 6))
    elif ds == "hotpot":
        out.setdefault("max_steps", 10)
    return out


def apply_multiagent_profile(
    dataset: str,
    agent_params: Mapping[str, Any],
    *,
    n_hops: int | None = None,
) -> dict[str, Any]:
    out = dict(agent_params)
    ds = (dataset or "").strip().lower()
    if ds not in OPEN_WIKI_QA_DATASETS:
        return out
    if ds == "musique":
        hops = n_hops or 4
        out.setdefault("tool_max_steps", max(10, hops + 2))
        out.setdefault("max_subtasks", min(4, max(hops, 2)))
        out.setdefault("worker_result_forward_chars", 300)
    elif ds == "hotpot":
        out.setdefault("tool_max_steps", 7)
        out.setdefault("max_subtasks", 2)
    return out


def _wiki_tool_subtasks(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in subtasks:
        if not st.get("needs_tool"):
            continue
        if str(st.get("tool_name", "none")).strip().lower() in _WIKI_TOOLS:
            out.append(st)
    return out


def validate_planner_subtasks(
    subtasks: list[dict[str, Any]],
    *,
    dataset: str,
    n_hops: int | None = None,
    hop_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    When ``n_hops`` is set, repair linear Wikipedia chains (planner often emits parallel roots).
    Skips ``parallel`` topology; does not synthesize missing subtasks.
    """
    if not subtasks or n_hops is None:
        return subtasks

    ds = (dataset or "").strip().lower()
    if ds not in ("musique", "hotpot"):
        return subtasks

    topo = coerce_hop_name(hop_name) or ""
    if topo != "linear":
        return subtasks

    wiki = _wiki_tool_subtasks(subtasks)
    if len(wiki) < 2:
        return subtasks

    wiki[0]["depends_on"] = []
    for i in range(1, len(wiki)):
        wiki[i]["depends_on"] = [wiki[i - 1]["id"]]

    return subtasks


def observation_text_from_tools_results(
    tools_results: list[dict[str, Any]] | None,
) -> str:
    """Last non-finish tool observation; prefer Wikipedia JSON ``content``."""
    for tr in reversed(tools_results or []):
        if str(tr.get("action", "")).lower() == "finish":
            continue
        obs = str(tr.get("observation", "") or "").strip()
        if not obs:
            continue
        try:
            data = json.loads(obs)
        except json.JSONDecodeError:
            return obs[:4000]
        if not isinstance(data, dict):
            return obs[:4000]
        if data.get("error"):
            continue
        content = data.get("content")
        if content:
            return str(content)
        return obs[:2000]
    return ""
