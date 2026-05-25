from __future__ import annotations

import json
import re
from typing import Any, Mapping

RUNTIME_HOP_KEYS: frozenset[str] = frozenset({"n_hops", "hop_name"})

# Multi-hop open QA datasets (profiles + planner repair + observation fallback).
OPEN_WIKI_QA_DATASETS: frozenset[str] = frozenset({"hotpot", "musique"})

# Hotpot / MuSiQue linear-hop repair only (not GAIA — GAIA uses files, code, mixed tools).
_WEB_LOOKUP_TOOLS = frozenset({"wikipedia_search", "wikipedia", "web_search", "web_fetch"})
_RETRIEVE_CHAIN_TOOLS = frozenset({"retrieve"}) | _WEB_LOOKUP_TOOLS


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


def agent_kwargs_from_row(
    row: Mapping[str, Any],
    *,
    dataset: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Build ``run_strategy`` / ``Router`` kwargs from a parquet / checkpoint row."""
    out: dict[str, Any] = dict(extra)
    meta = row.get("metadata")
    if not isinstance(meta, dict):
        meta = {}
    promote_hop_metadata(out, row, overwrite=True)
    promote_hop_metadata(out, meta, overwrite=False)

    # Per-row source wins (required for combined_raw: math / hotpot / musique stacked).
    ds = str(
        row.get("dataset_source")
        or row.get("dataset")
        or dataset
        or out.get("dataset")
        or ""
    ).strip().lower()
    if ds:
        from agent.episode_context import episode_context_from_row

        ctx = episode_context_from_row(row, ds)
        if ctx:
            out["context"] = ctx

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
    if ds == "gaia":
        out.setdefault("max_steps", 14)
        out.setdefault("max_format_retries", 5)
        return out
    if ds not in OPEN_WIKI_QA_DATASETS:
        return out
    if ds == "musique":
        hops = n_hops or 4
        out.setdefault("max_steps", max(10, hops + 2))
        out.setdefault("max_format_retries", 5)
    elif ds == "hotpot":
        hops = n_hops or 2
        out.setdefault("max_steps", max(7, hops + 2))
        out.setdefault("max_format_retries", 5)
    return out


_MULTIAGENT_FACT_CHARS: dict[str, int] = {
    "math": 80,
    "hotpot": 160,
    "musique": 180,
    "gaia": 220,
    "mmlu_pro": 80,
}


def multiagent_fact_chars(dataset: str) -> int:
    ds = (dataset or "").strip().lower()
    return _MULTIAGENT_FACT_CHARS.get(ds, 160)


def apply_multiagent_profile(
    dataset: str,
    agent_params: Mapping[str, Any],
    *,
    n_hops: int | None = None,
) -> dict[str, Any]:
    out = dict(agent_params)
    ds = (dataset or "").strip().lower()
    fact_cap = multiagent_fact_chars(ds)
    out.setdefault("worker_result_forward_chars", fact_cap)
    out.setdefault("memory_fact_chars", fact_cap)
    if ds == "math":
        out.setdefault("max_subtasks", 3)
        out.setdefault("tool_max_steps", 8)
    elif ds == "gaia":
        out.setdefault("max_subtasks", 6)
        out.setdefault("tool_max_steps", 14)
    elif ds == "mmlu_pro":
        out.setdefault("max_subtasks", 1)
    elif ds == "musique":
        hops = n_hops or 4
        out.setdefault("tool_max_steps", max(10, hops + 2))
        out.setdefault("max_subtasks", min(4, max(hops, 2)))
    elif ds == "hotpot":
        out.setdefault("tool_max_steps", 7)
        out.setdefault("max_subtasks", 2)
    return out


def _search_tool_subtasks(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in subtasks:
        if not st.get("needs_tool"):
            continue
        if str(st.get("tool_name", "none")).strip().lower() in _WEB_LOOKUP_TOOLS:
            out.append(st)
    return out


def _valid_subtask_ids(subtasks: list[dict[str, Any]]) -> frozenset[str]:
    return frozenset(
        str(st.get("id", "")).strip()
        for st in subtasks
        if str(st.get("id", "")).strip()
    )


# Planner placeholders: "from s1" (multiagent) or "from t1" (QCE).
_ST_ID = r"([st]\d+)"
_DEP_INFER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(rf"<\s*[^>]*?\bfrom\s+{_ST_ID}\b", re.IGNORECASE),
    re.compile(rf"\bfrom\s+{_ST_ID}\b", re.IGNORECASE),
    re.compile(rf"\busing\s+{_ST_ID}\b", re.IGNORECASE),
    re.compile(rf"\bafter\s+{_ST_ID}\b", re.IGNORECASE),
    re.compile(rf"\bresult\s+from\s+{_ST_ID}\b", re.IGNORECASE),
    re.compile(rf"\bdepends_on[\"'\s:\[]*{_ST_ID}\b", re.IGNORECASE),
)


def _mentioned_dep_ids(text: str, self_id: str, valid: frozenset[str]) -> list[str]:
    if not text or not valid:
        return []
    found: list[str] = []
    seen: set[str] = set()
    for pat in _DEP_INFER_PATTERNS:
        for m in pat.finditer(text):
            sid = str(m.group(1)).strip().lower()
            if sid in valid and sid != self_id and sid not in seen:
                seen.add(sid)
                found.append(sid)
    return found


def _merge_depends_on(st: dict[str, Any], new_deps: list[str]) -> None:
    existing = [
        str(d).strip()
        for d in (st.get("depends_on") or [])
        if str(d).strip()
    ]
    seen = set(existing)
    for dep in new_deps:
        if dep not in seen:
            existing.append(dep)
            seen.add(dep)
    st["depends_on"] = existing


_RELATIONAL_DEP_MARKERS: tuple[re.Pattern[str], ...] = (
    re.compile(
        r"\bthat (country|city|county|state|university|college|school|region|"
        r"place|nation|entity|alma mater)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bthe (identified|same|above|previous|resulting|bordering)\b", re.IGNORECASE),
    re.compile(
        r"\bidentified (country|city|county|university|alma mater|entity|nation)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bnamed after (what|whom|who)\b", re.IGNORECASE),
    re.compile(r"\bwhat (is )?the bordering county named after\b", re.IGNORECASE),
    re.compile(r"\bextract the (year|date|name|answer)\b", re.IGNORECASE),
    re.compile(r"\bfrom the retrieved\b", re.IGNORECASE),
    re.compile(r"\busing the (result|information|fact|identified)\b", re.IGNORECASE),
    re.compile(r"\bwho represented\b", re.IGNORECASE),
    re.compile(r"\b(current )?(minister|defence|defense|defense minister)\b", re.IGNORECASE),
    re.compile(r"\b(bordering county|that county|that city|that country)\b", re.IGNORECASE),
    re.compile(r"\b(of|in|for) the (identified|country|city|county|university)\b", re.IGNORECASE),
    re.compile(r"\bfind out who represented\b", re.IGNORECASE),
    re.compile(r"\bdetermine what the (bordering )?county is named after\b", re.IGNORECASE),
    re.compile(r"\bwhere (the|that|this)\b", re.IGNORECASE),
    re.compile(r"\b(the country|the city|the county|the university)\b", re.IGNORECASE),
)


def _needs_relational_dep(text: str) -> bool:
    return any(p.search(text or "") for p in _RELATIONAL_DEP_MARKERS)


def _propagate_semantic_dependencies(subtasks: list[dict[str, Any]]) -> None:
    """Link subtasks with relational language to the immediately prior subtask."""
    for i in range(1, len(subtasks)):
        st = subtasks[i]
        blob = f"{st.get('goal', '')} {st.get('search_query', '')}"
        if not _needs_relational_dep(blob):
            continue
        prior_id = str(subtasks[i - 1].get("id", "")).strip()
        if prior_id:
            _merge_depends_on(st, [prior_id])


def _retrieve_chain_subtasks(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for st in subtasks:
        if not st.get("needs_tool"):
            continue
        tool = str(st.get("tool_name", "none")).strip().lower()
        if tool in _RETRIEVE_CHAIN_TOOLS:
            out.append(st)
    return out


_MONTHS = (
    "january|february|march|april|may|june|july|august|"
    "september|october|november|december"
)


def query_relation_type(query: str) -> str | None:
    q = (query or "").strip().lower()
    if re.search(r"^(when|since when)\b", q) or re.search(r"\bwhen did\b", q):
        return "when"
    if re.search(r"^where\b", q) or re.search(r"\bwhat (city|place|country|portion)\b", q):
        return "where"
    if re.search(r"^in which country\b", q) or re.search(r"\bwhich country\b", q):
        return "country"
    if re.search(r"^who\b", q) or re.search(r"\bwho (is|was|represented|fathered)\b", q):
        return "who"
    if re.search(r"\bnamed after (what|whom|who)\b", q):
        return "what"
    return None


def subtask_relation_type(goal: str) -> str | None:
    g = (goal or "").strip().lower()
    if re.search(r"\b(when|date|year|era|since|died on|death date)\b", g):
        return "when"
    if re.search(r"\b(where|birthplace|city|country|place|located|live in)\b", g):
        return "where"
    if re.search(r"\bwhich country\b", g) or re.search(r"\bin which country\b", g):
        return "country"
    if re.search(r"\b(who|father|child|author|creator|represented)\b", g):
        return "who"
    if re.search(r"\bnamed after\b", g):
        return "what"
    return None


def looks_temporal(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    if re.search(r"\b(since|era|century|year|date|month)\b", t):
        return True
    if re.search(rf"\b(?:{_MONTHS})\b", t):
        return True
    if re.search(r"\b(?:19|20)\d{2}\b", t):
        return True
    if re.search(rf"\b\d{{1,2}}\s+(?:{_MONTHS})\s+\d{{4}}\b", t):
        return True
    return False


def looks_country_span(text: str) -> bool:
    t = (text or "").strip()
    if not t or looks_temporal(t):
        return False
    if "," in t:
        return True
    words = t.split()
    return 1 <= len(words) <= 3 and t[0].isupper()


_INNER_FIRST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bcountry where\b", re.IGNORECASE),
    re.compile(r"\bcity where\b", re.IGNORECASE),
    re.compile(r"\bplace where\b", re.IGNORECASE),
    re.compile(r"\bwhere .+ (died|born|located|live)\b", re.IGNORECASE),
    re.compile(r"\bwhere most\b", re.IGNORECASE),
    re.compile(r"\bnamed for\b", re.IGNORECASE),
    re.compile(r"\balong with\b", re.IGNORECASE),
    re.compile(r"\bthat .+ is in\b", re.IGNORECASE),
)

_OUTER_LAST_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^in which country\b", re.IGNORECASE),
    re.compile(r"^where do some\b", re.IGNORECASE),
    re.compile(r"^when did\b", re.IGNORECASE),
    re.compile(r"\binternational college\b", re.IGNORECASE),
    re.compile(r"\bbirthplace of\b", re.IGNORECASE),
    re.compile(r"\blarge art school\b", re.IGNORECASE),
    re.compile(r"\bfather of\b", re.IGNORECASE),
    re.compile(r"\bnetwork that first aired\b", re.IGNORECASE),
)


def _subtask_order_key(st: dict[str, Any]) -> tuple[int, int, int]:
    g = str(st.get("goal", "")).lower()
    outer = sum(1 for p in _OUTER_LAST_PATTERNS if p.search(g))
    inner = sum(1 for p in _INNER_FIRST_PATTERNS if p.search(g))
    if re.search(r"\b(previous step|subtask s|identified in|found in subtask)\b", g):
        outer += 1
    return (outer, -inner, len(g))


def _reorder_inside_out(subtasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Inner/nested clauses before outer/final-clause subtasks."""
    if len(subtasks) < 2:
        return subtasks
    for i, st in enumerate(subtasks):
        if i == 0:
            continue
        g = str(st.get("goal", "")).lower()
        if re.search(rf"\b(subtask s{i}|from s{i}|step s{i})\b", g):
            continue
        if re.search(r"\b(previous step|identified in the previous)\b", g):
            continue
        break
    else:
        return subtasks
    first_key = _subtask_order_key(subtasks[0])
    needs = any(
        _subtask_order_key(st)[0] < first_key[0]
        or _subtask_order_key(st)[1] < first_key[1]
        for st in subtasks[1:]
    )
    if not needs:
        return subtasks
    indexed = sorted(enumerate(subtasks), key=lambda ix: (_subtask_order_key(ix[1]), ix[0]))
    return [st for _, st in indexed]


def _enforce_linear_chain(chain: list[dict[str, Any]]) -> None:
    if not chain:
        return
    chain[0]["depends_on"] = []
    for i in range(1, len(chain)):
        _merge_depends_on(chain[i], [chain[i - 1]["id"]])


def terminal_subtask_ids(subtasks: list[dict[str, Any]]) -> list[str]:
    """Sink subtasks — not referenced as a dependency by any other subtask."""
    if not subtasks:
        return []
    all_ids = [str(st.get("id", "")).strip() for st in subtasks if str(st.get("id", "")).strip()]
    referenced: set[str] = set()
    for st in subtasks:
        for dep in st.get("depends_on") or []:
            dep_id = str(dep).strip()
            if dep_id:
                referenced.add(dep_id)
    sinks = [sid for sid in all_ids if sid not in referenced]
    if not sinks or len(sinks) == len(all_ids):
        return [all_ids[-1]] if all_ids else []
    return sinks


def infer_subtask_dependencies(
    subtasks: list[dict[str, Any]],
    *,
    dataset: str,
) -> list[dict[str, Any]]:
    """
    Repair ``depends_on`` when the planner leaves chains empty.

    - Text/placeholder inference (all datasets): goal or ``search_query`` mentions
      ``from s1`` / ``from t1``, ``<... from t1>``, etc.

    Web-hop retrieve chaining is applied in ``validate_planner_subtasks`` (Hotpot / MuSiQue).
    """
    if not subtasks:
        return subtasks

    valid = _valid_subtask_ids(subtasks)
    for st in subtasks:
        sid = str(st.get("id", "")).strip().lower()
        blob = f"{st.get('goal', '')} {st.get('search_query', '')}"
        inferred = _mentioned_dep_ids(blob, sid, valid)
        if inferred:
            _merge_depends_on(st, inferred)

    _propagate_semantic_dependencies(subtasks)
    return subtasks


def validate_planner_subtasks(
    subtasks: list[dict[str, Any]],
    *,
    dataset: str,
    n_hops: int | None = None,
    hop_name: str | None = None,
) -> list[dict[str, Any]]:
    """
    Repair planner ``depends_on`` edges.

    All datasets: ``infer_subtask_dependencies`` (text/placeholder + semantic propagation).
    Hotpot / MuSiQue linear: inside-out reorder + enforce linear retrieve chain.
    """
    if not subtasks:
        return subtasks

    ds = (dataset or "").strip().lower()
    subtasks = infer_subtask_dependencies(subtasks, dataset=ds)

    if ds in OPEN_WIKI_QA_DATASETS and (coerce_hop_name(hop_name) or "") == "linear":
        subtasks = _reorder_inside_out(subtasks)
        chain = _retrieve_chain_subtasks(subtasks)
        if len(chain) >= 2:
            for st in chain:
                st["depends_on"] = []
            _enforce_linear_chain(chain)

    return subtasks


def observation_text_from_tools_results(
    tools_results: list[dict[str, Any]] | None,
) -> str:
    """Last non-finish tool observation; Wikipedia ``content`` or web_search snippets."""
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
        results = data.get("results")
        if isinstance(results, list) and results:
            lines: list[str] = []
            for i, hit in enumerate(results[:5], start=1):
                if not isinstance(hit, dict):
                    continue
                title = str(hit.get("title", "") or "").strip()
                snippet = str(hit.get("snippet", hit.get("body", "")) or "").strip()
                url = str(hit.get("url", hit.get("href", "")) or "").strip()
                part = f"{i}. {title}: {snippet}" if title else f"{i}. {snippet}"
                if url:
                    part += f" ({url})"
                if part.strip():
                    lines.append(part)
            if lines:
                return "\n".join(lines)[:4000]
        return obs[:2000]
    return ""
