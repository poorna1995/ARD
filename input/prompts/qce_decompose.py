"""QCE decomposition prompts — executable procedure DAG."""

from __future__ import annotations

import json
import re
from typing import Any

from config.local.constants import DS, PLAN, TOOLS

DATASETS = DS.names
RELATION_TYPES = PLAN.relations
STEP_TYPES = PLAN.steps
TOOLS_PROMPT = TOOLS.prompt
ANSWER_GRANULARITY = PLAN.grans
_DEFAULT_RELATION = PLAN.default_relation


def _enum(xs: frozenset[str] | set[str]) -> str:
    return " | ".join(sorted(xs))


_SYSTEM_CORE = f"""QCE procedure planner: decompose the question into a small acyclic DAG of subtasks for structural + retrieval-planning + semantic complexity. Do not answer, leak the answer, or name agents (raw/cot/react/multiagent). Output one JSON object only (no markdown).

#1 RULE — terminal sink: the highest-id subtask (the sink) MUST directly satisfy the original question constraint (final_constraint). Its goal, relation_type, step_type, and answer_granularity target what the question asks — never stop on intermediate identification (entity, location, colony, suburb, director) if the question also requires who/when/how many/which state/year/etc. Add prerequisite hops first, then a terminal retrieve|reason|compute|verify|select|merge hop that resolves the asked fact; incomplete DAGs that omit this underestimate complexity.

Top-level: final_constraint (answer shape), subtasks[].
Each subtask — all keys required: id, goal (≤12 words), relation_type, answer_granularity, step_type, needs_tool, tool, search_query, depends_on.
Ids: t1, t2, … sequential, no gaps.

Enums:
  relation_type: {_enum(RELATION_TYPES)}
  step_type: {_enum(STEP_TYPES)}
  tool: none | {_enum(TOOLS_PROMPT - {"none"})}  (needs_tool=false ⇒ tool=none, search_query="")
  answer_granularity / final_constraint: "" | {_enum(ANSWER_GRANULARITY - {""})}

depends_on: prerequisite ids only; acyclic. Chain: t1[]; t2[t1]; …. Comparison: t1[]; t2[]; merge t3[t1,t2].
search_query: required on retrieve+needs_tool — short entity-focused query (3–12 tokens); never the full question, vague lookups, or answer text. Else "".
Dependency-semantics: if depends_on is non-empty and step_type=retrieve, search_query MUST refine retrieval using entities/relations from prerequisites (compositional); no weak edges where the query ignores depends_on; no placeholders ("from t1").
Inside-out multi-hop: resolve inner entities first; later hops use concrete names from earlier hops in search_query (not placeholders like "entity from t1" or "country from t3").
Goals: procedure only — no final answers, option letters, or "the answer is …"."""

_STRATEGY: dict[str, str] = {
    "hotpot": "hotpot: exactly 2 subtasks for bridge; t1 resolves bridge anchor; t2[t1] is the TERMINAL hop — goal+granularity must match what the question asks (span/phrase/name/year), not a different intermediate attribute (wrong: t2 finds suburb when question asks who founded — t2 must retrieve founder scoped to suburb from t1). comparison: t1[],t2[] entity facts, t3[t1,t2] verify/reason/select answers constraint; prefer wikipedia_search.",
    "musique": "musique: 2–5 hops; chain or merge DAG; every dependent retrieve uses compositional scoped search_query; sink MUST satisfy final_constraint (year/date/name/city/state) — if question asks a year or law date after entity/colony resolution, add terminal temporal retrieve depending on all prerequisites; never end on colony/country/continent identification alone when final_constraint is year|date|name; web_search|wikipedia_search.",
    "math": "math: 2–4 steps; sink compute|verify satisfies final_constraint (number|span|unit); parallel reason/compute merge OK; no web/wikipedia; search_query \"\".",
    "mmlu": "mmlu: exactly 3; sink t3 select|verify satisfies final_constraint letter.",
    "gaia": "gaia: 2–4 steps; sink verify|retrieve satisfies final_constraint; read_file first if attachment; dependent search_query uses prior-step entities.",
}

# Compact few-shots (full schema demonstrated per family).
_EXAMPLE_JSON: dict[str, str] = {
    "hotpot_bridge": (
        '{"final_constraint":"name","subtasks":['
        '{"id":"t1","goal":"Identify suburb of house","relation_type":"entity_lookup","answer_granularity":"",'
        '"step_type":"retrieve","needs_tool":true,"tool":"wikipedia_search","search_query":"Beaumont House suburb Adelaide","depends_on":[]},'
        '{"id":"t2","goal":"Find founder of suburb","relation_type":"compositional","answer_granularity":"name",'
        '"step_type":"retrieve","needs_tool":true,"tool":"wikipedia_search","search_query":"founder of Beaumont suburb Adelaide","depends_on":["t1"]}]}'
    ),
    "hotpot_comparison": (
        '{"final_constraint":"phrase","subtasks":['
        '{"id":"t1","goal":"Facts for entity A","relation_type":"comparison","answer_granularity":"",'
        '"step_type":"retrieve","needs_tool":true,"tool":"wikipedia_search","search_query":"India official languages count","depends_on":[]},'
        '{"id":"t2","goal":"Facts for entity B","relation_type":"comparison","answer_granularity":"",'
        '"step_type":"retrieve","needs_tool":true,"tool":"wikipedia_search","search_query":"China official languages count","depends_on":[]},'
        '{"id":"t3","goal":"Compare A and B","relation_type":"comparison","answer_granularity":"phrase",'
        '"step_type":"verify","needs_tool":false,"tool":"none","search_query":"","depends_on":["t1","t2"]}]}'
    ),
    "musique": (
        '{"final_constraint":"year","subtasks":['
        '{"id":"t1","goal":"Resolve country for place","relation_type":"entity_lookup","answer_granularity":"",'
        '"step_type":"retrieve","needs_tool":true,"tool":"web_search","search_query":"Prazeres country location","depends_on":[]},'
        '{"id":"t2","goal":"Identify former colonial holding","relation_type":"compositional","answer_granularity":"",'
        '"step_type":"retrieve","needs_tool":true,"tool":"web_search","search_query":"Portugal former colony Netherlands Antilles","depends_on":["t1"]},'
        '{"id":"t3","goal":"Find hazardous work law year","relation_type":"temporal","answer_granularity":"year",'
        '"step_type":"retrieve","needs_tool":true,"tool":"web_search","search_query":"Netherlands Antilles child labor law year","depends_on":["t2"]}]}'
    ),
    "math": (
        '{"final_constraint":"number","subtasks":['
        '{"id":"t1","goal":"Parse and set unknowns","relation_type":"numeric","answer_granularity":"",'
        '"step_type":"reason","needs_tool":false,"tool":"none","search_query":"","depends_on":[]},'
        '{"id":"t2","goal":"Compute value","relation_type":"numeric","answer_granularity":"",'
        '"step_type":"compute","needs_tool":true,"tool":"math_tool","search_query":"","depends_on":["t1"]},'
        '{"id":"t3","goal":"Verify constraints","relation_type":"verification","answer_granularity":"number",'
        '"step_type":"verify","needs_tool":false,"tool":"none","search_query":"","depends_on":["t2"]}]}'
    ),
    "mmlu": (
        '{"final_constraint":"letter","subtasks":['
        '{"id":"t1","goal":"Understand stem","relation_type":"entity_lookup","answer_granularity":"",'
        '"step_type":"reason","needs_tool":false,"tool":"none","search_query":"","depends_on":[]},'
        '{"id":"t2","goal":"Evaluate options","relation_type":"causal","answer_granularity":"",'
        '"step_type":"reason","needs_tool":false,"tool":"none","search_query":"","depends_on":["t1"]},'
        '{"id":"t3","goal":"Confirm best option","relation_type":"verification","answer_granularity":"letter",'
        '"step_type":"select","needs_tool":false,"tool":"none","search_query":"","depends_on":["t2"]}]}'
    ),
    "gaia": (
        '{"final_constraint":"phrase","subtasks":['
        '{"id":"t1","goal":"Read attachment","relation_type":"entity_lookup","answer_granularity":"",'
        '"step_type":"retrieve","needs_tool":true,"tool":"read_file","search_query":"inventory.xlsx","depends_on":[]},'
        '{"id":"t2","goal":"Fetch external fact for file entity","relation_type":"compositional","answer_granularity":"",'
        '"step_type":"retrieve","needs_tool":true,"tool":"web_search","search_query":"Acme Corp 2019 revenue statistic","depends_on":["t1"]},'
        '{"id":"t3","goal":"Merge evidence","relation_type":"verification","answer_granularity":"phrase",'
        '"step_type":"verify","needs_tool":false,"tool":"none","search_query":"","depends_on":["t2"]}]}'
    ),
}

_COMPARISON_RE = re.compile(
    r"(?i)(which\s+(is|are|was|were)\s+(more|less|older|younger|taller|longer|larger|smaller|greater|higher|lower)"
    r"|compare|comparison|more\s+.+\s+than|less\s+.+\s+than"
    r"|(more|fewer|less)\s+.+\s+or\s+|(older|younger|taller|longer|larger|smaller)\s+.*\bor\b"
    r"|who\s+is\s+(older|younger|taller))"
)


def example_json(key: str) -> str:
    if key not in _EXAMPLE_JSON:
        raise KeyError(key)
    return _EXAMPLE_JSON[key]


def normalize_relation_type(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    if val in RELATION_TYPES:
        return val
    if val in ("verify", "verification_step"):
        return "verification"
    return _DEFAULT_RELATION


def normalize_answer_granularity(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    return val if val in ANSWER_GRANULARITY else ""


def normalize_final_constraint(raw: Any) -> str:
    val = str(raw or "").strip().lower()
    return val if val in ANSWER_GRANULARITY else (val if val else "")


def _hotpot_topology(
    query: str, metadata: dict[str, str | int | float] | None
) -> str:
    if metadata:
        rt = str(metadata.get("reasoning_type", "")).strip().lower()
        if rt in ("comparison", "bridge", "intersection"):
            return rt
    return "comparison" if _COMPARISON_RE.search(query) else "bridge"


def _example_key(dataset: str, *, topology: str | None = None) -> str:
    ds = dataset.strip().lower()
    if ds == "hotpot":
        return "hotpot_comparison" if topology == "comparison" else "hotpot_bridge"
    return ds if ds in _EXAMPLE_JSON else "gaia"


def build_system_content(
    dataset: str,
    *,
    topology: str | None = None,
) -> str:
    ds = dataset.strip().lower()
    key = _example_key(ds, topology=topology)
    return "\n".join(
        (
            _SYSTEM_CORE.strip(),
            f"Example ({key}): {example_json(key)}",
            _STRATEGY.get(ds, _STRATEGY["gaia"]),
        )
    )


def build_user_content(
    query_norm: str,
    dataset: str,
    *,
    metadata: dict[str, str | int | float] | None = None,
    topology: str | None = None,
) -> str:
    ds = dataset.strip().lower()
    lines = [f"dataset={ds}"]
    if ds == "hotpot":
        lines.append(f"topology_hint={topology or _hotpot_topology(query_norm, metadata)}")
    if metadata:
        meta = "; ".join(
            f"{k}={v}" for k, v in metadata.items() if v is not None and str(v).strip()
        )
        if meta:
            lines.append(f"metadata: {meta}")
    lines.extend(("question:", query_norm.strip()))
    return "\n".join(lines)


def plan_msgs(
    query_norm: str,
    dataset: str,
    *,
    metadata: dict[str, str | int | float] | None = None,
) -> list[dict[str, str]]:
    ds = dataset.strip().lower()
    if ds not in DATASETS:
        raise ValueError(f"unsupported dataset {dataset!r}")
    topo = _hotpot_topology(query_norm, metadata) if ds == "hotpot" else None
    return [
        {"role": "system", "content": build_system_content(ds, topology=topo)},
        {
            "role": "user",
            "content": build_user_content(
                query_norm, ds, metadata=metadata, topology=topo
            ),
        },
    ]
