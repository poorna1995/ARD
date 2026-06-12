"""Step 2.2 — build procedure DAGs from plans and emit graph + planning features.

C(Q) main (v3.0) uses five dimensions for the primary router:
  structural, reasoning, evidence, tool, coordination_uncertainty.

Legacy five ``dim_legacy_*`` columns are retained as secondary/auxiliary outputs
for ablation and backward compatibility.
``complexity_graph`` remains a secondary scalar (Step 2.4).

  main dim_*          primary router dimensions (v3.0)
  legacy dim_legacy_* experiment 1 (kept as secondary)
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import networkx as nx

from agent.dataset_profile import infer_subtask_dependencies
from config.local.constants import DIMS, DS, GRAPH, PLAN, SCORE, STATUS, TOOLS
from qce.decompose import load_plans_jsonl

# Backward-compatible node ids (prefer config.local.qce.GRAPH).
START = GRAPH.start
END = GRAPH.end


@dataclass(frozen=True)
class TrainNorm:
    max_depth: float = 6.0
    width: float = 4.0
    n_tool_nodes: float = 4.0
    n_nodes: float = 6.0
    critical_path_tool_steps: float = 4.0
    n_repair_ops: float = 8.0
    avg_search_query_tokens: float = 12.0


@dataclass
class GraphBuildResult:
    training_id: str
    dataset: str
    graph: nx.DiGraph
    features: dict[str, Any]
    repair_log: list[str] = field(default_factory=list)
    status: str = STATUS.ok


def _step_type(st: dict[str, Any]) -> str:
    return str(st.get("step_type", "")).strip().lower()


def _sub_id(st: dict[str, Any]) -> str:
    return str(st.get("id", "")).strip()


def _sub_dag(g: nx.DiGraph, sub: nx.DiGraph | None) -> bool:
    return sub is None or nx.is_directed_acyclic_graph(sub)


def build_task_dag(plan: dict[str, Any]) -> GraphBuildResult:
    """Plan JSONL row → DAG + feature dict (atomics + dim_* + complexity_graph)."""
    tid = str(plan.get("training_id", ""))
    dataset = str(plan.get("dataset", "")).strip().lower()
    subtasks: list[dict[str, Any]] = list(plan.get("subtasks") or [])
    repair_log: list[str] = []
    fc = str(plan.get("final_constraint", ""))
    norm = TrainNorm()

    if not subtasks:
        g = nx.DiGraph()
        g.add_node(GRAPH.start, node_kind="start")
        g.add_node(GRAPH.end, node_kind="end")
        g.add_edge(GRAPH.start, GRAPH.end, edge_kind="virtual")
        feats = _assemble_features(g, [], set(), [], repair_log, STATUS.empty, tid, dataset, fc, norm)
        return GraphBuildResult(tid, dataset, g, feats, repair_log, feats["graph_status"])

    subtasks, prep_log = _prepare_subtasks(subtasks, dataset)
    repair_log.extend(prep_log)

    g = nx.DiGraph()
    order = _plan_order(subtasks)
    ids = list(order.keys())
    id_set = set(ids)

    _add_nodes(g, subtasks)
    _add_edges(g, subtasks, order, id_set, repair_log)

    sub = g.subgraph(ids) if ids else None
    if ids and not _sub_dag(g, sub):
        _drop_cycles(g, ids, repair_log)
        sub = g.subgraph(ids)
    if ids and not _sub_dag(g, sub):
        _chain_fallback(g, subtasks, ids, dataset, repair_log)
        sub = g.subgraph(ids)

    _add_virtual(g, ids, id_set)
    if ids and not _sub_dag(g, sub):
        raise RuntimeError(f"subtask graph still cyclic after repair: {tid}")

    graph_status = (
        STATUS.repair
        if any(e.startswith(SCORE.prefixes) for e in repair_log)
        else STATUS.ok
    )
    on_graph = _graph_subtasks(subtasks, id_set)
    feats = _assemble_features(
        g, ids, id_set, on_graph, repair_log, graph_status, tid, dataset, fc, norm
    )
    return GraphBuildResult(tid, dataset, g, feats, repair_log, feats["graph_status"])


def build_graphs_from_plans(plans: list[dict[str, Any]]) -> list[GraphBuildResult]:
    return [build_task_dag(p) for p in plans]


def features_dataframe(results: list[GraphBuildResult]):
    import pandas as pd

    return pd.DataFrame(
        [{**r.features, "graph_status": r.status} for r in results]
    )


def fit_train_norm(df) -> TrainNorm:
    def _cap(col: str, default: float) -> float:
        if col not in df.columns:
            return default
        vals = df[col].dropna()
        if len(vals) == 0:
            return default
        return max(float(vals.quantile(0.95)), 1.0)

    return TrainNorm(
        max_depth=_cap("max_depth", 6.0),
        width=_cap("width", 4.0),
        n_tool_nodes=_cap("n_tool_nodes", 4.0),
        n_nodes=_cap("n_nodes", 6.0),
        critical_path_tool_steps=_cap("critical_path_tool_steps", 4.0),
        n_repair_ops=_cap("n_repair_ops", 8.0),
        avg_search_query_tokens=_cap("avg_search_query_tokens", 12.0),
    )


def _rescore_row(f: dict[str, Any], norm: TrainNorm) -> dict[str, Any]:
    """Recompute scalar + all dimension columns from one feature row."""
    return {
        "complexity_graph": _complexity_graph(f, norm),
        **_conceptual_dims(f, norm),
        **_conceptual_dims_legacy(f, norm),
    }


def rescore_dataframe(df, norm: TrainNorm | None = None):
    norm = norm or TrainNorm()
    out = df.copy()
    scored = [_rescore_row(row, norm) for row in out.to_dict("records")]
    score_keys = ("complexity_graph", *DIMS.main, *DIMS.legacy)
    for key in score_keys:
        out[key] = [s[key] for s in scored]
    return out


def _assemble_features(
    g: nx.DiGraph,
    ids: list[str],
    id_set: set[str],
    on_graph: list[dict[str, Any]],
    repair_log: list[str],
    graph_status: str,
    tid: str,
    dataset: str,
    final_constraint: str,
    norm: TrainNorm,
) -> dict[str, Any]:
    feats = _integrity_and_dims(g, ids, repair_log, graph_status, norm)
    feats.update(_plan_step_signals(on_graph))
    feats.update(_plan_semantic_signals(on_graph, final_constraint))
    feats.update(_plan_interaction_signals(g, ids, on_graph))
    feats.update(_conceptual_dims(feats, norm))
    feats.update(_conceptual_dims_legacy(feats, norm))
    feats.update(training_id=tid, dataset=dataset, final_constraint=final_constraint)
    return feats


# --- plan step signals (verification dimension) ---


def _graph_subtasks(
    subtasks: list[dict[str, Any]], id_set: set[str]
) -> list[dict[str, Any]]:
    """Subtasks that became DAG nodes (non-empty id in planner order)."""
    return [st for st in subtasks if _sub_id(st) in id_set]


def _plan_step_signals(subtasks: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(subtasks)
    if not n:
        return {
            "n_verify_steps": 0,
            "verify_fraction": 0.0,
            "n_retrieve_steps": 0,
            "n_reason_steps": 0,
            "n_compute_steps": 0,
            "n_select_steps": 0,
        }
    counts = Counter(_step_type(st) for st in subtasks)
    n_verify = counts.get("verify", 0)
    return {
        "n_verify_steps": n_verify,
        "verify_fraction": round(n_verify / n, 4),
        "n_retrieve_steps": counts.get("retrieve", 0),
        "n_reason_steps": counts.get("reason", 0),
        "n_compute_steps": counts.get("compute", 0),
        "n_select_steps": counts.get("select", 0),
    }


def _sink_subtask(subtasks: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not subtasks:
        return None

    def _tid(st: dict[str, Any]) -> int:
        m = PLAN.task_id.match(_sub_id(st).lower())
        return int(m.group(1)) if m else 0

    return max(subtasks, key=_tid)


def _plan_semantic_signals(
    subtasks: list[dict[str, Any]], final_constraint: str
) -> dict[str, Any]:
    """Aggregate relation_type, answer_granularity, search_query into scalars for C(Q)."""
    n = len(subtasks)
    fc = str(final_constraint or "").strip().lower()
    if not n:
        return {
            "relation_diversity": 0.0,
            "relation_entropy": 0.0,
            "compositional_relation_fraction": 0.0,
            "avg_search_query_tokens": 0.0,
            "weak_search_query_fraction": 0.0,
            "scoped_retrieve_fraction": 0.0,
            "granularity_diversity": 0.0,
            "terminal_sink_ok": 0,
            "sink_intermediate_risk": 0,
        }

    relations: list[str] = []
    rel_clean: list[str] = []
    retrieves: list[dict[str, Any]] = []
    grans: list[str] = []
    for st in subtasks:
        r = str(st.get("relation_type", "")).strip().lower()
        relations.append(r)
        if r:
            rel_clean.append(r)
        if _step_type(st) == "retrieve":
            retrieves.append(st)
        g = str(st.get("answer_granularity", "")).strip().lower()
        if g:
            grans.append(g)

    relation_diversity = len(set(rel_clean)) / max(len(rel_clean), 1)
    if rel_clean:
        rel_counts = Counter(rel_clean)
        n_rel = len(rel_clean)
        probs = (c / n_rel for c in rel_counts.values())
        entropy = -sum(p * math.log(p + 1e-12) for p in probs)
        relation_entropy = (
            entropy / max(math.log(len(rel_counts)), 1e-12) if len(rel_counts) > 1 else 0.0
        )
    else:
        relation_entropy = 0.0
    compositional_relation_fraction = sum(1 for r in relations if r in PLAN.sem_rels) / n

    queries: list[str] = []
    scoped = 0
    for st in retrieves:
        q = str(st.get("search_query", "")).strip()
        if not q:
            continue
        queries.append(q)
        if (st.get("depends_on") or []) and not PLAN.bad_query.search(q):
            scoped += 1
    n_q = len(queries)
    avg_tokens = sum(len(q.split()) for q in queries) / n_q if n_q else 0.0
    weak_fraction = (
        sum(1 for q in queries if PLAN.bad_query.search(q)) / n_q if n_q else 0.0
    )
    scoped_retrieve_fraction = scoped / len(retrieves) if retrieves else 0.0
    granularity_diversity = len(set(grans)) / n if grans else 0.0

    sink = _sink_subtask(subtasks)
    sink_gran = str(sink.get("answer_granularity", "")).strip().lower() if sink else ""
    terminal_sink_ok = int(not fc or sink_gran == fc)
    sink_intermediate_risk = int(bool(fc and sink_gran != fc))

    return {
        "relation_diversity": round(relation_diversity, 4),
        "relation_entropy": round(float(relation_entropy), 4),
        "compositional_relation_fraction": round(compositional_relation_fraction, 4),
        "avg_search_query_tokens": round(avg_tokens, 4),
        "weak_search_query_fraction": round(weak_fraction, 4),
        "scoped_retrieve_fraction": round(scoped_retrieve_fraction, 4),
        "granularity_diversity": round(granularity_diversity, 4),
        "terminal_sink_ok": terminal_sink_ok,
        "sink_intermediate_risk": sink_intermediate_risk,
    }


# --- DAG topology + tool metrics ---


def _measure_graph(g: nx.DiGraph, ids: list[str]) -> dict[str, Any]:
    n = len(ids)
    if n == 0:
        return _empty_graph_metrics()

    id_set = set(ids)
    sub = g.subgraph(id_set)
    dep_edges = [
        (u, v)
        for u, v, d in g.edges(data=True)
        if u in id_set and v in id_set and d.get("edge_kind") == "dependency"
    ]
    n_edges = len(dep_edges)
    in_deg = dict(sub.in_degree())
    out_deg = dict(sub.out_degree())

    n_sources = n_sinks = n_merge = n_split = 0
    n_tool = 0
    unique_tools: set[str] = set()
    has_web = has_code = has_file = 0
    for i in ids:
        inde = in_deg.get(i, 0)
        outd = out_deg.get(i, 0)
        if inde == 0:
            n_sources += 1
        if outd == 0:
            n_sinks += 1
        if inde >= 2:
            n_merge += 1
        if outd >= 2:
            n_split += 1
        nd = g.nodes[i]
        t = str(nd.get("tool", "none")).lower()
        if nd.get("is_tool"):
            n_tool += 1
        if t and t != "none":
            unique_tools.add(t)
            if t in TOOLS.web:
                has_web = 1
            if t in TOOLS.code:
                has_code = 1
            if t in TOOLS.file:
                has_file = 1

    depth_map, max_width = _layer_depths(g, id_set)
    depths = [depth_map[i] for i in ids if i in depth_map]
    avg_depth = sum(depths) / len(depths) if depths else 0.0
    max_depth = _longest_subtask_path(g, id_set)
    crit_len, crit_tool_steps, on_crit = _critical_path(g, ids, id_set)
    n_comp = nx.number_weakly_connected_components(sub)
    denom = max(n * (n - 1), 1)
    hub = n_merge + n_split

    return {
        "n_nodes": n,
        "n_edges": n_edges,
        "max_depth": max_depth,
        "avg_depth": round(avg_depth, 4),
        "width": max_width,
        "dag_width": max_width,
        "n_sources": n_sources,
        "n_sinks": n_sinks,
        "parallel_ratio": round(hub / n, 4),
        "has_parallel": int(max_width >= 2 or n_sources >= 2 or n_merge >= 1),
        "n_merge_nodes": n_merge,
        "n_tool_nodes": n_tool,
        "tool_fraction": round(n_tool / n, 4),
        "n_unique_tools": len(unique_tools),
        "has_web": has_web,
        "has_code": has_code,
        "has_file": has_file,
        "critical_path_len": crit_len,
        "critical_path_tool_steps": crit_tool_steps,
        "tool_on_critical_path": int(on_crit),
        "edge_density": round(n_edges / denom, 4),
        "n_components": n_comp,
        "is_dag": nx.is_directed_acyclic_graph(sub),
        "is_weakly_connected": int(n_comp <= 1),
    }


def _empty_graph_metrics() -> dict[str, Any]:
    return {k: 0 for k in (
        "n_nodes", "n_edges", "max_depth", "avg_depth", "width", "dag_width",
        "n_sources", "n_sinks", "parallel_ratio", "has_parallel", "n_merge_nodes",
        "n_tool_nodes", "tool_fraction", "n_unique_tools", "has_web", "has_code",
        "has_file", "critical_path_len", "critical_path_tool_steps",
        "tool_on_critical_path", "edge_density", "n_components",
    )} | {"is_dag": True, "is_weakly_connected": 1}


def _layer_depths(g: nx.DiGraph, id_set: set[str]) -> tuple[dict[str, int], int]:
    depth: dict[str, int] = {}
    max_width = 0
    if not nx.is_directed_acyclic_graph(g):
        return depth, 0
    for layer in nx.topological_generations(g):
        sub = [node for node in layer if node in id_set]
        if not sub:
            continue
        max_width = max(max_width, len(sub))
        for node in sub:
            preds = [p for p in g.predecessors(node) if p in id_set or p == GRAPH.start]
            depth[node] = (
                0
                if not preds
                else 1 + max(-1 if p == GRAPH.start else depth.get(p, 0) for p in preds)
            )
    return depth, max_width


def _longest_subtask_path(g: nx.DiGraph, id_set: set[str]) -> int:
    if not id_set or not nx.is_directed_acyclic_graph(g):
        return 0
    try:
        return sum(1 for node in nx.dag_longest_path(g) if node in id_set)
    except nx.NetworkXError:
        return 0


def _critical_path(g: nx.DiGraph, ids: list[str], id_set: set[str]) -> tuple[int, int, bool]:
    h = g.subgraph(id_set).copy()
    if h.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(h):
        return 0, 0, False
    if GRAPH.start in g and GRAPH.end in g and nx.has_path(g, GRAPH.start, GRAPH.end):
        try:
            sub = [node for node in nx.dag_longest_path(g) if node in id_set]
            tools = sum(1 for node in sub if g.nodes[node].get("is_tool"))
            return len(sub), tools, tools > 0
        except nx.NetworkXError:
            pass
    s, t = "__cp_s__", "__cp_t__"
    h.add_node(s)
    h.add_node(t)
    for node in ids:
        if h.in_degree(node) == 0:
            h.add_edge(s, node, edge_kind="virtual")
        if h.out_degree(node) == 0:
            h.add_edge(node, t, edge_kind="virtual")
    try:
        sub = [node for node in nx.dag_longest_path(h) if node in id_set]
    except nx.NetworkXError:
        return 0, 0, False
    tools = sum(1 for node in sub if g.nodes[node].get("is_tool"))
    return len(sub), tools, tools > 0


def _weighted_critical_path_nodes(g: nx.DiGraph, id_set: set[str]) -> list[str]:
    """Weighted critical path on subtask DAG (semantic step-type weighting)."""
    h = g.subgraph(id_set).copy()
    if h.number_of_nodes() == 0 or not nx.is_directed_acyclic_graph(h):
        return []

    def _node_weight(node: str) -> float:
        return PLAN.step_wts.get(str(g.nodes[node].get("step_type", "")).strip().lower(), 1.0)

    score: dict[str, float] = {}
    parent: dict[str, str | None] = {}
    for node in nx.topological_sort(h):
        best_pred = None
        best_pred_score = 0.0
        for pred in h.predecessors(node):
            s = score.get(pred, 0.0)
            if s > best_pred_score:
                best_pred_score = s
                best_pred = pred
        score[node] = best_pred_score + _node_weight(node)
        parent[node] = best_pred

    end = max(score, key=score.get)
    path: list[str] = []
    cur: str | None = end
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


# --- integrity, composite score, conceptual dimensions ---


def _plan_trust(repair_log: list[str], graph_status: str) -> float:
    if graph_status == STATUS.empty:
        return 0.0
    penalty = sum(SCORE.repair.get(e.split(":", 1)[0], 0.05) for e in repair_log)
    return max(0.0, min(1.0, 1.0 - penalty))


def _norm01(val: float, cap: float) -> float:
    return min(1.0, max(0.0, val / max(cap, 1.0)))


def _norm01_log(val: float, cap: float) -> float:
    """Log-scale heavy-tailed count features before [0,1] normalization."""
    return _norm01(math.log1p(max(val, 0.0)), math.log1p(max(cap, 1.0)))


def complexity_graph_score(f: dict[str, Any], norm: TrainNorm | None = None) -> float:
    """Step 2.4 aggregate scalar in ~[0, 1] — plots/gating only, not primary router input."""
    if f.get("graph_status") == STATUS.empty:
        return 0.0
    return _complexity_graph(f, norm or TrainNorm())


def _complexity_graph(f: dict[str, Any], norm: TrainNorm) -> float:
    n = float(f.get("n_nodes", 0))
    terms = (
        _norm01(float(f.get("max_depth", 0)), norm.max_depth),
        _norm01_log(float(f.get("n_tool_nodes", 0)), norm.n_tool_nodes),
        min(1.0, float(f.get("tool_fraction", 0))),
        _norm01_log(float(f.get("critical_path_tool_steps", 0)), norm.critical_path_tool_steps),
        _norm01(math.log1p(n), math.log1p(norm.n_nodes)),
        min(1.0, float(f.get("verify_fraction", 0))),
    )
    w = SCORE.weights
    keys = (
        "max_depth",
        "n_tool_nodes",
        "tool_fraction",
        "critical_path_tool_steps",
        "log_nodes",
        "verify_fraction",
    )
    return round(sum(w[k] * t for k, t in zip(keys, terms)), 4)


def _plan_interaction_signals(
    g: nx.DiGraph,
    ids: list[str],
    subtasks: list[dict[str, Any]],
) -> dict[str, Any]:
    """Tool-sequence / dependency signals for main C(Q) dimensions."""
    n = len(subtasks)
    if not n or not ids:
        return {
            "retrieve_step_fraction": 0.0,
            "execution_external_fraction": 0.0,
            "dependent_hop_fraction": 0.0,
            "temporal_relation_fraction": 0.0,
            "comparison_relation_fraction": 0.0,
            "tool_switching": 0.0,
            "tool_switching_seq": 0.0,
            "tool_switching_edge": 0.0,
            "cross_tool_dep_fraction": 0.0,
            "execution_tool_diversity": 0.0,
            "critical_retrieve_fraction": 0.0,
        }

    id_set = set(ids)
    tools_by_id: dict[str, str] = {}
    n_retrieve = sum(1 for st in subtasks if _step_type(st) == "retrieve")
    n_with_deps = sum(1 for st in subtasks if st.get("depends_on"))
    relations = [str(st.get("relation_type", "")).strip().lower() for st in subtasks]
    n_temporal = sum(1 for r in relations if r in PLAN.temp_rels)
    n_comparison = sum(1 for r in relations if r in PLAN.comp_rels)
    for st in subtasks:
        sid = _sub_id(st)
        if sid in id_set:
            tools_by_id[sid] = str(st.get("tool", "none")).strip().lower() or "none"

    ordered_ids = [i for i in ids if i in tools_by_id]
    exec_nodes = sum(
        1 for t in tools_by_id.values() if t in TOOLS.exec and t not in TOOLS.retr
    )
    n_tool_nodes = sum(1 for i in ids if g.nodes[i].get("is_tool"))
    execution_external_fraction = exec_nodes / max(n_tool_nodes, 1) if n_tool_nodes else 0.0

    switches = sum(
        1
        for a, b in zip(ordered_ids, ordered_ids[1:])
        if (ta := tools_by_id[a]) != (tb := tools_by_id[b]) and ta != "none" and tb != "none"
    )
    tool_switching_seq = switches / max(len(ordered_ids) - 1, 1)

    cross_tool = dep_edges = 0
    for u, v, d in g.edges(data=True):
        if d.get("edge_kind") != "dependency" or u not in id_set or v not in id_set:
            continue
        dep_edges += 1
        tu = tools_by_id.get(u, "none")
        tv = tools_by_id.get(v, "none")
        if tu != tv and tu != "none" and tv != "none":
            cross_tool += 1
    cross_tool_dep_fraction = cross_tool / max(dep_edges, 1)
    tool_switching_edge = cross_tool_dep_fraction
    tool_switching = (tool_switching_seq + tool_switching_edge) / 2.0

    unique_exec = len({t for t in tools_by_id.values() if t in TOOLS.exec and t != "none"})
    execution_tool_diversity = unique_exec / max(n_tool_nodes, 1) if n_tool_nodes else 0.0

    crit_len, _, _ = _critical_path(g, ids, id_set)
    if crit_len > 0 and ordered_ids:
        path = _weighted_critical_path_nodes(g, id_set) or ordered_ids
        web_on_path = sum(1 for node in path if tools_by_id.get(node, "none") in TOOLS.retr)
        critical_retrieve_fraction = web_on_path / max(len(path), 1)
    else:
        critical_retrieve_fraction = float(
            sum(1 for t in tools_by_id.values() if t in TOOLS.retr)
        ) / max(n_tool_nodes, 1)

    return {
        "retrieve_step_fraction": round(n_retrieve / n, 4),
        "execution_external_fraction": round(execution_external_fraction, 4),
        "dependent_hop_fraction": round(n_with_deps / n, 4),
        "temporal_relation_fraction": round(n_temporal / n, 4),
        "comparison_relation_fraction": round(n_comparison / n, 4),
        "tool_switching": round(tool_switching, 4),
        "tool_switching_seq": round(tool_switching_seq, 4),
        "tool_switching_edge": round(tool_switching_edge, 4),
        "cross_tool_dep_fraction": round(cross_tool_dep_fraction, 4),
        "execution_tool_diversity": round(execution_tool_diversity, 4),
        "critical_retrieve_fraction": round(critical_retrieve_fraction, 4),
    }


def _avg_terms(terms: list[float]) -> float:
    return round(sum(terms) / max(len(terms), 1), 4)


def _conceptual_dims(f: dict[str, Any], norm: TrainNorm) -> dict[str, float]:
    """Main C(Q) v3.0 dimensions used by the primary router."""
    relation_signal = min(
        1.0, float(f.get("relation_entropy", 0.0)) * max(0.0, float(f.get("plan_trust", 1.0)))
    )
    structural = _avg_terms(
        [
            _norm01_log(float(f.get("n_nodes", 0)), norm.n_nodes),
            min(1.0, float(f.get("edge_density", 0)) * 4.0),
            _norm01(float(f.get("max_depth", 0)), norm.max_depth),
            _norm01(float(f.get("width", 0)), norm.width),
            min(1.0, float(f.get("parallel_ratio", 0))),
            _norm01_log(float(f.get("n_sinks", 0)), max(1.0, norm.n_nodes)),
            _norm01_log(float(f.get("n_merge_nodes", 0)), max(2.0, norm.n_nodes / 2)),
        ]
    )
    reasoning = _avg_terms(
        [
            _norm01_log(float(f.get("critical_path_len", 0)), norm.max_depth),
            _norm01(float(f.get("max_depth", 0)), norm.max_depth),
            min(1.0, float(f.get("dependent_hop_fraction", 0))),
            _norm01_log(
                float(f.get("n_reason_steps", 0)) + float(f.get("n_compute_steps", 0)),
                max(1.0, norm.n_nodes),
            ),
            min(1.0, float(f.get("verify_fraction", 0))),
        ]
    )
    evidence = _avg_terms(
        [
            min(1.0, float(f.get("retrieve_step_fraction", 0))),
            min(1.0, float(f.get("critical_retrieve_fraction", 0))),
            min(1.0, float(f.get("scoped_retrieve_fraction", 0))),
            _norm01(float(f.get("avg_search_query_tokens", 0)), norm.avg_search_query_tokens),
            1.0 - min(1.0, float(f.get("weak_search_query_fraction", 0))),
            relation_signal,
        ]
    )
    tool = _avg_terms(
        [
            _norm01_log(float(f.get("n_tool_nodes", 0)), norm.n_tool_nodes),
            min(1.0, float(f.get("tool_fraction", 0))),
            min(1.0, float(f.get("execution_external_fraction", 0))),
            _norm01_log(float(f.get("critical_path_tool_steps", 0)), norm.critical_path_tool_steps),
            float(f.get("has_web", 0)),
            float(f.get("has_code", 0)),
            float(f.get("has_file", 0)),
        ]
    )
    coordination_uncertainty = _avg_terms(
        [
            min(1.0, float(f.get("tool_switching", 0))),
            min(1.0, float(f.get("cross_tool_dep_fraction", 0))),
            1.0 - float(f.get("plan_trust", 1.0)),
            _norm01_log(float(f.get("n_repair_ops", 0)), norm.n_repair_ops),
            min(1.0, float(f.get("weak_search_query_fraction", 0))),
            float(f.get("heavily_repaired", 0)),
            float(f.get("sink_intermediate_risk", 0)),
        ]
    )
    return {
        "dim_structural": round(structural, 4),
        "dim_reasoning": round(reasoning, 4),
        "dim_evidence": round(evidence, 4),
        "dim_tool": round(tool, 4),
        "dim_coordination_uncertainty": round(coordination_uncertainty, 4),
    }


def _conceptual_dims_legacy(f: dict[str, Any], norm: TrainNorm) -> dict[str, float]:
    """Legacy 5D C(Q) kept as secondary signals for archived experiments."""
    structural = (
        _norm01(float(f.get("max_depth", 0)), norm.max_depth)
        + _norm01(float(f.get("critical_path_len", 0)), norm.max_depth)
        + _norm01(float(f.get("width", 0)), norm.width)
        + min(1.0, float(f.get("relation_diversity", 0)))
    ) / 4.0
    compositional = (
        min(1.0, float(f.get("parallel_ratio", 0)))
        + _norm01_log(float(f.get("n_merge_nodes", 0)), max(2.0, norm.n_nodes / 2))
        + _norm01_log(float(f.get("n_nodes", 0)), norm.n_nodes)
        + min(1.0, float(f.get("compositional_relation_fraction", 0)))
    ) / 4.0
    retrieval = (
        min(1.0, float(f.get("tool_fraction", 0)))
        + float(f.get("has_web", 0))
        + _norm01(float(f.get("critical_path_tool_steps", 0)), norm.critical_path_tool_steps)
        + _norm01(float(f.get("avg_search_query_tokens", 0)), norm.avg_search_query_tokens)
        + min(1.0, float(f.get("scoped_retrieve_fraction", 0)))
    ) / 5.0
    verification = (
        min(1.0, float(f.get("verify_fraction", 0)))
        + min(1.0, float(f.get("granularity_diversity", 0)))
    ) / 2.0
    uncertainty = (
        1.0 - float(f.get("plan_trust", 1.0))
        + _norm01_log(float(f.get("n_repair_ops", 0)), norm.n_repair_ops)
        + float(f.get("sink_intermediate_risk", 0))
        + float(f.get("weak_search_query_fraction", 0))
    ) / 4.0
    return {
        "dim_legacy_structural": round(structural, 4),
        "dim_legacy_compositional": round(compositional, 4),
        "dim_legacy_retrieval": round(retrieval, 4),
        "dim_legacy_verification": round(verification, 4),
        "dim_legacy_uncertainty": round(uncertainty, 4),
    }


def _integrity_and_dims(
    g: nx.DiGraph,
    ids: list[str],
    repair_log: list[str],
    graph_status: str,
    norm: TrainNorm | None = None,
) -> dict[str, Any]:
    norm = norm or TrainNorm()
    metrics = _measure_graph(g, ids) if ids else _empty_graph_metrics()
    tag_counts = Counter(e.split(":", 1)[0] for e in repair_log)

    metrics["graph_status"] = graph_status
    metrics["n_repair_ops"] = len(repair_log)
    metrics["used_chain_fallback"] = int(tag_counts.get("chain_fallback", 0) > 0)
    metrics["n_cycle_edges_dropped"] = tag_counts.get("drop_cycle_edge", 0)
    metrics["n_non_forward_deps_dropped"] = tag_counts.get("drop_non_forward_dep", 0)
    metrics["n_unknown_deps"] = tag_counts.get("unknown_parent", 0)
    metrics["plan_trust"] = round(_plan_trust(repair_log, graph_status), 4)
    metrics["heavily_repaired"] = int(
        metrics["used_chain_fallback"]
        or (tag_counts.get("infer_linear_chain", 0) > 0 and tag_counts.get("infer_dep", 0) == 0)
    )
    metrics["n_subtasks"] = metrics["n_nodes"]
    metrics["n_edges_dep"] = metrics["n_edges"]
    metrics["max_width"] = metrics["width"]
    metrics["complexity_graph"] = (
        0.0 if graph_status == STATUS.empty else complexity_graph_score(metrics, norm)
    )
    metrics["is_dag"] = True
    return metrics


# --- graph construction helpers ---


def _prepare_subtasks(
    subtasks: list[dict[str, Any]], dataset: str
) -> tuple[list[dict[str, Any]], list[str]]:
    prep: list[str] = []
    out = [dict(st) for st in subtasks]
    valid = {_sub_id(st) for st in out if _sub_id(st)}
    for st in out:
        sid = _sub_id(st)
        st["depends_on"] = [
            str(d).strip()
            for d in (st.get("depends_on") or [])
            if str(d).strip() in valid and str(d).strip() != sid
        ]
    before = {_sub_id(st): list(st.get("depends_on") or []) for st in out}
    out = infer_subtask_dependencies(out, dataset=dataset)
    for st in out:
        sid = _sub_id(st)
        for dep in st.get("depends_on") or []:
            if dep not in before.get(sid, []):
                prep.append(f"infer_dep:{dep}->{sid}")
    if len(out) >= 2 and not any(st.get("depends_on") for st in out):
        ds = (dataset or "").strip().lower()
        if ds in DS.chain or ds == "math":
            for i in range(1, len(out)):
                out[i]["depends_on"] = [str(out[i - 1]["id"]).strip()]
            prep.append(f"infer_linear_chain:{ds}")
    return out, prep


def _plan_order(subtasks: list[dict[str, Any]]) -> dict[str, int]:
    order: dict[str, int] = {}
    for i, st in enumerate(subtasks):
        sid = _sub_id(st)
        if sid and sid not in order:
            order[sid] = i
    return order


def _add_nodes(g: nx.DiGraph, subtasks: list[dict[str, Any]]) -> None:
    for st in subtasks:
        sid = _sub_id(st)
        if not sid:
            continue
        tool = str(st.get("tool", "none")).strip().lower() or "none"
        needs = bool(st.get("needs_tool")) and tool != "none"
        g.add_node(
            sid,
            node_kind="subtask",
            step_type=str(st.get("step_type", "reason")),
            relation_type=str(st.get("relation_type", "entity_lookup")),
            answer_granularity=str(st.get("answer_granularity", "")),
            search_query=str(st.get("search_query", "")),
            tool=tool,
            is_tool=needs,
        )


def _add_edges(
    g: nx.DiGraph,
    subtasks: list[dict[str, Any]],
    order: dict[str, int],
    id_set: set[str],
    repair_log: list[str],
) -> None:
    for st in subtasks:
        sid = _sub_id(st)
        if sid not in id_set:
            continue
        for dep in st.get("depends_on") or []:
            p = str(dep).strip()
            if not p or p == sid:
                if p == sid:
                    repair_log.append(f"self_loop:{sid}")
                continue
            if p not in id_set:
                repair_log.append(f"unknown_parent:{p}->{sid}")
                continue
            if order[p] >= order[sid]:
                repair_log.append(f"drop_non_forward_dep:{p}->{sid}")
                continue
            if nx.has_path(g, sid, p):
                repair_log.append(f"skip_back_edge:{p}->{sid}")
                continue
            g.add_edge(p, sid, edge_kind="dependency")


def _drop_cycles(g: nx.DiGraph, ids: list[str], repair_log: list[str]) -> None:
    sub = g.subgraph(ids).copy()
    while not nx.is_directed_acyclic_graph(sub):
        try:
            u, v, _ = nx.find_cycle(sub, orientation="original")[0]
        except nx.NetworkXNoCycle:
            break
        sub.remove_edge(u, v)
        if g.has_edge(u, v):
            g.remove_edge(u, v)
        repair_log.append(f"drop_cycle_edge:{u}->{v}")


def _chain_fallback(
    g: nx.DiGraph,
    subtasks: list[dict[str, Any]],
    ids: list[str],
    dataset: str,
    repair_log: list[str],
) -> None:
    if dataset not in DS.chain:
        _drop_cycles(g, ids, repair_log)
        return
    id_set = set(ids)
    ordered = [_sub_id(st) for st in subtasks if _sub_id(st) in id_set]
    if len(ordered) < 2:
        return
    for u in ids:
        for v in ids:
            if g.has_edge(u, v):
                g.remove_edge(u, v)
    for u, v in zip(ordered, ordered[1:]):
        g.add_edge(u, v, edge_kind="dependency")
    repair_log.append(f"chain_fallback:{dataset}")


def _add_virtual(g: nx.DiGraph, ids: list[str], id_set: set[str] | None = None) -> None:
    g.add_node(GRAPH.start, node_kind="start")
    g.add_node(GRAPH.end, node_kind="end")
    id_set = id_set if id_set is not None else set(ids)
    for sid in ids:
        if sid not in g:
            continue
        if not any(p in id_set for p in g.predecessors(sid)):
            g.add_edge(GRAPH.start, sid, edge_kind="start")
        if not any(s in id_set for s in g.successors(sid)):
            g.add_edge(sid, GRAPH.end, edge_kind="end")
    if not any(g.successors(GRAPH.start)):
        g.add_edge(GRAPH.start, GRAPH.end, edge_kind="virtual")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="QCE graph features from plans JSONL")
    p.add_argument(
        "--plans",
        type=Path,
        default=Path("datasets/decomposer_cache/qce_train_plans_pilot10.jsonl"),
    )
    args = p.parse_args()
    df = features_dataframe(build_graphs_from_plans(load_plans_jsonl(args.plans)))
    norm = fit_train_norm(df)
    df = rescore_dataframe(df, norm)
    print(f"rows={len(df)} norm={norm}")
    print(df[["training_id", "dataset", "n_nodes", "max_depth", "complexity_graph", *DIMS.main]].to_string())
