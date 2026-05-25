"""LLM query decomposition for QCE — procedure DAG plans (cached JSONL)."""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

from agent.base import COST_PER_1M
from agent.dataset_profile import coerce_hop_name, coerce_positive_int, validate_planner_subtasks
from evaluator.parse import extract_last_json_dict
from prompts.qce_decompose import (
    normalize_answer_granularity,
    normalize_final_constraint,
    normalize_relation_type,
    plan_msgs,
)
from qce.query_input import QueryIn, resolve_row

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECOMPOSE_MODEL = "gpt-4o-mini"
DEFAULT_CACHE_PATH = REPO_ROOT / "datasets/decomposer_cache/qce_train_plans.jsonl"

PLAN_OK = "ok"
PLAN_PARSE_FAIL = "parse_fail"

_VALID_TOOLS = frozenset(
    {
        "none",
        "web_search",
        "wikipedia",
        "math_tool",
        "python_exec",
        "read_file",
        "arxiv_search",
        "github_search",
        "pdb_parse",
        "retrieve_tool",
    }
)
_TOOL_ALIASES = {
    "math": "math_tool",
    "wikipedia_search": "wikipedia",
    "python": "python_exec",
    "code": "python_exec",
    "retrieve": "retrieve_tool",
}
_VALID_STEP_TYPES = frozenset({"retrieve", "reason", "compute", "verify", "select"})
_SUBTASK_BANDS: dict[str, tuple[int, int]] = {
    "hotpot": (2, 2),
    "musique": (2, 5),
    "math": (2, 4),
    "mmlu_pro": (3, 3),
    "gaia": (2, 4),
}
_ANSWER_IN_GOAL = re.compile(
    r"(?i)\b(the answer is|final answer|therefore the answer|answer:\s*\S)"
)
_MAX_SEARCH_LEN = 120

_client: OpenAI | None = None


def _openai() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    return _client


def _cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    rates = COST_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    return (
        prompt_tokens * rates["input"] + completion_tokens * rates["output"]
    ) / 1_000_000


def _normalize_subtasks(raw: list[Any], *, max_n: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[str] = set()
    for i, item in enumerate(raw[:max_n], start=1):
        if not isinstance(item, dict):
            continue
        sid = str(item.get("id", f"t{i}")).strip() or f"t{i}"
        if sid in seen:
            sid = f"{sid}_{i}"
        seen.add(sid)

        raw_tool = str(item.get("tool", "none")).strip().lower() or "none"
        tool = _TOOL_ALIASES.get(raw_tool, raw_tool)
        if tool not in _VALID_TOOLS:
            tool = "none"

        step_type = str(item.get("step_type", "reason")).strip().lower()
        if step_type not in _VALID_STEP_TYPES:
            step_type = "reason"

        needs_tool = bool(item.get("needs_tool", tool != "none"))
        if tool == "none":
            needs_tool = False

        goal = str(item.get("goal", "")).strip()
        if not goal:
            continue

        deps = item.get("depends_on", [])
        if not isinstance(deps, list):
            deps = []

        out.append(
            {
                "id": sid,
                "goal": goal,
                "relation_type": normalize_relation_type(item.get("relation_type")),
                "answer_granularity": normalize_answer_granularity(
                    item.get("answer_granularity")
                ),
                "step_type": step_type,
                "needs_tool": needs_tool,
                "tool": tool,
                "search_query": str(item.get("search_query", "")).strip(),
                "depends_on": [str(d).strip() for d in deps if str(d).strip()],
            }
        )

    ids = {st["id"] for st in out}
    for st in out:
        st["depends_on"] = [d for d in st["depends_on"] if d in ids and d != st["id"]]
    return out


def _has_cycle(subtasks: list[dict[str, Any]]) -> bool:
    ids = {st["id"] for st in subtasks}
    adj = {i: [] for i in ids}
    for st in subtasks:
        for dep in st.get("depends_on") or []:
            if dep in adj:
                adj[dep].append(st["id"])
    visiting: set[str] = set()
    done: set[str] = set()

    def dfs(node: str) -> bool:
        if node in done:
            return False
        if node in visiting:
            return True
        visiting.add(node)
        for child in adj.get(node, []):
            if dfs(child):
                return True
        visiting.remove(node)
        done.add(node)
        return False

    return any(dfs(n) for n in ids)


def _plan_warnings(
    *,
    dataset: str,
    query: str,
    subtasks: list[dict[str, Any]],
    metadata: dict[str, str | int | float],
) -> list[str]:
    warns: list[str] = []
    lo, hi = _SUBTASK_BANDS.get(dataset, (1, 6))
    n = len(subtasks)
    if n < lo or n > hi:
        warns.append(f"subtask_count:{n}_outside_{lo}_{hi}")

    id_set = {st["id"] for st in subtasks}
    for st in subtasks:
        sid = st["id"]
        for dep in st.get("depends_on") or []:
            if dep not in id_set:
                warns.append(f"unknown_dep:{dep}->{sid}")
            elif dep == sid:
                warns.append(f"self_dep:{sid}")

    if subtasks and _has_cycle(subtasks):
        warns.append("depends_on_cycle")

    q_len = max(len(query.strip()), 1)
    for st in subtasks:
        if _ANSWER_IN_GOAL.search(st["goal"]):
            warns.append(f"answer_like_goal:{st['id']}")
        if st["step_type"] == "retrieve" and st["needs_tool"]:
            sq = st["search_query"]
            if not sq:
                warns.append(f"empty_search_query:{st['id']}")
            elif len(sq) > _MAX_SEARCH_LEN:
                st["search_query"] = sq[:_MAX_SEARCH_LEN]
                warns.append(f"truncated_search_query:{st['id']}")

    if dataset == "gaia":
        attachment = str(metadata.get("file_name") or metadata.get("attachment") or "").strip()
        if attachment and subtasks and subtasks[0].get("tool") != "read_file":
            warns.append("gaia_file_not_first_step")
    if dataset == "mmlu_pro" and any(st.get("needs_tool") for st in subtasks):
        warns.append("mmlu_pro_unexpected_tool")

    return warns


def _repair_deps(
    subtasks: list[dict[str, Any]],
    *,
    dataset: str,
    metadata: dict[str, str | int | float],
) -> list[dict[str, Any]]:
    n_hops = coerce_positive_int(metadata.get("n_hops"))
    hop_name = coerce_hop_name(metadata.get("hop_name"))
    return validate_planner_subtasks(
        subtasks,
        dataset=dataset,
        n_hops=n_hops,
        hop_name=hop_name,
    )


def _llm_json(
    messages: list[dict[str, str]],
    *,
    model: str,
    temperature: float,
    max_tokens: int,
) -> tuple[dict[str, Any], float, int, int, str]:
    t0 = time.perf_counter()
    resp = _openai().chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    latency = time.perf_counter() - t0
    raw = resp.choices[0].message.content or ""
    usage = resp.usage
    pt = int(usage.prompt_tokens or 0)
    ct = int(usage.completion_tokens or 0)
    parsed = extract_last_json_dict(raw) or {}
    return parsed, latency, pt, ct, raw


def decompose_query(
    query: str,
    dataset: str,
    *,
    model: str = DEFAULT_DECOMPOSE_MODEL,
    metadata: dict[str, str | int | float] | None = None,
    training_id: str | None = None,
    query_in: QueryIn | None = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    """Decompose one question into a JSON-serializable plan record."""
    qin = query_in or QueryIn(
        training_id=training_id,
        dataset=dataset,
        query_text=query,
        metadata=metadata or {},
    )
    meta = qin.metadata
    messages = plan_msgs(qin.query_text, qin.dataset, metadata=meta or None)

    parsed: dict[str, Any] = {}
    latency = 0.0
    pt = ct = 0
    for _ in range(2):
        parsed, latency, pt, ct, _raw = _llm_json(
            messages, model=model, temperature=temperature, max_tokens=max_tokens
        )
        if parsed.get("subtasks") is not None:
            break

    status = PLAN_OK if parsed.get("subtasks") else PLAN_PARSE_FAIL
    _, hi = _SUBTASK_BANDS.get(qin.dataset, (1, 6))
    subtasks = _normalize_subtasks(parsed.get("subtasks") or [], max_n=hi)
    if subtasks:
        subtasks = _repair_deps(subtasks, dataset=qin.dataset, metadata=meta)

    warns = _plan_warnings(
        dataset=qin.dataset,
        query=qin.query_text,
        subtasks=subtasks,
        metadata=meta,
    )

    return {
        "training_id": qin.training_id,
        "dataset": qin.dataset,
        "query": qin.query_text,
        "final_constraint": normalize_final_constraint(parsed.get("final_constraint", "")),
        "subtasks": subtasks,
        "plan_status": status,
        "plan_warns": warns,
        "model": model,
        "latency_sec": round(latency, 4),
        "prompt_tokens": pt,
        "completion_tokens": ct,
        "cost_usd": round(_cost_usd(model, pt, ct), 8),
    }


def _cache_record(row: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    """JSONL row: ids + plan fields graph stage needs."""
    return {
        "training_id": plan.get("training_id") or row.get("training_id"),
        "dataset": plan["dataset"],
        "query": plan["query"],
        "final_constraint": plan["final_constraint"],
        "subtasks": plan["subtasks"],
        "plan_status": plan["plan_status"],
        "plan_warns": plan["plan_warns"],
        "model": plan["model"],
        "latency_sec": plan["latency_sec"],
        "prompt_tokens": plan["prompt_tokens"],
        "completion_tokens": plan["completion_tokens"],
        "cost_usd": plan["cost_usd"],
    }


def load_plans_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.is_file():
        return []
    plans: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                plans.append(json.loads(line))
    return plans


def write_plans_jsonl(path: str | Path, records: list[dict[str, Any]]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def plan_summary(plans: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate stats for logging after a batch run."""
    if not plans:
        return {"n": 0}
    n = len(plans)
    n_ok = sum(
        1
        for p in plans
        if p.get("plan_status") == PLAN_OK and (p.get("subtasks") or [])
    )
    n_warn = sum(1 for p in plans if p.get("plan_warns"))
    cost = sum(float(p.get("cost_usd") or 0) for p in plans)
    by_ds: dict[str, dict[str, int]] = {}
    for p in plans:
        ds = str(p.get("dataset", ""))
        bucket = by_ds.setdefault(ds, {"n": 0, "ok": 0, "warn": 0})
        bucket["n"] += 1
        if p.get("plan_status") == PLAN_OK and p.get("subtasks"):
            bucket["ok"] += 1
        if p.get("plan_warns"):
            bucket["warn"] += 1
    return {
        "n": n,
        "n_ok": n_ok,
        "ok_rate": round(n_ok / n, 4),
        "n_with_warnings": n_warn,
        "total_cost_usd": round(cost, 4),
        "by_dataset": by_ds,
    }


def decompose_batch(
    rows: list[dict[str, Any]],
    *,
    model: str = DEFAULT_DECOMPOSE_MODEL,
    cache_path: Path | None = DEFAULT_CACHE_PATH,
    id_key: str = "training_id",
    query_key: str = "query",
    dataset_key: str = "dataset",
    force_refresh: bool = False,
    sleep_sec: float = 0.0,
    limit: int = 0,
    log_every: int = 10,
) -> list[dict[str, Any]]:
    """
    Decompose many rows. Uses JSONL cache: skip cached ids unless ``force_refresh``.

    ``force_refresh=True`` deletes the cache file first, then writes only this run's rows.
    ``limit>0`` processes only the first N rows (after cache skip).
    """
    log = logging.getLogger(__name__)
    if limit > 0:
        rows = rows[:limit]

    cached: dict[str, dict[str, Any]] = {}
    if cache_path and cache_path.is_file() and not force_refresh:
        for rec in load_plans_jsonl(cache_path):
            cached[str(rec.get(id_key, ""))] = rec

    results: list[dict[str, Any]] = []
    to_write: list[dict[str, Any]] = []
    cache_path = Path(cache_path) if cache_path else None
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
    if force_refresh and cache_path and cache_path.is_file():
        cache_path.unlink()
        cached.clear()

    n_total = len(rows)
    n_api = 0
    log.info(
        "decompose n=%d model=%s cache=%s cached=%d force_refresh=%s",
        n_total,
        model,
        cache_path,
        len(cached),
        force_refresh,
    )

    for i, row in enumerate(rows, start=1):
        tid = str(row[id_key])
        if not force_refresh and tid in cached:
            results.append(cached[tid])
            continue

        qin = resolve_row(row, id_key=id_key, query_key=query_key, dataset_key=dataset_key)
        try:
            plan = decompose_query(
                qin.query_text,
                qin.dataset,
                model=model,
                query_in=qin,
            )
        except Exception as exc:
            log.error("[%d/%d] %s failed: %s", i, n_total, tid, exc)
            raise

        rec = _cache_record(row, plan)
        results.append(rec)
        to_write.append(rec)
        cached[tid] = rec
        n_api += 1
        if log_every > 0 and (n_api == 1 or n_api % log_every == 0):
            log.info(
                "[%d/%d] %s ok=%s (api_calls=%d)",
                i,
                n_total,
                tid,
                plan.get("plan_status"),
                n_api,
            )
        if sleep_sec > 0:
            time.sleep(sleep_sec)

    if cache_path and to_write:
        mode = "a"
        with cache_path.open(mode, encoding="utf-8") as f:
            for rec in to_write:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        log.info("wrote %d new rows -> %s", len(to_write), cache_path.resolve())

    return results


def _main() -> None:
    import argparse

    from qce.corpus_io import load_corpus

    parser = argparse.ArgumentParser(description="QCE decomposition batch (cached JSONL)")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=REPO_ROOT / "datasets/train_samples/v1/qce_train.csv",
    )
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE_PATH)
    parser.add_argument("--model", default=DEFAULT_DECOMPOSE_MODEL)
    parser.add_argument("--force-refresh", action="store_true")
    parser.add_argument("--sleep-sec", type=float, default=0.5)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--labeled-only", action="store_true", default=True)
    parser.add_argument(
        "--no-labeled-only",
        action="store_false",
        dest="labeled_only",
        help="Include all rows regardless of label_status",
    )
    args = parser.parse_args()

    corpus = load_corpus(args.corpus, labeled_only=args.labeled_only)
    rows = corpus.to_dict(orient="records")
    if args.limit > 0:
        rows = rows[: args.limit]

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    plans = decompose_batch(
        rows,
        model=args.model,
        cache_path=args.cache,
        force_refresh=args.force_refresh,
        sleep_sec=args.sleep_sec,
        limit=args.limit,
    )
    summary = plan_summary(plans)
    print(json.dumps(summary, indent=2))
    print(f"cache -> {args.cache.resolve()}")


if __name__ == "__main__":
    _main()
