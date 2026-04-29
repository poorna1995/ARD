from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from difficulty.feature_measure import TaskComplexityAnalyzer
from prompts.prompts import DATASETS
from routing.router import Router

DEFAULT_OUTPUT_ROOT = Path("results1/unified_baseline")
COMPACT_FIELDS = (
    "answer",
    "latency_total",
    "latency_llm",
    "latency_tools",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "cost_usd",
    "num_llm_calls",
    "num_steps",
    "num_tool_calls",
    "is_correct",
    "error",
    "is_failed",
)


def load_dataset(dataset: str, path: str | Path | None = None) -> pd.DataFrame:
    path = Path(path) if path is not None else Path("datasets/golden") / f"{dataset}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    # Read minimal columns only (memory optimization).
    try:
        return pd.read_parquet(path, columns=["query", "answer"])
    except Exception:
        return pd.read_parquet(path, columns=["query"])


def _response_to_dict(response: Any, compact: bool) -> dict[str, Any]:
    if hasattr(response, "__dataclass_fields__"):
        payload = asdict(response)
    elif isinstance(response, dict):
        payload = response
    else:
        payload = {"multiagent_response": str(response)}
    if not compact:
        return payload
    return {k: payload.get(k) for k in COMPACT_FIELDS}


def run_pipeline(
    dataset: str = "gaia",
    data_path: str | Path | None = None,
    output_path: str | Path | None = None,
    jsonl_output_path: str | Path | None = None,
    execute_agents: bool = True,
    assigned_multiagent_only: bool = True,
    limit: int | None = None,
    compact_response: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    if dataset not in DATASETS:
        raise ValueError(f"Unsupported dataset '{dataset}'. Expected one of: {DATASETS}")

    df = load_dataset(dataset=dataset, path=data_path)
    if "query" not in df.columns:
        raise KeyError(f"Missing required 'query' column in dataset '{dataset}'.")

    if limit is not None:
        df = df.head(limit)

    queries = df["query"].fillna("").astype(str).tolist()
    expected_answers = df["answer"].tolist() if "answer" in df.columns else [None] * len(df)

    analyzer = TaskComplexityAnalyzer()
    analyzer.fit(queries)

    records: list[dict[str, Any]] = []
    append_record = records.append
    total = len(queries)
    for idx, (query, expected) in enumerate(zip(queries, expected_answers), start=1):
        if verbose:
            q_preview = query[:100].replace("\n", " ")
            print(f"[{idx}/{total}] Routing query: {q_preview}")

        router = Router(query=query, analyzer=analyzer, dataset=dataset)
        routed = router.inspect()
        if verbose:
            print(
                f"[{idx}/{total}] -> assigned_agent={routed.get('assigned_agent')} "
                f"overall={routed.get('overall')}"
            )

        record: dict[str, Any] = {
            "dataset": dataset,
            "query": query,
            "expected_answer": expected,
            **routed,
        }
        if execute_agents:
            if assigned_multiagent_only and routed.get("assigned_agent") != "multiagent":
                if verbose:
                    print(
                        f"[{idx}/{total}] Skipping execution in multiagent-only mode "
                        f"(assigned_agent={routed.get('assigned_agent')})"
                    )
                append_record(record)
                continue
            if verbose:
                print(f"[{idx}/{total}] Executing agent...")
            try:
                run_result = router.run(expected_answer=expected)
                response = _response_to_dict(run_result.get("response"), compact=compact_response)
                record.update(
                    {
                        "run_agent": run_result.get("agent"),
                        "run_model": run_result.get("model"),
                        "run_overall": run_result.get("overall"),
                        **{f"response_{k}": v for k, v in response.items()},
                    }
                )
                if verbose:
                    print(
                        f"[{idx}/{total}] Done -> run_agent={run_result.get('agent')} "
                        f"model={run_result.get('model')} "
                        f"failed={response.get('is_failed')}"
                    )
                    print(
                        f"[{idx}/{total}] Answers -> final={response.get('answer')} "
                        f"expected={expected}"
                    )
            except Exception as exc:
                record.update(
                    {
                        "run_agent": routed.get("assigned_agent"),
                        "run_model": routed.get("model_primary"),
                        "response_error": str(exc),
                        "response_is_failed": True,
                    }
                )
                if verbose:
                    print(f"[{idx}/{total}] Failed -> error={exc}")
        elif verbose:
            print(f"[{idx}/{total}] Route-only mode; skipped execution.")
        append_record(record)

    results_df = pd.DataFrame.from_records(records)
    output = Path(output_path) if output_path is not None else DEFAULT_OUTPUT_ROOT / dataset / "orchestrator_pipeline.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    if "overall" not in results_df.columns:
        raise RuntimeError("Missing 'overall' in results; routing output is incomplete.")
    results_df.to_parquet(output, index=False)
    print(f"Saved pipeline results to: {output}")

    jsonl_output = (
        Path(jsonl_output_path)
        if jsonl_output_path is not None
        else output.with_suffix(".jsonl")
    )
    jsonl_output.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_output.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
    print(f"Saved per-query JSONL to: {jsonl_output}")

    return results_df


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Module-level orchestrator pipeline.")
    parser.add_argument("--dataset", choices=DATASETS, default="gaia")
    parser.add_argument("--data_path", default=None, help="Optional custom parquet path.")
    parser.add_argument("--output_path", default=None, help="Where to save pipeline output parquet.")
    parser.add_argument("--jsonl_output_path", default=None, help="Where to save per-query JSONL.")
    parser.add_argument(
        "--route_only",
        action="store_true",
        help="Only route tasks; skip agent execution.",
    )
    parser.add_argument(
        "--assigned_multiagent_only",
        action="store_true",
        help="Execute only queries assigned to multiagent; skip all others.",
    )
    parser.add_argument(
        "--all_assigned_agents",
        action="store_true",
        help="Override multiagent-only mode and execute whichever agent is assigned.",
    )
    parser.add_argument(
        "--full_response",
        action="store_true",
        help="Store full agent response payload (higher memory and disk usage).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit.")
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable per-query progress logs.",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    run_pipeline(
        dataset=args.dataset,
        data_path=args.data_path,
        output_path=args.output_path,
        jsonl_output_path=args.jsonl_output_path,
        execute_agents=not args.route_only,
        assigned_multiagent_only=(not args.all_assigned_agents),
        limit=args.limit,
        compact_response=not args.full_response,
        verbose=not args.quiet,
    )
