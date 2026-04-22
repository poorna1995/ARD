from __future__ import annotations

import argparse
import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.loaders import LOADER_REGISTRY, get_loader
from baselines.prompt_resolver import resolve_prompts

try:
    from routellm.controller import Controller
except ImportError as exc:
    raise ImportError(
        "Could not import RouteLLM Controller. Install it first, e.g. `pip install routellm`."
    ) from exc


DEFAULT_DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
DEFAULT_RESULTS_ROOT = REPO_ROOT / "results1" / "unified_baseline"

PRICING = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
}

DATASET_ALIASES = {
    "swe_bench_verified": "swe_bench",
}
PROMPT_DATASET_ALIASES = {
    "swe_bench": "swe_bench_verified",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run RouteLLM evaluation across one or more datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets to run (default: all datasets from configs/datasets.yaml).",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=None,
        help="Optional cap per dataset, applied after load.",
    )
    parser.add_argument(
        "--router",
        default="mf",
        help="RouteLLM router name (default: mf).",
    )
    parser.add_argument(
        "--strong-model",
        default="gpt-4o",
        help="Strong model for Controller.",
    )
    parser.add_argument(
        "--weak-model",
        default="gpt-4o-mini",
        help="Weak model for Controller.",
    )
    parser.add_argument(
        "--target-strong-pct",
        type=float,
        default=0.25,
        help="Desired fraction of queries routed to strong model if threshold not fixed.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed routing threshold. If omitted, computed from target-strong-pct per dataset.",
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Root directory to save outputs.",
    )
    parser.add_argument(
        "--force-redownload",
        action="store_true",
        help="Force dataset loaders to redownload/process source data.",
    )
    parser.add_argument(
        "--mmlu-categories",
        nargs="+",
        default=None,
        metavar="NAME",
        help=(
            "Only run these MMLU-Pro categories (case-insensitive). "
            "Example: --mmlu-categories biology chemistry"
        ),
    )
    parser.add_argument(
        "--list-mmlu-categories",
        action="store_true",
        help="List available MMLU-Pro categories and exit.",
    )
    return parser.parse_args()


def normalize_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return str(value).strip()


def get_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    key = "gpt-4o-mini" if "mini" in model else "gpt-4o"
    price = PRICING[key]
    return (input_tokens * price["input"] + output_tokens * price["output"]) / 1000.0


def resolve_requested_datasets(requested: list[str] | None, config_path: Path) -> list[str]:
    with config_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}

    configured = list(config.keys())
    if not requested:
        return configured

    out: list[str] = []
    valid = set(configured) | set(DATASET_ALIASES.keys())
    for ds in requested:
        if ds not in valid:
            raise ValueError(
                f"Unknown dataset '{ds}'. Allowed values: {sorted(valid)}"
            )
        out.append(ds)
    return out


def load_dataset_configs(config_path: Path) -> dict[str, dict]:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def build_router_scores(
    df: pd.DataFrame,
    client: Controller,
    router_name: str,
    threshold: float,
) -> pd.DataFrame:
    working = df.copy().reset_index(drop=True)
    working["query"] = working["query"].map(normalize_text)
    working = working[working["query"] != ""].copy().reset_index(drop=True)

    scores = [
        float(client.routers[router_name].calculate_strong_win_rate(q))
        for q in working["query"].tolist()
    ]
    scores_df = pd.DataFrame(
        {
            "id": working["id"].astype(str),
            "query": working["query"],
            "strong_win_rate": scores,
            "threshold": float(threshold),
        }
    )
    scores_df["route_to_strong"] = scores_df["strong_win_rate"] >= float(threshold)
    return scores_df


def run_routed_inference(
    dataset: str,
    base_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    client: Controller,
    router_name: str,
    threshold: float,
    target_strong_pct: float,
) -> pd.DataFrame:
    query_to_score = dict(zip(scores_df["id"], scores_df["strong_win_rate"]))
    results: list[dict[str, object]] = []
    router_model_name = f"router-{router_name}-{threshold:.5f}"

    for _, row in base_df.iterrows():
        query_id = str(row["id"])
        query = normalize_text(row["query"])
        if not query:
            continue
        system_prompt, user_prompt = build_vanilla_prompts(dataset=dataset, row=row)

        start_time = time.time()
        response = client.chat.completions.create(
            model=router_model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        latency = time.time() - start_time

        input_tokens = int(response.usage.prompt_tokens)
        output_tokens = int(response.usage.completion_tokens)
        total_tokens = int(response.usage.total_tokens)
        predicted = normalize_text(response.choices[0].message.content)
        model_used = str(response.model)
        cost = get_cost(model_used, input_tokens, output_tokens)

        results.append(
            {
                "id": query_id,
                "query": query,
                "model_used": model_used,
                "predicted": predicted,
                "score": float(query_to_score.get(query_id, np.nan)),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "cost_usd": round(cost, 6),
                "latency_sec": round(latency, 3),
            }
        )

        print(
            f"id={query_id} target_strong={target_strong_pct * 100:.1f}% "
            f"threshold={threshold:.5f} score={results[-1]['score']:.3f} "
            f"model={model_used} predicted={predicted} "
            f"tokens={total_tokens} cost=${cost:.5f} latency={latency:.2f}s"
        )

    return pd.DataFrame(results)


def build_metrics(
    dataset: str,
    merged: pd.DataFrame,
    router_name: str,
    threshold: float,
    strong_model: str,
    weak_model: str,
) -> dict[str, object]:
    total_queries = int(len(merged))
    correct_queries = int(merged["correct"].sum())
    accuracy = float(merged["correct"].mean()) if total_queries else 0.0

    total_prompt_tokens = int(merged["input_tokens"].sum()) if total_queries else 0
    total_completion_tokens = int(merged["output_tokens"].sum()) if total_queries else 0
    total_tokens = int(merged["total_tokens"].sum()) if total_queries else 0
    avg_tokens_per_query = round(total_tokens / total_queries, 3) if total_queries else 0.0

    total_cost_usd = round(float(merged["cost_usd"].sum()), 6) if total_queries else 0.0
    strong_mask = ~merged["model_used"].str.contains("mini", na=False)
    weak_mask = merged["model_used"].str.contains("mini", na=False)
    strong_cost = round(float(merged.loc[strong_mask, "cost_usd"].sum()), 6) if total_queries else 0.0
    weak_cost = round(float(merged.loc[weak_mask, "cost_usd"].sum()), 6) if total_queries else 0.0

    avg_latency_s = round(float(merged["latency_sec"].mean()), 3) if total_queries else 0.0
    p50_latency_s = round(float(merged["latency_sec"].quantile(0.50)), 3) if total_queries else 0.0
    p95_latency_s = round(float(merged["latency_sec"].quantile(0.95)), 3) if total_queries else 0.0

    metadata: dict[str, object] = {}
    if "level" in merged.columns and merged["level"].notna().any():
        per_level: dict[str, dict[str, object]] = {}
        for level, level_df in merged.groupby("level", dropna=True):
            per_level[str(level)] = {
                "total": int(len(level_df)),
                "correct": int(level_df["correct"].sum()),
                "accuracy": round(float(level_df["correct"].mean()), 6),
            }
        metadata["level_metrics"] = per_level

    return {
        "dataset": dataset,
        "modality": f"routellm_{router_name}",
        "strong_model": strong_model,
        "weak_model": weak_model,
        "router": router_name,
        "threshold": float(threshold),
        "system_prompt_version": "v1",
        "total_queries": total_queries,
        "correct_queries": correct_queries,
        "accuracy": round(accuracy, 6),
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "total_tokens": total_tokens,
        "avg_tokens_per_query": avg_tokens_per_query,
        "total_cost_usd": total_cost_usd,
        "strong_model_cost_usd": strong_cost,
        "weak_model_cost_usd": weak_cost,
        "avg_latency_s": avg_latency_s,
        "p50_latency_s": p50_latency_s,
        "p95_latency_s": p95_latency_s,
        "error_count": int((merged["predicted"] == "NO ACCESS").sum()) if total_queries else 0,
        "error_rate": round(float((merged["predicted"] == "NO ACCESS").mean()), 6) if total_queries else 0.0,
        "routing_split": {
            "strong_count": int(strong_mask.sum()) if total_queries else 0,
            "weak_count": int(weak_mask.sum()) if total_queries else 0,
            "strong_pct": round(float(strong_mask.mean() * 100), 2) if total_queries else 0.0,
            "weak_pct": round(float(weak_mask.mean() * 100), 2) if total_queries else 0.0,
        },
        "metadata": metadata,
    }


def resolve_loader_name(dataset: str) -> str:
    return DATASET_ALIASES.get(dataset, dataset)


def resolve_prompt_dataset_name(dataset: str) -> str:
    return PROMPT_DATASET_ALIASES.get(dataset, dataset)


def _ground_truth_hint(dataset: str) -> str:
    if dataset == "mmlu_pro":
        return "gold is one letter A-J; output that letter only"
    if dataset == "math_hard":
        return "gold is LaTeX/math final; match that form"
    if dataset in ("swe_bench", "swe_bench_verified"):
        return "gold is patch/diff text; output minimal patch"
    return "gold is short factual (word, number, name, date); match exactly"


def _format_options_for_prompt(raw_options: object) -> str:
    if isinstance(raw_options, (list, tuple)):
        return "\n".join(f"{chr(65 + i)}. {str(opt)}" for i, opt in enumerate(raw_options))
    if isinstance(raw_options, np.ndarray):
        as_list = raw_options.tolist()
        if isinstance(as_list, list):
            return "\n".join(f"{chr(65 + i)}. {str(opt)}" for i, opt in enumerate(as_list))
    if isinstance(raw_options, str):
        stripped = raw_options.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return "\n".join(
                        f"{chr(65 + i)}. {str(opt)}" for i, opt in enumerate(parsed)
                    )
            except Exception:
                pass
    return normalize_text(raw_options)


def build_vanilla_prompts(dataset: str, row: pd.Series) -> tuple[str, str]:
    prompt_dataset = resolve_prompt_dataset_name(dataset)
    options_text = _format_options_for_prompt(row.get("options"))
    if prompt_dataset == "mmlu_pro" and not options_text:
        raise ValueError(
            "mmlu_pro row is missing 'options'. MMLU prompts require query + options."
        )
    user_values = {
        "dataset": prompt_dataset,
        "query": normalize_text(row["query"]),
        "options": options_text,
        "ground_truth_format_hint": _ground_truth_hint(dataset),
    }
    resolved = resolve_prompts(
        modality="vanilla",
        dataset=prompt_dataset,
        model_name="routellm",
        system_values={
            "output_contract": "Return only the final answer in required format.",
        },
        user_values=user_values,
    )
    return resolved.system_prompt, resolved.user_prompt


def _pick_mmlu_category_column(df: pd.DataFrame) -> str | None:
    for col in ("category", "subject"):
        if col in df.columns:
            return col
    return None


def _list_mmlu_categories(df: pd.DataFrame) -> None:
    col = _pick_mmlu_category_column(df)
    if col is None:
        raise ValueError("mmlu_pro dataset has no category/subject column.")
    vc = df[col].astype(str).str.strip().value_counts().sort_index()
    print(f"[mmlu_pro] category column: {col}")
    for name, count in vc.items():
        print(f"  - {name}: {count}")


def _filter_mmlu_categories(df: pd.DataFrame, categories: list[str]) -> pd.DataFrame:
    col = _pick_mmlu_category_column(df)
    if col is None:
        raise ValueError(
            "mmlu_pro dataset has no category/subject column; cannot filter by category."
        )

    normalized_requested = {
        str(c).strip().lower() for c in categories if str(c).strip()
    }
    if not normalized_requested:
        return df

    normalized_present = (
        df[col].astype(str).str.strip().str.lower()
    )
    available = sorted(set(normalized_present.tolist()))
    missing = sorted(normalized_requested.difference(available))
    if missing:
        raise ValueError(
            f"Requested MMLU categories not found: {missing}. "
            f"Available categories: {available}"
        )

    filtered = df[normalized_present.isin(normalized_requested)].copy()
    if filtered.empty:
        raise ValueError("No rows left after MMLU category filtering.")
    return filtered.reset_index(drop=True)


def run_dataset(
    dataset: str,
    dataset_config: dict,
    args: argparse.Namespace,
    client: Controller,
) -> None:
    loader_name = resolve_loader_name(dataset)
    if loader_name not in LOADER_REGISTRY:
        raise ValueError(
            f"No loader registered for '{dataset}' (resolved to '{loader_name}')."
        )

    loader = get_loader(loader_name, config=dataset_config, data_root="datasets")
    df = loader.load(force_redownload=args.force_redownload).copy()

    required = {"id", "query", "answer"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{dataset}: missing required columns: {sorted(missing)}")

    if args.sample_limit is not None:
        df = df.iloc[: args.sample_limit].copy()

    if dataset == "mmlu_pro" and args.mmlu_categories:
        df = _filter_mmlu_categories(df, args.mmlu_categories)

    df["id"] = df["id"].astype(str)
    df["query"] = df["query"].map(normalize_text)
    df["answer"] = df["answer"].map(normalize_text)
    df = df[df["query"] != ""].copy().reset_index(drop=True)

    if df.empty:
        print(f"[{dataset}] No rows with non-empty query. Skipping.")
        return

    bootstrap_scores = [
        float(client.routers[args.router].calculate_strong_win_rate(q))
        for q in df["query"].tolist()
    ]
    threshold = (
        float(args.threshold)
        if args.threshold is not None
        else float(np.percentile(bootstrap_scores, 100 - (args.target_strong_pct * 100)))
    )
    print(
        f"[{dataset}] rows={len(df)} threshold={threshold:.5f} "
        f"(target strong={args.target_strong_pct * 100:.1f}%)"
    )

    scores_df = build_router_scores(
        df=df,
        client=client,
        router_name=args.router,
        threshold=threshold,
    )
    results_df = run_routed_inference(
        dataset=dataset,
        base_df=df,
        scores_df=scores_df,
        client=client,
        router_name=args.router,
        threshold=threshold,
        target_strong_pct=args.target_strong_pct,
    )

    keep_meta = [c for c in ("level", "category", "type", "repo") if c in df.columns]
    base_cols = ["id", "query", "answer"] + keep_meta
    merged = results_df.merge(df[base_cols], on=["id", "query"], how="left")
    merged = merged.rename(columns={"answer": "ground_truth"})
    merged["correct"] = (
        merged["predicted"].map(normalize_text).str.lower()
        == merged["ground_truth"].map(normalize_text).str.lower()
    )

    ordered_columns = (
        ["id", "query"] + keep_meta + [
            "ground_truth",
            "predicted",
            "correct",
            "model_used",
            "score",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "cost_usd",
            "latency_sec",
        ]
    )
    merged = merged[ordered_columns]

    out_dir = args.results_root / dataset / "routellm"
    out_dir.mkdir(parents=True, exist_ok=True)

    scores_df.to_parquet(out_dir / "routellm_scores.parquet", index=False)
    merged.to_parquet(out_dir / "routellm_final.parquet", index=False)
    merged.to_csv(out_dir / "routellm_final.csv", index=False)

    metrics = build_metrics(
        dataset=dataset,
        merged=merged,
        router_name=args.router,
        threshold=threshold,
        strong_model=args.strong_model,
        weak_model=args.weak_model,
    )
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"[{dataset}] Saved outputs to {out_dir}")


def main() -> None:
    args = parse_args()
    dataset_configs = load_dataset_configs(DEFAULT_DATASETS_CONFIG)
    datasets = resolve_requested_datasets(args.datasets, DEFAULT_DATASETS_CONFIG)

    client = Controller(
        routers=[args.router],
        strong_model=args.strong_model,
        weak_model=args.weak_model,
    )

    for dataset in datasets:
        loader_name = resolve_loader_name(dataset)
        config_key = loader_name if loader_name in dataset_configs else dataset
        dataset_config = dataset_configs.get(config_key)
        if dataset_config is None:
            raise ValueError(f"Missing dataset config for '{dataset}'.")
        if args.list_mmlu_categories and dataset == "mmlu_pro":
            loader = get_loader(loader_name, config=dataset_config, data_root="datasets")
            mmlu_df = loader.load(force_redownload=args.force_redownload).copy()
            _list_mmlu_categories(mmlu_df)
            return
        run_dataset(
            dataset=dataset,
            dataset_config=dataset_config,
            args=args,
            client=client,
        )


if __name__ == "__main__":
    main()