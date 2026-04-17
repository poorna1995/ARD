from __future__ import annotations

import argparse
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from baselines.results import split_and_save_all
from baselines.runner import QueryInput, RunSelection, run_selected
from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES

# -----------------------------
# Minimal run config (edit only this block)
# -----------------------------
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
AUTO_DISCOVER_PROCESSED = True

DATASET_PATHS: dict[str, str] = {
    # Fill these with actual files once present under datasets/processed:
    # "gaia": "../datasets/processed/gaia.parquet",
    # "mmlu_pro": "../datasets/processed/mmlu_pro.parquet",
    # "math_hard": "../datasets/processed/math_hard.parquet",
    # "swe_bench_verified": "../datasets/processed/swe_bench_verified.parquet",
}

SELECTION = RunSelection(
    datasets=("gaia",),       # Run only GAIA
    modalities=("vanilla",),  # Run only Vanilla
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.1,
)

RESULTS_DIR = "results/unified_baseline"
SAMPLE_LIMIT: int | None = None  # set e.g. 20 for quick smoke tests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run minimal unified baseline experiments.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(SELECTION.datasets),
        choices=ALLOWED_DATASETS,
        help="Dataset IDs to run.",
    )
    parser.add_argument(
        "--modalities",
        nargs="+",
        default=list(SELECTION.modalities),
        choices=ALLOWED_MODALITIES,
        help="Modalities to run.",
    )
    parser.add_argument(
        "--model",
        default=SELECTION.model,
        help="Model name (default from script config).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=SELECTION.max_tokens,
        help="Max completion tokens.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=SELECTION.temperature,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--sample-limit",
        type=int,
        default=SAMPLE_LIMIT,
        help="Optional per-dataset row cap.",
    )
    parser.add_argument(
        "--results-dir",
        default=RESULTS_DIR,
        help="Output directory for run artifacts.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=SELECTION.max_steps,
        help="Max reasoning/tool steps for agentic prompts.",
    )
    parser.add_argument(
        "--tool-budget",
        type=int,
        default=SELECTION.tool_budget,
        help="Tool budget for agentic prompts.",
    )
    return parser.parse_args()


def discover_processed_dataset_paths(processed_dir: Path) -> dict[str, str]:
    """
    Auto-map dataset IDs to parquet files under datasets/processed using filename hints.
    """
    if not processed_dir.exists():
        return {}

    aliases: dict[str, tuple[str, ...]] = {
        "gaia": ("gaia", "processed_gaia"),
        "mmlu_pro": ("mmlu_pro", "mmlupro", "processed_mmlu_pro"),
        "math_hard": ("math_hard", "mathhard", "processed_math_hard"),
        "swe_bench_verified": (
            "swe_bench_verified",
            "swebench_verified",
            "swe_bench",
            "processed_swe_bench_verified",
        ),
    }

    discovered: dict[str, str] = {}
    parquet_files = sorted(processed_dir.rglob("*.parquet"))
    for dataset, keys in aliases.items():
        for file_path in parquet_files:
            name = file_path.stem.lower().replace("-", "_")
            if any(key in name for key in keys):
                discovered[dataset] = str(file_path.resolve())
                break
    return discovered


def _pick(row: pd.Series, keys: tuple[str, ...], default: str = "") -> str:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return str(row[key])
    return default


def _to_query_input(dataset: str, row: pd.Series, idx: int) -> QueryInput:
    query_id = _pick(row, ("id", "query_id", "instance_id"), default=str(idx))

    if dataset == "gaia":
        query = _pick(row, ("query", "question"))
        ground_truth = _pick(row, ("answer", "ground_truth"))
        metadata = {
            "level": _pick(row, ("level",)),
            "file_name": _pick(row, ("file_name",)),
        }
        return QueryInput(
            query_id=query_id,
            dataset=dataset,
            query=query,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    if dataset == "mmlu_pro":
        query = _pick(row, ("query", "question"))
        ground_truth = _pick(row, ("answer", "ground_truth", "answer_index"))
        raw_options = row.get("options", "")
        if isinstance(raw_options, (list, tuple)):
            options = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(raw_options))
        elif hasattr(raw_options, "tolist") and not isinstance(raw_options, (str, bytes)):
            as_list = raw_options.tolist()
            if isinstance(as_list, list):
                options = "\n".join(f"{chr(65 + i)}. {opt}" for i, opt in enumerate(as_list))
            else:
                options = "" if pd.isna(as_list) else str(as_list)
        else:
            options = "" if pd.isna(raw_options) else str(raw_options)
        metadata = {"subject": _pick(row, ("subject", "category"))}
        return QueryInput(
            query_id=query_id,
            dataset=dataset,
            query=query,
            ground_truth=ground_truth,
            options=options,
            metadata=metadata,
        )

    if dataset == "math_hard":
        query = _pick(row, ("problem", "query", "question"))
        # Math-Hard GT should be final answer label, not full solution rationale.
        ground_truth = _pick(row, ("answer", "ground_truth", "solution"))
        metadata = {
            "level": _pick(row, ("level",)),
            "category": _pick(row, ("type", "category")),
        }
        return QueryInput(
            query_id=query_id,
            dataset=dataset,
            query=query,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    if dataset == "swe_bench_verified":
        query = _pick(row, ("problem_statement", "query", "question", "instruction"))
        ground_truth = _pick(row, ("patch", "answer", "ground_truth"))
        metadata = {"repo": _pick(row, ("repo",))}
        return QueryInput(
            query_id=query_id,
            dataset=dataset,
            query=query,
            ground_truth=ground_truth,
            metadata=metadata,
        )

    raise ValueError(f"Unsupported dataset: {dataset}")


def _validate_dataset_schema(dataset: str, df: pd.DataFrame, path: Path) -> None:
    cols = set(df.columns)
    expected_any: dict[str, tuple[set[str], ...]] = {
        "gaia": ({"query"}, {"question"}),
        "mmlu_pro": ({"query", "options"}, {"question", "options"}),
        "math_hard": ({"problem", "answer"}, {"question", "answer"}, {"query", "answer"}),
        "swe_bench_verified": ({"problem_statement"}, {"instruction"}, {"query"}),
    }
    if dataset not in expected_any:
        return
    if not any(required.issubset(cols) for required in expected_any[dataset]):
        raise ValueError(
            f"Dataset/path mismatch for '{dataset}'. File '{path}' has columns {sorted(cols)}. "
            f"Expected one of: {expected_any[dataset]}"
        )


def load_queries(dataset_paths: dict[str, str], sample_limit: int | None = None) -> list[QueryInput]:
    queries: list[QueryInput] = []
    for dataset, path in dataset_paths.items():
        p = Path(path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found for '{dataset}': {path}")

        df = pd.read_parquet(p)
        _validate_dataset_schema(dataset, df, p)
        if sample_limit:
            df = df.head(sample_limit)

        for i, (_, row) in enumerate(df.iterrows()):
            queries.append(_to_query_input(dataset, row, i))

    return queries


def main() -> None:
    args = parse_args()
    selection = RunSelection(
        datasets=tuple(args.datasets),
        modalities=tuple(args.modalities),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        max_steps=args.max_steps,
        tool_budget=args.tool_budget,
    )

    dataset_paths = dict(DATASET_PATHS)
    if AUTO_DISCOVER_PROCESSED:
        dataset_paths.update(discover_processed_dataset_paths(PROCESSED_DIR))

    # Keep only requested datasets so one-dataset runs stay minimal and fast.
    dataset_paths = {
        ds: p for ds, p in dataset_paths.items() if ds in set(selection.datasets)
    }

    if not dataset_paths:
        raise ValueError(
            "No dataset paths found for selected datasets. Add DATASET_PATHS manually or place parquet files in datasets/processed."
        )

    queries = load_queries(dataset_paths, sample_limit=args.sample_limit)
    results = run_selected(queries=queries, selection=selection)
    saved_paths = split_and_save_all(results, base_dir=args.results_dir)

    print(f"Queries loaded: {len(queries)}")
    print(f"Records produced: {len(results)}")
    print("Dataset paths used:")
    for ds, p in dataset_paths.items():
        print(f"  - {ds}: {p}")
    print("Saved run outputs:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
