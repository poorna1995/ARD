


from __future__ import annotations

import argparse
import sys
import warnings
import zlib
from dataclasses import asdict
from pathlib import Path

# `python baselines/run_baseline.py` puts `baselines/` on sys.path, not the
# repo root — insert the project root so `from baselines...` resolves.
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from baselines.constants import ALLOWED_DATASETS, ALLOWED_MODALITIES

# BUG FIX 1: validate_dataset / validate_modality / validate_model were
# imported but never called in this module.  Only ALLOWED_* are used here;
# the validate_* helpers belong in resolve_prompts / runner.
# Removed unused imports to avoid misleading readers into thinking
# per-value validation was happening here.

# ── Paths ─────────────────────────────────────────────────────────────────────
PROCESSED_DIR = REPO_ROOT / "datasets" / "processed"
AUTO_DISCOVER_PROCESSED = True

DATASET_PATHS: dict[str, str] = {
    # "gaia":               "../datasets/processed/gaia.parquet",
    # "mmlu_pro":           "../datasets/processed/mmlu_pro.parquet",
    # "math_hard":          "../datasets/processed/math_hard.parquet",
    # "swe_bench_verified": "../datasets/processed/swe_bench_verified.parquet",
}

from baselines.runner import RunSelection  # noqa: E402 — after path setup

SELECTION = RunSelection(
    datasets=("gaia",),
    modalities=("vanilla",),
    model="gpt-4o-mini",
    max_tokens=1024,
    temperature=0.0,
)

RESULTS_DIR = "results1/unified_baseline"
SAMPLE_LIMIT: int | None = None

# ── Column priority maps ───────────────────────────────────────────────────────
_MMLU_CAT_COLUMNS = ("category", "subject")

_QUERY_COLS: dict[str, tuple[str, ...]] = {
    "gaia":               ("query", "question"),
    "mmlu_pro":           ("query", "question"),
    "math_hard":          ("problem", "query", "question"),
    "swe_bench_verified": ("problem_statement", "query", "question", "instruction"),
}
_GT_COLS: dict[str, tuple[str, ...]] = {
    "gaia":               ("answer", "ground_truth"),
    "mmlu_pro":           ("answer", "ground_truth", "answer_index"),
    "math_hard":          ("answer", "ground_truth", "solution"),
    "swe_bench_verified": ("patch", "answer", "ground_truth"),
}
_ID_COLS = ("id", "query_id", "instance_id")


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _normalize_multi_choices(
    parser: argparse.ArgumentParser,
    values: list[str],
    allowed: tuple[str, ...],
    flag_label: str,
) -> list[str]:
    allowed_set = set(allowed)
    out: list[str] = []
    for chunk in values:
        for piece in str(chunk).split(","):
            p = piece.strip()
            if not p:
                continue
            if p.startswith("-"):
                parser.error(
                    f"{flag_label}: got {p!r}, which looks like a CLI flag inside the "
                    f"value list. Use space-separated names only, then other flags."
                )
            if p not in allowed_set:
                parser.error(
                    f"{flag_label}: invalid choice: {p!r} "
                    f"(choose from {', '.join(allowed)})"
                )
            out.append(p)
    if not out:
        parser.error(f"{flag_label}: at least one value is required")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run minimal unified baseline experiments."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=list(SELECTION.datasets), metavar="NAME",
        help=f"Dataset ids ({', '.join(ALLOWED_DATASETS)}).",
    )
    parser.add_argument(
        "--modalities", nargs="+", default=list(SELECTION.modalities), metavar="MOD",
        help=f"Run modes ({', '.join(ALLOWED_MODALITIES)}).",
    )
    parser.add_argument("--model",       default=SELECTION.model)
    parser.add_argument("--max-tokens",  type=int,   default=SELECTION.max_tokens)
    parser.add_argument("--temperature", type=float, default=SELECTION.temperature)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        metavar="N",
        help="OpenAI API seed (use with temperature 0 for best reproducibility).",
    )
    parser.add_argument(
        "--no-seed",
        action="store_true",
        help="Do not pass seed to the OpenAI API.",
    )
    parser.add_argument("--sample-limit", type=int,  default=SAMPLE_LIMIT, metavar="N")
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    parser.add_argument("--max-steps",   type=int,   default=SELECTION.max_steps)
    parser.add_argument("--tool-budget", type=int,   default=SELECTION.tool_budget)
    parser.add_argument("--routellm-router",       default=SELECTION.routellm_router)
    parser.add_argument("--routellm-threshold",    type=float, default=SELECTION.routellm_threshold)
    parser.add_argument("--routellm-strong-model", default=SELECTION.routellm_strong_model)
    parser.add_argument("--routellm-weak-model",   default=SELECTION.routellm_weak_model)
    parser.add_argument("--mmlu-categories",         nargs="+",  default=None, metavar="NAME")
    parser.add_argument("--list-mmlu-categories",    action="store_true")
    parser.add_argument("--mmlu-per-category-limit", type=int,   default=None, metavar="N")
    parser.add_argument("--mmlu-category-samples",   type=str,   default=None, metavar="SPEC")
    parser.add_argument("--mmlu-sample-seed",        type=int,   default=None, metavar="SEED")
    parser.add_argument(
        "--verbose-prompts",
        action="store_true",
        help="Print each query's task text, system prompt, user prompt, raw model output, "
        "predicted answer, and ground truth while running.",
    )
    args = parser.parse_args()
    args.datasets   = _normalize_multi_choices(parser, args.datasets,   ALLOWED_DATASETS,   "--datasets")
    args.modalities = _normalize_multi_choices(parser, args.modalities, ALLOWED_MODALITIES, "--modalities")
    return args


# ══════════════════════════════════════════════════════════════════════════════
# Dataset discovery and loading
# ══════════════════════════════════════════════════════════════════════════════

def discover_processed_dataset_paths(processed_dir: Path) -> dict[str, str]:
    if not processed_dir.exists():
        return {}
    aliases: dict[str, tuple[str, ...]] = {
        "gaia":               ("gaia", "processed_gaia"),
        "mmlu_pro":           ("mmlu_pro", "mmlupro", "processed_mmlu_pro"),
        "math_hard":          ("math_hard", "mathhard", "processed_math_hard"),
        "swe_bench_verified": ("swe_bench_verified", "swebench_verified",
                               "swe_bench", "processed_swe_bench_verified"),
    }
    keyword_to_dataset: dict[str, str] = {
        key: ds for ds, keys in aliases.items() for key in keys
    }
    discovered: dict[str, str] = {}
    for file_path in sorted(processed_dir.rglob("*.parquet")):
        name = file_path.stem.lower().replace("-", "_")
        for keyword, dataset in keyword_to_dataset.items():
            if keyword in name and dataset not in discovered:
                discovered[dataset] = str(file_path.resolve())
                break
    return discovered


def _resolve_col(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def _build_query_inputs(
    dataset: str, df: pd.DataFrame
) -> list:
    # BUG FIX 2: The original used itertuples() + dict(zip(cols, row)) which
    # rebuilds a full column→value mapping on every row (O(cols) allocations
    # per row).  df.to_dict("records") does the same work once in C and
    # returns a plain list of dicts — cleaner and faster for large frames.
    # Renamed from _build_query_inputs_vectorised: the function was never
    # vectorised; the old name was misleading.
    from baselines.runner import QueryInput
    cols = df.columns

    id_col    = _resolve_col(cols, _ID_COLS)
    query_col = _resolve_col(cols, _QUERY_COLS[dataset])
    gt_col    = _resolve_col(cols, _GT_COLS[dataset])

    extra: dict[str, str | None] = {}
    if dataset == "gaia":
        extra["level"]     = _resolve_col(cols, ("level",))
        extra["file_name"] = _resolve_col(cols, ("file_name",))
    elif dataset == "mmlu_pro":
        extra["opts"]    = _resolve_col(cols, ("options",))
        extra["subject"] = _resolve_col(cols, ("subject", "category"))
        extra["cat"]     = "category" if "category" in cols else None
    elif dataset == "math_hard":
        extra["level"] = _resolve_col(cols, ("level",))
        extra["cat"]   = _resolve_col(cols, ("type", "category"))
    elif dataset == "swe_bench_verified":
        extra["repo"] = _resolve_col(cols, ("repo",))

    records = df.to_dict("records")
    results: list[QueryInput] = []

    for idx, row_dict in enumerate(records):
        query_id     = str(row_dict.get(id_col, idx)) if id_col else str(idx)
        query_text   = str(row_dict.get(query_col, "")) if query_col else ""
        ground_truth = str(row_dict.get(gt_col, ""))    if gt_col    else ""

        if dataset == "gaia":
            results.append(QueryInput(
                query_id=query_id, dataset=dataset,
                query=query_text, ground_truth=ground_truth,
                metadata={
                    "level":     str(row_dict.get(extra["level"], ""))     if extra["level"]     else "",
                    "file_name": str(row_dict.get(extra["file_name"], "")) if extra["file_name"] else "",
                },
            ))

        elif dataset == "mmlu_pro":
            raw_opts = row_dict.get(extra["opts"], "") if extra["opts"] else ""
            if isinstance(raw_opts, (list, tuple)):
                options = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(raw_opts))
            elif hasattr(raw_opts, "tolist") and not isinstance(raw_opts, (str, bytes)):
                as_list = raw_opts.tolist()
                options = (
                    "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(as_list))
                    if isinstance(as_list, list)
                    else ("" if pd.isna(as_list) else str(as_list))
                )
            else:
                options = "" if (
                    raw_opts is None or (isinstance(raw_opts, float) and pd.isna(raw_opts))
                ) else str(raw_opts)

            subj = str(row_dict.get(extra["subject"], "")) if extra["subject"] else ""
            meta: dict = {"subject": subj}
            if extra["cat"]:
                v = row_dict.get(extra["cat"])
                if v is not None and not (isinstance(v, float) and pd.isna(v)):
                    meta["category"] = str(v)

            results.append(QueryInput(
                query_id=query_id, dataset=dataset,
                query=query_text, ground_truth=ground_truth,
                options=options, metadata=meta,
            ))

        elif dataset == "math_hard":
            results.append(QueryInput(
                query_id=query_id, dataset=dataset,
                query=query_text, ground_truth=ground_truth,
                metadata={
                    "level":    str(row_dict.get(extra["level"], "")) if extra["level"] else "",
                    "category": str(row_dict.get(extra["cat"],   "")) if extra["cat"]   else "",
                },
            ))

        elif dataset == "swe_bench_verified":
            results.append(QueryInput(
                query_id=query_id, dataset=dataset,
                query=query_text, ground_truth=ground_truth,
                metadata={"repo": str(row_dict.get(extra["repo"], "")) if extra["repo"] else ""},
            ))

        else:
            raise ValueError(f"Unsupported dataset: {dataset}")

    return results


def _validate_dataset_schema(dataset: str, df: pd.DataFrame, path: Path) -> None:
    cols = set(df.columns)
    expected_any: dict[str, tuple[set[str], ...]] = {
        "gaia":               ({"query"}, {"question"}),
        "mmlu_pro":           ({"query", "options"}, {"question", "options"}),
        "math_hard":          ({"problem", "answer"}, {"question", "answer"}, {"query", "answer"}),
        "swe_bench_verified": ({"problem_statement"}, {"instruction"}, {"query"}),
    }
    if dataset not in expected_any:
        return
    if not any(required.issubset(cols) for required in expected_any[dataset]):
        raise ValueError(
            f"Dataset/path mismatch for '{dataset}'. File '{path}' has columns {sorted(cols)}. "
            f"Expected one of: {expected_any[dataset]}"
        )


def _mmlu_filter_and_cap(
    df: pd.DataFrame, path: Path, *,
    categories: tuple[str, ...] | None,
    per_category_limit: int | None,
    category_samples: dict[str, int] | None,
    sample_seed: int | None,
) -> pd.DataFrame:
    col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(
            f"MMLU filtering needs a category column. Columns in {path}: {sorted(df.columns)}"
        )

    want:   set[str] | None = None
    limits: dict[str, int]  = {}

    if categories:
        want = {c.strip().lower() for c in categories if c and str(c).strip()}
    if category_samples:
        limits = {k.strip().lower(): int(v) for k, v in category_samples.items()}
        want = (want or set()) | set(limits.keys())

    if want is None and per_category_limit is None and not category_samples:
        return df

    norm_series = df[col].astype(str).str.strip().str.lower()
    present = set(norm_series.unique())

    if want:
        # BUG FIX 3: original only raised when ALL requested categories were
        # absent (`(want - present) == want`).  If even one category matched,
        # the missing ones were silently dropped, producing a smaller-than-
        # expected dataset with no feedback to the caller.
        # Now: always raise on a fully-absent request; warn on partial misses.
        missing_cats = want - present
        if missing_cats == want:
            avail = sorted({str(x) for x in df[col].dropna().unique()})
            raise ValueError(
                f"No MMLU-Pro rows match {sorted(want)!r}. "
                f"Available (sample): {avail[:30]}"
            )
        if missing_cats:
            warnings.warn(
                f"Some requested MMLU-Pro categories were not found and will be "
                f"skipped: {sorted(missing_cats)}",
                stacklevel=2,
            )

    buckets: dict[str, list[int]] = {}
    for pos, norm_val in enumerate(norm_series.tolist()):
        if want is not None and norm_val not in want:
            continue
        buckets.setdefault(norm_val, []).append(pos)

    if not buckets:
        return df.iloc[0:0]

    parts: list[pd.DataFrame] = []
    for cat, positions in buckets.items():
        cap = limits.get(cat, per_category_limit)
        if cap is None:
            parts.append(df.iloc[positions])
        else:
            n = min(max(0, int(cap)), len(positions))
            if n == 0:
                continue
            if sample_seed is not None and n < len(positions):
                import random
                rs = (int(sample_seed) + zlib.crc32(cat.encode())) % (2**31 - 1) or 1
                rng = random.Random(rs)
                chosen = sorted(rng.sample(positions, n))
                parts.append(df.iloc[chosen])
            else:
                parts.append(df.iloc[positions[:n]])

    return pd.concat(parts, ignore_index=True) if parts else df.iloc[0:0]


def _parse_mmlu_category_samples(raw: str | None) -> dict[str, int] | None:
    if raw is None or not str(raw).strip():
        return None
    out: dict[str, int] = {}
    for piece in str(raw).split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "=" not in piece:
            raise ValueError(f"Invalid --mmlu-category-samples segment {piece!r}; use name=count.")
        name, num = piece.split("=", 1)
        name = name.strip().lower()
        if not name:
            raise ValueError(f"Empty category name in segment {piece!r}")
        out[name] = int(num.strip())
    return out or None


def load_queries(
    dataset_paths: dict[str, str],
    sample_limit: int | None = None, *,
    mmlu_categories: tuple[str, ...] | None = None,
    mmlu_per_category_limit: int | None = None,
    mmlu_category_samples: dict[str, int] | None = None,
    mmlu_sample_seed: int | None = None,
) -> list:
    queries = []
    for dataset, path in dataset_paths.items():
        p = Path(path)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if not p.exists():
            raise FileNotFoundError(f"Dataset file not found for '{dataset}': {path}")

        df = pd.read_parquet(p)
        _validate_dataset_schema(dataset, df, p)

        if dataset == "mmlu_pro":
            needs_filter = (
                mmlu_categories or mmlu_category_samples or
                mmlu_per_category_limit is not None or mmlu_sample_seed is not None
            )
            if needs_filter:
                df = _mmlu_filter_and_cap(
                    df, p,
                    categories=mmlu_categories,
                    per_category_limit=mmlu_per_category_limit,
                    category_samples=mmlu_category_samples,
                    sample_seed=mmlu_sample_seed,
                )

        # BUG FIX 4: `if sample_limit:` treated sample_limit=0 as "no limit"
        # because 0 is falsy.  Use explicit `is not None` check instead.
        if sample_limit is not None:
            df = df.iloc[:sample_limit]

        queries.extend(_build_query_inputs(dataset, df))
    return queries


def list_mmlu_categories_from_paths(dataset_paths: dict[str, str]) -> None:
    if "mmlu_pro" not in dataset_paths:
        raise ValueError("No mmlu_pro path found.")
    p = Path(dataset_paths["mmlu_pro"])
    if not p.is_absolute():
        p = (REPO_ROOT / p).resolve()
    df = pd.read_parquet(p)
    col = next((c for c in _MMLU_CAT_COLUMNS if c in df.columns), None)
    if col is None:
        raise ValueError(f"No category column in {p}.")
    vc = df[col].astype(str).str.strip().value_counts().sort_index()
    print(f"MMLU-Pro file: {p}\nColumn: {col!r} ({len(vc)} distinct values)")
    for val, count in vc.items():
        print(f"  {val!r}  (n={count})")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args = parse_args()

    dataset_paths = dict(DATASET_PATHS)
    if AUTO_DISCOVER_PROCESSED:
        dataset_paths.update(discover_processed_dataset_paths(PROCESSED_DIR))

    if args.list_mmlu_categories:
        if "mmlu_pro" not in dataset_paths:
            raise ValueError("Could not find mmlu_pro parquet under datasets/processed.")
        list_mmlu_categories_from_paths({"mmlu_pro": dataset_paths["mmlu_pro"]})
        return

    selection = RunSelection(
        datasets=tuple(args.datasets),
        modalities=tuple(args.modalities),
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        seed=None if args.no_seed else args.seed,
        max_steps=args.max_steps,
        tool_budget=args.tool_budget,
        routellm_router=args.routellm_router,
        routellm_threshold=args.routellm_threshold,
        routellm_strong_model=args.routellm_strong_model,
        routellm_weak_model=args.routellm_weak_model,
    )

    dataset_paths = {ds: p for ds, p in dataset_paths.items() if ds in set(selection.datasets)}
    if not dataset_paths:
        raise ValueError(
            "No dataset paths found. Add DATASET_PATHS or place parquets in datasets/processed."
        )

    mmlu_cat: tuple[str, ...] | None = None
    if args.mmlu_categories:
        mmlu_cat = tuple(args.mmlu_categories)
        if "mmlu_pro" not in dataset_paths:
            raise ValueError("--mmlu-categories only applies when mmlu_pro is selected.")

    if (args.mmlu_per_category_limit is not None or args.mmlu_category_samples
            or args.mmlu_sample_seed is not None) and "mmlu_pro" not in dataset_paths:
        raise ValueError("MMLU per-category flags require mmlu_pro in --datasets.")

    mmlu_samples_parsed = _parse_mmlu_category_samples(args.mmlu_category_samples)

    queries = load_queries(
        dataset_paths,
        sample_limit=args.sample_limit,
        mmlu_categories=mmlu_cat,
        mmlu_per_category_limit=args.mmlu_per_category_limit,
        mmlu_category_samples=mmlu_samples_parsed,
        mmlu_sample_seed=args.mmlu_sample_seed,
    )

    from baselines.runner import run_selected
    from baselines.results import split_and_save_all
    from baselines.run_manifest import RunManifestContext

    results = run_selected(
        queries=queries,
        selection=selection,
        verbose_prompts=args.verbose_prompts,
    )
    manifest_ctx = RunManifestContext(
        repo_root=REPO_ROOT,
        cli_args=dict(vars(args)),
        dataset_paths=dict(dataset_paths),
        run_selection=asdict(selection),
    )
    saved_paths = split_and_save_all(
        results, base_dir=args.results_dir, manifest_context=manifest_ctx,
    )

    print(f"Queries loaded  : {len(queries)}")
    print(f"Records produced: {len(results)}")
    print("Dataset paths used:")
    for ds, p in dataset_paths.items():
        print(f"  - {ds}: {p}")
    print("Saved outputs:")
    for p in saved_paths:
        print(f"  - {p}")


if __name__ == "__main__":
    main()



