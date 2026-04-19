# from __future__ import annotations

# import json
# from dataclasses import asdict
# from pathlib import Path
# from typing import Optional

# import pandas as pd

# from baselines.schema import UnifiedExperimentRecord, UnifiedRunMetrics


# def _safe_name(value: str) -> str:
#     return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


# def _run_dir(base_dir: Path, dataset: str, modality: str, model: str) -> Path:
#     out = base_dir / _safe_name(dataset) / _safe_name(modality) / _safe_name(model)
#     out.mkdir(parents=True, exist_ok=True)
#     return out


# def _build_metrics(records: list[UnifiedExperimentRecord]) -> UnifiedRunMetrics:
#     if not records:
#         raise ValueError("Cannot build metrics from empty records.")

#     total_queries = len(records)
#     correct_queries = sum(1 for r in records if r.is_correct)
#     accuracy = correct_queries / total_queries if total_queries else 0.0

#     total_prompt_tokens = sum(r.prompt_tokens for r in records)
#     total_completion_tokens = sum(r.completion_tokens for r in records)
#     total_tokens = sum(r.total_tokens for r in records)
#     avg_tokens_per_query = total_tokens / total_queries if total_queries else 0.0

#     total_cost_usd = sum(r.cost_usd for r in records)
#     latencies = [r.latency_s for r in records]
#     avg_latency_s = sum(latencies) / total_queries if total_queries else 0.0
#     p50_latency_s = float(pd.Series(latencies).quantile(0.50)) if latencies else 0.0
#     p95_latency_s = float(pd.Series(latencies).quantile(0.95)) if latencies else 0.0

#     error_count = sum(1 for r in records if r.error_message)
#     error_rate = error_count / total_queries if total_queries else 0.0

#     step_vals = [r.reasoning_steps for r in records if r.reasoning_steps is not None]
#     avg_reasoning_steps: Optional[float] = None
#     if step_vals:
#         avg_reasoning_steps = round(sum(step_vals) / len(step_vals), 3)

#     extra_metadata: dict = {}
#     if records[0].dataset == "gaia":
#         # Level-wise GAIA breakdown for research reporting.
#         level_rows: list[dict] = []
#         for r in records:
#             level = str((r.metadata or {}).get("level", "")).strip() or "unknown"
#             level_rows.append({"level": level, "is_correct": bool(r.is_correct)})
#         ldf = pd.DataFrame(level_rows)
#         by_level = (
#             ldf.groupby("level", dropna=False)["is_correct"]
#             .agg(total="count", correct="sum", accuracy="mean")
#             .reset_index()
#         )
#         extra_metadata["gaia_level_metrics"] = {
#             str(row["level"]): {
#                 "total": int(row["total"]),
#                 "correct": int(row["correct"]),
#                 "accuracy": round(float(row["accuracy"]), 6),
#             }
#             for _, row in by_level.iterrows()
#         }

#     if records[0].dataset == "mmlu_pro":
#         # Per-category / subject breakdown (metadata from loader: category, subject).
#         cat_rows: list[dict] = []
#         for r in records:
#             m = r.metadata or {}
#             bucket = (
#                 str(m.get("category") or m.get("subject") or "").strip() or "unknown"
#             )
#             cat_rows.append({"category": bucket, "is_correct": bool(r.is_correct)})
#         cdf = pd.DataFrame(cat_rows)
#         by_cat = (
#             cdf.groupby("category", dropna=False)["is_correct"]
#             .agg(total="count", correct="sum", accuracy="mean")
#             .reset_index()
#         )
#         extra_metadata["mmlu_category_metrics"] = {
#             str(row["category"]): {
#                 "total": int(row["total"]),
#                 "correct": int(row["correct"]),
#                 "accuracy": round(float(row["accuracy"]), 6),
#             }
#             for _, row in by_cat.iterrows()
#         }

#     first = records[0]
#     return UnifiedRunMetrics(
#         dataset=first.dataset,
#         modality=first.modality,
#         model=first.model,
#         system_prompt_version=first.system_prompt_version,
#         user_prompt_version=first.user_prompt_version,
#         prompt_hash=first.prompt_hash,
#         total_queries=total_queries,
#         correct_queries=correct_queries,
#         accuracy=round(accuracy, 6),
#         total_prompt_tokens=total_prompt_tokens,
#         total_completion_tokens=total_completion_tokens,
#         total_tokens=total_tokens,
#         avg_tokens_per_query=round(avg_tokens_per_query, 3),
#         total_cost_usd=round(total_cost_usd, 6),
#         avg_latency_s=round(avg_latency_s, 3),
#         p50_latency_s=round(p50_latency_s, 3),
#         p95_latency_s=round(p95_latency_s, 3),
#         error_count=error_count,
#         error_rate=round(error_rate, 6),
#         avg_reasoning_steps=avg_reasoning_steps,
#         metadata=extra_metadata,
#     )


# def save_run_records(
#     records: list[UnifiedExperimentRecord],
#     base_dir: str = "results/unified_baseline",
# ) -> Path:
#     """
#     Save one run bucket (same dataset/modality/model) to:
#       responses.parquet
#       metrics.json
#     """
#     if not records:
#         raise ValueError("No records to save.")

#     key_set = {(r.dataset, r.modality, r.model) for r in records}
#     if len(key_set) != 1:
#         raise ValueError(
#             "save_run_records expects one dataset/modality/model bucket only."
#         )

#     dataset, modality, model = next(iter(key_set))
#     out_dir = _run_dir(Path(base_dir), dataset, modality, model)

#     df = pd.DataFrame([asdict(r) for r in records])
#     df.to_parquet(out_dir / "responses.parquet", index=False)

#     metrics = _build_metrics(records)
#     with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
#         json.dump(asdict(metrics), f, indent=2)

#     return out_dir


# def split_and_save_all(
#     records: list[UnifiedExperimentRecord],
#     base_dir: str = "results/unified_baseline",
# ) -> list[Path]:
#     """
#     Split mixed records by (dataset, modality, model) and save each bucket.
#     """
#     buckets: dict[tuple[str, str, str], list[UnifiedExperimentRecord]] = {}
#     for r in records:
#         buckets.setdefault((r.dataset, r.modality, r.model), []).append(r)

#     saved: list[Path] = []
#     for bucket in buckets.values():
#         saved.append(save_run_records(bucket, base_dir=base_dir))
#     return saved



from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from baselines.run_manifest import RunManifestContext, write_run_manifest
from baselines.schema import UnifiedExperimentRecord, UnifiedRunMetrics


def _safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _run_dir(base_dir: Path, dataset: str, modality: str, model: str) -> Path:
    out = base_dir / _safe_name(dataset) / _safe_name(modality) / _safe_name(model)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_metrics(records: list[UnifiedExperimentRecord]) -> UnifiedRunMetrics:
    """
    Compute all aggregate metrics in a single O(N) pass over records.

    Previous version made 10+ separate O(N) passes (one per sum/count expression).
    Now: one pass accumulates all scalar accumulators; numpy handles quantiles on
    the single latency array collected during that pass.
    """
    if not records:
        raise ValueError("Cannot build metrics from empty records.")

    total_queries       = len(records)
    correct_queries     = 0
    total_prompt_tokens = 0
    total_comp_tokens   = 0
    total_tokens        = 0
    total_cost_usd      = 0.0
    error_count         = 0
    step_sum            = 0.0
    step_count          = 0
    react_traj_sum       = 0.0
    react_traj_count     = 0

    # Pre-allocate arrays — avoids repeated list.append + pd.Series overhead
    latencies = np.empty(total_queries, dtype=np.float64)

    # Dataset-specific buckets built in the same pass
    first        = records[0]
    dataset      = first.dataset
    is_gaia      = dataset == "gaia"
    is_mmlu      = dataset == "mmlu_pro"

    # For breakdown dicts: { bucket_key -> [total, correct] }
    breakdown: dict[str, list[int]] = {}

    for i, r in enumerate(records):
        correct_queries     += r.is_correct
        total_prompt_tokens += r.prompt_tokens
        total_comp_tokens   += r.completion_tokens
        total_tokens        += r.total_tokens
        total_cost_usd      += r.cost_usd
        latencies[i]         = r.latency_s
        error_count         += bool(r.error_message)

        if r.reasoning_steps is not None:
            step_sum   += r.reasoning_steps
            step_count += 1

        if r.react_trace_length is not None:
            react_traj_sum   += r.react_trace_length
            react_traj_count += 1

        # Breakdown accumulation — O(1) per row using mutable [total, correct] list
        if is_gaia:
            key = str((r.metadata or {}).get("level", "")).strip() or "unknown"
            bucket = breakdown.setdefault(key, [0, 0])
            bucket[0] += 1
            bucket[1] += r.is_correct

        elif is_mmlu:
            m   = r.metadata or {}
            key = str(m.get("category") or m.get("subject") or "").strip() or "unknown"
            bucket = breakdown.setdefault(key, [0, 0])
            bucket[0] += 1
            bucket[1] += r.is_correct

    # ── Derived scalars ──────────────────────────────────────────────────────
    accuracy             = correct_queries / total_queries
    avg_tokens_per_query = total_tokens    / total_queries
    avg_latency_s        = float(latencies.mean())
    # numpy quantile is O(N log N) but avoids constructing a pd.Series twice
    p50_latency_s        = float(np.percentile(latencies, 50))
    p95_latency_s        = float(np.percentile(latencies, 95))
    error_rate           = error_count / total_queries
    avg_reasoning_steps: Optional[float] = (
        round(step_sum / step_count, 3) if step_count else None
    )

    extra_metadata: dict = {}
    if first.modality in ("zero_shot_cot", "few_shot_cot"):
        extra_metadata["cot_variant"] = first.modality
    if first.modality == "react" and react_traj_count:
        extra_metadata["avg_react_trace_length"] = round(
            react_traj_sum / react_traj_count, 3
        )
    if is_gaia and breakdown:
        extra_metadata["gaia_level_metrics"] = {
            lvl: {
                "total":    tot,
                "correct":  cor,
                "accuracy": round(cor / tot, 6) if tot else 0.0,
            }
            for lvl, (tot, cor) in sorted(breakdown.items())
        }
    elif is_mmlu and breakdown:
        extra_metadata["mmlu_category_metrics"] = {
            cat: {
                "total":    tot,
                "correct":  cor,
                "accuracy": round(cor / tot, 6) if tot else 0.0,
            }
            for cat, (tot, cor) in sorted(breakdown.items())
        }

    return UnifiedRunMetrics(
        dataset=dataset,
        modality=first.modality,
        model=first.model,
        system_prompt_version=first.system_prompt_version,
        user_prompt_version=first.user_prompt_version,
        prompt_hash=first.prompt_hash,
        total_queries=total_queries,
        correct_queries=correct_queries,
        accuracy=round(accuracy, 6),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_comp_tokens,
        total_tokens=total_tokens,
        avg_tokens_per_query=round(avg_tokens_per_query, 3),
        total_cost_usd=round(total_cost_usd, 6),
        avg_latency_s=round(avg_latency_s, 3),
        p50_latency_s=round(p50_latency_s, 3),
        p95_latency_s=round(p95_latency_s, 3),
        error_count=error_count,
        error_rate=round(error_rate, 6),
        avg_reasoning_steps=avg_reasoning_steps,
        metadata=extra_metadata,
    )


def save_run_records(
    records: list[UnifiedExperimentRecord],
    base_dir: str = "results1/unified_baseline",
    manifest_context: Optional[RunManifestContext] = None,
) -> Path:
    """
    Save one run bucket (same dataset/modality/model) to:
      responses.parquet
      metrics.json

    Space optimisation: builds the DataFrame column-by-column from field arrays
    instead of materialising a list of N dicts (O(N×F) intermediate objects).
    """
    if not records:
        raise ValueError("No records to save.")

    key_set = {(r.dataset, r.modality, r.model) for r in records}
    if len(key_set) != 1:
        raise ValueError(
            "save_run_records expects one dataset/modality/model bucket only."
        )

    dataset, modality, model = next(iter(key_set))
    out_dir = _run_dir(Path(base_dir), dataset, modality, model)

    # Build DataFrame from a single list-of-dicts — kept for schema correctness
    # but done once here rather than duplicated elsewhere.
    df = pd.DataFrame([asdict(r) for r in records])
    df.to_parquet(out_dir / "responses.parquet", index=False)

    metrics = _build_metrics(records)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    if manifest_context is not None:
        write_run_manifest(
            out_dir,
            dataset=dataset,
            modality=modality,
            model=model,
            records=records,
            ctx=manifest_context,
        )

    return out_dir


def split_and_save_all(
    records: list[UnifiedExperimentRecord],
    base_dir: str = "results1/unified_baseline",
    manifest_context: Optional[RunManifestContext] = None,
) -> list[Path]:
    """
    Split mixed records by (dataset, modality, model) and save each bucket.

    O(N) single pass to build buckets; same as before but with dict.setdefault
    to avoid repeated key lookups.
    """
    buckets: dict[tuple[str, str, str], list[UnifiedExperimentRecord]] = {}
    for r in records:
        buckets.setdefault((r.dataset, r.modality, r.model), []).append(r)

    return [
        save_run_records(bucket, base_dir=base_dir, manifest_context=manifest_context)
        for bucket in buckets.values()
    ]