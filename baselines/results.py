from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from baselines.schema import UnifiedExperimentRecord, UnifiedRunMetrics


def _safe_name(value: str) -> str:
    return value.lower().replace(" ", "_").replace("-", "_").replace("/", "_")


def _run_dir(base_dir: Path, dataset: str, modality: str, model: str) -> Path:
    out = base_dir / _safe_name(dataset) / _safe_name(modality) / _safe_name(model)
    out.mkdir(parents=True, exist_ok=True)
    return out


def _build_metrics(records: list[UnifiedExperimentRecord]) -> UnifiedRunMetrics:
    if not records:
        raise ValueError("Cannot build metrics from empty records.")

    total_queries = len(records)
    correct_queries = sum(1 for r in records if r.is_correct)
    accuracy = correct_queries / total_queries if total_queries else 0.0

    total_prompt_tokens = sum(r.prompt_tokens for r in records)
    total_completion_tokens = sum(r.completion_tokens for r in records)
    total_tokens = sum(r.total_tokens for r in records)
    avg_tokens_per_query = total_tokens / total_queries if total_queries else 0.0

    total_cost_usd = sum(r.cost_usd for r in records)
    latencies = [r.latency_s for r in records]
    avg_latency_s = sum(latencies) / total_queries if total_queries else 0.0
    p50_latency_s = float(pd.Series(latencies).quantile(0.50)) if latencies else 0.0
    p95_latency_s = float(pd.Series(latencies).quantile(0.95)) if latencies else 0.0

    error_count = sum(1 for r in records if r.error_message)
    error_rate = error_count / total_queries if total_queries else 0.0

    extra_metadata: dict = {}
    if records[0].dataset == "gaia":
        # Level-wise GAIA breakdown for research reporting.
        level_rows: list[dict] = []
        for r in records:
            level = str((r.metadata or {}).get("level", "")).strip() or "unknown"
            level_rows.append({"level": level, "is_correct": bool(r.is_correct)})
        ldf = pd.DataFrame(level_rows)
        by_level = (
            ldf.groupby("level", dropna=False)["is_correct"]
            .agg(total="count", correct="sum", accuracy="mean")
            .reset_index()
        )
        extra_metadata["gaia_level_metrics"] = {
            str(row["level"]): {
                "total": int(row["total"]),
                "correct": int(row["correct"]),
                "accuracy": round(float(row["accuracy"]), 6),
            }
            for _, row in by_level.iterrows()
        }

    first = records[0]
    return UnifiedRunMetrics(
        dataset=first.dataset,
        modality=first.modality,
        model=first.model,
        system_prompt_version=first.system_prompt_version,
        user_prompt_version=first.user_prompt_version,
        prompt_hash=first.prompt_hash,
        total_queries=total_queries,
        correct_queries=correct_queries,
        accuracy=round(accuracy, 6),
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        avg_tokens_per_query=round(avg_tokens_per_query, 3),
        total_cost_usd=round(total_cost_usd, 6),
        avg_latency_s=round(avg_latency_s, 3),
        p50_latency_s=round(p50_latency_s, 3),
        p95_latency_s=round(p95_latency_s, 3),
        error_count=error_count,
        error_rate=round(error_rate, 6),
        metadata=extra_metadata,
    )


def save_run_records(
    records: list[UnifiedExperimentRecord],
    base_dir: str = "results/unified_baseline",
) -> Path:
    """
    Save one run bucket (same dataset/modality/model) to:
      responses.parquet
      metrics.json
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

    df = pd.DataFrame([asdict(r) for r in records])
    df.to_parquet(out_dir / "responses.parquet", index=False)

    metrics = _build_metrics(records)
    with (out_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(metrics), f, indent=2)

    return out_dir


def split_and_save_all(
    records: list[UnifiedExperimentRecord],
    base_dir: str = "results/unified_baseline",
) -> list[Path]:
    """
    Split mixed records by (dataset, modality, model) and save each bucket.
    """
    buckets: dict[tuple[str, str, str], list[UnifiedExperimentRecord]] = {}
    for r in records:
        buckets.setdefault((r.dataset, r.modality, r.model), []).append(r)

    saved: list[Path] = []
    for bucket in buckets.values():
        saved.append(save_run_records(bucket, base_dir=base_dir))
    return saved
