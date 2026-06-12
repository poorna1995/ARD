"""Checkpoint, JSONL, and parquet persistence for orchestrator runs."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from orchestrator.load import _read_table
from orchestrator.run_manifest import orchestrator_output_paths


def response_to_dict(response: Any) -> dict[str, Any]:
    if hasattr(response, "__dataclass_fields__"):
        return asdict(response)
    if isinstance(response, dict):
        return response
    return {"raw": str(response)}


def dataframe_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        series = out[col]
        if series.dtype != object:
            continue
        if series.map(lambda x: isinstance(x, (dict, list))).any():
            out[col] = series.map(
                lambda x: (
                    json.dumps(x, ensure_ascii=False, default=str)
                    if isinstance(x, (dict, list))
                    else x
                )
            )
    return out


def resolve_output_paths(
    *,
    experiment_id: str,
    output_path: str | Path | None,
    jsonl_output_path: str | Path | None,
    append_to: str | Path | None,
    split: str | None,
    dataset: str | None,
) -> tuple[Path, Path, Path | None]:
    tag = split or (dataset or "eval")
    output, jsonl, run_dir = orchestrator_output_paths(
        experiment_id=experiment_id,
        tag=tag,
        output_path=output_path,
        jsonl_output_path=jsonl_output_path,
        append_to=append_to,
    )
    return output, jsonl, run_dir


def load_jsonl_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_existing_results(
    output: Path,
    jsonl_output: Path,
) -> pd.DataFrame | None:
    """Load partial/finished run from parquet (preferred) or streamed JSONL."""
    if output.is_file():
        return _read_table(output)
    if jsonl_output.is_file():
        records = load_jsonl_records(jsonl_output)
        if records:
            df = pd.DataFrame.from_records(records)
            if "training_id" in df.columns:
                df = df.drop_duplicates(subset="training_id", keep="last")
            return df
    return None


def append_jsonl_record(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")


def write_pipeline_artifacts(
    results_df: pd.DataFrame,
    *,
    output: Path,
    jsonl_output: Path,
    summary: dict[str, Any],
    stream: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    dataframe_for_parquet(results_df).to_parquet(output, index=False)
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not stream:
        jsonl_output.parent.mkdir(parents=True, exist_ok=True)
        with jsonl_output.open("w", encoding="utf-8") as f:
            for record in results_df.to_dict(orient="records"):
                f.write(json.dumps(record, ensure_ascii=True, default=str) + "\n")
