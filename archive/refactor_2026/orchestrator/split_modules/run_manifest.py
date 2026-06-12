"""Run directory layout and ``manifest.json`` for orchestrator runs."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from config.paths import REPO_ROOT, USE_NEW_DATA_LAYOUT, new_run_dir


def git_commit_short() -> str | None:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return out.strip() or None
    except (OSError, subprocess.CalledProcessError):
        return None


def orchestrator_output_paths(
    *,
    experiment_id: str,
    tag: str,
    output_path: str | Path | None,
    jsonl_output_path: str | Path | None,
    append_to: str | Path | None,
) -> tuple[Path, Path, Path | None]:
    """
    Return (parquet, jsonl, run_dir).

    When ``RESEARCH_USE_NEW_PATHS=1`` and no explicit output, writes under
    ``results/runs/{date}_{experiment_id}/orchestrator/{tag}/``.
    """
    run_dir: Path | None = None

    if output_path is not None:
        output = Path(output_path)
    elif append_to is not None:
        output = Path(append_to)
    elif USE_NEW_DATA_LAYOUT:
        run_dir = new_run_dir(experiment_id)
        output = run_dir / "orchestrator" / tag / "pipeline_results.parquet"
    else:
        from orchestrator.load import DEFAULT_OUTPUT_ROOT

        output = DEFAULT_OUTPUT_ROOT / tag / "pipeline_results.parquet"

    jsonl = Path(jsonl_output_path) if jsonl_output_path else output.with_suffix(".jsonl")
    return output, jsonl, run_dir


def manifest_path_for_output(output: Path, run_dir: Path | None) -> Path:
    if run_dir is not None:
        return run_dir / "manifest.json"
    return output.parent / "manifest.json"


def build_run_manifest(
    *,
    experiment_id: str,
    summary: dict[str, Any],
    router_path: str | Path | None = None,
    dataset: str | None = None,
    split: str | None = None,
    routes_path: str | Path | None = None,
    output_path: str | Path | None = None,
    model: str | None = None,
    execute_agents: bool | None = None,
    grade: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "created_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit_short(),
        "layout": "new" if USE_NEW_DATA_LAYOUT else "legacy",
        "router_path": str(router_path) if router_path else None,
        "dataset": dataset,
        "split": split,
        "routes_path": str(routes_path) if routes_path else None,
        "output_path": str(output_path) if output_path else None,
        "model": model,
        "execute_agents": execute_agents,
        "grade": grade,
        "summary": summary,
    }
    if extra:
        payload.update(extra)
    return payload


def write_run_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path
