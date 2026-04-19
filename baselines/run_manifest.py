from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from baselines.schema import UnifiedExperimentRecord


@dataclass(frozen=True)
class RunManifestContext:
    """Everything needed to write ``run_manifest.json`` beside ``metrics.json``."""

    repo_root: Path
    cli_args: dict[str, Any]
    dataset_paths: dict[str, str]
    run_selection: dict[str, Any]


def _json_safe(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj.resolve())
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return repr(obj)


def _sha256_file(path: Path, chunk: int = 1 << 20) -> tuple[Optional[str], Optional[str]]:
    """Return (hex_digest, error_message)."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            while True:
                block = f.read(chunk)
                if not block:
                    break
                h.update(block)
        return h.hexdigest(), None
    except OSError as exc:
        return None, str(exc)


def _git_info(repo: Path) -> dict[str, Any]:
    info: dict[str, Any] = {"commit": None, "is_dirty": None, "error": None}
    try:
        r = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if r.returncode == 0:
            info["commit"] = r.stdout.strip()
        r2 = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
        if r2.returncode == 0:
            info["is_dirty"] = bool(r2.stdout.strip())
    except (OSError, subprocess.TimeoutExpired) as exc:
        info["error"] = str(exc)
    return info


def _openai_version() -> Optional[str]:
    try:
        from importlib.metadata import version

        return version("openai")
    except Exception:
        return None


def build_run_manifest(
    *,
    dataset: str,
    modality: str,
    model: str,
    records: list[UnifiedExperimentRecord],
    ctx: RunManifestContext,
) -> dict[str, Any]:
    first = records[0]
    datasets_block: dict[str, Any] = {}
    for ds, pstr in sorted(ctx.dataset_paths.items()):
        p = Path(pstr)
        if not p.is_absolute():
            p = (ctx.repo_root / p).resolve()
        entry: dict[str, Any] = {"path": str(p)}
        if p.is_file():
            entry["size_bytes"] = p.stat().st_size
            digest, err = _sha256_file(p)
            entry["sha256"] = digest
            if err:
                entry["sha256_error"] = err
        else:
            entry["error"] = "path_not_found_or_not_file"

        datasets_block[ds] = entry

    return {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bucket": {
            "dataset": dataset,
            "modality": modality,
            "model": model,
            "n_records": len(records),
            "prompt_hash": first.prompt_hash,
            "system_prompt_version": first.system_prompt_version,
            "user_prompt_version": first.user_prompt_version,
        },
        "git": _git_info(ctx.repo_root),
        "environment": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
            "openai_package": _openai_version(),
        },
        "cli_args": _json_safe(ctx.cli_args),
        "run_selection": _json_safe(dict(ctx.run_selection)),
        "dataset_files": datasets_block,
    }


def write_run_manifest(
    out_dir: Path,
    *,
    dataset: str,
    modality: str,
    model: str,
    records: list[UnifiedExperimentRecord],
    ctx: RunManifestContext,
) -> Path:
    payload = build_run_manifest(
        dataset=dataset,
        modality=modality,
        model=model,
        records=records,
        ctx=ctx,
    )
    path = out_dir / "run_manifest.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=False)
        f.write("\n")
    return path
