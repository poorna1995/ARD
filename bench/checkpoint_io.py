from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _parse_json_objects_from_line(line: str) -> list[dict[str, Any]]:
    """Parse one or more JSON objects from a single line (handles glued checkpoints)."""
    out: list[dict[str, Any]] = []
    dec = json.JSONDecoder()
    pos = 0
    n = len(line)
    while pos < n:
        while pos < n and line[pos] in " \t":
            pos += 1
        if pos >= n:
            break
        obj, end = dec.raw_decode(line, pos)
        if not isinstance(obj, dict):
            raise ValueError(f"expected JSON object at char {pos}, got {type(obj).__name__}")
        out.append(obj)
        pos = end
    return out


def load_jsonl(path: Path, *, dedupe: bool = True) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.extend(_parse_json_objects_from_line(line))
    if dedupe:
        by_id: dict[str, dict[str, Any]] = {}
        for rec in records:
            tid = _record_key(rec)
            if tid is not None:
                by_id[str(tid)] = rec
        records = list(by_id.values())
    return _sort_training_records(records)


def _record_key(rec: dict[str, Any]) -> Any:
    """Stable row id for dedupe / resume (training bench or GAIA eval UUID)."""
    return rec.get("training_id") or rec.get("row_id") or rec.get("id")


def _sort_training_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda r: str(_record_key(r) or ""))


def write_checkpoint_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for rec in _sort_training_records(records):
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def load_done_ids(checkpoint_jsonl: Path) -> set[str]:
    return {
        str(k)
        for r in load_jsonl(checkpoint_jsonl, dedupe=True)
        if (k := _record_key(r)) is not None
    }


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
