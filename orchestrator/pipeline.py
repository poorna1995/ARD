"""
Eval orchestrator: route → execute agents → grade → persist.

Single module (consolidated).
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from agent.dataset_profile import agent_kwargs_from_row
from agent.episode_context import datasets_requiring_episode_context
from agent.registry import STRATEGY_REGISTRY
from config.common import REPO_ROOT, USE_NEW_DATA_LAYOUT
from config.local.constants.datasets import eval_parquet_stem
from config.global_config.paths import eval_samples_dir, new_run_dir, orchestrator_default_root
from config.global_config.runtime import resolve_output_root
from evaluator.grade import grade as grade_answer
from input.prompts.prompts_core import DATASETS, TASK_DESCRIPTION
from router.config import (
    DEFAULT_SELECTIVE_TAU,
    SELECTIVE_CHEAP_ROUTER_PATH,
    SELECTIVE_FULL_ROUTER_PATH,
    SELECTIVE_TAU_JSON,
)
from router.router import (
    DEFAULT_AGENT_MODEL,
    ROUTER_MODEL_PATH,
    RuntimeRouter,
    ensure_eval_features,
    eval_base_frame,
    load_eval_parquet,
    load_router,
    load_router_frame,
    load_split,
    normalize_loader_frame,
    resolve_dataset_name,
    top_k_from_row,
)
from eval.score import score_routed, write_score_summary
from research.selective import SelectiveGateConfig, route_selective_split, selective_build_and_route

DEFAULT_OUTPUT_ROOT = resolve_output_root(orchestrator_default_root())
QCE_SPLITS = ("train", "val", "test")
ROUTE_AGENT_COLS = ("router_pred", "assigned_agent")
EXECUTION_DONE_COLS = ("predicted_answer", "response_predicted_answer")
_EPISODE_CONTEXT_COLS = (
    "context",
    "paragraphs",
    "passages",
    "episode_context",
    "metadata",
)


# --- run manifest ---
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
        # DEFAULT_OUTPUT_ROOT

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

# --- load ---
def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported routes file type: {path.suffix} (use .parquet or .csv)")


def _routes_has_usable_episode_context(df: pd.DataFrame) -> bool:
    for col in _EPISODE_CONTEXT_COLS:
        if col not in df.columns:
            continue
        series = df[col]
        if series.notna().any():
            return True
    return False


def _ensure_training_id(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "training_id" not in out.columns:
        if "id" in out.columns:
            out["training_id"] = out["id"].astype(str)
        else:
            raise KeyError("Routes/eval frame needs training_id or id")
    else:
        out["training_id"] = out["training_id"].astype(str)
    return out


def load_routes_frame(
    routes_path: str | Path,
    *,
    dataset: str | None = None,
    data_path: str | Path | None = None,
    split: str | None = None,
) -> pd.DataFrame:
    """
    Load a route-only artifact (``router_pred`` / proba columns) for execution.

    Merges eval rows from ``--dataset``, ``--data_path``, or ``--split`` when
    ``query`` is missing **or** when Hotpot/MuSiQue passage ``context`` is missing
    (required for ReAct ``retrieve``).
    """
    path = Path(routes_path)
    if not path.is_file():
        raise FileNotFoundError(f"Routes file not found: {path}")

    df = _ensure_training_id(_read_table(path))
    df = normalize_loader_frame(df)
    if not any(c in df.columns for c in ROUTE_AGENT_COLS):
        raise ValueError(f"Routes file must include one of {ROUTE_AGENT_COLS}: {path}")

    if "router_pred" not in df.columns:
        df["router_pred"] = df["assigned_agent"].astype(str)
    if "assigned_agent" not in df.columns:
        df["assigned_agent"] = df["router_pred"].astype(str)

    needs_query_merge = (
        "query" not in df.columns
        or df["query"].isna().all()
        or df["query"].astype(str).str.strip().eq("").all()
    )
    row_ds = str(df["dataset"].iloc[0]) if len(df) and "dataset" in df.columns else ""
    ds_key = resolve_dataset_name(dataset or row_ds)
    needs_context_merge = (
        ds_key in datasets_requiring_episode_context()
        and not _routes_has_usable_episode_context(df)
    )

    if needs_query_merge or needs_context_merge:
        if split is None and not dataset and data_path is None and needs_query_merge:
            raise ValueError(
                "Routes file has no query column; pass --dataset, --data_path, or --split to merge eval rows."
            )
        if needs_context_merge and split is None and not dataset and not data_path:
            raise ValueError(
                f"Routes for {ds_key!r} lack passage context; pass --dataset {ds_key} "
                "(or --data_path to eval parquet) when executing."
            )
        eval_df, _ = load_eval_frame(
            dataset=dataset or ds_key,
            data_path=data_path,
            split=split,
        )
        eval_df = _ensure_training_id(eval_df)
        key = "training_id"
        if key not in eval_df.columns:
            raise KeyError(f"Cannot merge routes with eval data without {key!r}")

        if needs_query_merge:
            merge_cols = [c for c in eval_df.columns if c not in df.columns or c == key]
        else:
            merge_cols = [key] + [
                c
                for c in eval_df.columns
                if c != key and (c not in df.columns or c in _EPISODE_CONTEXT_COLS)
            ]
        df = df.merge(eval_df[merge_cols], on=key, how="left", suffixes=("", "_eval"))
        if needs_query_merge and "query" not in df.columns:
            raise KeyError("Merged eval frame still missing query")

    if "expected_answer" not in df.columns:
        for alt in ("answer", "gold", "reference"):
            if alt in df.columns:
                df["expected_answer"] = df[alt]
                break

    return df


def load_eval_frame(
    *,
    dataset: str | None,
    data_path: str | Path | None,
    split: str | None,
) -> tuple[pd.DataFrame, str]:
    """
    Return (dataframe, feature_tag).

    ``feature_tag`` names on-disk QCE artifacts (e.g. gaia, test).
    """
    if split is not None:
        if split not in QCE_SPLITS:
            raise ValueError(f"--split must be one of {QCE_SPLITS}")
        df = load_split(split, with_embeddings=True)
        return df, split

    if not dataset:
        raise ValueError("Provide --dataset or --split")

    ds_key = resolve_dataset_name(dataset)
    if ds_key not in TASK_DESCRIPTION and ds_key not in DATASETS:
        known = sorted(set(DATASETS) | set(TASK_DESCRIPTION))
        raise ValueError(f"Unknown dataset {dataset!r}. Known keys: {known}")

    if data_path is not None:
        df = eval_base_frame(Path(data_path), dataset=ds_key)
        tag = Path(data_path).stem
    else:
        df = load_eval_parquet(ds_key)
        tag = eval_parquet_stem(ds_key)
    return df, tag


# --- persist ---
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

# --- route stage ---
def resolve_selective_config(
    *,
    selective_qce: bool,
    selective_tau: float | None,
    selective_tau_margin: float | None,
    selective_cheap_router: str | Path | None,
    selective_full_router: str | Path | None,
) -> SelectiveGateConfig | None:
    if not selective_qce:
        return None
    tau = selective_tau
    if tau is None and SELECTIVE_TAU_JSON.is_file():
        try:
            data = json.loads(SELECTIVE_TAU_JSON.read_text(encoding="utf-8"))
            tau = float(data.get("recommended_tau", DEFAULT_SELECTIVE_TAU))
        except (json.JSONDecodeError, TypeError, ValueError):
            tau = DEFAULT_SELECTIVE_TAU
    if tau is None:
        tau = DEFAULT_SELECTIVE_TAU
    return SelectiveGateConfig(
        tau=tau,
        tau_margin=selective_tau_margin,
        cheap_router_path=Path(selective_cheap_router or SELECTIVE_CHEAP_ROUTER_PATH),
        full_router_path=Path(selective_full_router or SELECTIVE_FULL_ROUTER_PATH),
    )


def routing_summary(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "n": len(df),
        "router_distribution": df["router_pred"].value_counts().to_dict()
        if "router_pred" in df.columns
        else {},
    }
    if "selective_path" in df.columns:
        summary["selective_path_distribution"] = df["selective_path"].value_counts().to_dict()
    if "used_decompose" in df.columns:
        summary["pct_decompose"] = round(100.0 * float(df["used_decompose"].mean()), 1)
    if "selective_tau" in df.columns and len(df):
        summary["selective_tau"] = float(df["selective_tau"].iloc[0])
    if "oracle_agent" in df.columns and "router_pred" in df.columns:
        mask = df["oracle_agent"].astype(str).isin(STRATEGY_REGISTRY)
        sub = df[mask]
        summary["n_with_oracle"] = int(len(sub))
        if len(sub):
            summary["route_accuracy_vs_oracle"] = float(
                (sub["router_pred"] == sub["oracle_agent"]).mean()
            )
    if "is_correct" in df.columns:
        summary["execution_accuracy"] = float(
            pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).mean()
        )
    return summary


def row_execution_done(row: Mapping[str, Any], *, retry_failed: bool = False) -> bool:
    """True if this row already has a stored agent answer (skip on resume)."""
    if retry_failed and row.get("is_failed"):
        return False
    for key in EXECUTION_DONE_COLS:
        if str(row.get(key) or "").strip():
            return True
    return False


def done_training_ids(df: pd.DataFrame, *, retry_failed: bool = False) -> set[Any]:
    if df is None or df.empty or "training_id" not in df.columns:
        return set()
    return {
        r["training_id"]
        for r in df.to_dict(orient="records")
        if row_execution_done(r, retry_failed=retry_failed)
    }


def merge_pipeline_results(
    routes_df: pd.DataFrame,
    existing: pd.DataFrame | None,
    new_records: list[dict[str, Any]],
) -> pd.DataFrame:
    """Merge prior + new execution rows in ``routes_df`` order (full route set)."""
    by_id: dict[Any, dict[str, Any]] = {}
    if existing is not None and not existing.empty:
        for rec in existing.to_dict(orient="records"):
            by_id[rec.get("training_id")] = rec
    for rec in new_records:
        by_id[rec.get("training_id")] = rec

    out: list[dict[str, Any]] = []
    for row in routes_df.to_dict(orient="records"):
        tid = row.get("training_id")
        if tid in by_id:
            out.append(by_id[tid])
    return pd.DataFrame.from_records(out)


# --- pipeline ---
def run_pipeline(
    *,
    dataset: str | None = "gaia",
    data_path: str | Path | None = None,
    split: str | None = None,
    routes_path: str | Path | None = None,
    append_to: str | Path | None = None,
    retry_failed: bool = False,
    output_path: str | Path | None = None,
    jsonl_output_path: str | Path | None = None,
    router_path: str | Path | None = None,
    build_features: bool = False,
    force_refresh_features: bool = False,
    execute_agents: bool = True,
    react_only: bool = False,
    cascade_on_fail: bool = False,
    cascade_k: int = 3,
    grade: bool = False,
    model: str = DEFAULT_AGENT_MODEL,
    limit: int | None = None,
    stream: bool = True,
    checkpoint_every: int = 1,
    fresh: bool = False,
    verbose: bool = True,
    selective_qce: bool = False,
    selective_tau: float | None = None,
    selective_tau_margin: float | None = None,
    selective_cheap_router: str | Path | None = None,
    selective_full_router: str | Path | None = None,
    score_lookup: bool = True,
) -> pd.DataFrame:
    selective_config = resolve_selective_config(
        selective_qce=selective_qce,
        selective_tau=selective_tau,
        selective_tau_margin=selective_tau_margin,
        selective_cheap_router=selective_cheap_router,
        selective_full_router=selective_full_router,
    )
    if routes_path is not None and selective_config is not None and verbose:
        print("Note: --selective-qce ignored with --routes-path (pre-routed file).")
    if routes_path is not None and (build_features or force_refresh_features):
        if verbose:
            print("Note: --build-features / --force-refresh-features ignored with --routes-path.")

    score_dataset: str | None = None
    score_split: str | None = None

    if routes_path is not None:
        routed_df = load_routes_frame(
            routes_path,
            dataset=dataset,
            data_path=data_path,
            split=split,
        )
        tag = Path(routes_path).stem
        score_dataset = dataset.strip().lower() if dataset and split is None else None
        score_split = split
        if verbose:
            dist = routed_df["router_pred"].value_counts().to_dict()
            print(f"Loaded {len(routed_df)} pre-routed rows from {routes_path}")
            print(f"  router_pred distribution: {dist}")
    else:
        if selective_config is not None and split is not None:
            if build_features and verbose:
                print("Note: --build-features ignored for QCE splits (prebuilt parquets).")
            if verbose:
                print(
                    f"Selective QCE routing split={split!r} τ={selective_config.tau} "
                    f"(cheap={selective_config.cheap_router_path.name})…"
                )
            routed_df = route_selective_split(split, selective_config)
            tag = split
            score_split = split
        else:
            df, tag = load_eval_frame(dataset=dataset, data_path=data_path, split=split)
            score_split = split
            score_dataset = resolve_dataset_name(dataset) if split is None else None

            if selective_config is not None:
                base_cols = [
                    c
                    for c in df.columns
                    if not c.startswith("emb_")
                    and not c.startswith("dim_")
                    and c != "oracle_agent"
                ]
                base = df[base_cols].copy()
                if "training_id" not in base.columns:
                    raise KeyError("Eval frame missing training_id")
                if verbose:
                    print(
                        f"Selective QCE routing tag={tag!r} τ={selective_config.tau} "
                        f"(build_features={build_features})…"
                    )
                routed_df = selective_build_and_route(
                    base,
                    tag,
                    selective_config,
                    build=build_features,
                    force_refresh=force_refresh_features,
                    verbose=verbose,
                )
            elif split is None:
                base_cols = [c for c in df.columns if not c.startswith("emb_") and c != "oracle_agent"]
                base = df[base_cols].copy()
                if "training_id" not in base.columns:
                    raise KeyError("Eval frame missing training_id")
                df = ensure_eval_features(
                    base,
                    tag,
                    build=build_features,
                    force_refresh=force_refresh_features,
                    verbose=verbose,
                )
                if verbose:
                    print(f"Routing {len(df)} rows with {router_path or ROUTER_MODEL_PATH}…")
                routed_df = load_router_frame(df, router_path=router_path, feature_tag=tag)
            else:
                if build_features and verbose:
                    print("Note: --build-features ignored for QCE splits (use qce pipeline).")
                if verbose:
                    print(f"Routing {len(df)} rows with {router_path or ROUTER_MODEL_PATH}…")
                routed_df = load_router_frame(df, router_path=router_path, feature_tag=tag)

    routes_full_df = routed_df
    experiment_id = Path(router_path or ROUTER_MODEL_PATH).stem
    output, jsonl_output, run_dir = resolve_output_paths(
        experiment_id=experiment_id,
        output_path=output_path,
        jsonl_output_path=jsonl_output_path,
        append_to=append_to,
        split=split,
        dataset=dataset,
    )

    existing_df: pd.DataFrame | None = None
    resume_source: str | None = None
    if append_to is not None:
        append_path = Path(append_to)
        if not append_path.is_file():
            raise FileNotFoundError(f"--append-to file not found: {append_path}")
        existing_df = _read_table(append_path)
        resume_source = str(append_path)
    elif execute_agents and not fresh:
        existing_df = load_existing_results(output, jsonl_output)
        if existing_df is not None and not existing_df.empty:
            resume_source = str(output if output.is_file() else jsonl_output)

    if fresh and execute_agents and verbose:
        for path in (output, jsonl_output, output.with_suffix(".summary.json")):
            if path.is_file():
                path.unlink()

    done_ids: set[Any] = set()
    if existing_df is not None and not existing_df.empty:
        done_ids = done_training_ids(existing_df, retry_failed=retry_failed)
        pending_df = routes_full_df[~routes_full_df["training_id"].isin(done_ids)].copy()
        if verbose:
            print(
                f"Resume from {resume_source}: {len(done_ids)} done, "
                f"{len(pending_df)} pending (of {len(routes_full_df)} routes)"
            )
        routed_df = pending_df
    else:
        pending_df = routed_df

    if limit is not None:
        routed_df = routed_df.head(limit).copy()

    if stream and execute_agents and not fresh and existing_df is None and verbose:
        print(f"Streaming: {jsonl_output} (append) + checkpoint parquet every {checkpoint_every} row(s)")

    records: list[dict[str, Any]] = []
    total = len(routed_df)
    n_routes = len(routes_full_df)
    for idx, row in enumerate(routed_df.to_dict(orient="records"), start=1):
        query = str(row.get("query") or "").strip()
        expected = row.get("expected_answer") or row.get("answer")
        row_dataset = resolve_dataset_name(
            str(row.get("dataset_source") or row.get("dataset") or dataset or "gaia")
        )
        run_kw = agent_kwargs_from_row(row, dataset=row_dataset)
        assigned = str(row.get("assigned_agent") or row.get("router_pred") or "")

        if verbose:
            preview = query[:100].replace("\n", " ")
            prog = f"[{idx}/{total}]"
            if append_to is not None:
                prog = f"[{len(done_ids) + idx}/{n_routes}] ({idx}/{total} pending)"
            print(
                f"{prog} router={assigned} "
                f"max_prob={row.get('max_prob', 0):.3f} overall={row.get('overall')} "
                f"| {preview}"
            )

        top3 = top_k_from_row(row, k=cascade_k)
        record: dict[str, Any] = {
            "training_id": row.get("training_id"),
            "dataset": row_dataset,
            "query": query,
            "expected_answer": expected,
            "assigned_agent": assigned,
            "router_pred": assigned,
            "overall": row.get("overall"),
            "max_prob": row.get("max_prob"),
            "margin_top2": row.get("margin_top2"),
            "router_second": row.get("router_second") or (top3[1][0] if len(top3) > 1 else None),
            "router_third": row.get("router_third") or (top3[2][0] if len(top3) > 2 else None),
            "top3_agents": top3,
            "router_experiment": row.get("router_experiment"),
            "p_raw": row.get("p_raw"),
            "p_cot": row.get("p_cot"),
            "p_react": row.get("p_react"),
            "p_multiagent": row.get("p_multiagent"),
        }
        if row.get("oracle_agent") is not None:
            record["oracle_agent"] = row.get("oracle_agent")
        for key in (
            "selective_path",
            "used_decompose",
            "selective_tau",
            "max_prob_emb",
            "margin_top2_emb",
        ):
            if key in row:
                record[key] = row.get(key)

        should_run = execute_agents and assigned in STRATEGY_REGISTRY
        if react_only and assigned != "react":
            should_run = False

        if should_run:
            router = RuntimeRouter.from_dataframe_row(
                row, model=model, router_path=router_path, **run_kw
            )
            try:
                if cascade_on_fail:
                    if verbose:
                        agents_s = " → ".join(a for a, _ in top3)
                        print(f"[{idx}/{total}] Cascade (top-{cascade_k}): {agents_s} …")
                    run_result = router.run_cascade(expected_answer=expected, k=cascade_k)
                    response = response_to_dict(run_result.get("response"))
                    record.update(
                        {
                            "executed_agent": run_result.get("executed_agent"),
                            "cascade_rank": run_result.get("cascade_rank"),
                            "cascade_attempts": run_result.get("cascade_attempts"),
                        }
                    )
                else:
                    if verbose:
                        print(f"[{idx}/{total}] Executing {assigned} ({model})…")
                    run_result = router.run(expected_answer=expected)
                    response = response_to_dict(run_result.get("response"))
                    record["executed_agent"] = assigned
                    record["cascade_rank"] = 1

                pred = response.get("predicted_answer") or response.get("answer") or ""
                record.update(
                    {
                        "model": run_result.get("model"),
                        "predicted_answer": pred,
                        **{f"response_{k}": v for k, v in response.items()},
                    }
                )
                if grade and expected is not None:
                    record["is_correct"] = grade_answer(
                        str(pred), str(expected), dataset=row_dataset
                    )
                if verbose:
                    exec_agent = record.get("executed_agent", assigned)
                    print(
                        f"[{idx}/{total}] Done agent={exec_agent} "
                        f"rank={record.get('cascade_rank')} failed={response.get('is_failed')} "
                        f"correct={record.get('is_correct')} pred={str(pred)[:80]!r}"
                    )
            except Exception as exc:
                record["is_failed"] = True
                record["error"] = str(exc)
                if verbose:
                    print(f"[{idx}/{total}] Failed: {exc}")
        elif execute_agents and verbose:
            print(f"[{idx}/{total}] Skipped execution (react_only={react_only}).")

        records.append(record)

        if stream and execute_agents:
            append_jsonl_record(jsonl_output, record)
            if checkpoint_every > 0 and idx % checkpoint_every == 0:
                checkpoint_df = merge_pipeline_results(routes_full_df, existing_df, records)
                checkpoint_summary = routing_summary(checkpoint_df)
                if append_to is not None or existing_df is not None:
                    checkpoint_summary["n_pending_this_run"] = total
                    checkpoint_summary["n_done_before"] = len(done_ids)
                    checkpoint_summary["n_total_routes"] = n_routes
                if grade and "is_correct" in checkpoint_df.columns:
                    checkpoint_summary["graded_accuracy"] = float(
                        pd.to_numeric(checkpoint_df["is_correct"], errors="coerce").fillna(0).mean()
                    )
                write_pipeline_artifacts(
                    checkpoint_df,
                    output=output,
                    jsonl_output=jsonl_output,
                    summary=checkpoint_summary,
                    stream=True,
                )
                if verbose:
                    acc = checkpoint_summary.get("graded_accuracy") or checkpoint_summary.get(
                        "execution_accuracy"
                    )
                    acc_s = f"{acc:.3f}" if acc is not None else "n/a"
                    print(f"[checkpoint] {len(checkpoint_df)}/{n_routes} rows saved (acc={acc_s})")

    if not execute_agents:
        results_df = routes_full_df.copy()
    elif append_to is not None or existing_df is not None:
        results_df = merge_pipeline_results(routes_full_df, existing_df, records)
    else:
        results_df = (
            merge_pipeline_results(routes_full_df, None, records)
            if records
            else pd.DataFrame()
        )

    summary = routing_summary(results_df if len(results_df) else routes_full_df)
    if append_to is not None or existing_df is not None:
        summary["n_pending_this_run"] = total
        summary["n_done_before"] = len(done_ids)
        summary["n_total_routes"] = n_routes
    if grade and "is_correct" in results_df.columns:
        summary["graded_accuracy"] = float(
            pd.to_numeric(results_df["is_correct"], errors="coerce").fillna(0).mean()
        )
    if cascade_on_fail and "cascade_rank" in results_df.columns:
        ranks = pd.to_numeric(results_df["cascade_rank"], errors="coerce")
        summary["cascade_mean_rank"] = float(ranks.mean())
        summary["cascade_frac_needed_fallback"] = float((ranks > 1).mean())

    if score_lookup and not execute_agents:
        try:
            router_obj = load_router(router_path or ROUTER_MODEL_PATH)
            score_feature_set = router_obj.get("feature_set")
            scored, lookup = score_routed(
                routes_full_df,
                dataset=score_dataset,
                split=score_split,
                feature_set=score_feature_set,
            )
            results_df = scored
            summary["lookup_score"] = lookup
            write_score_summary(lookup, output.with_suffix(".lookup_score.json"))
            if verbose:
                em = lookup.get("em_pct")
                musd = lookup.get("musd")
                total_musd = lookup.get("total_musd")
                print(
                    f"Lookup score: EM={em}%  agent_mUSD={musd}  "
                    f"total_mUSD={total_musd}  (source={lookup.get('outcome_source')})"
                )
        except Exception as exc:
            summary["lookup_score_error"] = str(exc)
            if verbose:
                print(f"Lookup score skipped: {exc}")

    write_pipeline_artifacts(
        results_df,
        output=output,
        jsonl_output=jsonl_output,
        summary=summary,
        stream=stream,
    )

    if run_dir is not None:
        manifest_path = manifest_path_for_output(output, run_dir)
    else:
        manifest_path = manifest_path_for_output(output, None)
    manifest = build_run_manifest(
        experiment_id=experiment_id,
        summary=summary,
        router_path=router_path or ROUTER_MODEL_PATH,
        dataset=dataset,
        split=split,
        routes_path=routes_path,
        output_path=output,
        model=model,
        execute_agents=execute_agents,
        grade=grade,
    )
    write_run_manifest(manifest_path, manifest)
    if verbose:
        print(f"Run manifest: {manifest_path}")

    if verbose:
        print(f"Saved: {output}")
        print(f"Stream log: {jsonl_output}")
        print(f"Summary: {output.with_suffix('.summary.json')}")
        print(json.dumps(summary, indent=2))
    return results_df



def _build_parser() -> argparse.ArgumentParser:
    eval_datasets = sorted({p.stem for p in eval_samples_dir().glob("*.parquet")})
    parser = argparse.ArgumentParser(
        description="QCE features → G3 router → optional agent run (eval orchestrator)."
    )
    parser.add_argument(
        "--dataset",
        choices=eval_datasets,
        default="gaia",
        help="Eval parquet under datasets/eval_samples/ (ignored if --split set).",
    )
    parser.add_argument("--data_path", default=None, help="Custom eval parquet path.")
    parser.add_argument(
        "--split",
        choices=QCE_SPLITS,
        default=None,
        help="Use QCE v1 split (train/val/test) instead of eval_samples parquet.",
    )
    parser.add_argument("--output_path", default=None)
    parser.add_argument("--jsonl_output_path", default=None)
    parser.add_argument(
        "--router-path",
        dest="router_path",
        default=None,
        help=f"Trained router joblib (default: {ROUTER_MODEL_PATH.name}).",
    )
    parser.add_argument(
        "--build-features",
        action="store_true",
        help="Decompose + C(Q) + embeddings for eval parquet (needs OPENAI_API_KEY).",
    )
    parser.add_argument(
        "--force-refresh-features",
        action="store_true",
        help="Re-run LLM decompose even if cache exists.",
    )
    parser.add_argument(
        "--routes-path",
        dest="routes_path",
        default=None,
        help="Parquet/CSV with router_pred (and optional p_*). Skip routing; run agents on these labels.",
    )
    parser.add_argument(
        "--append-to",
        dest="append_to",
        default=None,
        help="Existing pipeline output; skip rows with predicted_answer, run the rest, merge back.",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="With --append-to, re-run rows that failed or have empty predicted_answer.",
    )
    parser.add_argument(
        "--route-only",
        action="store_true",
        help="Route only; skip agent execution. Incompatible with --routes-path.",
    )
    parser.add_argument(
        "--react-only",
        action="store_true",
        help="Execute only rows assigned to react.",
    )
    parser.add_argument(
        "--cascade",
        action="store_true",
        help="On agent failure, try 2nd then 3rd ranked agents (top-3 by router proba).",
    )
    parser.add_argument(
        "--cascade-k",
        type=int,
        default=3,
        help="Max agents to try in cascade mode (default: 3).",
    )
    parser.add_argument(
        "--grade",
        action="store_true",
        help="Grade predicted answers vs expected (when executing agents).",
    )
    parser.add_argument("--model", default=DEFAULT_AGENT_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing output/jsonl at --output_path and start from scratch.",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable per-row JSONL append and incremental parquet checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=1,
        metavar="N",
        help="Write parquet + summary every N completed rows when streaming (default: 1).",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--selective-qce",
        action="store_true",
        help="Phase 2: emb_only gate → decompose only if uncertain (see docs/selective_qce_phase2.md).",
    )
    parser.add_argument(
        "--selective-tau",
        type=float,
        default=None,
        help=f"Confidence threshold on cheap router max_prob (default: {SELECTIVE_TAU_JSON.name} or {DEFAULT_SELECTIVE_TAU}).",
    )
    parser.add_argument(
        "--selective-tau-margin",
        type=float,
        default=None,
        help="Also require (top1 - top2) prob >= this on cheap router.",
    )
    parser.add_argument(
        "--selective-cheap-router",
        default=None,
        help=f"Cheap gate model (default: {SELECTIVE_CHEAP_ROUTER_PATH.name}).",
    )
    parser.add_argument(
        "--selective-full-router",
        default=None,
        help=f"Full model for uncertain rows (default: {SELECTIVE_FULL_ROUTER_PATH.name}).",
    )
    parser.add_argument(
        "--no-score-lookup",
        action="store_true",
        help="Skip EM/mUSD from baseline or oracle lookup after route-only (no agent re-runs).",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    from config.global_config.runtime import configure_logging

    configure_logging()
    args = _build_parser().parse_args(argv)
    if args.routes_path and args.route_only:
        raise SystemExit("Use either --routes-path (execute saved routes) or --route-only, not both.")
    if args.append_to and not args.routes_path:
        raise SystemExit("--append-to requires --routes-path (full routed GAIA set).")
    run_pipeline(
        dataset=args.dataset if not args.split else None,
        data_path=args.data_path,
        split=args.split,
        routes_path=args.routes_path,
        append_to=args.append_to,
        retry_failed=args.retry_failed,
        output_path=args.output_path,
        jsonl_output_path=args.jsonl_output_path,
        router_path=args.router_path,
        build_features=args.build_features,
        force_refresh_features=args.force_refresh_features,
        execute_agents=not args.route_only,
        react_only=args.react_only,
        cascade_on_fail=args.cascade,
        cascade_k=args.cascade_k,
        grade=args.grade,
        model=args.model,
        limit=args.limit,
        stream=not args.no_stream,
        checkpoint_every=max(0, args.checkpoint_every),
        fresh=args.fresh,
        verbose=not args.quiet,
        selective_qce=args.selective_qce,
        selective_tau=args.selective_tau,
        selective_tau_margin=args.selective_tau_margin,
        selective_cheap_router=args.selective_cheap_router,
        selective_full_router=args.selective_full_router,
        score_lookup=not args.no_score_lookup,
    )


if __name__ == "__main__":
    main()
