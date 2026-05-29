"""
Cost-aware soft routing labels from oracle benchmark results.

Single module for:
  - per-question soft distribution over correct agents
  - attaching ``p_*`` columns to label tables
  - CLI to update QCE split CSVs from ``oracle_results*.csv``

Usage::

    uv run python -m src.utils.soft_labels --update-qce-splits
    uv run python src/utils/soft_labels.py --update-qce-splits
    uv run python src/utils/soft_labels.py --sweep-temperature
    uv run python src/utils/soft_labels.py --sweep-temperature --by-dataset
    uv run python src/utils/soft_labels.py --update-qce-splits --auto-temperature
    uv run python src/utils/soft_labels.py --update-qce-splits \\
        --temperature-map math:50,hotpot:100,musique:100
    uv run python -m src.utils.soft_labels --regenerate-oracle-agent --update-qce-splits
    uv run python -m src.utils.soft_labels --append-missing-solvable --with-qce-features
    uv run python scripts/describe_qce_split_filter.py --append-missing-solvable --with-qce-features
    # oracle_agent / y: utility cost-aware label, U = perf - DEFAULT_UTILITY_LAMBDA (25) * cost_usd
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

# Routing agents for this project (4 strategies).
DEFAULT_AGENTS: tuple[str, ...] = (
    "raw",
    "cot",
    "react",
    "multiagent",
)
DEFAULT_UTILITY_TIEBREAK: tuple[str, ...] = DEFAULT_AGENTS

ORACLE_AGENT_COL = "agent"
SOFT_COL_PREFIX = "p_"
SOFT_SUM_COL = "soft_label_sum"
SOFT_DOMINANT_COL = "soft_dominant_agent"
TARGET_COL = "y"
DATASET_COL = "dataset"

V1_SPLITS: tuple[str, ...] = (
    "qce_train.csv",
    "qce_val.csv",
    "qce_internal_test.csv",
)
POOL_LABELS_NAME = "train_labels1.csv"
SPLIT_KEYS: tuple[str, ...] = ("train", "val", "test")

# Utility λ for ``oracle_agent`` / ``y`` (cost-aware oracle): argmax_s (perf - λ * cost_usd).
# QCE v1 splits were built with λ=25.0 (``--regenerate-oracle-agent --update-qce-splits``).
DEFAULT_UTILITY_LAMBDA: float = 25.0

# Default λ values for ``--sweep-temperature`` (cost penalty, not LLM temperature).
DEFAULT_TEMPERATURE_SWEEP: tuple[float, ...] = (
    0.0,
    25.0,
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
)


def _tiebreak_index(agent: str, order: tuple[str, ...]) -> int:
    try:
        return order.index(agent)
    except ValueError:
        return len(order)


def repo_root() -> Path:
    """Project root (directory containing ``pyproject.toml``)."""
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd()


def default_v1_dir() -> Path:
    return repo_root() / "datasets" / "train_samples" / "v1"


def soft_probability_columns(agents: Iterable[str] = DEFAULT_AGENTS) -> list[str]:
    return [f"{SOFT_COL_PREFIX}{a}" for a in agents]


def load_oracle(path: str | Path, *, agents: tuple[str, ...] = DEFAULT_AGENTS) -> pd.DataFrame:
    """Load long-form oracle CSV; keep rows for ``agents`` only."""
    df = pd.read_csv(path)
    required = {
        "training_id",
        ORACLE_AGENT_COL,
        "is_correct",
        "is_failed",
        "predicted_answer",
        "cost_usd",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"oracle missing columns: {sorted(missing)}")
    df = df[df[ORACLE_AGENT_COL].isin(agents)].copy()
    return df


def regenerate_oracle_agent_labels(
    labels: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    utility_lambda: float,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    cost_col: str = "cost_usd",
    id_col: str = "training_id",
    dataset_col: str = DATASET_COL,
    query_col: str = "query",
    expected_col: str = "expected_answer",
    agent_col: str = ORACLE_AGENT_COL,
) -> pd.DataFrame:
    """
    Recompute ``oracle_agent`` using strict utility argmax:
    ``argmax_s(Perf - λ * Cost)`` where ``Perf = is_correct``.

    Project default: ``DEFAULT_UTILITY_LAMBDA`` (25.0) for QCE v1 ``oracle_agent`` / ``y``.
    """
    if id_col not in labels.columns:
        raise ValueError(f"labels missing {id_col!r}")
    if id_col not in oracle.columns:
        raise ValueError(f"oracle missing {id_col!r}")

    work = oracle.copy()
    work = work[work[agent_col].isin(agents)].copy()
    work["is_correct_num"] = pd.to_numeric(work["is_correct"], errors="coerce").fillna(0.0)
    work["is_failed_num"] = pd.to_numeric(work["is_failed"], errors="coerce").fillna(0.0)
    work[cost_col] = pd.to_numeric(work[cost_col], errors="coerce").fillna(0.0)
    work["latency_sec"] = pd.to_numeric(work.get("latency_sec"), errors="coerce").fillna(0.0)
    has_answer = work["predicted_answer"].fillna("").astype(str).str.strip().ne("")
    work["perf"] = (
        (work["is_correct_num"] > 0) & (work["is_failed_num"] <= 0) & has_answer
    ).astype(float)
    work["utility"] = work["perf"] - float(utility_lambda) * work[cost_col]
    work["_order"] = work[agent_col].map(lambda s: _tiebreak_index(str(s), DEFAULT_UTILITY_TIEBREAK))

    utility_pick = (
        work.sort_values(
            by=[id_col, "utility", cost_col, "latency_sec", "_order"],
            ascending=[True, False, True, True, True],
            kind="mergesort",
        )
        .groupby(id_col, sort=False, as_index=False)
        .first()
    )

    n_correct = (
        work[work["perf"] > 0]
        .groupby(id_col, sort=False)[agent_col]
        .nunique()
        .rename("n_correct_agents")
        .astype(int)
    )

    picked = utility_pick.set_index(id_col)
    out = labels.copy()
    out["oracle_agent"] = out[id_col].map(picked[agent_col]).astype("object")
    out["agent_tier"] = out["oracle_agent"].map({a: i for i, a in enumerate(agents)}).astype("Int64")
    out[cost_col] = out[id_col].map(picked[cost_col]).astype(float)
    out["latency_sec"] = out[id_col].map(picked["latency_sec"]).astype(float)
    out["cost_normalized"] = out[id_col].map(picked.get("cost_normalized")).astype(float)
    out["predicted_answer"] = out[id_col].map(picked["predicted_answer"]).astype("object")
    out["n_correct_agents"] = out[id_col].map(n_correct).fillna(0).astype(int)
    out["label_status"] = np.where(out["oracle_agent"].notna(), "labeled", "discarded")
    out["utility_lambda"] = float(utility_lambda)
    out[TARGET_COL] = out["oracle_agent"]

    # Fill core columns from oracle when absent in labels.
    fill_cols = [dataset_col, query_col, expected_col]
    for col in fill_cols:
        if col not in out.columns:
            out[col] = out[id_col].map(picked[col]).astype("object")

    return out


def regenerate_oracle_agent_labels_in_files(
    label_paths: Sequence[str | Path],
    oracle_path: str | Path,
    *,
    utility_lambda: float,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    cost_col: str = "cost_usd",
    write: bool = True,
) -> dict[Path, pd.DataFrame]:
    """Regenerate utility cost-aware oracle labels (``oracle_agent`` + metadata) for each CSV."""
    oracle = load_oracle(oracle_path, agents=agents)
    out: dict[Path, pd.DataFrame] = {}
    for path in label_paths:
        p = Path(path)
        labels = pd.read_csv(p)
        labeled = regenerate_oracle_agent_labels(
            labels,
            oracle,
            utility_lambda=utility_lambda,
            agents=agents,
            cost_col=cost_col,
        )
        out[p] = labeled
        if write:
            labeled.to_csv(p, index=False)
    return out


def _correct_oracle_rows(oracle: pd.DataFrame) -> pd.DataFrame:
    """Rows that count toward soft-label mass."""
    df = oracle.copy()
    ic = pd.to_numeric(df["is_correct"], errors="coerce").fillna(0).astype(int)
    iff = pd.to_numeric(df["is_failed"], errors="coerce").fillna(0).astype(int)
    has_answer = df["predicted_answer"].fillna("").astype(str).str.strip().ne("")
    return df[(ic == 1) & (iff == 0) & has_answer]


def soft_distribution_for_question(
    correct: pd.DataFrame,
    *,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    temperature: float = 100.0,
    cost_col: str = "cost_usd",
    agent_col: str = ORACLE_AGENT_COL,
) -> dict[str, float]:
    """
    Distribute probability mass over *correct* agents only.

    - 0 correct → all zeros
    - 1 correct → one-hot on that agent
    - 2+ correct → normalize ``exp(-temperature * cost)`` over correct agents
      (uses per-run ``cost_usd`` by default, not ``cost_normalized``)
    """
    probs = {a: 0.0 for a in agents}
    if correct.empty:
        return probs

    agents_present = [a for a in agents if a in set(correct[agent_col])]
    if not agents_present:
        return probs

    if len(agents_present) == 1:
        probs[agents_present[0]] = 1.0
        return probs

    costs = correct.set_index(agent_col)[cost_col].astype(float)
    costs = costs.reindex(agents_present).fillna(0.0)
    if costs.nunique() <= 1:
        w = np.ones(len(agents_present), dtype=float)
    else:
        w = np.exp(-float(temperature) * costs.to_numpy())
    w = w / w.sum()
    for agent, mass in zip(agents_present, w, strict=True):
        probs[agent] = float(mass)
    return probs


def _temperature_for_row(
    dataset: str | None,
    *,
    default: float,
    temperature_by_dataset: dict[str, float] | None,
) -> float:
    if not temperature_by_dataset or not dataset:
        return default
    return float(temperature_by_dataset.get(str(dataset), default))


def build_soft_label_frame(
    labels: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    temperature: float = 100.0,
    temperature_by_dataset: dict[str, float] | None = None,
    cost_col: str = "cost_usd",
    id_col: str = "training_id",
    dataset_col: str = DATASET_COL,
) -> pd.DataFrame:
    """Per-question soft columns aligned to ``labels[id_col]``."""
    if id_col not in labels.columns:
        raise ValueError(f"labels missing {id_col!r}")

    correct_all = _correct_oracle_rows(oracle)
    soft_cols = soft_probability_columns(agents)
    has_dataset = dataset_col in labels.columns

    rows: list[dict[str, float]] = []
    for idx, tid in enumerate(labels[id_col]):
        g = correct_all[correct_all["training_id"] == tid]
        temp = temperature
        if has_dataset:
            ds = labels.iloc[idx][dataset_col]
            temp = _temperature_for_row(
                None if pd.isna(ds) else str(ds),
                default=temperature,
                temperature_by_dataset=temperature_by_dataset,
            )
        dist = soft_distribution_for_question(
            g,
            agents=agents,
            temperature=temp,
            cost_col=cost_col,
        )
        rows.append({f"{SOFT_COL_PREFIX}{a}": dist[a] for a in agents})

    soft = pd.DataFrame(rows, columns=soft_cols)
    soft[SOFT_SUM_COL] = soft[soft_cols].sum(axis=1)
    dominant = pd.Series(pd.NA, index=soft.index, dtype="object")
    has_mass = soft[SOFT_SUM_COL] > 0
    if has_mass.any():
        dominant.loc[has_mass] = (
            soft.loc[has_mass, soft_cols]
            .idxmax(axis=1)
            .str.removeprefix(SOFT_COL_PREFIX)
        )
    soft[SOFT_DOMINANT_COL] = dominant
    return soft


def attach_soft_labels(
    labels: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    temperature: float = 100.0,
    temperature_by_dataset: dict[str, float] | None = None,
    cost_col: str = "cost_usd",
    target_col: str = TARGET_COL,
) -> pd.DataFrame:
    """
    Add ``p_*``, ``soft_label_sum``, ``soft_dominant_agent``, and optional ``y``.

    Existing soft columns are replaced. ``y`` mirrors ``oracle_agent`` when present.
    Use ``temperature_by_dataset`` for per-dataset λ (e.g. math vs hotpot).
    """
    drop_cols = [
        c
        for c in labels.columns
        if c.startswith(SOFT_COL_PREFIX)
        or c in (SOFT_SUM_COL, SOFT_DOMINANT_COL, target_col)
    ]
    out = labels.drop(columns=drop_cols, errors="ignore")
    soft = build_soft_label_frame(
        out,
        oracle,
        agents=agents,
        temperature=temperature,
        temperature_by_dataset=temperature_by_dataset,
        cost_col=cost_col,
    )
    out = pd.concat([out.reset_index(drop=True), soft], axis=1)
    if "oracle_agent" in out.columns:
        out[target_col] = out["oracle_agent"]
    return out


def apply_soft_labels_to_files(
    label_paths: Sequence[str | Path],
    oracle_path: str | Path,
    *,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    temperature: float = 100.0,
    temperature_by_dataset: dict[str, float] | None = None,
    cost_col: str = "cost_usd",
    write: bool = True,
) -> dict[Path, pd.DataFrame]:
    """Load oracle once, attach soft labels to each label CSV, optionally write back."""
    oracle = load_oracle(oracle_path, agents=agents)
    out: dict[Path, pd.DataFrame] = {}
    for path in label_paths:
        p = Path(path)
        labels = pd.read_csv(p)
        labeled = attach_soft_labels(
            labels,
            oracle,
            agents=agents,
            temperature=temperature,
            temperature_by_dataset=temperature_by_dataset,
            cost_col=cost_col,
        )
        out[p] = labeled
        if write:
            labeled.to_csv(p, index=False)
    return out


def _entropy(probs: np.ndarray) -> float:
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-(p * np.log(p)).sum())


def _sweep_temperature_one_group(
    work: pd.DataFrame,
    correct_all: pd.DataFrame,
    temperatures: Sequence[float],
    *,
    agents: tuple[str, ...],
    cost_col: str,
    id_col: str,
    group_name: str = "all",
) -> list[dict[str, float | int | str]]:
    """λ sweep metrics for one subset of label rows (e.g. one dataset)."""
    has_oracle_col = "oracle_agent" in work.columns
    tids = work[id_col].tolist()
    oracle_agent: dict[object, str] = {}
    if has_oracle_col:
        oracle_agent = work.set_index(id_col)["oracle_agent"].astype(str).to_dict()

    rows: list[dict[str, float | int | str]] = []
    for temp in temperatures:
        dom_match: list[bool] = []
        p_oracle: list[float] = []
        ent_multi: list[float] = []
        p_oracle_half: list[bool] = []
        n_multi = 0
        n_with_mass = 0

        for tid in tids:
            g = correct_all[correct_all["training_id"] == tid]
            dist = soft_distribution_for_question(
                g,
                agents=agents,
                temperature=temp,
                cost_col=cost_col,
            )
            mass = sum(dist.values())
            if mass <= 0:
                continue
            n_with_mass += 1
            probs = np.array([dist[a] for a in agents], dtype=float)

            oa = oracle_agent.get(tid) if has_oracle_col else None
            if oa and oa in dist:
                p_oracle.append(dist[oa])
                p_oracle_half.append(dist[oa] > 0.5)

            if sum(1 for v in dist.values() if v > 0) >= 2:
                n_multi += 1
                dom = max(dist, key=dist.get)
                if oa:
                    dom_match.append(dom == oa)
                ent_multi.append(_entropy(probs))

        rows.append(
            {
                DATASET_COL: group_name,
                "temperature": float(temp),
                "n_labeled": len(tids),
                "n_with_soft_mass": n_with_mass,
                "n_multi_correct": n_multi,
                "dominant_eq_oracle_pct": (
                    100.0 * float(np.mean(dom_match)) if dom_match else float("nan")
                ),
                "mean_p_oracle": (
                    float(np.mean(p_oracle)) if p_oracle else float("nan")
                ),
                "frac_p_oracle_gt_0_5": (
                    100.0 * float(np.mean(p_oracle_half)) if p_oracle_half else float("nan")
                ),
                "mean_entropy_multi": (
                    float(np.mean(ent_multi)) if ent_multi else float("nan")
                ),
            }
        )
    return rows


def sweep_temperature_metrics(
    labels: pd.DataFrame,
    oracle: pd.DataFrame,
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURE_SWEEP,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    cost_col: str = "cost_usd",
    labeled_only: bool = True,
    by_dataset: bool = False,
    id_col: str = "training_id",
    dataset_col: str = DATASET_COL,
) -> pd.DataFrame:
    """
    Summarize how λ (``temperature``) affects soft labels on a label table.

    With ``by_dataset=True``, returns one block per (dataset, λ) plus an
    ``all`` aggregate row group.
    """
    if id_col not in labels.columns:
        raise ValueError(f"labels missing {id_col!r}")

    work = labels
    if labeled_only and "label_status" in work.columns:
        work = work[work["label_status"].astype(str).str.strip().eq("labeled")]

    correct_all = _correct_oracle_rows(oracle)
    all_rows = _sweep_temperature_one_group(
        work,
        correct_all,
        temperatures,
        agents=agents,
        cost_col=cost_col,
        id_col=id_col,
        group_name="all",
    )

    if not by_dataset or dataset_col not in work.columns:
        out = pd.DataFrame(all_rows)
        if not by_dataset and DATASET_COL in out.columns:
            out = out.drop(columns=[DATASET_COL])
        return out

    rows = list(all_rows)
    for ds, g in work.groupby(dataset_col, sort=True):
        rows.extend(
            _sweep_temperature_one_group(
                g,
                correct_all,
                temperatures,
                agents=agents,
                cost_col=cost_col,
                id_col=id_col,
                group_name=str(ds),
            )
        )
    return pd.DataFrame(rows)


def recommend_temperature_per_dataset(
    summary: pd.DataFrame,
    *,
    dataset_col: str = DATASET_COL,
    default: float = 100.0,
) -> dict[str, float]:
    """
    Pick λ per dataset from a sweep table (use val split).

    Among λ with best ``dominant_eq_oracle_pct``, choose the one with highest
    ``mean_entropy_multi`` (softer labels while argmax still matches oracle).
    """
    if dataset_col not in summary.columns:
        return {"all": default}

    out: dict[str, float] = {}
    for ds, g in summary.groupby(dataset_col, sort=True):
        if str(ds) == "all":
            continue
        best_dom = g["dominant_eq_oracle_pct"].max()
        candidates = g[g["dominant_eq_oracle_pct"] >= best_dom - 0.5]
        if candidates.empty:
            out[str(ds)] = default
            continue
        pick = candidates.loc[candidates["mean_entropy_multi"].idxmax()]
        out[str(ds)] = float(pick["temperature"])
    return out


def _format_sweep_table(summary: pd.DataFrame) -> pd.DataFrame:
    fmt = summary.copy()
    for col in (
        "dominant_eq_oracle_pct",
        "mean_p_oracle",
        "frac_p_oracle_gt_0_5",
        "mean_entropy_multi",
    ):
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.2f}" if x == x else "nan")
    return fmt


def print_temperature_sweep(
    summary: pd.DataFrame,
    *,
    title: str | None = None,
    by_dataset: bool = False,
    dataset_col: str = DATASET_COL,
) -> None:
    """Pretty-print a sweep table from :func:`sweep_temperature_metrics`."""
    if title:
        print(title)
    if summary.empty:
        print("  (no rows)")
        return

    if by_dataset and dataset_col in summary.columns:
        for ds, g in summary.groupby(dataset_col, sort=True):
            print(f"  [{ds}]")
            print(_format_sweep_table(g).to_string(index=False))
            print()
        return

    print(_format_sweep_table(summary).to_string(index=False))


def run_temperature_sweep_on_files(
    label_paths: Sequence[str | Path],
    oracle_path: str | Path,
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURE_SWEEP,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    cost_col: str = "cost_usd",
    labeled_only: bool = True,
    by_dataset: bool = True,
) -> dict[Path, pd.DataFrame]:
    """Load oracle once and sweep λ for each label CSV path."""
    oracle = load_oracle(oracle_path, agents=agents)
    out: dict[Path, pd.DataFrame] = {}
    for path in label_paths:
        p = Path(path)
        labels = pd.read_csv(p)
        out[p] = sweep_temperature_metrics(
            labels,
            oracle,
            temperatures=temperatures,
            agents=agents,
            cost_col=cost_col,
            labeled_only=labeled_only,
            by_dataset=by_dataset,
        )
    return out


def _parse_temperature_map(raw: str | None) -> dict[str, float]:
    """Parse ``math:50,hotpot:100`` into a per-dataset λ map."""
    if not raw:
        return {}
    out: dict[str, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise SystemExit(
                f"--temperature-map: expected dataset:lambda, got {part!r}"
            )
        ds, val = part.split(":", 1)
        out[ds.strip()] = float(val.strip())
    return out


def infer_temperature_map_from_val(
    val_path: str | Path,
    oracle_path: str | Path,
    *,
    temperatures: Sequence[float] = DEFAULT_TEMPERATURE_SWEEP,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    cost_col: str = "cost_usd",
) -> dict[str, float]:
    """Run per-dataset λ sweep on val and return recommended map."""
    labels = pd.read_csv(val_path)
    oracle = load_oracle(oracle_path, agents=agents)
    summary = sweep_temperature_metrics(
        labels,
        oracle,
        temperatures=temperatures,
        agents=agents,
        cost_col=cost_col,
        by_dataset=True,
    )
    return recommend_temperature_per_dataset(summary)


def v1_split_paths(v1_dir: Path | None = None) -> dict[str, Path]:
    """Map split key → CSV path under v1/."""
    root = v1_dir or default_v1_dir()
    return {
        "train": root / "qce_train.csv",
        "val": root / "qce_val.csv",
        "test": root / "qce_internal_test.csv",
    }


def solvable_ids_missing_from_qce(
    *,
    v1_dir: Path | None = None,
    oracle_path: str | Path | None = None,
    pool_name: str = POOL_LABELS_NAME,
) -> list[str]:
    """``training_id`` values with ≥1 correct agent in oracle but not in any QCE split CSV."""
    root = v1_dir or default_v1_dir()
    pool = pd.read_csv(root / pool_name)
    paths = v1_split_paths(root)
    qce_ids: set[str] = set()
    for path in paths.values():
        qce_ids |= set(pd.read_csv(path)["training_id"].astype(str))
    oracle = load_oracle(oracle_path or (root / "oracle_results1.csv"))
    solvable = set(_correct_oracle_rows(oracle).groupby("training_id", sort=False).groups)
    return sorted(solvable - qce_ids)


def allocate_ids_to_splits(
    training_ids: list[str],
    *,
    split_sizes: dict[str, int],
) -> dict[str, list[str]]:
    """Assign ids across splits proportional to ``split_sizes`` (largest remainder)."""
    ids = sorted(training_ids)
    splits = [s for s in SPLIT_KEYS if s in split_sizes and split_sizes[s] >= 0]
    if not splits or not ids:
        return {s: [] for s in splits}
    total = sum(split_sizes[s] for s in splits) or 1
    n = len(ids)
    target = {s: n * split_sizes[s] / total for s in splits}
    alloc = {s: int(target[s]) for s in splits}
    remainder = n - sum(alloc.values())
    for s in sorted(splits, key=lambda s: target[s] - alloc[s], reverse=True)[:remainder]:
        alloc[s] += 1
    out: dict[str, list[str]] = {s: [] for s in splits}
    idx = 0
    for s in splits:
        out[s] = ids[idx : idx + alloc[s]]
        idx += alloc[s]
    return out


def append_missing_solvable_to_v1_splits(
    *,
    v1_dir: Path | None = None,
    oracle_path: str | Path | None = None,
    pool_name: str = POOL_LABELS_NAME,
    training_ids: list[str] | None = None,
    write: bool = True,
) -> dict[str, list[str]]:
    """
    Append solvable pool rows missing from QCE splits (proportional train/val/test).

    Rows are copied from ``train_labels1.csv`` (or ``pool_name``). Call
    ``apply_soft_labels_to_v1_splits`` afterward (or use ``--append-missing-solvable``).
    """
    root = v1_dir or default_v1_dir()
    oracle_path = Path(oracle_path or (root / "oracle_results1.csv"))
    missing = training_ids or solvable_ids_missing_from_qce(
        v1_dir=root, oracle_path=oracle_path, pool_name=pool_name
    )
    if not missing:
        return {k: [] for k in SPLIT_KEYS}

    pool = pd.read_csv(root / pool_name)
    pool = pool[pool["training_id"].astype(str).isin(missing)].copy()
    if len(pool) != len(missing):
        found = set(pool["training_id"])
        extra = [t for t in missing if t not in found]
        raise ValueError(f"pool missing training_ids: {extra[:5]}")

    paths = v1_split_paths(root)
    split_sizes: dict[str, int] = {}
    for key, path in paths.items():
        if path.is_file():
            split_sizes[key] = len(pd.read_csv(path))
        else:
            split_sizes[key] = 0

    by_split = allocate_ids_to_splits(missing, split_sizes=split_sizes)
    added: dict[str, list[str]] = {k: [] for k in SPLIT_KEYS}

    for key, ids in by_split.items():
        if not ids:
            continue
        path = paths[key]
        new_rows = pool[pool["training_id"].isin(ids)].copy()
        if path.is_file():
            existing = pd.read_csv(path)
            overlap = set(existing["training_id"]) & set(ids)
            if overlap:
                raise ValueError(f"{path.name}: already contains {sorted(overlap)[:3]}")
            combined = pd.concat([existing, new_rows], ignore_index=True)
        else:
            combined = new_rows
        if write:
            combined.to_csv(path, index=False)
        added[key] = ids

    return added


def append_missing_solvable_pipeline(
    *,
    v1_dir: Path | None = None,
    oracle_path: str | Path | None = None,
    with_qce_features: bool = True,
    temperature: float = 100.0,
    decompose_model: str = "gpt-4o",
    dry_run: bool = False,
) -> dict[str, list[str]]:
    """Append missing solvable ids, refresh soft labels, optionally build QCE features."""
    root = v1_dir or default_v1_dir()
    oracle_path = Path(oracle_path or (root / "oracle_results1.csv"))
    added = append_missing_solvable_to_v1_splits(
        v1_dir=root,
        oracle_path=oracle_path,
        write=not dry_run,
    )
    if not any(added.values()):
        print("No solvable training_ids missing from QCE splits.")
        return added

    for key, ids in added.items():
        print(f"  {key}: +{len(ids)} → {ids}")

    if dry_run:
        print("(dry-run: splits/features not written beyond append preview)")
        return added

    apply_soft_labels_to_v1_splits(
        v1_dir=root,
        oracle_path=oracle_path.name,
        temperature=temperature,
        write=True,
    )
    print("Refreshed soft labels on QCE splits.")

    if with_qce_features:
        build_qce_features_for_added_ids(
            added,
            v1_dir=root,
            decompose_model=decompose_model,
        )

    return added


def training_ids_missing_qce_features(
    *,
    v1_dir: Path | None = None,
    qce_features_dir: Path | None = None,
) -> dict[str, list[str]]:
    """Ids present in split CSV but absent from complexity_record_{split}.parquet."""
    root = v1_dir or default_v1_dir()
    repo = repo_root()
    qce_features_dir = qce_features_dir or (repo / "datasets/qce_features")
    paths = v1_split_paths(root)
    missing: dict[str, list[str]] = {k: [] for k in SPLIT_KEYS}
    for split_key, csv_path in paths.items():
        if not csv_path.is_file():
            continue
        ids_csv = set(pd.read_csv(csv_path)["training_id"].astype(str))
        c_path = qce_features_dir / f"complexity_record_{split_key}.parquet"
        if c_path.is_file():
            ids_c = set(pd.read_parquet(c_path)["training_id"].astype(str))
        else:
            ids_c = set()
        missing[split_key] = sorted(ids_csv - ids_c)
    return missing


def build_qce_features_for_added_ids(
    added_by_split: dict[str, list[str]],
    *,
    v1_dir: Path | None = None,
    decompose_model: str = "gpt-4o",
    qce_features_dir: Path | None = None,
    decomposer_cache_dir: Path | None = None,
    train_norm_json: Path | None = None,
) -> None:
    """Decompose + append complexity / embedding parquet rows for newly added split ids."""
    from qce.complexity import complexity_dataframe, load_train_norm, write_complexity_parquet
    from qce.decompose import decompose_batch
    from routing.router import plans_from_cache

    root = v1_dir or default_v1_dir()
    repo = repo_root()
    qce_features_dir = qce_features_dir or (repo / "datasets/qce_features")
    decomposer_cache_dir = decomposer_cache_dir or (repo / "datasets/decomposer_cache")
    train_norm_json = train_norm_json or (repo / "models/qce_graph/train_norm.json")
    norm = load_train_norm(train_norm_json)
    paths = v1_split_paths(root)

    for split_key, ids in added_by_split.items():
        if not ids:
            continue
        csv_path = paths[split_key]
        cache_path = decomposer_cache_dir / f"qce_{split_key}_plans.jsonl"
        c_path = qce_features_dir / f"complexity_record_{split_key}.parquet"
        e_path = qce_features_dir / f"query_embeddings_{split_key}.parquet"

        corpus = pd.read_csv(csv_path)
        sub = corpus[corpus["training_id"].astype(str).isin(ids)].copy()
        rows = sub.to_dict(orient="records")
        print(f"\n[{split_key}] decompose {len(rows)} queries → {cache_path.name}")
        decompose_batch(
            rows,
            cache_path=cache_path,
            model=decompose_model,
            force_refresh=False,
        )
        plans = plans_from_cache(rows, cache_path)
        new_c = complexity_dataframe(plans, norm=norm, fit_norm=False)
        if c_path.is_file():
            old_c = pd.read_parquet(c_path)
            old_c = old_c[~old_c["training_id"].astype(str).isin(ids)]
            c_df = pd.concat([old_c, new_c], ignore_index=True)
        else:
            c_df = new_c
        write_complexity_parquet(c_df, c_path)
        print(f"  complexity → {c_path} ({len(c_df)} rows)")

        from scripts.build_query_embeddings import append_embeddings_for_training_ids

        append_embeddings_for_training_ids(
            split_key,
            ids,
            labels_dir=root,
            out_dir=qce_features_dir,
        )
        print(f"  embeddings → {e_path}")


def apply_soft_labels_to_v1_splits(
    *,
    v1_dir: Path | None = None,
    oracle_name: str = "oracle_results1.csv",
    splits: tuple[str, ...] = V1_SPLITS,
    agents: tuple[str, ...] = DEFAULT_AGENTS,
    temperature: float = 100.0,
    temperature_by_dataset: dict[str, float] | None = None,
    cost_col: str = "cost_usd",
    write: bool = True,
) -> dict[Path, pd.DataFrame]:
    """Update default v1 QCE train/val/internal-test CSVs."""
    root = v1_dir or default_v1_dir()
    oracle_path = root / oracle_name
    label_paths = [root / name for name in splits]
    return apply_soft_labels_to_files(
        label_paths,
        oracle_path,
        agents=agents,
        temperature=temperature,
        temperature_by_dataset=temperature_by_dataset,
        cost_col=cost_col,
        write=write,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Attach cost-aware soft routing labels from oracle results.",
    )
    p.add_argument(
        "--oracle",
        type=Path,
        help="Long-form oracle CSV (default: v1/oracle_results1.csv).",
    )
    p.add_argument(
        "--labels",
        type=Path,
        action="append",
        help="Label CSV to update (repeatable). Ignored if --update-qce-splits.",
    )
    p.add_argument(
        "--update-qce-splits",
        action="store_true",
        help="Update qce_train.csv, qce_val.csv, qce_internal_test.csv under v1/.",
    )
    p.add_argument(
        "--v1-dir",
        type=Path,
        default=None,
        help="Directory containing QCE splits (default: datasets/train_samples/v1).",
    )
    p.add_argument(
        "--agents",
        nargs="+",
        default=list(DEFAULT_AGENTS),
        help=f"Agents to label (default: {' '.join(DEFAULT_AGENTS)}).",
    )
    p.add_argument(
        "--temperature",
        type=float,
        default=100.0,
        help="Soft-label temperature for exp(-temperature * cost_usd) among correct agents.",
    )
    p.add_argument(
        "--lambda",
        dest="utility_lambda",
        type=float,
        default=DEFAULT_UTILITY_LAMBDA,
        help=(
            "Utility cost penalty λ for oracle-agent regeneration: "
            f"U = quality - λ * cost (default: {DEFAULT_UTILITY_LAMBDA})."
        ),
    )
    p.add_argument(
        "--cost-col",
        default="cost_usd",
        choices=("cost_usd", "cost_normalized"),
        help="Cost column from oracle rows (default: cost_usd).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute labels but do not write CSV files.",
    )
    p.add_argument(
        "--sweep-temperature",
        action="store_true",
        help="Print λ sweep metrics (no CSV writes). Uses v1 splits if --labels omitted.",
    )
    p.add_argument(
        "--sweep-values",
        default=None,
        help=(
            "Comma-separated λ values for --sweep-temperature "
            f"(default: {','.join(str(int(t)) if t == int(t) else str(t) for t in DEFAULT_TEMPERATURE_SWEEP)})."
        ),
    )
    p.add_argument(
        "--by-dataset",
        action="store_true",
        default=True,
        help="With --sweep-temperature, report metrics per dataset (default: on).",
    )
    p.add_argument(
        "--no-by-dataset",
        action="store_true",
        help="Only aggregate metrics in --sweep-temperature.",
    )
    p.add_argument(
        "--temperature-map",
        default=None,
        help="Per-dataset λ for apply, e.g. math:50,hotpot:100,musique:100.",
    )
    p.add_argument(
        "--auto-temperature",
        action="store_true",
        help=(
            "Infer per-dataset λ from qce_val.csv (sweep on val), then apply to "
            "all splits being updated."
        ),
    )
    p.add_argument(
        "--regenerate-oracle-agent",
        action="store_true",
        help=(
            "Rebuild utility cost-aware oracle_agent and label metadata from oracle CSV using "
            "strict utility argmax: Perf - λ*Cost (λ from --lambda)."
        ),
    )
    p.add_argument(
        "--append-missing-solvable",
        action="store_true",
        help=(
            "Append pool rows with ≥1 correct agent that are missing from QCE splits "
            f"(from {POOL_LABELS_NAME}), proportional train/val/test; refresh p_*."
        ),
    )
    p.add_argument(
        "--with-qce-features",
        action="store_true",
        help=(
            "With --append-missing-solvable: decompose, complexity_record_*, "
            "query_embeddings_* for added ids only."
        ),
    )
    p.add_argument(
        "--decompose-model",
        default="gpt-4o",
        help="LLM for plan decompose when --with-qce-features (default: gpt-4o).",
    )
    p.add_argument(
        "--build-missing-qce-features",
        action="store_true",
        help="Decompose + complexity + embeddings for ids in QCE CSVs but not in feature parquets.",
    )
    return p


def _parse_sweep_values(raw: str | None) -> tuple[float, ...]:
    if not raw:
        return DEFAULT_TEMPERATURE_SWEEP
    out: list[float] = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise SystemExit("--sweep-values: no numeric values parsed.")
    return tuple(out)


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    agents = tuple(args.agents)
    v1_dir = args.v1_dir or default_v1_dir()
    oracle_path = args.oracle or (v1_dir / "oracle_results1.csv")
    by_dataset = args.by_dataset and not args.no_by_dataset
    temp_map = _parse_temperature_map(args.temperature_map)

    if args.sweep_temperature:
        temps = _parse_sweep_values(args.sweep_values)
        if args.labels:
            label_paths = [Path(p) for p in args.labels]
        else:
            label_paths = [v1_dir / name for name in V1_SPLITS]
        print(f"Oracle: {oracle_path}")
        print(f"λ sweep: {list(temps)}")
        print(f"Cost column: {args.cost_col}")
        print(f"By dataset: {by_dataset}\n")
        results = run_temperature_sweep_on_files(
            label_paths,
            oracle_path,
            temperatures=temps,
            agents=agents,
            cost_col=args.cost_col,
            by_dataset=by_dataset,
        )
        root = repo_root().resolve()
        for path, table in results.items():
            p = Path(path).resolve()
            try:
                rel = p.relative_to(root)
            except ValueError:
                rel = p
            print_temperature_sweep(
                table,
                title=f"=== {rel} ===",
                by_dataset=by_dataset,
            )
            print()
        return 0

    if args.build_missing_qce_features:
        gaps = training_ids_missing_qce_features(v1_dir=v1_dir)
        n_gap = sum(len(v) for v in gaps.values())
        if not n_gap:
            print("All QCE split rows have complexity_record_* rows.")
            return 0
        for key, ids in gaps.items():
            if ids:
                print(f"  {key}: {len(ids)} missing features")
        build_qce_features_for_added_ids(
            gaps,
            v1_dir=v1_dir,
            decompose_model=args.decompose_model,
        )
        return 0

    if args.append_missing_solvable:
        added = append_missing_solvable_pipeline(
            v1_dir=v1_dir,
            oracle_path=oracle_path,
            with_qce_features=args.with_qce_features,
            temperature=args.temperature,
            decompose_model=args.decompose_model,
            dry_run=args.dry_run,
        )
        if not args.dry_run:
            print(f"Done. Added {sum(len(v) for v in added.values())} rows across splits.")
        return 0

    if args.regenerate_oracle_agent:
        utility_lambda = float(args.utility_lambda)
        if args.update_qce_splits:
            label_paths = [v1_dir / name for name in V1_SPLITS]
        elif args.labels:
            label_paths = [Path(p) for p in args.labels]
        else:
            raise SystemExit(
                "Use --regenerate-oracle-agent with --update-qce-splits or --labels PATH."
            )
        results = regenerate_oracle_agent_labels_in_files(
            label_paths,
            oracle_path,
            utility_lambda=utility_lambda,
            agents=agents,
            cost_col=args.cost_col,
            write=not args.dry_run,
        )
        for path, df in results.items():
            n = len(df)
            labeled = int(df["label_status"].astype(str).eq("labeled").sum())
            print(f"{path}: {n} rows, {labeled} labeled (λ={utility_lambda})")
        if args.dry_run:
            print("(dry-run: no files written)")
        return 0

    if args.auto_temperature:
        val_path = v1_dir / "qce_val.csv"
        temps = _parse_sweep_values(args.sweep_values)
        temp_map = infer_temperature_map_from_val(
            val_path,
            oracle_path,
            temperatures=temps,
            agents=agents,
            cost_col=args.cost_col,
        )
        print("Auto λ from val (per dataset):")
        for ds, lam in sorted(temp_map.items()):
            print(f"  {ds}: {lam}")
        print()

    if args.update_qce_splits:
        results = apply_soft_labels_to_v1_splits(
            v1_dir=args.v1_dir,
            agents=agents,
            temperature=args.temperature,
            temperature_by_dataset=temp_map or None,
            cost_col=args.cost_col,
            write=not args.dry_run,
        )
    else:
        if not args.labels:
            raise SystemExit(
                "Provide --labels PATH (repeatable), --update-qce-splits, or --sweep-temperature."
            )
        if args.auto_temperature and not temp_map:
            val_path = next(
                (Path(p) for p in args.labels if "val" in Path(p).name.lower()),
                v1_dir / "qce_val.csv",
            )
            temp_map = infer_temperature_map_from_val(
                val_path,
                oracle_path,
                temperatures=_parse_sweep_values(args.sweep_values),
                agents=agents,
                cost_col=args.cost_col,
            )
        results = apply_soft_labels_to_files(
            args.labels,
            oracle_path,
            agents=agents,
            temperature=args.temperature,
            temperature_by_dataset=temp_map or None,
            cost_col=args.cost_col,
            write=not args.dry_run,
        )

    for path, df in results.items():
        n = len(df)
        with_mass = int((df[SOFT_SUM_COL] > 0).sum()) if SOFT_SUM_COL in df.columns else 0
        print(f"{path}: {n} rows, {with_mass} with soft mass")
        if temp_map and DATASET_COL in df.columns:
            for ds in sorted(df[DATASET_COL].dropna().unique()):
                lam = temp_map.get(str(ds), args.temperature)
                sub = df[df[DATASET_COL] == ds]
                wm = int((sub[SOFT_SUM_COL] > 0).sum()) if SOFT_SUM_COL in sub.columns else 0
                print(f"  {ds}: {len(sub)} rows (λ={lam}), {wm} with soft mass")
    if args.dry_run:
        print("(dry-run: no files written)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
