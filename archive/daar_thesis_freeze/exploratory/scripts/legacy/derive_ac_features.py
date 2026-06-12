"""
Derive normalized complexity features and labels from offline_dataset_raw.

Recomputable any time — does NOT require re-running agents.
Depends on train-split means (mu) and optional alpha weights.

Usage::

    uv run python scripts/derive_ac_features.py
    uv run python scripts/derive_ac_features.py --alpha-d 0.5 --alpha-m 0.5
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from config.global_config.paths import offline_dataset_dir

AGENTS: tuple[str, ...] = ("raw", "cot", "react", "multiagent")
TOOL_COL_PREFIX = "n_tool_"
TRAIN_SPLIT = "train"
DEFAULT_ALPHA_D = 0.5
DEFAULT_ALPHA_M = 0.5
R_ALPHA_ABLATIONS: tuple[tuple[float, float, str], ...] = (
    (1.0, 0.0, "R_alpha_1_0"),
    (0.5, 0.5, "R_alpha_0_5"),
    (0.0, 1.0, "R_alpha_0_1"),
)


def _reasoning_depth(agent: str, row: pd.Series) -> int:
    if agent == "raw":
        return 1
    if agent == "cot":
        return max(1, int(row.get("num_reasoning_steps") or row.get("n_reasoning_steps") or 1))
    if agent == "react":
        steps = row.get("steps_taken")
        if steps is not None and not (isinstance(steps, float) and math.isnan(steps)):
            st = int(steps)
            if st > 0:
                return st
        llm = int(row.get("num_llm_calls") or 1)
        return max(1, llm - 1)
    if agent == "multiagent":
        n_workers = int(row.get("n_sub_agents") or 0)
        return 1 + n_workers + 1
    return 1


def _safe_mean(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return 1.0
    m = float(s.mean())
    return m if m > 0 else 1.0


def _safe_div(numerator: float, denominator: float) -> float:
    if denominator <= 0 or math.isnan(denominator):
        denominator = 1.0
    return float(numerator) / float(denominator)


def _r_alpha(r_depth: float, r_llm: float, alpha_d: float, alpha_m: float) -> float:
    return alpha_d * r_depth + alpha_m * r_llm


def _tool_cols(df: pd.DataFrame) -> list[str]:
    return sorted(c for c in df.columns if c.startswith(TOOL_COL_PREFIX))


def add_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Solvability / success labels derived from raw success fields."""
    out = df.copy()
    out["s"] = out["is_correct"].astype(bool)
    q_any = out.groupby("training_id")["s"].any()
    out["y_solv"] = out["training_id"].map(q_any).astype(bool)
    out["query_solvable"] = out["y_solv"]
    out["query_unsolvable"] = ~out["y_solv"]
    out["n_correct_agents"] = out.groupby("training_id")["s"].transform("sum").astype(int)
    return out


def compute_normalization_stats(
    train_df: pd.DataFrame,
    tool_cols: list[str],
) -> dict[str, Any]:
    stats: dict[str, Any] = {
        "split_used": TRAIN_SPLIT,
        "n_rows": int(len(train_df)),
        "per_agent": {},
    }
    for agent in AGENTS:
        sub = train_df[train_df["agent"] == agent]
        agent_stats: dict[str, Any] = {
            "n_rows": int(len(sub)),
            "mu_d": _safe_mean(sub["d_depth"]),
            "mu_m": _safe_mean(sub["num_llm_calls"]),
            "mu_prompt": _safe_mean(sub["prompt_tokens"]),
            "mu_completion": _safe_mean(sub["completion_tokens"]),
            "mu_runtime": _safe_mean(sub["latency_total"]),
            "mu_tools": {},
        }
        for col in tool_cols:
            tool_key = col.removeprefix(TOOL_COL_PREFIX)
            agent_stats["mu_tools"][tool_key] = _safe_mean(sub[col])
        stats["per_agent"][agent] = agent_stats
    return stats


def apply_complexity_scores(
    df: pd.DataFrame,
    norm_stats: dict[str, Any],
    tool_cols: list[str],
    *,
    alpha_d: float = DEFAULT_ALPHA_D,
    alpha_m: float = DEFAULT_ALPHA_M,
) -> pd.DataFrame:
    out = df.copy()
    r_depth_vals: list[float] = []
    r_llm_vals: list[float] = []
    t_vals: list[float] = []
    tok_vals: list[float] = []
    l_vals: list[float] = []

    for _, row in out.iterrows():
        agent = str(row["agent"])
        ast = norm_stats["per_agent"][agent]
        d = float(row["d_depth"])
        m = float(row["num_llm_calls"])
        r_d = _safe_div(d, ast["mu_d"])
        r_m = _safe_div(m, ast["mu_m"])
        r_depth_vals.append(r_d)
        r_llm_vals.append(r_m)

        t_score = 0.0
        for col in tool_cols:
            tool_key = col.removeprefix(TOOL_COL_PREFIX)
            count = float(row[col])
            mu_j = float(ast["mu_tools"].get(tool_key, 1.0))
            t_score += _safe_div(count, mu_j)
        t_vals.append(t_score)

        tok_vals.append(
            _safe_div(float(row["prompt_tokens"] or 0), ast["mu_prompt"])
            + _safe_div(float(row["completion_tokens"] or 0), ast["mu_completion"])
        )
        l_vals.append(_safe_div(float(row["latency_total"] or 0), ast["mu_runtime"]))

    out["R_depth"] = r_depth_vals
    out["R_llm"] = r_llm_vals
    out["R_sum"] = [a + b for a, b in zip(r_depth_vals, r_llm_vals)]
    out["R_alpha"] = [_r_alpha(rd, rm, alpha_d, alpha_m) for rd, rm in zip(r_depth_vals, r_llm_vals)]
    for ad, am, col in R_ALPHA_ABLATIONS:
        out[col] = [_r_alpha(rd, rm, ad, am) for rd, rm in zip(r_depth_vals, r_llm_vals)]
    out["R"] = out["R_alpha"]
    out["T"] = t_vals
    out["Tok"] = tok_vals
    out["L"] = l_vals
    out["c_syn_baseline"] = out["R_sum"] + out["T"] + out["Tok"] + out["L"]
    out["c_syn"] = out["R_alpha"] + out["T"] + out["Tok"] + out["L"]
    if "cost_usd" in out.columns:
        out["c_obs_usd"] = pd.to_numeric(out["cost_usd"], errors="coerce").fillna(0.0)
    return out


def build_derived_frame(
    raw_df: pd.DataFrame,
    *,
    alpha_d: float = DEFAULT_ALPHA_D,
    alpha_m: float = DEFAULT_ALPHA_M,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = raw_df.copy()
    df["d_depth"] = df.apply(lambda row: _reasoning_depth(str(row["agent"]), row), axis=1).astype(int)

    tool_cols = _tool_cols(df)
    train_mask = df["split"] == TRAIN_SPLIT
    norm_stats = compute_normalization_stats(df.loc[train_mask], tool_cols)
    df = add_labels(df)
    df = apply_complexity_scores(df, norm_stats, tool_cols, alpha_d=alpha_d, alpha_m=alpha_m)

    norm_stats["tool_columns"] = tool_cols
    norm_stats["alpha_default"] = {"alpha_d": alpha_d, "alpha_m": alpha_m}
    norm_stats["layer"] = "derived"
    norm_stats["formulas"] = {
        "AC": "(R, T, Tok, L)",
        "R_alpha": "alpha_d * d_depth/mu_d + alpha_m * num_llm_calls/mu_m",
        "routing": "U = P_succ - w·AC (weights tuned on val)",
    }

    key_cols = [
        "training_id", "agent", "split",
        "d_depth", "s", "y_solv", "query_solvable", "query_unsolvable", "n_correct_agents",
        "R_depth", "R_llm", "R_sum", "R_alpha", "R_alpha_1_0", "R_alpha_0_5", "R_alpha_0_1",
        "R", "T", "Tok", "L", "c_syn_baseline", "c_syn", "c_obs_usd",
    ]
    extra = [c for c in df.columns if c not in key_cols]
    cols = [c for c in key_cols if c in df.columns] + sorted(extra)
    return df[cols], norm_stats


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Derive AC features from offline_dataset_raw.")
    parser.add_argument(
        "--raw",
        type=Path,
        default=offline_dataset_dir() / "offline_dataset_raw.parquet",
    )
    parser.add_argument("--out-dir", type=Path, default=offline_dataset_dir())
    parser.add_argument("--alpha-d", type=float, default=DEFAULT_ALPHA_D)
    parser.add_argument("--alpha-m", type=float, default=DEFAULT_ALPHA_M)
    args = parser.parse_args()

    raw = pd.read_parquet(args.raw)
    derived, norm_stats = build_derived_frame(raw, alpha_d=args.alpha_d, alpha_m=args.alpha_m)

    out_parquet = args.out_dir / "offline_dataset_derived.parquet"
    norm_path = args.out_dir / "offline_dataset_normalization.json"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    derived.to_parquet(out_parquet, index=False)
    norm_path.write_text(json.dumps(norm_stats, indent=2) + "\n", encoding="utf-8")

    print(f"Derived rows={len(derived)}  cols={len(derived.columns)}")
    print(f"  {out_parquet}")
    print(f"  {norm_path}")


if __name__ == "__main__":
    main()
