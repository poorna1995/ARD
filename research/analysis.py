"""Paper analyses: complexity regions, calibration, top-2 routing, cost–accuracy."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from router.config import (
    AGENTS,
    PROBA_COLS,
    ROUTER_MODEL_PATH,
    TARGET,
)
from router.config import DATASET_DISPLAY_NAMES, DISCUSSION_DATASET_ORDER
from eval.score import (
    StrategyMetrics,
    attach_outcomes,
    baseline_strategies,
    metrics_for_agent_column,
    metrics_for_routed,
)
from eval.score import DEFAULT_ORACLE_PATH, from_oracle_csv
from router.router import (
    attach_router_predictions,
    load_router,
    load_split,
)
from src.utils.soft_labels import DEFAULT_UTILITY_LAMBDA

DEFAULT_ORACLE = DEFAULT_ORACLE_PATH
oracle_outcome_matrix = from_oracle_csv
DEFAULT_ROUTER = ROUTER_MODEL_PATH

AGENT_TIER: dict[str, int] = {"raw": 0, "cot": 1, "react": 2, "multiagent": 3}
AGENT_COLORS: dict[str, str] = {
    "raw": "#4C78A8",
    "cot": "#F58518",
    "react": "#54A24B",
    "multiagent": "#E45756",
}

CONFIDENCE_BIN_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

Top2Mode = Literal["cheaper_tier", "cheaper_cost", "react_fallback", "argmax"]


def attach_router(
    df: pd.DataFrame,
    *,
    router_path: Path,
) -> pd.DataFrame:
    """Add ``router_pred``, ``max_prob``, ``margin_top2``, proba columns."""
    obj = load_router(router_path)
    pipe = obj["pipeline"]
    feature_cols: list[str] = list(obj["feature_cols"])
    out = attach_router_predictions(
        df,
        pipe,
        feature_cols,
        experiment_id=str(obj.get("experiment_id", router_path.stem)),
    )
    return out


def build_analysis_frame(
    split: str,
    *,
    router_path: Path | None = DEFAULT_ROUTER,
    pipe: Any | None = None,
    feature_cols: list[str] | None = None,
    experiment_id: str = "router",
    oracle_path: Path = DEFAULT_ORACLE,
    with_embeddings: bool = True,
) -> pd.DataFrame:
    df = load_split(split, with_embeddings=with_embeddings)
    if "complexity_graph" not in df.columns:
        raise ValueError(
            f"{split}: missing complexity_graph — rebuild complexity_record_{split}.parquet"
        )
    outcomes = oracle_outcome_matrix(oracle_path, df["training_id"])
    if pipe is not None and feature_cols is not None:
        df = attach_router_predictions(df, pipe, feature_cols, experiment_id=experiment_id)
    elif router_path is not None:
        df = attach_router(df, router_path=router_path)
    else:
        raise ValueError("provide router_path or (pipe, feature_cols)")
    return attach_outcomes(df, outcomes)


def _agent_cost_row(row: pd.Series, agent: str) -> float:
    v = row.get(f"cost_{agent}")
    return float(v) if pd.notna(v) else np.inf


def route_top2(
    row: pd.Series,
    *,
    tau: float,
    mode: Top2Mode,
) -> str:
    """Top-1 if margin ≥ τ; else cheaper-of-top-2, react fallback, or argmax."""
    if mode == "argmax":
        return str(row["router_pred"])
    p1 = float(row["max_prob"])
    p2 = float(row["second_prob"])
    if p1 - p2 >= tau:
        return str(row["router_pred"])
    a1 = str(row["router_pred"])
    a2 = str(row["router_second"])
    if mode == "react_fallback":
        return "react"
    if mode == "cheaper_tier":
        return a1 if AGENT_TIER[a1] <= AGENT_TIER[a2] else a2
    if mode == "cheaper_cost":
        c1 = _agent_cost_row(row, a1)
        c2 = _agent_cost_row(row, a2)
        return a1 if c1 <= c2 else a2
    raise ValueError(f"unknown top2 mode {mode!r}")


def top2_sweep(
    df: pd.DataFrame,
    *,
    taus: np.ndarray | None = None,
    modes: tuple[Top2Mode, ...] = ("cheaper_cost", "cheaper_tier", "react_fallback"),
) -> pd.DataFrame:
    if taus is None:
        taus = np.round(np.arange(0.0, 0.51, 0.05), 2)
    records: list[dict[str, Any]] = []
    base = metrics_for_routed(df, "router_pred", name="router_argmax")
    records.append(
        {
            "mode": "argmax",
            "tau": 0.0,
            "accuracy": base.accuracy,
            "mean_cost_usd": base.mean_cost_usd,
            "oracle_match_rate": base.oracle_match_rate,
        }
    )
    work = df.copy()
    for mode in modes:
        for tau in taus:
            col = f"route_{mode}_{tau}"
            work[col] = work.apply(
                lambda r, m=mode, t=tau: route_top2(r, tau=t, mode=m),
                axis=1,
            )
            m = metrics_for_routed(work, col, name=col)
            records.append(
                {
                    "mode": mode,
                    "tau": float(tau),
                    "accuracy": m.accuracy,
                    "mean_cost_usd": m.mean_cost_usd,
                    "oracle_match_rate": m.oracle_match_rate,
                }
            )
    return pd.DataFrame(records)


def router_confidence_sweep(
    df: pd.DataFrame,
    *,
    thresholds: np.ndarray | None = None,
    fallback: str = "raw",
) -> pd.DataFrame:
    """When ``max_prob < threshold``, route to ``fallback``."""
    if thresholds is None:
        thresholds = np.round(np.linspace(0.25, 0.99, 16), 2)
    records = []
    for thr in thresholds:
        chosen = []
        for _, row in df.iterrows():
            if float(row["max_prob"]) >= thr:
                chosen.append(str(row["router_pred"]))
            else:
                chosen.append(fallback)
        tmp = df.copy()
        tmp["_chosen"] = chosen
        m = metrics_for_routed(tmp, "_chosen", name=f"conf>={thr}")
        records.append(
            {
                "threshold": float(thr),
                "fallback": fallback,
                "accuracy": m.accuracy,
                "mean_cost_usd": m.mean_cost_usd,
                "oracle_match_rate": m.oracle_match_rate,
            }
        )
    return pd.DataFrame(records)


def oracle_macro_f1(df: pd.DataFrame, pred_col: str) -> float:
    y = df[TARGET].astype(str)
    pred = df[pred_col].astype(str)
    return float(
        f1_score(y, pred, average="macro", labels=list(AGENTS), zero_division=0)
    )


def _feature_group(name: str) -> str:
    if name == "dataset":
        return "dataset"
    if name.startswith("dim_"):
        return "graph_cq"
    if name.startswith("emb_"):
        return "embedding"
    return "other"


def complexity_region_table(
    df: pd.DataFrame,
    *,
    complexity_col: str = "complexity_graph",
    n_bins: int = 4,
) -> pd.DataFrame:
    """
    Per complexity bin: dominant oracle agent, router behavior, execution accuracy.

    Core interpretability table for the paper (adaptive allocation by region).
    """
    work = df.dropna(subset=[complexity_col, TARGET]).copy()
    if work.empty:
        return pd.DataFrame()
    work["region"] = pd.qcut(
        work[complexity_col],
        q=min(n_bins, work[complexity_col].nunique()),
        duplicates="drop",
    )
    rows: list[dict[str, Any]] = []
    for region, grp in work.groupby("region", observed=True):
        oracle_pct = grp[TARGET].value_counts(normalize=True)
        row: dict[str, Any] = {
            "region": str(region),
            "n": len(grp),
            f"{complexity_col}_median": float(grp[complexity_col].median()),
            "dominant_oracle": oracle_pct.idxmax(),
            "oracle_mode_pct": float(oracle_pct.max()),
        }
        for a in AGENTS:
            row[f"oracle_pct_{a}"] = round(float(oracle_pct.get(a, 0.0)), 4)
        if "router_pred" in grp.columns:
            router_pct = grp["router_pred"].value_counts(normalize=True)
            row["dominant_router"] = router_pct.idxmax()
            for a in AGENTS:
                row[f"router_pct_{a}"] = round(float(router_pct.get(a, 0.0)), 4)
        if "exec_correct" in grp.columns:
            row["router_exec_accuracy"] = round(float(grp["exec_correct"].mean()), 4)
        if "exec_cost_usd" in grp.columns:
            row["mean_exec_cost_usd"] = round(float(grp["exec_cost_usd"].mean()), 6)
        rows.append(row)
    return pd.DataFrame(rows)


def _macro_f1_scorer(estimator: Any, X: pd.DataFrame, y: np.ndarray) -> float:
    """Callable scorer for permutation_importance (string agent labels)."""
    pred = estimator.predict(X)
    return float(
        f1_score(
            y,
            pred,
            average="macro",
            labels=list(AGENTS),
            zero_division=0,
        )
    )


def permutation_importance_table(
    pipe: Any,
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    n_repeats: int = 10,
    random_state: int = 42,
) -> pd.DataFrame:
    """Sklearn permutation importance (macro-F1) on a routed analysis frame."""
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        pipe,
        df[feature_cols],
        df[TARGET].astype(str).to_numpy(),
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=-1,
        scoring=_macro_f1_scorer,
    )
    out = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    ).sort_values("importance_mean", ascending=False)
    out["group"] = out["feature"].map(_feature_group)
    return out


def feature_group_importance_summary(imp: pd.DataFrame) -> pd.DataFrame:
    return (
        imp.groupby("group", as_index=False)
        .agg(importance_sum=("importance_mean", "sum"), n_features=("feature", "count"))
        .sort_values("importance_sum", ascending=False)
    )


def run_feature_importance(
    split: str,
    *,
    router_path: Path = DEFAULT_ROUTER,
    oracle_path: Path = DEFAULT_ORACLE,
    out_dir: Path,
    n_repeats: int = 10,
) -> dict[str, Any]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = build_analysis_frame(split, router_path=router_path, oracle_path=oracle_path)
    obj = load_router(router_path)
    imp = permutation_importance_table(
        obj["pipeline"], df, list(obj["feature_cols"]), n_repeats=n_repeats
    )
    grp = feature_group_importance_summary(imp)
    imp.to_csv(out_dir / f"permutation_importance_{split}.csv", index=False)
    grp.to_csv(out_dir / f"importance_by_group_{split}.csv", index=False)
    summary = {
        "split": split,
        "router": str(router_path),
        "n": len(df),
        "top_features": imp.head(12)[["feature", "importance_mean", "group"]].to_dict(
            orient="records"
        ),
        "group_importance": grp.to_dict(orient="records"),
    }
    (out_dir / f"importance_summary_{split}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return summary


# ── Plots ─────────────────────────────────────────────────────────────────────


def plot_complexity_region(
    df: pd.DataFrame,
    out_path: Path,
    *,
    n_bins: int = 10,
    title_suffix: str = "",
) -> None:
    """Central figure: ``complexity_graph`` vs oracle agent (heatmap + marginals)."""
    work = df.dropna(subset=["complexity_graph", TARGET]).copy()
    work["cg_bin"] = pd.qcut(
        work["complexity_graph"],
        q=min(n_bins, work["complexity_graph"].nunique()),
        duplicates="drop",
    )
    bin_labels = [str(b) for b in work["cg_bin"].cat.categories]
    agents = list(AGENTS)

    counts = (
        work.groupby(["cg_bin", TARGET], observed=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=agents, fill_value=0)
    )
    frac = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.08, wspace=0.05)
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    im = ax_main.imshow(frac.values, aspect="auto", cmap="Blues", vmin=0, vmax=frac.values.max() or 1)
    ax_main.set_xticks(range(len(agents)))
    ax_main.set_xticklabels(agents, rotation=30, ha="right")
    ax_main.set_yticks(range(len(bin_labels)))
    ax_main.set_yticklabels([lbl[:18] for lbl in bin_labels], fontsize=8)
    ax_main.set_xlabel("Oracle agent")
    ax_main.set_ylabel("complexity_graph bin (low → high)")
    ax_main.set_title(
        f"P(oracle agent | complexity region){title_suffix}\n(n={len(work)})",
        fontsize=12,
    )
    plt.colorbar(im, ax=ax_main, fraction=0.046, pad=0.04, label="fraction")

    for i in range(frac.shape[0]):
        for j in range(frac.shape[1]):
            v = frac.values[i, j]
            if v >= 0.08:
                ax_main.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8, color="white" if v > 0.35 else "black")

    # Strip / rug: complexity vs agent
    rng = np.random.default_rng(42)
    y_map = {a: i for i, a in enumerate(agents)}
    y_jitter = work[TARGET].map(y_map) + rng.uniform(-0.22, 0.22, len(work))
    ax_top.scatter(
        work["complexity_graph"],
        y_jitter,
        c=[AGENT_COLORS.get(a, "#888") for a in work[TARGET]],
        alpha=0.65,
        s=22,
        edgecolors="none",
    )
    ax_top.set_yticks(range(len(agents)))
    ax_top.set_yticklabels(agents)
    ax_top.set_xlabel("complexity_graph")
    ax_top.set_title("Oracle agent by complexity (jittered)")
    ax_top.set_xlim(work["complexity_graph"].min() - 0.02, work["complexity_graph"].max() + 0.02)

    overall = work[TARGET].value_counts().reindex(agents, fill_value=0)
    overall = overall / overall.sum()
    ax_right.barh(range(len(agents)), overall.values, color=[AGENT_COLORS[a] for a in agents])
    ax_right.set_yticks(range(len(agents)))
    ax_right.set_yticklabels([])
    ax_right.set_xlabel("P(agent)")
    ax_right.set_title("Overall", fontsize=9)
    ax_right.invert_yaxis()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_router_on_complexity(
    df: pd.DataFrame,
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    """Overlay router predictions vs oracle on complexity_graph."""
    work = df.dropna(subset=["complexity_graph"]).copy()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    rng = np.random.default_rng(7)
    for agent in AGENTS:
        sub_o = work[work[TARGET] == agent]
        sub_r = work[work["router_pred"] == agent]
        ax.scatter(
            sub_o["complexity_graph"],
            rng.uniform(0.0, 0.45, len(sub_o)),
            c=AGENT_COLORS[agent],
            marker="o",
            alpha=0.35,
            s=28,
            label=f"oracle {agent}" if agent == AGENTS[0] else None,
        )
        ax.scatter(
            sub_r["complexity_graph"],
            0.55 + rng.uniform(0.0, 0.45, len(sub_r)),
            c=AGENT_COLORS[agent],
            marker="x",
            alpha=0.7,
            s=40,
        )
    ax.axhline(0.5, color="#333", lw=0.8, ls="--", alpha=0.5)
    ax.text(0.02, 0.12, "oracle (dots)", transform=ax.transAxes, fontsize=9)
    ax.text(0.02, 0.62, "router (x)", transform=ax.transAxes, fontsize=9)
    ax.set_xlabel("complexity_graph")
    ax.set_yticks([])
    ax.set_title(f"Oracle vs router by complexity{title_suffix}")
    handles = [plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=AGENT_COLORS[a], markersize=8, label=a) for a in AGENTS]
    ax.legend(handles=handles, ncol=4, loc="upper right", fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reliability_calibration(
    df: pd.DataFrame,
    out_path: Path,
    *,
    n_bins: int = 10,
    title_suffix: str = "",
) -> dict[str, float]:
    """Max softmax vs routing correctness (label match + execution correct)."""
    work = df.dropna(subset=["max_prob"]).copy()
    bins = np.linspace(0, 1, n_bins + 1)

    def _curve(y_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        conf, acc, cnt = [], [], []
        for i in range(n_bins):
            lo, hi = bins[i], bins[i + 1]
            if i < n_bins - 1:
                mask = (work["max_prob"] >= lo) & (work["max_prob"] < hi)
            else:
                mask = (work["max_prob"] >= lo) & (work["max_prob"] <= hi)
            sub = work[mask]
            if len(sub) == 0:
                continue
            conf.append(sub["max_prob"].mean())
            acc.append(sub[y_col].mean())
            cnt.append(len(sub))
        return np.array(conf), np.array(acc), np.array(cnt)

    conf_l, acc_l, cnt_l = _curve("oracle_label_match")
    conf_e, acc_e, cnt_e = _curve("exec_correct")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, conf, acc, cnt, ylab in [
        (axes[0], conf_l, acc_l, cnt_l, "P(pred = oracle label)"),
        (axes[1], conf_e, acc_e, cnt_e, "P(correct | run predicted agent)"),
    ]:
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
        ax.plot(conf, acc, "o-", color="#4C78A8", lw=2, label="empirical")
        ax.bar(conf, acc, width=0.06, alpha=0.15, color="#4C78A8")
        for x, y, n in zip(conf, acc, cnt):
            ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Mean max softmax in bin")
        ax.set_ylabel(ylab)
        ax.legend(loc="lower right", fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[0].set_title(f"Label calibration{title_suffix}")
    axes[1].set_title(f"Outcome calibration{title_suffix}")
    fig.suptitle("Router confidence vs correctness", fontsize=12, y=1.02)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    ece_label = float(np.sum(cnt_l * np.abs(acc_l - conf_l)) / max(cnt_l.sum(), 1))
    ece_exec = float(np.sum(cnt_e * np.abs(acc_e - conf_e)) / max(cnt_e.sum(), 1))
    return {"ece_label_match": ece_label, "ece_exec_correct": ece_exec}


def plot_cost_accuracy(
    points: list[StrategyMetrics],
    curve_df: pd.DataFrame | None,
    top2_df: pd.DataFrame | None,
    out_path: Path,
    *,
    title_suffix: str = "",
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for p in points:
        if p.name.startswith("always_"):
            ax.scatter(
                p.mean_cost_usd * 1000,
                p.accuracy,
                s=120,
                c=AGENT_COLORS.get(p.name.replace("always_", ""), "#888"),
                marker="D",
                edgecolors="black",
                linewidths=0.6,
                zorder=3,
            )
            ax.annotate(
                p.name.replace("always_", ""),
                (p.mean_cost_usd * 1000, p.accuracy),
                textcoords="offset points",
                xytext=(6, 4),
                fontsize=9,
            )
        else:
            ax.scatter(
                p.mean_cost_usd * 1000,
                p.accuracy,
                s=100,
                c="#333",
                marker="*",
                zorder=4,
            )
            ax.annotate(p.name, (p.mean_cost_usd * 1000, p.accuracy), fontsize=8, xytext=(6, -10))

    if curve_df is not None and len(curve_df):
        ax.plot(
            curve_df["mean_cost_usd"] * 1000,
            curve_df["accuracy"],
            "-",
            color="#72B7B2",
            lw=1.5,
            alpha=0.8,
            label="router + confidence threshold",
        )

    if top2_df is not None and len(top2_df):
        for mode, sub in top2_df.groupby("mode"):
            if mode == "argmax":
                continue
            ax.plot(
                sub["mean_cost_usd"] * 1000,
                sub["accuracy"],
                "--",
                lw=1.2,
                alpha=0.75,
                label=f"top2 {mode}",
            )

    ax.set_xlabel("Mean cost per question (milli-USD)")
    ax.set_ylabel("Accuracy (oracle execution)")
    ax.set_title(f"Cost vs accuracy{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=7, ncol=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def dataset_regret_slice_table(df: pd.DataFrame) -> pd.DataFrame:
    """Mean utility regret by dataset (includes rows with n=0 for paper table)."""
    rows: list[dict[str, Any]] = []
    for key in DISCUSSION_DATASET_ORDER:
        sub = df.loc[df["dataset"] == key]
        n = int(len(sub))
        rows.append(
            {
                "dataset_key": key,
                "dataset": DATASET_DISPLAY_NAMES[key],
                "n": n,
                "mean_utility_regret": float(sub["utility_regret"].mean()) if n else np.nan,
                "accuracy": float(sub["exec_correct"].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def confidence_regret_slice_table(
    df: pd.DataFrame,
    *,
    edges: tuple[float, ...] = CONFIDENCE_BIN_EDGES,
) -> pd.DataFrame:
    """Mean utility regret by router max-softmax bin."""
    work = df.dropna(subset=["max_prob", "utility_regret"])
    rows: list[dict[str, Any]] = []
    n_bins = len(edges) - 1
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i < n_bins - 1:
            mask = (work["max_prob"] >= lo) & (work["max_prob"] < hi)
        else:
            mask = (work["max_prob"] >= lo) & (work["max_prob"] <= hi)
        sub = work[mask]
        n = int(len(sub))
        rows.append(
            {
                "max_prob_bin": f"{lo:.1f}–{hi:.1f}",
                "bin_lo": lo,
                "bin_hi": hi,
                "n": n,
                "mean_utility_regret": float(sub["utility_regret"].mean()) if n else np.nan,
                "mean_max_prob": float(sub["max_prob"].mean()) if n else np.nan,
                "accuracy": float(sub["exec_correct"].mean()) if n else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _format_slice_markdown(
    dataset_tbl: pd.DataFrame,
    conf_tbl: pd.DataFrame,
    *,
    split: str,
    router: str,
) -> str:
    lines = [
        f"# Discussion slices — {split}",
        "",
        f"Router: `{router}`",
        "",
        "## Dataset slices",
        "",
        "| Dataset | n | Regret | Accuracy |",
        "|---------|---|--------|----------|",
    ]
    for _, r in dataset_tbl.iterrows():
        regret = "—" if pd.isna(r["mean_utility_regret"]) or int(r["n"]) == 0 else f"{r['mean_utility_regret']:.3f}"
        acc = "—" if pd.isna(r["accuracy"]) or int(r["n"]) == 0 else f"{r['accuracy']:.2f}"
        lines.append(f"| {r['dataset']} | {int(r['n'])} | {regret} | {acc} |")
    lines.extend(
        [
            "",
            "_MMLU and GAIA are not in the internal QCE val/test splits (100×3 from hotpot/musique/math only)._",
            "",
            "## Confidence slices (max softmax)",
            "",
            "| Max prob bin | n | Regret | Accuracy |",
            "|--------------|---|--------|----------|",
        ]
    )
    for _, r in conf_tbl.iterrows():
        regret = "—" if pd.isna(r["mean_utility_regret"]) else f"{r['mean_utility_regret']:.3f}"
        acc = "—" if pd.isna(r["accuracy"]) else f"{r['accuracy']:.2f}"
        lines.append(f"| {r['max_prob_bin']} | {int(r['n'])} | {regret} | {acc} |")
    lines.append("")
    return "\n".join(lines)


def write_discussion_slices(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    split: str,
    router: str = "",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write dataset/confidence regret CSVs and a markdown summary for one split."""
    out_dir = Path(out_dir)
    dataset_tbl = dataset_regret_slice_table(df)
    conf_tbl = confidence_regret_slice_table(df)
    dataset_tbl.to_csv(out_dir / "slices_dataset_regret.csv", index=False)
    conf_tbl.to_csv(out_dir / "slices_confidence_regret.csv", index=False)
    md = _format_slice_markdown(dataset_tbl, conf_tbl, split=split, router=router)
    (out_dir / "discussion_slices.md").write_text(md, encoding="utf-8")
    return dataset_tbl, conf_tbl


def plot_top2_sweep(top2_df: pd.DataFrame, out_path: Path, *, title_suffix: str = "") -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mode, sub in top2_df.groupby("mode"):
        sub = sub.sort_values("tau")
        ax.plot(sub["tau"], sub["accuracy"], marker="o", label=mode, lw=1.5)
    ax.set_xlabel("τ — route top-1 only if p₁ − p₂ ≥ τ")
    ax.set_ylabel("Execution accuracy")
    ax.set_title(f"Top-2 routing sweep{title_suffix}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run_full_analysis(
    split: str,
    *,
    out_dir: Path,
    router_path: Path = DEFAULT_ROUTER,
    oracle_path: Path = DEFAULT_ORACLE,
    with_embeddings: bool = True,
) -> dict[str, Any]:
    df = build_analysis_frame(
        split,
        router_path=router_path,
        oracle_path=oracle_path,
        with_embeddings=with_embeddings,
    )
    suffix = f" ({split}, n={len(df)})"
    out_dir = Path(out_dir)

    plot_complexity_region(df, out_dir / "complexity_region_oracle.png", title_suffix=suffix)
    plot_router_on_complexity(df, out_dir / "complexity_region_router_vs_oracle.png", title_suffix=suffix)
    region_tbl = complexity_region_table(df)
    if len(region_tbl):
        region_tbl.to_csv(out_dir / "complexity_region_table.csv", index=False)
    cal = plot_reliability_calibration(df, out_dir / "reliability_calibration.png", title_suffix=suffix)

    baselines = baseline_strategies(df)
    conf_curve = router_confidence_sweep(df)
    top2 = top2_sweep(df)
    plot_cost_accuracy(baselines, conf_curve, top2, out_dir / "cost_vs_accuracy.png", title_suffix=suffix)
    plot_top2_sweep(top2, out_dir / "top2_tau_sweep.png", title_suffix=suffix)

    # Best top-2 row by accuracy
    top2_best = top2.loc[top2["accuracy"].idxmax()].to_dict() if len(top2) else {}

    router_m = next(p for p in baselines if p.name == "router_argmax")
    summary = {
        "split": split,
        "n": len(df),
        "router": str(router_path),
        "utility_lambda": DEFAULT_UTILITY_LAMBDA,
        "router_macro_f1": oracle_macro_f1(df, "router_pred"),
        "router_accuracy_exec": router_m.accuracy,
        "router_accuracy_label": float(df["oracle_label_match"].mean()),
        "router_mean_cost_usd": router_m.mean_cost_usd,
        "router_mean_utility_regret": router_m.mean_utility_regret,
        "router_em_per_usd": router_m.em_per_usd,
        "mean_max_prob": float(df["max_prob"].mean()),
        "calibration": cal,
        "baselines": [p.__dict__ for p in baselines],
        "best_top2": top2_best,
        "complexity_regions": region_tbl.to_dict(orient="records") if len(region_tbl) else [],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    conf_curve.to_csv(out_dir / "confidence_threshold_sweep.csv", index=False)
    top2.to_csv(out_dir / "top2_sweep.csv", index=False)
    df[
        [
            "training_id",
            "dataset",
            "complexity_graph",
            TARGET,
            "router_pred",
            "max_prob",
            "margin_top2",
            "exec_correct",
            "exec_cost_usd",
            "utility_regret",
            "oracle_label_match",
        ]
    ].to_csv(out_dir / "per_question.csv", index=False)

    write_discussion_slices(df, out_dir, split=split, router=str(router_path))

    return summary


def main_analyze(argv: list[str] | None = None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Router paper analyses and figures.")
    p.add_argument("--split", choices=("val", "test", "both"), default="both")
    p.add_argument("--router", type=Path, default=DEFAULT_ROUTER)
    p.add_argument("--oracle", type=Path, default=DEFAULT_ORACLE)
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--no-embeddings", action="store_true")
    p.add_argument(
        "--importance",
        action="store_true",
        help="Also run permutation feature importance for each split.",
    )
    p.add_argument("--importance-repeats", type=int, default=10)
    args = p.parse_args(argv)
    splits = ["val", "test"] if args.split == "both" else [args.split]
    root = args.out_dir or (REPO_ROOT / "results/reports/router")
    all_summaries: dict[str, dict] = {}
    for split in splits:
        out = root / split
        print(f"\n=== Analyzing {split!r} → {out} ===")
        summary = run_full_analysis(
            split,
            out_dir=out,
            router_path=args.router,
            oracle_path=args.oracle,
            with_embeddings=not args.no_embeddings,
        )
        all_summaries[split] = summary
        print(json.dumps(summary, indent=2))
        if args.importance:
            imp_dir = out / "importance"
            imp_sum = run_feature_importance(
                split,
                router_path=args.router,
                oracle_path=args.oracle,
                out_dir=imp_dir,
                n_repeats=args.importance_repeats,
            )
            print(json.dumps(imp_sum, indent=2))
    if len(all_summaries) > 1:
        combined = root / "summary_all_splits.json"
        combined.write_text(json.dumps(all_summaries, indent=2), encoding="utf-8")
        print(f"\nCombined summary: {combined}")

    router_label = Path(args.router).stem
    parts: list[str] = [
        f"# Router analysis — {router_label}",
        "",
        f"Model: `{args.router}`",
        "",
    ]
    for split in splits:
        md_path = root / split / "discussion_slices.md"
        if md_path.is_file():
            parts.append(md_path.read_text(encoding="utf-8").strip())
            parts.append("")
    disc_path = root / "DISCUSSION_SLICES.md"
    disc_path.write_text("\n".join(parts).strip() + "\n", encoding="utf-8")
    print(f"Discussion slices: {disc_path}")


if __name__ == "__main__":
    main_analyze()
