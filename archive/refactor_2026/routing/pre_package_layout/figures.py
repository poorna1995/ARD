"""Paper figures for frozen router comparison."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from routing.config import REPO_ROOT, ROUTER_MODEL_PATH
from routing.soft_train import BASELINES_CSV, SOFT_KL_OUT_DIR, _test_regret_from_baselines

SOFT_KL_FIGURES_DIR = REPO_ROOT / "results/experiments/soft_kl_cvec5_emb/figures"

# Display order for the 4-way internal-test regret figure.
FOUR_WAY_LABELS: tuple[tuple[str, str, str], ...] = (
    ("Hard HGBM (legacy)", "hard", "hgbm"),
    ("Hard Logistic", "hard", "logreg"),
    ("Soft Logistic (secondary)", "soft", "logreg_soft"),
    ("Soft HGBM (primary)", "soft", "hgbm_soft"),
)

COLORS = {
    "hard": "#9CA3AF",
    "soft": "#2563EB",
    "best": "#16A34A",
}


def _test_regret_from_soft_metrics(classifier: str) -> float | None:
    path = SOFT_KL_OUT_DIR / f"soft_kl_metrics_{classifier}.csv"
    if not path.is_file():
        return None
    sub = pd.read_csv(path)
    test = sub[sub["split"] == "test"]
    if len(test):
        return float(test.iloc[0]["mean_utility_regret"])
    return None


def load_four_way_test_regret() -> tuple[tuple[str, str, float], ...]:
    """Hard baselines + soft-KL val metrics (test split, λ=25, n=100)."""
    hard_hgbm = _test_regret_from_baselines("hgbm", "frozen")
    hard_logreg = _test_regret_from_baselines("logreg", "default")
    soft_hgbm = _test_regret_from_soft_metrics("hgbm")
    soft_logreg = _test_regret_from_soft_metrics("logreg")

    values: dict[str, float | None] = {
        "hgbm": hard_hgbm,
        "logreg": hard_logreg,
        "hgbm_soft": soft_hgbm,
        "logreg_soft": soft_logreg,
    }
    rows: list[tuple[str, str, float]] = []
    for label, group, key in FOUR_WAY_LABELS:
        val = values[key]
        if val is None:
            raise FileNotFoundError(
                f"missing test regret for {key!r} — run soft-train / baselines first"
            )
        rows.append((label, group, val))
    return tuple(rows)


def write_four_way_regret_csv(
    rows: tuple[tuple[str, str, float], ...],
    out_path: Path | None = None,
) -> Path:
    out_path = Path(out_path or SOFT_KL_FIGURES_DIR / "four_way_test_regret.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"model": lab, "group": grp, "test_regret": r} for lab, grp, r in rows]
    ).to_csv(out_path, index=False)
    return out_path


def plot_regret_comparison_test(
    out_path: Path | None = None,
    *,
    rows: tuple[tuple[str, str, float], ...] | None = None,
    dpi: int = 200,
) -> Path:
    """Bar chart of mean utility regret on internal test (Figure 1)."""
    out_path = Path(out_path or SOFT_KL_FIGURES_DIR / "regret_comparison_test.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = rows or load_four_way_test_regret()
    write_four_way_regret_csv(data)

    labels = [r[0] for r in data]
    groups = [r[1] for r in data]
    regrets = [r[2] for r in data]
    best_idx = int(np.argmin(regrets))
    colors = [
        COLORS["best"] if i == best_idx else COLORS[g] for i, g in enumerate(groups)
    ]

    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(x, regrets, color=colors, edgecolor="#1F2937", linewidth=0.8, width=0.62)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=18, ha="right", fontsize=10)
    ax.set_ylabel("Mean utility regret (test, $n{=}100$, $\\lambda{=}25$)", fontsize=11)
    ax.set_title(
        "Router comparison: hard vs soft supervision\n"
        f"(deploy: {ROUTER_MODEL_PATH.name})",
        fontsize=11,
        pad=12,
    )
    ax.set_ylim(0, max(regrets) * 1.18)
    ax.axhline(
        0.274,
        color="#54A24B",
        linestyle="--",
        linewidth=1.2,
        alpha=0.85,
        label="Always ReAct (0.274)",
    )
    ax.grid(axis="y", alpha=0.35, linestyle="-", linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    for bar, val in zip(bars, regrets, strict=True):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.008,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="semibold",
        )

    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main() -> None:
    rows = load_four_way_test_regret()
    csv_path = write_four_way_regret_csv(rows)
    fig_path = plot_regret_comparison_test(rows=rows)
    meta = {
        "router_model_path": str(ROUTER_MODEL_PATH.relative_to(REPO_ROOT)),
        "four_way_test_regret": [
            {"model": lab, "group": grp, "test_regret": r} for lab, grp, r in rows
        ],
    }
    meta_path = SOFT_KL_FIGURES_DIR / "regret_comparison_test.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {fig_path}")
    print(f"      {csv_path}")
    print(f"      {meta_path}")
    for lab, _, r in rows:
        print(f"  {lab:<28} {r:.6f}")


if __name__ == "__main__":
    main()
