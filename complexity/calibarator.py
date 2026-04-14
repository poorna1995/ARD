"""
calibrate.py
============
Percentile-based calibration for the SurfaceFeatures normalisation bounds.

Run this notebook cell / script after collecting a representative dataset to
update the L1 normalisation ranges so they reflect actual query distributions
rather than hard-coded defaults.

Workflow
--------
1. ``extract_surface_features(datasets)``   → feature_df
2. ``percentile_report(feature_df)``        → inspect distribution
3. ``recommended_bounds(feature_df)``       → candidate LO/HI pairs
4. ``saturation_audit(feature_df)``         → check floor/ceiling %
5. ``apply_bounds("P5_P95")``               → write chosen bounds into config

Flags
-----
USE_LOG_SCALE_TC   : bool
    When True the patched ``SurfaceFeatures.score()`` applies log-compression
    to ``token_count`` (better for right-skewed distributions).
USE_DATASET_BOUNDS : bool
    When True, per-dataset bounds are stored and used during scoring
    (pass the ``dataset`` kwarg to ``score()``).
"""

from __future__ import annotations

import math
import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

import complexity.config as cfg
from complexity.features.surface import SurfaceFeatures
from complexity.utils import clip_to_range, log_scale, min_max_scale

logger = logging.getLogger(__name__)

# ── Toggle flags ─────────────────────────────────────────────────────────────
USE_DATASET_BOUNDS: bool = False   # True → per-dataset rulers
USE_LOG_SCALE_TC:   bool = True    # True → log-compress token_count

FEATURES: List[str] = cfg.SURFACE_FEATURE_NAMES   # ["token_count", "ner_density", "mattr"]
STRATEGIES: Dict[str, Tuple[int, int]] = cfg.CALIBRATION_STRATEGIES


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_surface_features(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Run ``SurfaceFeatures.raw_features()`` on every query in every dataset
    and collect results into a single DataFrame.

    Parameters
    ----------
    datasets : dict[name → DataFrame]
        Each DataFrame must have a ``query`` column.

    Returns
    -------
    pd.DataFrame with columns: dataset, token_count, ner_density, mattr
    """
    rows: List[dict] = []
    for name, df in datasets.items():
        logger.info("Extracting features: %s (%d queries)", name.upper(), len(df))
        for _, row in tqdm(df.iterrows(), total=len(df), desc=name, leave=False):
            sf  = SurfaceFeatures(str(row["query"]))
            raw = sf.raw_features()
            rows.append({"dataset": name, **raw})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Percentile report
# ─────────────────────────────────────────────────────────────────────────────

def percentile_report(df: pd.DataFrame, label: str = "GLOBAL") -> pd.DataFrame:
    """
    Print and return a DataFrame of per-feature statistics including key
    percentiles.  Use to understand distribution shape before choosing bounds.
    """
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    rows = []
    for feat in FEATURES:
        s = df[feat]
        rows.append({
            "feature": feat,
            "mean":    round(s.mean(), 4),
            "std":     round(s.std(),  4),
            "min":     round(s.min(),  4),
            "max":     round(s.max(),  4),
            **{f"p{p}": round(float(np.percentile(s, p)), 4) for p in pcts},
        })
    report = pd.DataFrame(rows).set_index("feature")
    sep = "═" * 60
    print(f"\n{sep}\n  {label}\n{sep}")
    print(report.to_string())
    return report


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Recommended bounds
# ─────────────────────────────────────────────────────────────────────────────

def recommended_bounds(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute P2/P5/P10 and P90/P95/P98 as candidate normalisation bounds.

    Returns a DataFrame indexed by feature with columns like
    ``P5_P95_LO``, ``P5_P95_HI``, etc.
    """
    rows = []
    for feat in FEATURES:
        s   = df[feat]
        row = {"feature": feat}
        for label, (lo_pct, hi_pct) in STRATEGIES.items():
            row[f"{label}_LO"] = round(float(np.percentile(s, lo_pct)), 4)
            row[f"{label}_HI"] = round(float(np.percentile(s, hi_pct)), 4)
        rows.append(row)
    return pd.DataFrame(rows).set_index("feature")


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Saturation audit
# ─────────────────────────────────────────────────────────────────────────────

def saturation_audit(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each feature, report the fraction of queries that score exactly
    0.0 (floor) or 1.0 (ceiling) under the *current* bounds.

    If either exceeds 10 %, the bound should be loosened.
    """
    current = {
        "token_count": (cfg.SURFACE_TC_LO,    cfg.SURFACE_TC_HI),
        "ner_density": (cfg.SURFACE_NER_LO,   cfg.SURFACE_NER_HI),
        "mattr":       (cfg.SURFACE_MATTR_LO, cfg.SURFACE_MATTR_HI),
    }
    rows = []
    for feat, (lo, hi) in current.items():
        s = df[feat]
        rows.append({
            "feature":          feat,
            "current_LO":       lo,
            "current_HI":       hi,
            "pct_floor (%)":    round(100 * (s <= lo).mean(), 1),
            "pct_ceiling (%)":  round(100 * (s >= hi).mean(), 1),
        })
    audit = pd.DataFrame(rows).set_index("feature")
    sep = "═" * 60
    print(f"\n{sep}\n  SATURATION AUDIT  (>10% = adjust that bound)\n{sep}")
    print(audit.to_string())
    return audit


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Apply chosen bounds + optional log-scale patch
# ─────────────────────────────────────────────────────────────────────────────

def _write_global_bounds(bounds_df: pd.DataFrame, strategy: str) -> None:
    """Write the chosen percentile bounds into the live ``complexity.config`` module."""
    row_tc  = bounds_df.loc["token_count"]
    row_ner = bounds_df.loc["ner_density"]
    row_ma  = bounds_df.loc["mattr"]

    cfg.SURFACE_TC_LO    = float(row_tc[f"{strategy}_LO"])
    cfg.SURFACE_TC_HI    = float(row_tc[f"{strategy}_HI"])
    cfg.SURFACE_NER_LO   = float(row_ner[f"{strategy}_LO"])
    cfg.SURFACE_NER_HI   = float(row_ner[f"{strategy}_HI"])
    cfg.SURFACE_MATTR_LO = float(row_ma[f"{strategy}_LO"])
    cfg.SURFACE_MATTR_HI = float(row_ma[f"{strategy}_HI"])

    logger.info(
        "Global bounds set — TC=[%.4f, %.4f]  NER=[%.4f, %.4f]  MATTR=[%.4f, %.4f]",
        cfg.SURFACE_TC_LO, cfg.SURFACE_TC_HI,
        cfg.SURFACE_NER_LO, cfg.SURFACE_NER_HI,
        cfg.SURFACE_MATTR_LO, cfg.SURFACE_MATTR_HI,
    )


def _build_patched_score(
    dataset_bounds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None,
):
    """
    Return a patched ``SurfaceFeatures.score()`` method that honours
    ``USE_LOG_SCALE_TC`` and (optionally) ``USE_DATASET_BOUNDS``.

    Parameters
    ----------
    dataset_bounds : dict[dataset_name → dict[feature → (lo, hi)]], optional
        Required when ``USE_DATASET_BOUNDS`` is True.
    """

    def patched_score(
        self: SurfaceFeatures,
        dataset: Optional[str] = None,
    ) -> Tuple[float, dict]:
        """
        Patched SurfaceFeatures.score() respecting calibration flags.

        Parameters
        ----------
        dataset : str, optional
            Dataset name when using per-dataset bounds.
        """
        raw = self.raw_features()

        if USE_DATASET_BOUNDS and dataset is not None:
            if dataset_bounds is None:
                raise RuntimeError("Dataset bounds were not computed — run apply_bounds() first.")
            if dataset not in dataset_bounds:
                raise ValueError(
                    f"Unknown dataset '{dataset}'. "
                    f"Available: {list(dataset_bounds.keys())}"
                )
            bmap   = dataset_bounds[dataset]
            tc_lo  = bmap["token_count"][0];  tc_hi  = bmap["token_count"][1]
            ner_lo = bmap["ner_density"][0];  ner_hi = bmap["ner_density"][1]
            ma_lo  = bmap["mattr"][0];        ma_hi  = bmap["mattr"][1]
        else:
            tc_lo, tc_hi   = cfg.SURFACE_TC_LO,    cfg.SURFACE_TC_HI
            ner_lo, ner_hi = cfg.SURFACE_NER_LO,   cfg.SURFACE_NER_HI
            ma_lo, ma_hi   = cfg.SURFACE_MATTR_LO, cfg.SURFACE_MATTR_HI

        scaler = log_scale if USE_LOG_SCALE_TC else min_max_scale
        s_tc    = scaler(raw["token_count"], tc_lo, tc_hi)
        s_ner   = min_max_scale(raw["ner_density"], ner_lo, ner_hi)
        s_mattr = min_max_scale(raw["mattr"],       ma_lo,  ma_hi)

        S = (
            cfg.SURFACE_W_TC    * s_tc
            + cfg.SURFACE_W_NER   * s_ner
            + cfg.SURFACE_W_MATTR * s_mattr
        )
        return clip_to_range(S), raw

    return patched_score


def apply_bounds(
    global_bounds: pd.DataFrame,
    dataset_bounds_map: Optional[Dict[str, pd.DataFrame]] = None,
    strategy: str = "P5_P95",
) -> None:
    """
    Apply calibration bounds to the running process.

    Updates the ``complexity.config`` module in-place and monkey-patches
    ``SurfaceFeatures.score()`` to respect the calibration flags.

    Parameters
    ----------
    global_bounds       : output of ``recommended_bounds(feature_df)``
    dataset_bounds_map  : dict[name → output of ``recommended_bounds``]
                          Required only when ``USE_DATASET_BOUNDS`` is True.
    strategy            : one of "P2_P98", "P5_P95", "P10_P90"
    """
    print(
        f"\nApplying {strategy} bounds "
        f"[dataset_bounds={USE_DATASET_BOUNDS}, log_scale_tc={USE_LOG_SCALE_TC}]"
    )

    _write_global_bounds(global_bounds, strategy)

    ds_bounds: Optional[Dict[str, Dict[str, Tuple[float, float]]]] = None

    if USE_DATASET_BOUNDS and dataset_bounds_map:
        ds_bounds = {
            ds: {
                feat: (
                    float(bdf.loc[feat, f"{strategy}_LO"]),
                    float(bdf.loc[feat, f"{strategy}_HI"]),
                )
                for feat in FEATURES
            }
            for ds, bdf in dataset_bounds_map.items()
        }
        print("  Per-dataset bounds stored.")
        for ds, bmap in ds_bounds.items():
            print(f"    {ds:<12}: {bmap}")

    SurfaceFeatures.score = _build_patched_score(ds_bounds)
    print("✅ Bounds applied and SurfaceFeatures.score() patched.\n")


# ─────────────────────────────────────────────────────────────────────────────
# Convenience: run full calibration in one call
# ─────────────────────────────────────────────────────────────────────────────

def run_calibration(
    datasets: Dict[str, pd.DataFrame],
    strategy: str = "P5_P95",
) -> Dict[str, pd.DataFrame]:
    """
    Execute the full five-step calibration pipeline and return a dict of
    diagnostic DataFrames.

    Parameters
    ----------
    datasets : dict[name → DataFrame]
    strategy : percentile strategy to apply

    Returns
    -------
    dict with keys:
      feature_df      : raw feature DataFrame
      global_report   : percentile stats
      global_bounds   : recommended LO/HI pairs
      audit           : saturation audit
      dataset_bounds  : dict[name → bounds DataFrame]
    """
    feature_df = extract_surface_features(datasets)
    print(f"\n✅ {len(feature_df):,} total queries extracted")

    global_report  = percentile_report(feature_df, "GLOBAL — all datasets pooled")
    global_bounds  = recommended_bounds(feature_df)

    print(f"\n{'═'*60}\n  RECOMMENDED BOUNDS — GLOBAL\n{'═'*60}")
    print(global_bounds.to_string())

    dataset_bounds: Dict[str, pd.DataFrame] = {}
    for name, group in feature_df.groupby("dataset"):
        percentile_report(group, f"{name.upper()} — {len(group):,} queries")
        dataset_bounds[name] = recommended_bounds(group)
        print(f"\n{'═'*60}\n  RECOMMENDED BOUNDS — {name.upper()}\n{'═'*60}")
        print(dataset_bounds[name].to_string())

    audit = saturation_audit(feature_df)

    apply_bounds(
        global_bounds     = global_bounds,
        dataset_bounds_map = dataset_bounds if USE_DATASET_BOUNDS else None,
        strategy           = strategy,
    )

    return {
        "feature_df":     feature_df,
        "global_report":  global_report,
        "global_bounds":  global_bounds,
        "audit":          audit,
        "dataset_bounds": dataset_bounds,
    }