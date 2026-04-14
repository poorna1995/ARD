"""
src/utils/helpers.py
────────────────────────────────────────────────────────────────
Shared utility functions used across loaders and estimators.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
import yaml
import math
import logging
from functools import lru_cache
import spacy

from ..config import BAND_THRESHOLDS

logger = logging.getLogger(__name__)



"""
utils.py
========
Shared helpers used across all feature layers.

- clip_to_range   : clamp float to [lo, hi]
- min_max_scale   : linear normalisation to [0, 1]
- log_scale       : log-compressed normalisation (good for skewed counts)
- tree_depth      : recursive dependency-tree depth
- get_doc         : spaCy parse, cached with a module-level singleton
- assign_band     : map composite score C → human-readable tier
"""



# ─────────────────────────────────────────────────────────────────────────────
# spaCy singleton
# ─────────────────────────────────────────────────────────────────────────────

_NLP: Optional[spacy.language.Language] = None


def get_nlp() -> spacy.language.Language:
    """
    Lazily load and return the shared spaCy model (en_core_web_md).

    The model is initialised once per process.  All feature layers call
    ``get_doc()`` which delegates here, so there is no risk of loading
    the model multiple times.
    """
    global _NLP
    if _NLP is None:
        try:
            _NLP = spacy.load("en_core_web_md")
            logger.info("spaCy model 'en_core_web_md' loaded.")
        except OSError:
            logger.warning(
                "en_core_web_md not found — falling back to en_core_web_sm. "
                "Run: python -m spacy download en_core_web_md"
            )
            _NLP = spacy.load("en_core_web_sm")
    return _NLP


def get_doc(text: str) -> spacy.tokens.Doc:
    """
    Parse *text* with the shared spaCy model and return a ``Doc``.

    The ``Doc`` contains tokens, POS tags, dependency arcs, and named
    entity spans — everything the L1 and L2 feature layers need.
    """
    return get_nlp()(text)


# ─────────────────────────────────────────────────────────────────────────────
# Numeric helpers
# ─────────────────────────────────────────────────────────────────────────────

def clip_to_range(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """
    Clamp *value* so it never falls outside [*lo*, *hi*].

    Needed because floating-point addition of weighted sub-scores can produce
    tiny over/underflows such as -0.0001 or 1.00003.

    Examples
    --------
    >>> clip_to_range(1.05)   # → 1.0
    >>> clip_to_range(-0.02)  # → 0.0
    >>> clip_to_range(0.73)   # → 0.73
    """
    return max(lo, min(hi, value))


def min_max_scale(value: float, lo: float, hi: float) -> float:
    """
    Map *value* from its natural range [*lo*, *hi*] into [0, 1].

    Formula: ``(value - lo) / (hi - lo)``

    Edge case: if ``lo == hi`` (zero variance), returns 0.0 to avoid
    division by zero.

    Examples
    --------
    >>> min_max_scale(50, 8, 167)   # token_count → 0.264
    >>> min_max_scale(3,  1, 6)     # bloom_level → 0.400
    """
    if hi == lo:
        return 0.0
    return clip_to_range((value - lo) / (hi - lo))


def log_scale(value: float, lo: float, hi: float) -> float:
    """
    Logarithmically compress *value* from [*lo*, *hi*] into [0, 1].

    Useful when the raw distribution is right-skewed (e.g. token counts).
    Short queries get more resolution; long ones are gently squished.

    Formula: ``(log(v) - log(lo)) / (log(hi) - log(lo))``
    where *v* is first clipped to [*lo*, *hi*].
    """
    value = max(lo, min(hi, float(value)))
    if lo <= 0:
        raise ValueError(f"log_scale requires lo > 0, got lo={lo}")
    return clip_to_range(
        (math.log(value) - math.log(lo)) / (math.log(hi) - math.log(lo))
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dependency-tree helpers
# ─────────────────────────────────────────────────────────────────────────────

def tree_depth(token: spacy.tokens.Token) -> int:
    """
    Recursively compute the maximum depth of *token*'s subtree.

    A leaf node (no children) has depth 1.  Every parent node is
    1 + max(depth of children).

    Deeper values → more embedded clause structure → harder grammar.

    Example tree for "The big dog quickly ran away"::

        ran (root, depth=3)
        ├── dog (depth=2)
        │   ├── The  (depth=1)
        │   └── big  (depth=1)
        └── away (depth=1)
    """
    children = list(token.children)
    if not children:
        return 1
    return 1 + max(tree_depth(child) for child in children)


# ─────────────────────────────────────────────────────────────────────────────
# Band assignment
# ─────────────────────────────────────────────────────────────────────────────

def assign_band(score: float) -> str:
    """
    Map a composite score *C* ∈ [0, 1] to a human-readable complexity tier.

    Thresholds (from ``config.BAND_THRESHOLDS``)::

        [0.00, 0.20) → LOW
        [0.20, 0.45) → MEDIUM
        [0.45, 0.60) → HIGH
        [0.60, 1.00] → VERY HIGH
    """
    for upper, label in BAND_THRESHOLDS:
        if score < upper:
            return label
    return BAND_THRESHOLDS[-1][1]





# ── Logging ────────────────────────────────────────────────────


def setup_logging(level: str = "INFO", log_file: str | None = None):
    """Configure root logger with console + optional file handler."""
    fmt = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s"
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
        force=True,
    )


# ── Config loading ─────────────────────────────────────────────


def load_config(path: str | Path) -> dict:
    """Load a YAML config file and return as dict."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ── Normalization ──────────────────────────────────────────────


def minmax_normalize(series: pd.Series, clip: bool = True) -> pd.Series:
    """
    Normalize a series to [0, 1] using min-max scaling.
    Handles NaN by leaving them as NaN.
    """
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    result = (series - mn) / (mx - mn)
    if clip:
        result = result.clip(0.0, 1.0)
    return result


def percentile_normalize(series: pd.Series) -> pd.Series:
    """
    Normalize using percentile rank (robust to outliers).
    Useful for structural estimators with skewed distributions.
    """
    return series.rank(pct=True)


def sigmoid_normalize(series: pd.Series, center: float | None = None) -> pd.Series:
    """
    Apply sigmoid to map any real-valued series to (0, 1).
    center: if provided, shifts sigmoid so that center maps to 0.5.
    """
    x = series.copy().astype(float)
    if center is not None:
        x = x - center
    return 1 / (1 + np.exp(-x))


# ── Text preprocessing ─────────────────────────────────────────


def truncate_text(text: str, max_chars: int = 2000) -> str:
    """Truncate long text to max_chars. Used before tokenizing."""
    if len(text) > max_chars:
        return text[:max_chars] + " [TRUNCATED]"
    return text


def count_words(text: str) -> int:
    """Simple whitespace-based word count."""
    return len(str(text).split())


def count_sentences(text: str) -> int:
    """Approximate sentence count by splitting on sentence-ending punctuation."""
    import re

    sentences = re.split(r"[.!?]+", str(text))
    return max(1, len([s for s in sentences if s.strip()]))


# ── Evaluation helpers ─────────────────────────────────────────


def spearman_rho(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Spearman rank correlation between predicted and true complexity."""
    from scipy.stats import spearmanr

    mask = y_true.notna() & y_pred.notna()
    if mask.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(y_true[mask], y_pred[mask])
    return round(float(rho), 4)


def auroc_binary(
    y_true: pd.Series, y_score: pd.Series, threshold: float = 0.5
) -> float:
    """
    AUROC treating samples above `threshold` as 'hard' (positive class).
    y_true: ground truth complexity [0,1]
    y_score: estimated complexity [0,1]
    """
    from sklearn.metrics import roc_auc_score

    labels = (y_true >= threshold).astype(int)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")  # only one class present
    mask = y_score.notna()
    return round(roc_auc_score(labels[mask], y_score[mask]), 4)


def summarize_results(results: dict[str, dict]) -> pd.DataFrame:
    """
    Convert a nested results dict to a summary DataFrame.
    results = {dataset_name: {estimator_name: {metric: value}}}
    """
    rows = []
    for dataset, estimators in results.items():
        for estimator, metrics in estimators.items():
            row = {"dataset": dataset, "estimator": estimator}
            row.update(metrics)
            rows.append(row)
    return pd.DataFrame(rows)
