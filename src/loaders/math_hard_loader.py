import re
from pathlib import Path

import pandas as pd
from datasets import load_dataset

import logging

from .base import BaseLoader

logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

KEEP_COLUMNS: list[str] = [
    "id",
    "query",
    "answer",
    "answer_extraction_method",
    "level",
    "type",
    "solution",
    "solution_length",
    "has_figure",
]

# ── Helpers ───────────────────────────────────────────────────────────────────

_ASY_RE    = re.compile(r"\[asy\]",          re.IGNORECASE)
_ANS_RE    = re.compile(
    r"(?:the\s+answer\s+is|answer\s*:)\s*(.+?)(?:[\n\r]|$)",
    re.IGNORECASE | re.DOTALL,
)


def _has_asymptote(text: str) -> bool:
    """Return True if the problem embeds an Asymptote figure ([asy] block)."""
    return bool(_ASY_RE.search(text))


def _extract_boxed_answer(solution: str) -> str | None:
    """
    Pull the final answer from a solution string using robust strategy:
      1) last balanced \\boxed{...}
      2) last balanced \\fbox{...}
      3) "the answer is ..." text pattern
      4) final non-empty line heuristic
    """
    if not isinstance(solution, str) or not solution.strip():
        return None

    boxed = _extract_last_balanced_macro_arg(solution, r"\boxed")
    if boxed:
        return _normalize_answer(boxed)

    fboxed = _extract_last_balanced_macro_arg(solution, r"\fbox")
    if fboxed:
        return _normalize_answer(fboxed)

    matches = _ANS_RE.findall(solution)
    if matches:
        return _normalize_answer(matches[-1])

    # Fallback: take the final meaningful line.
    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    if lines:
        return _normalize_answer(lines[-1])

    return None


def _extract_answer_with_method(solution: str) -> tuple[str | None, str]:
    """
    Return (answer, method) for auditing extraction quality.
    Methods: boxed, fbox, text_pattern, last_line, missing
    """
    if not isinstance(solution, str) or not solution.strip():
        return None, "missing"

    boxed = _extract_last_balanced_macro_arg(solution, r"\boxed")
    if boxed:
        return _normalize_answer(boxed), "boxed"

    fboxed = _extract_last_balanced_macro_arg(solution, r"\fbox")
    if fboxed:
        return _normalize_answer(fboxed), "fbox"

    matches = _ANS_RE.findall(solution)
    if matches:
        return _normalize_answer(matches[-1]), "text_pattern"

    lines = [ln.strip() for ln in solution.splitlines() if ln.strip()]
    if lines:
        return _normalize_answer(lines[-1]), "last_line"

    return None, "missing"


def _extract_last_balanced_macro_arg(text: str, macro: str) -> str | None:
    """
    Extract the argument content from the last balanced macro call, e.g.:
      \\boxed{...} or \\fbox{...}
    This handles nested braces, unlike simple regex matching.
    """
    positions: list[int] = []
    start = 0
    needle = f"{macro}{{"
    while True:
        i = text.find(needle, start)
        if i == -1:
            break
        positions.append(i + len(macro))  # points at "{"
        start = i + len(needle)

    for brace_start in reversed(positions):
        extracted = _extract_balanced_braces(text, brace_start)
        if extracted is not None:
            return extracted
    return None


def _extract_balanced_braces(text: str, brace_start: int) -> str | None:
    """
    Given index of '{', return content until matching '}' with nesting.
    """
    if brace_start >= len(text) or text[brace_start] != "{":
        return None

    depth = 0
    out: list[str] = []
    for i in range(brace_start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
            if depth > 1:
                out.append(ch)
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
            continue
        if depth >= 1:
            out.append(ch)
    return None


def _normalize_answer(answer: str) -> str:
    """
    Normalize common LaTeX escapes in boxed answers for cleaner display.
    Keeps mathematical structure but removes presentation-only wrappers.
    """
    text = answer.strip()

    # Remove outer math-mode delimiters like $...$
    while len(text) >= 2 and text.startswith("$") and text.endswith("$"):
        text = text[1:-1].strip()

    # Normalize common symbols/escapes.
    text = re.sub(r"\^\s*\\circ", "°", text)  # e.g. 156^\\circ -> 156°
    text = text.replace(r"\$", "$")
    text = text.replace(r"\%", "%")
    text = text.replace(r"\,", " ")
    text = text.replace(r"\!", "")
    text = text.replace(r"\ ", " ")

    # Collapse repeated spaces introduced by replacements.
    text = re.sub(r"\s+", " ", text).strip()

    # Remove common trailing punctuation from extracted free-text fallbacks.
    text = text.rstrip(" .,:;")

    return text


# ── Loader ────────────────────────────────────────────────────────────────────

class MathHardLoader(BaseLoader):
    """
    Loads the MATH-Hard benchmark (lighteval/MATH-Hard, test split).

    Download  →  raw parquet   (full test split, ~1 324 rows, all Level 5)
    Process   →  golden parquet (~617 rows after dropping Asymptote figures)

    The canonical '617-problem subset' used in the literature is produced by
    filtering out problems that embed [asy] figure code, which cannot be
    represented as plain text.  Solution length is used as a secondary
    within-level complexity proxy (all problems are already Level 5).
    """

    DATASET_NAME = "math_hard"

    # ── Paths ──────────────────────────────────────────────────────────────

    @property
    def _split(self) -> str:
        return self.config.get("split", "test")

    @property
    def _expected_rows_min(self) -> int | None:
        value = self.config.get("expected_rows_min")
        return int(value) if value is not None else None

    @property
    def _expected_rows_max(self) -> int | None:
        value = self.config.get("expected_rows_max")
        return int(value) if value is not None else None

    @property
    def _strict_expected_rows(self) -> bool:
        return bool(self.config.get("strict_expected_rows", False))

    @property
    def _raw_path(self) -> Path:
        # Keep raw files directly in datasets/raw (no dataset subfolder).
        path = self.data_root / "raw"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"{self.DATASET_NAME}_{self._split}.parquet"

    @property
    def _golden_path(self) -> Path:
        root = Path(self.config.get("data_dir", "data"))
        path = root / "golden"
        path.mkdir(parents=True, exist_ok=True)
        return path / f"golden_{self._split}.parquet"

    # ── Download ───────────────────────────────────────────────────────────

    def _download(self) -> pd.DataFrame:
        hf_repo = self.config.get("hf_repo", "lighteval/MATH-Hard")

        logger.info(f"  HF repo: {hf_repo}  split: {self._split}")
        raw_df = load_dataset(hf_repo, split=self._split).to_pandas()

        # ── Persist raw snapshot ──────────────────────────────────────────
        raw_df.to_parquet(self._raw_path, index=False)
        logger.info(f"  Raw data saved → {self._raw_path}  ({len(raw_df):,} rows)")

        return raw_df

    # ── Process ────────────────────────────────────────────────────────────

    def _process(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        df = raw_df.copy()

        # ── Rename to canonical columns ───────────────────────────────────
        df = df.rename(columns={"problem": "query"})

        # ── Flag problems that embed Asymptote figures ────────────────────
        # Filtering on [asy] blocks yields the canonical ~617-problem subset
        # used in the literature (all remaining problems are plain LaTeX).
        df["has_figure"] = df["query"].apply(_has_asymptote)
        before = len(df)
        df = df[~df["has_figure"]].copy()
        logger.info(
            f"  Dropped {before - len(df)} problems with [asy] figures → "
            f"{len(df)} remaining"
        )

        # ── Extract boxed answer from solution ────────────────────────────
        extracted = df["solution"].apply(_extract_answer_with_method)
        df["answer"] = extracted.apply(lambda x: x[0])
        df["answer_extraction_method"] = extracted.apply(lambda x: x[1])
        n_missing = df["answer"].isna().sum()
        if n_missing:
            logger.warning(f"  {n_missing} rows have no extracted final answer")
        method_counts = df["answer_extraction_method"].value_counts().to_dict()
        logger.info(f"  Answer extraction methods: {method_counts}")

        # ── Assign stable string IDs ──────────────────────────────────────
        df = df.reset_index(drop=True)
        df["id"] = [f"math_{i:04d}" for i in range(len(df))]

        # ── Secondary complexity proxy: solution length ───────────────────
        df["solution_length"] = df["solution"].str.len()
        logger.info(
            f"  solution_length: mean={df['solution_length'].mean():.0f} chars"
        )

        # ── Keep only needed columns ──────────────────────────────────────
        keep = [c for c in KEEP_COLUMNS if c in df.columns]
        df = df[keep].copy()

        # ── Drop rows with empty problem statements ───────────────────────
        before = len(df)
        df = df[df["query"].notna() & (df["query"].str.strip() != "")]
        logger.info(f"  Dropped {before - len(df)} rows with empty query")

        df = self._apply_max_samples(df)
        df = df.reset_index(drop=True)

        # ── Persist golden snapshot ───────────────────────────────────────
        df.to_parquet(self._golden_path, index=False)
        logger.info(f"  Golden data saved → {self._golden_path}  ({len(df):,} rows)")

        return df

    # ── Verify ─────────────────────────────────────────────────────────────

    def _extra_verify(self, df: pd.DataFrame) -> bool:
        ok = True

        if "type" in df.columns:
            dist = df["type"].value_counts().sort_index().to_dict()
            logger.info(f"  Subjects  : {dist}")

        if "level" in df.columns:
            dist = df["level"].value_counts().to_dict()
            logger.info(f"  Levels    : {dist}  (should be Level 5 only)")

        if "answer" in df.columns:
            n_null = df["answer"].isna().sum()
            if n_null:
                logger.info(f"  ⚠  {n_null} rows with null answer (no \\boxed{{}})")
            else:
                logger.info("  Answers   : OK (all rows have a boxed answer)")
        if "answer_extraction_method" in df.columns:
            method_counts = df["answer_extraction_method"].value_counts().to_dict()
            logger.info(f"  Extractor : {method_counts}")

        if "solution_length" in df.columns:
            logger.info(
                f"  Sol. len  : min={df['solution_length'].min()}  "
                f"max={df['solution_length'].max()}  "
                f"median={df['solution_length'].median():.0f}"
            )

        # Optional sanity check: expected row range can vary by dataset snapshot.
        # Keep this non-fatal by default so HF version updates do not fail verify().
        if (
            self._expected_rows_min is not None
            and self._expected_rows_max is not None
            and not (self._expected_rows_min <= len(df) <= self._expected_rows_max)
        ):
            logger.info(
                f"  ⚠  Row count {len(df)} outside expected range "
                f"{self._expected_rows_min}–{self._expected_rows_max}"
            )
            if self._strict_expected_rows:
                ok = False
        elif self._expected_rows_min is not None and self._expected_rows_max is not None:
            logger.info(
                f"  Row count : {len(df)} ✓ "
                f"(within expected {self._expected_rows_min}–{self._expected_rows_max})"
            )
        else:
            logger.info(f"  Row count : {len(df)}")

        return ok