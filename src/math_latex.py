"""Backward-compat shim → ``input.preprocess.math_latex``."""

from input.preprocess.math_latex import (
    extract_last_boxed,
    is_math_equiv,
    normalize_math_answer,
)

__all__ = ["extract_last_boxed", "is_math_equiv", "normalize_math_answer"]
