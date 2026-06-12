"""Backward-compat shim → ``input.loaders.mmlu_pro_loader``."""

from input.loaders.mmlu_pro_loader import MMLUProLoader, append_options_to_query

__all__ = ["MMLUProLoader", "append_options_to_query"]
