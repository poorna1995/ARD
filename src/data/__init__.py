"""Backward-compat shim → ``input.loaders``."""

from input.loaders import LOADER_REGISTRY, REGISTRY, get_loader

__all__ = ["REGISTRY", "LOADER_REGISTRY", "get_loader"]
