"""Backward-compat shim → ``input.loaders.registry``."""

from input.loaders.registry import REGISTRY, get_loader

__all__ = ["REGISTRY", "get_loader"]
