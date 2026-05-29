"""Dataset loaders — see ``registry.py`` for the canonical registry."""

from .registry import REGISTRY, get_loader

# Back-compat alias used by older scripts
LOADER_REGISTRY = REGISTRY

__all__ = ["REGISTRY", "LOADER_REGISTRY", "get_loader"]
