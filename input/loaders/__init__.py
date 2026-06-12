"""Benchmark download loaders (HF → raw → processed parquets)."""

from input.loaders.registry import REGISTRY, get_loader

LOADER_REGISTRY = REGISTRY

__all__ = ["REGISTRY", "LOADER_REGISTRY", "get_loader"]
