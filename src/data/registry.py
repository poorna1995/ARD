"""
src/data/registry.py
────────────────────────────────────────────────────────────────
Single place to register loaders.  Adding a new dataset = one line.
"""

from __future__ import annotations

from .math_loader     import MathLoader
from .musique_loader  import MuSiQueLoader
from .gaia_loader     import GAIALoader
from .mmlu_pro_loader import MMLUProLoader
from .hotpot_loader   import HotpotLoader

# name → loader class
REGISTRY: dict[str, type] = {
    "math":     MathLoader,
    "musique":  MuSiQueLoader,
    "gaia":     GAIALoader,
    "mmlu_pro": MMLUProLoader,
    "hotpot":   HotpotLoader,
}


def get_loader(name: str, cfg: dict, data_root: str = "datasets"):
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown dataset '{name}'. "
            f"Available: {sorted(REGISTRY)}"
        )
    return REGISTRY[name](cfg, data_root=data_root)