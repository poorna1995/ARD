from .swe_bench_loader import SWEBenchLoader
from .gaia_loader import GAIALoader
from .mmlu_pro_loader import MMLUProLoader
from .math_loader import MathLoader

LOADER_REGISTRY = {
    "swe_bench": SWEBenchLoader,
    "gaia": GAIALoader,
    "mmlu_pro": MMLUProLoader,
    "math_hard": MathLoader,
}


def get_loader(name, config, data_root="datasets"):
    if name not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'")
    return LOADER_REGISTRY[name](config=config, data_root=data_root)
