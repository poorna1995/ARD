from .swe_bench_loader import SWEBenchLoader
from .gaia_loader import GAIALoader
from .mmlu_pro_loader import MMLUProLoader
from .musique_loader import MuSiQueLoader
from .aime_loader import AIMELoader

LOADER_REGISTRY = {
    "swe_bench": SWEBenchLoader,
    "gaia": GAIALoader,
    "mmlu_pro": MMLUProLoader,
    "musique": MuSiQueLoader,
    "aime": AIMELoader,
}


def get_loader(name, config, data_root="datasets"):
    if name not in LOADER_REGISTRY:
        raise ValueError(f"Unknown dataset: '{name}'")
    return LOADER_REGISTRY[name](config=config, data_root=data_root)
