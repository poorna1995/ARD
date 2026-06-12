"""Dataset names, aliases, and metadata keys."""

from __future__ import annotations

from dataclasses import dataclass

_DS_NAMES = frozenset({"gaia", "math", "musique", "mmlu", "hotpot"})
_DS_ALIASES: dict[str, str] = {
    "mmlu-pro": "mmlu",
    "mmlu_pro": "mmlu",
    "hotpotqa": "hotpot",
    "hotpot_qa": "hotpot",
    "musiqueqa": "musique",
    "mathematics": "math",
    "math_hard": "math",
}

# Legacy on-disk folder names (``datasets/raw|processed``, baseline roots).
_DISK_DIR_ALIASES: dict[str, str] = {
    "mmlu": "mmlu_pro",
}


@dataclass(frozen=True)
class Datasets:
    names: frozenset[str]
    aliases: dict[str, str]
    wiki_qa: frozenset[str]
    math: frozenset[str]
    chain: frozenset[str]
    router_train: tuple[str, ...]

    def resolve(self, dataset: str, *, strict: bool = True) -> str:
        key = str(dataset).strip().lower().replace("-", "_")
        canonical = self.aliases.get(key, key)
        if strict and canonical not in self.names:
            raise ValueError(f"dataset must be one of {sorted(self.names)}, got {dataset!r}")
        return canonical


DS = Datasets(
    names=_DS_NAMES,
    aliases=dict(_DS_ALIASES),
    wiki_qa=frozenset({"hotpot", "musique"}),
    math=frozenset({"math", "math_hard"}),
    chain=_DS_NAMES - frozenset({"gaia", "math"}),
    router_train=("math", "hotpot", "musique"),
)

META_KEYS = frozenset(
    {
        "attachment",
        "file_name",
        "level",
        "n_hops",
        "hop_type",
        "hop_name",
        "reasoning_type",
    }
)

DATASET_DISPLAY_NAMES: dict[str, str] = {
    "hotpot": "HotpotQA",
    "musique": "MuSiQue",
    "math": "MATH",
    "mmlu": "MMLU",
    "gaia": "GAIA",
}
DATASET_DISPLAY = DATASET_DISPLAY_NAMES
DISCUSSION_DATASET_ORDER = ["hotpot", "musique", "math", "mmlu", "gaia"]


def resolve_dataset_name(name: str) -> str:
    return DS.resolve(name or "", strict=False)


def disk_dir_name(dataset: str) -> str:
    """Directory name under ``datasets/raw``, ``processed``, baseline roots."""
    canonical = resolve_dataset_name(dataset)
    return _DISK_DIR_ALIASES.get(canonical, canonical)


def eval_parquet_stem(dataset: str) -> str:
    """``datasets/eval_samples/{stem}.parquet`` filename stem."""
    return resolve_dataset_name(dataset)


__all__ = [
    "DATASET_DISPLAY",
    "DATASET_DISPLAY_NAMES",
    "DISCUSSION_DATASET_ORDER",
    "DS",
    "Datasets",
    "META_KEYS",
    "disk_dir_name",
    "eval_parquet_stem",
    "resolve_dataset_name",
]
