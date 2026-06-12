"""Router training pipeline and experiment CLIs."""

from typing import Any

__all__ = [
    "EvalResult",
    "TrainSpec",
    "evaluate",
    "fit_router",
    "main",
    "run_ablation_all",
    "run_ablation_cvec5",
    "run_ablation_cvec7",
    "run_ablation_trust",
    "run_feature_ablation",
    "run_leave_one_dataset_out",
    "run_seed_stability",
    "train_router",
]


def __getattr__(name: str) -> Any:
    if name in ("EvalResult", "TrainSpec", "evaluate", "fit_router", "train_router"):
        from routing.train import pipeline as _pipeline

        return getattr(_pipeline, name)
    if name in __all__:
        from routing.train import experiments as _experiments

        return getattr(_experiments, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
