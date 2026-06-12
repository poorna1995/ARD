"""Router batch inference and artifact I/O."""

from typing import Any

__all__ = [
    "attach_router_predictions",
    "load_router",
    "load_router_frame",
    "save_router",
    "top_k_agents",
    "top_k_from_row",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from routing.infer import predict as _predict

        return getattr(_predict, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
