from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RoutellmConfig:
    """Strong/weak pair + router id + cost–quality threshold (see RouteLLM paper)."""

    router: str = "mf"
    threshold: float = 0.11593
    strong_model: str = "gpt-4o"
    weak_model: str = "gpt-4o-mini"


def router_model_field(cfg: RoutellmConfig) -> str:
    """OpenAI `model` argument: `router-<name>-<threshold>`."""
    t = cfg.threshold
    ts = f"{t:.12f}".rstrip("0").rstrip(".")
    if ts == "" or ts == "-":
        ts = str(t)
    return f"router-{cfg.router}-{ts}"
