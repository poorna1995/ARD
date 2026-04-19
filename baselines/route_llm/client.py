from __future__ import annotations

import time
from typing import Any, Optional, Tuple

from baselines.route_llm.config import RoutellmConfig, router_model_field

# Import PyPI `routellm` (lm-sys). Local integration must NOT live in a top-level
# `routellm/` package folder: running `python baselines/run_baseline.py` puts
# `baselines/` on sys.path[0], which would shadow the real library.
ROUTELLM_IMPORT_ERROR: str | None = None
try:
    from routellm.controller import Controller

    ROUTELLM_AVAILABLE = True
except Exception as exc:
    Controller = Any  # type: ignore[misc, assignment]
    ROUTELLM_AVAILABLE = False
    ROUTELLM_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


def build_controller(cfg: RoutellmConfig) -> Any:
    if not ROUTELLM_AVAILABLE:
        msg = (
            "RouteLLM (PyPI) is not usable in this environment. From repo root run:\n"
            "  uv sync\n"
            "https://github.com/lm-sys/routellm"
        )
        if ROUTELLM_IMPORT_ERROR:
            msg += f"\n\nLast import attempt:\n  {ROUTELLM_IMPORT_ERROR}"
        raise ImportError(msg)
    return Controller(
        routers=[cfg.router],
        strong_model=cfg.strong_model,
        weak_model=cfg.weak_model,
    )


def routellm_chat_completion(
    client: Any,
    *,
    cfg: RoutellmConfig,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int,
    temperature: float,
    seed: Optional[int] = None,
) -> Tuple[str, dict[str, int], Optional[str], float, Optional[str]]:
    """
    Single chat completion via RouteLLM Controller (OpenAI-compatible).

    Returns
    -------
    text, usage dict, error, elapsed_s, resolved_model (from API if present).
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    predicted = ""
    error: Optional[str] = None
    resolved_model: Optional[str] = None
    model = router_model_field(cfg)

    start = time.perf_counter()
    try:
        create_kw: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if seed is not None:
            create_kw["seed"] = seed
        response = client.chat.completions.create(**create_kw)
        predicted = (response.choices[0].message.content or "").strip()
        if response.usage is not None:
            usage = {
                "prompt_tokens": response.usage.prompt_tokens or 0,
                "completion_tokens": response.usage.completion_tokens or 0,
                "total_tokens": response.usage.total_tokens or 0,
            }
        rid = getattr(response, "model", None)
        if rid:
            resolved_model = str(rid)
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - start
    return predicted, usage, error, elapsed, resolved_model
