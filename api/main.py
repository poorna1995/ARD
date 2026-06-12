"""
Minimal routing API skeleton (Phase 6).

Run locally::

    uv sync --group api
    uv run uvicorn api.main:app --reload --port 8080

Endpoints:
  GET  /health  — liveness
  POST /route   — batch route feature rows (JSON body)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config.global_config.runtime import REQUIRED_LLM_KEYS, configure_logging, verify_env_keys
from router.config import ROUTER_MODEL_PATH
from router.router import load_router_frame

configure_logging()

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError("Install API extras: uv sync --group api") from exc


class RouteRequest(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)
    router_path: str | None = None


class RouteResponse(BaseModel):
    n: int
    router_path: str
    rows: list[dict[str, Any]]


app = FastAPI(title="research-work router", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/route", response_model=RouteResponse)
def route_batch(body: RouteRequest) -> RouteResponse:
    import pandas as pd

    secret = verify_env_keys(require=())  # routing only; no LLM required
    if not secret.ok and not Path(ROUTER_MODEL_PATH).is_file():
        raise HTTPException(status_code=503, detail="router model not available")

    df = pd.DataFrame(body.rows)
    path = Path(body.router_path) if body.router_path else ROUTER_MODEL_PATH
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"router not found: {path}")

    routed = load_router_frame(df, router_path=path)
    keep = [
        c
        for c in routed.columns
        if c
        in {
            "training_id",
            "query",
            "dataset",
            "router_pred",
            "assigned_agent",
            "max_prob",
            "margin_top2",
            "p_raw",
            "p_cot",
            "p_react",
            "p_multiagent",
        }
        or c.startswith("p_")
    ]
    records = routed[keep].to_dict(orient="records")
    return RouteResponse(n=len(records), router_path=str(path), rows=records)


@app.get("/ready")
def ready() -> dict[str, Any]:
    """Readiness: primary router on disk (+ optional OpenAI key for full pipeline)."""
    router_ok = ROUTER_MODEL_PATH.is_file()
    llm = verify_env_keys(require=REQUIRED_LLM_KEYS)
    return {
        "router_ok": router_ok,
        "openai_configured": llm.ok,
        "router_path": str(ROUTER_MODEL_PATH),
    }
