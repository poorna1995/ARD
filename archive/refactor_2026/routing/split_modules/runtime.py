"""RuntimeRouter and agent cascade execution."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

import pandas as pd

from routing.datasets import resolve_dataset_name
from routing.features.columns import overall_complexity
from routing.infer.predict import top_k_agents, top_k_from_row
from routing.config import DEFAULT_AGENT_MODEL, PROBA_COLS
def agent_run_failed(response: dict[str, Any]) -> bool:
    """True if an agent run should trigger cascade (error flag or empty answer)."""
    if response.get("is_failed"):
        return True
    ans = str(response.get("predicted_answer") or response.get("answer") or "").strip()
    return not ans


def run_agent_cascade(
    row: pd.Series | dict[str, Any],
    *,
    query: str,
    dataset: str,
    model: str = DEFAULT_AGENT_MODEL,
    k: int = 3,
    expected_answer: Any = None,
    **run_kw: Any,
) -> dict[str, Any]:
    """
  Try up to ``k`` router-ranked agents until one succeeds.

    Returns ``executed_agent``, ``cascade_rank`` (1-based), ``cascade_attempts``, and final ``response``.
    """
    from agent.registry import STRATEGY_REGISTRY, run_strategy

    ranked = top_k_from_row(row, k=k)
    ds = resolve_dataset_name(dataset)
    attempts: list[dict[str, Any]] = []
    final: dict[str, Any] | None = None

    for rank, (agent, prob) in enumerate(ranked, start=1):
        if agent not in STRATEGY_REGISTRY:
            attempts.append({"rank": rank, "agent": agent, "prob": prob, "skipped": True})
            continue
        try:
            response = run_strategy(
                agent,
                query,
                model=model,
                dataset=ds,
                expected_answer=expected_answer,
                **run_kw,
            )
            payload = asdict(response) if hasattr(response, "__dataclass_fields__") else dict(response)
            failed = agent_run_failed(payload)
            attempts.append(
                {
                    "rank": rank,
                    "agent": agent,
                    "prob": prob,
                    "failed": failed,
                    "is_failed": bool(payload.get("is_failed")),
                }
            )
            final = {
                "agent": agent,
                "executed_agent": agent,
                "cascade_rank": rank,
                "model": model,
                "response": payload,
            }
            if not failed:
                break
        except Exception as exc:
            attempts.append(
                {"rank": rank, "agent": agent, "prob": prob, "failed": True, "error": str(exc)}
            )
            final = {
                "agent": agent,
                "executed_agent": agent,
                "cascade_rank": rank,
                "model": model,
                "response": {"is_failed": True, "error": str(exc), "predicted_answer": ""},
            }

    if final is None:
        final = {
            "agent": "",
            "executed_agent": "",
            "cascade_rank": 0,
            "model": model,
            "response": {"is_failed": True, "predicted_answer": ""},
        }

    final["top3_agents"] = ranked
    final["cascade_attempts"] = attempts
    final["router_pred"] = str(dict(row).get("router_pred") or dict(row).get("assigned_agent") or "")
    return final

class RuntimeRouter:
    """Per-query router + optional agent execution."""

    def __init__(
        self,
        query: str,
        *,
        feature_row: pd.Series | dict[str, Any],
        dataset: str,
        model: str = DEFAULT_AGENT_MODEL,
        **run_kw: Any,
    ) -> None:
        from agent.registry import run_strategy

        self._run_strategy = run_strategy
        self.query = query.strip()
        self.dataset = resolve_dataset_name(dataset)
        self.model = model
        self.run_kw = dict(run_kw)
        row = dict(feature_row)
        self._row = row
        self.assigned_agent = str(row.get("assigned_agent") or row.get("router_pred") or "")
        self.overall = row.get("overall") if row.get("overall") is not None else overall_complexity(row)

    @classmethod
    def from_dataframe_row(
        cls, row: pd.Series | dict[str, Any], *, model: str = DEFAULT_AGENT_MODEL, **run_kw: Any
    ) -> RuntimeRouter:
        return cls(
            str(row.get("query") or "").strip(),
            feature_row=row,
            dataset=str(row.get("dataset") or "gaia"),
            model=model,
            **run_kw,
        )

    def inspect(self) -> dict[str, Any]:
        row = self._row
        out: dict[str, Any] = {
            "assigned_agent": self.assigned_agent,
            "router_pred": self.assigned_agent,
            "overall": self.overall,
            "model": self.model,
            "max_prob": row.get("max_prob"),
            "margin_top2": row.get("margin_top2"),
            "router_second": row.get("router_second"),
        }
        for key in PROBA_COLS:
            if key in row:
                out[key] = row[key]
        if "p_raw" in row:
            out["top3_agents"] = top_k_agents(pd.Series({k: row[k] for k in PROBA_COLS if k in row}), k=3)
        return out

    def run(self, *, expected_answer: Any = None) -> dict[str, Any]:
        if not self.assigned_agent:
            raise ValueError("No assigned_agent; run batch routing first.")
        response = self._run_strategy(
            self.assigned_agent,
            self.query,
            model=self.model,
            dataset=self.dataset,
            expected_answer=expected_answer,
            **self.run_kw,
        )
        payload = asdict(response) if hasattr(response, "__dataclass_fields__") else response
        return {"agent": self.assigned_agent, "model": self.model, "response": payload}

    def run_cascade(self, *, expected_answer: Any = None, k: int = 3) -> dict[str, Any]:
        """Run top-``k`` agents in order until one does not fail."""
        return run_agent_cascade(
            self._row,
            query=self.query,
            dataset=self.dataset,
            model=self.model,
            k=k,
            expected_answer=expected_answer,
            **self.run_kw,
        )
