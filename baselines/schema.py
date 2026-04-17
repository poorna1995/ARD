from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from baselines.constants import (
    validate_dataset,
    validate_modality,
    validate_model,
)


@dataclass
class UnifiedExperimentRecord:
    # Run identi ddty
    query_id: str
    dataset: str
    modality: str
    model: str

    # Grounded evaluation fields
    query: str
    ground_truth: str
    predicted_answer: str
    is_correct: bool

    # Optional traces
    reasoning_trace: Optional[str] = None
    tool_trace: Optional[str] = None

    # Usage and cost
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0

    # Error tracking
    error_type: Optional[str] = None
    error_message: Optional[str] = None

    # Prompt reproducibility
    system_prompt_version: str = "unknown"
    user_prompt_version: str = "unknown"
    prompt_hash: Optional[str] = None

    # Flexible extension point for dataset/modality specific metadata
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_dataset(self.dataset)
        validate_modality(self.modality)
        validate_model(self.model)


@dataclass
class UnifiedRunMetrics:
    # Run identity
    dataset: str
    modality: str
    model: str

    # Prompt reproducibility
    system_prompt_version: str
    user_prompt_version: str
    prompt_hash: Optional[str] = None

    # Size and quality
    total_queries: int = 0
    correct_queries: int = 0
    accuracy: float = 0.0

    # Resource usage
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    avg_tokens_per_query: float = 0.0

    # Cost and latency
    total_cost_usd: float = 0.0
    avg_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p95_latency_s: float = 0.0

    # Reliability
    error_count: int = 0
    error_rate: float = 0.0

    # Optional modality diagnostics
    avg_reasoning_steps: Optional[float] = None
    avg_tool_calls: Optional[float] = None

    # Flexible extension point for additional aggregate metrics
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        validate_dataset(self.dataset)
        validate_modality(self.modality)
        validate_model(self.model)
