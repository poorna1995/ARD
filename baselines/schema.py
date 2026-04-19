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
    # Run identity
    query_id: str
    dataset: str
    modality: str
    model: str

    # Grounded evaluation fields
    query: str
    ground_truth: str
    predicted_answer: str
    is_correct: bool
    # Verbatim model completion from the API (before extraction / scoring heuristics).
    raw_model_output: Optional[str] = None

    # Optional traces
    reasoning_trace: Optional[str] = None
    tool_trace: Optional[str] = None
    # For CoT-style modalities: reasoning lines / markers in raw output.
    # For ReAct: non-finish trace steps (tool Thought/Action cycles before finish).
    reasoning_steps: Optional[int] = None
    # ReAct only: total entries in tool_trace (includes finish row when present).
    react_trace_length: Optional[int] = None

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
class ReActStep:
    """
    One Thought/Action/Observation cycle.
    Stored as a plain dataclass — cheaper than dict for large traces.
    Serialised to dict only once at the end for JSON storage.
    """
    step:         int
    thought:      str
    action:       str        # tool name  (e.g. "calculator")
    action_input: str        # raw string passed to the tool
    observation:  str        # tool return value
    error:        bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "step":         self.step,
            "thought":      self.thought,
            "action":       self.action,
            "action_input": self.action_input,
            "observation":  self.observation,
            "error":        self.error,
        }




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
