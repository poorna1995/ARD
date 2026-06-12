"""Live eval scoring, benchmarks, and oracle bounds."""

from eval.score import (
    DEFAULT_ORACLE,
    DEFAULT_ORACLE_PATH,
    StrategyMetrics,
    attach_outcomes,
    baseline_strategies,
    build_oracle_frame,
    from_live_baselines_for_ids,
    from_oracle_csv,
    oracle_outcome_matrix,
    score_routed,
    score_routed_eval,
    score_routed_split,
    write_score_summary,
)

__all__ = [
    "DEFAULT_ORACLE",
    "DEFAULT_ORACLE_PATH",
    "StrategyMetrics",
    "attach_outcomes",
    "baseline_strategies",
    "build_oracle_frame",
    "from_live_baselines_for_ids",
    "from_oracle_csv",
    "oracle_outcome_matrix",
    "score_routed",
    "score_routed_eval",
    "score_routed_split",
    "write_score_summary",
]
