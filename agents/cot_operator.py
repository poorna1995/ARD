from __future__ import annotations

"""
agents/cot_operator.py
=======================
Dedicated baseline operators for chain-of-thought (single completion).

- ``zero_shot_cot_operator``: no in-prompt exemplars (zero-shot); instructions
  come from resolved ``zero_shot_cot`` templates.
- ``few_shot_cot_operator``: same call shape; **fixed few-shot exemplars** are
  embedded in the resolved ``few_shot_cot`` system prompts in
  ``baselines/prompt_templates.py`` (not loaded from the dataset).
"""

from typing import Any, Optional, Tuple

from agents.vanilla_operator import vanilla_operator


def zero_shot_cot_operator(
    client: Any,
    model_name: str,
    question: str,
    system_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[str, dict, Optional[str], float]:
    """Run zero-shot CoT: one system + one user message, no tool loop."""
    return vanilla_operator(
        client=client,
        model_name=model_name,
        question=question,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,

    )


def few_shot_cot_operator(
    client: Any,
    model_name: str,
    question: str,
    system_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[str, dict, Optional[str], float]:
    """Run few-shot CoT: exemplars are embedded in prompts; still one completion."""
    return vanilla_operator(
        client=client,
        model_name=model_name,
        question=question,
        system_prompt=system_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        seed=seed,
    )
