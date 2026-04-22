from __future__ import annotations
import time
from typing import Optional, Tuple
from openai import OpenAI



def vanilla_operator(
    client: OpenAI,
    model_name: str,
    question: str,
    system_prompt: str,
    max_tokens: int = 1024,
    temperature: float = 0.0,
    seed: Optional[int] = None,
) -> Tuple[str, dict, Optional[str], float]:
    """Call an OpenAI-compatible chat model and return response details.

    Returns
    -------
    predicted : str
        Model text output (empty on error).
    usage : dict
        prompt_tokens / completion_tokens / total_tokens
    error : Optional[str]
        Error message, if the call fails.
    elapsed : float
        Time taken in seconds.
    """
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    predicted = ""
    error: Optional[str] = None

    start = time.perf_counter()
    try:
        create_kwargs = {
            "model": model_name,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": question},
            ],
        }
        if seed is not None:
            create_kwargs["seed"] = seed
        response = client.chat.completions.create(**create_kwargs)
        predicted = (response.choices[0].message.content or "").strip()
        
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        }
    except Exception as exc:
        error = str(exc)

    elapsed = time.perf_counter() - start
    return predicted, usage, error, elapsed
