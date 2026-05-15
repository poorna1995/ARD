"""Small decorator used by ReAct tool implementations."""

from __future__ import annotations

from typing import Callable


def tool(name: str, description: str):
    """Register a function as a ReAct tool (sets _tool_name / _tool_description)."""

    def decorator(fn: Callable) -> Callable:
        fn._tool_name = name
        fn._tool_description = description
        return fn

    return decorator
