"""ReAct tool implementations (search, fetch, Wikipedia, file reading, math)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agent.tools.decorator import tool
from agent.tools.arxiv import arxiv_search
from agent.tools.github import github_search
from agent.tools.math import math_tool
from agent.tools.pdb import pdb_parse
from agent.tools.search import web_search
from agent.tools.webfetch import web_fetch
from agent.tools.wikipedia import wikipedia_search

if TYPE_CHECKING:
    from collections.abc import Callable

# read_file pulls heavy optional deps (pdfplumber, etc.) — load on demand.
_LAZY_TOOL_IMPORTS: dict[str, tuple[str, str]] = {
    "read_file": ("agent.tools.readfile", "read_file"),
}

__all__ = [
    "tool",
    "arxiv_search",
    "github_search",
    "web_search",
    "web_fetch",
    "wikipedia_search",
    "pdb_parse",
    "read_file",
    "math_tool",
]


def __getattr__(name: str) -> "Callable":
    if name in _LAZY_TOOL_IMPORTS:
        module_path, attr = _LAZY_TOOL_IMPORTS[name]
        import importlib

        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
