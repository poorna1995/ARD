"""Tool names, aliases, and grouped tool sets."""

from __future__ import annotations

from dataclasses import dataclass

_NONE = frozenset({"none"})
_WEB = frozenset(
    {"web_search", "wikipedia", "wikipedia_search", "arxiv_search", "retrieve_tool"}
)
_CODE = frozenset({"python_exec", "math_tool", "math", "code"})
_FILE = frozenset({"read_file", "pdb_parse", "github_search"})
_PLANNER_CORE = frozenset({"math_tool", "python_exec"})
_PLAN_WEB = _WEB - frozenset({"wikipedia_search"})
_PROMPT_WEB = (_WEB - frozenset({"wikipedia", "retrieve_tool"})) | frozenset({"wikipedia_search"})

_TOOL_ALIASES: dict[str, str] = {
    "math": "math_tool",
    "wikipedia_search": "wikipedia",
    "python": "python_exec",
    "code": "python_exec",
    "retrieve": "retrieve_tool",
}


@dataclass(frozen=True)
class Tools:
    web: frozenset[str]
    code: frozenset[str]
    file: frozenset[str]
    aliases: dict[str, str]
    open_web: frozenset[str]
    retrieve: frozenset[str]
    web_lookup: frozenset[str]
    react_repeat: frozenset[str]
    plan: frozenset[str]
    prompt: frozenset[str]
    exec: frozenset[str]
    retr: frozenset[str]
    planner_names: tuple[str, ...]

    @property
    def retrieve_chain(self) -> frozenset[str]:
        return self.retrieve | self.web_lookup


def _make_tools() -> Tools:
    plan = _NONE | _PLAN_WEB | _PLANNER_CORE | _FILE
    return Tools(
        web=_WEB,
        code=_CODE,
        file=_FILE,
        aliases=dict(_TOOL_ALIASES),
        open_web=frozenset({"web_search", "web_fetch", "wikipedia_search"}),
        retrieve=frozenset({"retrieve"}),
        web_lookup=frozenset({"wikipedia_search", "wikipedia", "web_search", "web_fetch"}),
        react_repeat=frozenset({"python_exec", "web_fetch", "read_file", "pdb_parse"}),
        plan=plan,
        prompt=_NONE | _PROMPT_WEB | _PLANNER_CORE | _FILE,
        exec=(_FILE - frozenset({"github_search"})) | _CODE,
        retr=_WEB | frozenset({"retrieve"}),
        planner_names=tuple(sorted(plan - _NONE)),
    )


TOOLS = _make_tools()

__all__ = ["TOOLS", "Tools"]
