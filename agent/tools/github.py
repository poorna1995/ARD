from __future__ import annotations

import os
from typing import Any

from agent.tools.decorator import tool

_GITHUB_API = "https://api.github.com/search"


def _headers() -> dict[str, str]:
    token = os.environ.get("GITHUB_TOKEN", "").strip()
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "research-work-react-agent",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _search(endpoint: str, query: str, per_page: int = 5) -> list[dict[str, Any]]:
    try:
        import requests
    except ImportError as e:
        raise RuntimeError("Missing dependency `requests`. Install with: uv pip install requests") from e

    url = f"{_GITHUB_API}/{endpoint}"
    resp = requests.get(
        url,
        headers=_headers(),
        params={"q": query, "per_page": per_page},
        timeout=25,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"{resp.status_code} {resp.text[:220]}")
    payload = resp.json()
    return payload.get("items", [])


@tool(
    "github_search",
    (
        "Search GitHub. Input format: '<type>: <query>' where type is one of "
        "repos, issues, prs. Example: 'repos: langchain react agent', "
        "'issues: repo:openai/openai-python timeout', "
        "'prs: repo:pytorch/pytorch quantization'."
    ),
)
def github_search(user_input: str) -> str:
    raw = (user_input or "").strip()
    if not raw:
        return "Error: empty GitHub search input."

    if ":" in raw:
        mode, query = raw.split(":", 1)
        mode = mode.strip().lower()
        query = query.strip()
    else:
        mode, query = "repos", raw

    if not query:
        return "Error: empty GitHub query."

    try:
        if mode in {"repos", "repo", "repositories"}:
            items = _search("repositories", query)
            if not items:
                return f"No GitHub repositories found for '{query}'."
            lines = []
            for i, item in enumerate(items, start=1):
                lines.append(
                    "\n".join(
                        [
                            f"[{i}] {item.get('full_name', 'unknown')}",
                            f"URL   : {item.get('html_url', '')}",
                            f"Stars : {item.get('stargazers_count', 0)}",
                            f"Desc  : {(item.get('description') or '').strip()}",
                        ]
                    )
                )
            return "\n\n".join(lines)

        if mode in {"issues", "issue"}:
            items = _search("issues", f"{query} is:issue")
            if not items:
                return f"No GitHub issues found for '{query}'."
            lines = []
            for i, item in enumerate(items, start=1):
                repo = item.get("repository_url", "").replace("https://api.github.com/repos/", "")
                lines.append(
                    "\n".join(
                        [
                            f"[{i}] {item.get('title', '')}",
                            f"Repo  : {repo}",
                            f"URL   : {item.get('html_url', '')}",
                            f"State : {item.get('state', 'unknown')}",
                        ]
                    )
                )
            return "\n\n".join(lines)

        if mode in {"prs", "pr", "pulls", "pull_requests"}:
            items = _search("issues", f"{query} is:pr")
            if not items:
                return f"No GitHub pull requests found for '{query}'."
            lines = []
            for i, item in enumerate(items, start=1):
                repo = item.get("repository_url", "").replace("https://api.github.com/repos/", "")
                lines.append(
                    "\n".join(
                        [
                            f"[{i}] {item.get('title', '')}",
                            f"Repo  : {repo}",
                            f"URL   : {item.get('html_url', '')}",
                            f"State : {item.get('state', 'unknown')}",
                        ]
                    )
                )
            return "\n\n".join(lines)

        return "Invalid mode. Use one of: repos, issues, prs."
    except Exception as e:
        return (
            "GitHub search failed: "
            f"{e}. Tip: set GITHUB_TOKEN for higher rate limits."
        )
