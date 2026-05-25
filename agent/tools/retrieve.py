"""Retrieve over provided context paragraphs (Hotpot / MuSiQue / closed-book QA)."""

from __future__ import annotations

import json
import re
from typing import Any

from agent.tools.decorator import tool
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_TOP_K = 3
_SNIPPET_CHARS = 1200

_RETRIEVE_SEMANTIC_HINT = (
    "Use a short semantic phrase (entity + attribute), e.g. "
    "'South Korea Olympics' or 'Japan colony Korea' — not paragraph numbers."
)


def _normalize_passages(context: dict[str, Any] | None) -> list[dict[str, str]]:
    if not context:
        return []
    rows: list[dict[str, str]] = []
    if isinstance(context.get("paragraphs"), list):
        for p in context["paragraphs"]:
            if not isinstance(p, dict):
                continue
            text = str(p.get("text", "") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "title": str(p.get("title", "") or "").strip(),
                    "text": text,
                }
            )
    elif isinstance(context.get("documents"), list):
        for d in context["documents"]:
            if not isinstance(d, dict):
                continue
            text = str(d.get("text", "") or "").strip()
            if not text:
                continue
            rows.append(
                {
                    "title": str(d.get("title", "") or "").strip(),
                    "text": text,
                }
            )
    return rows


def _passage_document(row: dict[str, str]) -> str:
    title = row.get("title", "")
    text = row.get("text", "")
    return f"{title}\n{text}".strip() if title else text


def _rank_passages_semantic(
    query: str,
    passages: list[dict[str, str]],
) -> list[tuple[float, dict[str, str]]]:
    """Embed query + passages (TF-IDF) and rank by cosine similarity."""
    docs = [_passage_document(p) for p in passages]
    if not docs:
        return []
    try:
        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=8000,
        )
        matrix = vectorizer.fit_transform(docs + [query])
        q_vec = matrix[-1:]
        doc_matrix = matrix[:-1]
        scores = cosine_similarity(q_vec, doc_matrix).ravel()
    except ValueError:
        terms = {
            t
            for t in re.sub(r"[^\w\s]", " ", query.lower()).split()
            if len(t) > 1
        }
        scores = [
            float(sum(1 for t in terms if t in d.lower()))
            for d in docs
        ]
    return sorted(
        zip(scores.tolist() if hasattr(scores, "tolist") else list(scores), passages),
        key=lambda x: x[0],
        reverse=True,
    )


def _format_hit(row: dict[str, str], snippet: str) -> dict[str, str]:
    title = row["title"]
    if title:
        header = title
    else:
        header = snippet[:80].replace("\n", " ").strip() or "Passage"
    return {
        "title": title,
        "snippet": snippet,
        "header": header,
    }


@tool(
    "retrieve",
    (
        "Semantic search over the provided Context paragraphs. "
        "Input: short factual query string (entity + attribute) only. "
        "Returns top-ranked passage snippets — use only retrieved text as evidence."
    ),
)
def retrieve(tool_input: str, context: dict[str, Any] | None = None) -> str:
    """Rank passages by embedded query similarity (not the open web)."""
    query = (tool_input or "").strip()
    passages = _normalize_passages(context)

    if not passages:
        return json.dumps(
            {
                "tool": "retrieve",
                "ok": False,
                "error": "NO_CONTEXT",
                "message": "No paragraphs in Context. Pass context={'paragraphs': [...]} to the agent.",
            },
            ensure_ascii=False,
        )

    if not query:
        return json.dumps(
            {
                "tool": "retrieve",
                "ok": False,
                "error": "EMPTY_QUERY",
                "message": f"Provide a short semantic search query. {_RETRIEVE_SEMANTIC_HINT}",
            },
            ensure_ascii=False,
        )

    if query.strip().isdigit():
        return json.dumps(
            {
                "tool": "retrieve",
                "ok": False,
                "error": "INDEX_QUERY_NOT_ALLOWED",
                "message": (
                    "Numeric paragraph ids are not allowed. "
                    f"{_RETRIEVE_SEMANTIC_HINT}"
                ),
            },
            ensure_ascii=False,
        )

    ranked = _rank_passages_semantic(query, passages)
    hits: list[dict[str, str]] = []
    for score, row in ranked:
        if score <= 0.0 and hits:
            continue
        snippet = row["text"][:_SNIPPET_CHARS]
        hit = _format_hit(row, snippet)
        hit["score"] = round(float(score), 4)
        hits.append(hit)
        if len(hits) >= _TOP_K:
            break

    if not hits and ranked:
        row = ranked[0][1]
        snippet = row["text"][:_SNIPPET_CHARS]
        hit = _format_hit(row, snippet)
        hit["score"] = round(float(ranked[0][0]), 4)
        hits = [hit]

    return json.dumps(
        {
            "tool": "retrieve",
            "ok": True,
            "mode": "semantic",
            "query": query,
            "results": hits,
        },
        ensure_ascii=False,
    )

# Mark for ReAct runner — tool reads runtime context.
retrieve._uses_context = True  # type: ignore[attr-defined]
