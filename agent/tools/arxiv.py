from __future__ import annotations

from agent.tools.decorator import tool


def _format_arxiv_results(results: list[dict], query: str) -> str:
    if not results:
        return f"No arXiv results found for '{query}'."

    blocks: list[str] = []
    for i, item in enumerate(results, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{i}] {item['title']}",
                    f"arXiv ID : {item['id']}",
                    f"URL      : {item['url']}",
                    f"Published: {item['published']}",
                    f"Authors  : {item['authors']}",
                    f"Summary  : {item['summary']}",
                ]
            )
        )
    return "\n\n".join(blocks)


@tool(
    "arxiv_search",
    (
        "Search arXiv papers. Input: short query string, arXiv id, or topic "
        "(e.g. 'attention is all you need', '1608.03637', 'graph neural networks')."
    ),
)
def arxiv_search(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return "Error: empty arXiv query."

    try:
        import arxiv
    except ImportError:
        return "arXiv tool unavailable. Install with: uv pip install arxiv"

    try:
        client = arxiv.Client(page_size=5, delay_seconds=0.5, num_retries=2)
        search = arxiv.Search(query=q, max_results=5, sort_by=arxiv.SortCriterion.Relevance)

        rows: list[dict] = []
        for result in client.results(search):
            authors = ", ".join(a.name for a in result.authors[:6])
            if len(result.authors) > 6:
                authors += ", et al."
            rows.append(
                {
                    "title": " ".join(result.title.split()),
                    "id": result.get_short_id(),
                    "url": result.entry_id,
                    "published": result.published.date().isoformat(),
                    "authors": authors,
                    "summary": " ".join(result.summary.split())[:500],
                }
            )
        return _format_arxiv_results(rows, q)
    except Exception as e:
        return f"arXiv search failed: {e}"
