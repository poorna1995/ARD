from agent.tools.decorator import tool


def _wikipedia_with_wikipediaapi(query: str) -> str | None:
    """Use Wikipedia-API package if installed; return None to fall back."""
    try:
        import wikipediaapi
    except ImportError:
        return None
    try:
        wiki = wikipediaapi.Wikipedia(
            language="en",
            user_agent="ReActAgent/1.0 (research-bot)",
        )
        page = wiki.page(query)
        if not page.exists():
            return None
        summary = page.summary
        if len(summary) > 1500:
            summary = summary[:1500].rsplit(" ", 1)[0] + " …"
        return f"Wikipedia – {page.title}\nURL: {page.fullurl}\n\n{summary}"
    except Exception:
        return None


def _wikipedia_fallback(query: str) -> str:
    try:
        import wikipedia

        wikipedia.set_lang("en")
        try:
            summary = wikipedia.summary(query, sentences=5, auto_suggest=True)
            page = wikipedia.page(query, auto_suggest=True)
            return f"Wikipedia – {page.title}\nURL: {page.url}\n\n{summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            options = ", ".join(e.options[:6])
            return (
                f"Disambiguation: '{query}' may refer to multiple topics. "
                f"Try being more specific. Suggestions: {options}"
            )
        except wikipedia.exceptions.PageError:
            return f"No Wikipedia page found for '{query}'."
    except ImportError:
        return (
            "Wikipedia tool unavailable. "
            "Install: pip install wikipedia-api  OR  pip install wikipedia"
        )


@tool(
    "wikipedia",
    (
        "Look up a topic on Wikipedia and return a concise summary. "
        "Input: the topic or entity name "
        "(e.g. 'Albert Einstein', 'Python programming language')."
    ),
)
def wikipedia_search(query: str) -> str:
    primary = _wikipedia_with_wikipediaapi(query)
    if primary is not None:
        return primary
    return _wikipedia_fallback(query)
