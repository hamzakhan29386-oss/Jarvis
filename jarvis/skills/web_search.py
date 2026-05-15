"""Live web search skill for JARVIS.

Uses DuckDuckGo search to gather fresh context before an LLM response.
"""

from __future__ import annotations
try:
    from duckduckgo_search import AsyncDDGS
except ImportError:  # pragma: no cover - defensive fallback for newer package builds
    AsyncDDGS = None


async def search_web(query: str, max_results: int = 5) -> str:
    """Search the live web and return clean, citation-friendly text."""
    cleaned_query = " ".join((query or "").split())
    if not cleaned_query:
        return "No web search query was provided."

    result_limit = max(3, min(max_results, 5))

    if AsyncDDGS is None:
        return (
            "Live web search is unavailable because duckduckgo-search with "
            "AsyncDDGS is not installed."
        )

    last_error = None
    raw_results = []
    for backend in ("api", "html", "lite"):
        try:
            async with AsyncDDGS(timeout=10) as ddgs:
                raw_results = ddgs.text(
                    cleaned_query,
                    region="wt-wt",
                    safesearch="moderate",
                    backend=backend,
                    max_results=result_limit,
                ) or []
            if raw_results:
                break
        except Exception as exc:
            last_error = exc

    if not raw_results and last_error is not None:
        return f"Live web search failed: {last_error}"

    formatted_results = []
    for index, item in enumerate(raw_results[:result_limit], start=1):
        title = (item.get("title") or "Untitled result").strip()
        snippet = (item.get("body") or item.get("snippet") or "").strip()
        url = (item.get("href") or item.get("url") or "").strip()

        if not title and not snippet and not url:
            continue

        formatted_results.append(
            f"{index}. {title}\n"
            f"   Snippet: {snippet or 'No snippet available.'}\n"
            f"   URL: {url or 'No URL available.'}"
        )

    if not formatted_results:
        return f"No live web results found for: {cleaned_query}"

    return (
        f"Live web search results for: {cleaned_query}\n\n"
        + "\n\n".join(formatted_results)
    )
