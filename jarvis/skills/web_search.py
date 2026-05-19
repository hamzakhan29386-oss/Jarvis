"""
Live web search skill for JARVIS.

Uses Tavily API for deep, agentic web search with raw content extraction.
Replaces the previous DuckDuckGo implementation which was rate-limited (HTTP 202).

The Tavily client is synchronous, so we wrap it in asyncio.to_thread()
to prevent blocking the async event bus.
"""

from __future__ import annotations

import asyncio
import logging
import os

log = logging.getLogger("jarvis.skills.web_search")

# ── Tavily client (lazy-initialised singleton) ───────────────────────────────

_tavily_client = None


def _get_tavily_client():
    """Lazy-init the Tavily client. Returns None if unavailable."""
    global _tavily_client
    if _tavily_client is not None:
        return _tavily_client

    api_key = os.getenv("TAVILY_API_KEY", "").strip()
    if not api_key:
        log.warning("[WebSearch] TAVILY_API_KEY not set in environment / .env")
        return None

    try:
        from tavily import TavilyClient
        _tavily_client = TavilyClient(api_key=api_key)
        log.info("[WebSearch] Tavily client initialised (advanced search mode)")
        return _tavily_client
    except ImportError:
        log.error(
            "[WebSearch] tavily-python not installed. "
            "Run: pip install tavily-python"
        )
        return None
    except Exception as exc:
        log.error("[WebSearch] Tavily client init failed: %s", exc)
        return None


# ── Synchronous search (runs inside asyncio.to_thread) ───────────────────────

def _tavily_search_sync(query: str, max_results: int = 5) -> dict | None:
    """
    Execute the Tavily search synchronously.
    This function is designed to be called via asyncio.to_thread().
    """
    client = _get_tavily_client()
    if client is None:
        return None

    return client.search(
        query=query,
        search_depth="advanced",
        include_raw_content=True,
        max_results=max_results,
    )


# ── Public async interface ───────────────────────────────────────────────────

async def search_web(query: str, max_results: int = 5) -> str:
    """
    Search the live web via Tavily and return formatted, citation-friendly text.

    This is the primary entry point consumed by brain.py and the async event bus.
    The synchronous Tavily client call is offloaded to a thread so it never
    blocks the event loop.

    Returns:
        A formatted string of search results, or a failure sentinel string
        that brain.py can detect and handle gracefully.
    """
    cleaned_query = " ".join((query or "").split())
    if not cleaned_query:
        return "No web search query was provided."

    result_limit = max(3, min(max_results, 10))

    # ── Execute search in a thread (non-blocking) ────────────────────────────
    try:
        raw_response = await asyncio.to_thread(
            _tavily_search_sync, cleaned_query, result_limit
        )
    except Exception as exc:
        log.error("[WebSearch] Tavily search failed: %s", exc)
        return "[SEARCH_FAILED: External uplink is currently offline.]"

    if raw_response is None:
        return "[SEARCH_FAILED: External uplink is currently offline.]"

    # ── Extract and format results ───────────────────────────────────────────
    results = raw_response.get("results", [])
    if not results:
        return f"No live web results found for: {cleaned_query}"

    formatted_results = []
    for index, item in enumerate(results[:result_limit], start=1):
        title = (item.get("title") or "Untitled").strip()
        snippet = (item.get("content") or "").strip()
        url = (item.get("url") or "").strip()
        raw_content = (item.get("raw_content") or "").strip()

        # Prefer raw_content for richer context, truncate to avoid token bloat
        body = raw_content[:1500] if raw_content else snippet
        if not body:
            body = "No content available."

        formatted_results.append(
            f"{index}. {title}\n"
            f"   Content: {body}\n"
            f"   Source: {url}"
        )

    if not formatted_results:
        return f"No live web results found for: {cleaned_query}"

    # Include the Tavily answer summary if available
    answer = (raw_response.get("answer") or "").strip()
    header = f"Live web intelligence for: {cleaned_query}"
    if answer:
        header += f"\n\nQuick Answer: {answer}"

    return header + "\n\n" + "\n\n".join(formatted_results)
