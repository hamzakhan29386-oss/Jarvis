"""YouTube-focused automation actions."""

from __future__ import annotations

import urllib.parse
import webbrowser

from actions import action


def _youtube_search_url(query: str) -> str:
    return "https://www.youtube.com/results?search_query=" + urllib.parse.quote_plus(query)


@action("youtube_search", "Search YouTube for videos")
def youtube_search(query: str) -> str:
    url = _youtube_search_url(query)
    webbrowser.open(url)
    return f"Opened YouTube search for: {query}"


@action("youtube_play", "Search YouTube and play the first video")
def youtube_play(query: str) -> str:
    url = _youtube_search_url(query)
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        webbrowser.open(url)
        return f"Playwright is not installed; opened YouTube search for: {query}"

    video_url = None
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        href = page.locator("a#video-title").first.get_attribute("href", timeout=10000)
        if href:
            video_url = "https://www.youtube.com" + href
        browser.close()
    webbrowser.open(video_url or url)
    return f"Playing on YouTube: {query}"
