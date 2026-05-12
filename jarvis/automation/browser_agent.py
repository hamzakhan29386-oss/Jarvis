"""Browser automation actions powered by Playwright when available."""

from __future__ import annotations

import urllib.parse
import webbrowser

from actions import action


def _with_playwright(url: str, callback=None) -> str:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        webbrowser.open(url)
        return "Playwright is not installed; opened in the default browser."

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, wait_until="domcontentloaded")
        if callback:
            callback(page)
        title = page.title() or page.url
        browser.close()
        return title


@action("browser_open", "Open a website in an automated browser")
def browser_open(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    webbrowser.open(url)
    return f"Opened browser: {url}"


@action("browser_google_search", "Search Google in a browser")
def browser_google_search(query: str) -> str:
    url = "https://www.google.com/search?q=" + urllib.parse.quote_plus(query)
    webbrowser.open(url)
    return f"Google search opened: {query}"


@action("browser_click_text", "Click a visible text button or link")
def browser_click_text(text: str, url: str = "about:blank") -> str:
    def click(page):
        page.get_by_text(text, exact=False).first.click(timeout=5000)

    title = _with_playwright(url, click)
    return f"Clicked text: {text} ({title})"


@action("browser_type_text", "Type text into the focused browser field")
def browser_type_text(text: str, url: str = "about:blank") -> str:
    def type_on_page(page):
        page.keyboard.type(text, delay=15)

    title = _with_playwright(url, type_on_page)
    return f"Typed in browser: {text[:60]} ({title})"


@action("browser_scrape_text", "Scrape visible page text")
def browser_scrape_text(url: str, max_chars: int = 2000) -> str:
    captured = {"text": ""}

    def scrape(page):
        captured["text"] = page.locator("body").inner_text(timeout=10000)[:max_chars]

    _with_playwright(url, scrape)
    return captured["text"]
