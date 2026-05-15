"""
browser_operator.py - Autonomous browser operation through Playwright.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from core.paths import user_data_dir
from event_bus import emit
from safety import get_safety_manager

BROWSER_DIR = user_data_dir() / "browser_operator"


class BrowserOperator:
    def __init__(self):
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None

    def start(self, headless: bool = False) -> Dict[str, Any]:
        try:
            from playwright.sync_api import sync_playwright
            if self._playwright is None:
                self._playwright = sync_playwright().start()
            if self._browser is None:
                self._browser = self._playwright.chromium.launch(headless=headless)
                self._context = self._browser.new_context()
                self._page = self._context.new_page()
            result = {"ok": True, "headless": headless}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        emit("browser_operator_started", result, source="browser_operator")
        return result

    def open_url(self, url: str) -> Dict[str, Any]:
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        decision = get_safety_manager().assess("browser_open_url", {"url": url}, safety_level="medium")
        if not decision.allowed:
            return decision.__dict__
        if self._page is None:
            started = self.start()
            if not started.get("ok"):
                return started
        try:
            self._page.goto(url, wait_until="domcontentloaded", timeout=30000)
            result = self.snapshot()
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        emit("browser_tab_changed", {"url": url, "result": result}, source="browser_operator")
        return result

    def snapshot(self) -> Dict[str, Any]:
        if self._page is None:
            return {"ok": False, "error": "browser not started"}
        BROWSER_DIR.mkdir(parents=True, exist_ok=True)
        path = BROWSER_DIR / f"page_{time.strftime('%Y%m%d_%H%M%S')}.png"
        try:
            self._page.screenshot(path=str(path), full_page=True)
            title = self._page.title()
            url = self._page.url
            text = self._page.locator("body").inner_text(timeout=3000)[:5000]
            result = {"ok": True, "title": title, "url": url, "screenshot": str(path), "text_preview": text}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        emit("browser_context_updated", result, source="browser_operator")
        return result

    def click(self, selector: str) -> Dict[str, Any]:
        decision = get_safety_manager().assess("browser_click", {"selector": selector}, safety_level="medium")
        if not decision.allowed:
            return decision.__dict__
        try:
            self._page.locator(selector).first.click(timeout=10000)
            return self.snapshot()
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def fill(self, selector: str, value: str) -> Dict[str, Any]:
        decision = get_safety_manager().assess("browser_fill", {"selector": selector, "value": value[:100]}, safety_level="medium")
        if not decision.allowed:
            return decision.__dict__
        try:
            self._page.locator(selector).first.fill(value, timeout=10000)
            return {"ok": True, "selector": selector}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def search(self, query: str, engine: str = "duckduckgo") -> Dict[str, Any]:
        url = f"https://duckduckgo.com/?q={query}" if engine == "duckduckgo" else f"https://www.google.com/search?q={query}"
        return self.open_url(url)

    def stop(self) -> None:
        try:
            if self._browser:
                self._browser.close()
            if self._playwright:
                self._playwright.stop()
        finally:
            self._browser = self._context = self._page = self._playwright = None


_browser: Optional[BrowserOperator] = None


def get_browser_operator() -> BrowserOperator:
    global _browser
    if _browser is None:
        _browser = BrowserOperator()
    return _browser
