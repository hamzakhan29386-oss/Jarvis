"""
Structured command intent parsing for JARVIS.

This is intentionally lightweight and deterministic. Brain-level model routing
still lives in brain.py; this parser catches direct desktop commands first so
JARVIS can act quickly without spending a model call.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class ParsedIntent:
    action: str
    args: dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    source_text: str = ""
    needs_ai: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_APP_OPEN = re.compile(r"^(?:open|launch|start)\s+(?P<name>.+)$", re.I)
_APP_CLOSE = re.compile(r"^(?:close|quit|kill)\s+(?P<name>.+)$", re.I)
_PLAY_YOUTUBE = re.compile(
    r"^(?:play|put on)\s+(?P<query>.+?)(?:\s+(?:on|in)\s+youtube)?$",
    re.I,
)
_SEARCH_YOUTUBE = re.compile(r"^(?:search)\s+(?P<query>.+?)\s+(?:on|in)\s+youtube$", re.I)
_SEARCH_WEB = re.compile(r"^(?:search|google|look up|find)\s+(?P<query>.+)$", re.I)
_OPEN_SITE = re.compile(r"^(?:go to|open website|open site|visit)\s+(?P<url>.+)$", re.I)
_TYPE_TEXT = re.compile(r"^(?:type|write)\s+(?P<text>.+)$", re.I)
_VOLUME = re.compile(r"^(?:set\s+)?volume\s+(?P<level>\d{1,3})(?:\s*%)?$", re.I)
_BRIGHTNESS = re.compile(r"^(?:set\s+)?brightness\s+(?P<level>\d{1,3})(?:\s*%)?$", re.I)
_LIVE_WEB_HINTS = (
    "today", "today's", "latest", "current", "currently", "recent",
    "breaking", "news", "headline", "headlines", "this week", "this month",
    "right now", "live", "update", "updates", "post-2023", "after 2023",
    "2024", "2025", "2026",
)
_FUTURE_YEAR = re.compile(r"\b20(2[4-9]|[3-9]\d)\b")

_WORKFLOWS = {
    "coding setup": "coding_setup",
    "code setup": "coding_setup",
    "study mode": "study_mode",
    "focus mode": "study_mode",
    "entertainment mode": "entertainment_mode",
}


def _looks_like_live_web_query(text: str) -> bool:
    lower = (text or "").lower()
    return _FUTURE_YEAR.search(lower) is not None or any(hint in lower for hint in _LIVE_WEB_HINTS)


def parse_user_intent(text: str) -> ParsedIntent:
    cleaned = " ".join((text or "").strip().split())
    lower = cleaned.lower()
    if not cleaned:
        return ParsedIntent("none", source_text=text, confidence=1.0)

    if lower in {"screenshot", "take screenshot", "capture screen"}:
        return ParsedIntent("take_screenshot", source_text=cleaned, confidence=0.95)

    if lower in {"system status", "status", "pc status", "computer status"}:
        return ParsedIntent("get_system_status", source_text=cleaned, confidence=0.9)

    if lower in {"mute", "mute volume", "unmute", "volume mute"}:
        return ParsedIntent("control_media", {"action_name": "mute"}, 0.9, cleaned)

    if lower in {"pause", "play pause", "resume", "next", "previous", "prev"}:
        media = {"resume": "play", "previous": "prev"}.get(lower, lower)
        return ParsedIntent("control_media", {"action_name": media}, 0.9, cleaned)

    if lower in {"minimize window", "minimize active window"}:
        return ParsedIntent("window_minimize", source_text=cleaned, confidence=0.9)

    if lower in {"maximize window", "maximize active window"}:
        return ParsedIntent("window_maximize", source_text=cleaned, confidence=0.9)

    for phrase, workflow in _WORKFLOWS.items():
        if phrase in lower and lower.startswith(("open", "start", "run", "activate")):
            return ParsedIntent("run_workflow", {"name": workflow}, 0.9, cleaned)

    match = _SEARCH_YOUTUBE.match(cleaned)
    if match:
        return ParsedIntent("youtube_search", {"query": match.group("query")}, 0.95, cleaned)

    match = _PLAY_YOUTUBE.match(cleaned)
    if match and ("youtube" in lower or lower.startswith(("play ", "put on "))):
        return ParsedIntent("youtube_play", {"query": match.group("query")}, 0.88, cleaned)

    match = _OPEN_SITE.match(cleaned)
    if match:
        return ParsedIntent("browser_open", {"url": match.group("url")}, 0.9, cleaned)

    match = _SEARCH_WEB.match(cleaned)
    if match and "youtube" not in lower:
        query = match.group("query")
        if _looks_like_live_web_query(query):
            return ParsedIntent(
                "chat",
                {"message": cleaned, "web_search": True, "query": query},
                0.82,
                cleaned,
                needs_ai=True,
            )
        return ParsedIntent("browser_google_search", {"query": query}, 0.85, cleaned)

    match = _APP_OPEN.match(cleaned)
    if match:
        target = match.group("name")
        if "." in target or target.lower() in {"youtube", "google", "github"}:
            url = {
                "youtube": "youtube.com",
                "google": "google.com",
                "github": "github.com",
            }.get(target.lower(), target)
            return ParsedIntent("browser_open", {"url": url}, 0.85, cleaned)
        return ParsedIntent("open_app", {"name": target}, 0.8, cleaned)

    match = _APP_CLOSE.match(cleaned)
    if match:
        return ParsedIntent("close_app", {"name": match.group("name")}, 0.85, cleaned)

    match = _TYPE_TEXT.match(cleaned)
    if match:
        return ParsedIntent("type_text", {"text": match.group("text")}, 0.8, cleaned)

    match = _VOLUME.match(cleaned)
    if match:
        return ParsedIntent("set_volume", {"level": int(match.group("level"))}, 0.9, cleaned)

    match = _BRIGHTNESS.match(cleaned)
    if match:
        return ParsedIntent("set_brightness", {"level": int(match.group("level"))}, 0.9, cleaned)

    return ParsedIntent("chat", {"message": cleaned}, 0.35, cleaned, needs_ai=True)
