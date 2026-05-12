"""Path helpers for source and PyInstaller-frozen JARVIS runtimes."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def app_root() -> Path:
    """Return the source root or PyInstaller extraction root."""
    if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return Path(__file__).resolve().parents[1]


def resource_path(*parts: str) -> Path:
    return app_root().joinpath(*parts)


def user_data_dir() -> Path:
    """Return a writable directory for generated runtime files."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
    if base:
        path = Path(base) / "JARVIS"
    else:
        path = Path.home() / ".jarvis"
    path.mkdir(parents=True, exist_ok=True)
    return path

