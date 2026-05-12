"""Windows desktop and application control actions."""

from __future__ import annotations

import subprocess

from actions import action


@action("close_app", "Close an application by process name")
def close_app(name: str) -> str:
    process = name.strip().lower()
    aliases = {
        "chrome": "chrome.exe",
        "edge": "msedge.exe",
        "spotify": "spotify.exe",
        "discord": "discord.exe",
        "notepad": "notepad.exe",
        "calculator": "calculator.exe",
        "vscode": "Code.exe",
        "vs code": "Code.exe",
    }
    exe = aliases.get(process, process if process.endswith(".exe") else f"{process}.exe")
    result = subprocess.run(["taskkill", "/IM", exe, "/F"], capture_output=True, text=True)
    if result.returncode == 0:
        return f"Closed {name}"
    return f"Could not close {name}: {(result.stderr or result.stdout).strip()}"


@action("window_minimize", "Minimize the active window")
def window_minimize() -> str:
    try:
        import pyautogui

        pyautogui.hotkey("win", "down")
        return "Minimized active window"
    except Exception as exc:
        return f"Minimize failed: {exc}"


@action("window_maximize", "Maximize the active window")
def window_maximize() -> str:
    try:
        import pyautogui

        pyautogui.hotkey("win", "up")
        return "Maximized active window"
    except Exception as exc:
        return f"Maximize failed: {exc}"


@action("switch_window", "Switch to the next window")
def switch_window() -> str:
    try:
        import pyautogui

        pyautogui.hotkey("alt", "tab")
        return "Switched window"
    except Exception as exc:
        return f"Window switch failed: {exc}"


@action("focus_app", "Focus an open app window by title")
def focus_app(title: str) -> str:
    try:
        import pygetwindow as gw

        matches = [w for w in gw.getAllWindows() if title.lower() in w.title.lower()]
        if not matches:
            return f"No open window matched: {title}"
        matches[0].activate()
        return f"Focused {matches[0].title}"
    except ImportError:
        return "pygetwindow not installed. Run: pip install pygetwindow"
    except Exception as exc:
        return f"Focus failed: {exc}"

