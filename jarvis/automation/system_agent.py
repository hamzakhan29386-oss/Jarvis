"""System-control actions for JARVIS."""

from __future__ import annotations

import os
import subprocess

from actions import action


@action("set_volume", "Set system volume approximately")
def set_volume(level: int) -> str:
    level = max(0, min(100, int(level)))
    try:
        import pyautogui

        pyautogui.press("volumemute")
        pyautogui.press("volumemute")
        pyautogui.press("volumedown", presses=50)
        pyautogui.press("volumeup", presses=max(0, level // 2))
        return f"Volume set near {level}%"
    except Exception as exc:
        return f"Volume control failed: {exc}"


@action("set_brightness", "Set screen brightness")
def set_brightness(level: int) -> str:
    level = max(0, min(100, int(level)))
    try:
        import screen_brightness_control as sbc

        sbc.set_brightness(level)
        return f"Brightness set to {level}%"
    except ImportError:
        return "screen-brightness-control not installed. Run: pip install screen-brightness-control"
    except Exception as exc:
        return f"Brightness control failed: {exc}"


@action("shutdown_pc", "Shut down Windows")
def shutdown_pc(confirm: bool = False) -> str:
    if not confirm:
        return "Shutdown requires confirm=true."
    subprocess.Popen(["shutdown", "/s", "/t", "5"])
    return "Shutdown scheduled."


@action("restart_pc", "Restart Windows")
def restart_pc(confirm: bool = False) -> str:
    if not confirm:
        return "Restart requires confirm=true."
    subprocess.Popen(["shutdown", "/r", "/t", "5"])
    return "Restart scheduled."


@action("open_folder", "Open a folder in File Explorer")
def open_folder(path: str) -> str:
    if not os.path.exists(path):
        return f"Folder not found: {path}"
    os.startfile(path)
    return f"Opened folder: {path}"

