"""
computer_use.py - Multimodal desktop operator primitives.

The module is conservative by default: high-impact operations return
confirmation-required decisions through the safety layer before moving or
typing. Screenshot and analysis functions work even when OCR is unavailable.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.paths import user_data_dir
from event_bus import emit
from safety import get_safety_manager

SCREEN_DIR = user_data_dir() / "vision" / "screenshots"


class ComputerUse:
    def capture_screenshot(self, label: str = "desktop") -> Dict[str, Any]:
        SCREEN_DIR.mkdir(parents=True, exist_ok=True)
        path = SCREEN_DIR / f"{label}_{time.strftime('%Y%m%d_%H%M%S')}.png"
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            img.save(path)
            result = {"ok": True, "path": str(path), "size": img.size}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        emit("screenshot_captured", result, source="computer_use")
        return result

    def ocr_image(self, image_path: str) -> Dict[str, Any]:
        try:
            from PIL import Image
            import pytesseract
            text = pytesseract.image_to_string(Image.open(image_path))
            return {"ok": True, "text": text}
        except Exception as exc:
            return {"ok": False, "text": "", "error": "OCR unavailable or failed: " + str(exc)}

    def analyze_screen(self) -> Dict[str, Any]:
        shot = self.capture_screenshot("analysis")
        text = self.ocr_image(shot["path"]) if shot.get("ok") else {"ok": False, "text": ""}
        windows = self.detect_windows()
        result = {"screenshot": shot, "ocr": text, "windows": windows, "timestamp": time.time()}
        emit("screen_state_analyzed", result, source="computer_use")
        return result

    def detect_windows(self) -> List[Dict[str, Any]]:
        try:
            import pygetwindow as gw
            return [
                {"title": w.title, "left": w.left, "top": w.top, "width": w.width, "height": w.height}
                for w in gw.getAllWindows() if getattr(w, "title", "")
            ]
        except Exception:
            return []

    def move_and_click(self, x: int, y: int, *, verify: bool = True) -> Dict[str, Any]:
        decision = get_safety_manager().assess("desktop_click", {"x": x, "y": y}, safety_level="medium")
        if not decision.allowed:
            return decision.__dict__
        before = self.capture_screenshot("before_click") if verify else None
        try:
            import pyautogui
            pyautogui.moveTo(x, y, duration=0.15)
            pyautogui.click()
            after = self.capture_screenshot("after_click") if verify else None
            result = {"ok": True, "before": before, "after": after}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        emit("desktop_click_completed", result, source="computer_use")
        return result

    def type_text(self, text: str, interval: float = 0.01) -> Dict[str, Any]:
        decision = get_safety_manager().assess("desktop_type_text", {"text": text[:200]}, safety_level="high")
        if not decision.allowed:
            return decision.__dict__
        try:
            import pyautogui
            pyautogui.write(text, interval=interval)
            result = {"ok": True, "chars": len(text)}
        except Exception as exc:
            result = {"ok": False, "error": str(exc)}
        emit("desktop_text_typed", result, source="computer_use")
        return result

    def press_keys(self, keys: List[str]) -> Dict[str, Any]:
        decision = get_safety_manager().assess("desktop_press_keys", {"keys": keys}, safety_level="medium")
        if not decision.allowed:
            return decision.__dict__
        try:
            import pyautogui
            if len(keys) > 1:
                pyautogui.hotkey(*keys)
            else:
                pyautogui.press(keys[0])
            return {"ok": True, "keys": keys}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    def locate_text(self, text: str) -> Optional[Tuple[int, int]]:
        # Placeholder for OCR bounding-box grounding; keeps API stable.
        analysis = self.analyze_screen()
        if text.lower() in analysis.get("ocr", {}).get("text", "").lower():
            size = analysis.get("screenshot", {}).get("size") or (0, 0)
            return (int(size[0] / 2), int(size[1] / 2))
        return None


_computer: Optional[ComputerUse] = None


def get_computer_use() -> ComputerUse:
    global _computer
    if _computer is None:
        _computer = ComputerUse()
    return _computer
