"""Contextual cowork intelligence: active-window, screen, OCR, and visual memory hooks."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from core.paths import user_data_dir
from event_bus import emit


class ContextualCoworkService:
    """Supervised perception service with privacy-preserving defaults."""

    def __init__(self):
        self.capture_dir = user_data_dir() / "vision_context"
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self._last_context: dict[str, Any] = {}
        self._ocr = None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": os.getenv("COWORK_MODE_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            "screen_capture_enabled": os.getenv("SCREEN_CAPTURE_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            "ocr_enabled": os.getenv("OCR_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            "vlm_enabled": os.getenv("VLM_ENABLED", "false").lower() in {"1", "true", "yes", "on"},
            "last_context": self._last_context,
        }

    def active_window(self) -> dict[str, Any]:
        title = ""
        backend = "none"
        try:
            import pygetwindow as gw

            win = gw.getActiveWindow()
            if win:
                title = getattr(win, "title", "") or ""
                backend = "pygetwindow"
        except Exception:
            try:
                from pywinauto import Desktop

                win = Desktop(backend="uia").get_active()
                title = win.window_text() if win else ""
                backend = "pywinauto"
            except Exception:
                pass
        return {"title": title, "backend": backend, "captured_at": time.time()}

    def capture_screen(self, *, save: bool = False) -> dict[str, Any]:
        image = None
        backend = "none"
        try:
            import dxcam

            camera = dxcam.create(output_idx=0)
            image = camera.grab()
            if image is None:
                raise RuntimeError("dxcam returned no frame")
            backend = "dxcam"
        except Exception:
            try:
                from PIL import ImageGrab

                image = ImageGrab.grab()
                backend = "PIL.ImageGrab"
            except Exception as exc:
                return {"ok": False, "error": str(exc), "backend": backend}

        width = height = None
        path = ""
        try:
            if hasattr(image, "size") and not isinstance(image.size, int):
                width, height = image.size
            else:
                height, width = image.shape[:2]
        except Exception:
            pass

        if save:
            path = str(self._save_image(image))

        payload = {
            "ok": True,
            "backend": backend,
            "width": width,
            "height": height,
            "path": path,
            "captured_at": time.time(),
        }
        emit("screen_captured", payload, source="contextual_cowork")
        return payload

    def extract_context(self, *, include_ocr: bool = False, save_screenshot: bool = False) -> dict[str, Any]:
        window = self.active_window()
        capture = self.capture_screen(save=save_screenshot or include_ocr)
        ocr_text = ""
        if include_ocr and capture.get("ok") and capture.get("path"):
            ocr_text = self._ocr_file(capture["path"])
            if not save_screenshot:
                try:
                    Path(capture["path"]).unlink(missing_ok=True)
                    capture["path"] = ""
                except Exception:
                    pass

        context = {
            "active_window": window,
            "screen": capture,
            "ocr_text_preview": ocr_text[:2000],
            "captured_at": time.time(),
        }
        self._last_context = context
        try:
            from world_state import get_world_state

            get_world_state().update_state(
                {
                    "focused_application": window.get("title", ""),
                    "desktop_session_context": context,
                },
                source="contextual_cowork",
            )
        except Exception:
            pass
        emit("context_updated", context, source="contextual_cowork")
        return context

    def _save_image(self, image) -> Path:
        path = self.capture_dir / f"screen-{int(time.time() * 1000)}.png"
        if hasattr(image, "save"):
            image.save(path)
        else:
            from PIL import Image

            Image.fromarray(image).save(path)
        return path

    def _ocr_file(self, path: str) -> str:
        try:
            if self._ocr is None:
                from paddleocr import PaddleOCR

                self._ocr = PaddleOCR(use_angle_cls=True, lang=os.getenv("PADDLEOCR_LANG", "en"))
            result = self._ocr.ocr(path, cls=True)
            lines: list[str] = []
            for page in result or []:
                for item in page or []:
                    if len(item) >= 2 and item[1]:
                        lines.append(str(item[1][0]))
            return "\n".join(lines).strip()
        except Exception:
            try:
                import pytesseract
                from PIL import Image

                return pytesseract.image_to_string(Image.open(path)).strip()
            except Exception:
                return ""


_service: ContextualCoworkService | None = None


def get_contextual_cowork_service() -> ContextualCoworkService:
    global _service
    if _service is None:
        _service = ContextualCoworkService()
    return _service
