"""Windows system tray app for the always-running JARVIS runtime."""

from __future__ import annotations

import time
from pathlib import Path

from core.assistant_runtime import get_runtime
from core.paths import resource_path
from tray.tray_menu import build_menu


def load_tray_icon():
    """Load the packaged icon, falling back to a generated bitmap."""
    from PIL import Image, ImageDraw

    icon_path = resource_path("assets", "jarvis.ico")
    if icon_path.exists():
        return Image.open(icon_path)

    image = Image.new("RGBA", (64, 64), (8, 14, 20, 255))
    draw = ImageDraw.Draw(image)
    draw.ellipse((8, 8, 56, 56), outline=(0, 220, 255, 255), width=4)
    draw.ellipse((22, 22, 42, 42), fill=(0, 220, 255, 255))
    return image


class TrayApp:
    """Owns the pystray icon and forwards menu actions to AssistantRuntime."""

    def __init__(self):
        self.runtime = get_runtime()
        self.icon = None

    def run(self, open_ui: bool = False, enable_wake: bool = True) -> None:
        self.runtime.start(enable_wake=enable_wake, open_ui=open_ui)
        try:
            import pystray
        except ImportError:
            print("pystray is not installed. Runtime is running without a tray icon.")
            self._run_forever_without_tray()
            return

        self.icon = pystray.Icon(
            "JARVIS",
            load_tray_icon(),
            "JARVIS Desktop Assistant",
            menu=build_menu(self),
        )
        self.icon.run()

    def open_ui(self, icon=None, item=None) -> None:
        self.runtime.open_ui()

    def toggle_mute(self, icon=None, item=None) -> None:
        self.runtime.set_voice_muted(not self.runtime.state.voice_muted)

    def toggle_wake_word(self, icon=None, item=None) -> None:
        self.runtime.toggle_wake_word()

    def restart(self, icon=None, item=None) -> None:
        self.runtime.restart()

    def quit(self, icon=None, item=None) -> None:
        self.runtime.stop()
        if self.icon:
            self.icon.stop()

    def _run_forever_without_tray(self) -> None:
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            self.runtime.stop()
