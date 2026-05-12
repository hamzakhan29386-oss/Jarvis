"""Tray menu construction for JARVIS."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tray.tray_app import TrayApp


def build_menu(app: "TrayApp"):
    """Build the pystray menu lazily so imports stay optional."""
    import pystray

    return pystray.Menu(
        pystray.MenuItem("Open JARVIS", app.open_ui),
        pystray.MenuItem(
            "Toggle wake word",
            app.toggle_wake_word,
            checked=lambda _: app.runtime.state.wake_enabled,
        ),
        pystray.MenuItem(
            "Mute voice",
            app.toggle_mute,
            checked=lambda _: app.runtime.state.voice_muted,
        ),
        pystray.MenuItem("Restart assistant", app.restart),
        pystray.MenuItem("Quit", app.quit),
    )

