"""Windows desktop launcher for the persistent JARVIS runtime."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from core.assistant_runtime import get_runtime
from tray.tray_app import TrayApp


def install_startup_shortcut() -> Path:
    """Create a Windows Startup shortcut that launches JARVIS silently."""
    startup = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs" / "Startup"
    startup.mkdir(parents=True, exist_ok=True)
    shortcut = startup / "JARVIS Desktop Assistant.lnk"
    target = Path(sys.executable)
    launcher = Path(__file__).resolve()
    frozen = bool(getattr(sys, "frozen", False))
    working_dir = target.parent if frozen else launcher.parents[1]
    arguments = "--tray" if frozen else "-m core.launcher --tray"

    try:
        import win32com.client

        shell = win32com.client.Dispatch("WScript.Shell")
        link = shell.CreateShortcut(str(shortcut))
        link.TargetPath = str(target)
        link.Arguments = arguments
        link.WorkingDirectory = str(working_dir)
        icon_path = working_dir / "assets" / "jarvis.ico"
        link.IconLocation = str(icon_path if icon_path.exists() else target)
        link.Description = "Start JARVIS Desktop Assistant"
        link.Save()
    except Exception:
        fallback = startup / "Start JARVIS Desktop Assistant.bat"
        if frozen:
            command = f'start "" "{target}" --tray'
        else:
            command = f'start "" "{target}" -m core.launcher --tray'
        fallback.write_text(f'@echo off\ncd /d "{working_dir}"\n{command}\n', encoding="utf-8")
        return fallback
    return shortcut


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the JARVIS desktop assistant runtime.")
    parser.add_argument("--tray", action="store_true", help="Run with Windows system tray integration.")
    parser.add_argument("--open-ui", action="store_true", help="Open the HUD after startup.")
    parser.add_argument("--no-wake", action="store_true", help="Do not enable wake-word detection on launch.")
    parser.add_argument("--install-startup", action="store_true", help="Install a Windows Startup shortcut.")
    args = parser.parse_args()

    if args.install_startup:
        path = install_startup_shortcut()
        print(f"Startup launcher installed: {path}")
        return

    if args.tray:
        TrayApp().run(open_ui=args.open_ui, enable_wake=not args.no_wake)
        return

    runtime = get_runtime()
    runtime.start(enable_wake=not args.no_wake, open_ui=args.open_ui)
    print(f"JARVIS runtime online: {runtime.url}")
    try:
        while True:
            import time

            time.sleep(3600)
    except KeyboardInterrupt:
        runtime.stop()


if __name__ == "__main__":
    main()
