# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

ROOT = Path(SPECPATH)
ICON = ROOT / "assets" / "jarvis.ico"

datas = [
    ("index.html", "."),
    ("style.css", "."),
    ("script.js", "."),
    ("FRONTEND.md", "."),
    ("README.md", "."),
    ("assets", "assets"),
    ("wake/models", "wake/models"),
]

hiddenimports = [
    "automation.browser_agent",
    "automation.youtube_agent",
    "automation.desktop_agent",
    "automation.workflow_agent",
    "automation.system_agent",
    "core.assistant_runtime",
    "core.background_threads",
    "core.intent_parser",
    "core.task_router",
    "tray.tray_app",
    "tray.tray_menu",
    "wake.service",
    "wake.detector",
    "wake.transcriber",
    "PIL._tkinter_finder",
]

a = Analysis(
    ["jarvis_desktop.py"],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="JARVIS",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(ICON) if ICON.exists() else None,
)
