@echo off
cd /d "%~dp0\.."
if exist "dist\JARVIS.exe" (
  start "JARVIS Desktop Assistant" "dist\JARVIS.exe" --tray
) else (
  start "JARVIS Desktop Assistant" python -m core.launcher --tray
)
