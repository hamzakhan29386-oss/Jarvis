@echo off
cd /d "%~dp0\.."
if exist "dist\JARVIS.exe" (
  "dist\JARVIS.exe" --install-startup
) else (
  python -m core.launcher --install-startup
)

