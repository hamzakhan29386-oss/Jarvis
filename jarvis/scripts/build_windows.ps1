$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot

python scripts\generate_icon.py
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install pyinstaller pywin32
python -m playwright install chromium
python -m PyInstaller --clean --noconfirm JARVIS.spec

Write-Host ""
Write-Host "Build complete: dist\JARVIS.exe"

