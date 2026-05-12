$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $ProjectRoot
python -m core.launcher --install-startup
Write-Host "JARVIS startup launcher installed."

