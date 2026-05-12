# Installer Notes

The project is now installer-ready through PyInstaller.

Build output:

```text
dist/JARVIS.exe
```

For a full Windows installer, point Inno Setup, WiX, or MSIX packaging at
`dist/JARVIS.exe` and include the generated `assets/jarvis.ico` as the product
icon. Startup registration is handled by:

```powershell
JARVIS.exe --install-startup
```

