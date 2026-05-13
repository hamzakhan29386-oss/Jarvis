# 🤖 JARVIS — Cognitive AI Assistant with Iron Man HUD

A highly sophisticated, 100% free, local-first AI assistant. JARVIS features a multi-model cognitive architecture, persistent memory, a modular voice stack, and a comprehensive action automation engine, all wrapped in a futuristic Iron Man-inspired HUD.

## ⚡ Features

- **Multi-Model Cognitive Architecture** — Intelligent query classification routes tasks to the best free model (Local Ollama or OpenRouter free tier).
- **Persistent Memory Engine** — Features episodic, semantic (user profiles), and procedural memory. JARVIS remembers past conversations, procedures, and learns about you over time.
- **Modular Voice Stack** — Voice control powered by Whisper (STT) and pyttsx3 (TTS) for a seamless hands-free experience.
- **Agent-Based Action System** — Over 13 system commands, multi-step action planning, and system telemetry monitoring (CPU, RAM, Battery).
- **True SSE Streaming** — Token-by-token streaming responses directly from the AI engines.
- **Iron Man HUD** — Reactive animations, rotating rings, neon glows, and real-time telemetry display.

## 📁 Project Structure

```
jarvis/
├── index.html       # HUD interface (Iron Man style)
├── style.css        # Animations, glows, rings, reactive effects
├── script.js        # Frontend logic (streaming, voice, actions, state)
├── server.py        # Flask backend (REST + SSE, memory, actions endpoints)
├── brain.py         # Cognitive Control Layer (Routing, OpenRouter + Ollama)
├── actions.py       # Action executor (System stats, multi-step plans)
├── memory.py        # Persistent memory engine (Episodic, Semantic, Procedural)
├── voice.py         # Voice stack (Whisper STT, pyttsx3 TTS)
├── main.py          # Terminal mode (alternative)
├── requirements.txt # Python dependencies
└── FRONTEND.md      # Frontend documentation (detailed)
```

## 🚀 Setup

### Step 1: Install dependencies

```powershell
cd jarvis
pip install -r requirements.txt
```

*(Ensure you also have [Ollama](https://ollama.com/) installed if you plan to use local models).*

### Step 2: Set your API keys (Optional but Recommended)

JARVIS uses OpenRouter's free tier for cloud processing and Gemini as a fallback.

```powershell
# PowerShell
$env:OPENROUTER_API_KEY="your-free-openrouter-key"
$env:GEMINI_API_KEY="your-gemini-key" # Optional fallback

# CMD
set OPENROUTER_API_KEY=your-free-openrouter-key
set GEMINI_API_KEY=your-gemini-key
```

### Step 3: Start the local AI (Optional)
If using Ollama, run it in the background:
```powershell
ollama serve
```

### Step 4: Run the server

```powershell
python server.py
```

### Step 5: Open the HUD

Open **http://localhost:5000** in your browser.

## Desktop Runtime

JARVIS can now run as a persistent Windows desktop assistant with a tray icon,
wake-word listener, supervised background threads, and fast desktop intent
routing.

```powershell
# Run in the background with tray controls
python -m core.launcher --tray --open-ui

# Install Windows Startup launcher
powershell -ExecutionPolicy Bypass -File scripts\install_startup.ps1
```

Tray menu:
- Open JARVIS
- Mute voice
- Toggle wake word
- Restart assistant
- Quit

Direct commands such as "Play Interstellar docking scene", "Search AI news on
YouTube", "Open coding setup", "set volume 40", and "take screenshot" are parsed
locally first and executed through `actions.py`. General conversation still flows
through `brain.py` with the existing cloud/local fallback routing.

Runtime modules:
- `core/launcher.py` starts the desktop runtime.
- `core/assistant_runtime.py` supervises Flask, wake-word events, and routing.
- `tray/tray_app.py` and `tray/tray_menu.py` own system tray behavior.
- `automation/` contains modular browser, YouTube, desktop, system, and workflow agents.

## Windows EXE Build

Build an installer-ready desktop executable:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\build_windows.ps1
```

Output:

```text
dist\JARVIS.exe
```

The build script generates `assets\jarvis.ico`, installs packaging tools, installs
Playwright Chromium, and runs PyInstaller with `JARVIS.spec`.

Useful commands:

```powershell
# Launch packaged app from the tray
dist\JARVIS.exe --tray

# Register packaged app for Windows startup
dist\JARVIS.exe --install-startup

# Source fallback startup registration
scripts\install_startup.bat
```

For Playwright-powered browser automation, run once after installing
dependencies:

```powershell
python -m playwright install chromium
```

## 🧠 Cognitive Architecture

JARVIS routes your queries dynamically based on complexity to the following tiers:
- **Reflex (FAST):** Simple commands & greetings → `qwen2.5:0.5b` (~300MB RAM, instant)
- **Analyst (AUTO):** Reasoning & planning → `phi3:latest` (balanced speed/quality)
- **Coder (SMART):** Code generation & debugging → `llama3:8b-instruct-q4_0` / `llama3:latest`
- **Backup:** Ultra-light fallback → `qwen2.5:0.5b`

**Modes:**
| Mode | Primary Model | Best For |
|------|--------------|----------|
| `fast` | qwen2.5:0.5b | Instant responses, low RAM |
| `auto` | Auto-classified | Balanced (default) |
| `smart` | llama3:latest | Maximum quality |

**Auto-fallback:** If a model times out or fails, JARVIS automatically tries the next model in the chain, ending with the ultra-light `qwen2.5:0.5b` backup.

## 💾 Memory Engine

JARVIS features a three-tier memory system (`memory.py`):
1. **Episodic Memory:** Remembers past conversations and interactions.
2. **Semantic Memory:** Builds a profile of your preferences and facts over time.
3. **Procedural Memory:** Can memorize custom multi-step action sequences. (Say *"Remember this procedure..."*)

## ⚙️ Action & Agent System

JARVIS can execute multi-step plans and monitor your device:
- Control browser (Open YouTube, Google, Chrome).
- Search the web.
- Retrieve system telemetry (CPU, RAM, Disk, Battery status).
- Execute complex customized macros and agent plans via `/execute-plan`.

## 🛣️ New API Endpoints

Alongside the core `/ask` and `/ask-stream` endpoints, the updated architecture adds:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/system-status` | GET | CPU/RAM/disk/battery telemetry |
| `/execute-action` | POST | Execute a single action |
| `/execute-plan` | POST | Execute a multi-step agent plan |
| `/memory/episodes` | GET | Retrieve recent memory episodes |
| `/memory/semantic` | GET | Retrieve user profile (semantic memory) |
| `/memory/procedure` | POST | Store a new named procedure |
| `/voice/speak` | POST | Trigger text-to-speech manually |
| `/session-stats` | GET | Per-model usage statistics |

## 🔧 Troubleshooting

| Problem | Fix |
|---------|-----|
| "All engines offline" | Ensure Ollama is running (`ollama serve`) or set `OPENROUTER_API_KEY` |
| Slow first response | Ollama is loading the model into memory. Subsequent requests are instant. |
| Voice not working | Check the native Python audio logs, `sounddevice` device selection, and the `/voice/wake-status` endpoint. The browser does not own the microphone. |
| Missing modules | Run `pip install -r requirements.txt` to grab the new memory and voice requirements. |
