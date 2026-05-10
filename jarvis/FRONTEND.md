# 🛡️ JARVIS Frontend — Iron Man HUD Interface

A futuristic AI assistant interface inspired by **Iron Man's JARVIS HUD**.
Built with pure HTML, CSS, and JavaScript — no frameworks, no build tools.

Connects to a local Flask backend at `http://localhost:5000` and displays AI responses with smooth animations and a typing effect.

---

## ⚡ Features

| Feature | Description |
|---------|-------------|
| **Animated HUD Core** | Central circle with glowing border and "JARVIS" label |
| **4 Rotating Rings** | Concentric rings spinning at different speeds and directions |
| **36 Tick Marks** | Dynamic tick marks generated via JavaScript |
| **Thinking State** | Rings speed up + core turns orange while AI is processing |
| **Typing Animation** | Response appears character-by-character with a blinking cursor |
| **Floating Particles** | 20 ambient particles rising from the bottom of the screen |
| **Scanline Overlay** | Subtle horizontal lines for an authentic HUD look |
| **Neon Glow Effects** | Cyan glows, borders, and text shadows throughout |
| **Status Indicator** | Top-right dot shows ONLINE / THINKING / ERROR / OFFLINE |
| **Toast Notifications** | Error and success messages slide in from the top |
| **Glow on Input** | Core brightens when the user starts typing |
| **Responsive Design** | Adapts to small screens and short viewports |
| **Health Checks** | Polls the backend every 30 seconds to verify connection |
| **Engine Display** | Bottom bar dynamically shows which AI engine is active |

---

## 📁 File Breakdown

### `index.html` — Structure

The main HTML file that defines the layout of every UI element:

| Section | Element | Purpose |
|---------|---------|---------|
| **Top Bar** | `.top-bar` | Logo ("J.A.R.V.I.S") and status indicator |
| **HUD Center** | `.hud-container` | 4 rotating rings, tick marks, corner brackets, and the AI core |
| **Response Area** | `.response-container` | Scrollable area where AI output appears |
| **Input Area** | `.input-container` | Command prefix (`CMD ›`), text input, and SEND button |
| **Bottom Bar** | `.bottom-bar` | Shows model name, active engine, and fallback engine |
| **Toast** | `.toast` | Pop-up notification for errors and status changes |
| **Overlays** | `.scanlines`, `.particles` | Visual atmosphere layers |

No external dependencies — just loads `style.css` and `script.js`.

---

### `style.css` — Visuals & Animations

All styling, animations, and effects are defined here using **CSS variables** for easy customization.

#### Color System (CSS Variables)

```css
--cyan: #00f0ff;          /* Primary accent */
--cyan-dim: #00a8b4;      /* Muted accent */
--cyan-glow: rgba(0, 240, 255, 0.4);  /* Glow effects */
--orange: #ff6a00;        /* Thinking state */
--bg-primary: #020a13;    /* Main background */
--bg-card: rgba(6, 18, 32, 0.85);     /* Card backgrounds */
--text-primary: #e0f4ff;  /* Main text */
--text-secondary: #6b9bbb;/* Muted text */
```

#### Key Animations

| Animation | Element | Speed |
|-----------|---------|-------|
| `spin-cw` | Ring 1, Ring 3, Tick marks | 20s, 25s, 60s |
| `spin-ccw` | Ring 2, Ring 4 | 15s, 35s |
| `pulse-dot` | Status indicator | 2s (normal), 0.6s (thinking) |
| `blink` | Typing cursor | 0.7s |
| `float-particle` | Ambient particles | 8-20s (random) |

#### Thinking State

When the AI is processing, CSS classes change to:
- Rings speed up (20s → 4s, 15s → 3s, etc.)
- Core border turns orange
- Core glow shifts from cyan to orange
- Status dot pulses faster

#### Fonts

- **Orbitron** — Used for labels, status text, and buttons (futuristic mono look)
- **Rajdhani** — Used for response text and input (clean, readable)

Both loaded from Google Fonts.

---

### `script.js` — Logic & Interaction

Handles all frontend behavior. No libraries, just vanilla JavaScript.

#### Functions Overview

| Function | What it does |
|----------|-------------|
| `checkHealth()` | Polls `GET /health` every 30s, updates status dot and engine label |
| `sendMessage()` | Reads input, sends `POST /ask`, handles response or error |
| `typeResponse(text)` | Types text character-by-character (18ms per char) with blinking cursor |
| `showThinkingDots()` | Shows "Analyzing..." animation while waiting for response |
| `clearResponse()` | Clears the output area and cancels any active typing |
| `setStatus(text, state)` | Updates the top-right status dot and label |
| `setCoreState(state, label)` | Toggles the core between idle and thinking mode |
| `showToast(message, type)` | Shows a slide-in notification (auto-hides after 4s) |
| `createParticles()` | Generates 20 floating particle elements |
| `createTickMarks()` | Generates 36 tick marks on the HUD ring |

#### API Communication

```
POST /ask
Content-Type: application/json
Body: { "message": "user text" }

Response: { "response": "AI text", "action": "NONE" }
```

- Timeout: 65 seconds
- Uses `AbortSignal.timeout()` for automatic cancellation
- Errors display via toast notification

---

## 🔄 How It Works

```
User types message
       ↓
  Press Enter or click SEND
       ↓
  UI locks (input disabled, button greyed out)
  Core switches to "PROCESSING" (orange)
  Rings speed up
  "Analyzing..." animation starts
       ↓
  POST /ask → Flask backend → AI engine
       ↓
  Response received
       ↓
  Core returns to "JARVIS" (cyan)
  Rings slow back down
  Response types out character-by-character
       ↓
  UI unlocks, input refocuses
  Ready for next message
```

---

## 🚀 How to Run

#### 1. Start the backend

```powershell
cd jarvis
python server.py
```

> The Flask server starts on `http://localhost:5000` and serves the frontend files automatically.

#### 2. Open the frontend

Navigate to **http://localhost:5000** in your browser.

> You do NOT need to open `index.html` directly — the Flask server serves it.

#### 3. Interact

- Type a message in the `CMD ›` input field
- Press **Enter** or click **SEND**
- Watch the HUD animate and the response type out

---

## 📌 Important Notes

- **Backend must be running** — the frontend communicates with `localhost:5000`
- **No internet required** if using Ollama (local model)
- **Faster with Gemini** — set your API key for cloud-powered responses
- **Health checks run automatically** every 30 seconds
- **Engine label updates dynamically** based on the active AI engine
- **Works in any modern browser** — Chrome, Edge, Firefox, Safari

---

## 🎨 Customization Guide

### Change Colors

Edit the CSS variables in `:root` at the top of `style.css`:

```css
/* Change the primary accent from cyan to green */
--cyan: #00ff88;
--cyan-dim: #00b45e;
--cyan-glow: rgba(0, 255, 136, 0.4);
```

### Change Ring Speeds

Modify the `animation-duration` values on `.ring-1` through `.ring-4`:

```css
/* Make rings spin faster */
.ring-1 { animation: spin-cw 10s linear infinite; }  /* was 20s */
.ring-2 { animation: spin-ccw 8s linear infinite; }  /* was 15s */
```

### Change Typing Speed

In `script.js`, adjust the `speed` variable in `typeResponse()`:

```javascript
const speed = 10;  // faster (was 18ms per character)
```

### Change Fonts

Replace the Google Fonts import in `style.css` line 6:

```css
@import url('https://fonts.googleapis.com/css2?family=YourFont&display=swap');
```

Then update `font-family` references throughout the file.

### Adjust Layout

Key sizing is controlled by:

```css
--ring-size: min(420px, 80vw);  /* HUD diameter */
```

Response and input widths use `min(600px, 88vw)` for responsive sizing.

---

## 🔮 Future Improvements

- 🎤 **Voice input** — add speech-to-text for hands-free interaction
- 🔊 **Voice output** — text-to-speech for JARVIS-like spoken responses
- 🌀 **3D HUD** — WebGL/Three.js for a true holographic interface
- 📊 **Dashboard panels** — system stats, weather, calendar widgets
- 🖥️ **Fullscreen mode** — immersive fullscreen HUD experience
- ⌨️ **Command history** — up/down arrow to recall previous messages
- 🎵 **Sound effects** — subtle audio cues for thinking and response states
- 🌐 **Multi-language** — i18n support for the interface

---

## 🛠️ Tech Stack

| Technology | Usage |
|------------|-------|
| **HTML5** | Semantic structure |
| **CSS3** | Animations, variables, gradients, blur, responsive design |
| **JavaScript (ES6+)** | Async/await, fetch API, DOM manipulation |
| **Google Fonts** | Orbitron + Rajdhani |
| **No frameworks** | Zero dependencies — pure vanilla code |
