/**
 * script.js — JARVIS HUD Engine (Full State Machine)
 * =====================================================
 * State machine, true SSE streaming, audio visualization,
 * conversation log, system monitor, voice toggle, sound effects.
 */

// ── DOM Elements ───────────────────────────────────────────────
const $ = id => document.getElementById(id);
const userInput = $("user-input"), sendBtn = $("send-btn");
const responseText = $("response-text"), responseContainer = $("response-container");
const statusText = $("status-text"), statusDot = $("status-dot");
const coreLabel = $("core-label"), hudCore = $("hud-core");
const hudContainer = $("hud-container"), toast = $("toast");
const modelLabel = $("model-label"), engineLabel = $("engine-label");
const fallbackBadge = $("fallback-badge"), memoryBadge = $("memory-badge");
const memoryIndicator = $("memory-indicator");
const badgeTier = $("badge-tier"), badgeModel = $("badge-model");
const voiceBtn = $("voice-btn"), voiceRing = $("voice-ring");
const convLog = $("conv-log"), convLogBody = $("conv-log-body");
const actionPanel = $("action-panel"), actionSteps = $("action-steps");
const actionProgressBar = $("action-progress-bar");
const waveformCanvas = $("waveform-canvas");
const sysCpu = $("sys-cpu"), sysRam = $("sys-ram"), sysBattery = $("sys-battery");

// ── State ──────────────────────────────────────────────────────
let currentState = "dormant";
let isProcessing = false;
let typingTimer = null;
let currentMode = "auto";
let voiceActive = false;
let audioCtx = null;
let ttsEnabled = false;   // proactive TTS — muted by default, toggled via button

// ── API URLs ───────────────────────────────────────────────────
const API = { ask:"/ask", stream:"/ask-stream", health:"/health",
  setMode:"/set-mode", getMode:"/get-mode", sysStatus:"/system-status",
  convHistory:"/conversation-history", parseIntent:"/intent/parse",
  command:"/assistant/command" };

// ═══════════════════════════════════════════════════════════════
//  STATE MACHINE
// ═══════════════════════════════════════════════════════════════

function setState(state) {
  const prev = currentState;
  currentState = state;
  document.body.setAttribute("data-state", state);

  const labels = {
    dormant:"JARVIS", idle:"JARVIS", listening:"LISTENING",
    processing:"PROCESSING", streaming:"STREAMING",
    acting:"EXECUTING", cooldown:"JARVIS"
  };
  coreLabel.textContent = labels[state] || "JARVIS";

  const statuses = {
    dormant:["STANDBY","online"], idle:["ONLINE","online"],
    listening:["LISTENING","thinking"], processing:["THINKING","thinking"],
    streaming:["STREAMING","online"], acting:["EXECUTING","thinking"],
    cooldown:["ONLINE","online"]
  };
  if (statuses[state]) setStatus(statuses[state][0], statuses[state][1]);

  // Cooldown auto-transitions to idle
  if (state === "cooldown") {
    playTone(800, 0.08, 0.15);
    setTimeout(() => { if (currentState === "cooldown") setState("idle"); }, 1500);
  }
  if (state === "processing") playTone(400, 0.05, 0.1);
}

// ═══════════════════════════════════════════════════════════════
//  INIT
// ═══════════════════════════════════════════════════════════════

document.addEventListener("DOMContentLoaded", () => {
  createParticles();
  createTickMarks();
  checkHealth();
  fetchMode();
  pollSystemStatus();
  userInput.focus();
  setTimeout(() => setState("idle"), 2000);
  setInterval(() => { if (!isProcessing) checkHealth(); }, 30000);  // skip health during generation
  setInterval(() => { if (!isProcessing) pollSystemStatus(); }, 8000);  // slower polls, skip during gen
});

// ═══════════════════════════════════════════════════════════════
//  HEALTH CHECK
// ═══════════════════════════════════════════════════════════════

async function checkHealth() {
  try {
    const r = await fetch(API.health, {signal:AbortSignal.timeout(5000)});
    const d = await r.json();
    const e = d.engines || {};
    const ollamaOnline = e.ollama || e.ollama_local;
    if (engineLabel) {
      if (ollamaOnline) engineLabel.textContent = "ENGINE: OLLAMA";
      else if (e.openrouter) engineLabel.textContent = "ENGINE: OPENROUTER";
      else if (e.nvidia_nim) engineLabel.textContent = "ENGINE: NVIDIA";
      else engineLabel.textContent = "ENGINE: NONE";
    }
    if (e.openrouter || e.nvidia_nim || ollamaOnline) {
      if (currentState === "dormant" || currentState === "idle") setStatus("ONLINE","online");
    } else {
      setStatus("NO ENGINE","offline");
      showToast("No AI engine. Start Ollama or set OPENROUTER_API_KEY.","error");
    }
    // Update session stats in badge
    if (d.session_stats) {
      const total = d.session_stats.total || 0;
      if (total > 0 && badgeModel) badgeModel.textContent = `${total} calls`;
    }
  } catch {
    setStatus("DISCONNECTED","offline");
  }
}

// ═══════════════════════════════════════════════════════════════
//  MODE MANAGEMENT
// ═══════════════════════════════════════════════════════════════

async function fetchMode() {
  try {
    const r = await fetch(API.getMode);
    const d = await r.json();
    currentMode = d.mode || "auto";
    updateModeUI(currentMode);
    updateModelDisplay(d.active_model || "auto");
  } catch {}
}

async function switchMode(mode) {
  if (isProcessing) return;
  try {
    const r = await fetch(API.setMode, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({mode})
    });
    const d = await r.json();
    if (d.success) {
      currentMode = d.mode;
      updateModeUI(currentMode);
      updateModelDisplay(d.active_model || currentMode);
      showToast(`Mode: ${currentMode.toUpperCase()}`,"success");
    }
  } catch { showToast("Could not change mode","error"); }
}

function updateModeUI(mode) {
  document.querySelectorAll(".mode-btn").forEach(b => b.classList.remove("active"));
  const b = document.querySelector(`.mode-btn[data-mode="${mode}"]`);
  if (b) b.classList.add("active");
  if (modelLabel) modelLabel.textContent = `MODE: ${mode.toUpperCase()}`;
}

function updateModelDisplay(model) {
  if (modelLabel) modelLabel.textContent = `MODEL: ${model.toUpperCase()}`;
}

// ═══════════════════════════════════════════════════════════════
//  SEND MESSAGE (True SSE Streaming)
// ═══════════════════════════════════════════════════════════════

async function sendMessage() {
  const message = userInput.value.trim();
  if (!message || isProcessing) return;

  isProcessing = true;
  sendBtn.disabled = true;
  userInput.disabled = true;

  setState("processing");
  clearResponse();
  hide(fallbackBadge); hide(memoryBadge);
  showThinkingDots();

  try {
    const directHandled = await tryDirectCommand(message);
    if (directHandled) return;

    const endpoint = ttsEnabled ? "/voice/realtime-ask" : API.stream;
const resp = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, tts: ttsEnabled }),
    signal: AbortSignal.timeout(200000),
});
    if (!resp.ok) throw new Error(`Stream failed: ${resp.status}`);

    clearResponse();
    setState("streaming");

    const cursor = document.createElement("span");
    cursor.className = "cursor-blink";
    responseText.appendChild(cursor);

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    let wordBuffer = "";

    while (true) {
      const {done, value} = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, {stream:true});

      const lines = buffer.split("\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const js = line.slice(6).trim();
        if (!js) continue;
        try {
          const data = JSON.parse(js);
          if (data.done) {
            handleStreamComplete(data, cursor);
            break;
          }
          if (data.char !== undefined) {
            const node = document.createTextNode(data.char);
            responseText.insertBefore(node, cursor);
            responseContainer.scrollTop = responseContainer.scrollHeight;

            // Flash effect on word boundaries
            wordBuffer += data.char;
            if (data.char === " " || data.char === "\n") {
              wordBuffer = "";
            }
          }
        } catch {}
      }
    }
  } catch (err) {
    console.warn("Stream failed, fallback:", err.message);
    if (err.name === "TimeoutError") {
      showToast("Model took too long. Switching to faster fallback...", "error");
      setResponse("Generation timed out, sir. Try switching to FAST mode for quicker responses.");
      setState("cooldown");
    } else {
      await sendMessageFallback(message);
    }
  } finally {
    isProcessing = false;
    sendBtn.disabled = false;
    userInput.disabled = false;
    userInput.value = "";
    userInput.focus();
  }
}

async function tryDirectCommand(message) {
  const intentResp = await fetch(API.parseIntent, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message}), signal:AbortSignal.timeout(5000)
  });
  if (!intentResp.ok) return false;

  const intent = await intentResp.json();
  if (intent.needs_ai || intent.action === "chat") return false;

  setState("acting");
  renderActionIntent(intent);

  const commandResp = await fetch(API.command, {
    method:"POST", headers:{"Content-Type":"application/json"},
    body:JSON.stringify({message, speak:ttsEnabled}), signal:AbortSignal.timeout(60000)
  });
  const result = await commandResp.json();
  if (!commandResp.ok || result.error) throw new Error(result.error || "Command failed");

  renderActionResult(result);
  setResponse(result.response || result.action_result || "Done.");
  if (result.ok === false) showToast(result.action_result || "Action reported a problem", "error");
  else showToast(result.response || "Action complete", "success");
  setState("cooldown");
  return true;
}

function renderActionIntent(intent) {
  if (!actionSteps) return;
  actionSteps.innerHTML = "";
  const item = document.createElement("div");
  item.className = "action-step running";
  item.textContent = `${(intent.action || "action").replaceAll("_"," ")} ${JSON.stringify(intent.args || {})}`;
  actionSteps.appendChild(item);
  if (actionProgressBar) actionProgressBar.style.width = "45%";
}

function renderActionResult(result) {
  if (!actionSteps) return;
  const item = actionSteps.querySelector(".action-step");
  if (item) {
    item.classList.remove("running");
    item.classList.add(result.ok === false ? "failed" : "success");
    item.textContent = result.action_result || result.response || "Complete";
  }
  if (actionProgressBar) actionProgressBar.style.width = "100%";
}

function handleStreamComplete(data, cursor) {
  setTimeout(() => { if (cursor.parentNode) cursor.remove(); }, 1200);

  // Update badges
  if (data.model) {
    updateModelDisplay(data.model);
    if (badgeTier) badgeTier.textContent = (data.tier || "auto").toUpperCase();
    if (badgeModel) badgeModel.textContent = data.model.split("/").pop().split(":")[0];
  }
  if (data.fallback) show(fallbackBadge);
  if (data.memory_used) { show(memoryBadge); showMemoryIndicator(); }
  if (data.error) showToast(data.error, "error");

  // Highlight code blocks
  responseText.querySelectorAll("pre code").forEach(b => hljs.highlightElement(b));

  // Speak the full response if TTS is enabled
  const fullText = responseText.innerText || responseText.textContent || "";
  speakResponse(fullText);

  setState("cooldown");
}

async function sendMessageFallback(message) {
  try {
    const r = await fetch(API.ask, {
      method:"POST", headers:{"Content-Type":"application/json"},
      body:JSON.stringify({message}), signal:AbortSignal.timeout(200000)
    });
    const d = await r.json();
    if (d.error) {
      setStatus("ERROR","offline"); showToast(d.error,"error");
      setResponse("Systems encountered an anomaly, sir.");
    } else {
      typeResponse(d.response);
      if (d.model) updateModelDisplay(d.model);
      if (d.fallback) show(fallbackBadge);
      if (d.memory_used) { show(memoryBadge); showMemoryIndicator(); }
      speakResponse(d.response);
    }
    setState("cooldown");
  } catch (err) {
    setStatus("ERROR","offline");
    showToast(err.name==="TimeoutError"?"Request timed out.":"Server unreachable.","error");
    setResponse("Connection failed. Is the server running?");
    setState("idle");
  }
}

// ═══════════════════════════════════════════════════════════════
//  TYPING ANIMATION
// ═══════════════════════════════════════════════════════════════

function typeResponse(text) {
  clearResponse();
  setState("streaming");
  let i = 0;
  const cursor = document.createElement("span");
  cursor.className = "cursor-blink";
  responseText.appendChild(cursor);

  function next() {
    if (i < text.length) {
      responseText.insertBefore(document.createTextNode(text[i]), cursor);
      i++;
      responseContainer.scrollTop = responseContainer.scrollHeight;
      typingTimer = setTimeout(next, 18);
    } else {
      setTimeout(() => { if(cursor.parentNode) cursor.remove(); }, 1500);
      setState("cooldown");
    }
  }
  typingTimer = setTimeout(next, 18);
}

function showThinkingDots() {
  responseText.innerHTML = "";
  let dots = 0;
  const el = document.createElement("span");
  el.style.color = "var(--cyan-dim)";
  el.style.letterSpacing = "2px";
  responseText.appendChild(el);
  function animate() {
    if (!isProcessing) return;
    dots = (dots % 3) + 1;
    el.textContent = "Analyzing" + ".".repeat(dots);
    typingTimer = setTimeout(animate, 400);
  }
  animate();
}

function clearResponse() {
  if (typingTimer) { clearTimeout(typingTimer); typingTimer = null; }
  responseText.innerHTML = "";
}
function setResponse(text) { clearResponse(); responseText.textContent = text; }

// ═══════════════════════════════════════════════════════════════
//  SYSTEM MONITOR
// ═══════════════════════════════════════════════════════════════

async function pollSystemStatus() {
  try {
    const r = await fetch(API.sysStatus, {signal:AbortSignal.timeout(3000)});
    const d = await r.json();
    if (sysCpu) sysCpu.textContent = `CPU: ${d.cpu_percent || 0}%`;
    if (sysRam) sysRam.textContent = `RAM: ${d.ram_percent || 0}%`;
    if (sysBattery) {
      if (d.battery_percent !== null && d.battery_percent !== undefined) {
        sysBattery.textContent = `BAT: ${d.battery_percent}%${d.battery_plugged?" ⚡":""}`;
      } else {
        sysBattery.textContent = "BAT: AC";
      }
    }
  } catch {}
}

// ═══════════════════════════════════════════════════════════════
//  CONVERSATION LOG
// ═══════════════════════════════════════════════════════════════

function toggleConvLog() {
  convLog.classList.toggle("hidden");
  if (!convLog.classList.contains("hidden")) loadConversationHistory();
}

async function loadConversationHistory() {
  try {
    const r = await fetch(API.convHistory);
    const d = await r.json();
    convLogBody.innerHTML = "";
    (d.history || []).forEach(msg => {
      const div = document.createElement("div");
      div.className = `conv-msg ${msg.role}`;
      div.innerHTML = `
        <div class="conv-msg__role">${msg.role.toUpperCase()}</div>
        <div class="conv-msg__text">${escapeHtml(msg.text).slice(0,300)}</div>
        <div class="conv-msg__meta">${msg.timestamp || ""}${msg.model?" · "+msg.model:""}</div>
      `;
      convLogBody.appendChild(div);
    });
    convLogBody.scrollTop = convLogBody.scrollHeight;
  } catch {}
}

function escapeHtml(t) {
  const d = document.createElement("div"); d.textContent = t; return d.innerHTML;
}

// ═══════════════════════════════════════════════════════════════
//  NATIVE VOICE TOGGLE
// ═══════════════════════════════════════════════════════════════

function toggleVoice() {
  toggleWakeWord();
}

// ═══════════════════════════════════════════════════════════════
//  SOUND ENGINE (Web Audio API — no files needed)
// ═══════════════════════════════════════════════════════════════

function playTone(freq, dur, vol) {
  try {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const osc = audioCtx.createOscillator();
    const gain = audioCtx.createGain();
    osc.connect(gain); gain.connect(audioCtx.destination);
    osc.frequency.value = freq;
    osc.type = "sine";
    gain.gain.setValueAtTime(vol || 0.1, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + dur);
    osc.start(); osc.stop(audioCtx.currentTime + dur);
  } catch {}
}

// ═══════════════════════════════════════════════════════════════
//  PROACTIVE TTS
// ═══════════════════════════════════════════════════════════════

async function speakResponse(text) {
  if (!ttsEnabled || !text || !text.trim()) return;

  // Strip to first ~400 chars so pyttsx3 doesn't choke on huge responses.
  // Trim at the last sentence boundary within that limit.
  let trimmed = text.trim().replace(/\s+/g, " ");
  if (trimmed.length > 400) {
    const cut = trimmed.slice(0, 400).search(/[.!?][^.!?]*$/);
    trimmed = cut > 60 ? trimmed.slice(0, cut + 1) : trimmed.slice(0, 400);
  }

  try {
    const res = await fetch("/voice/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: trimmed }),
      signal: AbortSignal.timeout(8000),
    });
    const json = await res.json();
    if (json.error) {
      console.error("[TTS] server error:", json.error);
      showToast("TTS error: " + json.error, "error");
    }
  } catch (e) {
    console.warn("[TTS] speak failed:", e.message);
    showToast("TTS failed — check server logs", "error");
  }
}

async function testTTS() {
  // Fires a hardcoded test phrase so we can verify the whole pipeline.
  try {
    const res = await fetch("/voice/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: "JARVIS voice systems online, sir." }),
      signal: AbortSignal.timeout(8000),
    });
    const json = await res.json();
    if (json.error) showToast("TTS test failed: " + json.error, "error");
    else showToast("TTS test sent — you should hear JARVIS now", "success");
  } catch (e) {
    showToast("TTS test error: " + e.message, "error");
  }
}

function toggleTTS() {
  ttsEnabled = !ttsEnabled;
  const btn = $("tts-toggle-btn");
  if (!btn) return;
  if (ttsEnabled) {
    btn.textContent = "TTS: ON";
    btn.classList.add("tts-active");
    showToast("Voice output ON — testing now...", "success");
    playTone(660, 0.12, 0.12);
    testTTS();   // immediately verify the pipeline works
  } else {
    btn.textContent = "TTS: OFF";
    btn.classList.remove("tts-active");
    showToast("Voice output muted", "success");
    playTone(330, 0.12, 0.08);
  }
}

// ═══════════════════════════════════════════════════════════════
//  UI HELPERS
// ═══════════════════════════════════════════════════════════════

function setStatus(text, state) {
  statusText.textContent = text;
  statusDot.className = "status-dot";
  if (state === "offline") statusDot.classList.add("offline");
  if (state === "thinking") statusDot.classList.add("thinking");
}

function show(el) { if (el) el.classList.remove("hidden"); }
function hide(el) { if (el) el.classList.add("hidden"); }

function showMemoryIndicator() {
  if (memoryIndicator) {
    memoryIndicator.classList.remove("hidden");
    memoryIndicator.classList.add("active");
    setTimeout(() => memoryIndicator.classList.remove("active"), 5000);
  }
}

function showToast(message, type) {
  toast.textContent = message;
  toast.className = "toast visible";
  if (type === "success") toast.classList.add("success");
  setTimeout(() => toast.classList.remove("visible"), 4000);
}

// ═══════════════════════════════════════════════════════════════
//  VISUAL EFFECTS
// ═══════════════════════════════════════════════════════════════

function createParticles() {
  const c = $("particles");
  if (!c) return;
  for (let i = 0; i < 25; i++) {
    const p = document.createElement("div");
    p.className = "particle";
    p.style.left = Math.random()*100 + "%";
    p.style.animationDuration = (8 + Math.random()*12) + "s";
    p.style.animationDelay = (Math.random()*10) + "s";
    p.style.width = (1+Math.random()*2) + "px";
    p.style.height = p.style.width;
    c.appendChild(p);
  }
}

function createTickMarks() {
  const c = $("ring-ticks");
  if (!c) return;
  for (let i = 0; i < 36; i++) {
    const t = document.createElement("div");
    t.className = "tick";
    t.style.transform = `rotate(${i*(360/36)}deg)`;
    t.style.opacity = i%3===0 ? "0.5" : "0.2";
    t.style.height = i%3===0 ? "8px" : "4px";
    c.appendChild(t);
  }
}

// ═══════════════════════════════════════════════════════════════
//  WAKE WORD CLIENT
// ═══════════════════════════════════════════════════════════════

let wakeEnabled = false;
let wakeEventSource = null;

function setWakeButtonState(enabled) {
  const btn = $("wake-toggle-btn");
  if (!btn) return;
  wakeEnabled = enabled;
  voiceActive = enabled;
  btn.textContent = `WAKE: ${enabled ? "ON" : "OFF"}`;
  btn.style.color = enabled ? "var(--green)" : "";
  btn.style.borderColor = enabled ? "var(--green)" : "";
  if (voiceBtn) voiceBtn.classList.toggle("active", enabled);
}

async function toggleWakeWord() {
  try {
    if (!wakeEnabled) {
      // Enable on server
      const r = await fetch("/voice/wake-enable", { method: "POST" });
      const d = await r.json();
      if (d.error) { showToast("Wake word error: " + d.error, "error"); return; }

      setWakeButtonState(true);
      showToast("Native wake word active - say 'Hey JARVIS'", "success");
      connectWakeStream();
    } else {
      // Disable on server
      await fetch("/voice/wake-disable", { method: "POST" });
      if (wakeEventSource) { wakeEventSource.close(); wakeEventSource = null; }
      setWakeButtonState(false);
      showToast("Wake word disabled", "success");
      if (currentState === "listening") setState("idle");
    }
  } catch (e) {
    showToast("Could not toggle wake word", "error");
  }
}

function connectWakeStream() {
  if (wakeEventSource) { wakeEventSource.close(); }

  wakeEventSource = new EventSource("/voice/wake-stream");

  wakeEventSource.onmessage = (e) => {
    let data;
    try { data = JSON.parse(e.data); } catch { return; }

    switch (data.event) {
      case "detected":
        // HUD reacts: pulse the core, show LISTENING state
        setState("listening");
        setResponse("Listening...");
        playTone(880, 0.12, 0.15);   // rising ping
        playTone(1100, 0.08, 0.1);
        showToast("Hey JARVIS — I'm listening", "success");
        break;

      case "listening":
        // Already in listening state — update label
        coreLabel.textContent = "LISTENING";
        break;

      case "transcript":
        // Got transcribed text — populate input and auto-send
        if (data.text && !isProcessing) {
          userInput.value = data.text;
          // Brief pause so user can see what was heard
          setTimeout(() => sendMessage(), 300);
        } else if (isProcessing) {
          showToast(`Heard: "${data.text}" (busy — try again)`, "success");
          setState("idle");
        }
        break;

      case "error":
        showToast(data.message || "Wake word error", "error");
        if (currentState === "listening") setState("idle");
        break;

      case "unavailable":
        showToast("Wake word unavailable: " + data.message, "error");
        setWakeButtonState(false);
        wakeEventSource.close();
        wakeEventSource = null;
        break;

      case "disabled":
        // Server says service is off — sync button state
        setWakeButtonState(false);
        wakeEventSource.close();
        wakeEventSource = null;
        break;

      case "heartbeat":
        // Keep-alive — no UI action needed
        break;
    }
  };

  wakeEventSource.onerror = () => {
    // Reconnect after 3s if wake word is still enabled
    if (wakeEnabled) {
      setTimeout(() => { if (wakeEnabled) connectWakeStream(); }, 3000);
    }
  };
}

// ═══════════════════════════════════════════════════════════════
//  EVENT LISTENERS
// ═══════════════════════════════════════════════════════════════

sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keydown", e => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});
userInput.addEventListener("input", () => {
  hudCore.style.boxShadow = userInput.value.length > 0
    ? "0 0 40px rgba(0,240,255,0.15),inset 0 0 40px rgba(0,240,255,0.06)" : "";
});

document.querySelectorAll(".mode-btn").forEach(b => {
  b.addEventListener("click", () => {
    const m = b.dataset.mode;
    if (m && m !== currentMode) switchMode(m);
  });
});

if (voiceBtn) voiceBtn.addEventListener("click", toggleVoice);
if ($("toggle-log-btn")) $("toggle-log-btn").addEventListener("click", toggleConvLog);
if ($("conv-log-close")) $("conv-log-close").addEventListener("click", () => hide(convLog));
if ($("wake-toggle-btn")) $("wake-toggle-btn").addEventListener("click", toggleWakeWord);
if ($("tts-toggle-btn")) $("tts-toggle-btn").addEventListener("click", toggleTTS);
if ($("toggle-stats-btn")) $("toggle-stats-btn").addEventListener("click", async () => {
  try {
    const r = await fetch("/session-stats");
    const d = await r.json();
    showToast(`Calls: ${d.total} | Fallbacks: ${d.fallbacks}`, "success");
  } catch { showToast("Stats unavailable","error"); }
});
