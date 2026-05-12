"""
server.py — JARVIS Flask Backend (Extended)
==============================================
API bridge between the frontend HUD and all JARVIS subsystems.
Preserves all original endpoints + adds new ones for memory,
actions, voice, and system monitoring.

Original Endpoints (preserved):
    GET  /           -> Serve the HUD frontend
    GET  /health     -> Engine status check
    POST /ask        -> Full JSON response
    POST /ask-stream -> SSE streaming response (now with TRUE streaming)
    POST /set-mode   -> Switch AI mode
    GET  /get-mode   -> Get current mode info

New Endpoints:
    GET  /system-status       -> CPU/RAM/disk/battery
    POST /execute-action      -> Execute a single action
    POST /execute-plan        -> Execute a multi-step plan
    GET  /memory/episodes     -> Recent memory episodes
    GET  /memory/semantic     -> User profile (semantic memory)
    POST /memory/procedure    -> Store a named procedure
    GET  /conversation-history -> Conversation log
    POST /voice/speak         -> Trigger TTS
    GET  /voice/status        -> Voice engine status
    GET  /session-stats       -> Per-model usage stats
"""
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from flask import Flask, request, jsonify, send_from_directory, Response
import traceback
import json
import time
import threading
from brain import (
    think, think_stream, check_nvidia, check_openrouter, check_gemini,
    set_mode, get_mode, get_session_stats, CURRENT_MODE,
)
# ── Production wake word system ──────────────────────────────────────────────
from wake.service import get_wake_service as _get_production_wake_service
app = Flask(__name__, static_folder=".", static_url_path="")

# ── CORS support ────────────────────────────────────────────────────────────
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    return response


# ── Conversation History (in-memory, simple) ───────────────────────────────
_conversation_history = []
_history_lock = threading.Lock()
MAX_HISTORY = 100


def _add_to_history(role: str, text: str, model: str = "", tier: str = ""):
    with _history_lock:
        _conversation_history.append({
            "role": role,
            "text": text,
            "model": model,
            "tier": tier,
            "timestamp": time.strftime("%H:%M:%S"),
        })
        if len(_conversation_history) > MAX_HISTORY:
            _conversation_history.pop(0)


# ═══════════════════════════════════════════════════════════════════════════════
#  ORIGINAL ENDPOINTS (preserved exactly)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/health")
def health():
    """Extended health check — checks cloud API key presence."""
    nvidia_ok = check_nvidia()
    openrouter_ok = check_openrouter()
    gemini_ok = check_gemini()
    mode_info = get_mode()

    # Overall status: at least one provider must be configured
    status = "online" if (nvidia_ok or openrouter_ok) else "no_keys"

    # Memory status
    mem_stats = {}
    try:
        from memory import get_memory
        mem = get_memory()
        mem_stats = mem.get_stats()
    except Exception:
        mem_stats = {"error": "memory unavailable"}

    # Voice status
    voice_status = {}
    try:
        from voice import get_voice_engine
        ve = get_voice_engine()
        voice_status = ve.get_status()
    except Exception:
        voice_status = {"error": "voice unavailable"}

    return jsonify({
        "status": status,
        "mode": mode_info["mode"],
        "active_model": mode_info["active_model"],
        "engines": {
            "nvidia_nim": nvidia_ok,
            "openrouter": openrouter_ok,
            "gemini": gemini_ok,
        },
        "memory": mem_stats,
        "voice": voice_status,
        "session_stats": get_session_stats(),
    })


@app.route("/ask", methods=["POST"])
def ask():
    """Full JSON response with memory integration."""
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    try:
        _add_to_history("user", user_message)
        result = think(user_message)
        _add_to_history(
            "jarvis", result.get("response", ""),
            model=result.get("model", ""),
            tier=result.get("tier", ""),
        )
        return jsonify(result)

    except Exception as e:
        traceback.print_exc()
        return jsonify({
            "action": "NONE",
            "error": f"Server error: {str(e)}",
            "response": "Systems encountered an anomaly, sir. Recovering.",
            "model": "none", "fallback": True,
        }), 500


@app.route("/ask-stream", methods=["POST"])
def ask_stream():
    """
    TRUE streaming SSE — tokens arrive as the AI generates them.
    No more fake char-by-char delay. Real token-by-token streaming.
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    _add_to_history("user", user_message)

    def generate():
        full_response = ""
        try:
            for chunk in think_stream(user_message):
                if "token" in chunk:
                    full_response += chunk["token"]
                    payload = json.dumps({"char": chunk["token"]})
                    yield f"data: {payload}\n\n"
                elif "done" in chunk and chunk["done"]:
                    _add_to_history(
                        "jarvis", full_response,
                        model=chunk.get("model", ""),
                        tier=chunk.get("tier", ""),
                    )
                    done_payload = json.dumps({
                        "done": True,
                        "action": "NONE",
                        "model": chunk.get("model", "unknown"),
                        "tier": chunk.get("tier", "unknown"),
                        "fallback": chunk.get("fallback", False),
                        "memory_used": chunk.get("memory_used", False),
                    })
                    yield f"data: {done_payload}\n\n"
        except Exception as e:
            traceback.print_exc()
            error_payload = json.dumps({
                "done": True, "error": str(e),
                "action": "NONE", "model": "none", "fallback": True,
            })
            yield f"data: {error_payload}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/set-mode", methods=["POST"])
def set_mode_endpoint():
    data = request.get_json(silent=True)
    if not data or "mode" not in data:
        return jsonify({"error": "Missing 'mode' field"}), 400
    mode = data["mode"].strip().lower()
    success = set_mode(mode)
    if not success:
        return jsonify({"error": f"Invalid mode: '{mode}'."}), 400
    return jsonify({"success": True, **get_mode()})


@app.route("/get-mode")
def get_mode_endpoint():
    return jsonify(get_mode())


# ═══════════════════════════════════════════════════════════════════════════════
#  NEW ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/system-status")
def system_status():
    """Return system resource usage (CPU, RAM, disk, battery)."""
    try:
        from actions import get_system_status
        status = get_system_status()
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/execute-action", methods=["POST"])
def execute_action_endpoint():
    """Execute a single action by name."""
    data = request.get_json(silent=True)
    if not data or "action" not in data:
        return jsonify({"error": "Missing 'action' field"}), 400

    try:
        from actions import execute_action
        action_name = data["action"]
        args = data.get("args", {})
        result = execute_action(action_name, args)
        return jsonify({"result": result, "action": action_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/execute-plan", methods=["POST"])
def execute_plan_endpoint():
    """Execute a multi-step agent plan."""
    data = request.get_json(silent=True)
    if not data or "steps" not in data:
        return jsonify({"error": "Missing 'steps' field"}), 400

    try:
        from actions import execute_plan, AgentPlan, PlanStep
        steps = [
            PlanStep(action=s["action"], args=s.get("args", {}),
                     delay_ms=s.get("delay_ms", 0))
            for s in data["steps"]
        ]
        plan = AgentPlan(
            plan_id=data.get("plan_id", "custom"),
            label=data.get("label", "Custom Plan"),
            steps=steps,
        )
        result = execute_plan(plan)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/episodes")
def memory_episodes():
    """Return recent memory episodes."""
    try:
        from memory import get_memory
        mem = get_memory()
        count = request.args.get("count", 20, type=int)
        episodes = mem.get_recent_episodes(count)
        # Strip embeddings for response size
        clean = []
        for ep in episodes:
            e = dict(ep)
            e.pop("embedding", None)
            clean.append(e)
        return jsonify({"episodes": clean})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/semantic")
def memory_semantic():
    """Return semantic memory (user profile)."""
    try:
        from memory import get_memory
        mem = get_memory()
        return jsonify(mem.get_semantic())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/procedure", methods=["POST"])
def memory_procedure():
    """Store a named procedure."""
    data = request.get_json(silent=True)
    if not data or "name" not in data or "steps" not in data:
        return jsonify({"error": "Missing 'name' or 'steps'"}), 400

    try:
        from memory import get_memory
        mem = get_memory()
        mem.add_procedure(
            data["name"],
            data["steps"],
            triggers=data.get("triggers", [data["name"].replace("_", " ")]),
        )
        return jsonify({"success": True, "procedure": data["name"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/conversation-history")
def conversation_history():
    """Return conversation log."""
    count = request.args.get("count", 50, type=int)
    with _history_lock:
        return jsonify({"history": _conversation_history[-count:]})


@app.route("/voice/speak", methods=["POST"])
def voice_speak():
    """Trigger TTS for given text."""
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Missing 'text' field"}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "Empty text"}), 400
    try:
        from voice import get_voice_engine
        engine = get_voice_engine()
        tts = engine._tts
        # Report init errors back to the frontend immediately
        if hasattr(tts, "_init_error") and tts._init_error:
            return jsonify({"error": f"TTS init failed: {tts._init_error}"}), 500
        if hasattr(tts, "_ready") and not tts._ready.is_set():
            return jsonify({"error": "TTS engine still initialising — try again in a moment"}), 503
        engine.speak_async(text)
        return jsonify({"success": True, "chars": len(text)})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/voice/test")
def voice_test():
    """Quick smoke-test: speaks a fixed phrase. Useful for diagnostics."""
    try:
        from voice import get_voice_engine
        engine = get_voice_engine()
        tts = engine._tts
        if hasattr(tts, "_init_error") and tts._init_error:
            return jsonify({"ok": False, "error": tts._init_error})
        engine.speak_async("JARVIS voice systems online, sir.")
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)})


@app.route("/voice/status")
def voice_status():
    """Return voice engine status."""
    try:
        from voice import get_voice_engine
        engine = get_voice_engine()
        return jsonify(engine.get_status())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/session-stats")
def session_stats():
    """Return per-model usage statistics."""
    return jsonify(get_session_stats())


@app.route("/voice/wake-stream")
def voice_wake_stream():
    """SSE stream — production wake word events."""
    def generate():
        try:
            svc = _get_production_wake_service()
            if not svc.is_enabled():
                yield f"data: {json.dumps({'event': 'disabled'})}\n\n"
                return
            for event in svc.subscribe():
                yield f"data: {json.dumps(event)}\n\n"
        except GeneratorExit:
            pass
        except Exception as e:
            yield f"data: {json.dumps({'event': 'error', 'message': str(e)})}\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/voice/wake-enable", methods=["POST"])
def voice_wake_enable():
    """Enable wake word detection (production service)."""
    try:
        svc = _get_production_wake_service()
        svc.enable()
        return jsonify({"success": True, "wake_word": True, "status": svc.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voice/wake-disable", methods=["POST"])
def voice_wake_disable():
    """Disable wake word detection."""
    try:
        svc = _get_production_wake_service()
        svc.disable()
        return jsonify({"success": True, "wake_word": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/voice/wake-status")
def voice_wake_status():
    """Return wake word service status."""
    try:
        svc = _get_production_wake_service()
        return jsonify({"enabled": svc.is_enabled(), **svc.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ═══════════════════════════════════════════════════════════════════════════════
#  STARTUP
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    mode_info = get_mode()

    print("\n  +==========================================+")
    print("  |   J.A.R.V.I.S  -  Cognitive AI Engine   |")
    print("  +==========================================+")
    print(f"  |   Mode:    {mode_info['mode'].upper():<30s}|")
    print(f"  |   Model:   {mode_info['active_model']:<30s}|")
    print("  |   Port:    5000                          |")
    print("  |   Stack:   100% FREE (Cloud APIs)        |")
    print("  +==========================================+")
    print("  |   Models: NVIDIA NIM + OpenRouter :free  |")
    print("  |   Memory: Episodic + Semantic + Vector   |")
    print("  |   Voice:  Whisper + pyttsx3              |")
    print("  |   Actions: 13 system commands            |")
    print("  +==========================================+\n")
    print("  -> Open http://localhost:5000")
    print("  -> Set: NVIDIA_API_KEY + OPENROUTER_API_KEY in .env")
    print("  -> No Ollama required!\n")

    # ── Warm-up: pre-heat the reflex endpoint so first query is fast ───
    def _warmup():
        """Send a silent dummy request on startup to pre-warm the API endpoint."""
        import time as _time
        _time.sleep(3)  # wait for server to fully start
        try:
            from brain import _call_model_stream
            # Silent warm-up — result is discarded
            list(_call_model_stream("reflex", [{"role": "user", "content": "ping"}]))
            print("  [INFO] Warm-up complete — reflex tier ready")
        except Exception as e:
            print(f"  [INFO] Warm-up skipped: {e}")

    threading.Thread(target=_warmup, daemon=True).start()

    app.run(host="0.0.0.0", port=5000, debug=False)