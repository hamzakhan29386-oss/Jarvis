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
import queue
import time
import threading
from core.paths import resource_path
from core.assistant_state import get_assistant_state
from core.metrics import get_metrics
from brain import (
    think, think_stream, check_nvidia, check_openrouter, check_gemini,
    check_ollama, set_mode, get_mode, get_session_stats, CURRENT_MODE,
)
# ── Production wake word system ──────────────────────────────────────────────
from wake.service import get_wake_service as _get_production_wake_service
WEB_ROOT = resource_path(".")
app = Flask(__name__, static_folder=str(WEB_ROOT), static_url_path="")

try:
    from event_bus import emit, get_event_bus
    from world_state import get_world_state
    from attention_manager import get_attention_manager
    from agent_loop import get_agent_loop, AgentJob
    from operating_modes import get_operating_mode, set_operating_mode
    from goal_planner import get_goal_planner
    from tool_registry import get_tool_registry
    from skills.skill_manager import get_skill_manager
    from self_optimizer import get_self_optimizer
    from execution_replay import get_execution_replay
except Exception:
    emit = None

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
    try:
        if emit:
            emit(
                "conversation_updated",
                {"role": role, "text": text[:500], "model": model, "tier": tier},
                source="server",
            )
    except Exception:
        pass


# ═══════════════════════════════════════════════════════════════════════════════
def _set_assistant_state(state: str, reason: str, **metadata):
    try:
        return get_assistant_state().set_state(
            state,
            reason=reason,
            source="server",
            metadata=metadata,
        )
    except Exception:
        return None


def _metric(name: str, amount: int = 1):
    try:
        get_metrics().increment(name, amount)
    except Exception:
        pass


#  ORIGINAL ENDPOINTS (preserved exactly)
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return send_from_directory(str(WEB_ROOT), "index.html")


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

    cognitive_status = {}
    try:
        cognitive_status = {
            "world": get_world_state().get_state(),
            "attention": get_attention_manager().status(),
            "agent_loop": get_agent_loop().status(),
            "operating_mode": get_operating_mode(),
        }
    except Exception as e:
        cognitive_status = {"error": str(e)}

    return jsonify({
        "status": status,
        "mode": mode_info["mode"],
        "active_model": mode_info["active_model"],
        "engines": {
            "nvidia_nim": nvidia_ok,
            "openrouter": openrouter_ok,
            "gemini": gemini_ok,
            "ollama_local": check_ollama(),
        },
        "memory": mem_stats,
        "voice": voice_status,
        "session_stats": get_session_stats(),
        "assistant_state": get_assistant_state().get_state(),
        "cognitive": cognitive_status,
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
        _metric("requests.ask")
        _set_assistant_state("thinking", "ask", message_preview=user_message[:120])
        if emit:
            emit("user_prompt_received", {"message": user_message[:500]}, source="server")
        _add_to_history("user", user_message)
        result = think(user_message)
        _add_to_history(
            "jarvis", result.get("response", ""),
            model=result.get("model", ""),
            tier=result.get("tier", ""),
        )
        _set_assistant_state("idle", "ask_complete", model=result.get("model", ""))
        return jsonify(result)

    except Exception as e:
        _metric("requests.ask.errors")
        _set_assistant_state("recovering", "ask_error", error=str(e))
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

    _metric("requests.ask_stream")
    _set_assistant_state("thinking", "ask_stream", message_preview=user_message[:120])
    _add_to_history("user", user_message)
    try:
        if emit:
            emit("user_prompt_received", {"message": user_message[:500], "streaming": True}, source="server")
    except Exception:
        pass

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
                    _set_assistant_state("idle", "ask_stream_complete", model=chunk.get("model", ""))
                    yield f"data: {done_payload}\n\n"
        except Exception as e:
            _metric("requests.ask_stream.errors")
            _set_assistant_state("recovering", "ask_stream_error", error=str(e))
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
        from agents.closed_loop import get_closed_loop_executor

        action_name = data["action"]
        args = data.get("args", {})
        result = get_closed_loop_executor().run_action(action_name, args)
        return jsonify({"result": result.get("result"), "action": action_name, "execution": result})
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


@app.route("/intent/parse", methods=["POST"])
def parse_intent_endpoint():
    """Parse raw user text into a structured desktop intent."""
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    try:
        from core.intent_parser import parse_user_intent

        return jsonify(parse_user_intent(data["message"]).to_dict())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/assistant/command", methods=["POST"])
def assistant_command_endpoint():
    """Route a command through intent parsing, actions, and brain fallback."""
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400
    try:
        from core.task_router import route_text

        result = route_text(data["message"], speak=bool(data.get("speak", False)))
        _add_to_history("user", data["message"])
        _add_to_history("jarvis", result.get("response", ""), tier=result.get("tier", "router"))
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/runtime/status")
def runtime_status_endpoint():
    """Return status for the persistent desktop runtime when active."""
    try:
        from core.assistant_runtime import get_runtime

        return jsonify(get_runtime().status())
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


@app.route("/memory/consolidate", methods=["POST"])
def memory_consolidate_endpoint():
    try:
        from memory import get_memory

        data = request.get_json(silent=True) or {}
        return jsonify(get_memory().consolidate(limit=int(data.get("limit", 50))))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/memory/prune", methods=["POST"])
def memory_prune_endpoint():
    try:
        from memory import get_memory

        data = request.get_json(silent=True) or {}
        return jsonify(get_memory().prune_memories(
            retention_days=int(data.get("retention_days", 365)),
            min_importance=float(data.get("min_importance", 0.2)),
        ))
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
        from voice.voice_manager import get_voice_manager

        engine = get_voice_engine()
        return jsonify({
            **engine.get_status(),
            "native_runtime": get_voice_manager().get_status(),
        })
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
@app.route("/cognitive/status")
def cognitive_status_endpoint():
    try:
        return jsonify({
            "world": get_world_state().get_state(),
            "attention": get_attention_manager().status(),
            "agent_loop": get_agent_loop().status(),
            "operating_mode": get_operating_mode(),
            "events": get_event_bus().history(50),
            "goals": get_goal_planner().list_goals(),
            "optimizer": get_self_optimizer().recommend_tools(),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/world-state", methods=["GET", "POST"])
def world_state_endpoint():
    try:
        ws = get_world_state()
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            return jsonify(ws.update_state(data.get("updates", data), source="api"))
        return jsonify(ws.get_state())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/goals", methods=["GET", "POST", "DELETE"])
def goals_endpoint():
    try:
        planner = get_goal_planner()
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            goal = planner.create_goal(
                data.get("title", "Untitled mission"),
                data.get("subtasks", []),
                int(data.get("priority", 5)),
            )
            get_world_state().set_goal({"id": goal.id, "title": goal.title, "priority": goal.priority})
            return jsonify(planner.goal_to_dict(goal))
        if request.method == "DELETE":
            goal_id = (request.get_json(silent=True) or {}).get("id", "")
            return jsonify({"cleared": get_world_state().clear_goal(goal_id)})
        return jsonify(planner.list_goals())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/operating-mode", methods=["GET", "POST"])
def operating_mode_endpoint():
    try:
        if request.method == "POST":
            data = request.get_json(silent=True) or {}
            return jsonify(set_operating_mode(data.get("mode", "ASSIST")))
        return jsonify(get_operating_mode())
    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/autonomous/start", methods=["POST"])
def autonomous_start():
    return jsonify(get_agent_loop().start())


@app.route("/autonomous/pause", methods=["POST"])
def autonomous_pause():
    return jsonify(get_agent_loop().pause())


@app.route("/autonomous/resume", methods=["POST"])
def autonomous_resume():
    return jsonify(get_agent_loop().resume())


@app.route("/autonomous/stop", methods=["POST"])
def autonomous_stop():
    return jsonify(get_agent_loop().stop())


@app.route("/autonomous/enqueue", methods=["POST"])
def autonomous_enqueue():
    data = request.get_json(silent=True) or {}
    job = AgentJob(kind=data.get("kind", "tool"), payload=data.get("payload", {}), priority=int(data.get("priority", 5)))
    get_agent_loop().enqueue(job)
    return jsonify({"queued": True, "job": job.__dict__})


@app.route("/tools")
def tools_endpoint():
    return jsonify(get_tool_registry().list_tools())


@app.route("/skills")
def skills_endpoint():
    return jsonify(get_skill_manager().list_skills())


@app.route("/events")
def events_endpoint():
    return jsonify(get_event_bus().history(int(request.args.get("limit", 100))))


@app.route("/execution-replay")
def execution_replay_endpoint():
    return jsonify(get_execution_replay().recent(int(request.args.get("limit", 50))))


@app.route("/assistant/state")
def assistant_state_endpoint():
    return jsonify(get_assistant_state().get_state())


@app.route("/metrics")
def metrics_endpoint():
    payload = get_metrics().snapshot()
    payload["assistant_state"] = get_assistant_state().get_state()
    payload["session_stats"] = get_session_stats()
    return jsonify(payload)


@app.route("/cognition/router")
def cognition_router_endpoint():
    from cognition.router import get_cognitive_router

    return jsonify(get_cognitive_router().snapshot())


@app.route("/voice/interrupt", methods=["POST"])
def voice_interrupt_endpoint():
    errors = []
    try:
        from voice import get_voice_engine

        engine = get_voice_engine()
        if hasattr(engine, "stop_speaking"):
            engine.stop_speaking()
        elif hasattr(engine, "stop"):
            engine.stop()
    except Exception as e:
        errors.append(str(e))

    _metric("voice.interrupt")
    state = _set_assistant_state("interrupted", "voice_interrupt", errors=errors)
    return jsonify({"ok": not errors, "errors": errors, "state": state})


@app.route("/voice/realtime-ask", methods=["POST"])
def voice_realtime_ask():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field"}), 400

    user_message = data["message"].strip()
    if not user_message:
        return jsonify({"error": "Empty message"}), 400

    tts_enabled = data.get("tts", True)
    if isinstance(tts_enabled, str):
        tts_enabled = tts_enabled.strip().lower() not in {"0", "false", "no", "off"}

    _metric("voice.realtime_ask")
    _set_assistant_state("thinking", "voice_realtime_ask", message_preview=user_message[:120])
    _add_to_history("user", user_message)

    def _mark_idle_when_speech_finishes(speaker):
        try:
            speaker.wait_done(timeout=90.0)
        finally:
            _set_assistant_state("idle", "voice_realtime_complete")

    def generate():
        full_response = ""
        speaker = None

        if tts_enabled:
            try:
                from voice import StreamingSpeaker, get_voice_engine

                engine = get_voice_engine()

                def _on_sentence(sentence: str):
                    _metric("voice.sentences_spoken")
                    _set_assistant_state("speaking", "tts_sentence", sentence_preview=sentence[:120])

                speaker = StreamingSpeaker(
                    getattr(engine, "_tts", engine),
                    on_sentence=_on_sentence,
                )
            except Exception as e:
                _metric("voice.tts_unavailable")
                app.logger.warning("[realtime-ask] TTS unavailable: %s", e)

        try:
            for chunk in think_stream(user_message):
                if "token" in chunk:
                    token = chunk["token"]
                    full_response += token
                    if speaker is not None:
                        speaker.feed_token(token)
                    yield f"data: {json.dumps({'char': token})}\n\n"

                elif chunk.get("done"):
                    if speaker is not None:
                        speaker.flush()
                        threading.Thread(
                            target=_mark_idle_when_speech_finishes,
                            args=(speaker,),
                            daemon=True,
                            name="JARVISRealtimeTTSIdleMarker",
                        ).start()
                    else:
                        _set_assistant_state("idle", "voice_realtime_complete")

                    _add_to_history(
                        "jarvis",
                        full_response,
                        model=chunk.get("model", ""),
                        tier=chunk.get("tier", ""),
                    )
                    done_payload = json.dumps({
                        "done": True,
                        "action": "NONE",
                        "model": chunk.get("model", "unknown"),
                        "tier": chunk.get("tier", "unknown"),
                        "provider": chunk.get("provider", "unknown"),
                        "fallback": chunk.get("fallback", False),
                        "memory_used": chunk.get("memory_used", False),
                        "tts_active": speaker is not None,
                    })
                    yield f"data: {done_payload}\n\n"
        except Exception as e:
            _metric("voice.realtime_ask.errors")
            _set_assistant_state("recovering", "voice_realtime_error", error=str(e))
            traceback.print_exc()
            if speaker is not None:
                try:
                    speaker.stop()
                except Exception:
                    pass
            error_payload = json.dumps({
                "done": True,
                "error": str(e),
                "action": "NONE",
                "model": "none",
                "fallback": True,
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


@app.route("/voice/tts-engine", methods=["POST", "GET"])
def voice_tts_engine():
    try:
        from voice import get_voice_engine

        engine = get_voice_engine()
        if request.method == "GET":
            status = engine.get_status()
            return jsonify({
                "tts_engine": status.get("tts_engine", "unknown"),
                "is_speaking": status.get("is_speaking", False),
            })

        data = request.get_json(silent=True) or {}
        new_engine = data.get("engine", "pyttsx3")
        if new_engine not in ("pyttsx3", "piper", "coqui"):
            return jsonify({"error": f"Unknown engine: {new_engine}"}), 400

        engine.switch_tts(new_engine)
        _metric("voice.tts_engine.switch")
        return jsonify({"success": True, "engine": new_engine})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ws/events")
def ws_events_endpoint():
    """SSE-compatible realtime event stream at the planned WebSocket path."""
    bus = get_event_bus()
    event_queue: queue.Queue = queue.Queue(maxsize=256)

    def _push(event):
        try:
            event_queue.put_nowait(event.to_dict())
        except queue.Full:
            try:
                event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                event_queue.put_nowait(event.to_dict())
            except queue.Full:
                pass

    token = bus.subscribe("*", _push)

    def generate():
        yield f"data: {json.dumps({'event': 'connected', 'transport': 'sse', 'path': '/ws/events'})}\n\n"
        try:
            while True:
                try:
                    event = event_queue.get(timeout=20.0)
                    yield f"data: {json.dumps(event)}\n\n"
                except queue.Empty:
                    yield f"data: {json.dumps({'event': 'heartbeat', 'ts': time.time()})}\n\n"
        finally:
            bus.unsubscribe(token)

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.route("/context/status")
def context_status_endpoint():
    from core.contextual_cowork import get_contextual_cowork_service

    return jsonify(get_contextual_cowork_service().status())


@app.route("/context/screen", methods=["GET", "POST"])
def context_screen_endpoint():
    from core.contextual_cowork import get_contextual_cowork_service

    data = request.get_json(silent=True) or {}
    save = str(data.get("save", request.args.get("save", "false"))).lower() in {"1", "true", "yes", "on"}
    return jsonify(get_contextual_cowork_service().capture_screen(save=save))


@app.route("/context/explain", methods=["POST"])
def context_explain_endpoint():
    from core.contextual_cowork import get_contextual_cowork_service

    data = request.get_json(silent=True) or {}
    include_ocr = bool(data.get("ocr", False))
    save = bool(data.get("save_screenshot", False))
    return jsonify(get_contextual_cowork_service().extract_context(include_ocr=include_ocr, save_screenshot=save))


@app.route("/world/state", methods=["GET", "POST"])
def world_state_alias_endpoint():
    ws = get_world_state()
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        return jsonify(ws.update_state(data.get("updates", data), source="api"))
    return jsonify(ws.get_state())


@app.route("/world/refresh", methods=["POST"])
def world_refresh_endpoint():
    return jsonify(get_world_state().refresh_environment())


@app.route("/world/resume")
def world_resume_endpoint():
    ws = get_world_state()
    state = ws.get_state()
    return jsonify({
        "active_goals": state.get("active_goals", []),
        "current_task": state.get("current_task"),
        "active_coding_project": state.get("active_coding_project"),
        "current_workspace": state.get("current_workspace"),
        "recent_actions": state.get("recent_actions", [])[-10:],
        "ongoing_workflows": state.get("ongoing_workflows", []),
        "operating_context": ws.operating_context(),
    })


@app.route("/optimization/status")
def optimization_status_endpoint():
    return jsonify(get_self_optimizer().status())


@app.route("/optimization/recommendations")
def optimization_recommendations_endpoint():
    return jsonify(get_self_optimizer().recommend_tools())


@app.route("/presence/status")
def presence_status_endpoint():
    from core.presence import get_presence_manager

    return jsonify(get_presence_manager().status())


@app.route("/presence/profile", methods=["POST"])
def presence_profile_endpoint():
    from core.presence import get_presence_manager

    data = request.get_json(silent=True) or {}
    return jsonify(get_presence_manager().set_profile(
        mode=data.get("mode"),
        voice_profile=data.get("voice_profile"),
    ))


@app.route("/presence/ack", methods=["POST"])
def presence_ack_endpoint():
    from core.presence import get_presence_manager

    data = request.get_json(silent=True) or {}
    return jsonify(get_presence_manager().acknowledge(
        state=data.get("state", get_assistant_state().get_state().get("state", "idle")),
        force=bool(data.get("force", False)),
    ))


@app.route("/runtime/tasks", methods=["GET", "POST"])
def runtime_tasks_endpoint():
    from core.long_task_runtime import get_long_task_runtime

    runtime = get_long_task_runtime()
    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        objective = (data.get("objective") or "").strip()
        if not objective:
            return jsonify({"error": "Missing 'objective' field"}), 400
        return jsonify(runtime.create_task(
            objective,
            constraints=data.get("constraints") or [],
            plan=data.get("plan") or None,
            requires_confirmation=bool(data.get("requires_confirmation", True)),
        ))
    return jsonify({"tasks": runtime.list_tasks()})


@app.route("/runtime/tasks/<task_id>/pause", methods=["POST"])
def runtime_task_pause_endpoint(task_id):
    from core.long_task_runtime import get_long_task_runtime

    return jsonify(get_long_task_runtime().update_status(task_id, "paused"))


@app.route("/runtime/tasks/<task_id>/resume", methods=["POST"])
def runtime_task_resume_endpoint(task_id):
    from core.long_task_runtime import get_long_task_runtime

    return jsonify(get_long_task_runtime().update_status(task_id, "active"))


@app.route("/runtime/tasks/<task_id>/cancel", methods=["POST"])
def runtime_task_cancel_endpoint(task_id):
    from core.long_task_runtime import get_long_task_runtime

    return jsonify(get_long_task_runtime().update_status(task_id, "cancelled"))


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
