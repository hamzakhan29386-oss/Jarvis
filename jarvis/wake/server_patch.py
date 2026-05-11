"""
server_wake_patch.py — Drop-in patch for server.py
====================================================
Replace the three wake word endpoint blocks in your existing server.py
with the versions below. The only change is importing from wake.service
instead of voice.py's get_wake_service.

CHANGES NEEDED IN server.py:
=============================================================

1. Add this import near the top (after existing imports):

    from wake.service import get_wake_service as _get_production_wake_service

2. Replace the four wake endpoints with the versions below.

3. Add a new /voice/wake-status and /voice/speaker-status endpoint.

=============================================================
"""

# ── PASTE THESE FOUR ROUTES INTO server.py ────────────────────────────────────

# -- ROUTE 1 ------------------------------------------------------------------
# Replace the existing /voice/wake-stream with this:

"""
@app.route("/voice/wake-stream")
def voice_wake_stream():
    \"\"\"SSE stream — production wake word events.\"\"\"
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
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )
"""

# -- ROUTE 2 ------------------------------------------------------------------
# Replace /voice/wake-enable with this:

"""
@app.route("/voice/wake-enable", methods=["POST"])
def voice_wake_enable():
    \"\"\"Enable wake word detection (production service).\"\"\"
    try:
        svc = _get_production_wake_service()
        svc.enable()
        return jsonify({"success": True, "wake_word": True, "status": svc.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""

# -- ROUTE 3 ------------------------------------------------------------------
# Replace /voice/wake-disable with this:

"""
@app.route("/voice/wake-disable", methods=["POST"])
def voice_wake_disable():
    \"\"\"Disable wake word detection.\"\"\"
    try:
        svc = _get_production_wake_service()
        svc.disable()
        return jsonify({"success": True, "wake_word": False})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""

# -- ROUTE 4 ------------------------------------------------------------------
# Replace /voice/wake-status with this:

"""
@app.route("/voice/wake-status")
def voice_wake_status():
    \"\"\"Return wake word service status.\"\"\"
    try:
        svc = _get_production_wake_service()
        return jsonify({"enabled": svc.is_enabled(), **svc.get_status()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""

# -- ROUTE 5 (NEW) ------------------------------------------------------------
# Add this new endpoint for speaker enrollment:

"""
@app.route("/voice/speaker-enroll", methods=["POST"])
def voice_speaker_enroll():
    \"\"\"Trigger speaker enrollment. Blocks until complete.\"\"\"
    try:
        svc = _get_production_wake_service()
        ok = svc.enroll_speaker()
        return jsonify({"success": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""

# -- ROUTE 6 (NEW) ------------------------------------------------------------
# Threshold tuning endpoint:

"""
@app.route("/voice/wake-threshold", methods=["POST"])
def voice_wake_threshold():
    \"\"\"Adjust detection threshold at runtime.\"\"\"
    data = request.get_json(silent=True) or {}
    threshold = data.get("threshold")
    if threshold is None:
        return jsonify({"error": "Missing 'threshold'"}), 400
    try:
        svc = _get_production_wake_service()
        svc.set_threshold(float(threshold))
        return jsonify({"success": True, "threshold": float(threshold)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
"""
