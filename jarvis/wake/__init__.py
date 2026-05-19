"""
wake/ — JARVIS Production Wake Word System
===========================================
Drop this folder into your jarvis/ directory.

Quick start:
    from wake.service import get_wake_service

    svc = get_wake_service()
    svc.enable()

    # In your Flask SSE endpoint:
    for event in svc.subscribe():
        yield f"data: {json.dumps(event)}\n\n"

Training:
    python -m wake.train all

Speaker enrollment:
    python -m wake.enroll
"""

__all__ = [
    "ProductionWakeService",
    "get_wake_service",
    "WakeWordEngine",
    "SpeakerVerifier",
    "PostWakeTranscriber",
    "NoisePipeline",
]


_EXPORTS = {
    "ProductionWakeService": (".service", "ProductionWakeService"),
    "get_wake_service": (".service", "get_wake_service"),
    "WakeWordEngine": (".detector", "WakeWordEngine"),
    "SpeakerVerifier": (".speaker", "SpeakerVerifier"),
    "PostWakeTranscriber": (".transcriber", "PostWakeTranscriber"),
    "NoisePipeline": (".noise", "NoisePipeline"),
}


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    from importlib import import_module

    module_name, attr_name = _EXPORTS[name]
    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
