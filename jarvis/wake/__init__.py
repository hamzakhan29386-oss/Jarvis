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

from .service import ProductionWakeService, get_wake_service
from .detector import WakeWordEngine
from .speaker import SpeakerVerifier
from .transcriber import PostWakeTranscriber
from .noise import NoisePipeline

__all__ = [
    "ProductionWakeService",
    "get_wake_service",
    "WakeWordEngine",
    "SpeakerVerifier",
    "PostWakeTranscriber",
    "NoisePipeline",
]
