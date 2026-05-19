"""Lifecycle manager for the native JARVIS voice runtime."""

from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Iterator

from .stt import NativeSTT
from .wake_listener import NativeWakeListener

log = logging.getLogger("jarvis.voice.manager")


def _evt(name: str, **kwargs) -> dict:
    return {"event": name, "ts": time.time(), **kwargs}


class VoiceManager:
    """Owns wake listening, post-wake STT, pub/sub, and lifecycle state."""

    def __init__(self):
        self._lock = threading.RLock()
        self._enabled = False
        self._in_command_capture = False
        self._restart_count = 0
        self._subscribers: list[queue.Queue] = []
        self._sub_lock = threading.Lock()
        self._listener: NativeWakeListener | None = None
        self._stt: NativeSTT | None = None
        self._command_thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._enabled:
                return
            self._ensure_components()
            self._listener.on_detected = self._handle_wake
            self._listener.on_error = self._handle_error
            self._listener.start()
            self._enabled = True
            self._publish(_evt("enabled", status=self.get_status()))
            log.info("[VoiceManager] Started")

    def stop(self) -> None:
        with self._lock:
            self._enabled = False
            listener = self._listener
        if listener:
            listener.stop()
        self._publish(_evt("disabled"))
        log.info("[VoiceManager] Stopped")

    def restart(self) -> None:
        self.stop()
        self._restart_count += 1
        time.sleep(0.25)
        self.start()

    def get_status(self) -> dict:
        return {
            "enabled": self._enabled,
            "in_command_capture": self._in_command_capture,
            "restart_count": self._restart_count,
            "listener": self._listener.get_status() if self._listener else None,
            "stt": self._stt.get_status() if self._stt else None,
        }

    def is_enabled(self) -> bool:
        return self._enabled

    def subscribe(self, timeout_s: float = 30.0) -> Iterator[dict]:
        q: queue.Queue = queue.Queue(maxsize=64)
        with self._sub_lock:
            self._subscribers.append(q)
        try:
            while True:
                try:
                    yield q.get(timeout=timeout_s)
                except queue.Empty:
                    yield _evt("heartbeat", status=self.get_status())
        finally:
            with self._sub_lock:
                try:
                    self._subscribers.remove(q)
                except ValueError:
                    pass

    def set_threshold(self, threshold: float) -> None:
        self._ensure_components()
        self._listener.detector.set_threshold(threshold)

    def enroll_speaker(self):
        self._ensure_components()
        return self._listener.speaker.enroll_from_mic()

    def _ensure_components(self) -> None:
        if self._listener is None:
            self._listener = NativeWakeListener()
        if self._stt is None:
            self._stt = NativeSTT()

    def _handle_wake(self, data: dict) -> None:
        self._publish(_evt("detected", **data))
        if self._command_thread and self._command_thread.is_alive():
            log.info("[VoiceManager] Wake ignored while command capture is active")
            return
        self._command_thread = threading.Thread(target=self._capture_command, daemon=True, name="JARVISCommandCapture")
        self._command_thread.start()

    def _capture_command(self) -> None:
        self._in_command_capture = True
        self._publish(_evt("listening"))
        listener = self._listener
        if listener:
            listener.pause()
        try:
            text = self._stt.capture_and_transcribe(status_callback=lambda e: self._publish(_evt(e)))
        finally:
            self._in_command_capture = False
            if self._enabled and listener:
                try:
                    listener.resume()
                except Exception as exc:
                    self._handle_error(exc)

        if text and text.strip():
            self._publish(_evt("transcript", text=text.strip()))
        else:
            self._publish(_evt("error", message="Didn't catch that - please try again"))

    def _handle_error(self, exc: Exception) -> None:
        log.error("[VoiceManager] Error: %s", exc)
        self._publish(_evt("error", message=str(exc)))

    def _publish(self, event: dict) -> None:
        with self._sub_lock:
            for q in list(self._subscribers):
                try:
                    q.put_nowait(event)
                except queue.Full:
                    log.debug("[VoiceManager] Dropping event for slow subscriber: %s", event.get("event"))


_instance: VoiceManager | None = None
_instance_lock = threading.Lock()


def get_voice_manager() -> VoiceManager:
    global _instance
    if _instance is None:
        with _instance_lock:
            if _instance is None:
                _instance = VoiceManager()
    return _instance
