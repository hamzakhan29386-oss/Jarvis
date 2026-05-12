"""Always-running desktop runtime for JARVIS."""

from __future__ import annotations

import logging
import threading
import time
import webbrowser
from dataclasses import dataclass

from core.background_threads import ThreadSupervisor
from core.task_router import get_task_router

log = logging.getLogger("jarvis.core.runtime")


@dataclass
class RuntimeState:
    started_at: float
    server_started: bool = False
    wake_enabled: bool = False
    voice_muted: bool = False
    last_transcript: str = ""
    last_response: str = ""


class AssistantRuntime:
    """Owns Flask, wake-word handling, task routing, and runtime status."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5000):
        self.host = host
        self.port = port
        self.state = RuntimeState(started_at=time.time())
        self.supervisor = ThreadSupervisor()
        self._stop = threading.Event()
        self._server_thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start(self, enable_wake: bool = True, open_ui: bool = False) -> None:
        self._stop.clear()
        self._start_server()
        self.supervisor.add("JARVISWakeCommandConsumer", self._wake_event_consumer, restart=True)
        self.supervisor.start_watchdog()
        if enable_wake:
            self.enable_wake_word()
        if open_ui:
            self.open_ui()

    def stop(self) -> None:
        self._stop.set()
        self.disable_wake_word()
        self.supervisor.stop()

    def restart(self) -> None:
        wake_enabled = self.state.wake_enabled
        self.stop()
        time.sleep(1.0)
        self.supervisor = ThreadSupervisor()
        self.start(enable_wake=wake_enabled)

    def open_ui(self) -> None:
        webbrowser.open(self.url)

    def enable_wake_word(self) -> None:
        from wake.service import get_wake_service

        get_wake_service().enable()
        self.state.wake_enabled = True

    def disable_wake_word(self) -> None:
        try:
            from wake.service import get_wake_service

            get_wake_service().disable()
        finally:
            self.state.wake_enabled = False

    def toggle_wake_word(self) -> bool:
        if self.state.wake_enabled:
            self.disable_wake_word()
        else:
            self.enable_wake_word()
        return self.state.wake_enabled

    def set_voice_muted(self, muted: bool) -> None:
        self.state.voice_muted = muted

    def status(self) -> dict:
        return {
            "url": self.url,
            "uptime_s": int(time.time() - self.state.started_at),
            "server_started": self.state.server_started,
            "wake_enabled": self.state.wake_enabled,
            "voice_muted": self.state.voice_muted,
            "last_transcript": self.state.last_transcript,
            "last_response": self.state.last_response,
            "threads": self.supervisor.status(),
        }

    def _start_server(self) -> None:
        if self._server_thread and self._server_thread.is_alive():
            return

        def run_server() -> None:
            from server import app

            self.state.server_started = True
            app.run(host=self.host, port=self.port, debug=False, use_reloader=False, threaded=True)

        self._server_thread = threading.Thread(target=run_server, daemon=True, name="JARVISFlaskServer")
        self._server_thread.start()

    def _wake_event_consumer(self) -> None:
        from wake.service import get_wake_service

        service = get_wake_service()
        for event in service.subscribe():
            if self._stop.is_set():
                return
            if event.get("event") != "transcript":
                continue
            text = (event.get("text") or "").strip()
            if not text:
                continue
            self.state.last_transcript = text
            result = get_task_router().route(text, speak=not self.state.voice_muted)
            self.state.last_response = result.response


_runtime: AssistantRuntime | None = None


def get_runtime() -> AssistantRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AssistantRuntime()
    return _runtime
