"""Single-owner microphone fanout for wake, VAD, STT, interrupts, and telemetry."""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field

from .audio_input import AudioFrame, AudioInputConfig, NativeAudioInput

log = logging.getLogger("jarvis.voice.audio_multiplexer")


@dataclass
class AudioSubscriber:
    name: str
    queue: "queue.Queue[AudioFrame]"
    created_at: float = field(default_factory=time.time)
    frames_received: int = 0
    frames_dropped: int = 0

    def read(self, timeout: float = 0.5) -> AudioFrame | None:
        try:
            frame = self.queue.get(timeout=timeout)
            self.queue.task_done()
            return frame
        except queue.Empty:
            return None

    def status(self) -> dict:
        return {
            "name": self.name,
            "queue_size": self.queue.qsize(),
            "queue_max": self.queue.maxsize,
            "frames_received": self.frames_received,
            "frames_dropped": self.frames_dropped,
            "age_s": round(time.time() - self.created_at, 3),
        }


class AudioMultiplexer:
    """Owns the microphone exactly once and fans frames out to named queues."""

    def __init__(self, config: AudioInputConfig | None = None):
        self.input = NativeAudioInput(config or AudioInputConfig())
        self._lock = threading.RLock()
        self._subscribers: dict[str, AudioSubscriber] = {}
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._running = False
        self._frames_dispatched = 0

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            self._stop.clear()
            self.input.start()
            self._thread = threading.Thread(target=self._fanout_loop, daemon=True, name="JARVISAudioMultiplexer")
            self._thread.start()
            self._running = True
        log.info("[AudioMultiplexer] Started")

    def stop(self) -> None:
        with self._lock:
            self._running = False
            self._stop.set()
            thread = self._thread
            self._thread = None
        self.input.stop()
        if thread and thread.is_alive():
            thread.join(timeout=2.0)
        log.info("[AudioMultiplexer] Stopped")

    def is_running(self) -> bool:
        with self._lock:
            return self._running

    def subscribe(self, name: str, *, maxsize: int = 128) -> AudioSubscriber:
        with self._lock:
            token = f"{name}:{uuid.uuid4().hex[:8]}"
            subscriber = AudioSubscriber(token, queue.Queue(maxsize=maxsize))
            self._subscribers[token] = subscriber
            return subscriber

    def unsubscribe(self, subscriber: AudioSubscriber | str | None) -> None:
        if subscriber is None:
            return
        name = subscriber if isinstance(subscriber, str) else subscriber.name
        with self._lock:
            self._subscribers.pop(name, None)

    def get_status(self) -> dict:
        with self._lock:
            subscribers = [subscriber.status() for subscriber in self._subscribers.values()]
        return {
            "running": self._running,
            "frames_dispatched": self._frames_dispatched,
            "input": self.input.get_status(),
            "subscribers": subscribers,
        }

    def _fanout_loop(self) -> None:
        while not self._stop.is_set():
            frame = self.input.read(timeout=0.5)
            if frame is None:
                continue
            with self._lock:
                subscribers = list(self._subscribers.values())
            for subscriber in subscribers:
                try:
                    subscriber.queue.put_nowait(frame)
                    subscriber.frames_received += 1
                except queue.Full:
                    subscriber.frames_dropped += 1
                    try:
                        subscriber.queue.get_nowait()
                        subscriber.queue.task_done()
                    except queue.Empty:
                        pass
                    try:
                        subscriber.queue.put_nowait(frame)
                        subscriber.frames_received += 1
                    except queue.Full:
                        pass
            self._frames_dispatched += 1


_multiplexer: AudioMultiplexer | None = None
_lock = threading.Lock()


def get_audio_multiplexer() -> AudioMultiplexer:
    global _multiplexer
    with _lock:
        if _multiplexer is None:
            _multiplexer = AudioMultiplexer()
        return _multiplexer
