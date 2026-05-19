"""Sentence-level streaming TTS helper for realtime responses."""

from __future__ import annotations

import queue
import re
import threading
import time
from collections.abc import Callable


_SENTENCE_END = re.compile(r"([.!?;]\s+|\n+)")


class StreamingSpeaker:
    """Buffers streamed tokens into sentences and speaks them off-thread."""

    def __init__(
        self,
        tts,
        *,
        on_sentence: Callable[[str], None] | None = None,
        max_queue_size: int = 16,
        min_chars: int = 24,
    ):
        self.tts = tts
        self.on_sentence = on_sentence
        self.min_chars = min_chars
        self._buffer = ""
        self._queue: queue.Queue[str | None] = queue.Queue(maxsize=max_queue_size)
        self._stop = threading.Event()
        self._worker = threading.Thread(target=self._run, daemon=True, name="JARVISStreamingSpeaker")
        self._worker.start()

    def feed_token(self, token: str) -> None:
        if self._stop.is_set() or not token:
            return
        self._buffer += token
        while True:
            match = _SENTENCE_END.search(self._buffer)
            if not match or match.end() < self.min_chars:
                return
            sentence = self._buffer[: match.end()].strip()
            self._buffer = self._buffer[match.end() :]
            if sentence:
                self._enqueue(sentence)

    def flush(self) -> None:
        sentence = self._buffer.strip()
        self._buffer = ""
        if sentence and not self._stop.is_set():
            self._enqueue(sentence)

    def stop(self) -> None:
        self._stop.set()
        self._buffer = ""
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def status(self) -> dict:
        return {"queued": self._queue.qsize(), "stopped": self._stop.is_set()}

    def wait_done(self, timeout: float | None = None) -> bool:
        deadline = time.monotonic() + timeout if timeout else None
        while getattr(self._queue, "unfinished_tasks", 0):
            if deadline and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return True

    def _enqueue(self, sentence: str) -> None:
        try:
            self._queue.put_nowait(sentence)
        except queue.Full:
            # Drop the oldest queued utterance to keep realtime speech responsive.
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            self._queue.put_nowait(sentence)

    def _run(self) -> None:
        while not self._stop.is_set():
            item = self._queue.get()
            if item is None:
                self._queue.task_done()
                break
            try:
                if self.on_sentence:
                    self.on_sentence(item)
                self.tts.speak(item)
            except Exception:
                pass
            finally:
                self._queue.task_done()
