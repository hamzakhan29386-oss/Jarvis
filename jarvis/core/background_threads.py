"""Small supervised background-thread helpers for the desktop runtime."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable

log = logging.getLogger("jarvis.core.background")


@dataclass
class ManagedThread:
    name: str
    target: Callable[[], None]
    restart: bool = True
    backoff_s: float = 2.0
    thread: threading.Thread | None = None
    restarts: int = 0
    last_error: str = ""


class ThreadSupervisor:
    """Runs named daemon workers and restarts crashable listeners."""

    def __init__(self):
        self._workers: dict[str, ManagedThread] = {}
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._watchdog: threading.Thread | None = None

    def add(self, name: str, target: Callable[[], None], restart: bool = True) -> None:
        with self._lock:
            if name in self._workers:
                return
            worker = ManagedThread(name=name, target=target, restart=restart)
            self._workers[name] = worker
            self._start_worker(worker)

    def start_watchdog(self) -> None:
        if self._watchdog and self._watchdog.is_alive():
            return
        self._watchdog = threading.Thread(target=self._watchdog_loop, daemon=True, name="JARVISRuntimeWatchdog")
        self._watchdog.start()

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict:
        with self._lock:
            return {
                name: {
                    "alive": bool(worker.thread and worker.thread.is_alive()),
                    "restarts": worker.restarts,
                    "last_error": worker.last_error,
                }
                for name, worker in self._workers.items()
            }

    def _start_worker(self, worker: ManagedThread) -> None:
        def runner() -> None:
            try:
                worker.target()
            except Exception as exc:
                worker.last_error = str(exc)
                log.exception("Worker %s crashed", worker.name)

        worker.thread = threading.Thread(target=runner, daemon=True, name=worker.name)
        worker.thread.start()

    def _watchdog_loop(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                for worker in self._workers.values():
                    if worker.restart and worker.thread and not worker.thread.is_alive():
                        worker.restarts += 1
                        time.sleep(worker.backoff_s)
                        self._start_worker(worker)
            time.sleep(2.0)

