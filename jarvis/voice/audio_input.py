"""Native sounddevice microphone capture for the JARVIS runtime."""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import sounddevice as sd

log = logging.getLogger("jarvis.voice.audio_input")

MIC_DEVICE = 12
MIC_SAMPLE_RATE = 48000
CHANNELS = 1
BLOCKSIZE = 2048
DTYPE = "float32"


@dataclass(frozen=True)
class AudioInputConfig:
    device: int | None = MIC_DEVICE
    samplerate: int = MIC_SAMPLE_RATE
    channels: int = CHANNELS
    blocksize: int = BLOCKSIZE
    dtype: str = DTYPE
    queue_size: int = 128


@dataclass
class AudioFrame:
    data: np.ndarray
    samplerate: int
    captured_at: float
    frame_count: int
    status: str = ""


@dataclass
class AudioInputStats:
    frames_captured: int = 0
    frames_dropped: int = 0
    callback_errors: int = 0
    started_at: float | None = None
    stopped_at: float | None = None
    selected_device: int | None = None
    selected_device_name: str = ""
    hostapi: str = ""
    last_status: str = ""
    last_queue_size: int = 0


def _hostapi_name(index: int | None) -> str:
    try:
        if index is None:
            return ""
        return str(sd.query_hostapis(index).get("name", ""))
    except Exception:
        return ""


def _device_info(index: int) -> dict[str, Any]:
    return dict(sd.query_devices(index))


def resolve_input_device(preferred_device: int | None = MIC_DEVICE) -> int | None:
    """Select a Windows microphone with a production-friendly fallback order."""
    devices = sd.query_devices()

    if preferred_device is not None:
        try:
            info = _device_info(preferred_device)
            if int(info.get("max_input_channels", 0)) > 0:
                log.info("[AudioInput] Using configured mic device=%s name=%s", preferred_device, info.get("name"))
                return preferred_device
        except Exception as exc:
            log.warning("[AudioInput] Configured mic device=%s unavailable: %s", preferred_device, exc)

    candidates: list[tuple[int, dict[str, Any], str]] = []
    for idx, info in enumerate(devices):
        if int(info.get("max_input_channels", 0)) <= 0:
            continue
        hostapi = _hostapi_name(info.get("hostapi"))
        name = str(info.get("name", ""))
        candidates.append((idx, dict(info), hostapi))

    def pick(predicate):
        for idx, info, hostapi in candidates:
            if predicate(str(info.get("name", "")).lower(), hostapi.lower()):
                return idx
        return None

    for label, predicate in (
        ("WASAPI", lambda name, host: "wasapi" in host),
        ("Realtek microphone", lambda name, host: "realtek" in name and ("mic" in name or "microphone" in name)),
        ("Intel Smart Array", lambda name, host: "intel" in name and ("array" in name or "smart" in name)),
    ):
        idx = pick(predicate)
        if idx is not None:
            log.info("[AudioInput] Auto-selected %s device=%s name=%s", label, idx, _device_info(idx).get("name"))
            return idx

    try:
        default_input = sd.default.device[0]
        if default_input is not None and default_input >= 0:
            log.info("[AudioInput] Falling back to default input device=%s", default_input)
            return int(default_input)
    except Exception:
        pass

    log.warning("[AudioInput] No explicit input device found; letting sounddevice choose default")
    return None


class NativeAudioInput:
    """Owns the microphone and feeds a bounded, non-blocking queue."""

    def __init__(self, config: AudioInputConfig | None = None):
        self.config = config or AudioInputConfig()
        self.queue: queue.Queue[AudioFrame] = queue.Queue(maxsize=self.config.queue_size)
        self.stats = AudioInputStats()
        self._stream: sd.InputStream | None = None
        self._lock = threading.RLock()
        self._running = False

    def start(self) -> None:
        with self._lock:
            if self._running:
                return
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            device = resolve_input_device(self.config.device)
            self.stats.selected_device = device
            if device is not None:
                info = _device_info(device)
                self.stats.selected_device_name = str(info.get("name", ""))
                self.stats.hostapi = _hostapi_name(info.get("hostapi"))
                log.info(
                    "[AudioInput] Device info | device=%s name=%s hostapi=%s default_sr=%s channels=%s",
                    device,
                    self.stats.selected_device_name,
                    self.stats.hostapi,
                    info.get("default_samplerate"),
                    info.get("max_input_channels"),
                )

            self._stream = sd.InputStream(
                device=device,
                samplerate=self.config.samplerate,
                channels=self.config.channels,
                dtype=self.config.dtype,
                blocksize=self.config.blocksize,
                callback=self._callback,
                latency="low",
            )
            self._stream.start()
            self._running = True
            self.stats.started_at = time.time()
            self.stats.stopped_at = None
            log.info(
                "[AudioInput] Started | device=%s rate=%d channels=%d dtype=%s blocksize=%d queue_max=%d",
                device,
                self.config.samplerate,
                self.config.channels,
                self.config.dtype,
                self.config.blocksize,
                self.config.queue_size,
            )

    def stop(self) -> None:
        with self._lock:
            self._running = False
            stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.stop()
                stream.close()
            except Exception as exc:
                log.warning("[AudioInput] Error closing stream: %s", exc)
        self.stats.stopped_at = time.time()
        log.info(
            "[AudioInput] Stopped | captured=%d dropped=%d callback_errors=%d",
            self.stats.frames_captured,
            self.stats.frames_dropped,
            self.stats.callback_errors,
        )

    def read(self, timeout: float = 0.5) -> AudioFrame | None:
        try:
            return self.queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _callback(self, indata, frames: int, time_info, status) -> None:
        try:
            if status:
                self.stats.last_status = str(status)
                log.debug("[AudioInput] Callback status=%s", status)
            data = indata[:, 0].copy() if getattr(indata, "ndim", 1) > 1 else indata.copy()
            frame = AudioFrame(
                data=data.astype(np.float32, copy=False),
                samplerate=self.config.samplerate,
                captured_at=time.perf_counter(),
                frame_count=frames,
                status=str(status) if status else "",
            )
            self.queue.put_nowait(frame)
            self.stats.frames_captured += 1
            self.stats.last_queue_size = self.queue.qsize()
        except queue.Full:
            self.stats.frames_dropped += 1
            if self.stats.frames_dropped == 1 or self.stats.frames_dropped % 50 == 0:
                log.warning(
                    "[AudioInput] Dropped frames=%d queue_size=%d",
                    self.stats.frames_dropped,
                    self.queue.qsize(),
                )
        except Exception as exc:
            self.stats.callback_errors += 1
            log.debug("[AudioInput] Callback error: %s", exc)

    def get_status(self) -> dict:
        return {
            "running": self._running,
            "device": self.stats.selected_device,
            "device_name": self.stats.selected_device_name,
            "hostapi": self.stats.hostapi,
            "sample_rate": self.config.samplerate,
            "channels": self.config.channels,
            "blocksize": self.config.blocksize,
            "dtype": self.config.dtype,
            "queue_size": self.queue.qsize(),
            "queue_max": self.config.queue_size,
            "frames_captured": self.stats.frames_captured,
            "dropped_frames": self.stats.frames_dropped,
            "callback_errors": self.stats.callback_errors,
            "last_status": self.stats.last_status,
        }
