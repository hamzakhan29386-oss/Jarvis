"""Deterministic provider orchestration, health, metrics, and circuit breakers."""

from __future__ import annotations

import threading
import time
import uuid
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from event_bus import emit


@dataclass
class ProviderMetrics:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    total_latency_ms: float = 0.0
    last_success_at: float | None = None
    last_failure_at: float | None = None
    last_error: str = ""

    @property
    def success_rate(self) -> float:
        return self.successes / self.calls if self.calls else 1.0

    @property
    def avg_latency_ms(self) -> float | None:
        return round(self.total_latency_ms / self.successes, 2) if self.successes else None

    def snapshot(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["success_rate"] = round(self.success_rate, 3)
        payload["avg_latency_ms"] = self.avg_latency_ms
        return payload


class CircuitBreaker:
    """Simple half-open circuit breaker for model providers."""

    def __init__(self, failure_threshold: int = 3, recovery_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_seconds = recovery_seconds
        self.failures = 0
        self.opened_at: float | None = None

    def allow(self) -> bool:
        if self.opened_at is None:
            return True
        if time.time() - self.opened_at >= self.recovery_seconds:
            return True
        return False

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = None

    def record_failure(self) -> None:
        self.failures += 1
        if self.failures >= self.failure_threshold:
            self.opened_at = time.time()

    def snapshot(self) -> dict[str, Any]:
        return {
            "state": "open" if self.opened_at and not self.allow() else "closed",
            "failures": self.failures,
            "opened_at": self.opened_at,
            "recovery_seconds": self.recovery_seconds,
        }


class MetricsTracker:
    def __init__(self):
        self._metrics: dict[str, ProviderMetrics] = {}

    def ensure(self, provider: str) -> ProviderMetrics:
        return self._metrics.setdefault(provider, ProviderMetrics())

    def success(self, provider: str, latency_ms: float) -> None:
        metrics = self.ensure(provider)
        metrics.calls += 1
        metrics.successes += 1
        metrics.total_latency_ms += latency_ms
        metrics.last_success_at = time.time()

    def failure(self, provider: str, error: str) -> None:
        metrics = self.ensure(provider)
        metrics.calls += 1
        metrics.failures += 1
        metrics.last_failure_at = time.time()
        metrics.last_error = error[:500]

    def snapshot(self) -> dict[str, Any]:
        return {name: metrics.snapshot() for name, metrics in self._metrics.items()}


class HealthMonitor:
    def score(self, metrics: ProviderMetrics, breaker: CircuitBreaker) -> float:
        if not breaker.allow():
            return 0.0
        latency = metrics.avg_latency_ms or 500.0
        latency_score = max(0.1, min(1.0, 1000.0 / max(latency, 1.0)))
        return round((metrics.success_rate * 0.7) + (latency_score * 0.3), 3)


class ProviderManager:
    """Central provider health and cooldown registry."""

    def __init__(self):
        self._lock = threading.RLock()
        self.metrics = MetricsTracker()
        self.health = HealthMonitor()
        self.breakers: dict[str, CircuitBreaker] = {}

    def can_call(self, provider: str) -> bool:
        with self._lock:
            return self._breaker(provider).allow()

    def record_success(self, provider: str, latency_ms: float) -> None:
        with self._lock:
            self._breaker(provider).record_success()
            self.metrics.success(provider, latency_ms)
        emit("provider_call_succeeded", {"provider": provider, "latency_ms": round(latency_ms, 2)}, source="provider_manager")

    def record_failure(self, provider: str, error: str) -> None:
        with self._lock:
            self._breaker(provider).record_failure()
            self.metrics.failure(provider, error)
        emit("provider_call_failed", {"provider": provider, "error": error[:500]}, priority=2, source="provider_manager")

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            metrics = self.metrics.snapshot()
            providers = sorted(set(metrics) | set(self.breakers))
            return {
                "providers": {
                    provider: {
                        "metrics": metrics.get(provider, ProviderMetrics().snapshot()),
                        "circuit_breaker": self._breaker(provider).snapshot(),
                        "health_score": self.health.score(self.metrics.ensure(provider), self._breaker(provider)),
                    }
                    for provider in providers
                }
            }

    def _breaker(self, provider: str) -> CircuitBreaker:
        return self.breakers.setdefault(provider, CircuitBreaker())


@dataclass
class RequestLifecycle:
    id: str
    prompt_preview: str
    streaming: bool
    started_at: float = field(default_factory=time.time)
    cancelled: bool = False


class RequestLifecycleManager:
    def __init__(self):
        self._lock = threading.RLock()
        self._requests: dict[str, RequestLifecycle] = {}

    def start(self, prompt: str, *, streaming: bool) -> RequestLifecycle:
        request = RequestLifecycle(id=uuid.uuid4().hex, prompt_preview=prompt[:200], streaming=streaming)
        with self._lock:
            self._requests[request.id] = request
        emit("request_lifecycle_started", asdict(request), source="request_lifecycle")
        return request

    def finish(self, request_id: str, **metadata) -> None:
        with self._lock:
            request = self._requests.pop(request_id, None)
        if request:
            emit(
                "request_lifecycle_finished",
                {"id": request_id, "elapsed_ms": int((time.time() - request.started_at) * 1000), **metadata},
                source="request_lifecycle",
            )

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {"active_requests": [asdict(request) for request in self._requests.values()]}


class UnifiedStreamEngine:
    """Normalizes stream/non-stream responses into one token lifecycle."""

    def from_text(self, text: str) -> Iterator[str]:
        yield text

    def collect(self, tokens: Iterable[str]) -> str:
        return "".join(tokens)


class CognitiveRouter:
    def __init__(self):
        self.providers = get_provider_manager()
        self.lifecycle = RequestLifecycleManager()
        self.streams = UnifiedStreamEngine()

    def snapshot(self) -> dict[str, Any]:
        return {
            "providers": self.providers.snapshot(),
            "requests": self.lifecycle.snapshot(),
        }


_provider_manager: ProviderManager | None = None
_cognitive_router: CognitiveRouter | None = None
_lock = threading.RLock()


def get_provider_manager() -> ProviderManager:
    global _provider_manager
    with _lock:
        if _provider_manager is None:
            _provider_manager = ProviderManager()
        return _provider_manager


def get_cognitive_router() -> CognitiveRouter:
    global _cognitive_router
    with _lock:
        if _cognitive_router is None:
            _cognitive_router = CognitiveRouter()
        return _cognitive_router
