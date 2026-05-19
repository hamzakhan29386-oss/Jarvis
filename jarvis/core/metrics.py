"""Runtime metrics snapshots for JARVIS health and HUD telemetry."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class Counter:
    count: int = 0
    last_at: float | None = None

    def inc(self, amount: int = 1) -> None:
        self.count += amount
        self.last_at = time.time()


@dataclass
class MetricsRegistry:
    started_at: float = field(default_factory=time.time)
    counters: dict[str, Counter] = field(default_factory=dict)
    gauges: dict[str, Any] = field(default_factory=dict)

    def increment(self, name: str, amount: int = 1) -> None:
        self.counters.setdefault(name, Counter()).inc(amount)

    def set_gauge(self, name: str, value: Any) -> None:
        self.gauges[name] = value

    def snapshot(self) -> dict[str, Any]:
        return {
            "uptime_s": round(time.time() - self.started_at, 3),
            "counters": {name: asdict(counter) for name, counter in self.counters.items()},
            "gauges": dict(self.gauges),
        }


_metrics = MetricsRegistry()


def get_metrics() -> MetricsRegistry:
    return _metrics
