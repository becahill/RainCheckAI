"""In-process telemetry primitives for RainCheckAI."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from statistics import mean
from threading import Lock
from typing import Any


def _freeze_tags(tags: Mapping[str, str] | None) -> str:
    """Create a stable key for an optional tag mapping."""
    if not tags:
        return ""
    return ",".join(f"{key}={value}" for key, value in sorted(tags.items()))


class TelemetryCollector:
    """Thread-safe metric collector for counters, gauges, and latencies."""

    def __init__(self) -> None:
        """Initialize empty metric stores."""
        self._counters: dict[str, float] = defaultdict(float)
        self._gauges: dict[str, float] = {}
        self._latencies: dict[str, list[float]] = defaultdict(list)
        self._lock = Lock()

    def increment_counter(
        self,
        name: str,
        value: float = 1.0,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Increment a named counter."""
        key = f"{name}|{_freeze_tags(tags)}"
        with self._lock:
            self._counters[key] += value

    def set_gauge(
        self,
        name: str,
        value: float,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Set a gauge value."""
        key = f"{name}|{_freeze_tags(tags)}"
        with self._lock:
            self._gauges[key] = value

    def record_latency(
        self,
        name: str,
        value_ms: float,
        tags: Mapping[str, str] | None = None,
    ) -> None:
        """Record a latency in milliseconds."""
        key = f"{name}|{_freeze_tags(tags)}"
        with self._lock:
            self._latencies[key].append(value_ms)

    def snapshot(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot of current metrics."""
        with self._lock:
            latencies = {
                key: {
                    "count": len(values),
                    "mean_ms": mean(values) if values else 0.0,
                    "max_ms": max(values) if values else 0.0,
                    "min_ms": min(values) if values else 0.0,
                }
                for key, values in self._latencies.items()
            }
            return {
                "counters": dict(self._counters),
                "gauges": dict(self._gauges),
                "latencies": latencies,
            }
