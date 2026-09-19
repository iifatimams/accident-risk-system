"""Bounded thread-safe acquisition queue; no barrier across sensor modalities."""

from collections import deque
from collections.abc import Iterator
from threading import RLock

from accident_risk.schemas import SENSOR_FIELDS, SensorFrame
from accident_risk.sources.base import DataSource


class LiveBuffer(DataSource):
    """Nonblocking drain iterator; overflow is explicit, never silent data loss.

    The latest-observation cache is diagnostic. The shared inference synchronizer
    applies staleness; raw queued frames retain original observations.
    """

    def __init__(self, capacity: int = 10000) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = RLock()
        self._queue: deque[SensorFrame] = deque()
        self._latest: dict[str, dict[str, tuple[float, float]]] = {}
        self._running = True

    def start(self) -> None:
        """Accept updates."""
        with self._lock:
            self._running = True

    def stop(self) -> None:
        """Reject new updates while allowing queued observations to drain."""
        with self._lock:
            self._running = False

    def push(self, frame: SensorFrame) -> None:
        """Enqueue immediately, without waiting for other sensors."""
        with self._lock:
            if not self._running:
                raise RuntimeError("Live session is stopped")
            if len(self._queue) >= self.capacity:
                raise BufferError("Live buffer full; consumer must drain or apply backpressure")
            latest = self._latest.setdefault(frame.stream_id, {})
            for name in SENSOR_FIELDS:
                value = getattr(frame, name)
                observed = frame.sensor_timestamps.get(name, frame.timestamp)
                if value is not None and (name not in latest or observed >= latest[name][1]):
                    latest[name] = (value, observed)
            self._queue.append(frame)

    def ages(self, stream_id: str, now: float) -> dict[str, float]:
        """Return ages in seconds for observations at or before `now`."""
        with self._lock:
            return {
                key: now - t for key, (_, t) in self._latest.get(stream_id, {}).items() if t <= now
            }

    def __iter__(self) -> Iterator[SensorFrame]:
        """Drain what is available; never block waiting for HR or SpO2."""
        while True:
            with self._lock:
                if not self._queue:
                    return
                frame = self._queue.popleft()
            yield frame
