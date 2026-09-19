"""Causal rolling windows. A future label never changes the feature cutoff."""

from collections import deque
from dataclasses import dataclass

from accident_risk.schemas import SensorFrame


@dataclass(frozen=True)
class Window:
    """Closed feature interval [start, end], in Unix seconds, within one stream."""

    start: float
    end: float
    stream_id: str
    frames: tuple[SensorFrame, ...]

    def __post_init__(self) -> None:
        if self.start > self.end:
            raise ValueError("Invalid window bounds")
        if any(
            f.stream_id != self.stream_id or not self.start <= f.timestamp <= self.end
            for f in self.frames
        ):
            raise ValueError("Window contains cross-stream or future/past-outside data")


class RollingWindows:
    """Bound history in event time; retain complementary equal-time packets."""

    def __init__(self, seconds: float, stride_seconds: float = 1) -> None:
        if seconds <= 0 or stride_seconds <= 0:
            raise ValueError("Window and stride must be positive")
        self.seconds, self.stride = seconds, stride_seconds
        self.buffers: dict[str, deque[SensorFrame]] = {}
        self.last_emitted: dict[str, float] = {}

    def update(self, frame: SensorFrame) -> Window | None:
        """Append and emit when stride has elapsed; features never include later frames."""
        buffer = self.buffers.setdefault(frame.stream_id, deque())
        if buffer and frame.timestamp < buffer[-1].timestamp:
            raise ValueError("Out-of-order window input")
        buffer.append(frame)
        while buffer and buffer[0].timestamp < frame.timestamp - self.seconds:
            buffer.popleft()
        if (
            frame.timestamp - self.last_emitted.get(frame.stream_id, -float("inf"))
            < self.stride - 1e-9
        ):
            return None
        self.last_emitted[frame.stream_id] = frame.timestamp
        return self.at(frame.stream_id, frame.timestamp)

    def at(self, stream_id: str, timestamp: float, seconds: float | None = None) -> Window:
        """Return a strictly cutoff-filtered view of currently retained history."""
        start = max(0, timestamp - (seconds or self.seconds))
        return Window(
            start,
            timestamp,
            stream_id,
            tuple(f for f in self.buffers.get(stream_id, ()) if start <= f.timestamp <= timestamp),
        )


def future_event_target(
    window: Window, event_times: list[float], horizon_seconds: float, observed_until: float
) -> int:
    """Target in (t, t+h]; reject right-censored labels without complete follow-up."""
    if horizon_seconds <= 0:
        raise ValueError("Future-event horizon must be positive")
    if observed_until < window.end + horizon_seconds:
        raise ValueError("Right-censored target: insufficient follow-up")
    return int(any(window.end < t <= window.end + horizon_seconds for t in event_times))
