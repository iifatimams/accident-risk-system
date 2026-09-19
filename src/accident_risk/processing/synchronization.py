"""Causal per-channel forward fill and optional fixed-rate resampling."""

import math
from collections.abc import Iterable, Iterator

from accident_risk.schemas import SENSOR_FIELDS, SensorFrame
from accident_risk.settings import PipelineSettings


class Synchronizer:
    """Per-stream clock and observation cache. Late frames are rejected explicitly."""

    def __init__(self, settings: PipelineSettings | None = None) -> None:
        self.settings = settings or PipelineSettings()
        self.values: dict[str, dict[str, tuple[float, float]]] = {}
        self.last: dict[str, float] = {}

    def update(self, frame: SensorFrame) -> SensorFrame:
        """Carry only past, nonstale readings; never refresh an old reading's age."""
        if frame.timestamp < self.last.get(frame.stream_id, -1):
            raise ValueError("Out-of-order frame: reorder upstream or start a new stream")
        state = self.values.setdefault(frame.stream_id, {})
        # Check conflicts before mutating any state.
        for name in SENSOR_FIELDS:
            value = getattr(frame, name)
            observed = frame.sensor_timestamps.get(name, frame.timestamp)
            if (
                value is not None
                and name in state
                and observed == state[name][1]
                and value != state[name][0]
            ):
                raise ValueError(f"Conflicting duplicate observation: {name}")
        for name in SENSOR_FIELDS:
            value = getattr(frame, name)
            observed = frame.sensor_timestamps.get(name, frame.timestamp)
            if value is not None and (name not in state or observed >= state[name][1]):
                state[name] = (value, observed)
        self.last[frame.stream_id] = frame.timestamp
        return self.snapshot(frame.stream_id, frame.timestamp, frame.source)

    def snapshot(
        self, stream_id: str, timestamp: float, source: str = "synchronized"
    ) -> SensorFrame:
        """Create a causal snapshot at a requested time and preserve observation timestamps."""
        payload: dict = {"timestamp": timestamp, "stream_id": stream_id, "source": source}
        observed_at = {}
        for name, (value, observed) in self.values.get(stream_id, {}).items():
            age = timestamp - observed
            limit = self.settings.max_age_seconds.get(name, self.settings.default_max_age_seconds)
            if 0 <= age <= limit:
                payload[name] = value
                observed_at[name] = observed
        return SensorFrame(**payload, sensor_timestamps=observed_at)

    def ages(self, stream_id: str, timestamp: float) -> dict[str, float]:
        """Include expired values in diagnostics, but never in model snapshots."""
        return {
            name: timestamp - t
            for name, (_, t) in self.values.get(stream_id, {}).items()
            if t <= timestamp
        }


def resample(
    frames: Iterable[SensorFrame], target_hz: float, settings: PipelineSettings | None = None
) -> Iterator[SensorFrame]:
    """Resample one ordered stream, anchored at its first observation, with causal ZOH.

    Output ends at the final observed timestamp; no interpolation/backfill is used.
    This is an explicit optional acquisition step, not a different training transform.
    """
    if not math.isfinite(target_hz) or target_hz <= 0:
        raise ValueError("target_hz must be positive and finite")
    sync = Synchronizer(settings)
    origin: float | None = None
    index = 0
    stream_id: str | None = None
    last = -1.0
    for frame in frames:
        if origin is None:
            origin, stream_id = frame.timestamp, frame.stream_id
        if frame.stream_id != stream_id or frame.timestamp < last:
            raise ValueError("resample requires one ordered stream")
        # Emit strictly earlier ticks before observing the next packet.
        while origin + index / target_hz < frame.timestamp - 1e-9:
            yield sync.snapshot(frame.stream_id, origin + index / target_hz, frame.source)
            index += 1
        sync.update(frame)
        last = frame.timestamp
    # Delaying a coincident tick until all equal-time observations are consumed
    # makes the result independent of modality order at that timestamp.
    if origin is not None and origin + index / target_hz <= last + 1e-9:
        yield sync.snapshot(str(stream_id), origin + index / target_hz)
