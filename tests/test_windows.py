"""Boundary and adversarial future-data leakage tests."""

import pytest

from accident_risk.processing.windows import RollingWindows, Window, future_event_target
from accident_risk.schemas import SensorFrame


def test_boundaries_streams_and_horizon():
    rolling = RollingWindows(2)
    for t in range(6):
        window = rolling.update(SensorFrame(timestamp=t, stream_id="test", source="synthetic"))
    assert [f.timestamp for f in window.frames] == [3, 4, 5]
    assert future_event_target(window, [5, 7], 2, observed_until=7) == 1
    assert future_event_target(window, [5, 7.01], 2, observed_until=8) == 0
    with pytest.raises(ValueError, match="censored"):
        future_event_target(window, [6], 2, observed_until=6)
    assert rolling.at("another", 5).frames == ()
    assert all(f.timestamp <= 4 for f in rolling.at("test", 4).frames)


def test_future_frame_rejected():
    future = SensorFrame(timestamp=11, stream_id="test", source="synthetic")
    with pytest.raises(ValueError, match="future"):
        Window(0, 10, "test", (future,))
