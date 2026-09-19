"""Synthetic replay tests with a fake clock, never realtime delays."""

import pytest

from accident_risk.schemas import SensorFrame
from accident_risk.sources.replay import ReplaySource


@pytest.mark.parametrize(
    "speed,maximum,total", [(1, False, 4), (2, False, 2), (100, False, 0.04), (1, True, 0)]
)
def test_timing_and_order(speed, maximum, total):
    clock = [0.0]

    def sleep(seconds):
        clock[0] += seconds

    frames = [
        SensorFrame(timestamp=t, stream_id="test", source="synthetic", vehicle_speed=t)
        for t in [104, 100, 102]
    ]
    source = ReplaySource(frames, speed, maximum, clock=lambda: clock[0], sleep=sleep)
    output = list(source)
    assert [f.timestamp for f in output] == [100, 102, 104]
    assert output[-1].vehicle_speed == 104
    assert clock[0] == pytest.approx(total)


def test_stop_restart_and_invalid_speed():
    frames = [SensorFrame(timestamp=t, stream_id="t", source="synthetic") for t in [1, 2]]
    source = ReplaySource(frames, maximum_speed=True)
    iterator = iter(source)
    next(iterator)
    source.stop()
    assert list(iterator) == []
    assert list(source) == frames
    with pytest.raises(ValueError):
        ReplaySource([], speed=0)
