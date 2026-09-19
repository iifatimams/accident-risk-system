"""Synthetic schema tests; not clinical thresholds or performance evidence."""

import math

import pytest
from pydantic import ValidationError

from accident_risk.schemas import SensorFrame


def test_optional_and_units():
    frame = SensorFrame(
        timestamp=100,
        stream_id="test",
        source="synthetic",
        heart_rate=72,
        spo2=98,
        tire_pressure_fl=235,
    )
    assert frame.accel_x is None
    assert frame.heart_rate == 72
    assert frame.tire_pressure_fl == 235


@pytest.mark.parametrize(
    "values",
    [
        {"spo2": 400},
        {"heart_rate": -1},
        {"heart_rate": 400},
        {"vehicle_speed": -1},
        {"accel_x": math.nan},
        {"rpm": math.inf},
        {"gps_latitude": 95},
        {"gps_heading": 7},
        {"steering_angle": 20},
        {"tire_pressure_fl": 0},
        {"tire_pressure_rr": 1200},
        {"timestamp": -1},
        {"heart_rate": 70, "sensor_timestamps": {"heart_rate": 101}},
    ],
)
def test_invalid(values):
    with pytest.raises(ValidationError):
        SensorFrame(**({"timestamp": 100, "stream_id": "t", "source": "synthetic"} | values))
