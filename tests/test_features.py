"""Small known synthetic arrays establish feature definitions."""

import math

import pytest

from accident_risk.processing.physiology_features import physiology_features
from accident_risk.processing.vehicle_features import vehicle_features
from accident_risk.processing.windows import Window
from accident_risk.schemas import SensorFrame


def test_vehicle_statistics():
    frames = tuple(
        SensorFrame(
            timestamp=t,
            stream_id="test",
            source="synthetic",
            vehicle_speed=2 * t,
            accel_x=t,
            accel_y=0,
            accel_z=0,
            gyro_x=0,
            gyro_y=0,
            gyro_z=1,
        )
        for t in range(3)
    )
    features = vehicle_features(Window(0, 2, "test", frames))
    assert features["speed_mean"] == 2
    assert features["speed_std"] == pytest.approx((8 / 3) ** 0.5)
    assert features["speed_slope"] == 2
    assert features["jerk_mean"] == 1
    assert math.isnan(features["rpm_mean"])
    assert math.isnan(features["longitudinal_acceleration_mean"])


def test_physiology_and_no_fake_hrv():
    frames = tuple(
        SensorFrame(timestamp=t, stream_id="test", source="synthetic", heart_rate=60 + t, spo2=s)
        for t, s in enumerate([99, 95, 94])
    )
    features = physiology_features(Window(0, 2, "test", frames))
    assert features["heart_rate_mean"] == 61
    assert features["spo2_drop_count"] == 1
    assert math.isnan(features["rmssd_ms"])
    assert math.isnan(features["sdnn_ms"])


def test_carried_hr_not_counted_as_new_and_rr():
    frames = tuple(
        SensorFrame(
            timestamp=t,
            stream_id="test",
            source="synthetic",
            heart_rate=60,
            rr_interval_ms=rr,
            sensor_timestamps={"heart_rate": 0},
        )
        for t, rr in enumerate([1000, 1100, 1000])
    )
    features = physiology_features(Window(0, 2, "test", frames))
    assert features["heart_rate_age_seconds"] == 2
    assert features["rmssd_ms"] == 100
    assert features["sdnn_ms"] == pytest.approx(57.7350269)
