"""Causality, quality reports, and nonblocking acquisition tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from accident_risk.processing import collision_detector
from accident_risk.processing.pipeline import FeaturePipeline
from accident_risk.processing.synchronization import Synchronizer, resample
from accident_risk.processing.validation import validate_records
from accident_risk.schemas import SensorFrame
from accident_risk.settings import PipelineSettings, load_config
from accident_risk.sources.live_buffer import LiveBuffer


def frame(t, **kwargs):
    return SensorFrame(timestamp=t, stream_id="synthetic", source="test", **kwargs)


def accel_frame(t, x, y, z):
    return frame(t, accel_x=x, accel_y=y, accel_z=z)


def test_collision_detector_ignores_frames_without_accel():
    assert collision_detector.check(frame(100)) is None


def test_collision_detector_normal_driving_does_not_trigger():
    # ~1.0g, gravity-dominated, as on a phone resting in a mount.
    suspicion = collision_detector.check(accel_frame(100, x=0.1, y=0.1, z=9.8))
    assert suspicion.triggered is False
    assert suspicion.score < 1.0


def test_collision_detector_hard_braking_is_a_negative_example():
    # Hard braking adds real deceleration (~1.5g combined) but is ordinary
    # driving, not a collision -- must not trigger.
    suspicion = collision_detector.check(accel_frame(100, x=11.0, y=0.2, z=9.8))
    assert suspicion.triggered is False


def test_collision_detector_a_bump_is_a_negative_example():
    # A pothole/speed bump jolt (~1.8g vertical) -- still ordinary driving.
    suspicion = collision_detector.check(accel_frame(100, x=0.2, y=0.2, z=17.0))
    assert suspicion.triggered is False


def test_collision_detector_phone_handling_is_a_negative_example():
    # Picking up/dropping the phone can spike accelerometer readings sharply
    # but briefly -- below the collision threshold, same as any other
    # non-collision jolt, since this detector has no notion of "in-vehicle".
    suspicion = collision_detector.check(accel_frame(100, x=5.0, y=5.0, z=12.0))
    assert suspicion.triggered is False


def test_collision_detector_triggers_on_a_simulated_impact():
    # ~2.6g combined, above the 2.5g threshold -- a simulated impact spike,
    # well under the VZCrash research's observed mean crash peak (~3.9g).
    suspicion = collision_detector.check(accel_frame(100, x=25.0, y=0.0, z=5.0))
    assert suspicion.triggered is True
    assert suspicion.score >= 1.0
    assert suspicion.onset_estimate == 100
    assert suspicion.model_version == collision_detector.MODEL_VERSION


def test_resample_is_causal_and_not_spo2_paced():
    frames = [
        frame(100, vehicle_speed=0, heart_rate=60),
        frame(100.5, vehicle_speed=10),
        frame(101, vehicle_speed=20),
    ]
    output = list(resample(frames, target_hz=4))
    assert [f.timestamp for f in output] == [100, 100.25, 100.5, 100.75, 101]
    assert [f.vehicle_speed for f in output] == [0, 0, 10, 10, 20]
    assert output[-1].sensor_timestamps["heart_rate"] == 100


def test_equal_time_resampling_combines_channels():
    output = list(
        resample([frame(100, accel_x=1), frame(100, heart_rate=60), frame(101, accel_x=2)], 1)
    )
    assert output[0].accel_x == 1 and output[0].heart_rate == 60


def test_staleness_conflicts_and_reports():
    sync = Synchronizer(PipelineSettings(max_age_seconds={"heart_rate": 2}))
    sync.update(frame(100, heart_rate=60))
    assert sync.update(frame(101, vehicle_speed=10)).heart_rate == 60
    assert sync.update(frame(103, vehicle_speed=11)).heart_rate is None
    with pytest.raises(ValueError, match="Conflicting"):
        sync.update(frame(103, vehicle_speed=12))
    assert sync.snapshot("synthetic", 103).vehicle_speed == 11
    report = validate_records(
        [
            frame(100, heart_rate=60),
            frame(100, heart_rate=60),
            frame(99),
            {"timestamp": 104, "stream_id": "synthetic", "source": "test", "spo2": float("nan")},
        ]
    )
    assert not report.valid
    assert any("duplicate" in w for w in report.warnings)
    assert len(report.errors) == 2


def test_nonblocking_threadsafe_buffer():
    buffer = LiveBuffer(capacity=100)
    with ThreadPoolExecutor(4) as pool:
        list(pool.map(buffer.push, [frame(t, vehicle_speed=t) for t in range(100)]))
    assert len(list(buffer)) == 100
    assert list(buffer) == []
    assert buffer.ages("synthetic", 100)["vehicle_speed"] == 1
    buffer.stop()
    with pytest.raises(RuntimeError):
        buffer.push(frame(101))
    full = LiveBuffer(capacity=1)
    full.push(frame(1))
    with pytest.raises(BufferError):
        full.push(frame(2))


def test_pipeline_future_mutation_cannot_change_past():
    a, b = FeaturePipeline(), FeaturePipeline()
    for t in range(10):
        sa = a.update(frame(100 + t, vehicle_speed=t))
        sb = b.update(frame(100 + t, vehicle_speed=t))
    prior_a = sa.vehicle.copy()
    b.update(frame(110, vehicle_speed=99999))
    assert sa.vehicle["speed_mean"] == sb.vehicle["speed_mean"]
    assert sa.vehicle["speed_mean"] == prior_a["speed_mean"]


def test_config_inheritance_and_cycles(tmp_path):
    (tmp_path / "base.yaml").write_text("pipeline:\n  stride_seconds: 2\n")
    (tmp_path / "child.yaml").write_text("extends: base.yaml\npipeline:\n  min_samples: 3\n")
    assert load_config(tmp_path / "child.yaml")["pipeline"] == {
        "stride_seconds": 2,
        "min_samples": 3,
    }
    (tmp_path / "base.yaml").write_text("extends: child.yaml\n")
    with pytest.raises(ValueError, match="cycle"):
        load_config(tmp_path / "child.yaml")


def test_rejected_physiology_does_not_return_after_quality_expires():
    pipeline = FeaturePipeline(
        PipelineSettings(max_age_seconds={"heart_rate": 30, "physiology_signal_quality": 1})
    )
    pipeline.update(frame(100, heart_rate=100, physiology_signal_quality=0.1))
    snapshot = pipeline.update(frame(102, vehicle_speed=10))
    assert snapshot.frame.heart_rate is None
