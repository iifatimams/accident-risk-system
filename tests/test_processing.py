"""Causality, quality reports, and nonblocking acquisition tests."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from accident_risk.processing.pipeline import FeaturePipeline
from accident_risk.processing.synchronization import Synchronizer, resample
from accident_risk.processing.validation import validate_records
from accident_risk.schemas import SensorFrame
from accident_risk.settings import PipelineSettings, load_config
from accident_risk.sources.live_buffer import LiveBuffer


def frame(t, **kwargs):
    return SensorFrame(timestamp=t, stream_id="synthetic", source="test", **kwargs)


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
