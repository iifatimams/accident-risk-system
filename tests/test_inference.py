"""Mode independence, missing modalities, stale HR, and model contracts."""

import pytest

from accident_risk.inference.engine import InferenceEngine
from accident_risk.inference.replay_runner import run_source
from accident_risk.schemas import SensorFrame
from accident_risk.sources.live_buffer import LiveBuffer
from accident_risk.sources.replay import ReplaySource


def frame(t, **values):
    return SensorFrame(timestamp=t, stream_id="test", source="synthetic_test", **values)


def test_vehicle_only_and_no_models(models):
    engine = InferenceEngine(vehicle_model=models[0])
    assert engine.process(frame(100, vehicle_speed=10)) is None
    prediction = engine.process(frame(101, vehicle_speed=11))
    assert prediction.event_detection_score is not None
    assert prediction.driver_state_risk is None
    empty = InferenceEngine()
    empty.process(frame(100, vehicle_speed=10))
    assert empty.process(frame(101, vehicle_speed=10)).overall_risk_index is None


def test_physiology_only_missing_spo2_staleness_and_combined(models):
    engine = InferenceEngine(models[0], models[1])
    engine.process(frame(100, heart_rate=70))
    physiological = engine.process(frame(101, heart_rate=72))
    assert physiological.driver_state_risk is not None
    assert physiological.event_detection_score is None
    combined = engine.process(frame(102, vehicle_speed=10))
    assert combined.driver_state_risk is not None
    assert combined.event_detection_score is not None
    stale = engine.process(frame(105, vehicle_speed=11))
    assert stale.driver_state_risk is None
    assert stale.event_detection_score is not None
    assert "stale:heart_rate" in stale.quality_flags
    assert stale.sensor_age_seconds["heart_rate"] == 4


def test_live_and_replay_identical(models):
    frames = [frame(100 + t, vehicle_speed=t, heart_rate=70 if t == 0 else None) for t in range(6)]
    live = LiveBuffer()
    for f in frames:
        live.push(f)
    a, _ = run_source(live, InferenceEngine(*models))
    b, _ = run_source(ReplaySource(frames, maximum_speed=True), InferenceEngine(*models))
    assert [p.model_dump(exclude={"inference_latency_ms"}) for p in a] == [
        p.model_dump(exclude={"inference_latency_ms"}) for p in b
    ]


def test_quality_order_and_contract(models, settings):
    engine = InferenceEngine(*models)
    engine.process(frame(100, heart_rate=70, physiology_signal_quality=1))
    p = engine.process(frame(101, heart_rate=70, physiology_signal_quality=0.1, vehicle_speed=10))
    assert p.driver_state_risk is None
    assert p.event_detection_score is not None
    with pytest.raises(ValueError, match="order"):
        engine.process(frame(99))
    with pytest.raises(ValueError, match="configuration"):
        InferenceEngine(models[0], settings=settings.model_copy(update={"stride_seconds": 2}))
