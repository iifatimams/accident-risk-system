"""Tests for the RoadSense_Synthetic_Driving_Scenarios_v1 fixture package.

SYNTHETIC SOFTWARE TEST DATA ONLY -- see tests/fixtures/roadsense_synthetic_v1/README.md.
Nothing here is evidence of real-world model accuracy (AGENTS.md: "Synthetic
fixtures are software tests, never evidence of model performance"). These
tests exercise schema validation, replay plumbing, the INDEPENDENT rule-based
collision detector, and pipeline robustness to missing sensors -- never a
trained production accident predictor, and never a live/production server.
Every HTTP-level test here uses a fresh tmp_path sqlite database and the
app's default MockNotificationSender (same as every other test in this
suite), so no simulated event can ever reach a real family contact.
"""

from __future__ import annotations

import math

import pytest
from fastapi.testclient import TestClient

from accident_risk.api.main import create_app
from accident_risk.inference.engine import InferenceEngine
from accident_risk.processing.collision_detector import check as collision_check
from accident_risk.sources.replay import ReplaySource
from tests.synthetic_fixtures import SCENARIO_NAMES, iter_scenarios, load_scenario

HARD_NEGATIVES = (
    "normal_cruise",
    "gentle_brake",
    "normal_roundabout",
    "stopped_pedestrian",
    "lead_vehicle_safe",
    "gps_obd_dropout",
)
NEAR_MISSES = ("lead_vehicle_nearmiss", "pedestrian_crossing_nearmiss")
COLLISIONS = ("lead_vehicle_collision", "pedestrian_crossing_collision")


def test_manifest_lists_exactly_the_ten_documented_scenarios():
    assert len(SCENARIO_NAMES) == 10
    assert set(HARD_NEGATIVES) | set(NEAR_MISSES) | set(COLLISIONS) == set(SCENARIO_NAMES)


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_fixture_matches_schema_and_has_no_leaked_ground_truth(name):
    """Every frame must be a real, valid SensorFrame (extra="forbid" would
    already reject anything unexpected), and no ground-truth-only column
    (event_in_next_5s, contact_now, the *_truth distances) may appear as a
    SensorFrame field -- that's the whole reason the package ships them as
    a separate CSV instead of embedding them in the JSONL."""
    scenario = load_scenario(name)
    assert len(scenario.frames) == 401
    assert len(scenario.labels) == 401
    model_fields = set(type(scenario.frames[0]).model_fields)
    for forbidden in ("event_in_next_5s", "contact_now", "lead_distance_m_truth", "pedestrian_distance_m_truth"):
        assert forbidden not in model_fields
    # No fabricated physiology: the package explicitly omits WHOOP/Helio fields.
    assert all(f.heart_rate is None and f.rr_interval_ms is None for f in scenario.frames)
    # Strictly increasing timestamps at the documented 10 Hz cadence.
    for a, b in zip(scenario.frames, scenario.frames[1:]):
        assert round(b.timestamp - a.timestamp, 3) == 0.1


@pytest.mark.parametrize("name", HARD_NEGATIVES + NEAR_MISSES)
def test_hard_negatives_and_near_misses_never_trigger_collision_detector(name):
    """Normal driving, gentle braking, ordinary roundabout turning, a
    stopped-but-visible pedestrian, safe lead-vehicle following, GPS/OBD
    dropout, and BOTH near-miss scenarios must never trip the independent
    kinematic collision detector. A near miss is, by construction, exactly
    the case where avoidance worked -- it must read identically to "normal"
    here, since the detector only looks at realized impact acceleration."""
    scenario = load_scenario(name)
    triggers = [collision_check(f) for f in scenario.frames]
    assert not any(t is not None and t.triggered for t in triggers)


@pytest.mark.parametrize("name", COLLISIONS)
def test_collision_scenarios_trigger_exactly_at_the_labeled_contact_tick(name):
    """The independent collision detector must fire at the synthetic impact
    tick (the one row with contact_now=True, elapsed_s=28.0) and NOWHERE
    else in the other 400 frames -- a single false negative or any false
    positive elsewhere would defeat the point of an independent detector."""
    scenario = load_scenario(name)
    assert scenario.onset_s == 28.0
    triggered_indices = [
        i for i, f in enumerate(scenario.frames)
        if (result := collision_check(f)) is not None and result.triggered
    ]
    contact_indices = [i for i, label in enumerate(scenario.labels) if label["contact_now"] == "True"]
    assert contact_indices == [280]  # 28.0s at 10 Hz, 0-indexed from t=0
    assert triggered_indices == contact_indices, (
        f"{name}: detector triggered at {triggered_indices}, labeled contact at {contact_indices}"
    )


@pytest.mark.parametrize("name", COLLISIONS)
def test_five_second_event_label_correctness(name):
    """event_in_next_5s must be True for exactly the 50 ticks (5s at 10Hz)
    strictly before the collision tick, and false everywhere else --
    including AT and AFTER the collision itself, which is "it happened",
    not "it will happen in the next 5s"."""
    scenario = load_scenario(name)
    positives = [i for i, label in enumerate(scenario.labels) if label["event_in_next_5s"] == "True"]
    assert positives == list(range(230, 280)), name  # elapsed_s in [23.0, 27.9]
    assert scenario.labels[280]["event_in_next_5s"] == "False"  # the impact tick itself


@pytest.mark.parametrize("name", SCENARIO_NAMES)
def test_full_pipeline_handles_every_scenario_without_crashing(name):
    """Regression/robustness guard, not a model-accuracy claim: every
    scenario (including gps_obd_dropout, which has stretches of missing GPS
    and OBD-like fields) must replay through the real ReplaySource +
    InferenceEngine pipeline without raising. No trained model is loaded
    here (matching this repo's existing bare-`InferenceEngine()` test
    convention), so overall_risk_index stays None throughout -- that is
    correct abstention behavior, not a defect, and must not be read as
    "the model said low risk" for frames that include a simulated impact."""
    scenario = load_scenario(name)
    engine = InferenceEngine()
    predictions = []
    for frame in ReplaySource(list(scenario.frames), maximum_speed=True):
        predictions.append(engine.process(frame))
    assert len(predictions) == len(scenario.frames)
    assert all(p.overall_risk_index is None for p in predictions if p is not None)


def test_gps_obd_dropout_scenario_actually_contains_dropouts():
    """Sanity check on the fixture itself, not just the code under test --
    if the generator ever stopped actually omitting fields, the robustness
    test above would stop meaning anything."""
    scenario = load_scenario("gps_obd_dropout")
    assert any(f.gps_latitude is None for f in scenario.frames)
    assert any(f.vehicle_speed is None for f in scenario.frames)
    assert any(f.gps_latitude is not None for f in scenario.frames)  # also recovers


def test_ingesting_a_synthetic_collision_never_reaches_a_real_notification_provider():
    """Posts a synthetic collision frame (source="synthetic_replay", matching
    the fixture's own convention) through the real /ingest HTTP path, against
    an isolated tmp_path database -- never a running production/ngrok
    instance, and never this repo's real data/app.sqlite3. The app's
    notification_sender is hardcoded to MockNotificationSender in
    create_app() regardless of input; this asserts that stays true for a
    synthetic payload specifically, so a future change can't silently start
    routing synthetic events to a real provider."""
    import tempfile
    import time
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        app = create_app(InferenceEngine(), database_path=Path(tmp) / "test_synthetic_ingest.sqlite3")
        assert type(app.state.notification_sender).__name__ == "MockNotificationSender"
        client = TestClient(app)
        scenario = load_scenario("lead_vehicle_collision")
        contact_frame = scenario.frames[280]
        payload = contact_frame.model_dump(mode="json", exclude_none=True)
        payload["client_incident_id"] = "synthetic-test-incident-1"
        # The fixture's own arbitrary synthetic epoch (see its README) is far
        # from real wall-clock time, which /ingest correctly rejects as
        # "too far in the future" -- rebased here only so this test can
        # exercise the real HTTP path at all; the point of this test is the
        # notification sender, not timestamp fidelity. sensor_timestamps is
        # dropped too, since it's expressed in that same synthetic epoch and
        # would otherwise read as "future" relative to the rebased timestamp.
        payload["timestamp"] = time.time()
        payload.pop("sensor_timestamps", None)
        response = client.post("/ingest", json=payload)
        assert response.status_code == 200
        assert type(app.state.notification_sender).__name__ == "MockNotificationSender"
