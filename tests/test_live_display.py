"""Actual-observation freshness and validation for dashboard ingestion."""

import time

from fastapi.testclient import TestClient

from accident_risk.api.main import create_app
from accident_risk.inference.engine import InferenceEngine


def test_empty_partial_stale_and_zero(tmp_path):
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_live_display.sqlite3")
    )
    empty = client.get("/api/v1/predict/live").json()
    assert empty["status"] == "waiting_for_hardware_stream"
    assert empty["heart"] is None and empty["risk_score"] is None
    now = time.time()
    packet = dict(
        timestamp=now,
        source="test",
        stream_id="hardware",
        vehicle_speed=0,
        heart_rate=82,
        sensor_timestamps={"heart_rate": now - 10},
    )
    assert client.post("/ingest", json=packet).status_code == 200
    data = client.get("/api/v1/predict/live").json()
    assert data["speed"] == 0 and data["heart"] == 82
    assert data["location"]["latitude"] is None and data["rpm"] is None
    assert data["risk_score"] is None
    # Re-sending an old measurement must not renew it.
    stale = dict(
        timestamp=now + 0.01,
        stream_id="old",
        source="test",
        heart_rate=82,
        sensor_timestamps={"heart_rate": now - 40},
    )
    assert client.post("/ingest", json=stale).status_code == 200
    data = client.get("/api/v1/predict/live?stream_id=old").json()
    assert data["heart"] is None and data["status"] == "stale"
    assert client.get("/api/v1/predict/live").status_code == 400


def test_invalid_packets_and_reset(tmp_path):
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_invalid_packets.sqlite3")
    )
    packet = dict(timestamp=time.time(), source="test", stream_id="hardware")
    for update in (
        {"spo2": 110},
        {"gps_latitude": 200},
        {"vehicle_speed": -1},
        {"timestamp": time.time() + 100},
    ):
        assert client.post("/ingest", json=packet | update).status_code == 422
    assert client.post("/ingest", json=packet | {"vehicle_speed": 10}).status_code == 200
    assert client.get("/api/v1/predict/live").json()["speed"] == 36
    client.post("/session/stop")
    assert client.get("/api/v1/predict/live").json()["speed"] is None
    client.post("/session/start")
    assert client.get("/api/v1/predict/live").json()["status"] == "waiting_for_hardware_stream"

    # Legacy `speed` (km/h) is a supported alias for `vehicle_speed` (m/s),
    # not an invalid field -- a 40 km/h legacy reading must convert and
    # round-trip back to 40, not be rejected. Checked last, on its own stream,
    # so it doesn't disturb the single-stream assumptions above.
    legacy = dict(packet, stream_id="hardware-legacy")
    assert client.post("/ingest", json=legacy | {"speed": 40}).status_code == 200
    assert client.get("/api/v1/predict/live?stream_id=hardware-legacy").json()["speed"] == 40


def test_attention_state_round_trips_through_live_dashboard(tmp_path):
    """cv_attention_state is dashboard-only diagnostics (see
    DriverAttentionState.swift / Phase 2); defaults to "attentive" when a
    client never sends it, and must not be silently dropped when it does."""
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_attention_state.sqlite3")
    )
    default_only = client.get("/api/v1/predict/live?stream_id=none-yet").json()
    assert default_only["cv_attention_state"] == "attentive"

    frame = dict(
        timestamp=time.time(),
        source="ios",
        stream_id="driver-1",
        risk_active=True,
        risk_score=10,
        eyes_on_road=False,
        attention_state="looking_down",
    )
    assert client.post("/ingest", json=frame).status_code == 200
    data = client.get("/api/v1/predict/live?stream_id=driver-1").json()
    assert data["cv_eyes_on_road"] is False
    assert data["cv_attention_state"] == "looking_down"


def test_forward_hazard_state_round_trips_through_live_dashboard(tmp_path):
    """cv_forward_hazard_state is dashboard-only diagnostics (see
    ForwardCollisionHazard.swift / Phase 6); defaults to
    "no_relevant_lead_object" and must not be silently dropped when sent."""
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_forward_hazard.sqlite3")
    )
    default_only = client.get("/api/v1/predict/live?stream_id=none-yet").json()
    assert default_only["cv_forward_hazard_state"] == "no_relevant_lead_object"

    frame = dict(
        timestamp=time.time(),
        source="ios",
        stream_id="driver-2",
        risk_active=True,
        risk_score=10,
        lead_closing_fast=True,
        forward_hazard_state="increasing_closing_hazard",
    )
    assert client.post("/ingest", json=frame).status_code == 200
    data = client.get("/api/v1/predict/live?stream_id=driver-2").json()
    assert data["cv_closing_fast"] is True
    assert data["cv_forward_hazard_state"] == "increasing_closing_hazard"
