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


def test_multimodal_risk_reaches_dashboard_without_gating_alerts(tmp_path, monkeypatch):
    """The on-device multimodal risk breakdown (vehicle/attention/road
    contributions, dominant_factor, evidence_quality -- multimodal_risk in
    the ingest payload) must reach the dashboard as explainability detail,
    but must never affect collision_alert_id or trigger a family
    notification -- the 5-second-ahead estimate IS risk_score itself now
    (see BackgroundCaptureService.swift), kept separate from
    CollisionDetector's pure IMU-threshold alerting path."""
    sent = []
    monkeypatch.setattr(
        "accident_risk.api.notifications.MockNotificationSender.send",
        lambda self, **kwargs: sent.append(kwargs) or "mock_sent",
    )
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_multimodal.sqlite3")
    )
    now = time.time()
    frame = dict(
        timestamp=now,
        source="ios",
        stream_id="edge-multimodal",
        risk_active=True,
        risk_score=72.5,
        multimodal_risk={
            "current_risk": 70.0, "five_second_ahead_risk": 72.5,
            "risk_level": "high", "vehicle_contribution": 40.0,
            "attention_contribution": 20.0, "road_hazard_contribution": 10.0,
            "contextual_adjustment": 0.0, "evidence_quality": 0.9,
            "dominant_factor": "vehicle-dynamics",
        },
    )
    response = client.post("/ingest", json=frame)
    assert response.status_code == 200
    assert response.json()["collision_alert_id"] is None

    data = client.get("/api/v1/predict/live?stream_id=edge-multimodal").json()
    assert data["multimodal_risk"]["dominant_factor"] == "vehicle-dynamics"
    assert data["multimodal_risk"]["evidence_quality"] == 0.9
    assert not sent  # the risk score never notifies family by itself

    # A frame omitting the key must not crash.
    quiet_frame = dict(frame, risk_score=5, multimodal_risk=None)
    assert client.post("/ingest", json=quiet_frame).status_code == 200
    data = client.get("/api/v1/predict/live?stream_id=edge-multimodal").json()
    assert data["multimodal_risk"] is None


def test_invalid_packets_and_reset(tmp_path):
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_invalid_packets.sqlite3")
    )
    packet = dict(timestamp=time.time(), source="test", stream_id="hardware")
    for update in (
        {"heart_rate": -5},
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


def test_pedestrian_hazard_state_round_trips_through_live_dashboard(tmp_path):
    """cv_pedestrian_hazard_state is dashboard-only diagnostics (see
    PedestrianHazard.swift / Phase 7); defaults to "insufficient_evidence"
    and must not be silently dropped when sent."""
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_pedestrian_hazard.sqlite3")
    )
    default_only = client.get("/api/v1/predict/live?stream_id=none-yet").json()
    assert default_only["cv_pedestrian_hazard_state"] == "insufficient_evidence"

    frame = dict(
        timestamp=time.time(),
        source="ios",
        stream_id="driver-3",
        risk_active=True,
        risk_score=10,
        pedestrians_detected=True,
        pedestrian_count=1,
        pedestrian_hazard_state="entering_conflict_zone",
    )
    assert client.post("/ingest", json=frame).status_code == 200
    data = client.get("/api/v1/predict/live?stream_id=driver-3").json()
    assert data["cv_pedestrians_detected"] is True
    assert data["cv_pedestrian_hazard_state"] == "entering_conflict_zone"


def test_risk_breakdown_round_trips_through_live_dashboard(tmp_path):
    """risk_breakdown is dashboard-only explainability (see
    RiskFusionHeuristics.swift / Phase 9.1): absent/None when never sent,
    and must not be silently dropped when it is."""
    client = TestClient(
        create_app(InferenceEngine(), database_path=tmp_path / "test_risk_breakdown.sqlite3")
    )
    default_only = client.get("/api/v1/predict/live?stream_id=none-yet").json()
    assert default_only["risk_breakdown"] is None

    frame = dict(
        timestamp=time.time(),
        source="ios",
        stream_id="driver-4",
        risk_active=True,
        risk_score=42,
        risk_breakdown={"ml_fusion_base": 20.0, "eyes_closed_penalty": 22.0, "hard_override_applied": False},
    )
    assert client.post("/ingest", json=frame).status_code == 200
    data = client.get("/api/v1/predict/live?stream_id=driver-4").json()
    assert data["risk_breakdown"]["ml_fusion_base"] == 20.0
    assert data["risk_breakdown"]["eyes_closed_penalty"] == 22.0
    assert data["risk_breakdown"]["hard_override_applied"] is False
