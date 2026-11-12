"""Contract tests for the product API: login, role gating and the session flow."""

import time

import pytest
from fastapi.testclient import TestClient

from accident_risk.api.main import create_app
from accident_risk.inference.engine import InferenceEngine


@pytest.fixture
def app(tmp_path, models):
    # Explicit test-only opt-in: sign_in() below needs the code back in the
    # HTTP response. This must never be ambient/default -- see auth.py.
    return create_app(
        InferenceEngine(*models),
        database_path=tmp_path / "app.sqlite3",
        dev_mock_sms_delivery=True,
    )


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def sign_in(client, phone):
    """Complete the two-step phone login and return an auth header."""
    requested = client.post("/app/api/auth/request-code", json={"phone": phone}).json()
    verified = client.post(
        "/app/api/auth/verify", json={"phone": phone, "code": requested["code"]}
    )
    assert verified.status_code == 200, verified.text
    return {"Authorization": f"Bearer {verified.json()['token']}"}


@pytest.fixture
def people(app):
    database = app.state.db
    database.create_user("+971501110001", "Driver One", "driver")
    database.create_user("+971501110002", "Manager One", "account_manager")
    database.create_user("+971501110003", "Admin One", "administrator")
    return database


def test_interface_is_served(client):
    page = client.get("/app")
    assert page.status_code == 200
    assert "RoadSense" in page.text
    assert client.get("/static/app/app.js").status_code == 200
    assert client.get("/static/app/app.css").status_code == 200


def test_original_dashboard_still_works(client):
    """The research page and ingestion API must be unaffected by the new routes."""
    assert client.get("/").status_code == 200
    assert client.get("/health").json()["status"] == "ok"


def test_everything_requires_authentication(client, people):
    for path in ("/app/api/me", "/app/api/sessions", "/app/api/live", "/app/api/users"):
        assert client.get(path).status_code == 401, path


def test_wrong_code_is_refused(client, people):
    client.post("/app/api/auth/request-code", json={"phone": "+971501110001"})
    response = client.post(
        "/app/api/auth/verify", json={"phone": "+971501110001", "code": "000000"}
    )
    assert response.status_code == 401


def test_driver_cannot_reach_manager_or_admin_routes(client, people):
    headers = sign_in(client, "+971501110001")
    assert client.get("/app/api/me", headers=headers).json()["role"] == "driver"
    assert client.get("/app/api/live", headers=headers).status_code == 403
    assert client.get("/app/api/drivers", headers=headers).status_code == 403
    assert client.get("/app/api/users", headers=headers).status_code == 403
    assert client.put(
        "/app/api/settings", headers=headers, json={"values": {"risk_high_threshold": 0.6}}
    ).status_code == 403


def test_manager_cannot_reach_admin_routes(client, people):
    headers = sign_in(client, "+971501110002")
    assert client.get("/app/api/live", headers=headers).status_code == 200
    assert client.get("/app/api/drivers", headers=headers).status_code == 200
    assert client.get("/app/api/users", headers=headers).status_code == 403
    assert client.post(
        "/app/api/devices", headers=headers, json={"kind": "adapter", "identifier": "X"}
    ).status_code == 403


def test_administrator_inherits_manager_routes(client, people):
    headers = sign_in(client, "+971501110003")
    for path in ("/app/api/live", "/app/api/drivers", "/app/api/devices", "/app/api/users"):
        assert client.get(path, headers=headers).status_code == 200, path


def test_session_lifecycle_and_ingestion(client, people, monkeypatch):
    """Keep live-session time and synthetic frame time in Unix seconds UTC aligned."""
    clock = [1_000_000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={})
    assert started.status_code == 200
    stream_id = started.json()["stream_id"]

    # Starting again while a session is active is idempotent -- it returns the
    # SAME running session rather than 409, so a client retry (e.g. after a
    # flaky network response) can't accidentally orphan the first session.
    # See routes_app.py start_session: "Idempotently return the running
    # session rather than failing."
    again = client.post("/app/api/driver/session/start", headers=headers, json={})
    assert again.status_code == 200
    assert again.json()["id"] == started.json()["id"]

    for step in range(6):
        clock[0] = 1_000_000.0 + step
        frame = {
            "timestamp": clock[0],
            "source": "synthetic_test",
            "stream_id": stream_id,
            "vehicle_speed": 10 + step,
            "accel_x": step,
            "accel_y": 0,
            "accel_z": 9.8,
            "heart_rate": 70,
            "physiology_signal_quality": 1,
            "gps_latitude": 25.3,
            "gps_longitude": 55.4,
        }
        assert client.post("/ingest", json=frame).status_code == 200

    state = client.get("/app/api/driver/state", headers=headers).json()
    assert state["session"]["stream_id"] == stream_id

    stopped = client.post("/app/api/driver/session/stop", headers=headers)
    assert stopped.status_code == 200
    assert stopped.json()["summary"]["points"] > 0

    detail = client.get(
        f"/app/api/sessions/{started.json()['id']}", headers=headers
    ).json()
    assert len(detail["points"]) > 0
    # Stopping with nothing active is also a graceful no-op, not an error --
    # same idempotent philosophy as session/start, for the same reason (a
    # flaky mobile network must not turn a retried stop into a hard failure).
    again_stop = client.post("/app/api/driver/session/stop", headers=headers)
    assert again_stop.status_code == 200
    assert again_stop.json()["status"] == "no_active_session"


def test_research_export_requires_consent_is_authorized_and_pseudonymous(
    client, people, monkeypatch
):
    """Recording/export milestone: explicit consent gates recording AND export,
    independent of role; CSV+JSON export is pseudonymous, precise GPS is off by
    default and gated to account_manager+, missing sensors stay null."""
    clock = [2_000_000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    driver_headers = sign_in(client, "+971501110001")

    # No consent decision yet -- reads as not granted.
    status = client.get("/app/api/driver/research-consent", headers=driver_headers)
    assert status.status_code == 200 and status.json()["granted"] is False

    started = client.post("/app/api/driver/session/start", headers=driver_headers, json={})
    assert started.status_code == 200
    stream_id, session_id = started.json()["stream_id"], started.json()["id"]

    def send_frames(start_step: int = 0) -> None:
        for step in range(start_step, start_step + 6):
            clock[0] = 2_000_000.0 + step
            frame = {
                "timestamp": clock[0],
                "source": "synthetic_test",
                "stream_id": stream_id,
                "vehicle_speed": 15 + step,
                "accel_x": 0.1, "accel_y": 0.2, "accel_z": 9.8,
                "gyro_x": 0.0, "gyro_y": 0.0, "gyro_z": 0.0,
                "heart_rate": 72,
                "physiology_signal_quality": 1,
                "gps_latitude": 25.3, "gps_longitude": 55.4,
            }
            assert client.post("/ingest", json=frame).status_code == 200

    # Frames sent WITHOUT consent still drive the live dashboard, but produce
    # no research-grade samples, and export is refused outright.
    send_frames(start_step=0)
    assert client.get(
        f"/app/api/sessions/{session_id}/export.json", headers=driver_headers
    ).status_code == 403

    # Consent is self-granted -- nothing else authorizes research collection.
    granted = client.post(
        "/app/api/driver/research-consent", headers=driver_headers, json={"granted": True}
    )
    assert granted.status_code == 200 and granted.json()["granted"] is True

    send_frames(start_step=10)

    # Driver can export their own session; GPS is absent by default.
    export = client.get(f"/app/api/sessions/{session_id}/export.json", headers=driver_headers)
    assert export.status_code == 200
    body = export.json()
    assert body["includes_precise_location"] is False
    rows = body["rows"]
    assert len(rows) > 0
    row = rows[0]
    assert row["gps_latitude"] is None and row["gps_longitude"] is None
    assert row["accel_x"] == 0.1 and row["heart_rate"] == 72
    assert row["rpm"] is None  # never sent -- stays null, not coerced to 0
    pseudonym = row["pseudonymous_participant_id"]
    assert pseudonym.startswith("P-") and str(started.json().get("user_id", "")) not in pseudonym
    # Only the post-consent frames produced samples -- not the pre-consent ones.
    assert len(rows) == 6
    # "synthetic_test" has no assigned external device -- marked explicitly,
    # not silently null and not guessed from an unrelated assignment.
    assert row["device_id"] == "not_applicable:source_has_no_external_device"

    # A driver cannot request precise location, even on their own session.
    gated = client.get(
        f"/app/api/sessions/{session_id}/export.json?include_location=true", headers=driver_headers
    )
    assert gated.status_code == 403

    # A different driver cannot export someone else's session.
    manager_headers = sign_in(client, "+971501110002")
    client.post(
        "/app/api/drivers", headers=manager_headers,
        json={"name": "Driver Two", "phone": "+971501110099"},
    )
    driver_two_headers = sign_in(client, "+971501110099")
    forbidden = client.get(f"/app/api/sessions/{session_id}/export.json", headers=driver_two_headers)
    assert forbidden.status_code == 403

    # account_manager can opt into precise location.
    unlocked = client.get(
        f"/app/api/sessions/{session_id}/export.json?include_location=true", headers=manager_headers
    )
    assert unlocked.status_code == 200
    unlocked_row = unlocked.json()["rows"][0]
    assert unlocked_row["gps_latitude"] == 25.3 and unlocked_row["gps_longitude"] == 55.4
    # Pseudonym is stable across exports/roles, not derived from who's asking.
    assert unlocked_row["pseudonymous_participant_id"] == pseudonym

    csv_export = client.get(f"/app/api/sessions/{session_id}/export.csv", headers=manager_headers)
    assert csv_export.status_code == 200
    assert csv_export.headers["content-type"].startswith("text/csv")
    lines = csv_export.text.strip().splitlines()
    assert len(lines) >= 2  # header + at least one data row
    assert "pseudonymous_participant_id" in lines[0]

    # Revocation stops FUTURE research recording and blocks export again --
    # it does not retroactively delete the samples written while consent was
    # active (preserve, don't destroy, past research data on a revoke).
    revoked = client.post(
        "/app/api/driver/research-consent", headers=driver_headers, json={"granted": False}
    )
    assert revoked.status_code == 200 and revoked.json()["granted"] is False
    send_frames(start_step=20)
    assert client.get(
        f"/app/api/sessions/{session_id}/export.json", headers=driver_headers
    ).status_code == 403
    # `people` is the live Database instance (see its fixture) -- reading it
    # directly, bypassing the now-blocked export route, confirms the earlier
    # rows are still there on disk, just no longer exportable.
    assert len(people.session_samples(session_id)) == 6


def test_research_export_carries_camera_derived_features(client, people, monkeypatch):
    """Export verification: camera-derived fields (eyes-on-road, traffic,
    pedestrians) reach the export, not just the sensor/physiology columns."""
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={})
    stream_id, session_id = started.json()["stream_id"], started.json()["id"]
    client.post("/app/api/driver/research-consent", headers=headers, json={"granted": True})

    # edge_mode (iOS + risk_active) produces a prediction on the FIRST frame,
    # no warmup loop needed, and is where eyes_on_road/traffic_level/etc live.
    frame = {
        "timestamp": time.time(),
        "source": "ios",
        "stream_id": stream_id,
        "risk_active": True,
        "risk_score": 42,
        "eyes_on_road": False,
        "traffic_level": "moderate",
        "pedestrians_detected": True,
        "pedestrian_count": 2,
        "lead_closing_fast": True,
    }
    assert client.post("/ingest", json=frame).status_code == 200

    export = client.get(f"/app/api/sessions/{session_id}/export.json", headers=headers).json()
    assert len(export["rows"]) > 0
    row = export["rows"][0]
    assert row["eyes_on_road"] is False
    assert row["traffic_level"] == "moderate"
    assert row["pedestrians_detected"] is True
    assert row["pedestrian_count"] == 2
    assert row["lead_closing_fast"] is True
    assert row["overall_risk_index"] == pytest.approx(0.42)


def test_drivers_only_see_their_own_sessions(client, people):
    driver = sign_in(client, "+971501110001")
    manager = sign_in(client, "+971501110002")
    created = client.post("/app/api/driver/session/start", headers=driver, json={}).json()

    other = people.create_user("+971501110009", "Driver Two", "driver")
    hidden = people.start_session(other["id"], "other-stream")

    assert [s["id"] for s in client.get("/app/api/sessions", headers=driver).json()] == [
        created["id"]
    ]
    assert client.get(
        f"/app/api/sessions/{hidden['id']}", headers=driver
    ).status_code == 403
    assert client.get(f"/app/api/sessions/{hidden['id']}", headers=manager).status_code == 200


def test_manager_enrols_a_driver_with_a_kit(client, people):
    admin = sign_in(client, "+971501110003")
    adapter = client.post(
        "/app/api/devices", headers=admin, json={"kind": "adapter", "identifier": "ELM-010"}
    ).json()

    manager = sign_in(client, "+971501110002")
    enrolled = client.post(
        "/app/api/drivers",
        headers=manager,
        json={"phone": "+971501110020", "name": "New Driver", "adapter_id": adapter["id"]},
    )
    assert enrolled.status_code == 201
    assert enrolled.json()["kit"][0]["identifier"] == "ELM-010"
    # The new driver can sign in immediately.
    assert client.get(
        "/app/api/me", headers=sign_in(client, "+971501110020")
    ).json()["role"] == "driver"


def test_threshold_updates_are_validated(client, people):
    headers = sign_in(client, "+971501110003")
    ok = client.put(
        "/app/api/settings", headers=headers, json={"values": {"risk_high_threshold": 0.6}}
    )
    assert ok.status_code == 200 and ok.json()["risk_high_threshold"] == 0.6
    bad = client.put(
        "/app/api/settings", headers=headers, json={"values": {"risk_medium_threshold": 0.95}}
    )
    assert bad.status_code == 422
    assert client.get("/app/api/settings", headers=headers).json()["risk_high_threshold"] == 0.6


def test_administrator_cannot_lock_themselves_out(client, people):
    headers = sign_in(client, "+971501110003")
    me = client.get("/app/api/me", headers=headers).json()
    assert client.patch(
        f"/app/api/users/{me['id']}", headers=headers, json={"active": False}
    ).status_code == 409
    assert client.patch(
        f"/app/api/users/{me['id']}", headers=headers, json={"role": "driver"}
    ).status_code == 409


def test_collision_alert_can_be_cancelled_by_the_driver(client, people):
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    raised = client.post(
        "/app/api/driver/collision",
        headers=headers,
        json={"stream_id": started["stream_id"], "latitude": 25.3, "longitude": 55.4},
    )
    assert raised.status_code == 200 and raised.json()["status"] == "pending"

    state = client.get("/app/api/driver/state", headers=headers).json()
    assert state["pending_collision"]["id"] == raised.json()["id"]

    cancelled = client.post(f"/app/api/alerts/{raised.json()['id']}/cancel", headers=headers)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"


def test_one_driver_cannot_act_on_another_drivers_alert(client, people):
    first = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=first, json={}).json()
    alert = client.post(
        "/app/api/driver/collision", headers=first, json={"stream_id": started["stream_id"]}
    ).json()

    people.create_user("+971501110030", "Driver Three", "driver")
    second = sign_in(client, "+971501110030")
    assert client.post(
        f"/app/api/alerts/{alert['id']}/cancel", headers=second
    ).status_code == 403
    assert client.get("/app/api/alerts", headers=second).json() == []


def test_logout_invalidates_the_token(client, people):
    headers = sign_in(client, "+971501110001")
    assert client.post("/app/api/auth/logout", headers=headers).status_code == 200
    assert client.get("/app/api/me", headers=headers).status_code == 401
