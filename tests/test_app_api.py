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
        demo_access_key="synthetic-test-pairing-key-00000000",
        demo_phone_numbers=[f"+971501110{i:03}" for i in (1, 2, 3, 20, 30, 31, 99)],
    )


@pytest.fixture
def client(app):
    with TestClient(app, headers={"X-RoadSense-Demo-Key": "synthetic-test-pairing-key-00000000"}) as test_client:
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


def test_automatic_collision_detection_is_independent_and_suppresses_duplicates(
    client, people
):
    """Independent collision detection: synthetic fixtures only -- this is a
    rule-based kinematic trigger, not validated against a real impact, and
    this test does not claim otherwise."""
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    stream_id = started["stream_id"]

    def send(accel_x, accel_y, accel_z):
        return client.post("/ingest", json={
            "timestamp": time.time(),
            "source": "synthetic_test",
            "stream_id": stream_id,
            "accel_x": accel_x, "accel_y": accel_y, "accel_z": accel_z,
        })

    # Ordinary driving and a hard brake: no incident.
    assert send(0.1, 0.1, 9.8).status_code == 200
    assert send(11.0, 0.2, 9.8).status_code == 200
    assert client.get("/app/api/alerts", headers=headers).json() == []

    # A simulated impact spike: creates a suspected-then-awaiting-confirmation
    # incident automatically, independent of any risk/prediction score --
    # the frame above carries no risk fields at all.
    assert send(25.0, 0.0, 5.0).status_code == 200
    alerts = client.get("/app/api/alerts", headers=headers).json()
    assert len(alerts) == 1
    incident = alerts[0]
    assert incident["kind"] == "collision"
    assert incident["status"] == "awaiting_confirmation"
    assert incident["detail"]["source"] == "automatic_detector"
    assert incident["detail"]["model_version"]
    assert incident["detail"]["detection_score"] >= 1.0
    assert incident["detail"]["onset_estimate"] is not None

    # A second impact-level spike right after is suppressed as a duplicate.
    assert send(25.0, 0.0, 5.0).status_code == 200
    assert len(client.get("/app/api/alerts", headers=headers).json()) == 1


def test_client_incident_id_survives_offline_retry_after_the_cooldown_window(
    client, people, monkeypatch
):
    """Synthetic fixture: simulates the iOS app's offline retry -- the SAME
    client_incident_id re-uploaded long after the normal cooldown window
    must still resolve to the ORIGINAL incident, not create a second one.
    This is what makes reconnection-after-offline duplicate-safe."""
    clock = [4_000_000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()

    first = client.post("/ingest", json={
        "timestamp": clock[0], "source": "ios", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "client_incident_id": "test-incident-abc123",
    })
    assert first.status_code == 200
    original_alert_id = first.json()["collision_alert_id"]
    assert original_alert_id is not None

    # The driver confirms while still "offline" from the server's point of
    # view (this call itself succeeds -- confirming simulates the moment
    # connectivity returns and the app syncs the locally-recorded decision).
    client.post(f"/app/api/alerts/{original_alert_id}/confirm", headers=headers)

    # Far past COLLISION_DETECTION_COOLDOWN_SECONDS (120s) -- the retried
    # upload of the exact same detected event, as an offline queue would
    # eventually resend it.
    clock[0] += 600.0
    retry = client.post("/ingest", json={
        "timestamp": clock[0], "source": "ios", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "client_incident_id": "test-incident-abc123",
    })
    assert retry.status_code == 200
    assert retry.json()["collision_alert_id"] == original_alert_id

    alerts = client.get("/app/api/alerts", headers=headers).json()
    assert len(alerts) == 1
    assert alerts[0]["status"] == "confirmed"  # the retry did not reset it to awaiting_confirmation


def test_collision_incident_lifecycle_dismiss_resolve_and_invalid_transitions(
    client, people
):
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
    })
    incident = client.get("/app/api/alerts", headers=headers).json()[0]

    # Cannot resolve directly from awaiting_confirmation -- must go through
    # confirmed/dismissed/unconfirmed_timeout first.
    invalid = client.post(f"/app/api/alerts/{incident['id']}/resolve", headers=headers)
    assert invalid.status_code == 409

    dismissed = client.post(f"/app/api/alerts/{incident['id']}/dismiss", headers=headers)
    assert dismissed.status_code == 200 and dismissed.json()["status"] == "dismissed"

    resolved = client.post(f"/app/api/alerts/{incident['id']}/resolve", headers=headers)
    assert resolved.status_code == 200 and resolved.json()["status"] == "resolved"

    # A resolved incident is terminal -- no further transition is valid.
    again = client.post(f"/app/api/alerts/{incident['id']}/resolve", headers=headers)
    assert again.status_code == 409


def test_unconfirmed_collision_incident_times_out(client, app, people, monkeypatch):
    clock = [3_000_000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    headers = sign_in(client, "+971501110001")

    # A family contact, so the timeout notification's wording can be
    # checked directly -- it must never claim a verified accident.
    contact = client.post(
        "/app/api/driver/family-contacts", headers=headers,
        json={"name": "Watchful Uncle", "phone": "+971500000555"},
    ).json()
    client.post("/app/api/family/accept", json={"token": contact["invite_token"]})

    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": clock[0], "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
    })
    incident = client.get("/app/api/alerts", headers=headers).json()[0]
    assert incident["status"] == "awaiting_confirmation"

    # Still within the window: no change yet, and nothing sent to family.
    clock[0] += 60.0
    still_waiting = client.get("/app/api/alerts", headers=headers).json()[0]
    assert still_waiting["status"] == "awaiting_confirmation"
    assert app.state.notification_sender.sent == []

    # Past the timeout window: the call-on-access sweep moves it AND
    # notifies family -- but the message must be honest that this is an
    # UNCONFIRMED sensor trigger, never worded as a verified accident.
    clock[0] += 180.0
    timed_out = client.get("/app/api/alerts", headers=headers).json()[0]
    assert timed_out["status"] == "unconfirmed_timeout"
    sent = app.state.notification_sender.sent
    assert len(sent) == 1
    message = sent[0]["message"].lower()
    assert "not been confirmed" in message or "unconfirmed" in message
    assert "confirmed they were in a collision" not in message


def test_family_contact_invite_accept_and_revoke(client, people):
    """Explicit consent: a contact only ever becomes notifiable by using
    their own invite_token -- never assumed, never driver-granted on their
    behalf."""
    headers = sign_in(client, "+971501110001")
    invited = client.post(
        "/app/api/driver/family-contacts", headers=headers,
        json={"name": "Aunt Mona", "phone": "+971500000111", "relationship": "aunt"},
    )
    assert invited.status_code == 201
    contact = invited.json()
    assert contact["status"] == "invited"
    token = contact["invite_token"]

    # Not yet notifiable: invited, not accepted.
    assert client.get("/app/api/driver/family-contacts", headers=headers).json()[0]["status"] == "invited"

    accepted = client.post("/app/api/family/accept", json={"token": token})
    assert accepted.status_code == 200 and accepted.json()["status"] == "accepted"

    listed = client.get("/app/api/driver/family-contacts", headers=headers).json()
    assert listed[0]["status"] == "accepted"

    # The same token cannot be used again (not reusable to re-authorize).
    reused = client.post("/app/api/family/accept", json={"token": token})
    assert reused.status_code == 404

    revoked = client.post(
        f"/app/api/driver/family-contacts/{contact['id']}/revoke", headers=headers
    )
    assert revoked.status_code == 200 and revoked.json()["status"] == "revoked"


def test_one_driver_cannot_revoke_another_drivers_family_contact(client, people):
    first = sign_in(client, "+971501110001")
    contact = client.post(
        "/app/api/driver/family-contacts", headers=first,
        json={"name": "Dad", "phone": "+971500000222"},
    ).json()

    people.create_user("+971501110031", "Driver Four", "driver")
    second = sign_in(client, "+971501110031")
    forbidden = client.post(
        f"/app/api/driver/family-contacts/{contact['id']}/revoke", headers=second
    )
    assert forbidden.status_code == 404


def test_confirmed_collision_notifies_family_respecting_location_consent(
    client, app, people, monkeypatch
):
    """Synthetic fixture only -- this is a local MockNotificationSender
    (see notifications.py); no real SMS/push/email was sent or could have
    been, and this test does not claim otherwise."""
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    headers = sign_in(client, "+971501110001")

    near = client.post(
        "/app/api/driver/family-contacts", headers=headers,
        json={"name": "Nearby Contact", "phone": "+971500000333", "share_location": True},
    ).json()
    far = client.post(
        "/app/api/driver/family-contacts", headers=headers,
        json={"name": "Distant Contact", "phone": "+971500000444", "share_location": False},
    ).json()
    client.post("/app/api/family/accept", json={"token": near["invite_token"]})
    client.post("/app/api/family/accept", json={"token": far["invite_token"]})

    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "gps_latitude": 25.3, "gps_longitude": 55.4,
    })
    incident = client.get("/app/api/alerts", headers=headers).json()[0]

    confirmed = client.post(f"/app/api/alerts/{incident['id']}/confirm", headers=headers)
    assert confirmed.status_code == 200 and confirmed.json()["status"] == "confirmed"

    sent = app.state.notification_sender.sent
    assert len(sent) == 2
    by_name = {record["contact_name"]: record["message"] for record in sent}
    assert "25.3" in by_name["Nearby Contact"]
    assert "25.3" not in by_name["Distant Contact"]

    # Calling confirm again must not double-notify either contact.
    client.post(f"/app/api/alerts/{incident['id']}/confirm", headers=headers)
    assert len(app.state.notification_sender.sent) == 2

    # The driver's own view of who was told, and whether they've acknowledged.
    notifications = client.get(
        f"/app/api/alerts/{incident['id']}/family-notifications", headers=headers
    ).json()
    assert len(notifications) == 2
    assert all(n["acknowledged_at"] is None for n in notifications)
    ack = client.post("/app/api/family/acknowledge", json={"token": notifications[0]["acknowledge_token"]})
    assert ack.status_code == 200 and ack.json()["status"] == "acknowledged"
    refreshed = client.get(
        f"/app/api/alerts/{incident['id']}/family-notifications", headers=headers
    ).json()
    assert sum(1 for n in refreshed if n["acknowledged_at"] is not None) == 1


def test_accepting_an_invite_does_not_itself_grant_location_access(client, people):
    """Acceptance is the CONTACT's consent to being notified at all.
    share_location is the DRIVER's separate, independent choice -- neither
    side's consent implies the other."""
    headers = sign_in(client, "+971501110001")
    contact = client.post(
        "/app/api/driver/family-contacts", headers=headers,
        json={"name": "Grandpa", "phone": "+971500000666"},  # share_location omitted -> False
    ).json()
    assert contact["share_location"] is False

    accepted = client.post("/app/api/family/accept", json={"token": contact["invite_token"]})
    assert accepted.status_code == 200
    # Acceptance alone must not have flipped it.
    listed = client.get("/app/api/driver/family-contacts", headers=headers).json()[0]
    assert listed["share_location"] is False

    # Only the driver, explicitly, can turn it on.
    updated = client.patch(
        f"/app/api/driver/family-contacts/{contact['id']}", headers=headers,
        json={"share_location": True},
    )
    assert updated.status_code == 200 and updated.json()["share_location"] is True
    # notify_accidents is configured independently and is untouched by that call.
    assert updated.json()["notify_accidents"] is True


def test_revoked_family_contact_is_not_notified(client, app, people, monkeypatch):
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    headers = sign_in(client, "+971501110001")
    contact = client.post(
        "/app/api/driver/family-contacts", headers=headers,
        json={"name": "Ex Contact", "phone": "+971500000777"},
    ).json()
    client.post("/app/api/family/accept", json={"token": contact["invite_token"]})
    client.post(f"/app/api/driver/family-contacts/{contact['id']}/revoke", headers=headers)

    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
    })
    incident = client.get("/app/api/alerts", headers=headers).json()[0]
    client.post(f"/app/api/alerts/{incident['id']}/confirm", headers=headers)
    assert app.state.notification_sender.sent == []


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
