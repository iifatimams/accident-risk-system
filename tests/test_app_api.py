"""Contract tests for the product API: login, role gating and the session flow."""

import time

import pytest
from fastapi.testclient import TestClient

from accident_risk.api.main import create_app
from accident_risk.inference.engine import InferenceEngine


@pytest.fixture
def app(tmp_path, models):
    return create_app(InferenceEngine(*models), database_path=tmp_path / "app.sqlite3")


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def sign_in(client, phone):
    """Complete the two-step phone login and return an auth header.

    The code is read via request_code_for_operator directly -- the SAME
    trusted path the loopback-only operator console (api/operator.py) uses
    -- not via the public HTTP request-code response, which never
    contains it (see test_login_delivery.py)."""
    app_state = client.app.state
    requested = app_state.auth.request_code_for_operator(phone)
    verified = client.post(
        "/app/api/auth/verify", json={"phone": phone, "code": requested["code"]}
    )
    assert verified.status_code == 200, verified.text
    return {"Authorization": f"Bearer {verified.json()['token']}"}


@pytest.fixture
def people(app):
    database = app.state.db
    database.create_user("+971501110001", "Driver One", "driver")
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


def test_driver_cannot_reach_administrator_routes(client, people):
    headers = sign_in(client, "+971501110001")
    assert client.get("/app/api/me", headers=headers).json()["role"] == "driver"
    assert client.get("/app/api/live", headers=headers).status_code == 403
    assert client.get("/app/api/drivers", headers=headers).status_code == 403
    assert client.get("/app/api/users", headers=headers).status_code == 403
    assert client.put(
        "/app/api/settings", headers=headers, json={"values": {"risk_high_threshold": 0.6}}
    ).status_code == 403


def test_administrator_can_reach_all_staff_routes(client, people):
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


def test_full_sample_is_recorded_unconditionally_with_weather_and_pose(client, people, monkeypatch):
    """The full per-frame record has no consent gate -- every session gets
    one -- and carries the raw head-pose/eye diagnostics and weather context
    alongside the vehicle and physiology fields, not just what feeds the
    live risk score."""
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    headers = sign_in(client, "+971501110001")
    me = client.get("/app/api/me", headers=headers).json()
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "ios", "stream_id": started["stream_id"],
        "risk_active": True, "risk_score": 12.5,
        "vehicle_speed": 15.0, "heart_rate": 80,
        "debug_pitch": -0.3, "debug_yaw": 0.1, "debug_roll": 0.05, "debug_ear": 0.22,
        "debug_face_detected": True, "camera_status": "active",
        "eyes_on_road": False, "eyes_closed_sec": 1.2,
        "weather_multiplier": 1.25, "weather_condition_code": 8.0, "weather_is_live": True,
    })
    session_id = people.active_session(me["id"])["id"]
    samples = people.session_samples(session_id)
    assert len(samples) == 1
    sample = samples[0]
    assert sample["debug_pitch"] == -0.3
    assert sample["debug_yaw"] == 0.1
    assert sample["debug_roll"] == 0.05
    assert sample["debug_ear"] == 0.22
    assert sample["debug_face_detected"] is True
    assert sample["camera_status"] == "active"
    assert sample["eyes_on_road"] is False
    assert sample["eyes_closed_sec"] == 1.2
    assert sample["weather_multiplier"] == 1.25
    assert sample["weather_condition_code"] == 8.0
    assert sample["weather_is_live"] is True
    assert sample["heart_rate"] == 80


def test_drivers_only_see_their_own_sessions(client, people):
    driver = sign_in(client, "+971501110001")
    admin = sign_in(client, "+971501110003")
    created = client.post("/app/api/driver/session/start", headers=driver, json={}).json()

    other = people.create_user("+971501110009", "Driver Two", "driver")
    hidden = people.start_session(other["id"], "other-stream")

    assert [s["id"] for s in client.get("/app/api/sessions", headers=driver).json()] == [
        created["id"]
    ]
    assert client.get(
        f"/app/api/sessions/{hidden['id']}", headers=driver
    ).status_code == 403
    assert client.get(f"/app/api/sessions/{hidden['id']}", headers=admin).status_code == 200


def test_administrator_enrols_a_driver_with_a_kit(client, people):
    admin = sign_in(client, "+971501110003")
    adapter = client.post(
        "/app/api/devices", headers=admin, json={"kind": "adapter", "identifier": "ELM-010"}
    ).json()

    enrolled = client.post(
        "/app/api/drivers",
        headers=admin,
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


def test_collision_alert_can_be_cancelled_by_the_driver(client, people, monkeypatch):
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "gps_latitude": 25.3, "gps_longitude": 55.4,
    })
    raised = client.get("/app/api/alerts", headers=headers).json()[0]
    assert raised["status"] == "awaiting_confirmation"

    state = client.get("/app/api/driver/state", headers=headers).json()
    assert state["pending_collision"]["id"] == raised["id"]

    cancelled = client.post(f"/app/api/alerts/{raised['id']}/dismiss", headers=headers)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "dismissed"


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


def test_client_incident_id_retry_after_session_ended_still_resolves_to_existing_alert(
    client, people
):
    """Checkpoint resume task #3: a retry can arrive after the stream was
    released (session stopped), e.g. the app relaunched and re-registered
    the same still-pending client_incident_id, or an offline-queued upload
    simply lands late. SessionRecorder._sessions no longer has this
    stream_id at all once the session ends -- this must still resolve to
    the SAME alert via the database, not silently return null or create a
    user_id=None row."""
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    stream_id = started["stream_id"]

    first = client.post("/ingest", json={
        "timestamp": time.time(), "source": "ios", "stream_id": stream_id,
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "client_incident_id": "test-incident-after-session-end",
    })
    assert first.status_code == 200
    original_alert_id = first.json()["collision_alert_id"]
    assert original_alert_id is not None

    client.post("/app/api/driver/session/stop", headers=headers, json={})

    retry = client.post("/ingest", json={
        "timestamp": time.time(), "source": "ios", "stream_id": stream_id,
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "client_incident_id": "test-incident-after-session-end",
    })
    assert retry.status_code == 200
    assert retry.json()["collision_alert_id"] == original_alert_id

    alerts = client.get("/app/api/alerts", headers=headers).json()
    assert len(alerts) == 1


def test_client_incident_id_dedup_survives_server_restart(tmp_path, models):
    """Checkpoint resume task #3: the previous dedup cache
    (SessionRecorder._incident_ids_seen) was an in-process dict, lost on
    every server restart -- a retry arriving after a restart would create a
    SECOND alert for the same real event. Simulates a restart by creating
    a brand-new app/InferenceEngine/SessionRecorder against the SAME sqlite
    file, with no in-memory state carried over at all."""
    db_path = tmp_path / "restart_test.sqlite3"

    app1 = create_app(InferenceEngine(*models), database_path=db_path)
    with TestClient(app1) as client1:
        app1.state.db.create_user("+971501110001", "Driver One", "driver")
        headers = sign_in(client1, "+971501110001")
        started = client1.post(
            "/app/api/driver/session/start", headers=headers, json={}
        ).json()
        first = client1.post("/ingest", json={
            "timestamp": time.time(), "source": "ios", "stream_id": started["stream_id"],
            "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
            "client_incident_id": "test-incident-survives-restart",
        })
        assert first.status_code == 200
        original_alert_id = first.json()["collision_alert_id"]
        assert original_alert_id is not None

    # A fresh app -- fresh InferenceEngine, fresh SessionRecorder, nothing
    # registered, nothing cached -- opened against the SAME database file.
    app2 = create_app(InferenceEngine(*models), database_path=db_path)
    with TestClient(app2) as client2:
        retry = client2.post("/ingest", json={
            "timestamp": time.time(), "source": "ios", "stream_id": "a-different-stream-id",
            "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
            "client_incident_id": "test-incident-survives-restart",
        })
        assert retry.status_code == 200
        assert retry.json()["collision_alert_id"] == original_alert_id

        headers2 = sign_in(client2, "+971501110001")
        alerts = client2.get("/app/api/alerts", headers=headers2).json()
        assert len(alerts) == 1


def test_concurrent_get_or_create_collision_alert_creates_only_one_row(people):
    """Checkpoint resume task #3: the old check-then-create was not atomic
    -- two concurrent callers for the SAME client_incident_id (the
    immediate on-device report and a retry-timer upload landing before
    either response returns) could both pass a "not seen yet" check before
    either recorded it. Hits Database.get_or_create_collision_alert
    directly from real concurrent threads (not through the HTTP/pipeline
    layer, which has its own unrelated frame-ordering constraints) and
    asserts exactly one row is ever created, with every caller agreeing on
    its id."""
    import concurrent.futures

    user_id = people.get_user_by_phone("+971501110001")["id"]

    def attempt(_):
        incident, created = people.get_or_create_collision_alert(
            "test-incident-concurrent-retry", user_id, None,
            status="suspected", detail={"source": "automatic_detector"},
        )
        return incident["id"], created

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(attempt, range(16)))

    ids = {r[0] for r in results}
    assert len(ids) == 1, f"expected every caller to agree on one alert id, got {ids}"
    created_flags = [r[1] for r in results]
    assert created_flags.count(True) == 1, "exactly one caller must have actually created the row"

    with people.connect() as connection:
        count = connection.execute(
            "SELECT COUNT(*) AS n FROM alerts WHERE client_incident_id = ?",
            ("test-incident-concurrent-retry",),
        ).fetchone()["n"]
    assert count == 1


def test_different_drivers_with_distinct_client_incident_ids_each_get_their_own_alert(
    client, people
):
    """Sanity check in the other direction: dedup must be keyed on the
    actual client_incident_id, not accidentally merge unrelated incidents
    from different drivers/streams."""
    headers1 = sign_in(client, "+971501110001")
    client.app.state.db.create_user("+971501110004", "Driver Two", "driver")
    headers2 = sign_in(client, "+971501110004")
    started1 = client.post("/app/api/driver/session/start", headers=headers1, json={}).json()
    started2 = client.post("/app/api/driver/session/start", headers=headers2, json={}).json()

    r1 = client.post("/ingest", json={
        "timestamp": time.time(), "source": "ios", "stream_id": started1["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "client_incident_id": "driver-one-incident",
    })
    r2 = client.post("/ingest", json={
        "timestamp": time.time(), "source": "ios", "stream_id": started2["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
        "client_incident_id": "driver-two-incident",
    })
    assert r1.status_code == 200 and r2.status_code == 200
    id1, id2 = r1.json()["collision_alert_id"], r2.json()["collision_alert_id"]
    assert id1 is not None and id2 is not None and id1 != id2

    alerts1 = client.get("/app/api/alerts", headers=headers1).json()
    alerts2 = client.get("/app/api/alerts", headers=headers2).json()
    assert len(alerts1) == 1 and len(alerts2) == 1


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


def test_confirm_and_cancel_are_also_lifecycle_enforced_once_an_incident_is_in_the_rich_vocabulary(
    client, people
):
    """Checkpoint resume task #4: confirm/cancel were NEVER lifecycle-
    enforced (only dismiss/resolve were), so a stray "confirm" could reopen
    an already-dismissed/resolved automatic incident. Once an incident has
    entered the rich lifecycle (anything past awaiting_confirmation),
    confirm/cancel must be enforced exactly like dismiss/resolve."""
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
    })
    incident = client.get("/app/api/alerts", headers=headers).json()[0]
    assert incident["status"] == "awaiting_confirmation"

    # "cancel" is not a reachable state from awaiting_confirmation at all
    # (only confirmed/dismissed/unconfirmed_timeout are) -- must be rejected.
    bad_cancel = client.post(f"/app/api/alerts/{incident['id']}/cancel", headers=headers)
    assert bad_cancel.status_code == 409

    dismissed = client.post(f"/app/api/alerts/{incident['id']}/dismiss", headers=headers)
    assert dismissed.status_code == 200 and dismissed.json()["status"] == "dismissed"

    # A dismissed incident must not be reopened by "confirm" -- this is
    # the exact gap checkpoint task #4 describes.
    stray_confirm = client.post(f"/app/api/alerts/{incident['id']}/confirm", headers=headers)
    assert stray_confirm.status_code == 409
    assert client.get("/app/api/alerts", headers=headers).json()[0]["status"] == "dismissed"

    resolved = client.post(f"/app/api/alerts/{incident['id']}/resolve", headers=headers)
    assert resolved.status_code == 200 and resolved.json()["status"] == "resolved"

    # A resolved incident is terminal for EVERY action, not just resolve
    # again -- confirm/cancel/dismiss must all be rejected too.
    for action in ("confirm", "cancel", "dismiss", "resolve"):
        rejected = client.post(f"/app/api/alerts/{incident['id']}/{action}", headers=headers)
        assert rejected.status_code == 409, f"{action} on a resolved incident must be rejected"


def test_legacy_pending_alert_confirm_and_cancel_remain_unrestricted(client, app, people, monkeypatch):
    """The legacy high_risk vocabulary (status "pending") was never part of
    INCIDENT_LIFECYCLE and must stay exactly as permissive as before this
    session's fix -- only the richer automatic-incident vocabulary gained
    enforcement on confirm/cancel."""
    clock = [5_000_000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    headers = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    app.state.recorder.register(started["stream_id"], started["id"])
    alert = app.state.db.create_alert(
        user_id=app.state.db.get_user_by_phone("+971501110001")["id"],
        kind="high_risk", session_id=started["id"], risk_index=0.9,
    )
    assert alert["status"] == "pending"

    confirmed = client.post(f"/app/api/alerts/{alert['id']}/confirm", headers=headers)
    assert confirmed.status_code == 200 and confirmed.json()["status"] == "confirmed"
    # Legacy behavior: even a second "confirm" (or "cancel") on an already-
    # confirmed legacy alert is still allowed -- this was never enforced
    # and this fix must not start enforcing it now.
    cancelled = client.post(f"/app/api/alerts/{alert['id']}/cancel", headers=headers)
    assert cancelled.status_code == 200 and cancelled.json()["status"] == "cancelled"


def test_unconfirmed_collision_incident_times_out(client, app, people, monkeypatch):
    clock = [3_000_000.0]
    monkeypatch.setattr("time.time", lambda: clock[0])
    headers = sign_in(client, "+971501110001")

    # A linked family member, so the timeout notification's wording can be
    # checked directly -- it must never claim a verified accident.
    people.create_user("+971500000555", "Watchful Uncle", "driver")
    uncle_headers = sign_in(client, "+971500000555")
    request = client.post(
        "/app/api/driver/family-request", headers=headers, json={"phone": "+971500000555"}
    ).json()
    client.post(
        f"/app/api/driver/family-links/{request['id']}/respond",
        headers=uncle_headers, json={"accept": True},
    )

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


def test_family_link_request_accept_and_remove(client, people):
    """Explicit consent: a link only ever becomes accepted by the
    RECIPIENT's own response -- never assumed, never requester-granted on
    their behalf."""
    headers = sign_in(client, "+971501110001")
    people.create_user("+971500000111", "Aunt Mona", "driver")
    aunt_headers = sign_in(client, "+971500000111")

    requested = client.post(
        "/app/api/driver/family-request", headers=headers, json={"phone": "+971500000111"}
    )
    assert requested.status_code == 201
    link = requested.json()
    assert link["status"] == "requested"

    # Not yet accepted.
    assert client.get("/app/api/driver/family-links", headers=headers).json()[0]["status"] == "requested"

    # Only the recipient can respond -- the requester trying to "accept"
    # their own request must not be able to.
    forbidden = client.post(
        f"/app/api/driver/family-links/{link['id']}/respond", headers=headers, json={"accept": True}
    )
    assert forbidden.status_code == 404

    accepted = client.post(
        f"/app/api/driver/family-links/{link['id']}/respond",
        headers=aunt_headers, json={"accept": True},
    )
    assert accepted.status_code == 200 and accepted.json()["status"] == "accepted"

    listed = client.get("/app/api/driver/family-links", headers=headers).json()
    assert listed[0]["status"] == "accepted"

    # Already answered -- not reusable to re-authorize or re-decide.
    reused = client.post(
        f"/app/api/driver/family-links/{link['id']}/respond", headers=aunt_headers, json={"accept": False}
    )
    assert reused.status_code == 404

    removed = client.post(f"/app/api/driver/family-links/{link['id']}/remove", headers=headers)
    assert removed.status_code == 200 and removed.json()["status"] == "removed"
    assert client.get("/app/api/driver/family-links", headers=headers).json() == []


def test_one_driver_cannot_remove_another_drivers_family_link(client, people):
    first = sign_in(client, "+971501110001")
    people.create_user("+971500000222", "Dad", "driver")
    link = client.post(
        "/app/api/driver/family-request", headers=first, json={"phone": "+971500000222"}
    ).json()

    people.create_user("+971501110031", "Driver Four", "driver")
    second = sign_in(client, "+971501110031")
    forbidden = client.post(f"/app/api/driver/family-links/{link['id']}/remove", headers=second)
    assert forbidden.status_code == 404


def test_unregistered_phone_creates_a_new_driver_account_with_devices(client, people):
    """Requesting a family link with a phone number that isn't a driver yet
    creates a real, loggable-in driver account for them, with its own
    device assignment -- not just a lightweight, no-login contact."""
    headers = sign_in(client, "+971501110001")

    missing_name = client.post(
        "/app/api/driver/family-request", headers=headers, json={"phone": "+971500000333"}
    )
    assert missing_name.status_code == 422

    created = client.post(
        "/app/api/driver/family-request", headers=headers,
        json={
            "phone": "+971500000333", "name": "Grandma",
            "obd_identifier": "OBD-GMA-01", "wearable_identifier": "WHOOP-GMA-01",
        },
    )
    assert created.status_code == 201

    new_driver = people.get_user_by_phone("+971500000333")
    assert new_driver is not None and new_driver["role"] == "driver"
    kit = people.kit_for_user(new_driver["id"])
    assert {d["identifier"] for d in kit} == {"OBD-GMA-01", "WHOOP-GMA-01"}

    # The new account can sign in and see the pending request itself.
    grandma_headers = sign_in(client, "+971500000333")
    pending = client.get("/app/api/driver/family-links", headers=grandma_headers).json()
    assert len(pending) == 1 and pending[0]["status"] == "requested"


def test_family_page_serves_for_any_token(client):
    """The static family page itself doesn't validate the token (the POST
    to /app/api/family/accept-or-acknowledge does) -- it just needs to
    serve so a contact's link actually opens something, closing the
    previous gap where no frontend existed for this use case at all."""
    response = client.get("/family/some-invite-token")
    assert response.status_code == 200
    assert "RoadSense" in response.text


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

    people.create_user("+971500000333", "Nearby Contact", "driver")
    people.create_user("+971500000444", "Distant Contact", "driver")
    near_headers = sign_in(client, "+971500000333")
    far_headers = sign_in(client, "+971500000444")

    near = client.post(
        "/app/api/driver/family-request", headers=headers, json={"phone": "+971500000333"}
    ).json()
    far = client.post(
        "/app/api/driver/family-request", headers=headers, json={"phone": "+971500000444"}
    ).json()
    client.post(
        f"/app/api/driver/family-links/{near['id']}/respond", headers=near_headers, json={"accept": True}
    )
    client.post(
        f"/app/api/driver/family-links/{far['id']}/respond", headers=far_headers, json={"accept": True}
    )
    # The driver controls what THEY disclose to each linked family member,
    # independently per link -- near sees location, far does not.
    client.patch(
        f"/app/api/driver/family-links/{near['id']}/permissions", headers=headers,
        json={"share_location": True},
    )
    client.patch(
        f"/app/api/driver/family-links/{far['id']}/permissions", headers=headers,
        json={"share_location": False},
    )

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

    # Who was told, and whether they've acknowledged -- read directly off the
    # database, since no screen in the app surfaces this list (the HTTP route
    # that used to expose it was unused and removed).
    notifications = people.list_family_notifications(incident["id"])
    assert len(notifications) == 2
    assert all(n["acknowledged_at"] is None for n in notifications)
    ack = client.post("/app/api/family/acknowledge", json={"token": notifications[0]["acknowledge_token"]})
    assert ack.status_code == 200 and ack.json()["status"] == "acknowledged"
    refreshed = people.list_family_notifications(incident["id"])
    assert sum(1 for n in refreshed if n["acknowledged_at"] is not None) == 1


def test_administrator_confirmed_incident_notifies_family_with_different_wording(
    client, app, people, monkeypatch
):
    """An administrator can confirm another driver's alert
    (role_allows(..., "administrator")) -- the family message must say so,
    never attribute the administrator's action to the driver as if the
    driver confirmed it themselves."""
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    driver_headers = sign_in(client, "+971501110001")
    admin_headers = sign_in(client, "+971501110003")

    people.create_user("+971500000666", "Watchful Aunt", "driver")
    aunt_headers = sign_in(client, "+971500000666")
    link = client.post(
        "/app/api/driver/family-request", headers=driver_headers, json={"phone": "+971500000666"}
    ).json()
    client.post(
        f"/app/api/driver/family-links/{link['id']}/respond", headers=aunt_headers, json={"accept": True}
    )

    started = client.post("/app/api/driver/session/start", headers=driver_headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
    })
    incident = client.get("/app/api/alerts", headers=driver_headers).json()[0]

    confirmed = client.post(f"/app/api/alerts/{incident['id']}/confirm", headers=admin_headers)
    assert confirmed.status_code == 200 and confirmed.json()["status"] == "confirmed"

    sent = app.state.notification_sender.sent
    assert len(sent) == 1
    assert "administrator, not the driver" in sent[0]["message"]
    assert "Driver One confirmed they were in a collision" not in sent[0]["message"]


def test_accepting_a_request_does_not_itself_grant_location_access(client, people):
    """Acceptance is the RECIPIENT's consent to being notified at all.
    share_location is each side's own separate, independent choice --
    neither side's acceptance implies the other turning location on."""
    headers = sign_in(client, "+971501110001")
    people.create_user("+971500000666", "Grandpa", "driver")
    grandpa_headers = sign_in(client, "+971500000666")
    link = client.post(
        "/app/api/driver/family-request", headers=headers, json={"phone": "+971500000666"}
    ).json()
    assert link["requester_share_location"] is False  # off by default

    accepted = client.post(
        f"/app/api/driver/family-links/{link['id']}/respond", headers=grandpa_headers, json={"accept": True}
    )
    assert accepted.status_code == 200
    # Acceptance alone must not have flipped it.
    listed = client.get("/app/api/driver/family-links", headers=headers).json()[0]
    assert listed["requester_share_location"] is False

    # Only the requester, explicitly, can turn on THEIR OWN half.
    updated = client.patch(
        f"/app/api/driver/family-links/{link['id']}/permissions", headers=headers,
        json={"share_location": True},
    )
    assert updated.status_code == 200 and updated.json()["requester_share_location"] is True
    # notify_accidents is configured independently and is untouched by that call.
    assert updated.json()["requester_notify_accidents"] is True
    # The recipient's own half is completely untouched by the requester's change.
    assert updated.json()["recipient_share_location"] is False


def test_removed_family_link_is_not_notified(client, app, people, monkeypatch):
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    headers = sign_in(client, "+971501110001")
    people.create_user("+971500000777", "Ex Contact", "driver")
    ex_headers = sign_in(client, "+971500000777")
    link = client.post(
        "/app/api/driver/family-request", headers=headers, json={"phone": "+971500000777"}
    ).json()
    client.post(
        f"/app/api/driver/family-links/{link['id']}/respond", headers=ex_headers, json={"accept": True}
    )
    client.post(f"/app/api/driver/family-links/{link['id']}/remove", headers=headers)

    started = client.post("/app/api/driver/session/start", headers=headers, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
    })
    incident = client.get("/app/api/alerts", headers=headers).json()[0]
    client.post(f"/app/api/alerts/{incident['id']}/confirm", headers=headers)
    assert app.state.notification_sender.sent == []


def test_one_driver_cannot_act_on_another_drivers_alert(client, people, monkeypatch):
    monkeypatch.setattr(
        "accident_risk.api.main.MapContextResolver.is_in_roundabout",
        lambda self, latitude, longitude: False,
    )
    first = sign_in(client, "+971501110001")
    started = client.post("/app/api/driver/session/start", headers=first, json={}).json()
    client.post("/ingest", json={
        "timestamp": time.time(), "source": "synthetic_test", "stream_id": started["stream_id"],
        "accel_x": 25.0, "accel_y": 0.0, "accel_z": 5.0,
    })
    alert = client.get("/app/api/alerts", headers=first).json()[0]

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
