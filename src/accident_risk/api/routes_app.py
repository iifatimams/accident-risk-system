"""Product API behind the operator and driver interface. Mounted at /app/api."""

import csv
import hashlib
import io
import json
import time
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from accident_risk.api.auth import AuthError, AuthService, Principal, bearer_from_header
from accident_risk.api.db import Database, role_allows
from accident_risk.api.notifications import MockNotificationSender, build_message
from accident_risk.api.recorder import SessionRecorder
from accident_risk.inference.engine import InferenceEngine
from accident_risk.schemas import SENSOR_FIELDS

SESSION_TIMEOUT_SECONDS = 180.0
EDGE_FRESHNESS_SECONDS = 10.0
UNCONFIRMED_INCIDENT_TIMEOUT_SECONDS = 180.0


def has_sensor_values(fields: dict[str, Any]) -> bool:
    """Don't mistake the frame's metadata for fresh sensor measurements."""
    return any(fields.get(name) is not None for name in SENSOR_FIELDS)


def movement_confirmed(fields: dict[str, Any]) -> bool:
    """Fallback for legacy streams without an explicit phone driving state.

    Do not treat an elevated heart rate or idling RPM as evidence of driving.
    """
    vehicle_speed = fields.get("vehicle_speed")
    gps_speed = fields.get("gps_speed")
    speed_mps = vehicle_speed if vehicle_speed is not None else gps_speed
    return speed_mps is not None and speed_mps >= 8.0 / 3.6


def speed_in_kmh(fields: dict[str, Any]) -> float | None:
    vehicle_speed = fields.get("vehicle_speed")
    gps_speed = fields.get("gps_speed")
    speed_mps = vehicle_speed if vehicle_speed is not None else gps_speed
    return speed_mps * 3.6 if speed_mps is not None else None


class CodeRequest(BaseModel):
    phone: str = Field(min_length=3, max_length=30)


class VerifyRequest(BaseModel):
    phone: str = Field(min_length=3, max_length=30)
    code: str = Field(min_length=4, max_length=10)


class StartSessionRequest(BaseModel):
    stream_id: str | None = Field(None, min_length=1, max_length=120)


class CollisionRequest(BaseModel):
    stream_id: str
    latitude: float | None = None
    longitude: float | None = None


class ResearchConsentRequest(BaseModel):
    granted: bool


class FamilyContactRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    phone: str = Field(min_length=3, max_length=30)
    relationship: str = Field("", max_length=60)
    notify_accidents: bool = True
    share_location: bool = False


class FamilyTokenRequest(BaseModel):
    token: str = Field(min_length=1, max_length=200)


class FamilyPermissionsRequest(BaseModel):
    notify_accidents: bool | None = None
    share_location: bool | None = None


class DriverRequest(BaseModel):
    phone: str = Field(min_length=3, max_length=30)
    name: str = Field(min_length=1, max_length=120)
    adapter_id: int | None = None
    wearable_id: int | None = None


class UserRequest(BaseModel):
    phone: str = Field(min_length=3, max_length=30)
    name: str = Field(min_length=1, max_length=120)
    role: str


class UserPatch(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=120)
    role: str | None = None
    active: bool | None = None


class DeviceRequest(BaseModel):
    kind: str
    identifier: str = Field(min_length=1, max_length=120)
    note: str = ""


class DevicePatch(BaseModel):
    identifier: str | None = Field(None, min_length=1, max_length=120)
    status: str | None = None
    assigned_to: int | None = None
    clear_assignment: bool = False


class SettingsRequest(BaseModel):
    values: dict[str, float]


def risk_level_for(index: float | None, thresholds: dict[str, float]) -> str | None:
    if index is None:
        return None
    if index >= thresholds["risk_critical_threshold"]:
        return "CRITICAL"
    if index >= thresholds["risk_high_threshold"]:
        return "HIGH"
    if index >= thresholds["risk_medium_threshold"]:
        return "MEDIUM"
    return "LOW"


# Research export (recording/export milestone). Keep in sync with the fields
# SessionRecorder._add_research_sample actually writes.
EXPORT_COLUMNS = (
    "ts", "pseudonymous_participant_id", "session_id", "device_id", "source",
    "speed_kmh", "rpm", "throttle_position", "engine_load", "steering_angle",
    "tire_pressure_fl", "tire_pressure_fr", "tire_pressure_rl", "tire_pressure_rr",
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    "gps_latitude", "gps_longitude", "gps_heading", "is_roundabout",
    "heart_rate", "spo2", "hrv_rmssd", "rr_interval_ms", "physiology_signal_quality",
    "eyes_on_road", "eyes_closed_sec", "traffic_level", "pedestrians_detected",
    "pedestrian_count", "lead_closing_fast",
    "event_detection_score", "vehicle_risk_score", "driver_state_risk",
    "overall_risk_index", "risk_level", "data_quality", "model_version",
    "quality_flags", "sensor_age_seconds", "inference_latency_ms", "warning_fired",
    "sensor_timestamps",
)


def pseudonymize_participant(user_id: int) -> str:
    """Stable, one-way per-participant id for research export -- never the raw
    user_id, name, or phone number. A fixed per-deployment string salt, not a
    cryptographic secret: good enough to avoid exporting a directly-identifying
    field, not a claim of formal anonymity."""
    digest = hashlib.sha256(f"roadsense-research-export:{user_id}".encode()).hexdigest()[:10]
    return f"P-{digest}"


def build_export_rows(
    session: dict[str, Any], samples: list[dict[str, Any]], include_location: bool
) -> list[dict[str, Any]]:
    """Flatten research samples for export: pseudonymous id (no name/phone),
    precise GPS blanked unless include_location, missing readings stay null
    rather than becoming zero, original per-sensor timestamps preserved."""
    pseudonym = pseudonymize_participant(session["user_id"])
    rows = []
    for sample in samples:
        row = {column: sample.get(column) for column in EXPORT_COLUMNS}
        row["pseudonymous_participant_id"] = pseudonym
        row["session_id"] = session["id"]
        if not include_location:
            row["gps_latitude"] = None
            row["gps_longitude"] = None
        rows.append(row)
    return rows


def create_router(
    database: Database,
    auth: AuthService,
    recorder: SessionRecorder,
    engine: InferenceEngine,
    edge_state: dict[str, dict[str, Any]] | None = None,
    notification_sender: MockNotificationSender | None = None,
) -> APIRouter:
    router = APIRouter(prefix="/app/api", tags=["application"])
    edge_state = edge_state if edge_state is not None else {}
    # MockNotificationSender only -- see notifications.py. No real SMS/push/
    # email provider is wired up; never report a notification from this as
    # really delivered.
    notification_sender = notification_sender if notification_sender is not None else MockNotificationSender()

    def notify_family_of_incident(alert: dict[str, Any]) -> None:
        """Called when a collision incident reaches a status family should
        know about. Respects each contact's own notify_accidents AND
        share_location permissions individually; duplicate-suppressed per
        (alert, contact) by the database's own UNIQUE constraint."""
        driver = database.get_user(alert["user_id"])
        if driver is None:
            return
        for contact in database.accepted_notifiable_contacts(alert["user_id"]):
            if database.already_notified(alert["id"], contact["id"]):
                continue
            message = build_message(
                driver["name"], alert["status"], contact["share_location"],
                alert.get("latitude"), alert.get("longitude"),
            )
            status = notification_sender.send(
                contact_phone=contact["phone"], contact_name=contact["name"], message=message
            )
            database.record_family_notification(alert["id"], contact["id"], status)

    def driving_state(stream_id: str, fields: dict[str, Any], now: float) -> bool | None:
        """True=driving; False=not driving; None=no recent confirmation.

        An edge-owned stream never falls back to a legacy predictor or RPM/HR.
        """
        state = edge_state.get(stream_id, {})
        if state.get("edge_owned"):
            if not 0 <= now - state.get("received_at", 0.0) <= EDGE_FRESHNESS_SECONDS:
                return None
            return state.get("risk_active")
        return movement_confirmed(fields) if has_sensor_values(fields) else None

    def current_risk(stream_id: str, fields: dict[str, Any], now: float) -> tuple[float | None, bool | None]:
        active = driving_state(stream_id, fields, now)
        prediction = engine.latest.get(stream_id)
        index = prediction.overall_risk_index if active is True and prediction else None
        return index, active

    def get_principal(authorization: Annotated[str | None, Header()] = None) -> Principal:
        try:
            return auth.resolve(bearer_from_header(authorization))
        except AuthError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    def require(minimum: str):
        def dependency(principal: Annotated[Principal, Depends(get_principal)]) -> Principal:
            if not role_allows(principal.role, minimum):
                raise HTTPException(403, "Your role does not permit this action")
            return principal
        return dependency

    any_user = Depends(get_principal)
    manager = Depends(require("account_manager"))
    admin = Depends(require("administrator"))

    def visible_session(session_id: int, principal: Principal) -> dict[str, Any]:
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        if not role_allows(principal.role, "account_manager") and (session["user_id"] != principal.user_id):
            raise HTTPException(403, "You can only review your own sessions")
        database.log_access(principal.user_id, principal.role, f"session:{session_id}")
        return session

    def close_stale_sessions() -> None:
        for stream_id in recorder.stale_streams(SESSION_TIMEOUT_SECONDS):
            session_id = recorder.session_for(stream_id)
            if session_id is not None:
                recorder.finish(session_id, stream_id, "timeout")

    def timeout_unconfirmed_incidents() -> None:
        """Call-on-access sweep (same pattern as close_stale_sessions): a
        collision incident nobody confirms or dismisses in time moves to
        unconfirmed_timeout rather than sitting open forever -- and family
        is notified, since the driver not responding is, if anything, MORE
        reason to let them know than a confirmed response would be."""
        for timed_out in database.timeout_unconfirmed_incidents(UNCONFIRMED_INCIDENT_TIMEOUT_SECONDS):
            notify_family_of_incident(timed_out)

    @router.get("/connection")
    def connection() -> dict[str, Any]:
        """Non-sensitive identity probe used by the physical phone's Settings screen."""
        return {"service": "roadsense", "api_version": 1, "demo_mode": auth.dev_mock_sms_delivery}

    @router.post("/auth/request-code")
    def request_code(
        payload: CodeRequest, response: Response,
        x_roadsense_demo_key: Annotated[str | None, Header()] = None,
    ) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        try:
            return auth.request_code(payload.phone, demo_key=x_roadsense_demo_key)
        except AuthError as exc:
            raise HTTPException(exc.status, exc.message) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/auth/verify")
    def verify(payload: VerifyRequest, response: Response) -> dict[str, Any]:
        response.headers["Cache-Control"] = "no-store"
        try:
            return auth.verify_code(payload.phone, payload.code)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        except AuthError as exc:
            raise HTTPException(exc.status, exc.message) from exc

    @router.post("/auth/logout")
    def logout(authorization: Annotated[str | None, Header()] = None) -> dict[str, bool]:
        token = bearer_from_header(authorization)
        if token:
            auth.revoke(token)
        return {"ok": True}

    @router.get("/me")
    def me(principal: Principal = any_user) -> dict[str, Any]:
        return {
            "id": principal.user_id,
            "name": principal.name,
            "phone": principal.phone,
            "role": principal.role,
            "kit": database.kit_for_user(principal.user_id),
            "thresholds": database.get_settings(),
        }

    @router.get("/driver/state")
    def driver_state(principal: Principal = any_user) -> dict[str, Any]:
        close_stale_sessions()
        timeout_unconfirmed_incidents()
        session = database.active_session(principal.user_id)
        thresholds = database.get_settings()
        snapshot: dict[str, Any] = {}
        ages: dict[str, float] = {}
        now = time.time()
        synchronizer = engine.pipeline.synchronizer
        stream_id = session["stream_id"] if session is not None else f"u{principal.user_id}-default"
        
        # Never borrow telemetry or predictions from a different stream.
        if stream_id in synchronizer.values:
            snapshot = synchronizer.snapshot(stream_id, now).model_dump()
            ages = synchronizer.ages(stream_id, now)

        last_seen = synchronizer.last.get(stream_id)
        fresh = last_seen is not None and 0 <= now - last_seen <= EDGE_FRESHNESS_SECONDS
        if not fresh:
            snapshot, ages = {}, {}

        index, active = current_risk(stream_id, snapshot, now)
        if session is None:
            # An ended/not-yet-started trip must never keep displaying risk.
            index, active = None, None

        raw_speed = snapshot.get("vehicle_speed")
        gps_speed = snapshot.get("gps_speed")
        speed_kmh = speed_in_kmh(snapshot)
        rpm_val = snapshot.get("rpm")
        hr_val = snapshot.get("heart_rate")
        vehicle_active = raw_speed is not None or (rpm_val or 0) > 0
        wearable_active = hr_val is not None and hr_val > 0
        upload_active = fresh and has_sensor_values(snapshot)
        computed_level = risk_level_for(index, thresholds)

        return {
            "session": session,
            "vehicle_connected": vehicle_active,
            "wearable_connected": wearable_active,
            "uploading": upload_active,
            "risk_index": index,
            "risk_level": computed_level,
            "risk_active": active,
            "speed_kmh": speed_kmh if speed_kmh is not None else 0.0,
            "rpm": rpm_val if vehicle_active else 0.0,
            "engine_load": snapshot.get("engine_load") if vehicle_active else 0.0,
            "throttle": (snapshot.get("throttle_position") or snapshot.get("throttle")) if vehicle_active else 0.0,
            "heart_rate": hr_val if wearable_active else 0.0,
            "latitude": snapshot.get("gps_latitude"),
            "longitude": snapshot.get("gps_longitude"),
            "sensor_age_seconds": ages,
            "pending_collision": next(
                (
                    a
                    for a in database.list_alerts(user_id=principal.user_id, kind="collision", limit=5)
                    if a["status"] == "pending"
                ),
                None,
            ),
            "models_loaded": len(engine.models),
        }

    @router.get("/driver/research-consent")
    def get_research_consent(principal: Principal = any_user) -> dict[str, Any]:
        """Always the caller's OWN consent -- never settable or readable on
        someone else's behalf, and independent of family-monitoring consent."""
        return database.get_research_consent(principal.user_id)

    @router.post("/driver/research-consent")
    def set_research_consent(
        payload: ResearchConsentRequest, principal: Principal = any_user
    ) -> dict[str, Any]:
        database.log_access(principal.user_id, principal.role, "research-consent")
        return database.set_research_consent(principal.user_id, payload.granted)

    @router.get("/driver/family-contacts")
    def list_family_contacts(principal: Principal = any_user) -> list[dict[str, Any]]:
        return database.list_family_contacts(principal.user_id)

    @router.post("/driver/family-contacts", status_code=201)
    def invite_family_contact(
        payload: FamilyContactRequest, principal: Principal = any_user
    ) -> dict[str, Any]:
        """invite_token is returned here because the DRIVER is its creator
        and has no other way to relay it (no bundled SMS/email provider) --
        this is handing someone their own new resource's token, not
        leaking anyone else's credential (contrast with the auth code
        leak fixed earlier: that handed back someone ELSE's login code to
        an unauthenticated caller who merely guessed a phone number)."""
        database.log_access(principal.user_id, principal.role, "family-contacts:invite")
        return database.invite_family_contact(
            principal.user_id, payload.name, payload.phone,
            relationship=payload.relationship,
            notify_accidents=payload.notify_accidents,
            share_location=payload.share_location,
        )

    @router.post("/driver/family-contacts/{contact_id}/revoke")
    def revoke_family_contact(contact_id: int, principal: Principal = any_user) -> dict[str, Any]:
        updated = database.revoke_family_contact(contact_id, principal.user_id)
        if updated is None:
            raise HTTPException(404, "Contact not found")
        return updated

    @router.patch("/driver/family-contacts/{contact_id}")
    def update_family_contact_permissions(
        contact_id: int, payload: FamilyPermissionsRequest, principal: Principal = any_user
    ) -> dict[str, Any]:
        """notify_accidents and share_location are configured independently
        here -- the driver's own choice, separate from whether the contact
        has accepted. Accepting never grants location access by itself."""
        updated = database.update_family_contact_permissions(
            contact_id, principal.user_id,
            notify_accidents=payload.notify_accidents, share_location=payload.share_location,
        )
        if updated is None:
            raise HTTPException(404, "Contact not found")
        return updated

    @router.post("/family/accept")
    def accept_family_invite(payload: FamilyTokenRequest) -> dict[str, Any]:
        """Public, token-only -- the contact is not a users() row and never
        logs in. This is the explicit consent action the roadmap asks for:
        accepting is a deliberate act by the contact, never assumed."""
        accepted = database.accept_family_contact(payload.token)
        if accepted is None:
            raise HTTPException(404, "Invalid or already-used invite token")
        return {"status": "accepted", "driver_user_id": accepted["driver_user_id"]}

    @router.post("/family/acknowledge")
    def acknowledge_family_notification(payload: FamilyTokenRequest) -> dict[str, Any]:
        """Public, token-only -- lets a notified contact confirm they saw
        it, without needing an account."""
        acknowledged = database.acknowledge_family_notification(payload.token)
        if acknowledged is None:
            raise HTTPException(404, "Invalid acknowledge token")
        return {"status": "acknowledged"}

    @router.post("/driver/session/start")
    def start_session(payload: StartSessionRequest, principal: Principal = any_user) -> dict[str, Any]:
        existing = database.active_session(principal.user_id)
        if existing is not None:
            # Idempotently return the running session rather than failing
            recorder.register(existing["stream_id"], existing["id"])
            return existing

        stream_id = payload.stream_id or f"u{principal.user_id}-{uuid4().hex[:8]}"
        try:
            session = database.start_session(principal.user_id, stream_id)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

        recorder.register(stream_id, session["id"])
        return session

    @router.post("/driver/session/stop")
    def stop_session(principal: Principal = any_user) -> dict[str, Any]:
        session = database.active_session(principal.user_id)
        if session is None:
            return {"status": "no_active_session"}

        summary = recorder.finish(session["id"], session["stream_id"], "driver_stopped")
        # Clear any remaining risk so a stopped session cannot show Critical.
        engine.latest.pop(session["stream_id"], None)
        state = edge_state.get(session["stream_id"])
        if state is not None:
            state["risk_active"] = False
        return {"session": database.get_session(session["id"]), "summary": summary}

    @router.post("/driver/collision")
    def report_collision(payload: CollisionRequest, principal: Principal = any_user) -> dict[str, Any]:
        session = database.active_session(principal.user_id)
        if session is None or session["stream_id"] != payload.stream_id:
            raise HTTPException(409, "That stream has no running session")
        alert = recorder.report_collision(payload.stream_id, payload.latitude, payload.longitude)
        if alert is None:
            raise HTTPException(409, "That stream is not being recorded")
        return alert

    @router.get("/alerts")
    def alerts(
        principal: Principal = any_user,
        kind: str | None = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        timeout_unconfirmed_incidents()
        scope = None if role_allows(principal.role, "account_manager") else principal.user_id
        database.log_access(principal.user_id, principal.role, "alerts")
        return database.list_alerts(user_id=scope, kind=kind, limit=limit)

    @router.post("/alerts/{alert_id}/{action}")
    def act_on_alert(alert_id: int, action: str, principal: Principal = any_user) -> dict[str, Any]:
        # confirm/cancel: unchanged from before (high_risk alerts, manual
        # collision reports). dismiss/resolve are new, lifecycle-enforced
        # actions for an automatically DETECTED incident (see
        # recorder.check_collision / INCIDENT_LIFECYCLE) -- "dismiss" is the
        # driver saying a suspected impact wasn't real; "resolve" is an
        # explicit administrative close-out after confirm/dismiss/timeout.
        status_by_action = {
            "confirm": "confirmed", "cancel": "cancelled",
            "dismiss": "dismissed", "resolve": "resolved",
        }
        if action not in status_by_action:
            raise HTTPException(404, "Unknown action")
        existing = next((a for a in database.list_alerts(limit=500) if a["id"] == alert_id), None)
        if existing is None:
            raise HTTPException(404, "Alert not found")
        if not role_allows(principal.role, "account_manager") and (existing["user_id"] != principal.user_id):
            raise HTTPException(403, "You can only act on your own alerts")

        try:
            updated = database.set_alert_status(
                alert_id, status_by_action[action], principal.user_id,
                enforce_lifecycle=action in {"dismiss", "resolve"},
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if updated is None:
            raise HTTPException(404, "Alert not found")
        if updated["kind"] == "collision" and updated["status"] == "confirmed":
            notify_family_of_incident(updated)
        return updated

    @router.get("/alerts/{alert_id}/family-notifications")
    def alert_family_notifications(alert_id: int, principal: Principal = any_user) -> list[dict[str, Any]]:
        """Which permitted (accepted + notify_accidents) family contacts
        were actually notified about this incident, and whether each has
        acknowledged -- the driver's own view of who was told."""
        existing = next((a for a in database.list_alerts(limit=500) if a["id"] == alert_id), None)
        if existing is None:
            raise HTTPException(404, "Alert not found")
        if not role_allows(principal.role, "account_manager") and (existing["user_id"] != principal.user_id):
            raise HTTPException(403, "You can only view your own alerts")
        return database.list_family_notifications(alert_id)

    @router.get("/sessions")
    def sessions(
        principal: Principal = any_user,
        driver_id: int | None = Query(None),
        limit: int = Query(50, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        scope = driver_id if role_allows(principal.role, "account_manager") else principal.user_id
        database.log_access(principal.user_id, principal.role, "sessions")
        return database.list_sessions(user_id=scope, limit=limit)

    @router.get("/sessions/{session_id}")
    def session_detail(session_id: int, principal: Principal = any_user) -> dict[str, Any]:
        session = visible_session(session_id, principal)
        return {
            "session": session,
            "points": database.session_points(session_id),
            "gaps": database.session_gaps(session_id),
        }

    def _prepare_export(
        session_id: int, principal: Principal, include_location: bool
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        session = visible_session(session_id, principal)
        if include_location and not role_allows(principal.role, "account_manager"):
            raise HTTPException(403, "Precise location requires account_manager role or higher")
        # Consent gates export regardless of who is asking or what role they
        # hold -- a manager cannot export a participant's research data just
        # because they're allowed to SEE the session.
        if not database.get_research_consent(session["user_id"])["granted"]:
            raise HTTPException(403, "This participant has not granted research consent")
        rows = build_export_rows(session, database.session_samples(session_id), include_location)
        database.log_access(principal.user_id, principal.role, f"session:{session_id}:export")
        return session, rows

    @router.get("/sessions/{session_id}/export.json")
    def export_session_json(
        session_id: int,
        principal: Principal = any_user,
        include_location: bool = Query(False, description="Include precise GPS; requires account_manager+"),
    ) -> dict[str, Any]:
        session, rows = _prepare_export(session_id, principal, include_location)
        return {
            "pseudonymous_participant_id": pseudonymize_participant(session["user_id"]),
            "session_id": session_id,
            "includes_precise_location": include_location,
            "rows": rows,
        }

    @router.get("/sessions/{session_id}/export.csv")
    def export_session_csv(
        session_id: int,
        principal: Principal = any_user,
        include_location: bool = Query(False, description="Include precise GPS; requires account_manager+"),
    ) -> Response:
        _, rows = _prepare_export(session_id, principal, include_location)
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=EXPORT_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            flat = dict(row)
            for key in ("quality_flags", "sensor_age_seconds", "sensor_timestamps"):
                if flat.get(key) is not None:
                    flat[key] = json.dumps(flat[key])
            writer.writerow(flat)
        return Response(
            content=buffer.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f'attachment; filename="session_{session_id}_research_export.csv"'},
        )

    @router.get("/live")
    def live(principal: Principal = manager) -> dict[str, Any]:
        close_stale_sessions()
        timeout_unconfirmed_incidents()
        thresholds = database.get_settings()
        synchronizer = engine.pipeline.synchronizer
        now = time.time()
        drivers = []
        for session in database.list_sessions(limit=500):
            if session["ended_at"] is not None:
                continue
            stream_id = session["stream_id"]
            fields: dict[str, Any] = {}
            if stream_id in synchronizer.values:
                fields = synchronizer.snapshot(stream_id, now).model_dump()

            last_seen = synchronizer.last.get(stream_id)
            fresh = last_seen is not None and 0 <= now - last_seen <= EDGE_FRESHNESS_SECONDS
            if not fresh:
                fields = {}

            index, active = current_risk(stream_id, fields, now)
            drivers.append(
                {
                    "session_id": session["id"],
                    "driver_id": session["user_id"],
                    "driver_name": session["driver_name"],
                    "stream_id": stream_id,
                    "started_at": session["started_at"],
                    "risk_index": index,
                    "risk_level": risk_level_for(index, thresholds),
                    "risk_active": active,
                    "speed_kmh": speed_in_kmh(fields),
                    "heart_rate": fields.get("heart_rate"),
                    "latitude": fields.get("gps_latitude"),
                    "longitude": fields.get("gps_longitude"),
                    "last_seen": last_seen,
                    "receiving": fresh and has_sensor_values(fields),
                }
            )
        database.log_access(principal.user_id, principal.role, "live")
        return {"drivers": drivers, "thresholds": thresholds, "server_time": now}

    @router.get("/drivers")
    def list_drivers(principal: Principal = manager) -> list[dict[str, Any]]:
        drivers = database.list_users(role="driver")
        for driver in drivers:
            driver["kit"] = database.kit_for_user(driver["id"])
        return drivers

    @router.post("/drivers", status_code=201)
    def enrol_driver(payload: DriverRequest, principal: Principal = manager) -> dict[str, Any]:
        try:
            driver = database.create_user(payload.phone, payload.name, "driver")
            for device_id in (payload.adapter_id, payload.wearable_id):
                if device_id is not None:
                    database.update_device(device_id, assigned_to=driver["id"])
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        driver["kit"] = database.kit_for_user(driver["id"])
        return driver

    @router.get("/users")
    def list_users(principal: Principal = admin) -> list[dict[str, Any]]:
        return database.list_users()

    @router.post("/users", status_code=201)
    def create_user(payload: UserRequest, principal: Principal = admin) -> dict[str, Any]:
        try:
            return database.create_user(payload.phone, payload.name, payload.role)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.patch("/users/{user_id}")
    def patch_user(user_id: int, payload: UserPatch, principal: Principal = admin) -> dict[str, Any]:
        if user_id == principal.user_id and payload.active is False:
            raise HTTPException(409, "You cannot deactivate your own account")
        if user_id == principal.user_id and payload.role not in (None, "administrator"):
            raise HTTPException(409, "You cannot remove your own administrator role")
        try:
            user = database.update_user(user_id, name=payload.name, role=payload.role, active=payload.active)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if user is None:
            raise HTTPException(404, "User not found")
        return user

    @router.get("/devices")
    def list_devices(principal: Principal = manager) -> list[dict[str, Any]]:
        return database.list_devices()

    @router.post("/devices", status_code=201)
    def create_device(payload: DeviceRequest, principal: Principal = admin) -> dict[str, Any]:
        try:
            return database.create_device(payload.kind, payload.identifier, payload.note)
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.patch("/devices/{device_id}")
    def patch_device(device_id: int, payload: DevicePatch, principal: Principal = admin) -> dict[str, Any]:
        try:
            device = database.update_device(
                device_id,
                identifier=payload.identifier,
                status=payload.status,
                assigned_to=payload.assigned_to,
                clear_assignment=payload.clear_assignment,
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        if device is None:
            raise HTTPException(404, "Device not found")
        return device

    @router.get("/settings")
    def get_settings(principal: Principal = manager) -> dict[str, float]:
        return database.get_settings()

    @router.put("/settings")
    def put_settings(payload: SettingsRequest, principal: Principal = admin) -> dict[str, float]:
        try:
            return database.update_settings(payload.values)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/access-log")
    def access_log(principal: Principal = admin, limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
        return database.recent_access(limit)

    return router