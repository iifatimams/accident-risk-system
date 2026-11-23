"""Product API behind the operator and driver interface. Mounted at /app/api."""

import json
import time
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Response
from pydantic import BaseModel, Field

from accident_risk.api.auth import AuthError, AuthService, Principal, bearer_from_header
from accident_risk.api.db import ALERT_VERIFICATION_RESULTS, Database, role_allows
from accident_risk.api.notifications import MockNotificationSender, build_message
from accident_risk.api.recorder import SessionRecorder
from accident_risk.inference.engine import InferenceEngine
from accident_risk.schemas import SENSOR_FIELDS

SESSION_TIMEOUT_SECONDS = 180.0
EDGE_FRESHNESS_SECONDS = 10.0
UNCONFIRMED_INCIDENT_TIMEOUT_SECONDS = 180.0

# ios/RoadSense/Services/MultimodalRiskFusion.swift's fiveSecondAheadRisk
# (PolidrivingFiveSecondModel.mlpackage, driver-held-out ROC-AUC 0.918 --
# docs/FINAL_MULTIMODAL_RESULTS.md section 2.1, blended with live
# attention/road evidence) is now risk_score/risk_index itself, not a
# second advisory number -- see BackgroundCaptureService.swift. It must
# still never gate collision detection or a family notification (those
# stay driven solely by CollisionDetector's IMU threshold), and it is a
# model score, not a calibrated "probability of an accident."


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


class FamilyRequestRequest(BaseModel):
    phone: str = Field(min_length=3, max_length=30)
    # Only needed when `phone` isn't a registered driver yet -- supplying
    # them creates a brand-new driver account (self-service, not just
    # admin-only) and assigns the two named devices to it in one step.
    name: str | None = Field(None, min_length=1, max_length=120)
    obd_identifier: str | None = Field(None, min_length=1, max_length=120)
    wearable_identifier: str | None = Field(None, min_length=1, max_length=120)


class FamilyRespondRequest(BaseModel):
    accept: bool


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


class AlertVerificationRequest(BaseModel):
    result: str
    notes: str = ""


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
        know about. Respects each linked family member's own notify AND
        share_location permissions individually; duplicate-suppressed per
        (alert, recipient) by the database's own UNIQUE constraint."""
        driver = database.get_user(alert["user_id"])
        if driver is None:
            return
        for member in database.accepted_family_members_to_notify(alert["user_id"]):
            if database.already_notified(alert["id"], member["id"]):
                continue
            message = build_message(
                driver["name"], alert["status"], member["share_location"],
                alert.get("latitude"), alert.get("longitude"),
                confirmed_by_driver=alert.get("acknowledged_by") == alert["user_id"],
            )
            status = notification_sender.send(
                contact_phone=member["phone"], contact_name=member["name"], message=message
            )
            database.record_family_notification(alert["id"], member["id"], status)

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
    admin = Depends(require("administrator"))

    def visible_session(session_id: int, principal: Principal) -> dict[str, Any]:
        session = database.get_session(session_id)
        if session is None:
            raise HTTPException(404, "Session not found")
        if not role_allows(principal.role, "administrator") and (session["user_id"] != principal.user_id):
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
        return {"service": "roadsense", "api_version": 1}

    @router.post("/auth/request-code")
    def request_code(payload: CodeRequest, response: Response) -> dict[str, Any]:
        """Public, ngrok-reachable. Never returns a verification code --
        see api/operator.py for the Mac-local console that does."""
        response.headers["Cache-Control"] = "no-store"
        try:
            return auth.request_code(payload.phone)
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
                    if a["status"] == "awaiting_confirmation"
                ),
                None,
            ),
            "models_loaded": len(engine.models),
        }

    @router.get("/driver/family-links")
    def list_family_links(principal: Principal = any_user) -> list[dict[str, Any]]:
        return database.list_family_links(principal.user_id)

    @router.post("/driver/family-request", status_code=201)
    def request_family_link(
        payload: FamilyRequestRequest, principal: Principal = any_user
    ) -> dict[str, Any]:
        """Driver-to-driver, by phone number -- replaces the earlier admin-
        managed family contacts entirely. If the phone number isn't a
        registered driver yet, name (+ device identifiers, if known)
        creates a real new driver account right here, with login access,
        rather than requiring an administrator (FR2.1 now allows this path
        too, not just admin-created accounts)."""
        recipient = database.get_user_by_phone(payload.phone)
        if recipient is None:
            if not payload.name:
                raise HTTPException(
                    422,
                    "That phone number isn't registered yet -- provide a name "
                    "(and OBD/wearable identifiers, if known) to create their account",
                )
            try:
                recipient = database.create_user(payload.phone, payload.name, "driver")
            except ValueError as exc:
                raise HTTPException(409, str(exc)) from exc
            for kind, identifier in (
                ("adapter", payload.obd_identifier), ("wearable", payload.wearable_identifier),
            ):
                if not identifier:
                    continue
                # identifier text is no longer unique (duplicate OBD/wearable
                # names are expected), so reuse an unassigned spare with this
                # identifier if one exists, rather than always creating a new
                # device row for what may be the same physical unit.
                device = next(
                    (
                        d for d in database.list_devices()
                        if d["kind"] == kind and d["identifier"] == identifier
                        and d["assigned_to"] is None
                    ),
                    None,
                )
                if device is None:
                    device = database.create_device(kind, identifier)
                database.update_device(device["id"], assigned_to=recipient["id"])
        database.log_access(principal.user_id, principal.role, "family-links:request")
        try:
            return database.request_family_link(principal.user_id, recipient["id"])
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc

    @router.post("/driver/family-links/{link_id}/respond")
    def respond_to_family_link(
        link_id: int, payload: FamilyRespondRequest, principal: Principal = any_user
    ) -> dict[str, Any]:
        """Only the RECIPIENT of the request can accept or decline it -- a
        deliberate act by them, never assumed."""
        updated = database.respond_to_family_link(link_id, principal.user_id, payload.accept)
        if updated is None:
            raise HTTPException(404, "Request not found, already answered, or not yours to answer")
        return updated

    @router.post("/driver/family-links/{link_id}/remove")
    def remove_family_link(link_id: int, principal: Principal = any_user) -> dict[str, Any]:
        """Either party can remove a link at any time -- a pending
        request, or an already-accepted one (leaving a family group)."""
        if not database.remove_family_link(link_id, principal.user_id):
            raise HTTPException(404, "Link not found or not yours")
        return {"status": "removed"}

    @router.patch("/driver/family-links/{link_id}/permissions")
    def update_family_link_permissions(
        link_id: int, payload: FamilyPermissionsRequest, principal: Principal = any_user
    ) -> dict[str, Any]:
        """Each side configures only THEIR OWN half of the link -- what
        THEY share with the other party, independent of what the other
        party shares back."""
        updated = database.update_family_link_permissions(
            link_id, principal.user_id,
            notify_accidents=payload.notify_accidents, share_location=payload.share_location,
        )
        if updated is None:
            raise HTTPException(404, "Link not found or not yours")
        return updated

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

    @router.get("/alerts")
    def alerts(
        principal: Principal = any_user,
        kind: str | None = Query(None),
        limit: int = Query(100, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        timeout_unconfirmed_incidents()
        scope = None if role_allows(principal.role, "administrator") else principal.user_id
        database.log_access(principal.user_id, principal.role, "alerts")
        results = database.list_alerts(user_id=scope, kind=kind, limit=limit)
        # Evaluator verification (correct/false/unreviewed) is a separate,
        # research-only judgement -- distinct from the alert's own
        # operational status/lifecycle above -- so it's attached here
        # rather than folded into list_alerts itself.
        for alert in results:
            verification = database.get_alert_verification(alert["id"])
            alert["verification_result"] = (
                verification["verification_result"] if verification else "unreviewed"
            )
        return results

    @router.post("/alerts/{alert_id}/verify")
    def verify_alert_route(
        alert_id: int, payload: AlertVerificationRequest, principal: Principal = admin,
    ) -> dict[str, Any]:
        """An administrator's research judgement of one alert event (did it
        actually correspond to a real hazard/collision), kept separate from
        the driver-facing confirm/cancel/dismiss/resolve lifecycle above."""
        if payload.result not in ALERT_VERIFICATION_RESULTS:
            raise HTTPException(422, f"Unknown verification result: {payload.result}")
        try:
            return database.verify_alert(
                alert_id, payload.result, verified_by=principal.user_id, notes=payload.notes
            )
        except ValueError as exc:
            raise HTTPException(404, str(exc)) from exc

    @router.post("/alerts/{alert_id}/{action}")
    def act_on_alert(alert_id: int, action: str, principal: Principal = any_user) -> dict[str, Any]:
        # confirm/cancel: legacy vocabulary for high_risk alerts (status
        # "pending") -- never lifecycle-enforced for those, unchanged from
        # before. dismiss/resolve are the richer,
        # always-enforced vocabulary for an automatically DETECTED incident
        # (see recorder.check_collision / INCIDENT_LIFECYCLE).
        #
        # The two vocabularies share one table AND can share status
        # STRINGS (an automatic incident also reaches "confirmed"), so
        # status alone cannot distinguish them -- checking the alert's own
        # detail.source ("automatic_detector" is set only by
        # recorder._check_collision, never by a manual report or a plain
        # high_risk alert) is unambiguous. Without this, confirm/cancel
        # were NEVER lifecycle-enforced at all, so a stray "confirm" could
        # reopen an already-resolved/dismissed/timed-out automatic
        # incident (checkpoint resume task #4) -- while a legacy alert
        # that happens to reach "confirmed" must still accept a later
        # "cancel" exactly as before.
        status_by_action = {
            "confirm": "confirmed", "cancel": "cancelled",
            "dismiss": "dismissed", "resolve": "resolved",
        }
        if action not in status_by_action:
            raise HTTPException(404, "Unknown action")
        existing = next((a for a in database.list_alerts(limit=500) if a["id"] == alert_id), None)
        if existing is None:
            raise HTTPException(404, "Alert not found")
        if not role_allows(principal.role, "administrator") and (existing["user_id"] != principal.user_id):
            raise HTTPException(403, "You can only act on your own alerts")

        is_automatic_incident = existing.get("detail", {}).get("source") == "automatic_detector"
        try:
            updated = database.set_alert_status(
                alert_id, status_by_action[action], principal.user_id,
                enforce_lifecycle=action in {"dismiss", "resolve"} or is_automatic_incident,
            )
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from exc
        if updated is None:
            raise HTTPException(404, "Alert not found")
        if updated["kind"] == "collision" and updated["status"] == "confirmed":
            notify_family_of_incident(updated)
        return updated

    @router.get("/sessions")
    def sessions(
        principal: Principal = any_user,
        driver_id: int | None = Query(None),
        limit: int = Query(50, ge=1, le=500),
    ) -> list[dict[str, Any]]:
        scope = driver_id if role_allows(principal.role, "administrator") else principal.user_id
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

    @router.get("/live")
    def live(principal: Principal = admin) -> dict[str, Any]:
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
    def list_drivers(principal: Principal = admin) -> list[dict[str, Any]]:
        drivers = database.list_users(role="driver")
        for driver in drivers:
            driver["kit"] = database.kit_for_user(driver["id"])
        return drivers

    @router.post("/drivers", status_code=201)
    def enrol_driver(payload: DriverRequest, principal: Principal = admin) -> dict[str, Any]:
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
    def list_devices(principal: Principal = admin) -> list[dict[str, Any]]:
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
    def get_settings(principal: Principal = admin) -> dict[str, float]:
        return database.get_settings()

    @router.put("/settings")
    def put_settings(payload: SettingsRequest, principal: Principal = admin) -> dict[str, float]:
        try:
            return database.update_settings(payload.values, changed_by=principal.user_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.get("/access-log")
    def access_log(principal: Principal = admin, limit: int = Query(100, ge=1, le=500)) -> list[dict[str, Any]]:
        return database.recent_access(limit)

    return router