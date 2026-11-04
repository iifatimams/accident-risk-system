"""Product API behind the operator and driver interface. Mounted at /app/api."""
import time
from typing import Annotated, Any
from uuid import uuid4

from fastapi import APIRouter, Depends, Header, HTTPException, Query
from pydantic import BaseModel, Field

from accident_risk.api.auth import AuthError, AuthService, Principal, bearer_from_header
from accident_risk.api.db import Database, role_allows
from accident_risk.api.recorder import SessionRecorder
from accident_risk.inference.engine import InferenceEngine

SESSION_TIMEOUT_SECONDS = 180.0


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


def create_router(
    database: Database,
    auth: AuthService,
    recorder: SessionRecorder,
    engine: InferenceEngine,
) -> APIRouter:
    router = APIRouter(prefix="/app/api", tags=["application"])

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

    @router.post("/auth/request-code")
    def request_code(payload: CodeRequest) -> dict[str, Any]:
        try:
            return auth.request_code(payload.phone)
        except AuthError as exc:
            raise HTTPException(exc.status, exc.message) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @router.post("/auth/verify")
    def verify(payload: VerifyRequest) -> dict[str, Any]:
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
        session = database.active_session(principal.user_id)
        thresholds = database.get_settings()
        snapshot: dict[str, Any] = {}
        ages: dict[str, float] = {}
        now = time.time()
        synchronizer = engine.pipeline.synchronizer
        stream_id = session["stream_id"] if session is not None else f"u{principal.user_id}-default"
        
        if stream_id in synchronizer.values and any(synchronizer.snapshot(stream_id, now).model_dump().values()):
            snapshot = synchronizer.snapshot(stream_id, now).model_dump()
            ages = synchronizer.ages(stream_id, now)
        elif "hardware" in synchronizer.values:
            snapshot = synchronizer.snapshot("hardware", now).model_dump()
            ages = synchronizer.ages("hardware", now)

        prediction = engine.latest.get(stream_id) or engine.latest.get("hardware")
        index = prediction.overall_risk_index if prediction else None

        raw_speed = snapshot.get("vehicle_speed")
        rpm_val = snapshot.get("rpm")
        hr_val = snapshot.get("heart_rate")

        vehicle_active = (raw_speed is not None and raw_speed > 0) or (rpm_val is not None and rpm_val > 0)
        wearable_active = hr_val is not None and hr_val > 0
        upload_active = bool(snapshot and any(v is not None for v in snapshot.values()))

        speed_kmh = (raw_speed * 3.6) if raw_speed is not None else snapshot.get("speed")
        computed_level = risk_level_for(index, thresholds) if (index is not None and (vehicle_active or wearable_active)) else None

        return {
            "session": session,
            "vehicle_connected": vehicle_active,
            "wearable_connected": wearable_active,
            "uploading": upload_active,
            "risk_index": index if (vehicle_active or wearable_active) else None,
            "risk_level": computed_level,
            "speed_kmh": speed_kmh if vehicle_active else 0.0,
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
            "models_loaded": len(engine.models) or (1 if prediction is not None else 0),
        }

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
        scope = None if role_allows(principal.role, "account_manager") else principal.user_id
        database.log_access(principal.user_id, principal.role, "alerts")
        return database.list_alerts(user_id=scope, kind=kind, limit=limit)

    @router.post("/alerts/{alert_id}/{action}")
    def act_on_alert(alert_id: int, action: str, principal: Principal = any_user) -> dict[str, Any]:
        if action not in {"confirm", "cancel"}:
            raise HTTPException(404, "Unknown action")
        existing = next((a for a in database.list_alerts(limit=500) if a["id"] == alert_id), None)
        if existing is None:
            raise HTTPException(404, "Alert not found")
        if not role_allows(principal.role, "account_manager") and (existing["user_id"] != principal.user_id):
            raise HTTPException(403, "You can only act on your own alerts")
        status = "confirmed" if action == "confirm" else "cancelled"
        updated = database.set_alert_status(alert_id, status, principal.user_id)
        if updated is None:
            raise HTTPException(404, "Alert not found")
        return updated

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

    @router.get("/live")
    def live(principal: Principal = manager) -> dict[str, Any]:
        close_stale_sessions()
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
            elif "hardware" in synchronizer.values:
                fields = synchronizer.snapshot("hardware", now).model_dump()
            prediction = engine.latest.get(stream_id) or engine.latest.get("hardware")
            index = prediction.overall_risk_index if prediction else None
            drivers.append(
                {
                    "session_id": session["id"],
                    "driver_id": session["user_id"],
                    "driver_name": session["driver_name"],
                    "stream_id": stream_id,
                    "started_at": session["started_at"],
                    "risk_index": index,
                    "risk_level": risk_level_for(index, thresholds),
                    "speed_kmh": fields.get("vehicle_speed") * 3.6 if fields.get("vehicle_speed") is not None else fields.get("speed"),
                    "heart_rate": fields.get("heart_rate"),
                    "latitude": fields.get("gps_latitude"),
                    "longitude": fields.get("gps_longitude"),
                    "last_seen": synchronizer.last.get(stream_id) or synchronizer.last.get("hardware"),
                    "receiving": bool(fields) and any(v is not None for v in fields.values()),
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