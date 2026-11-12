"""Multi-stream ingestion API; handles per-driver telemetry and dashboard feeds."""
import logging
import math
import time
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError

from accident_risk.api.auth import AuthService
from accident_risk.api.db import Database
from accident_risk.api.notifications import MockNotificationSender
from accident_risk.api.recorder import SessionRecorder
from accident_risk.api.routes_app import create_router, risk_level_for
from accident_risk.inference.engine import InferenceEngine
from accident_risk.logging_utils import configure_logging
from accident_risk.models.fusion import RiskFusion
from accident_risk.models.physiology_model import PhysiologyModel
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.processing.map_context import MapContextResolver
from accident_risk.schemas import SENSOR_FIELDS, PredictionResult, SensorFrame
from accident_risk.settings import EnvironmentSettings, PipelineSettings, load_config
from accident_risk.sources.live_buffer import LiveBuffer

# Kept for compatibility with code that imports this name; each app instance
# uses its own cache so tests and app factories never share driver state.
APP_UI_STATE: dict[str, dict] = {}

# UI must not show a last-known danger score after the phone stops transmitting.
EDGE_FRESHNESS_SECONDS = 10.0

# Tolerate ordinary client clock drift, but reject frames whose declared
# timestamp is implausibly ahead of the server's own clock (e.g. a buggy
# client or corrupted replay), rather than silently accepting them and
# having them simply never show up as "fresh" until the clock catches up.
MAX_CLOCK_SKEW_SECONDS = 5.0
LEGACY_RUNNER_SOURCES = {"ml_runner", "terminal_multimodal"}


def optional_value(body: dict, *keys: str):
    """Preserve legitimate zeros (dict.get(a) or dict.get(b) does not)."""
    for key in keys:
        if body.get(key) is not None:
            return body[key]
    return None


def parse_risk(body: dict) -> float | None:
    """Accept risk_index in [0,1], and risk_score/fused_risk in [0,100]."""
    for key, divisor in (("risk_index", 1.0), ("risk_score", 100.0), ("fused_risk", 100.0)):
        value = body.get(key)
        if value is None:
            continue
        try:
            number = float(value) / divisor
        except (ValueError, TypeError, OverflowError) as exc:
            raise HTTPException(422, f"Invalid {key}") from exc
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise HTTPException(422, f"{key} is outside its expected range")
        return number
    return None


class Batch(BaseModel):
    frames: list[SensorFrame] = Field(min_length=1, max_length=1000)


def bootstrap_administrator(database: Database, env: EnvironmentSettings) -> None:
    if database.list_users(role="administrator"):
        return
    phone = env.bootstrap_admin_phone or "+971500000000"
    try:
        database.create_user(phone, env.bootstrap_admin_name, "administrator")
        print(f"\n[SYSTEM] Initialized Admin user: Phone={phone}\n")
    except ValueError:
        pass


def create_app(
    engine: InferenceEngine | None = None,
    database_path: Path | str | None = None,
) -> FastAPI:
    """Build the API. No real SMS provider or demo-mode pairing-key system
    is wired up -- see api/auth.py and api/operator.py for how verification
    codes are actually revealed to a Mac-local operator instead."""
    env = EnvironmentSettings()
    if engine is None:
        config = load_config(env.config) if env.config.exists() else {}
        configure_logging(config.get("logging", {}).get("level", "INFO"))
        vehicle = VehicleModel.load(env.vehicle_model) if env.vehicle_model else None
        physiology = PhysiologyModel.load(env.physiology_model) if env.physiology_model else None
        engine = InferenceEngine(
            vehicle,
            physiology,
            settings=None
            if vehicle or physiology
            else PipelineSettings(**config.get("pipeline", {})),
            fusion=RiskFusion(**config.get("fusion", {})),
        )

    application = FastAPI(title="RoadSense Multi-Driver API", version="0.2.0")
    buffer = LiveBuffer()
    lock = RLock()
    map_resolver = MapContextResolver()
    # State is keyed by the same stream_id as the iOS app's SensorFrame.
    ui_state: dict[str, dict] = {}

    application.state.ui_state = ui_state
    application.state.engine = engine
    application.state.active = True

    static_dir = Path(__file__).with_name("static")
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    database = Database(database_path if database_path is not None else env.database)
    auth = AuthService(database)
    recorder = SessionRecorder(database)
    bootstrap_administrator(database, env)
    notification_sender = MockNotificationSender()

    application.state.db = database
    application.state.auth = auth
    application.state.recorder = recorder
    application.state.notification_sender = notification_sender

    application.include_router(
        create_router(
            database, auth, recorder, engine,
            edge_state=ui_state, notification_sender=notification_sender,
        )
    )

    @application.get("/", include_in_schema=False)
    @application.get("/app", include_in_schema=False)
    @application.get("/app/{path:path}", include_in_schema=False)
    def application_shell(path: str = "") -> FileResponse:
        # FastAPI requires the `path` argument for the catch-all route.
        if path == "api" or path.startswith("api/"):
            raise HTTPException(404, "Unknown RoadSense API route")
        return FileResponse(
            static_dir / "app" / "index.html", headers={"Cache-Control": "no-store"}
        )

    def ingest_frame(
        frame: SensorFrame, direct_risk: float | None = None, edge_mode: bool = False
    ) -> PredictionResult | None:
        if not application.state.active:
            raise HTTPException(409, "Session is stopped; POST /session/start")

        try:
            result = None

            if frame.source != "ml_runner":
                if frame.gps_latitude is not None and frame.gps_longitude is not None:
                    try:
                        map_resolver.update_location(frame.gps_latitude, frame.gps_longitude)
                        in_roundabout = map_resolver.is_in_roundabout()
                        frame = frame.model_copy(update={"is_roundabout": in_roundabout})
                    except (ValueError, TypeError, RuntimeError) as exc:
                        logging.getLogger("accident_risk").warning("Map context unavailable: %s", exc)

                if edge_mode:
                    # Direct Core ML inference is already complete on the iPhone.
                    # Update telemetry only; do not run a second Python model.
                    engine.pipeline.synchronizer.update(frame)
                else:
                    buffer.push(frame)
                    for pending in buffer:
                        result = engine.process(pending)

            active_sessions = [s for s in database.list_sessions(limit=50) if s.get("ended_at") is None]
            session = next((s for s in active_sessions if s["stream_id"] == frame.stream_id), None)
            target_stream = session["stream_id"] if session else frame.stream_id

            prediction = result or engine.latest.get(target_stream)
            if edge_mode and direct_risk is None:
                # An inactive/unknown edge risk is NOT a LOW probability.
                engine.latest.pop(target_stream, None)
                engine.latest.pop(frame.stream_id, None)
                recorder.observe(frame, None, ui_state.get(frame.stream_id))
                return None
            if direct_risk is not None:
                risk_level = risk_level_for(direct_risk, database.get_settings())

                prediction = PredictionResult(
                    timestamp=frame.timestamp,
                    stream_id=target_stream,
                    overall_risk_index=direct_risk,
                    risk_level=risk_level,
                    data_quality=1.0,
                    model_version="ios_coreml" if edge_mode else "terminal_fusion",
                    inference_latency_ms=0.0,
                )
                engine.latest[target_stream] = prediction
                engine.latest[frame.stream_id] = prediction

            try:
                if prediction is not None:
                    recorder.observe(frame, prediction, ui_state.get(frame.stream_id))
            except (ValueError, TypeError, RuntimeError, OSError) as exc:
                logging.getLogger("accident_risk").warning("Session recording failed: %s", exc)

            return prediction
        except HTTPException:
            raise
        except Exception as exc:
            logging.getLogger("accident_risk").error("Ingest processing error: %s", exc)
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/v1/ingest")
    @application.post("/ingest")
    @application.post("/")
    async def ingest(request: Request) -> dict:
        with lock:
            try:
                body = await request.json()
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, "Invalid JSON body") from exc

            if isinstance(body, list):
                body = body[0] if body else {}
            if not isinstance(body, dict):
                raise HTTPException(400, "JSON body must be an object")

            source_tag = body.get("source") or "ios"
            stream_id = str(body.get("stream_id") or "driver-default")
            edge_mode = source_tag == "ios" and "risk_active" in body

            risk_active = body.get("risk_active") if edge_mode else None
            if edge_mode and type(risk_active) is not bool:
                raise HTTPException(422, "risk_active must be a JSON boolean")

            # A false flag takes priority over any accidentally attached score.
            direct_risk = parse_risk(body) if not edge_mode or risk_active else None
            if edge_mode and risk_active and direct_risk is None:
                raise HTTPException(422, "risk_active=true requires risk_score or risk_index")

            state = ui_state.setdefault(stream_id, {})
            # Once a trip has sent Core ML output, legacy terminal messages
            # must never take ownership of that SAME stream again.
            if source_tag in LEGACY_RUNNER_SOURCES and state.get("edge_owned"):
                return {"status": "ignored_legacy_prediction", "stream_id": stream_id, "risk_index": None}

            if source_tag == "ios" and "camera_frame" in body:
                state["camera_frame"] = body["camera_frame"]

            if edge_mode:
                state.update({
                    "edge_owned": True,
                    "risk_active": risk_active,
                    "received_at": time.time(),
                    "eyes_on_road": body.get("eyes_on_road", True),
                    # Discrete reading behind eyes_on_road, e.g. "looking_down",
                    # "face_not_detected", "uncertain" -- see
                    # ios/RoadSense/Services/DriverAttentionState.swift.
                    # Dashboard-only diagnostic, same as camera_status/debug_*
                    # below; not validated against the strict SensorFrame schema.
                    "attention_state": body.get("attention_state", "attentive"),
                    "eyes_closed_sec": body.get("eyes_closed_sec", 0.0),
                    "traffic_level": body.get("traffic_level", "empty"),
                    "pedestrians_detected": body.get("pedestrians_detected", False),
                    "pedestrian_count": body.get("pedestrian_count", 0),
                    # Discrete reading combining pedestrian position/motion
                    # -- see ios/RoadSense/Services/PedestrianHazard.swift
                    # (Phase 7). The most severe of any currently tracked
                    # pedestrians.
                    "pedestrian_hazard_state": body.get("pedestrian_hazard_state", "insufficient_evidence"),
                    "lead_closing_fast": body.get("lead_closing_fast", False),
                    # Discrete reading behind lead_closing_fast -- see
                    # ios/RoadSense/Services/ForwardCollisionHazard.swift
                    # (Phase 6). Experimental, derived from a still-crude
                    # rectangle-heuristic detection, never a calibrated
                    # collision probability.
                    "forward_hazard_state": body.get("forward_hazard_state", "no_relevant_lead_object"),
                    "camera_status": body.get("camera_status", "idle"),
                    "debug_face_detected": body.get("debug_face_detected", False),
                    "debug_pitch": body.get("debug_pitch", 0.0),
                    "debug_yaw": body.get("debug_yaw", 0.0),
                    "debug_roll": body.get("debug_roll", 0.0),
                    "debug_ear": body.get("debug_ear", 0.0),
                    # Phase 9.1 explainable breakdown (ml_fusion_base, each
                    # named heuristic adjustment) -- see
                    # ios/RoadSense/Services/RiskFusionHeuristics.swift.
                    # Dashboard-only, like everything else in this block;
                    # absent entirely when risk is inactive.
                    "risk_breakdown": body.get("risk_breakdown"),
                })
            elif source_tag in LEGACY_RUNNER_SOURCES:
                state.update({
                    "eyes_on_road": body.get("eyes_on_road", True),
                    "eyes_closed_sec": body.get("eyes_closed_sec", 0.0),
                    "traffic_level": body.get("traffic_level", "empty"),
                    "pedestrians_detected": body.get("pedestrians_detected", False),
                    "pedestrian_count": body.get("pedestrian_count", 0),
                    "lead_closing_fast": body.get("lead_closing_fast", False),
                })

            obd_connected = optional_value(body, "is_obd_connected", "obd_connected")
            vehicle_speed = optional_value(body, "vehicle_speed", "speed")
            if source_tag == "ios":
                # SensorFrame.toDictionary() writes 0 for missing OBD speed.
                # Ignore that synthetic 0 when OBD isn't connected.
                if obd_connected is False:
                    vehicle_speed = None
                elif vehicle_speed is not None:
                    vehicle_speed = float(vehicle_speed) / 3.6  # km/h -> m/s
            elif vehicle_speed is not None:
                # Legacy `speed` is km/h; schema `vehicle_speed` is m/s.
                vehicle_speed = float(vehicle_speed) if body.get("vehicle_speed") is not None else float(vehicle_speed) / 3.6

            request_timestamp = body.get("timestamp", time.time())
            if (
                isinstance(request_timestamp, (int, float))
                and request_timestamp > time.time() + MAX_CLOCK_SKEW_SECONDS
            ):
                raise HTTPException(422, "timestamp is too far in the future")

            try:
                frame = SensorFrame(
                    timestamp=request_timestamp,
                    stream_id=stream_id,
                    source=source_tag,
                    vehicle_speed=vehicle_speed,
                    gps_speed=body.get("gps_speed"),  # Core Location reports m/s
                    gps_heading=body.get("gps_heading"),
                    rpm=body.get("rpm"),
                    throttle_position=optional_value(body, "throttle_position", "throttle"),
                    engine_load=optional_value(body, "engine_load", "load"),
                    gps_latitude=optional_value(body, "gps_latitude", "lat"),
                    gps_longitude=optional_value(body, "gps_longitude", "lon"),
                    heart_rate=(optional_value(body, "heart_rate", "instant_hr", "bpm") or None),
                    rr_interval_ms=body.get("rr_interval_ms"),
                    spo2=body.get("spo2"),
                    hrv_rmssd=body.get("hrv_rmssd"),
                    physiology_signal_quality=body.get("physiology_signal_quality"),
                    steering_angle=body.get("steering_angle"),
                    tire_pressure_fl=body.get("tire_pressure_fl"),
                    tire_pressure_fr=body.get("tire_pressure_fr"),
                    tire_pressure_rl=body.get("tire_pressure_rl"),
                    tire_pressure_rr=body.get("tire_pressure_rr"),
                    sensor_timestamps=body.get("sensor_timestamps") or {},
                    accel_x=body.get("accel_x"),
                    accel_y=body.get("accel_y"),
                    accel_z=body.get("accel_z"),
                    gyro_x=body.get("gyro_x"),
                    gyro_y=body.get("gyro_y"),
                    gyro_z=body.get("gyro_z"),
                    client_incident_id=body.get("client_incident_id"),
                )
            except (ValidationError, ValueError, TypeError) as exc:
                raise HTTPException(422, f"Invalid sensor frame: {exc}") from exc

            # Independent collision detection: runs on this frame's raw
            # accelerometer data alone, before and regardless of the
            # prediction/edge_mode/direct_risk branching inside
            # ingest_frame, so it is never gated by (or dependent on) the
            # future-risk score. Called here (not inside ingest_frame) so
            # its result -- the collision alert, if one was created or
            # matched by client_incident_id -- can be returned to the
            # caller, letting an offline-queued client learn the server-
            # side alert id once its upload finally succeeds.
            collision_alert = recorder.check_collision(frame)
            prediction = ingest_frame(frame, direct_risk=direct_risk, edge_mode=edge_mode)
            return {
                "status": "ok",
                "stream_id": stream_id,
                "risk_index": prediction.overall_risk_index if prediction else direct_risk,
                "collision_alert_id": collision_alert["id"] if collision_alert else None,
            }

    @application.post("/session/stop")
    def stop_global_session() -> dict[str, bool]:
        """Global ingestion halt (distinct from a per-driver /app/api session):
        new frames are refused with 409 until /session/start. Also clears
        cached per-stream state so a resumed /api/v1/predict/live read doesn't
        show data from before the stop."""
        with lock:
            application.state.active = False
            engine.pipeline.synchronizer.values.clear()
            engine.pipeline.synchronizer.last.clear()
            engine.latest.clear()
        return {"active": False}

    @application.post("/session/start")
    def start_global_session() -> dict[str, bool]:
        with lock:
            application.state.active = True
        return {"active": True}

    @application.get("/api/v1/predict/live")
    def live_dashboard(stream_id: str | None = Query(None)) -> dict:
        with lock:
            sync = engine.pipeline.synchronizer
            streams = list(sync.values)
            if stream_id is None:
                # Silently picking one of several active streams would show
                # the wrong driver's data; the caller must disambiguate.
                if len(streams) > 1:
                    raise HTTPException(
                        400, "stream_id is required when multiple streams are active"
                    )
                stream_id = streams[0] if streams else "driver-default"
            now = time.time()
            snapshot = sync.snapshot(stream_id, now)
            fields = snapshot.model_dump()
            fresh = any(fields[key] is not None for key in SENSOR_FIELDS)
            status = (
                "active" if fresh
                else "stale" if stream_id in sync.values
                else "waiting_for_hardware_stream"
            )

            state = ui_state.get(stream_id, {})
            edge_owned = bool(state.get("edge_owned"))
            edge_fresh = (
                edge_owned and 0 <= now - state.get("received_at", 0.0) <= EDGE_FRESHNESS_SECONDS
            )
            risk_active = (
                state.get("risk_active") if edge_fresh else None
            ) if edge_owned else None
            prediction = engine.latest.get(stream_id)
            if edge_owned:
                score = prediction.overall_risk_index if risk_active is True and prediction else None
            else:
                score = prediction.overall_risk_index if status == "active" and prediction else None

            risk_status = (
                "Not driving" if risk_active is False
                else "Stale telemetry" if edge_owned and not edge_fresh
                else "Active ML Output" if score is not None
                else "Waiting for sensor frames"
            )
            result = {
                "status": status,
                "stream_id": stream_id,
                "received_at": sync.last.get(stream_id),
                "sensor_age_seconds": sync.ages(stream_id, now),
                "risk_score": score,
                "risk_index": score,
                "risk_active": risk_active,
                "risk_status": risk_status,
                "speed": (fields["vehicle_speed"] if fields["vehicle_speed"] is not None else fields["gps_speed"]) * 3.6
                if (fields["vehicle_speed"] is not None or fields["gps_speed"] is not None) else None,
                "rpm": fields["rpm"],
                "throttle": fields["throttle_position"],
                "engine_load": fields["engine_load"],
                "heart": fields["heart_rate"],
                "spo2": fields["spo2"],
                "quality": fields["physiology_signal_quality"],
                "heading": fields["gps_heading"],
                "location": {
                    "latitude": fields["gps_latitude"],
                    "longitude": fields["gps_longitude"],
                },
            }
            for key in ("accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z"):
                result[key] = fields.get(key)
                
            # APPEND UI DIAGNOSTICS TO RESPONSE
            ui_data = state
            result["camera_frame"] = ui_data.get("camera_frame")
            result["cv_eyes_on_road"] = ui_data.get("eyes_on_road", True)
            result["cv_attention_state"] = ui_data.get("attention_state", "attentive")
            result["cv_eyes_closed_sec"] = ui_data.get("eyes_closed_sec", 0.0)
            result["cv_traffic"] = ui_data.get("traffic_level", "empty")
            result["cv_pedestrians_detected"] = ui_data.get("pedestrians_detected", False)
            result["cv_pedestrian_count"] = ui_data.get("pedestrian_count", 0)
            result["cv_pedestrian_hazard_state"] = ui_data.get("pedestrian_hazard_state", "insufficient_evidence")
            result["cv_closing_fast"] = ui_data.get("lead_closing_fast", False)
            result["cv_forward_hazard_state"] = ui_data.get("forward_hazard_state", "no_relevant_lead_object")
            result["cv_camera_status"] = ui_data.get("camera_status", "idle")
            result["cv_debug_face_detected"] = ui_data.get("debug_face_detected", False)
            result["cv_debug_pitch"] = ui_data.get("debug_pitch", 0.0)
            result["cv_debug_yaw"] = ui_data.get("debug_yaw", 0.0)
            result["cv_debug_roll"] = ui_data.get("debug_roll", 0.0)
            result["cv_debug_ear"] = ui_data.get("debug_ear", 0.0)
            result["risk_breakdown"] = ui_data.get("risk_breakdown")

            return result

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "session_id": engine.session_id,
            "active": application.state.active,
            "models_loaded": len(engine.models),
            # A weighted risk index is not a calibrated crash probability.
            "calibrated": False,
        }

    return application


def __getattr__(name: str):
    """Lazily build the production ASGI app (uvicorn's `main:app` target) only
    when something actually accesses `main.app` -- not merely on import. Every
    test file imports `create_app` from this module; an eager `app =
    create_app()` here would open the real database (env.database, normally
    data/app.sqlite3) as a side effect of that import alone."""
    if name == "app":
        return create_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
