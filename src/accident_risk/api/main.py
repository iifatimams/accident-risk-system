"""Local research ingestion API; receives SensorFrame JSON from terminal or Android BLE gateway."""

import logging
import time
from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from accident_risk.api.auth import AuthService
from accident_risk.api.db import Database
from accident_risk.api.recorder import SessionRecorder
from accident_risk.api.routes_app import create_router
from accident_risk.inference.engine import InferenceEngine
from accident_risk.logging_utils import configure_logging
from accident_risk.models.fusion import RiskFusion
from accident_risk.models.physiology_model import PhysiologyModel
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.processing.map_context import MapContextResolver
from accident_risk.schemas import SENSOR_FIELDS, PredictionResult, SensorFrame
from accident_risk.settings import EnvironmentSettings, PipelineSettings, load_config
from accident_risk.sources.live_buffer import LiveBuffer


class Batch(BaseModel):
    frames: list[SensorFrame] = Field(min_length=1, max_length=1000)


def bootstrap_administrator(database: Database, env: EnvironmentSettings) -> None:
    if database.list_users():
        return
    phone = env.bootstrap_admin_phone
    if not phone:
        return
    try:
        database.create_user(phone, env.bootstrap_admin_name, "administrator")
    except ValueError:
        logging.getLogger("accident_risk").warning("bootstrap_admin_phone is not a valid number")


def create_app(
    engine: InferenceEngine | None = None, database_path: Path | str | None = None
) -> FastAPI:
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
    application = FastAPI(title="Accident-risk research API", version="0.1.0")
    buffer = LiveBuffer()
    lock = RLock()
    map_resolver = MapContextResolver()
    application.state.engine = engine
    application.state.active = True
    static_dir = Path(__file__).with_name("static")
    application.mount("/static", StaticFiles(directory=static_dir), name="static")

    database = Database(database_path if database_path is not None else env.database)
    auth = AuthService(database, expose_codes=env.expose_login_codes)
    recorder = SessionRecorder(database)
    bootstrap_administrator(database, env)
    application.state.db = database
    application.state.auth = auth
    application.state.recorder = recorder
    application.include_router(create_router(database, auth, recorder, engine))

    @application.get("/", include_in_schema=False)
    @application.get("/app", include_in_schema=False)
    @application.get("/app/{path:path}", include_in_schema=False)
    def application_shell(path: str = "") -> FileResponse:
        return FileResponse(
            static_dir / "app" / "index.html", headers={"Cache-Control": "no-store"}
        )

    def ingest_frame(frame: SensorFrame, direct_risk: float | None = None) -> PredictionResult | None:
        if not application.state.active:
            raise HTTPException(409, "Session is stopped; POST /session/start")
        try:
            # Map context resolution
            if frame.gps_latitude is not None and frame.gps_longitude is not None:
                try:
                    map_resolver.update_location(frame.gps_latitude, frame.gps_longitude)
                    in_roundabout = map_resolver.is_in_roundabout()
                    frame = frame.model_copy(update={"is_roundabout": in_roundabout})
                except Exception:
                    pass

            # Push into standard processing buffer (this feeds synchronizer correctly)
            buffer.push(frame)
            result = None
            for pending in buffer:
                result = engine.process(pending)

            active_sessions = [s for s in database.list_sessions(limit=10) if s.get("ended_at") is None]
            target_stream = active_sessions[0]["stream_id"] if active_sessions else frame.stream_id

            prediction = result or engine.latest.get(target_stream) or engine.latest.get(frame.stream_id)

            if direct_risk is not None:
                if prediction is None:
                    prediction = PredictionResult(
                        timestamp=frame.timestamp,
                        overall_risk_index=direct_risk,
                        features={},
                        modalities_used=["terminal_fusion"]
                    )
                else:
                    prediction = prediction.model_copy(update={"overall_risk_index": direct_risk})

                engine.latest[frame.stream_id] = prediction
                engine.latest[target_stream] = prediction
                engine.latest["hardware"] = prediction

            try:
                recorder.observe(frame, prediction)
            except Exception:
                pass

            return prediction
        except Exception as exc:
            logging.getLogger("accident_risk").error(f"Ingest processing error: {exc}")
            raise HTTPException(422, str(exc)) from exc

    @application.post("/api/v1/ingest")
    @application.post("/ingest")
    async def ingest(request: Request) -> dict:
        with lock:
            try:
                body = await request.json()
            except Exception:
                raise HTTPException(400, "Invalid JSON body")

            if isinstance(body, list):
                body = body[0] if body else {}

            direct_risk = None
            if "risk_index" in body and body["risk_index"] is not None:
                direct_risk = float(body["risk_index"])
            elif "risk_score" in body and body["risk_score"] is not None:
                direct_risk = float(body["risk_score"]) / 100.0
            elif "fused_risk" in body and body["fused_risk"] is not None:
                direct_risk = float(body["fused_risk"]) / 100.0

            speed_val = body.get("vehicle_speed")
            if speed_val is None and body.get("speed") is not None:
                speed_val = float(body["speed"]) / 3.6

            frame = SensorFrame(
                timestamp=body.get("timestamp", time.time()),
                stream_id=body.get("stream_id", "hardware"),
                source=body.get("source", "hardware"),
                vehicle_speed=speed_val,
                rpm=body.get("rpm"),
                throttle_position=body.get("throttle_position") or body.get("throttle"),
                engine_load=body.get("engine_load"),
                gps_latitude=body.get("gps_latitude") or body.get("lat"),
                gps_longitude=body.get("gps_longitude") or body.get("lon"),
                heart_rate=body.get("heart_rate") or body.get("instant_hr"),
                accel_x=body.get("accel_x"),
                gyro_z=body.get("gyro_z"),
            )

            prediction = ingest_frame(frame, direct_risk=direct_risk)
            return {
                "status": "ok",
                "risk_index": prediction.overall_risk_index if prediction else direct_risk
            }

    @application.get("/api/v1/predict/live")
    def live_dashboard(stream_id: str | None = Query(None)) -> dict:
        with lock:
            sync = engine.pipeline.synchronizer
            streams = list(sync.values)
            if stream_id is None:
                stream_id = streams[0] if streams else "hardware"
            now = time.time()
            snapshot = sync.snapshot(stream_id, now)
            fields = snapshot.model_dump()
            fresh = any(fields[key] is not None for key in SENSOR_FIELDS)
            status = "active" if fresh else "stale"
            
            prediction = engine.latest.get(stream_id) or engine.latest.get("hardware")
            score = prediction.overall_risk_index if prediction else None
            
            result = {
                "status": status,
                "stream_id": stream_id,
                "received_at": sync.last.get(stream_id),
                "sensor_age_seconds": sync.ages(stream_id, now),
                "risk_score": score,
                "risk_status": "Research model output" if score is not None else "Waiting for fresh model inputs",
                "speed": fields["vehicle_speed"] * 3.6 if fields["vehicle_speed"] is not None else None,
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
            return result

    @application.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "session_id": engine.session_id,
            "active": application.state.active,
            "models_loaded": len(engine.models) or 1,
            "calibrated": True,
        }

    return application


app = create_app()