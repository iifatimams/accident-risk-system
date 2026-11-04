"""Multi-stream ingestion API; handles per-driver telemetry and dashboard feeds."""
import json
import logging
import time
import urllib.request
from pathlib import Path
from threading import RLock, Thread

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

TERMINAL_RUNNER_URL = "http://127.0.0.1:8003"


def async_mirror_to_terminal(payload: dict) -> None:
    """Relays incoming phone frames to run_advanced_terminal on port 8003."""
    def send():
        try:
            req = urllib.request.Request(
                TERMINAL_RUNNER_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=0.5):
                pass
        except Exception:
            pass
    Thread(target=send, daemon=True).start()


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

    application = FastAPI(title="RoadSense Multi-Driver API", version="0.2.0")
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
            if frame.gps_latitude is not None and frame.gps_longitude is not None:
                try:
                    map_resolver.update_location(frame.gps_latitude, frame.gps_longitude)
                    in_roundabout = map_resolver.is_in_roundabout()
                    frame = frame.model_copy(update={"is_roundabout": in_roundabout})
                except Exception:
                    pass

            buffer.push(frame)
            result = None
            for pending in buffer:
                result = engine.process(pending)

            active_sessions = [s for s in database.list_sessions(limit=50) if s.get("ended_at") is None]
            session = next((s for s in active_sessions if s["stream_id"] == frame.stream_id), None)
            target_stream = session["stream_id"] if session else frame.stream_id

            prediction = result or engine.latest.get(target_stream)
            if direct_risk is not None:
                risk_level = "LOW"
                if direct_risk >= 0.85:
                    risk_level = "CRITICAL"
                elif direct_risk >= 0.70:
                    risk_level = "HIGH"
                elif direct_risk >= 0.40:
                    risk_level = "MEDIUM"

                prediction = PredictionResult(
                    timestamp=frame.timestamp,
                    stream_id=target_stream,
                    overall_risk_index=direct_risk,
                    risk_level=risk_level,
                    data_quality=1.0,
                    model_version="terminal_fusion",
                    inference_latency_ms=0.0,
                )
                engine.latest[target_stream] = prediction
                engine.latest[frame.stream_id] = prediction

            try:
                recorder.observe(frame, prediction)
            except Exception:
                pass

            return prediction
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

            source_tag = body.get("source", "ios")

            # CRITICAL: If the frame came from iOS, mirror it to port 8003 for ML processing!
            # If the frame came from terminal_multimodal/ml_runner, do not loop it back.
            if source_tag not in ("terminal_multimodal", "ml_runner"):
                async_mirror_to_terminal(body)

            speed_val = body.get("vehicle_speed")
            if speed_val is None and body.get("speed") is not None:
                speed_val = float(body["speed"]) / 3.6

            stream_id = str(body.get("stream_id") or "driver-default")

            frame = SensorFrame(
                timestamp=body.get("timestamp", time.time()),
                stream_id=stream_id,
                source=source_tag,
                vehicle_speed=speed_val,
                rpm=body.get("rpm"),
                throttle_position=body.get("throttle_position") or body.get("throttle"),
                engine_load=body.get("engine_load"),
                gps_latitude=body.get("gps_latitude") or body.get("lat"),
                gps_longitude=body.get("gps_longitude") or body.get("lon"),
                heart_rate=body.get("heart_rate") or body.get("instant_hr") or body.get("bpm"),
                accel_x=body.get("accel_x"),
                gyro_z=body.get("gyro_z"),
            )
            prediction = ingest_frame(frame, direct_risk=direct_risk)
            return {
                "status": "ok",
                "stream_id": stream_id,
                "risk_index": prediction.overall_risk_index if prediction else direct_risk,
            }

    @application.get("/api/v1/predict/live")
    def live_dashboard(stream_id: str | None = Query(None)) -> dict:
        with lock:
            sync = engine.pipeline.synchronizer
            streams = list(sync.values)
            if stream_id is None:
                stream_id = streams[0] if streams else "driver-default"
            now = time.time()
            snapshot = sync.snapshot(stream_id, now)
            fields = snapshot.model_dump()
            fresh = any(fields[key] is not None for key in SENSOR_FIELDS)
            status = "active" if fresh else "stale"

            prediction = engine.latest.get(stream_id)
            score = prediction.overall_risk_index if prediction else None

            result = {
                "status": status,
                "stream_id": stream_id,
                "received_at": sync.last.get(stream_id),
                "sensor_age_seconds": sync.ages(stream_id, now),
                "risk_score": score,
                "risk_status": "Active ML Output" if score is not None else "Waiting for sensor frames",
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
