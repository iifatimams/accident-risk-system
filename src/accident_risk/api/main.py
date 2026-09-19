"""Local research ingestion API; a future Android/BLE gateway sends SensorFrame JSON."""

from pathlib import Path
from threading import RLock

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from accident_risk.inference.engine import InferenceEngine
from accident_risk.logging_utils import configure_logging
from accident_risk.models.fusion import RiskFusion
from accident_risk.models.physiology_model import PhysiologyModel
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.schemas import PredictionResult, SensorFrame
from accident_risk.settings import EnvironmentSettings, PipelineSettings, load_config
from accident_risk.sources.live_buffer import LiveBuffer


class Batch(BaseModel):
    """Bounded request size, with independent accepted/rejected results."""

    frames: list[SensorFrame] = Field(min_length=1, max_length=1000)


def create_app(engine: InferenceEngine | None = None) -> FastAPI:
    """Create isolated session state; serve with a single worker to retain in-memory state."""
    if engine is None:
        env = EnvironmentSettings()
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
    app = FastAPI(title="Accident-risk research API", version="0.1.0")
    buffer = LiveBuffer()
    lock = RLock()
    app.state.engine = engine
    app.state.active = True
    static_dir = Path(__file__).with_name("static")
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        """Serve the approachable live monitoring dashboard."""
        return FileResponse(static_dir / "index.html")

    def ingest_frame(frame: SensorFrame) -> PredictionResult | None:
        """Consume immediately under the session lock; slow modalities never gate ingestion."""
        if not app.state.active:
            raise HTTPException(409, "Session is stopped; POST /session/start")
        try:
            buffer.push(frame)
            result = None
            for pending in buffer:
                result = engine.process(pending)
            return result or engine.latest.get(frame.stream_id)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc

    @app.get("/health")
    def health() -> dict:
        """Expose process/session/model readiness separately."""
        return {
            "status": "ok",
            "session_id": engine.session_id,
            "active": app.state.active,
            "models_loaded": len(engine.models),
            "calibrated": False,
        }

    @app.post("/ingest", response_model=PredictionResult | None)
    def ingest(frame: SensorFrame) -> PredictionResult | None:
        """Accept a timestamped partial sensor update."""
        with lock:
            return ingest_frame(frame)

    @app.post("/ingest/batch")
    def ingest_batch(batch: Batch) -> dict:
        """Structural errors reject the request; temporal errors are reported per item.

        Valid earlier items are retained if a later item is out of order.
        """
        with lock:
            if not app.state.active:
                raise HTTPException(409, "Session is stopped")
            results = []
            for i, frame in enumerate(batch.frames):
                try:
                    prediction = ingest_frame(frame)
                    results.append({"index": i, "accepted": True, "prediction": prediction})
                except HTTPException as exc:
                    results.append({"index": i, "accepted": False, "error": exc.detail})
            return {"results": results}

    @app.get("/prediction/latest", response_model=PredictionResult | None)
    def latest(stream_id: str | None = Query(None)) -> PredictionResult | None:
        """Select a stream explicitly when more than one has predictions."""
        with lock:
            if stream_id:
                return engine.latest.get(stream_id)
            if len(engine.latest) > 1:
                raise HTTPException(400, "Specify stream_id for multiple streams")
            return next(iter(engine.latest.values()), None)

    @app.post("/reset")
    def reset() -> dict:
        """Clear session buffers and previous predictions."""
        nonlocal buffer
        with lock:
            engine.reset()
            buffer = LiveBuffer()
            if not app.state.active:
                buffer.stop()
            return {"session_id": engine.session_id}

    @app.post("/session/start")
    def start() -> dict:
        """Begin a fresh independent acquisition session."""
        with lock:
            app.state.active = True
            reset()
            return {"session_id": engine.session_id, "active": True}

    @app.post("/session/stop")
    def stop() -> dict:
        """Stop accepting packets while retaining the last prediction for inspection."""
        with lock:
            app.state.active = False
            buffer.stop()
            return {"session_id": engine.session_id, "active": False}

    return app


app = create_app()
