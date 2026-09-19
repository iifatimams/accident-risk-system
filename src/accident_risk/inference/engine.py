"""Source-independent, causal inference with explicit abstention and latency measurement."""

import logging
from threading import RLock
from time import perf_counter
from uuid import uuid4

import pandas as pd

from accident_risk.models.fusion import RiskFusion
from accident_risk.models.physiology_model import PhysiologyModel
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.processing.pipeline import FeaturePipeline
from accident_risk.schemas import PredictionResult, SensorFrame
from accident_risk.settings import PipelineSettings
from accident_risk.training.train_vehicle import usable


class InferenceEngine:
    """One synchronized state store for independent streams; no acquisition-source logic."""

    def __init__(
        self,
        vehicle_model: VehicleModel | None = None,
        physiology_model: PhysiologyModel | None = None,
        event_model: VehicleModel | None = None,
        settings: PipelineSettings | None = None,
        fusion: RiskFusion | None = None,
    ) -> None:
        self.models = [m for m in (vehicle_model, physiology_model, event_model) if m is not None]
        if len({m.target for m in self.models}) != len(self.models):
            raise ValueError("Only one model per score component is supported")
        # The saved preprocessing contract wins unless a matching one is explicitly supplied.
        saved = [
            PipelineSettings(**m.metadata.get("configuration", {}).get("pipeline", {}))
            for m in self.models
        ]
        self.settings = settings or (saved[0] if saved else PipelineSettings())
        if any(s != self.settings for s in saved):
            raise ValueError(
                "Preprocessing configuration differs from training; align window/feature/age settings"
            )
        self.fusion = fusion or RiskFusion()
        self._lock = RLock()
        self.pipeline = FeaturePipeline(self.settings)
        self.latest: dict[str, PredictionResult] = {}
        self.session_id = uuid4().hex
        self._log_session("session_created")

    def _log_session(self, event: str) -> None:
        """Log lifecycle events once rather than logging every IMU packet at INFO."""
        logging.getLogger("accident_risk").info(
            event,
            extra={
                "context": {
                    "session_id": self.session_id,
                    "model_version": [m.metadata.get("model_version") for m in self.models],
                }
            },
        )

    def reset(self) -> None:
        """Clear all buffers/predictions and begin an independent session."""
        with self._lock:
            self.pipeline = FeaturePipeline(self.settings)
            self.latest.clear()
            self.session_id = uuid4().hex
            self._log_session("session_reset")

    def process(self, frame: SensorFrame) -> PredictionResult | None:
        """Process each packet immediately; return null until cadence/warmup permits inference."""
        started = perf_counter()
        with self._lock:
            snapshot = self.pipeline.update(frame)
            if snapshot is None:
                return None
            scores: dict[str, float | None] = {
                "event_detection_score": None,
                "vehicle_risk_score": None,
                "driver_state_risk": None,
            }
            versions = []
            flags = list(snapshot.flags)
            for model in self.models:
                physiology = model.target == "driver_state_risk"
                if not usable(snapshot, physiology):
                    flags.append(f"unavailable:{model.target}")
                    continue
                features = snapshot.physiology if physiology else snapshot.vehicle
                scores[model.target] = float(model.predict_score(pd.DataFrame([features]))[0])
                versions.append(model.metadata["model_version"])
            if not self.models:
                flags.append("no_models_loaded")
            index, level = self.fusion.combine(scores)
            quality = sum(
                getattr(snapshot.frame, field) is not None for field in self.settings.quality_fields
            ) / len(self.settings.quality_fields)
            prediction = PredictionResult(
                timestamp=frame.timestamp,
                stream_id=frame.stream_id,
                **scores,
                overall_risk_index=index,
                risk_level=level,
                data_quality=quality,
                model_version="+".join(versions) or "unavailable",
                quality_flags=flags,
                sensor_age_seconds=snapshot.ages,
                inference_latency_ms=(perf_counter() - started) * 1000,
            )
            self.latest[frame.stream_id] = prediction
            logging.getLogger("accident_risk").debug(
                "inference",
                extra={
                    "context": {
                        "session_id": self.session_id,
                        "timestamp": frame.timestamp,
                        "source": frame.source,
                        "stream_id": frame.stream_id,
                        "inference_latency_ms": prediction.inference_latency_ms,
                        "model_version": prediction.model_version,
                        "quality_flags": flags,
                    }
                },
            )
            return prediction
