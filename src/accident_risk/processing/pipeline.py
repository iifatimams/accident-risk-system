"""The sole feature path shared by training, replay, and live ingestion."""

from dataclasses import dataclass

from accident_risk.processing.physiology_features import physiology_features
from accident_risk.processing.synchronization import Synchronizer
from accident_risk.processing.vehicle_features import vehicle_features
from accident_risk.processing.windows import RollingWindows, Window
from accident_risk.schemas import PHYSIOLOGY_FIELDS, SensorFrame
from accident_risk.settings import PipelineSettings


@dataclass
class FeatureSnapshot:
    """One causal feature cutoff with quality and freshness evidence."""

    frame: SensorFrame
    window: Window
    vehicle: dict[str, float]
    physiology: dict[str, float]
    ages: dict[str, float]
    flags: list[str]


class FeaturePipeline:
    """Online transformations; no fitted preprocessing, labels, or source branches."""

    def __init__(self, settings: PipelineSettings | None = None) -> None:
        self.settings = settings or PipelineSettings()
        self.synchronizer = Synchronizer(self.settings)
        self.windows = RollingWindows(
            max(self.settings.vehicle_window_seconds, self.settings.physiology_window_seconds),
            self.settings.stride_seconds,
        )

    def update(self, frame: SensorFrame) -> FeatureSnapshot | None:
        """Update every packet; emit features at configured stride without modality waits."""
        frame = SensorFrame.model_validate(frame)
        current = self.synchronizer.update(frame)
        ages = self.synchronizer.ages(frame.stream_id, frame.timestamp)
        flags = [
            f"stale:{name}"
            for name, age in ages.items()
            if age > self.settings.max_age_seconds.get(name, self.settings.default_max_age_seconds)
        ]
        quality = current.physiology_signal_quality
        if quality is None:
            flags.append("physiology_quality_unknown")
        if quality is not None and quality < self.settings.physiology_quality_threshold:
            flags.append("physiology_low_quality")
            payload = current.model_dump()
            for name in PHYSIOLOGY_FIELDS:
                if name != "physiology_signal_quality":
                    payload[name] = None
                    payload["sensor_timestamps"].pop(name, None)
                    # An expired quality flag must never resurrect rejected readings.
                    self.synchronizer.values[frame.stream_id].pop(name, None)
            current = SensorFrame(**payload)
        window = self.windows.update(current)
        if window is None:
            return None
        if len({f.timestamp for f in window.frames}) < self.settings.min_samples:
            return None
        vehicle_window = self.windows.at(
            frame.stream_id, frame.timestamp, self.settings.vehicle_window_seconds
        )
        physiology_window = self.windows.at(
            frame.stream_id, frame.timestamp, self.settings.physiology_window_seconds
        )
        return FeatureSnapshot(
            current,
            vehicle_window,
            vehicle_features(
                vehicle_window, self.settings.longitudinal_axis, self.settings.lateral_axis
            ),
            physiology_features(physiology_window, self.settings.spo2_drop_threshold),
            ages,
            flags,
        )
