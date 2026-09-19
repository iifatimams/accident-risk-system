"""Validation reports keep every rejection or questionable observation visible."""

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from accident_risk.schemas import SENSOR_FIELDS, SensorFrame


@dataclass
class ValidationReport:
    """Accepted frames and indexed errors/warnings; no silent row removal."""

    frames: list[SensorFrame] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def valid(self) -> bool:
        """Whether every submitted record passed structural validation."""
        return not self.errors


def validate_records(records: list[dict[str, Any] | SensorFrame]) -> ValidationReport:
    """Reject NaN/infinity/range errors; report ordering, duplicates, and quality."""
    report = ValidationReport()
    last: dict[str, float] = {}
    seen: set[tuple[str, float, str]] = set()
    for index, record in enumerate(records):
        try:
            frame = SensorFrame.model_validate(record)
        except ValidationError as exc:
            report.errors.append(f"row {index}: {exc}")
            continue
        if frame.timestamp < last.get(frame.stream_id, -1):
            report.errors.append(f"row {index}: out-of-order timestamp")
        if frame.timestamp == last.get(frame.stream_id):
            report.warnings.append(
                f"row {index}: duplicate timestamp (may be complementary sensors)"
            )
        for sensor in SENSOR_FIELDS:
            if getattr(frame, sensor) is None:
                continue
            key = (frame.stream_id, frame.sensor_timestamps.get(sensor, frame.timestamp), sensor)
            if key in seen:
                report.warnings.append(f"row {index}: repeated {sensor} observation")
            seen.add(key)
        if frame.physiology_signal_quality is not None and frame.physiology_signal_quality < 0.5:
            report.warnings.append(f"row {index}: low physiology signal quality")
        last[frame.stream_id] = frame.timestamp
        report.frames.append(frame)
    return report
