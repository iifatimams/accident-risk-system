"""Transparent heuristic fusion; no claim of statistical or medical validation."""

import math
from dataclasses import dataclass, field

COMPONENTS = {"vehicle_risk_score", "driver_state_risk", "event_detection_score"}


@dataclass
class RiskFusion:
    """Normalize configured weights over available components; abstain if none."""

    weights: dict[str, float] = field(
        default_factory=lambda: {
            "vehicle_risk_score": 0.5,
            "driver_state_risk": 0.2,
            "event_detection_score": 0.3,
        }
    )
    thresholds: tuple[float, float, float] = (0.3, 0.6, 0.8)

    def __post_init__(self) -> None:
        if (
            not self.weights
            or set(self.weights) - COMPONENTS
            or any(not math.isfinite(v) or v < 0 for v in self.weights.values())
            or sum(self.weights.values()) <= 0
        ):
            raise ValueError(
                "Fusion requires finite nonnegative known-component weights and a positive total"
            )
        if (
            len(self.thresholds) != 3
            or not 0 < self.thresholds[0] < self.thresholds[1] < self.thresholds[2] < 1
        ):
            raise ValueError("Three strictly increasing thresholds in (0, 1) are required")

    def combine(self, scores: dict[str, float | None]) -> tuple[float | None, str | None]:
        """Return a display index and category; missing evidence never means LOW."""
        available = [(self.weights.get(k, 0), v) for k, v in scores.items() if v is not None]
        if any(not math.isfinite(v) or not 0 <= v <= 1 for _, v in available):
            raise ValueError("Fusion scores must be finite and in [0, 1]")
        total = sum(w for w, _ in available)
        if total == 0:
            return None, None
        index = sum(w * v for w, v in available) / total
        level = ("LOW", "MEDIUM", "HIGH", "CRITICAL")[sum(index >= t for t in self.thresholds)]
        return index, level
