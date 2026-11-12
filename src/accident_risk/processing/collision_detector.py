"""Independent collision detection: a rule-based kinematic trigger that never
reads overall_risk_index, event_detection_score, or any other future-risk
output. Per the roadmap, actual impact detection must not be gated by (or
even aware of) the separate, still-unvalidated future-risk prediction work in
scripts/build_future_risk_dataset.py -- the two must stay independently
assessable, and never conflated in a report.

This is a simple threshold on raw |acceleration|, not a trained model. The
threshold (COLLISION_ACCEL_THRESHOLD_MS2) is chosen with reference to the
VZCrash research (round 2/3): real crash clips showed a mean peak of ~3.9g
vs ~1.5g for normal/near-miss driving, so a trigger well above normal-driving
noise but below the observed crash-clip peaks is defensible as a FIRST, rule-
based pass -- it is explicitly NOT validated against labeled real-world
impacts for this project's actual sensor placement/orientation, and must not
be reported as if it were.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from accident_risk.schemas import SensorFrame

GRAVITY_MS2 = 9.80665
# ~2.5g of combined (gravity-inclusive) acceleration magnitude. Ordinary
# driving, including hard braking, rarely approaches this; real impacts in
# the VZCrash data peaked far higher (~3.9g mean). Deliberately conservative
# (a lower threshold trades false positives for not missing a real impact),
# but unvalidated for this project's own sensors -- see module docstring.
COLLISION_ACCEL_THRESHOLD_MS2 = 2.5 * GRAVITY_MS2
MODEL_VERSION = "kinematic_threshold_v1_unvalidated"


@dataclass(frozen=True)
class CollisionSuspicion:
    triggered: bool
    accel_magnitude_ms2: float
    score: float  # accel_magnitude / threshold; >=1.0 means triggered
    onset_estimate: float  # the frame's own timestamp -- no smoothing/lag
    model_version: str = MODEL_VERSION


def accel_magnitude(frame: SensorFrame) -> float | None:
    """sqrt(x^2+y^2+z^2) from the frame's own raw axes, or None if absent."""
    if frame.accel_x is None or frame.accel_y is None or frame.accel_z is None:
        return None
    return math.sqrt(frame.accel_x**2 + frame.accel_y**2 + frame.accel_z**2)


def check(frame: SensorFrame) -> CollisionSuspicion | None:
    """Independent, stateless, per-frame check. None if accel isn't present
    on this frame (e.g. an OBD-only or wearable-only reading) -- absence is
    not evidence of anything, so it's distinct from a non-triggering score."""
    magnitude = accel_magnitude(frame)
    if magnitude is None:
        return None
    return CollisionSuspicion(
        triggered=magnitude >= COLLISION_ACCEL_THRESHOLD_MS2,
        accel_magnitude_ms2=magnitude,
        score=magnitude / COLLISION_ACCEL_THRESHOLD_MS2,
        onset_estimate=frame.timestamp,
    )
