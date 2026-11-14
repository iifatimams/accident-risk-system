"""Pure, deterministic helpers for synthesizing OBD-II/physiology/label columns.

SYNTHETIC SOFTWARE TEST FIXTURE INFRASTRUCTURE -- see synthetic_data/README.md.
These are hand-picked heuristics for plausible-looking correlated signals,
not a calibrated vehicle or cardiovascular model, and must never be
presented as evidence of real-world behavior (AGENTS.md: "Synthetic
fixtures are software tests, never evidence of model performance"; "Never
call an uncalibrated score a probability of an accident").

Units are SI/percent throughout and named explicitly, per AGENTS.md ("Use
SI units internally where practical. Document timestamp units explicitly"):
  speed_mps        m/s
  rpm              rev/min
  throttle_frac    0.0-1.0 (fraction of pedal travel, NOT a percent)
  brake_frac       0.0-1.0 (fraction of pedal travel, NOT a percent)
  engine_load_pct  0-100 percent
  heart_rate_bpm   beats per minute
  timestamp_s      seconds since episode start

This generator authors full-episode synthetic ground truth (it knows, at
authoring time, whether and when a given episode's collision occurs) in
order to write a plausible pre-impact physiological narrative -- that is
ordinary synthetic-dataset authorship, not the causal/future-leakage
concern AGENTS.md raises for feature engineering. Downstream feature
extraction and model training must still only consume point-in-time,
causal information (see src/accident_risk/processing/windows.py); nothing
here is wired into that pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

GRAVITY_MS2 = 9.80665

# Approximate top speed (m/s) usable in each of 6 gears, used only to pick a
# plausible gear index for the synthetic RPM curve below. Not a real
# transmission's ratios.
_GEAR_TOP_SPEED_MPS = (4.0, 9.0, 16.0, 24.0, 33.0, 45.0)
IDLE_RPM = 800.0
MAX_RPM = 6500.0

RESTING_BPM_RANGE = (68.0, 78.0)
DROWSY_BPM_RANGE = (55.0, 62.0)
PANIC_BPM_RANGE = (125.0, 150.0)
PANIC_ONSET_SECONDS_BEFORE_IMPACT = 1.5
IMMINENT_ACCIDENT_WINDOW_S = 3.0


def gear_for_speed(speed_mps: float) -> int:
    """Index (0-based) of the synthetic gear whose top speed first covers speed_mps."""
    for gear, top_speed in enumerate(_GEAR_TOP_SPEED_MPS):
        if speed_mps <= top_speed:
            return gear
    return len(_GEAR_TOP_SPEED_MPS) - 1


def derive_rpm(speed_mps: float, throttle_frac: float) -> float:
    """Heuristic RPM from speed-implied gear plus a throttle-driven offset.

    Not a real drivetrain model: deliberately simple so the correlation
    structure (higher throttle -> higher RPM at a given speed; RPM resets
    near idle at each gear change) is inspectable and testable.
    """
    speed_mps = max(0.0, speed_mps)
    throttle_frac = min(max(throttle_frac, 0.0), 1.0)
    gear = gear_for_speed(speed_mps)
    top_speed = _GEAR_TOP_SPEED_MPS[gear]
    fraction_of_gear = min(speed_mps / top_speed, 1.0) if top_speed > 0 else 0.0
    base_rpm = IDLE_RPM + fraction_of_gear * (MAX_RPM - IDLE_RPM) * 0.55
    throttle_boost = throttle_frac * (MAX_RPM - IDLE_RPM) * 0.35
    rpm = base_rpm + throttle_boost
    return min(max(rpm, IDLE_RPM), MAX_RPM)


def derive_engine_load(throttle_frac: float, accel_mps2: float) -> float:
    """Heuristic engine load percent from throttle demand and positive acceleration."""
    throttle_frac = min(max(throttle_frac, 0.0), 1.0)
    positive_accel = max(accel_mps2, 0.0)
    load = 15.0 + throttle_frac * 65.0 + min(positive_accel, 4.0) * 5.0
    return min(max(load, 0.0), 100.0)


@dataclass
class BpmState:
    """Per-episode mutable heart-rate state, advanced one frame at a time."""

    baseline_bpm: float
    current_bpm: float

    @classmethod
    def for_driver_state(cls, driver_state: str, rng) -> BpmState:
        low, high = DROWSY_BPM_RANGE if driver_state == "drowsy_asleep" else RESTING_BPM_RANGE
        baseline = rng.uniform(low, high)
        return cls(baseline_bpm=baseline, current_bpm=baseline)


def next_bpm(
    state: BpmState,
    driver_state: str,
    seconds_to_impact: float | None,
    hard_braking: bool,
    rng,
) -> float:
    """Advance one frame of heart rate and return the new value.

    - alert/distracted, no imminent impact: small random walk around baseline
      (a hard brake nudges it up slightly -- a real, if modest, startle
      response -- but it stays within a normal range).
    - drowsy_asleep, no imminent impact: baseline drifts slowly toward the
      low drowsy range (already sampled as the baseline) with small jitter.
    - any state, within PANIC_ONSET_SECONDS_BEFORE_IMPACT of a real
      collision: ramps sharply toward the panic range as impact nears.
    """
    jitter = rng.uniform(-1.0, 1.0)
    target = state.baseline_bpm

    if seconds_to_impact is not None and 0.0 <= seconds_to_impact <= PANIC_ONSET_SECONDS_BEFORE_IMPACT:
        panic_low, panic_high = PANIC_BPM_RANGE
        progress = 1.0 - (seconds_to_impact / PANIC_ONSET_SECONDS_BEFORE_IMPACT)
        target = state.baseline_bpm + progress * ((panic_low + panic_high) / 2.0 - state.baseline_bpm)
        new_bpm = state.current_bpm + (target - state.current_bpm) * 0.6 + jitter
    elif hard_braking and driver_state == "alert":
        target = state.baseline_bpm + 8.0
        new_bpm = state.current_bpm + (target - state.current_bpm) * 0.3 + jitter
    else:
        new_bpm = state.current_bpm + (target - state.current_bpm) * 0.1 + jitter

    new_bpm = min(max(new_bpm, 40.0), 190.0)
    state.current_bpm = new_bpm
    return new_bpm


def relative_forward_lateral(
    ego_x: float, ego_y: float, ego_heading_rad: float, other_x: float, other_y: float
) -> tuple[float, float]:
    """Project another object's position into the ego vehicle's forward/lateral frame (meters)."""
    dx, dy = other_x - ego_x, other_y - ego_y
    cos_h, sin_h = math.cos(ego_heading_rad), math.sin(ego_heading_rad)
    forward = dx * cos_h + dy * sin_h
    lateral = -dx * sin_h + dy * cos_h
    return forward, lateral


def min_obstacle_distance(
    ego_x: float,
    ego_y: float,
    ego_heading_rad: float,
    other_positions: list[tuple[float, float]],
    lateral_half_width_m: float = 2.2,
    max_range_m: float = 60.0,
) -> float:
    """Nearest in-lane obstacle ahead of the ego vehicle, in meters.

    Returns NaN if nothing is within max_range_m ahead and within
    lateral_half_width_m of the ego vehicle's own lane -- "no obstacle
    detected", not "obstacle is infinitely far".
    """
    best = math.nan
    for other_x, other_y in other_positions:
        forward, lateral = relative_forward_lateral(ego_x, ego_y, ego_heading_rad, other_x, other_y)
        if 0.0 < forward <= max_range_m and abs(lateral) <= lateral_half_width_m:
            if math.isnan(best) or forward < best:
                best = forward
    return best


def label_imminent_accident(
    timestamps_s: list[float], collision_timestamp_s: float | None, window_s: float = IMMINENT_ACCIDENT_WINDOW_S
) -> list[int]:
    """1 for frames within window_s seconds before (inclusive) the collision, else 0.

    collision_timestamp_s=None (no collision in this episode) labels every
    frame 0.
    """
    if collision_timestamp_s is None:
        return [0] * len(timestamps_s)
    return [
        1 if collision_timestamp_s - window_s <= t <= collision_timestamp_s else 0 for t in timestamps_s
    ]
