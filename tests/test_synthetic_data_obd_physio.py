"""Tests for synthetic_data/obd_physio_synthesis.py.

SYNTHETIC SOFTWARE TEST FIXTURE ONLY -- see synthetic_data/README.md.
These exercise the pure, deterministic helpers used by the MetaDrive
scenario generator (AGENTS.md: "Every new feature must have meaningful
tests"). They do not import MetaDrive and do not run the simulator, so
they stay fast and need no GPU/display.
"""

from __future__ import annotations

import math
import random

import pytest

from synthetic_data.obd_physio_synthesis import (
    DROWSY_BPM_RANGE,
    PANIC_BPM_RANGE,
    RESTING_BPM_RANGE,
    BpmState,
    derive_engine_load,
    derive_rpm,
    label_imminent_accident,
    min_obstacle_distance,
    next_bpm,
)


def test_derive_rpm_increases_with_throttle_at_fixed_speed():
    low_throttle = derive_rpm(speed_mps=10.0, throttle_frac=0.1)
    high_throttle = derive_rpm(speed_mps=10.0, throttle_frac=0.9)
    assert high_throttle > low_throttle


def test_derive_rpm_stays_within_idle_and_max_bounds():
    for speed in (0.0, 5.0, 20.0, 60.0, 1000.0):
        for throttle in (0.0, 0.5, 1.0, 5.0, -1.0):
            rpm = derive_rpm(speed, throttle)
            assert 800.0 <= rpm <= 6500.0


def test_derive_engine_load_increases_with_throttle_and_acceleration():
    baseline = derive_engine_load(throttle_frac=0.0, accel_mps2=0.0)
    more_throttle = derive_engine_load(throttle_frac=1.0, accel_mps2=0.0)
    more_accel = derive_engine_load(throttle_frac=0.0, accel_mps2=4.0)
    assert more_throttle > baseline
    assert more_accel > baseline
    assert 0.0 <= baseline <= 100.0
    assert 0.0 <= more_throttle <= 100.0


def test_min_obstacle_distance_finds_nearest_obstacle_directly_ahead():
    others = [(0.0, 30.0), (0.0, 10.0), (0.0, 50.0)]  # all straight ahead on the y axis
    dist = min_obstacle_distance(ego_x=0.0, ego_y=0.0, ego_heading_rad=math.pi / 2, other_positions=others)
    assert dist == pytest.approx(10.0)


def test_min_obstacle_distance_ignores_behind_and_out_of_lane():
    others = [
        (0.0, -10.0),  # directly behind
        (10.0, 10.0),  # ahead but far outside the lane (large lateral offset)
    ]
    dist = min_obstacle_distance(ego_x=0.0, ego_y=0.0, ego_heading_rad=math.pi / 2, other_positions=others)
    assert math.isnan(dist)


def test_min_obstacle_distance_nan_when_nothing_in_range():
    dist = min_obstacle_distance(ego_x=0.0, ego_y=0.0, ego_heading_rad=0.0, other_positions=[], max_range_m=60.0)
    assert math.isnan(dist)


def test_label_imminent_accident_marks_exactly_the_window_before_collision():
    timestamps = [t / 10.0 for t in range(0, 101)]  # 0.0 .. 10.0s at 0.1s steps
    collision_at = 10.0
    labels = label_imminent_accident(timestamps, collision_at, window_s=3.0)
    for t, label in zip(timestamps, labels, strict=True):
        expected = 1 if 7.0 <= t <= 10.0 else 0
        assert label == expected, (t, label, expected)


def test_label_imminent_accident_all_zero_without_a_collision():
    timestamps = [0.0, 1.0, 2.0, 3.0]
    assert label_imminent_accident(timestamps, None) == [0, 0, 0, 0]


def test_bpm_state_drowsy_baseline_is_in_the_drowsy_range():
    rng = random.Random(0)
    state = BpmState.for_driver_state("drowsy_asleep", rng)
    low, high = DROWSY_BPM_RANGE
    assert low <= state.baseline_bpm <= high


def test_bpm_state_alert_and_distracted_baseline_is_in_the_resting_range():
    rng = random.Random(0)
    for driver_state in ("alert", "distracted_phone"):
        state = BpmState.for_driver_state(driver_state, rng)
        low, high = RESTING_BPM_RANGE
        assert low <= state.baseline_bpm <= high


def test_next_bpm_ramps_toward_panic_range_as_impact_nears():
    rng = random.Random(0)
    state = BpmState.for_driver_state("drowsy_asleep", rng)
    bpm_far = next_bpm(state, "drowsy_asleep", seconds_to_impact=1.5, hard_braking=False, rng=rng)
    for seconds_to_impact in (1.2, 0.9, 0.6, 0.3, 0.0):
        bpm_near = next_bpm(state, "drowsy_asleep", seconds_to_impact=seconds_to_impact, hard_braking=False, rng=rng)
    panic_low, _panic_high = PANIC_BPM_RANGE
    assert bpm_near > bpm_far
    assert bpm_near > panic_low - 10.0  # should be approaching the panic range by t=0


def test_next_bpm_without_imminent_impact_stays_near_baseline():
    rng = random.Random(1)
    state = BpmState.for_driver_state("alert", rng)
    baseline = state.baseline_bpm
    for _ in range(50):
        bpm = next_bpm(state, "alert", seconds_to_impact=None, hard_braking=False, rng=rng)
    assert abs(bpm - baseline) < 15.0
