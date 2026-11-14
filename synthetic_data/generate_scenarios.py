"""Generate synthetic multimodal driving episodes with MetaDrive.

SYNTHETIC SOFTWARE TEST FIXTURE -- see synthetic_data/README.md. Output is
never evidence of real-world model performance (AGENTS.md: "Synthetic
fixtures are software tests, never evidence of model performance") and must
never be substituted for real OBD-II, physiology, driver-camera, or
front-camera data when evaluating the actual accident-risk model.

Requires the optional "simulation" extra (metadrive-simulator,
opencv-python-headless); not part of the core accident_risk runtime:

    pip install -e ".[simulation]"

Writes:
  <out-dir>/images/ep_{episode_id}_step_{step}.jpg   front-camera RGB frames
  <out-dir>/telemetry_log.csv                         one row per saved frame

Episode generation:
  - "crash" episodes: driver_state is distracted_phone or drowsy_asleep for
    the whole episode; throttle never responds to the obstacle ahead, so a
    collision with dense traffic is the expected (not guaranteed) outcome.
  - "safe" episodes: driver_state is alert; throttle/brake react to the
    nearest in-lane obstacle (simple safe-following-distance controller),
    so the vehicle is expected to avoid collision in the same traffic.

Both policies share the same lane-centering steering and the same traffic
density/map family, so the only deliberate difference between the two
classes is attentiveness -> braking behavior, matching the brief's
"exact same crowded traffic" requirement as closely as a procedurally
generated map allows (seeds differ per episode; MetaDrive does not support
replaying one fixed traffic layout under two different control policies in
the same process run).
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from synthetic_data import macos_metadrive_patch
from synthetic_data.obd_physio_synthesis import (
    BpmState,
    derive_engine_load,
    derive_rpm,
    label_imminent_accident,
    min_obstacle_distance,
    next_bpm,
)

STEP_DT_S = 0.1  # MetaDrive default: decision_repeat(5) * physics_world_step_size(0.02)
IMAGE_WIDTH, IMAGE_HEIGHT = 84, 60
SAFE_FOLLOWING_GAP_M = 12.0
MAX_EPISODE_STEPS = 200  # 20s ceiling; episodes end earlier on collision or crash-free timeout


@dataclass
class EpisodeRow:
    episode_id: int
    scenario_type: str
    step: int
    timestamp_s: float
    image_path: str
    speed_mps: float
    rpm: float
    throttle_frac: float
    brake_frac: float
    engine_load_pct: float
    heart_rate_bpm: float
    driver_state: str
    min_obstacle_dist_m: float
    collision_occurred: int
    imminent_accident_3s: int = 0


@dataclass
class EpisodeResult:
    rows: list[EpisodeRow] = field(default_factory=list)
    collision_timestamp_s: float | None = None


def build_env(num_scenarios: int, start_seed: int, traffic_density: float):
    macos_metadrive_patch.apply()
    from metadrive.component.sensors.rgb_camera import RGBCamera
    from metadrive.envs.metadrive_env import MetaDriveEnv

    return MetaDriveEnv(
        dict(
            use_render=False,
            image_observation=True,
            vehicle_config=dict(image_source="rgb_camera"),
            sensors=dict(rgb_camera=(RGBCamera, IMAGE_WIDTH, IMAGE_HEIGHT)),
            traffic_density=traffic_density,
            map="S",
            num_scenarios=num_scenarios,
            start_seed=start_seed,
            accident_prob=0.0,
            horizon=MAX_EPISODE_STEPS,
        )
    )


def lane_centering_steering(agent) -> float:
    """Small proportional controller to keep the ego vehicle near its lane center."""
    lane = agent.lane
    if lane is None:
        return 0.0
    longitudinal, lateral = lane.local_coordinates(agent.position)
    target_heading = lane.heading_theta_at(longitudinal)
    heading_error = math.atan2(math.sin(target_heading - agent.heading_theta), math.cos(target_heading - agent.heading_theta))
    steering = -0.6 * lateral + 1.2 * heading_error
    return min(max(steering, -1.0), 1.0)


def choose_action(scenario_type: str, driver_state: str, min_dist_m: float) -> tuple[float, float, float]:
    """Return (steering placeholder unused here, throttle_frac, brake_frac)."""
    del scenario_type  # behavior is fully determined by driver_state below
    if driver_state == "alert":
        if not math.isnan(min_dist_m) and min_dist_m < SAFE_FOLLOWING_GAP_M:
            brake_frac = min(1.0, (SAFE_FOLLOWING_GAP_M - min_dist_m) / SAFE_FOLLOWING_GAP_M + 0.3)
            return 0.0, 0.0, brake_frac
        return 0.0, 0.5, 0.0
    # distracted_phone / drowsy_asleep: ignore the obstacle ahead entirely.
    return 0.0, 0.6, 0.0


def run_episode(env, episode_id: int, scenario_type: str, out_dir: Path, rng: random.Random) -> EpisodeResult:
    driver_state = rng.choice(["distracted_phone", "drowsy_asleep"]) if scenario_type == "crash" else "alert"
    obs, _info = env.reset()
    bpm_state = BpmState.for_driver_state(driver_state, rng)

    pending_rows: list[EpisodeRow] = []
    collision_timestamp_s: float | None = None
    previous_speed_mps = env.agent.speed

    for step in range(MAX_EPISODE_STEPS):
        others = [tuple(v.position) for v in env.engine.traffic_manager.vehicles]
        min_dist_m = min_obstacle_distance(
            env.agent.position[0], env.agent.position[1], env.agent.heading_theta, others
        )
        steering = lane_centering_steering(env.agent)
        _, throttle_frac, brake_frac = choose_action(scenario_type, driver_state, min_dist_m)
        action_scalar = throttle_frac if brake_frac == 0.0 else -brake_frac

        speed_mps_before = env.agent.speed
        obs, _reward, terminated, truncated, info = env.step([steering, action_scalar])
        timestamp_s = (step + 1) * STEP_DT_S
        speed_mps = env.agent.speed
        accel_mps2 = (speed_mps - previous_speed_mps) / STEP_DT_S
        previous_speed_mps = speed_mps

        collided_this_step = bool(info.get("crash_vehicle") or info.get("crash_object"))
        if collided_this_step and collision_timestamp_s is None:
            collision_timestamp_s = timestamp_s

        hard_braking = brake_frac > 0.5
        heart_rate_bpm = next_bpm(
            bpm_state,
            driver_state,
            seconds_to_impact=None,  # filled in after the episode, once collision time is known
            hard_braking=hard_braking,
            rng=rng,
        )

        image = np.asarray(obs["image"][..., -1] * 255.0, dtype=np.uint8)  # most recent frame of the stack
        image_rel_path = f"images/ep_{episode_id}_step_{step}.jpg"
        _save_jpg(image, out_dir / image_rel_path)

        pending_rows.append(
            EpisodeRow(
                episode_id=episode_id,
                scenario_type=scenario_type,
                step=step,
                timestamp_s=timestamp_s,
                image_path=image_rel_path,
                speed_mps=speed_mps_before,
                rpm=derive_rpm(speed_mps_before, throttle_frac),
                throttle_frac=throttle_frac,
                brake_frac=brake_frac,
                engine_load_pct=derive_engine_load(throttle_frac, accel_mps2),
                heart_rate_bpm=heart_rate_bpm,
                driver_state=driver_state,
                min_obstacle_dist_m=min_dist_m,
                collision_occurred=0,  # patched below once collision_timestamp_s is known
            )
        )

        if collided_this_step or terminated or truncated:
            break

    if collision_timestamp_s is not None:
        for row in pending_rows:
            if math.isclose(row.timestamp_s, collision_timestamp_s, abs_tol=1e-9):
                row.collision_occurred = 1

    timestamps = [row.timestamp_s for row in pending_rows]
    imminent = label_imminent_accident(timestamps, collision_timestamp_s)
    for row, flag in zip(pending_rows, imminent, strict=True):
        row.imminent_accident_3s = flag

    return EpisodeResult(rows=pending_rows, collision_timestamp_s=collision_timestamp_s)


def _save_jpg(image_hwc_uint8: np.ndarray, path: Path) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(image_hwc_uint8, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr)


def generate(
    out_dir: Path, n_crash: int, n_safe: int, seed: int, traffic_density: float = 0.35
) -> pd.DataFrame:
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    total_episodes = n_crash + n_safe
    env = build_env(num_scenarios=total_episodes, start_seed=seed, traffic_density=traffic_density)
    rng = random.Random(seed)

    all_rows: list[dict] = []
    try:
        episode_id = 0
        for scenario_type, count in (("crash", n_crash), ("safe", n_safe)):
            for _ in range(count):
                result = run_episode(env, episode_id, scenario_type, out_dir, rng)
                all_rows.extend(row.__dict__ for row in result.rows)
                episode_id += 1
    finally:
        env.close()

    columns = [
        "episode_id",
        "scenario_type",
        "step",
        "timestamp_s",
        "image_path",
        "speed_mps",
        "rpm",
        "throttle_frac",
        "brake_frac",
        "engine_load_pct",
        "heart_rate_bpm",
        "driver_state",
        "min_obstacle_dist_m",
        "collision_occurred",
        "imminent_accident_3s",
    ]
    df = pd.DataFrame(all_rows, columns=columns)
    df.to_csv(out_dir / "telemetry_log.csv", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-crash", type=int, default=5)
    parser.add_argument("--n-safe", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--traffic-density", type=float, default=0.35)
    args = parser.parse_args()

    if "raw" in args.out_dir.parts:
        raise SystemExit("Refusing to write synthetic output into data/raw (see AGENTS.md).")

    df = generate(args.out_dir, args.n_crash, args.n_safe, args.seed, args.traffic_density)
    n_collisions = int(df.groupby("episode_id")["collision_occurred"].max().sum())
    print(
        f"Wrote {len(df)} rows across {df['episode_id'].nunique()} episodes "
        f"({n_collisions} with a recorded collision) to {args.out_dir}"
    )


if __name__ == "__main__":
    main()
