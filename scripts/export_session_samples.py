"""Export one driving session's full per-frame record (session_samples) to a
CSV, for reviewing a drive after the fact -- pitch/yaw/roll/EAR, eyes-closed
duration, speed/RPM/throttle, GPS, weather, pedestrian/forward-hazard state,
every field api/recorder.py's _record_full_sample captures, one row per
frame, in time order.

Read-only: never writes to data/app.sqlite3, only reads it.

Run:
  .venv/bin/python scripts/export_session_samples.py --session-id 31
  .venv/bin/python scripts/export_session_samples.py --latest
  .venv/bin/python scripts/export_session_samples.py --latest --phone "+9715..."
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import Database, normalise_phone  # noqa: E402

# Column order for the CSV -- every field _record_full_sample writes, plus
# the two timestamp columns and sample metadata. A field absent from a given
# row's payload (older samples recorded before a field was added) renders
# as an empty cell, never a fabricated 0/False.
FIELDS = [
    "ts", "clock_utc", "device_id", "source",
    "speed_kmh", "rpm", "throttle_position", "engine_load",
    "accel_x", "accel_y", "accel_z", "gyro_x", "gyro_y", "gyro_z",
    "gps_latitude", "gps_longitude", "gps_heading", "is_roundabout",
    "heart_rate", "spo2", "hrv_rmssd", "rr_interval_ms", "physiology_signal_quality",
    "weather_multiplier", "weather_condition_code", "weather_is_live",
    "eyes_on_road", "attention_state", "eyes_closed_sec",
    "debug_pitch", "debug_yaw", "debug_roll", "debug_ear",
    "debug_face_detected", "camera_status",
    "traffic_level", "pedestrians_detected", "pedestrian_count", "pedestrian_hazard_state",
    "lead_closing_fast", "forward_hazard_state",
    "warning_fired",
    "event_detection_score", "vehicle_risk_score", "driver_state_risk",
    "overall_risk_index", "risk_level", "data_quality", "model_version",
    "quality_flags", "sensor_age_seconds", "inference_latency_ms",
]


def resolve_session_id(db: Database, args: argparse.Namespace) -> int:
    if args.session_id is not None:
        return args.session_id
    user_id: int | None = None
    if args.phone:
        user = db.get_user_by_phone(normalise_phone(args.phone))
        if user is None:
            raise SystemExit(f"No driver registered with phone {args.phone!r}")
        user_id = user["id"]
    sessions = db.list_sessions(user_id=user_id, limit=1)
    if not sessions:
        raise SystemExit("No sessions found" + (f" for {args.phone}" if args.phone else ""))
    return sessions[0]["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--latest", action="store_true", help="Use the most recent session.")
    parser.add_argument("--phone", default=None, help="With --latest, restrict to this driver's most recent session.")
    parser.add_argument("--out", default=None, help="Output CSV path. Defaults next to --db.")
    args = parser.parse_args()
    if args.session_id is None and not args.latest:
        parser.error("Pass --session-id or --latest")

    db = Database(Path(args.db))
    session_id = resolve_session_id(db, args)
    session = db.get_session(session_id)
    if session is None:
        raise SystemExit(f"Session {session_id} not found")
    samples = db.session_samples(session_id)
    if not samples:
        raise SystemExit(f"Session {session_id} has no recorded samples")

    out_path = Path(args.out) if args.out else Path(args.db).parent / f"session_{session_id}_samples.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in samples:
            row = dict(row)
            row["clock_utc"] = datetime.fromtimestamp(row["ts"], tz=timezone.utc).isoformat()
            writer.writerow(row)

    driver_name = session.get("driver_name", f"user {session['user_id']}")
    print(f"Session {session_id} ({driver_name}): {len(samples)} frames -> {out_path}")


if __name__ == "__main__":
    main()
