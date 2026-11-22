"""Export one row per session: duration, GPS-derived distance, recording
completeness, sensor availability by source, and reliability status.

Run:
  .venv/bin/python scripts/export_session_summaries.py
  .venv/bin/python scripts/export_session_summaries.py --phone "+9715..."
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import Database, normalise_phone  # noqa: E402

FIELDS = [
    "session_id", "driver_id", "driver_name", "driver_phone", "stream_id",
    "started_at_utc", "ended_at_utc", "end_reason", "complete",
    "duration_seconds", "distance_km", "distance_source", "distance_calculation_method",
    "max_speed_kmh", "mean_speed_kmh", "mean_heart_rate",
    "mean_risk_index", "max_risk_index", "alerts", "gap_count", "longest_gap_seconds",
    "overall_sensor_availability", "sensor_availability_by_source",
    "reliability_status",
]


def _reliability_status(session: dict) -> str:
    reason = session.get("end_reason")
    if reason == "orphaned_on_restart":
        return "application_interruption"
    if reason == "timeout":
        return "temporary_sensor_interruption_or_timeout"
    summary = session.get("summary") or {}
    if not summary.get("complete", True):
        return "session_recording_incomplete"
    return "session_ended_normally"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--phone", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    db = Database(Path(args.db))
    user_id = None
    if args.phone:
        user = db.get_user_by_phone(normalise_phone(args.phone))
        if user is None:
            raise SystemExit(f"No driver registered with phone {args.phone!r}")
        user_id = user["id"]

    sessions = db.list_sessions(user_id=user_id, limit=100000)
    out_path = Path(args.out) if args.out else Path(args.db).parent / "session_summaries.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for s in sessions:
            summary = s.get("summary") or {}
            writer.writerow({
                "session_id": s["id"],
                "driver_id": s["user_id"],
                "driver_name": s.get("driver_name"),
                "driver_phone": s.get("driver_phone"),
                "stream_id": s["stream_id"],
                "started_at_utc": datetime.fromtimestamp(s["started_at"], tz=UTC).isoformat(),
                "ended_at_utc": (
                    datetime.fromtimestamp(s["ended_at"], tz=UTC).isoformat()
                    if s.get("ended_at") else None
                ),
                "end_reason": s.get("end_reason"),
                "complete": bool(s.get("complete")),
                "duration_seconds": summary.get("duration_seconds"),
                "distance_km": summary.get("distance_km"),
                "distance_source": summary.get("distance_source"),
                "distance_calculation_method": summary.get("distance_calculation_method"),
                "max_speed_kmh": summary.get("max_speed_kmh"),
                "mean_speed_kmh": summary.get("mean_speed_kmh"),
                "mean_heart_rate": summary.get("mean_heart_rate"),
                "mean_risk_index": summary.get("mean_risk_index"),
                "max_risk_index": summary.get("max_risk_index"),
                "alerts": summary.get("alerts"),
                "gap_count": summary.get("gap_count"),
                "longest_gap_seconds": summary.get("longest_gap_seconds"),
                "overall_sensor_availability": summary.get("overall_sensor_availability"),
                "sensor_availability_by_source": json.dumps(
                    summary.get("sensor_availability_by_source") or {}
                ),
                "reliability_status": _reliability_status(s),
            })

    print(f"{len(sessions)} sessions -> {out_path}")


if __name__ == "__main__":
    main()
