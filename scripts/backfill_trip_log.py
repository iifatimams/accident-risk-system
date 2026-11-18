"""One-off backfill: load real trip heart-rate data (from a personal Apple Health/
Maps export) into data/app.sqlite3 as reconstructed demo sessions, for the user's
own experimentation only.

NOT for the report/submission and NOT evidence of model performance. Every row
this writes is tagged as synthetic/reconstructed:
  - session_points.heart_rate is the one genuinely real value per trip (copied
    verbatim, in order, no interpolation).
  - speed_kmh/rpm/throttle/engine_load and the GPS route are generated to be
    *plausible*, anchored to the real distance/duration/avg-speed reported by
    each trip's Apple Maps "Journey Details" screenshot where available.
  - risk_index/risk_level are left NULL: no trained inference model with a
    real (non-synthetic-fixture) training provenance was available to score
    these frames, and AGENTS.md forbids treating a synthetic-fixture-trained
    model's output as real evidence -- so no score is fabricated here.
  - camera_events are entirely simulated (no real camera frames exist for
    these trips) and every row says so in its `note` column.
  - sessions.summary carries "synthetic": true and a "source" tag.

Run: .venv/bin/python scripts/backfill_trip_log.py
"""

from __future__ import annotations

import json
import random
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

DB_PATH = "data/app.sqlite3"
DRIVER_USER_ID = 2  # Fatima Al Sinan
UAE_TZ = timezone(timedelta(hours=4))

random.seed(20261117)

# Approximate waypoint coordinates, derived from the neighbourhood/road labels
# actually visible in each trip's Apple Maps "Journey Details" screenshot
# (Al Tarfa, Al Shahba, Al Khezamia, University City/Sharjah Healthcare City,
# Al Yash, Saif Zone, House of Wisdom/Mowaileh Camp, Al Juraina, Sharjah
# Airport). A plain screenshot carries no embedded GPS/EXIF, so these are
# still estimates from the map's own street/area labels, NOT extracted
# coordinates -- but each trip now gets its own distinct, direction-matching
# route instead of 3 generic points reused across all five trips.
PLACES = {
    "ibn_katheer": (25.3330, 55.4730),       # Al Tarfa / Ibn Katheer St
    "al_shahba": (25.3395, 55.4660),
    "al_khezamia": (25.3460, 55.4610),
    "sharjah_generic": (25.3480, 55.4550),   # "Sharjah" pin, trip 1's start area
    "university_city": (25.3155, 55.4935),   # near Vox Cinemas / Sharjah Healthcare City
    "al_yash": (25.3230, 55.4870),
    "saif_zone": (25.3230, 55.5080),
    "house_of_wisdom": (25.3050, 55.5000),   # Mowaileh Camp area
    "al_juraina": (25.2950, 55.5050),        # near Sharjah Airport (SHJ)
}


@dataclass
class Trip:
    date: str  # "2026-10-31"
    start_time: str  # "02:18"
    end_time: str  # "02:44" (exclusive, per the user's own note)
    heart_rates: list[float]
    waypoints: list[str]  # PLACES keys, in order, tracing the screenshot's drawn path
    distance_km: float | None = None
    avg_speed_kmh: float | None = None


TRIPS = [
    Trip(
        "2026-10-31", "02:18", "02:44",
        [86, 92, 92, 90, 85, 86, 96, 89, 92, 86,
         87, 89, 94, 88, 85, 91, 89, 91, 87, 90,
         95, 88, 93, 93, 89, 90],
        ["sharjah_generic", "al_khezamia", "al_shahba", "ibn_katheer"],
        distance_km=31.6, avg_speed_kmh=75.0,
    ),
    Trip(
        "2026-10-31", "13:22", "13:52",
        [98, 82, 87, 83, 87, 100, 90, 93, 96, 91,
         95, 95, 91, 88, 98, 108, 90, 94, 95, 91,
         94, 114, 90, 87, 87, 94, 99, 95, 91, 89],
        ["ibn_katheer", "al_shahba", "al_yash", "university_city"],
        distance_km=6.8, avg_speed_kmh=14.0,
    ),
    Trip(
        "2026-10-31", "14:19", "14:51",
        [115, 102, 97, 103, 98, 103, 98, 98, 97, 95,
         95, 98, 97, 93, 101, 130, 124, 121, 125, 108,
         106, 110, 115, 118, 122, 128, 109, 99, 103, 101,
         104, 98],
        # The screenshot shows a rectangular loop (north through Al Yash,
        # then back south-west) rather than retracing trip 2's path.
        ["university_city", "al_yash", "al_khezamia", "al_shahba", "ibn_katheer"],
        distance_km=13.7, avg_speed_kmh=28.0,
    ),
    Trip(
        "2026-11-01", "09:30", "09:49",
        [112, 103, 99, 92, 97, 97, 94, 92, 93, 96,
         95, 98, 99, 93, 92, 92, 97, 96, 98],
        ["ibn_katheer", "saif_zone", "house_of_wisdom"],
        distance_km=8.9, avg_speed_kmh=31.0,
    ),
    Trip(
        "2026-11-01", "12:39", "14:35",
        [100, 83, 85, 85, 84, 82, 87, 85, 82, 88,
         83, 83, 85, 81, 80, 78, 83, 82, 76, 77,
         82, 79, 80, 81, 84, 95, 90, 93, 87, 92,
         88, 97, 92, 96, 94, 94, 90, 100, 98, 96,
         99, 94, 90, 92, 88, 87, 88, 89, 84, 94,
         83, 85, 90, 97, 96, 85, 84, 88, 88, 85,
         89, 87, 93, 86, 89, 90, 90, 80, 83, 89,
         81, 85, 89, 100, 100, 106, 98, 94, 82, 87,
         87, 82, 83, 86, 80, 82, 89, 91, 89, 91,
         88, 108, 104, 90, 85, 87, 87, 92, 81, 81,
         81, 91, 89, 90, 96, 86, 86, 86, 87, 84,
         91, 89, 88, 86, 93, 88],
        # Compass-rotated screenshot (view faces W); interpreted as a south
        # corridor past Saif Zone toward Al Juraina/the airport, distinct
        # from trips 1-4's route.
        ["ibn_katheer", "saif_zone", "al_juraina"],
        # Duration field wasn't visible in the screenshot; derive distance from
        # the HR block's own 116-minute window at a plausible mixed-traffic
        # average speed rather than inventing a precise reported figure.
        distance_km=None, avg_speed_kmh=25.0,
    ),
]

FRONT_LABELS = [
    ("looking_road", 0.78),
    ("brief_glance", 0.12),
    ("looking_down", 0.06),
    ("phone_glance", 0.04),
]
REAR_LABELS = [
    ("no_relevant_lead_object", 0.75),
    ("vehicle_ahead", 0.15),
    ("pedestrian_detected_far", 0.07),
    ("vehicle_ahead_closing", 0.03),
]


def weighted_choice(options: list[tuple[str, float]]) -> str:
    labels, weights = zip(*options)
    return random.choices(labels, weights=weights, k=1)[0]


def uae_local_to_epoch(date: str, time_str: str) -> float:
    hour, minute = (int(part) for part in time_str.split(":"))
    year, month, day = (int(part) for part in date.split("-"))
    dt = datetime(year, month, day, hour, minute, tzinfo=UAE_TZ)
    return dt.timestamp()


def interpolate_route(waypoints: list[tuple[float, float]], n: int) -> list[tuple[float, float]]:
    """Interpolate n points along a multi-waypoint polyline (equal time per
    leg), with light jitter so each leg isn't a perfectly straight line."""
    legs = len(waypoints) - 1
    points = []
    for i in range(n):
        t = i / max(n - 1, 1) * legs
        leg = min(int(t), legs - 1)
        leg_t = t - leg
        start, end = waypoints[leg], waypoints[leg + 1]
        lat = start[0] + (end[0] - start[0]) * leg_t
        lon = start[1] + (end[1] - start[1]) * leg_t
        jitter = 0.0008 * (random.random() - 0.5)
        points.append((lat + jitter, lon + jitter))
    return points


def speed_profile(n: int, avg_speed: float) -> list[float]:
    if n <= 2:
        return [avg_speed] * n
    profile = []
    ramp = max(1, n // 6)
    for i in range(n):
        if i < ramp:
            factor = (i + 1) / ramp
        elif i >= n - ramp:
            factor = (n - i) / ramp
        else:
            factor = 1.0
        noise = 1.0 + 0.12 * (random.random() - 0.5)
        profile.append(max(0.0, avg_speed * factor * noise))
    # Rescale so the mean matches the reported average speed.
    mean = sum(profile) / len(profile)
    if mean > 0:
        profile = [v * (avg_speed / mean) for v in profile]
    return profile


def vehicle_fields_from_speed(speed_kmh: float) -> tuple[float, float, float]:
    idle_rpm = 750.0
    rpm = idle_rpm + speed_kmh * 28.0 + random.uniform(-80, 80)
    throttle = min(100.0, max(0.0, speed_kmh * 0.9 + random.uniform(-4, 4)))
    engine_load = min(100.0, max(0.0, throttle * 0.7 + speed_kmh * 0.2 + random.uniform(-3, 3)))
    return rpm, throttle, engine_load


def build_camera_events(session_id: int, start_ts: float, end_ts: float) -> list[tuple]:
    rows = []
    for camera, labels in (("front", FRONT_LABELS), ("rear", REAR_LABELS)):
        ts = start_ts
        while ts < end_ts:
            label = weighted_choice(labels)
            confidence = round(random.uniform(0.55, 0.95), 3)
            rows.append((
                session_id, ts, camera, label, confidence,
                "simulated — no real camera frames available for this trip",
            ))
            ts += random.uniform(30, 90)
    return rows


def main() -> None:
    connection = sqlite3.connect(DB_PATH)
    connection.execute("PRAGMA foreign_keys = ON")
    cursor = connection.cursor()

    for trip_index, trip in enumerate(TRIPS, start=1):
        started_at = uae_local_to_epoch(trip.date, trip.start_time)
        ended_at = uae_local_to_epoch(trip.date, trip.end_time)
        n_minutes = len(trip.heart_rates)
        assert ended_at - started_at == n_minutes * 60, (
            f"Trip {trip_index}: window does not match reading count "
            f"({ended_at - started_at}s vs {n_minutes} readings)"
        )

        stream_id = f"u{DRIVER_USER_ID}-hist-{trip.date.replace('-', '')}{trip.start_time.replace(':', '')}"
        avg_speed = trip.avg_speed_kmh or 0.0
        distance_km = trip.distance_km
        if distance_km is None and avg_speed:
            distance_km = round(avg_speed * (n_minutes / 60.0), 1)

        route = interpolate_route([PLACES[key] for key in trip.waypoints], n_minutes)
        speeds = speed_profile(n_minutes, avg_speed)

        cursor.execute(
            "INSERT INTO sessions (stream_id, user_id, started_at, ended_at, end_reason, complete, summary) "
            "VALUES (?, ?, ?, ?, ?, 1, ?)",
            (stream_id, DRIVER_USER_ID, started_at, ended_at, "driver_stopped", "{}"),
        )
        session_id = cursor.lastrowid

        mean_hr = sum(trip.heart_rates) / len(trip.heart_rates)
        for i in range(n_minutes):
            ts = started_at + i * 60
            lat, lon = route[i]
            speed_kmh = speeds[i]
            rpm, throttle, engine_load = vehicle_fields_from_speed(speed_kmh)
            cursor.execute(
                "INSERT INTO session_points "
                "(session_id, ts, risk_index, risk_level, speed_kmh, rpm, engine_load, throttle, "
                " heart_rate, latitude, longitude) VALUES (?, ?, NULL, NULL, ?, ?, ?, ?, ?, ?, ?)",
                (session_id, ts, speed_kmh, rpm, engine_load, throttle, trip.heart_rates[i], lat, lon),
            )

        camera_rows = build_camera_events(session_id, started_at, ended_at)
        cursor.executemany(
            "INSERT INTO camera_events (session_id, ts, camera, label, confidence, note) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            camera_rows,
        )

        summary = {
            "duration_seconds": ended_at - started_at,
            "distance_km": distance_km,
            "max_speed_kmh": round(max(speeds), 1) if speeds else None,
            "mean_speed_kmh": round(sum(speeds) / len(speeds), 1) if speeds else None,
            "idling_seconds": 0,
            "mean_risk_index": None,
            "max_risk_index": None,
            "risk_level_seconds": {},
            "mean_heart_rate": round(mean_hr, 1),
            "points": n_minutes,
            "gap_count": 0,
            "longest_gap_seconds": 0.0,
            "complete": True,
            "alerts": 0,
            "synthetic": True,
            "source": "backfilled_from_trip_log_2026-11-17",
            "route_source": "geocoded_from_journey_screenshot_approximate",
        }
        cursor.execute(
            "UPDATE sessions SET summary = ? WHERE id = ?",
            (json.dumps(summary), session_id),
        )

        print(f"Trip {trip_index} ({trip.date} {trip.start_time}-{trip.end_time}): "
              f"session_id={session_id}, {n_minutes} HR points, "
              f"{len(camera_rows)} simulated camera events")

    connection.commit()
    connection.close()


if __name__ == "__main__":
    main()
