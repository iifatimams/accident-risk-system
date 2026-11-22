"""GPS-based trip distance. Never inferred from reported speed alone."""

import math

EARTH_RADIUS_KM = 6371.0088
# A generous upper bound (~300 km/h) on real vehicle speed between two GPS
# fixes; an implied speed above this is treated as a GPS glitch/jump, not
# travelled distance.
MAX_REALISTIC_SPEED_MPS = 83.3
# Two fixes closer together than this are indistinguishable from ordinary
# GPS drift while stationary; the gap between them is not counted as travel.
STATIONARY_DRIFT_METERS = 5.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance between two lat/lon points, in kilometers."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(min(1.0, math.sqrt(a)))


def _valid_coordinate(lat: float | None, lon: float | None) -> bool:
    return lat is not None and lon is not None and -90 <= lat <= 90 and -180 <= lon <= 180


def trip_distance_km(points: list[dict]) -> float | None:
    """Sum of great-circle distances between consecutive valid, ordered GPS
    fixes (each a dict with at least ts/latitude/longitude).

    - Invalid coordinates (missing, or outside the valid lat/lon range) are
      excluded entirely, not treated as zero movement.
    - A jump implying an unrealistic speed between two fixes is excluded as
      a likely GPS glitch, not counted as travel.
    - A hop smaller than ordinary GPS drift while stationary is excluded.
    - Returns None (not 0.0) when fewer than two valid fixes exist, since
      "no distance measurable" is different from "measured zero distance".
    """
    valid = [
        p for p in points
        if _valid_coordinate(p.get("latitude"), p.get("longitude"))
    ]
    if len(valid) < 2:
        return None

    total_km = 0.0
    for previous, current in zip(valid, valid[1:], strict=False):
        dt = current["ts"] - previous["ts"]
        if dt <= 0:
            continue
        segment_km = haversine_km(
            previous["latitude"], previous["longitude"],
            current["latitude"], current["longitude"],
        )
        implied_speed_mps = (segment_km * 1000.0) / dt
        if implied_speed_mps > MAX_REALISTIC_SPEED_MPS:
            continue
        if segment_km * 1000.0 < STATIONARY_DRIFT_METERS:
            continue
        total_km += segment_km
    return total_km
