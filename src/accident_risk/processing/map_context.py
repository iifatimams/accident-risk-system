"""Module for querying non-blocking spatial road context from OpenStreetMap."""

import math
import threading
import time
import requests


class MapContextResolver:
    """Queries OpenStreetMap asynchronously for roundabouts, intersections, and road context."""

    def __init__(self, cache_ttl_sec: float = 15.0, radius_meters: float = 35.0):
        self.cache_ttl_sec = cache_ttl_sec
        self.radius_meters = radius_meters
        self._last_lat = None
        self._last_lon = None
        self._last_query_time = 0.0

        self._in_roundabout = False
        self._near_intersection = False

        self._lock = threading.Lock()
        self._fetching = False

    def _haversine_meters(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points in meters."""
        r = 6371000.0
        phi1, phi2 = math.radians(lat1), math.radians(lat2)
        dphi = math.radians(lat2 - lat1)
        dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
        return r * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    def update_location(self, lat: float | None, lon: float | None):
        """Triggers a non-blocking background query if position or cache time has expired."""
        if lat is None or lon is None or lat == 0.0 or lon == 0.0:
            return

        now = time.time()
        should_fetch = False

        if self._last_lat is None or self._last_lon is None:
            should_fetch = True
        else:
            dist = self._haversine_meters(self._last_lat, self._last_lon, lat, lon)
            if dist > 20.0 or (now - self._last_query_time) > self.cache_ttl_sec:
                should_fetch = True

        if should_fetch and not self._fetching:
            self._fetching = True
            threading.Thread(target=self._query_overpass, args=(lat, lon), daemon=True).start()

    def _query_overpass(self, lat: float, lon: float):
        """Background worker thread executing Overpass API queries."""
        query = f"""
        [out:json][timeout:3];
        (
          // Roundabouts
          way(around:{self.radius_meters},{lat},{lon})["junction"="roundabout"];
          way(around:{self.radius_meters},{lat},{lon})["junction"="circular"];
          node(around:{self.radius_meters},{lat},{lon})["highway"="mini_roundabout"];
          
          // Intersections & Junction Features
          node(around:{self.radius_meters},{lat},{lon})["highway"="traffic_signals"];
          node(around:{self.radius_meters},{lat},{lon})["highway"="give_way"];
          node(around:{self.radius_meters},{lat},{lon})["highway"="stop"];
          node(around:{self.radius_meters},{lat},{lon})["highway"="turning_circle"];
          node(around:{self.radius_meters},{lat},{lon})["highway"="motorway_junction"];
        );
        out body;
        """
        try:
            response = requests.post(
                "https://overpass-api.de/api/interpreter",
                data={"data": query},
                timeout=2.5,
            )
            if response.status_code == 200:
                elements = response.json().get("elements", [])
                
                is_rb = False
                is_intersect = False

                for elem in elements:
                    tags = elem.get("tags", {})
                    if tags.get("junction") in ("roundabout", "circular") or tags.get("highway") == "mini_roundabout":
                        is_rb = True
                    if tags.get("highway") in ("traffic_signals", "give_way", "stop", "turning_circle", "motorway_junction"):
                        is_intersect = True

                with self._lock:
                    self._in_roundabout = is_rb
                    self._near_intersection = is_intersect or is_rb
                    self._last_lat = lat
                    self._last_lon = lon
                    self._last_query_time = time.time()
        except Exception:
            pass
        finally:
            self._fetching = False

    def is_in_roundabout(self, lat: float | None = None, lon: float | None = None) -> bool:
        """Check for roundabout context. Accepts optional lat/lon for API backward compatibility."""
        if lat is not None and lon is not None:
            self.update_location(lat, lon)
        with self._lock:
            return self._in_roundabout

    def is_near_intersection(self, lat: float | None = None, lon: float | None = None) -> bool:
        """Check for intersection context. Accepts optional lat/lon for API backward compatibility."""
        if lat is not None and lon is not None:
            self.update_location(lat, lon)
        with self._lock:
            return self._near_intersection