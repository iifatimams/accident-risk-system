"""
accident_risk/processing/map_context.py
Autonomous real-time map context and road speed resolver for the UAE.
Operates using Overpass OSM API with local spatial hashing and Emirate boundary detection.
"""

import json
import math
import threading
import time
import urllib.parse
import urllib.request


class MapContextResolver:
    """
    Self-updating road speed limit and geometry resolver across UAE Emirates.
    """

    EMIRATE_DEFAULTS = {
        "dubai_sharjah_northern": {
            "motorway": 120.0,
            "trunk": 100.0,
            "primary": 80.0,
            "secondary": 60.0,
            "tertiary": 50.0,
            "residential": 40.0,
            "roundabout": 40.0,
            "intersection": 50.0,
            "buffer": 20.0,
        },
        "abu_dhabi": {
            "motorway": 140.0,
            "trunk": 120.0,
            "primary": 80.0,
            "secondary": 60.0,
            "tertiary": 40.0,
            "residential": 40.0,
            "roundabout": 40.0,
            "intersection": 40.0,
            "buffer": 0.0,
        },
    }

    def __init__(self):
        self._lock = threading.Lock()
        self.current_lat = 0.0
        self.current_lon = 0.0

        self.current_road_name = "Connecting Highway"
        self.current_emirate = "Sharjah"
        self.design_speed = 80.0
        self.radar_threshold = 100.0
        self.is_roundabout_state = False
        self.is_intersection_state = False

        self._cache = {}
        self._last_fetch_time = 0.0
        self._last_coords = (0.0, 0.0)

    def _detect_emirate(self, lat: float, lon: float) -> str:
        """Determines the active Emirate jurisdiction based on coordinate boundary."""
        if lat < 24.85 and lon < 55.10:
            return "abu_dhabi"
        return "dubai_sharjah_northern"

    def update_location(self, lat: float, lon: float):
        """Updates internal coordinate buffer and checks spatial cache."""
        if lat == 0.0 or lon == 0.0:
            return

        with self._lock:
            self.current_lat = lat
            self.current_lon = lon

        grid_key = (round(lat, 4), round(lon, 4))
        now = time.time()

        with self._lock:
            if grid_key in self._cache:
                cached_data, _ = self._cache[grid_key]
                self._apply_road_data(cached_data)
                return

        dist_moved = math.hypot(lat - self._last_coords[0], lon - self._last_coords[1]) * 111000
        if dist_moved > 40.0 or (now - self._last_fetch_time) > 6.0:
            self._last_fetch_time = now
            self._last_coords = (lat, lon)
            threading.Thread(
                target=self._query_overpass_road_graph,
                args=(lat, lon, grid_key),
                daemon=True,
            ).start()

    def _query_overpass_road_graph(self, lat: float, lon: float, grid_key: tuple):
        """Dispatches an Overpass API request to find live speed limit tags."""
        radius_m = 45
        query = f"[out:json][timeout:2];way(around:{radius_m},{lat},{lon})[\"highway\"];out tags 1;"
        url = "https://overpass-api.de/api/interpreter?data=" + urllib.parse.quote(query)

        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "RoadSense-Telematics-UAE/1.0"}
            )
            with urllib.request.urlopen(req, timeout=2.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                elements = data.get("elements", [])

                if elements:
                    tags = elements[0].get("tags", {})
                    road_payload = self._parse_road_tags(tags, lat, lon)

                    with self._lock:
                        self._cache[grid_key] = (road_payload, time.time())
                        if len(self._cache) > 1000:
                            self._cache.clear()
                        self._apply_road_data(road_payload)
        except (OSError, json.JSONDecodeError, KeyError):
            self._apply_heuristic_fallback(lat, lon)

    def _parse_road_tags(self, tags: dict, lat: float, lon: float) -> dict:
        """Extracts road name and maps statutory UAE road limits and buffers."""
        highway_type = tags.get("highway", "primary")
        junction = tags.get("junction", "")
        name = tags.get("name:en") or tags.get("name") or "Public Street"

        is_rbt = junction == "roundabout" or highway_type == "roundabout"
        emirate = self._detect_emirate(lat, lon)
        rules = self.EMIRATE_DEFAULTS[emirate]

        design_spd = None
        if "maxspeed" in tags:
            digits = "".join(c for c in tags["maxspeed"] if c.isdigit())
            if digits:
                design_spd = float(digits)

        if design_spd is None:
            if is_rbt:
                design_spd = rules["roundabout"]
            else:
                design_spd = rules.get(highway_type, 80.0)

        radar_spd = design_spd + rules["buffer"]

        return {
            "road_name": name,
            "highway_type": highway_type,
            "design_speed": design_spd,
            "radar_threshold": radar_spd,
            "is_roundabout": is_rbt,
            "emirate": emirate,
        }

    def _apply_road_data(self, data: dict):
        """Applies parsed navigation dictionary to instance attributes."""
        self.current_road_name = data["road_name"]
        self.design_speed = data["design_speed"]
        self.radar_threshold = data["radar_threshold"]
        self.is_roundabout_state = data["is_roundabout"]
        self.current_emirate = data["emirate"]

    def _apply_heuristic_fallback(self, lat: float, lon: float):
        """Applies local statutory fallback when map tiles are unreachable."""
        emirate = self._detect_emirate(lat, lon)
        rules = self.EMIRATE_DEFAULTS[emirate]
        with self._lock:
            self.design_speed = rules["primary"]
            self.radar_threshold = rules["primary"] + rules["buffer"]
            self.current_emirate = emirate

    def get_design_speed(self) -> float:
        """Returns the current legal speed limit in km/h."""
        with self._lock:
            return self.design_speed

    def get_radar_threshold(self) -> float:
        """Returns the speed limit plus regional radar tolerance buffer in km/h."""
        with self._lock:
            return self.radar_threshold

    def get_road_name(self) -> str:
        """Returns the human-readable street or expressway name."""
        with self._lock:
            return self.current_road_name

    def is_in_roundabout(self) -> bool:
        """Checks if current position matches an active roundabout node."""
        with self._lock:
            return self.is_roundabout_state

    def is_near_intersection(self) -> bool:
        """Checks if the vehicle is approaching a signalized junction."""
        with self._lock:
            return self.is_intersection_state
        