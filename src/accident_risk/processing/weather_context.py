"""
Real-time weather context resolver using OpenWeatherMap.
Runs asynchronously to avoid blocking the main inference loop.
"""

import json
import threading
import time
import urllib.request

class WeatherContextResolver:
    """Fetches and caches local weather conditions to apply environmental risk multipliers."""
    
    def __init__(self, api_key: str, poll_interval_sec: int = 300):
        self.api_key = api_key
        self.poll_interval = poll_interval_sec
        self._lock = threading.Lock()
        
        self._last_fetch_time = 0.0
        self._last_coords = (0.0, 0.0)
        
        # Default state (assumes clear weather if API fails or is unconfigured)
        self.condition = "CLEAR"
        self.multiplier = 1.0
        self.description = "Clear sky"

    def update_location(self, lat: float, lon: float):
        """Updates internal coordinate buffer and checks time since last fetch."""
        if lat == 0.0 or lon == 0.0:
            return
            
        now = time.time()
        with self._lock:
            # Only poll the API once every 5 minutes to avoid rate limits
            if (now - self._last_fetch_time) < self.poll_interval:
                return
            self._last_fetch_time = now
            self._last_coords = (lat, lon)
            
        # Fetch asynchronously so we don't stall the camera and OBD feeds
        threading.Thread(
            target=self._query_weather_api,
            args=(lat, lon),
            daemon=True
        ).start()

    def _query_weather_api(self, lat: float, lon: float):
        """Queries OpenWeather API and calculates risk multiplier based on conditions."""
        if self.api_key == "YOUR_OPENWEATHER_API_KEY_HERE" or not self.api_key:
            # If the user hasn't set up a real API key, fallback gracefully
            with self._lock:
                self.condition = "CLEAR"
                self.multiplier = 1.0
                self.description = "Clear sky (No API Key)"
            return

        url = f"https://api.openweathermap.org/data/2.5/weather?lat={lat}&lon={lon}&appid={self.api_key}"
        
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "RoadSense/1.0"})
            with urllib.request.urlopen(req, timeout=3.0) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                
                weather_main = data["weather"][0]["main"].upper()
                weather_desc = data["weather"][0]["description"].capitalize()
                
                # Apply risk scaling multipliers for UAE environmental hazards
                mult = 1.0
                if weather_main in ["RAIN", "DRIZZLE"]:
                    mult = 1.25  # Slippery roads, braking distance increased
                elif weather_main in ["THUNDERSTORM"]:
                    mult = 1.40  # Severe hazard
                elif weather_main in ["MIST", "FOG", "HAZE", "SAND", "DUST"]:
                    mult = 1.15  # Low visibility
                    
                with self._lock:
                    self.condition = weather_main
                    self.multiplier = mult
                    self.description = weather_desc
                    
        except Exception:
            # If API fails, internet drops, or timeout occurs, retain the last known weather
            pass

    def get_context(self) -> tuple[str, float, str]:
        """Returns the active weather state safely for the main thread."""
        with self._lock:
            return self.condition, self.multiplier, self.description