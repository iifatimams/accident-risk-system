"""Real-time multimodal terminal runner with proactive context-aware risk logic."""
from collections import deque
import csv
import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import math
import os
import sys
import threading
import time
import urllib.request
import joblib
import numpy as np
import pandas as pd

from accident_risk.processing.map_context import MapContextResolver

# =========================================================
# PATH & MODEL CONFIGURATION
# =========================================================
FUSION_MODEL_PATH = "artifacts/fusion_model/fusion_model.joblib"
FUSION_SCALER_PATH = "artifacts/fusion_model/scaler.joblib"
BENCHMARK_MODEL_PATH = "artifacts/benchmark_models/vzcrash_xgb.joblib"
ANOMALY_MODEL_PATH = "artifacts/benchmark_models/unsupervised_iso.joblib"
BENCHMARK_SCALER_PATH = "artifacts/benchmark_models/benchmark_scaler.joblib"

CLASS_MAP = {0: "CRASH (Class 0)", 1: "NEAR-MISS (Class 1)", 2: "NORMAL (Class 2)"}
SERVE_INGEST_URL = "http://127.0.0.1:8000/api/v1/ingest"
HARDWARE_TIMEOUT_SEC = 10.0

POLIDRIVING_FEATURES = [
    "observation_hour", "speed", "rpm", "acceleration", "throttle_position",
    "engine_temperature", "engine_load_value", "heart_rate", "current_weather",
    "visibility", "precipitation", "accidents_onsite", "design_speed", "accidents_time",
]

state_lock = threading.Lock()


class DriverContext:
    def __init__(self, stream_id: str):
        self.stream_id = stream_id
        self.lat = 0.0
        self.lon = 0.0
        self.accel_mag = 0.0
        self.gyro_mag = 0.0
        self.speed = 0.0
        self.rpm = 0.0
        self.throttle = 0.0
        self.load = 0.0
        self.instant_hr = 0.0
        self.accel_obd = 0.0

        # --- New Context Variables ---
        self.resting_hr = None  # Will auto-calibrate on first connection
        self.eyes_on_road = True
        self.lead_closing_fast = False

        self.last_stream_time = 0.0
        self.last_obd_time = 0.0
        self.last_whoop_time = 0.0

        self.obd_connected = False
        self.whoop_connected = False

        self.prev_speed_mps = None
        self.prev_time = None
        self.smoothed_accel = 0.0
        self.smoothed_fused_risk = 0.0

        self.history_buffer = deque(maxlen=30)


drivers: dict[str, DriverContext] = {}


def get_driver(stream_id: str) -> DriverContext:
    if stream_id not in drivers:
        drivers[stream_id] = DriverContext(stream_id)
    return drivers[stream_id]


def async_forward_to_dashboard(payload: dict):
    """Pushes calculated ML metrics directly to the FastAPI server."""
    def send():
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                SERVE_INGEST_URL,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1.0):
                pass
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
    threading.Thread(target=send, daemon=True).start()


class UnifiedIngestHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        now = time.time()
        try:
            payload = json.loads(body.decode("utf-8"))
            frames = payload if isinstance(payload, list) else [payload]
            with state_lock:
                for frame in frames:
                    if not isinstance(frame, dict):
                        continue
                    sid = str(frame.get("stream_id") or "driver-default")
                    driver = get_driver(sid)
                    driver.last_stream_time = now

                    # Parse standard sensors
                    lat = frame.get("lat") or frame.get("gps_latitude")
                    lon = frame.get("lon") or frame.get("gps_longitude")
                    if lat is not None and lon is not None:
                        driver.lat = float(lat)
                        driver.lon = float(lon)

                    if "accel_mag" in frame:
                        driver.accel_mag = float(frame["accel_mag"])
                    if "gyro_mag" in frame:
                        driver.gyro_mag = float(frame["gyro_mag"])

                    obd_conn = bool(frame.get("obd_connected") or frame.get("is_obd_connected"))
                    driver.obd_connected = obd_conn

                    driver.speed = float(frame.get("speed") or frame.get("vehicle_speed") or 0.0)
                    driver.rpm = float(frame.get("rpm") or frame.get("engine_speed") or 0.0)
                    driver.throttle = float(frame.get("throttle") or frame.get("throttle_position") or 0.0)
                    driver.load = float(frame.get("load") or frame.get("engine_load") or 0.0)

                    if obd_conn or driver.speed > 0 or driver.rpm > 0:
                        driver.last_obd_time = now

                    hr_raw = float(frame.get("heart_rate") or frame.get("bpm") or frame.get("hr") or 0.0)
                    if hr_raw > 0:
                        driver.instant_hr = hr_raw
                        driver.whoop_connected = True
                        driver.last_whoop_time = now
                        
                        # Auto-calibrate resting HR to the first valid heart rate seen
                        if driver.resting_hr is None:
                            driver.resting_hr = hr_raw
                    else:
                        if (now - driver.last_whoop_time) > HARDWARE_TIMEOUT_SEC:
                            driver.whoop_connected = False

                    # Parse new Vision Camera Flags (Default to safe if not provided yet)
                    driver.eyes_on_road = frame.get("eyes_on_road", True)
                    driver.lead_closing_fast = frame.get("lead_closing_fast", False)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            self.send_response(400)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass


def start_receiver():
    server = HTTPServer(("0.0.0.0", 8003), UnifiedIngestHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()


def main():
    if not os.path.exists(BENCHMARK_MODEL_PATH) or not os.path.exists(FUSION_MODEL_PATH):
        print("Model files missing! Please run training scripts first.")
        sys.exit(1)

    print("Loading Model Artifacts...")
    fusion_model = joblib.load(FUSION_MODEL_PATH)
    fusion_scaler = joblib.load(FUSION_SCALER_PATH)
    benchmark_model = joblib.load(BENCHMARK_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    benchmark_scaler = joblib.load(BENCHMARK_SCALER_PATH)
    map_resolver = MapContextResolver()

    print("Listening for multi-driver phone telemetry stream on port 8003...")
    start_receiver()

    print("\n" + "=" * 105)
    print("PROACTIVE CONTEXT-AWARE INFERENCE ENGINE ONLINE (STUDENT LEVEL LOGIC)")
    print("=" * 105 + "\n")

    try:
        while True:
            current_time = time.time()
            active_streams = []

            with state_lock:
                for sid, d in list(drivers.items()):
                    if (current_time - d.last_stream_time) <= HARDWARE_TIMEOUT_SEC:
                        active_streams.append(sid)

            if not active_streams:
                time.sleep(0.5)
                continue

            for sid in active_streams:
                with state_lock:
                    driver = drivers[sid]
                    lat, lon = driver.lat, driver.lon
                    speed_kmh, rpm = driver.speed, driver.rpm
                    throttle, load = driver.throttle, driver.load
                    raw_accel_mag = driver.accel_mag
                    gyro_mag = driver.gyro_mag
                    hr = driver.instant_hr
                    
                    # Ensure resting_hr has a fallback just in case
                    resting_hr = driver.resting_hr if driver.resting_hr is not None else 65.0
                    
                    eyes_on_road = driver.eyes_on_road
                    lead_closing_fast = driver.lead_closing_fast

                    obd_connected = driver.obd_connected
                    whoop_connected = driver.whoop_connected
                    obd_active = obd_connected or (
                        (current_time - driver.last_obd_time) <= HARDWARE_TIMEOUT_SEC
                        and (speed_kmh > 0 or rpm > 0)
                    )
                    whoop_active = whoop_connected and hr > 0

                map_resolver.update_location(lat, lon)
                design_speed = map_resolver.get_design_speed()
                road_name = map_resolver.get_road_name()
                speed_bump_ahead = map_resolver.is_speed_bump_ahead()

                if driver.smoothed_accel > 0:
                    driver.smoothed_accel = (0.3 * raw_accel_mag) + (0.7 * driver.smoothed_accel)
                else:
                    driver.smoothed_accel = raw_accel_mag

                speed_mps = speed_kmh * (1000.0 / 3600.0)
                accel_obd = 0.0
                if driver.prev_speed_mps is not None and driver.prev_time is not None:
                    dt = current_time - driver.prev_time
                    if dt > 0:
                        accel_obd = (speed_mps - driver.prev_speed_mps) / dt
                driver.prev_speed_mps = speed_mps
                driver.prev_time = current_time

                # --- 1. POLIDRIVING BASELINE MODEL ---
                if obd_active or whoop_active:
                    fusion_df = pd.DataFrame([{
                        "observation_hour": float(datetime.datetime.now().hour),
                        "speed": round(speed_kmh, 2),
                        "rpm": round(rpm, 2),
                        "acceleration": round(accel_obd, 3),
                        "throttle_position": round(throttle, 2),
                        "engine_temperature": 92.0,
                        "engine_load_value": round(load, 2),
                        "heart_rate": float(hr) if hr > 0 else 75.0,
                        "current_weather": 6.0,
                        "visibility": 10.0,
                        "precipitation": 0.0,
                        "accidents_onsite": 2.0,
                        "design_speed": float(design_speed),
                        "accidents_time": 1.0,
                    }])[POLIDRIVING_FEATURES]
                    fusion_probs = fusion_model.predict_proba(fusion_scaler.transform(fusion_df))[0]
                    base_risk = float(np.sum(fusion_probs[1:])) * 100.0
                else:
                    base_risk = 0.0

                # --- 2. VZCRASH DETECTION (NOW SEPARATED FROM PREDICTION RISK) ---
                effective_accel = driver.smoothed_accel if driver.smoothed_accel > 0 else abs(accel_obd)
                
                # If a speed bump is coming up, ignore vertical G-force spikes!
                if speed_bump_ahead:
                    effective_accel = min(effective_accel, 9.8)

                benchmark_df = pd.DataFrame([{
                    "effective_accel": effective_accel,
                    "gyro_mag": gyro_mag,
                    "speed_kmh": speed_kmh,
                    "abs_accel_obd": abs(accel_obd),
                }])
                benchmark_scaled = benchmark_scaler.transform(benchmark_df)
                vz_probs = benchmark_model.predict_proba(benchmark_scaled)[0]
                class_pred = int(np.argmax(vz_probs))
                is_anomaly = anomaly_model.predict(benchmark_scaled)[0] == -1 if effective_accel >= 22.0 else False

                # Did an actual accident just happen?
                accident_detected = False
                if class_pred == 0 or is_anomaly:
                    accident_detected = True

                # --- 3. THE NEW CONTEXT-AWARE RISK LOGIC ---
                
                target_risk = base_risk
                context_label = "NORMAL"

                # Calculate Heart Rate Stress (Delta from Resting)
                hr_delta = hr - resting_hr if whoop_active else 0.0
                driver_is_calm = hr_delta < 20.0

                if accident_detected:
                    target_risk = 100.0
                    context_label = "CRASH DETECTED"
                    
                # PARKED / IDLING OVERRIDE
                elif speed_kmh < 2.0 and effective_accel < 12.0:
                    target_risk = 0.0
                    context_label = "PARKED / IDLING"

                elif speed_kmh > 0:
                    # RULE 1: Immediate Hazard (Distracted + Car braking ahead)
                    if not eyes_on_road and lead_closing_fast:
                        target_risk = 95.0
                        context_label = "CRITICAL: IMMINENT HAZARD"
                    
                    # RULE 2: Just Distracted (Empty road ahead)
                    elif not eyes_on_road and not lead_closing_fast:
                        target_risk = 35.0  # Keep it at medium warning, not critical
                        context_label = "WARNING: DISTRACTED"
                    
                    # RULE 3: Speeding but Calm ("Driver in Control")
                    elif speed_kmh > design_speed and driver_is_calm and eyes_on_road:
                        target_risk = min(target_risk, 20.0) # Suppress risk score
                        context_label = "SPEEDING (CALM/CONTROLLED)"

                # Exponential smoothing for UI stability
                if driver.smoothed_fused_risk == 0.0:
                    driver.smoothed_fused_risk = target_fused_risk = target_risk
                else:
                    driver.smoothed_fused_risk = (0.25 * target_risk) + (0.75 * driver.smoothed_fused_risk)

                # Floor the visual risk to 0 if the math pushes it slightly below 0
                driver.smoothed_fused_risk = max(0.0, driver.smoothed_fused_risk)

                risk_index_val = round(float(driver.smoothed_fused_risk) / 100.0, 2)
                if risk_index_val < 0.25:
                    risk_level_val = "LOW"
                elif risk_index_val < 0.55:
                    risk_level_val = "MEDIUM"
                elif risk_index_val < 0.80:
                    risk_level_val = "HIGH"
                else:
                    risk_level_val = "CRITICAL"

                t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                score_color = "\033[91m" if driver.smoothed_fused_risk >= 60.0 else ("\033[93m" if driver.smoothed_fused_risk >= 25.0 else "\033[92m")
                reset = "\033[0m"
                
                bump_tag = "[BUMP AHEAD]" if speed_bump_ahead else ""
                
                # Format RHR strictly as an integer for cleaner output
                rhr_display = f"{resting_hr:.0f}" if resting_hr else "---"
                
                print(
                    f"[{t_now}] {road_name[:14]:<14} {bump_tag:<13} | "
                    f"Speed:{speed_kmh:4.0f} | HR:{hr:3.0f} (RHR:{rhr_display}) | "
                    f"Risk:{score_color}{driver.smoothed_fused_risk:5.1f}% [{risk_level_val}]{reset} | "
                    f"Status: {context_label}"
                )

                # Forward to FastAPI Dashboard
                async_forward_to_dashboard({
                    "timestamp": current_time,
                    "stream_id": sid,
                    "source": "ml_runner",
                    "speed": float(speed_kmh) if obd_active else 0.0,
                    "vehicle_speed": float(speed_mps) if obd_active else 0.0,
                    "rpm": float(rpm) if obd_active else 0.0,
                    "throttle": float(throttle) if obd_active else 0.0,
                    "engine_load": float(load) if obd_active else 0.0,
                    "gps_latitude": float(lat) if lat != 0.0 else None,
                    "gps_longitude": float(lon) if lon != 0.0 else None,
                    "heart_rate": float(hr) if whoop_active else None,
                    "accel_x": float(driver.smoothed_accel),
                    "gyro_z": float(gyro_mag),
                    "risk_index": risk_index_val,
                    "risk_level": risk_level_val,
                    "risk_score": float(driver.smoothed_fused_risk),
                    "fused_risk": float(driver.smoothed_fused_risk),
                    "class_label": context_label,
                    "obd_connected": bool(obd_active),
                    "wearable_connected": bool(whoop_active),
                })

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped live multi-stream inference.")

if __name__ == "__main__":
    main()