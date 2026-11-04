"""Real-time multimodal terminal runner with per-stream states for multiple users."""
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
        self.hr_history = deque(maxlen=30)


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

                    speed = frame.get("speed") or frame.get("vehicle_speed") or 0.0
                    rpm = frame.get("rpm") or frame.get("engine_speed") or 0.0
                    throttle = frame.get("throttle") or frame.get("throttle_position") or 0.0
                    load = frame.get("load") or frame.get("engine_load") or 0.0

                    driver.speed = float(speed)
                    driver.rpm = float(rpm)
                    driver.throttle = float(throttle)
                    driver.load = float(load)

                    if obd_conn or driver.speed > 0 or driver.rpm > 0:
                        driver.last_obd_time = now

                    hr_raw = (
                        frame.get("heart_rate")
                        or frame.get("bpm")
                        or frame.get("hr")
                        or 0.0
                    )
                    hr_val = float(hr_raw)
                    if hr_val > 0:
                        driver.instant_hr = hr_val
                        driver.whoop_connected = True
                        driver.last_whoop_time = now
                        driver.hr_history.append(hr_val)
                    else:
                        if (now - driver.last_whoop_time) > HARDWARE_TIMEOUT_SEC:
                            driver.whoop_connected = False

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


def compute_predictive_metrics(buf):
    if len(buf) < 4:
        return 0.0, 0.0
    t_now = buf[-1]["time"]
    t_past = buf[0]["time"]
    dt = t_now - t_past
    if dt <= 0.05:
        return 0.0, 0.0
    jerk = (buf[-1]["accel"] - buf[0]["accel"]) / dt
    hr_velocity = 0.0
    if buf[-1]["hr"] > 0 and buf[0]["hr"] > 0:
        hr_velocity = (buf[-1]["hr"] - buf[0]["hr"]) / dt
    return jerk, hr_velocity


def calculate_gps_bearing_delta(buf):
    if len(buf) < 3:
        return 0.0
    lat1, lon1 = buf[0]["lat"], buf[0]["lon"]
    lat2, lon2 = buf[-1]["lat"], buf[-1]["lon"]
    if lat1 == 0.0 or lat2 == 0.0 or (lat1 == lat2 and lon1 == lon2):
        return 0.0
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(math.radians(lat2))
    x = (
        math.cos(math.radians(lat1)) * math.sin(math.radians(lat2))
        - math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)
    )
    return abs(math.degrees(math.atan2(y, x)))


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

    csv_dir = "data/recordings"
    os.makedirs(csv_dir, exist_ok=True)
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"multi_telemetry_{session_timestamp}.csv")
    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "time_str", "stream_id", "latitude", "longitude", "speed_kmh", "rpm",
        "throttle", "engine_load", "accel_obd", "accel_mag", "gyro_mag",
        "jerk", "hr_velocity", "heart_rate", "in_roundabout", "near_intersection",
        "polidriving_risk", "vzcrash_risk", "fused_risk_score", "class_pred",
        "class_label", "is_anomaly",
    ])
    csv_file.flush()

    print("\n" + "=" * 105)
    print("MULTI-USER CONCURRENT RISK INFERENCE ENGINE ONLINE (UAE LOCAL CONTEXT)")
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
                t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                print(f"[{t_now}] WAITING FOR ACTIVE STREAMS ON PORT 8003...")
                time.sleep(0.5)
                continue

            for sid in active_streams:
                with state_lock:
                    driver = drivers[sid]
                    lat = driver.lat
                    lon = driver.lon
                    speed_kmh = driver.speed
                    rpm = driver.rpm
                    throttle = driver.throttle
                    load = driver.load
                    raw_accel_mag = driver.accel_mag
                    gyro_mag = driver.gyro_mag
                    hr = driver.instant_hr
                    obd_connected = driver.obd_connected
                    whoop_connected = driver.whoop_connected

                    obd_active = obd_connected or (
                        (current_time - driver.last_obd_time) <= HARDWARE_TIMEOUT_SEC
                        and (speed_kmh > 0 or rpm > 0)
                    )
                    whoop_active = whoop_connected and hr > 0

                map_resolver.update_location(lat, lon)
                in_roundabout = map_resolver.is_in_roundabout()
                near_intersection = map_resolver.is_near_intersection()
                design_speed = map_resolver.get_design_speed()
                road_name = map_resolver.get_road_name()

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

                driver.history_buffer.append({
                    "time": current_time,
                    "accel": driver.smoothed_accel,
                    "hr": hr,
                    "speed": speed_kmh,
                    "lat": lat,
                    "lon": lon,
                })

                jerk, hr_velocity = compute_predictive_metrics(driver.history_buffer)
                bearing_delta = calculate_gps_bearing_delta(driver.history_buffer)

                # --- 1. AUTHENTIC 14-FEATURE POLIDRIVING MODEL ---
                if obd_active or whoop_active:
                    obs_hour = float(datetime.datetime.now().hour)
                    accidents_onsite = 8.0 if near_intersection else 2.0
                    accidents_time = 1.0
                    fusion_df = pd.DataFrame([{
                        "observation_hour": obs_hour,
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
                        "accidents_onsite": accidents_onsite,
                        "design_speed": float(design_speed),
                        "accidents_time": accidents_time,
                    }])[POLIDRIVING_FEATURES]
                    fusion_probs = fusion_model.predict_proba(fusion_scaler.transform(fusion_df))[0]
                    raw_polidriving_risk = float(np.sum(fusion_probs[1:]))
                    if speed_kmh <= design_speed and throttle < 45.0 and abs(accel_obd) < 2.0:
                        polidriving_risk = min(raw_polidriving_risk, 0.20)
                    else:
                        polidriving_risk = raw_polidriving_risk
                else:
                    polidriving_risk = 0.0

                # --- 2. VZCRASH CLASSIFICATION MODEL ---
                effective_accel = driver.smoothed_accel if driver.smoothed_accel > 0 else abs(accel_obd)
                benchmark_df = pd.DataFrame([{
                    "effective_accel": effective_accel,
                    "gyro_mag": gyro_mag,
                    "speed_kmh": speed_kmh,
                    "abs_accel_obd": abs(accel_obd),
                }])
                benchmark_scaled = benchmark_scaler.transform(benchmark_df)
                vz_probs = benchmark_model.predict_proba(benchmark_scaled)[0]
                class_pred = int(np.argmax(vz_probs))
                p_crash, p_near_miss = vz_probs[0], vz_probs[1]
                vzcrash_risk = float(1.0 * p_crash + 0.6 * p_near_miss)

                # --- 3. FUSED RISK COMPUTATION ---
                if not obd_active and not whoop_active and speed_kmh == 0.0 and effective_accel < 12.0:
                    target_fused_risk = 0.0
                    class_pred = 2
                    class_label = "NORMAL (Parked / Sensor Standby)"
                else:
                    target_fused_risk = ((0.5 * polidriving_risk) + (0.5 * vzcrash_risk)) * 100.0
                    predictive_boost = 0.0
                    if jerk > 14.0:
                        predictive_boost += min(25.0, jerk * 1.5)
                    if hr_velocity > 4.0 and speed_kmh > 40.0:
                        predictive_boost += min(15.0, hr_velocity * 2.5)
                    target_fused_risk = min(99.9, target_fused_risk + predictive_boost)

                    is_turn = (gyro_mag > 0.35 and bearing_delta > 10.0) or (bearing_delta > 20.0)
                    is_slowdown = (speed_kmh < 40.0 and effective_accel < 20.0 and not is_turn)

                    if in_roundabout or is_turn or near_intersection or is_slowdown:
                        if class_pred in (0, 1) and effective_accel < 22.0:
                            class_pred = 2
                            reason = "Roundabout" if in_roundabout else ("Turn" if is_turn else "Slowdown")
                            class_label = f"NORMAL ({reason})"
                            target_fused_risk = min(target_fused_risk, 15.0)
                        else:
                            class_label = CLASS_MAP.get(class_pred, "UNKNOWN")
                    else:
                        class_label = CLASS_MAP.get(class_pred, "UNKNOWN")

                    if speed_kmh <= design_speed and throttle < 40.0 and effective_accel < 16.0 and jerk < 8.0:
                        target_fused_risk = min(target_fused_risk, 15.0)

                # Exponential smoothing
                if driver.smoothed_fused_risk == 0.0:
                    driver.smoothed_fused_risk = target_fused_risk
                else:
                    driver.smoothed_fused_risk = (0.25 * target_fused_risk) + (0.75 * driver.smoothed_fused_risk)

                risk_index_val = round(float(driver.smoothed_fused_risk) / 100.0, 2)
                if risk_index_val < 0.25:
                    risk_level_val = "LOW"
                elif risk_index_val < 0.55:
                    risk_level_val = "MEDIUM"
                elif risk_index_val < 0.80:
                    risk_level_val = "HIGH"
                else:
                    risk_level_val = "CRITICAL"

                is_anomaly = anomaly_model.predict(benchmark_scaled)[0] == -1 if effective_accel >= 22.0 else False

                t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
                score_color = "\033[91m" if driver.smoothed_fused_risk >= 60.0 else ("\033[93m" if driver.smoothed_fused_risk >= 25.0 else "\033[92m")
                reset = "\033[0m"
                obd_status_str = f"{speed_kmh:.0f}km/h" if obd_active else "NO OBD"
                whoop_status_str = f"{hr:.0f}bpm" if whoop_active else "NO WHOOP"

                print(
                    f"[{t_now}] [{sid[:10]:<10}] {road_name[:14]:<14} | "
                    f"OBD:{obd_status_str} | RPM:{rpm:4.0f} | "
                    f"HR:{whoop_status_str} | "
                    f"Risk:{score_color}{driver.smoothed_fused_risk:5.1f}% [{risk_level_val}]{reset} | "
                    f"VZ:{class_label}"
                )

                csv_writer.writerow([
                    round(current_time, 3), t_now, sid, round(lat, 6), round(lon, 6),
                    round(speed_kmh, 2), round(rpm, 1), round(throttle, 2), round(load, 2),
                    round(accel_obd, 3), round(raw_accel_mag, 3), round(gyro_mag, 3),
                    round(jerk, 3), round(hr_velocity, 3), int(hr), bool(in_roundabout),
                    bool(near_intersection), round(polidriving_risk, 4), round(vzcrash_risk, 4),
                    round(driver.smoothed_fused_risk, 2), class_pred, class_label, bool(is_anomaly),
                ])
                csv_file.flush()

                # FORWARD ML OUTPUT DIRECTLY TO SERVE.PY DASHBOARD
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
                    "risk_level": risk_level_val if (obd_active or whoop_active or speed_kmh > 0) else None,
                    "risk_score": float(driver.smoothed_fused_risk),
                    "fused_risk": float(driver.smoothed_fused_risk),
                    "class_label": class_label,
                    "obd_connected": bool(obd_active),
                    "wearable_connected": bool(whoop_active),
                })

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nStopped live multi-stream inference.")
    finally:
        if "csv_file" in locals() and not csv_file.closed:
            csv_file.close()


if __name__ == "__main__":
    main()