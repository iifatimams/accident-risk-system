"""Real-time multimodal terminal runner with async map context, speed bump detection, turn suppression, and dashboard forwarding."""

import asyncio
from collections import deque
import csv
import datetime
import json
import math
import os
import sys
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer

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
HARDWARE_TIMEOUT_SEC = 5.0

state_lock = threading.Lock()

# Global state reflecting genuine phone data
live_state = {
    "lat": 0.0,
    "lon": 0.0,
    "accel_mag": 0.0,
    "gyro_mag": 0.0,
    "last_android_time": 0.0,
    "instant_hr": 0.0,
    "last_whoop_time": 0.0,
    "speed": 0.0,
    "rpm": 0.0,
    "throttle": 0.0,
    "load": 0.0,
    "accel_obd": 0.0,
    "last_obd_time": 0.0,
    "obd_connected": False,
    "whoop_connected": False,
}

history_buffer = deque(maxlen=30)
hr_history = deque(maxlen=30)
debug_print_counter = 0


# =========================================================
# 1. NON-BLOCKING DASHBOARD FORWARDING & INGEST HANDLER
# =========================================================
def async_forward_to_dashboard(payload):
    """Non-blocking forwarding with error printing."""
    def send():
        try:
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                SERVE_INGEST_URL,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=1.0) as resp:
                pass
        except Exception:
            pass

    threading.Thread(target=send, daemon=True).start()


class UnifiedIngestHandler(BaseHTTPRequestHandler):
    """Handler for remote Android TelemetryService POSTs."""

    def do_POST(self):
        global debug_print_counter
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        now = time.time()

        try:
            payload = json.loads(body.decode("utf-8"))
            frames = payload if isinstance(payload, list) else [payload]

            debug_print_counter += 1
            if debug_print_counter % 25 == 1 and frames:
                print(f"\n📱 RAW TELEMETRY RECEIVED FROM PHONE: {frames[0]}\n")

            with state_lock:
                for frame in frames:
                    if not isinstance(frame, dict):
                        continue

                    live_state["last_android_time"] = now

                    # Coordinates
                    lat = frame.get("lat") or frame.get("gps_latitude")
                    lon = frame.get("lon") or frame.get("gps_longitude")
                    if lat is not None and lon is not None:
                        live_state["lat"] = float(lat)
                        live_state["lon"] = float(lon)

                    # IMU Accelerometer & Gyroscope
                    if "accel_mag" in frame:
                        live_state["accel_mag"] = float(frame["accel_mag"])
                    if "gyro_mag" in frame:
                        live_state["gyro_mag"] = float(frame["gyro_mag"])

                    # Genuine OBD Telematics
                    obd_conn = bool(frame.get("obd_connected"))
                    live_state["obd_connected"] = obd_conn

                    speed = frame.get("speed") or frame.get("vehicle_speed") or 0.0
                    rpm = frame.get("rpm") or frame.get("engine_speed") or 0.0
                    throttle = frame.get("throttle") or 0.0
                    load = frame.get("load") or 0.0

                    live_state["speed"] = float(speed)
                    live_state["rpm"] = float(rpm)
                    live_state["throttle"] = float(throttle)
                    live_state["load"] = float(load)

                    if obd_conn or live_state["speed"] > 0 or live_state["rpm"] > 0:
                        live_state["last_obd_time"] = now

                    # Genuine WHOOP Heart Rate
                    hr_raw = frame.get("heart_rate") or frame.get("bpm") or frame.get("hr") or 0.0
                    hr_val = float(hr_raw)

                    if hr_val > 0:
                        live_state["instant_hr"] = hr_val
                        live_state["whoop_connected"] = True
                        live_state["last_whoop_time"] = now
                        hr_history.append(hr_val)
                    else:
                        live_state["instant_hr"] = 0.0
                        if (now - live_state["last_whoop_time"]) > HARDWARE_TIMEOUT_SEC:
                            live_state["whoop_connected"] = False

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status":"ok"}')
        except Exception:
            self.send_response(400)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def start_receiver():
    """Starts background HTTP server listening on port 8003."""
    server = HTTPServer(("0.0.0.0", 8003), UnifiedIngestHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()


# =========================================================
# 2. PROACTIVE PREDICTIVE & KINEMATIC FEATURE CALCULATORS
# =========================================================
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
    x = math.cos(math.radians(lat1)) * math.sin(math.radians(lat2)) - \
        math.sin(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.cos(dlon)

    return abs(math.degrees(math.atan2(y, x)))


# =========================================================
# 3. MAIN INFERENCE LOOP
# =========================================================
def main():
    if not os.path.exists(BENCHMARK_MODEL_PATH) or not os.path.exists(FUSION_MODEL_PATH):
        print("❌ Model files missing! Please run training scripts first.")
        sys.exit(1)

    print("🧠 Loading Model Artifacts (POLIDriving Fusion, VZCrash Benchmark, & Anomaly Detector)...")
    fusion_model = joblib.load(FUSION_MODEL_PATH)
    fusion_scaler = joblib.load(FUSION_SCALER_PATH)

    benchmark_model = joblib.load(BENCHMARK_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    benchmark_scaler = joblib.load(BENCHMARK_SCALER_PATH)

    map_resolver = MapContextResolver()

    print("📡 Listening for remote phone telemetry stream on port 8003...")
    start_receiver()

    fusion_cols = [
        "speed", "rpm", "throttle_position", "engine_load_value",
        "acceleration", "mean_hr_30s", "rmssd_30s", "sdnn_30s", "pnn50_30s"
    ]
    prev_speed_mps, prev_time = None, None
    smoothed_accel = 0.0
    smoothed_fused_risk = 0.0  # Smooth EMA tracker

    csv_dir = "data/recordings"
    os.makedirs(csv_dir, exist_ok=True)
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"realtime_telemetry_{session_timestamp}.csv")

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "time_str", "latitude", "longitude", "speed_kmh", "rpm",
        "throttle", "engine_load", "accel_obd", "accel_mag", "gyro_mag",
        "jerk", "hr_velocity", "heart_rate", "in_roundabout", "near_intersection",
        "polidriving_risk", "vzcrash_risk", "fused_risk_score", "class_pred",
        "class_label", "is_anomaly"
    ])
    csv_file.flush()

    print("\n" + "=" * 110)
    print(" 🚀 PROACTIVE MULTIMODAL ACCIDENT RISK INFERENCE ENGINE ONLINE")
    print("=" * 110 + "\n")

    try:
        while True:
            t_now = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
            current_time = time.time()

            with state_lock:
                lat = live_state["lat"]
                lon = live_state["lon"]
                speed_kmh = live_state["speed"]
                rpm = live_state["rpm"]
                throttle = live_state["throttle"]
                load = live_state["load"]
                raw_accel_mag = live_state["accel_mag"]
                gyro_mag = live_state["gyro_mag"]
                hr = live_state["instant_hr"]
                obd_connected = live_state["obd_connected"]
                whoop_connected = live_state["whoop_connected"]

                android_active = (current_time - live_state["last_android_time"]) <= HARDWARE_TIMEOUT_SEC
                obd_active = obd_connected or ((current_time - live_state["last_obd_time"]) <= HARDWARE_TIMEOUT_SEC and (speed_kmh > 0 or rpm > 0))
                whoop_active = whoop_connected and hr > 0

            if not android_active:
                print(f"[{t_now}] ⚠️ WAITING FOR PHONE STREAM (PORT 8003)...")
                time.sleep(0.5)
                continue

            try:
                map_resolver.update_location(lat, lon)
                in_roundabout = map_resolver.is_in_roundabout()
                near_intersection = map_resolver.is_near_intersection()
            except Exception:
                in_roundabout = False
                near_intersection = False

            smoothed_accel = (0.3 * raw_accel_mag) + (0.7 * smoothed_accel) if smoothed_accel > 0 else raw_accel_mag

            speed_mps = speed_kmh * (1000.0 / 3600.0)
            accel_obd = 0.0
            if prev_speed_mps is not None and prev_time is not None:
                dt = current_time - prev_time
                if dt > 0:
                    accel_obd = (speed_mps - prev_speed_mps) / dt
            prev_speed_mps = speed_mps
            prev_time = current_time

            history_buffer.append({
                "time": current_time,
                "accel": smoothed_accel,
                "hr": hr,
                "speed": speed_kmh,
                "lat": lat,
                "lon": lon
            })
            jerk, hr_velocity = compute_predictive_metrics(history_buffer)
            bearing_delta = calculate_gps_bearing_delta(history_buffer)

            # Compute statistical window features
            mean_hr = float(np.mean(hr_history)) if len(hr_history) > 0 else 0.0
            rmssd = float(np.sqrt(np.mean(np.diff(hr_history) ** 2))) if len(hr_history) > 1 else 0.0
            sdnn = float(np.std(hr_history)) if len(hr_history) > 0 else 0.0
            pnn50 = float(np.sum(np.abs(np.diff(hr_history)) > 5) / len(np.diff(hr_history)) * 100.0) if len(hr_history) > 1 else 0.0

            # --- 1. POLIDRIVING REGRESSION MODEL WITH COGNITIVE/BEHAVIORAL GATING ---
            if obd_active or whoop_active:
                fusion_df = pd.DataFrame([{
                    "speed": round(speed_kmh, 2),
                    "rpm": round(rpm, 2),
                    "throttle_position": round(throttle, 2),
                    "engine_load_value": round(load, 2),
                    "acceleration": round(accel_obd, 3),
                    "mean_hr_30s": mean_hr,
                    "rmssd_30s": rmssd,
                    "sdnn_30s": sdnn,
                    "pnn50_30s": pnn50,
                }])[fusion_cols]
                raw_polidriving_risk = float(fusion_model.predict_proba(fusion_scaler.transform(fusion_df))[0][1])

                # Heart rate context gating: if driving calmly under 60 km/h with low throttle,
                # do not allow elevated pulse alone to pin the risk score at 1.0.
                if speed_kmh < 60.0 and throttle < 45.0 and abs(accel_obd) < 2.0:
                    polidriving_risk = min(raw_polidriving_risk, 0.20)
                else:
                    polidriving_risk = raw_polidriving_risk
            else:
                polidriving_risk = 0.0

            # --- 2. VZCRASH CLASSIFICATION MODEL ---
            effective_accel = smoothed_accel if smoothed_accel > 0 else abs(accel_obd)
            benchmark_df = pd.DataFrame([{
                "effective_accel": effective_accel,
                "gyro_mag": gyro_mag,
                "speed_kmh": speed_kmh,
                "abs_accel_obd": abs(accel_obd)
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
                class_label = "NORMAL (Parked / Waiting for Sensors)"
            else:
                target_fused_risk = ((0.5 * polidriving_risk) + (0.5 * vzcrash_risk)) * 100.0

                # Predictive boost only on genuine physical spikes
                predictive_boost = 0.0
                if jerk > 14.0:
                    predictive_boost += min(25.0, jerk * 1.5)
                if hr_velocity > 4.0 and speed_kmh > 40.0:
                    predictive_boost += min(15.0, hr_velocity * 2.5)
                target_fused_risk = min(99.9, target_fused_risk + predictive_boost)

                # Road context / turn suppression: require BOTH gyro and bearing to confirm a turn
                is_turning_or_swerving = (gyro_mag > 0.35 and bearing_delta > 10.0) or (bearing_delta > 20.0)
                is_linear_slowdown = (speed_kmh < 40.0 and effective_accel < 20.0 and not is_turning_or_swerving)

                if in_roundabout or is_turning_or_swerving or near_intersection or is_linear_slowdown:
                    if class_pred in (0, 1) and effective_accel < 22.0:
                        class_pred = 2
                        reason = "Roundabout" if in_roundabout else ("Turn" if is_turning_or_swerving else "Slowdown")
                        class_label = f"NORMAL ({reason})"
                        target_fused_risk = min(target_fused_risk, 15.0)
                    else:
                        class_label = CLASS_MAP.get(class_pred, "UNKNOWN")
                else:
                    class_label = CLASS_MAP.get(class_pred, "UNKNOWN")

                # Baseline safety clamp: smooth driving under 45 km/h stays low
                if speed_kmh < 45.0 and throttle < 40.0 and effective_accel < 16.0 and jerk < 8.0:
                    target_fused_risk = min(target_fused_risk, 15.0)

            # --- 4. EXPONENTIAL SMOOTHING FILTER (NO MORE DISRUPTIVE JUMPS) ---
            if smoothed_fused_risk == 0.0:
                smoothed_fused_risk = target_fused_risk
            else:
                smoothed_fused_risk = (0.25 * target_fused_risk) + (0.75 * smoothed_fused_risk)

            # Map smoothed value to standard bands
            risk_index_val = round(float(smoothed_fused_risk) / 100.0, 2)
            if risk_index_val < 0.25:
                risk_level_val = "LOW"
            elif risk_index_val < 0.55:
                risk_level_val = "MEDIUM"
            elif risk_index_val < 0.80:
                risk_level_val = "HIGH"
            else:
                risk_level_val = "CRITICAL"

            is_anomaly = anomaly_model.predict(benchmark_scaled)[0] == -1 if effective_accel >= 22.0 else False

            # --- 5. TERMINAL DISPLAY ---
            score_color = "\033[91m" if smoothed_fused_risk >= 60.0 else ("\033[93m" if smoothed_fused_risk >= 25.0 else "\033[92m")
            reset = "\033[0m"

            obd_status_str = f"🟢 {speed_kmh:.0f}km/h" if obd_active else "🔴 NO OBD"
            whoop_status_str = f"🟢 {hr:.0f}bpm" if whoop_active else "🔴 NO WHOOP"

            print(
                f"[{t_now}] 📍 Lat:{lat:.6f}, Lon:{lon:.6f} | "
                f"🚗 OBD:{obd_status_str} | RPM:{rpm:4.0f} | "
                f"🫀 Wearable:{whoop_status_str} | "
                f"🎯 Risk:{score_color}{smoothed_fused_risk:5.1f}% [{risk_level_val}]{reset} | "
                f"🏆 VZCrash:{class_label}"
            )

            # --- 6. LOG TO CSV ---
            csv_writer.writerow([
                round(current_time, 3), t_now, round(lat, 6), round(lon, 6),
                round(speed_kmh, 2), round(rpm, 1), round(throttle, 2), round(load, 2),
                round(accel_obd, 3), round(raw_accel_mag, 3), round(gyro_mag, 3),
                round(jerk, 3), round(hr_velocity, 3), int(hr), bool(in_roundabout),
                bool(near_intersection), round(polidriving_risk, 4), round(vzcrash_risk, 4),
                round(smoothed_fused_risk, 2), class_pred, class_label, bool(is_anomaly)
            ])
            csv_file.flush()

            # --- 7. ASYNC DASHBOARD FORWARDING ---
            async_forward_to_dashboard({
                "timestamp": current_time,
                "stream_id": "hardware",
                "source": "hardware",
                "speed": float(speed_kmh) if obd_active else 0.0,
                "vehicle_speed": float(speed_mps) if obd_active else 0.0,
                "rpm": float(rpm) if obd_active else 0.0,
                "throttle": float(throttle) if obd_active else 0.0,
                "engine_load": float(load) if obd_active else 0.0,
                "gps_latitude": float(lat) if lat != 0.0 else None,
                "gps_longitude": float(lon) if lon != 0.0 else None,
                "heart_rate": float(hr) if whoop_active else None,
                "accel_x": float(smoothed_accel),
                "gyro_z": float(gyro_mag),
                "risk_index": risk_index_val,
                "risk_level": risk_level_val if (obd_active or whoop_active or speed_kmh > 0) else None,
                "risk_score": float(smoothed_fused_risk),
                "fused_risk": float(smoothed_fused_risk),
                "class_label": class_label,
                "obd_connected": bool(obd_active),
                "wearable_connected": bool(whoop_active),
            })

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped live inference stream.")
    finally:
        if "csv_file" in locals() and not csv_file.closed:
            csv_file.close()


if __name__ == "__main__":
    main()