"""Real-time multimodal terminal runner with weather risk scaling, proactive prediction, turn suppression, and sub-second latency."""

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
from accident_risk.processing.weather_context import WeatherContextResolver

# =========================================================
# PATH & MODEL CONFIGURATION
# =========================================================
FUSION_MODEL_PATH = "artifacts/fusion_model/fusion_model.joblib"
FUSION_SCALER_PATH = "artifacts/fusion_model/scaler.joblib"
BENCHMARK_MODEL_PATH = "artifacts/benchmark_models/vzcrash_xgb.joblib"
ANOMALY_MODEL_PATH = "artifacts/benchmark_models/unsupervised_iso.joblib"
BENCHMARK_SCALER_PATH = "artifacts/benchmark_models/benchmark_scaler.joblib"

# Set your OpenWeatherMap API Key here or pass via OPENWEATHER_API_KEY environment variable
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "YOUR_OPENWEATHER_API_KEY_HERE")

CLASS_MAP = {0: "CRASH (Class 0)", 1: "NEAR-MISS (Class 1)", 2: "NORMAL (Class 2)"}
SERVE_INGEST_URL = "http://127.0.0.1:8000/ingest"
HARDWARE_TIMEOUT_SEC = 5.0

state_lock = threading.Lock()

live_state = {
    "lat": 0.0,
    "lon": 0.0,
    "accel_mag": 9.8,
    "gyro_mag": 0.0,
    "last_android_time": 0.0,
    "instant_hr": 72.0,
    "mean_hr_30s": 72.0,
    "rmssd_30s": 45.0,
    "sdnn_30s": 40.0,
    "pnn50_30s": 15.0,
    "last_whoop_time": 0.0,
    "speed": 0.0,
    "rpm": 0.0,
    "throttle": 0.0,
    "load": 0.0,
    "accel_obd": 0.0,
    "last_obd_time": 0.0,
}

history_buffer = deque(maxlen=30)


def async_forward_to_dashboard(payload):
    def send():
        try:
            req = urllib.request.Request(
                SERVE_INGEST_URL,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=0.2):
                pass
        except Exception:
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

                    live_state["last_android_time"] = now

                    lat = frame.get("lat") or frame.get("latitude") or frame.get("gps_latitude")
                    lon = frame.get("lon") or frame.get("longitude") or frame.get("gps_longitude")
                    if lat is not None and lon is not None:
                        live_state["lat"] = float(lat)
                        live_state["lon"] = float(lon)

                    ax = frame.get("accel_x", 0.0)
                    ay = frame.get("accel_y", 0.0)
                    az = frame.get("accel_z", 0.0)
                    if ax or ay or az:
                        live_state["accel_mag"] = math.sqrt(float(ax) ** 2 + float(ay) ** 2 + float(az) ** 2)
                    elif "accel_mag" in frame:
                        live_state["accel_mag"] = float(frame["accel_mag"])

                    if "gyro_mag" in frame:
                        live_state["gyro_mag"] = float(frame["gyro_mag"])
                    elif "gyro_z" in frame:
                        live_state["gyro_mag"] = float(frame["gyro_z"])

                    is_obd_connected = frame.get("obd_connected", False)
                    speed_val = frame.get("speed") or frame.get("vehicle_speed") or 0.0
                    rpm_val = frame.get("rpm") or frame.get("engine_speed") or 0.0
                    throttle_val = frame.get("throttle") or frame.get("throttle_position") or 0.0
                    load_val = frame.get("load") or frame.get("engine_load") or 0.0

                    if is_obd_connected or speed_val > 0 or rpm_val > 0:
                        live_state["speed"] = float(speed_val)
                        live_state["rpm"] = float(rpm_val)
                        live_state["throttle"] = float(throttle_val)
                        live_state["load"] = float(load_val)
                        live_state["last_obd_time"] = now

                    hr_val = frame.get("bpm") or frame.get("heart_rate") or frame.get("hr")
                    if hr_val is not None:
                        hr = float(hr_val)
                        live_state["instant_hr"] = hr
                        live_state["mean_hr_30s"] = hr
                        live_state["last_whoop_time"] = now

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
    hr_velocity = (buf[-1]["hr"] - buf[0]["hr"]) / dt
    return jerk, hr_velocity


def main():
    if not os.path.exists(BENCHMARK_MODEL_PATH) or not os.path.exists(FUSION_MODEL_PATH):
        print("❌ Model files missing! Ensure model training artifacts are present.")
        sys.exit(1)

    print("🧠 Loading Model Artifacts (POLIDriving Fusion, VZCrash Benchmark, & Anomaly Detector)...")
    fusion_model = joblib.load(FUSION_MODEL_PATH)
    fusion_scaler = joblib.load(FUSION_SCALER_PATH)

    benchmark_model = joblib.load(BENCHMARK_MODEL_PATH)
    anomaly_model = joblib.load(ANOMALY_MODEL_PATH)
    benchmark_scaler = joblib.load(BENCHMARK_SCALER_PATH)

    map_resolver = MapContextResolver()
    weather_resolver = WeatherContextResolver(api_key=OPENWEATHER_API_KEY, poll_interval_sec=300)

    print("📡 Listening for remote phone telemetry stream on port 8003...")
    start_receiver()

    fusion_cols = [
        "speed", "rpm", "throttle_position", "engine_load_value",
        "acceleration", "mean_hr_30s", "rmssd_30s", "sdnn_30s", "pnn50_30s"
    ]
    prev_speed_mps, prev_time = None, None
    smoothed_accel = 9.8

    csv_dir = "data/recordings"
    os.makedirs(csv_dir, exist_ok=True)
    session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(csv_dir, f"realtime_telemetry_{session_timestamp}.csv")

    csv_file = open(csv_path, mode="w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow([
        "timestamp", "time_str", "latitude", "longitude", "speed_kmh", "rpm",
        "throttle", "engine_load", "accel_obd", "accel_mag", "gyro_mag",
        "jerk", "hr_velocity", "heart_rate", "weather_cond", "weather_mult",
        "in_roundabout", "polidriving_risk", "vzcrash_risk", "fused_risk_score",
        "class_pred", "class_label", "is_anomaly"
    ])
    csv_file.flush()
    print(f"📁 CSV Logging initialized: {csv_path}")

    print("\n" + "=" * 110)
    print(" 🚀 PROACTIVE MULTIMODAL INFERENCE ENGINE + LIVE WEATHER SCALING ONLINE")
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
                mean_hr = live_state["mean_hr_30s"]
                rmssd = live_state["rmssd_30s"]
                sdnn = live_state["sdnn_30s"]
                pnn50 = live_state["pnn50_30s"]

                android_active = (current_time - live_state["last_android_time"]) <= HARDWARE_TIMEOUT_SEC
                obd_active = (current_time - live_state["last_obd_time"]) <= HARDWARE_TIMEOUT_SEC

            if not android_active:
                status_android = "🔴 WAITING"
                status_obd = "🟢 OK" if obd_active else "🔴 WAITING"
                print(f"[{t_now}] ⚠️ WAITING FOR REMOTE STREAM | GPS/Sensors: {status_android} | OBD-II: {status_obd}")
                time.sleep(0.5)
                continue

            # Update Weather Context (Triggers non-blocking fetch if cache expired)
            weather_resolver.update_location(lat, lon)
            weather_cond, weather_mult, weather_desc = weather_resolver.get_context()

            smoothed_accel = (0.4 * raw_accel_mag) + (0.6 * smoothed_accel)

            speed_mps = speed_kmh * (1000.0 / 3600.0)
            accel_obd = 0.0
            if prev_speed_mps is not None and prev_time is not None:
                dt = current_time - prev_time
                if dt > 0:
                    accel_obd = (speed_mps - prev_speed_mps) / dt
            prev_speed_mps = speed_mps
            prev_time = current_time

            history_buffer.append({"time": current_time, "accel": smoothed_accel, "hr": hr, "speed": speed_kmh})
            jerk, hr_velocity = compute_predictive_metrics(history_buffer)

            in_roundabout = map_resolver.is_in_roundabout(lat, lon)

            # --- 1. POLIDRIVING REGRESSION MODEL ---
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

            polidriving_risk = float(fusion_model.predict_proba(fusion_scaler.transform(fusion_df))[0][1])

            # --- 2. VZCRASH CLASSIFICATION MODEL ---
            effective_accel = smoothed_accel if smoothed_accel > 0 else abs(accel_obd)
            benchmark_sample = np.array([[effective_accel, gyro_mag, speed_kmh, abs(accel_obd)]])
            benchmark_scaled = benchmark_scaler.transform(benchmark_sample)

            vz_probs = benchmark_model.predict_proba(benchmark_scaled)[0]
            class_pred = int(np.argmax(vz_probs))
            p_crash, p_near_miss = vz_probs[0], vz_probs[1]
            vzcrash_risk = float(1.0 * p_crash + 0.6 * p_near_miss)

            base_fused_risk = ((0.5 * polidriving_risk) + (0.5 * vzcrash_risk)) * 100.0

            # --- 3. PROACTIVE HAZARD BOOST & WEATHER SCALING ---
            predictive_boost = 0.0
            if jerk > 12.0:
                predictive_boost += min(25.0, jerk * 1.5)
            if hr_velocity > 3.0:
                predictive_boost += min(15.0, hr_velocity * 2.5)

            # Apply Weather Multiplier to final risk score
            fused_risk_score = min(99.9, (base_fused_risk + predictive_boost) * weather_mult)

            # --- 4. SWERVING / SHARP TURN & ROUNDABOUT SUPPRESSION ---
            is_turning_or_swerving = (gyro_mag > 0.8 and effective_accel < 24.0)

            if in_roundabout or is_turning_or_swerving:
                if class_pred in (0, 1) and effective_accel < 24.0:
                    class_pred = 2
                    reason = "Roundabout Curve" if in_roundabout else "Sharp Turn / Swerve"
                    class_label = f"NORMAL ({reason})"
                    fused_risk_score = min(fused_risk_score, 18.0 * weather_mult)
                else:
                    class_label = CLASS_MAP.get(class_pred, "UNKNOWN")
            else:
                class_label = CLASS_MAP.get(class_pred, "UNKNOWN")

            is_anomaly = anomaly_model.predict(benchmark_scaled)[0] == -1
            if (in_roundabout or is_turning_or_swerving) and is_anomaly and effective_accel < 24.0:
                is_anomaly = False

            # --- 5. TERMINAL LOG DISPLAY ---
            score_color = "\033[91m" if fused_risk_score >= 60.0 else ("\033[93m" if fused_risk_score >= 30.0 else "\033[92m")
            anomaly_color = "\033[91m" if is_anomaly else "\033[92m"
            reset = "\033[0m"

            rb_tag = ""
            if in_roundabout:
                rb_tag = " 🔄 [ROUNDABOUT]"
            elif is_turning_or_swerving:
                rb_tag = " ↩️ [SWERVE/TURN]"
            elif predictive_boost > 10.0:
                rb_tag = " ⚠️ [PREDICTIVE WARNING]"

            obd_tag = "🟢 OBD" if obd_active else "⚪ NO OBD"
            weather_tag = f" 🌧️ [{weather_cond} {weather_mult}x]" if weather_mult > 1.0 else ""

            print(
                f"[{t_now}] 📍 Lat:{lat:.6f}, Lon:{lon:.6f}{rb_tag}{weather_tag} | "
                f"🚗 Speed:{speed_kmh:4.1f}km/h ({obd_tag}) | RPM:{rpm:4.0f} | "
                f"🫀 HR:{hr:3.0f}bpm | "
                f"🎯 Risk:{score_color}{fused_risk_score:5.1f}%{reset} | "
                f"🏆 VZCrash:{class_label} | "
                f"🔍 Anomaly:{anomaly_color}{'YES' if is_anomaly else 'NO'}{reset}"
            )

            # --- 6. LOG FRAME TO CSV ---
            csv_writer.writerow([
                round(current_time, 3), t_now, round(lat, 6), round(lon, 6),
                round(speed_kmh, 2), round(rpm, 1), round(throttle, 2), round(load, 2),
                round(accel_obd, 3), round(raw_accel_mag, 3), round(gyro_mag, 3),
                round(jerk, 3), round(hr_velocity, 3), int(hr), weather_cond,
                weather_mult, bool(in_roundabout), round(polidriving_risk, 4),
                round(vzcrash_risk, 4), round(fused_risk_score, 2), class_pred,
                class_label, bool(is_anomaly)
            ])
            csv_file.flush()

            # --- 7. ASYNCHRONOUS DASHBOARD FORWARDING ---
            async_forward_to_dashboard({
                "timestamp": current_time,
                "stream_id": "hardware",
                "source": "advanced_terminal",
                "vehicle_speed": float(speed_mps),
                "rpm": float(rpm) if rpm > 0 else None,
                "throttle_position": float(throttle) if throttle > 0 else None,
                "engine_load": float(load) if load > 0 else None,
                "gps_latitude": float(lat) if lat != 0.0 else None,
                "gps_longitude": float(lon) if lon != 0.0 else None,
                "heart_rate": float(hr) if hr > 0 else None,
                "accel_x": float(smoothed_accel),
                "gyro_z": float(gyro_mag),
                "weather_condition": weather_cond,
                "weather_multiplier": weather_mult,
            })

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\n🛑 Stopped live inference stream.")
    finally:
        if "csv_file" in locals() and not csv_file.closed:
            csv_file.close()
            print(f"💾 Telemetry log saved: {csv_path}")


if __name__ == "__main__":
    main()