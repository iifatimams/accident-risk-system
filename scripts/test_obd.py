import csv
import datetime
import os
import time
import obd

# Configuration
PORT = "/dev/cu.OBDII"

# Target raw directory (creates 'data/raw' relative to script location)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data", "raw")

# Ensure raw data folder exists
os.makedirs(RAW_DIR, exist_ok=True)

# Path to dataset file in raw folder
CSV_FILE = os.path.join(RAW_DIR, "obd_entries.csv")

# Dataset column headers matching your risk model
FIELDNAMES = [
    "timestamp",
    "speed",
    "rpm",
    "throttle_position",
    "engine_load_value",
    "engine_temperature",
    "acceleration"
]

print("Connecting to OBD-II adapter at:", PORT)

connection = obd.OBD(
    portstr=PORT,
    baudrate=38400,
    fast=False,
    timeout=2.0
)

print("Connection status:", connection.status())

if connection.is_connected():
    print(f"ECU Connected! Logging raw telemetry to '{CSV_FILE}'...\n")
    
    # Check if CSV file already exists in raw directory
    file_exists = os.path.isfile(CSV_FILE)
    
    # Variables for tracking acceleration
    prev_speed_mps = None
    prev_time = None

    try:
        # Open CSV file in append mode ('a')
        with open(CSV_FILE, mode='a', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
            
            # Write header row if creating a new file
            if not file_exists:
                writer.writeheader()
                csv_file.flush()

            while True:
                # 1. Timestamp
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                current_time = time.time()

                # 2. Query OBD Sensors
                speed_resp = connection.query(obd.commands.SPEED)
                rpm_resp = connection.query(obd.commands.RPM)
                throttle_resp = connection.query(obd.commands.THROTTLE_POS)
                load_resp = connection.query(obd.commands.ENGINE_LOAD)
                temp_resp = connection.query(obd.commands.COOLANT_TEMP)

                # Extract magnitudes safely
                speed_kmh = speed_resp.value.magnitude if not speed_resp.is_null() else 0.0
                rpm = rpm_resp.value.magnitude if not rpm_resp.is_null() else 0.0
                throttle_position = throttle_resp.value.magnitude if not throttle_resp.is_null() else 0.0
                engine_load_value = load_resp.value.magnitude if not load_resp.is_null() else 0.0
                engine_temperature = temp_resp.value.magnitude if not temp_resp.is_null() else 0.0

                # 3. Calculate Acceleration (m/s^2)
                speed_mps = speed_kmh * (1000.0 / 3600.0)
                acceleration = 0.0

                if prev_speed_mps is not None and prev_time is not None:
                    dt = current_time - prev_time
                    if dt > 0:
                        acceleration = (speed_mps - prev_speed_mps) / dt

                prev_speed_mps = speed_mps
                prev_time = current_time

                # 4. Build row dictionary
                row_data = {
                    "timestamp": timestamp,
                    "speed": round(speed_kmh, 2),
                    "rpm": round(rpm, 2),
                    "throttle_position": round(throttle_position, 2),
                    "engine_load_value": round(engine_load_value, 2),
                    "engine_temperature": round(engine_temperature, 2),
                    "acceleration": round(acceleration, 3)
                }

                # 5. Write to CSV and flush immediately to disk
                writer.writerow(row_data)
                csv_file.flush()

                # 6. Terminal display
                print(
                    f"[SAVED -> RAW] {timestamp} | "
                    f"Speed: {speed_kmh:.1f} km/h | "
                    f"RPM: {rpm:.0f} | "
                    f"Throttle: {throttle_position:.1f}% | "
                    f"Load: {engine_load_value:.1f}% | "
                    f"Temp: {engine_temperature:.0f}°C | "
                    f"Accel: {acceleration:.2f} m/s²"
                )

                time.sleep(0.2)

    except KeyboardInterrupt:
        print(f"\nStopped collection. Data saved cleanly to {CSV_FILE}.")
    finally:
        connection.close()
        print("Serial connection closed.")
else:
    print("Failed to connect to car ECU.")