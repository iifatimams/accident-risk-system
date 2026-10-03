import asyncio
import csv
import os
import time
from datetime import datetime
from bleak import BleakClient, BleakScanner
import numpy as np

CSV_FILE_PATH = "data/raw/whoop_ble_30s_vitals.csv"
HEART_RATE_MEASUREMENT_UUID = "00002a37-0000-1000-8000-00805f9b34fb"

# 30-Second Data Buffers
bpm_buffer = []
rr_buffer = []


def initialize_csv():
    """Ensure destination directory and CSV headers exist."""
    os.makedirs(os.path.dirname(CSV_FILE_PATH), exist_ok=True)
    if not os.path.exists(CSV_FILE_PATH):
        with open(CSV_FILE_PATH, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "timestamp",
                    "datetime",
                    "mean_hr_30s",
                    "std_hr_30s",
                    "min_hr_30s",
                    "max_hr_30s",
                    "range_hr_30s",
                    "median_hr_30s",
                    "mean_rr_30s",
                    "sdnn_30s",
                    "rmssd_30s",
                    "pnn50_30s",
                    "pnn20_30s",
                ]
            )
        print(f"📁 Initialized maximal feature dataset at: {CSV_FILE_PATH}")


def parse_ble_payload(sender, data: bytearray):
    """Parses incoming BLE notification packets into buffers."""
    global bpm_buffer, rr_buffer

    flags = data[0]
    hr_format = flags & 0x01

    if hr_format == 0:
        bpm = data[1]
        offset = 2
    else:
        bpm = (data[2] << 8) | data[1]
        offset = 3

    if bpm > 0:
        bpm_buffer.append(bpm)

    # Parse R-R intervals if present (bit 4)
    if flags & 0x10:
        while offset + 1 < len(data):
            rr_raw = (data[offset + 1] << 8) | data[offset]
            rr_ms = (rr_raw / 1024.0) * 1000.0
            if 300.0 <= rr_ms <= 2000.0:  # Valid human physiological range
                rr_buffer.append(rr_ms)
            offset += 2


def write_30s_window_to_csv():
    """Computes all mathematical features across the 30s buffer and logs a row."""
    global bpm_buffer, rr_buffer

    if len(bpm_buffer) == 0:
        print("⚠️ Waiting for BLE samples...")
        return

    now = int(time.time())
    dt_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. HR Statistics
    mean_hr = round(float(np.mean(bpm_buffer)), 2)
    std_hr = round(float(np.std(bpm_buffer)), 2)
    min_hr = round(float(np.min(bpm_buffer)), 2)
    max_hr = round(float(np.max(bpm_buffer)), 2)
    range_hr = round(max_hr - min_hr, 2)
    median_hr = round(float(np.median(bpm_buffer)), 2)

    # 2. HRV (R-R Interval) Statistics
    if len(rr_buffer) > 1:
        mean_rr = round(float(np.mean(rr_buffer)), 2)
        sdnn = round(float(np.std(rr_buffer)), 2)

        rr_diffs = np.abs(np.diff(rr_buffer))
        rmssd = round(float(np.sqrt(np.mean(rr_diffs**2))), 2)

        pnn50 = round(
            float((np.sum(rr_diffs > 50.0) / len(rr_diffs)) * 100.0), 2
        )
        pnn20 = round(
            float((np.sum(rr_diffs > 20.0) / len(rr_diffs)) * 100.0), 2
        )
    else:
        mean_rr, sdnn, rmssd, pnn50, pnn20 = 0.0, 0.0, 0.0, 0.0, 0.0

    # Write full vector to CSV
    with open(CSV_FILE_PATH, mode="a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                now,
                dt_str,
                mean_hr,
                std_hr,
                min_hr,
                max_hr,
                range_hr,
                median_hr,
                mean_rr,
                sdnn,
                rmssd,
                pnn50,
                pnn20,
            ]
        )

    print(
        f"[{dt_str}] 💾 30s Window | HR: {mean_hr} BPM (Max: {max_hr}) | "
        f"RMSSD: {rmssd} ms | pNN50: {pnn50}%"
    )

    # Clear buffers for next 30-second window
    bpm_buffer.clear()
    rr_buffer.clear()


async def main():
    initialize_csv()
    print("🔍 Scanning for WHOOP device over BLE...")

    devices = await BleakScanner.discover()
    whoop_device = None

    for d in devices:
        if d.name and "WHOOP" in d.name.upper():
            whoop_device = d
            break

    if not whoop_device:
        print(
            "❌ WHOOP band not found. Ensure 'Broadcast Heart Rate' is ON in WHOOP App."
        )
        return

    print(f"✅ Found WHOOP: {whoop_device.name} [{whoop_device.address}]")

    async with BleakClient(whoop_device.address) as client:
        print("⚡ Connected! Subscribing to 1 Hz BLE stream...")
        await client.start_notify(
            HEART_RATE_MEASUREMENT_UUID, parse_ble_payload
        )

        print("📡 Streaming active. Outputting 30-second feature vectors...")

        while True:
            await asyncio.sleep(30)
            write_30s_window_to_csv()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n🛑 Stopped BLE Harvester.")