# Future live hardware integration

```text
Android accelerometer + gyroscope + GPS
Bluetooth OBD-II vehicle telemetry       -> timestamped gateway -> POST /ingest
BLE wearable HR / optional SpO2
```

The Android gateway should assign a stable trip `stream_id`, align sensor clocks to Unix
seconds UTC, and send partial SensorFrames. Accelerometer/gyro may update at 50–100 Hz,
OBD approximately 5–20 Hz, GPS around 1 Hz, and wearable measurements more slowly. These are
integration targets, not tested vendor specifications. HR/SpO2 never form a synchronization
barrier. Keep original observation timestamps when forwarding cached values.

Wire units: acceleration m/s², angular velocity rad/s, speed m/s, RPM rev/min, throttle/load
percent, HR bpm, SpO2 percent, RR/IBI/RMSSD ms, quality 0–1. Convert OBD km/h upstream. Do
not convert BPM to synthetic RR samples. Quality scores require a documented device meaning.
The schema accepts individual optional fields and rejects impossible percentages and nonfinite values.
Optional GPS heading and steering angle use radians; tire pressures use kPa. A real steering
angle or four-wheel tire-pressure feed requires a compatible vehicle/adapter, confirmed
supported parameter IDs, and a documented gateway mapping. The web demo simulates these
fields; it does not verify that a generic OBD-II adapter can supply them.

Example wearable update:

```json
{
  "timestamp": 1700000001.25,
  "stream_id": "trip-1",
  "source": "wearable-gateway",
  "heart_rate": 74,
  "physiology_signal_quality": 0.9
}
```

The server holds the latest observed value and age for each channel, expires stale values
according to configuration, and can continue vehicle inference without physiology. The
default feature stride is one second; tune it using measured latency for the target device
and sensor rate. The live buffer raises an explicit overflow if a consumer falls behind.
No hidden dropping or waiting for SpO2 is used.

Choose a stable mounting orientation and document whether acceleration includes gravity.
The server does not infer anatomical/vehicle axes from phone axes. Calibrate and test
orientation and motion artifacts per device. Handle disconnect/reconnect and clock resets
in the gateway; begin a new stream or session after a clock discontinuity. Out-of-order
frames receive an explicit error rather than being inserted into prior predictions.

Use `/session/start`, `/session/stop`, and `/reset` for acquisition lifecycle. Latest outputs
are per stream. When a request arrives between feature strides, `/ingest` may return an
earlier latest prediction; compare its timestamp with the submitted frame's timestamp.
If no new packets arrive at all, no timer creates new predictions: clients must check output age.

Vendor-specific Android/Bluetooth/BLE code is intentionally not implemented. Some wearables
do not expose SpO2 continuously or through a public interface; availability must be confirmed
when hardware is selected. The local API is ready for a gateway prototype, not an authenticated
network deployment or vehicle control system. It does not operate brakes, steering, or
emergency calls. No real hardware connectivity or end-to-end radio timing has been tested.
