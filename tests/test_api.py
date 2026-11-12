"""In-process offline API contract tests with synthetic measurements.

NOTE: this file previously asserted a pre-refactor contract (legacy
top-level `/session/start`, `/session/stop`, `/prediction/latest`,
`/ingest/batch`, `/reset` routes and a single-page tabbed dashboard with
inline tire-pressure gauges). None of those routes or dashboard strings
exist anymore -- the app was rebuilt as a role-based, sign-in-gated SPA
(see `routes_app.py`'s `/app/api/...` endpoints and
`static/app/{index.html,app.js,app.css}`). This rewrite asserts the
contract that actually exists today, confirmed by exercising the real
running app rather than guessing.
"""

import time

from fastapi.testclient import TestClient

from accident_risk.api.main import create_app
from accident_risk.inference.engine import InferenceEngine


def test_api(models):
    with TestClient(create_app(InferenceEngine(*models))) as client:
        dashboard = client.get("/")
        assert dashboard.status_code == 200
        assert "RoadSense" in dashboard.text

        assert client.get("/static/app/app.js").status_code == 200
        assert client.get("/static/app/app.css").status_code == 200
        assert client.get("/docs").status_code == 200
        assert client.get("/health").json()["status"] == "ok"

        payload = {
            "timestamp": time.time(),
            "source": "synthetic",
            "stream_id": "trip",
            "vehicle_speed": 10,
        }
        # First frame in a fresh stream produces no prediction yet.
        first = client.post("/ingest", json=payload)
        assert first.status_code == 200
        assert first.json()["risk_index"] is None

        # Each later frame is timestamped with a freshly captured wall-clock
        # value (never offset into the future) so the live snapshot taken
        # right after is guaranteed to see it as current, not pending.
        demo_frame = payload | {
            "timestamp": time.time(),
            "gps_latitude": 25.1869,
            "gps_longitude": 55.2671,
            "gps_speed": 12.5,
            "gps_heading": 1.2,
            "steering_angle": 0.12,
            "tire_pressure_fl": 235,
            "tire_pressure_fr": 236,
            "tire_pressure_rl": 238,
            "tire_pressure_rr": 237,
            "heart_rate": 74,
            "spo2": 98,
        }
        assert client.post("/ingest", json=demo_frame).status_code == 200

        live = client.get("/api/v1/predict/live", params={"stream_id": "trip"}).json()
        assert live["status"] == "active"
        assert live["heart"] == 74
        assert live["spo2"] == 98
        assert live["location"]["latitude"] == 25.1869

        # Out-of-range / malformed fields are rejected, not silently dropped.
        bad_ts = time.time()
        assert client.post("/ingest", json=payload | {"timestamp": bad_ts, "spo2": 400}).status_code == 422
        assert client.post(
            "/ingest", json=payload | {"timestamp": bad_ts, "gps_latitude": 200}
        ).status_code == 422
        assert client.post(
            "/ingest", json=payload | {"timestamp": bad_ts, "vehicle_speed": -1}
        ).status_code == 422

        # A duplicate/out-of-order timestamp on the same stream is rejected.
        assert client.post("/ingest", json=payload).status_code == 422
