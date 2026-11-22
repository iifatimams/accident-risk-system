"""Aggregate real, measured system latencies: mean, median, 95th percentile,
maximum, valid-measurement count, and missing-measurement count. Never
substitutes a default duration for a missing measurement -- a session with
no reported value for a field is counted as missing, not zero.

Limitation this script does NOT paper over: the on-device risk engine
measures and reports its full prediction+fusion latency as ONE number
(inference_latency_ms, from risk_computation_latency_ms) -- there is no
separately-instrumented "vehicle-only inference" timer distinct from the
camera-inclusive path, so this script cannot report that split. Camera
processing latency (front/back) and alert delivery latency ARE independently
measured and reported here.

Run:
  .venv/bin/python scripts/report_latency_stats.py
  .venv/bin/python scripts/report_latency_stats.py --session-id 38
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import Database  # noqa: E402


def _stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "n": len(arr),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--session-id", type=int, default=None)
    args = parser.parse_args()

    db = Database(Path(args.db))
    session_ids = (
        [args.session_id] if args.session_id is not None
        else [s["id"] for s in db.list_sessions(limit=100000)]
    )

    inference_ms: list[float] = []
    front_ms: list[float] = []
    back_ms: list[float] = []
    total_samples = 0

    for sid in session_ids:
        for row in db.session_samples(sid):
            total_samples += 1
            value = row.get("inference_latency_ms")
            if value is not None and value > 0:
                inference_ms.append(value)
            front = row.get("front_processing_latency_ms")
            if front is not None and front > 0:
                front_ms.append(front)
            back = row.get("back_processing_latency_ms")
            if back is not None and back > 0:
                back_ms.append(back)

    alert_delivery_s: list[float] = []
    missing_delivery = 0
    for sid in session_ids:
        for alert in db.list_alerts(session_id=sid, limit=100000):
            notifications = db.list_family_notifications(alert["id"])
            if not notifications:
                continue  # no family notification was ever attempted for this alert
            delay = notifications[0]["sent_at"] - alert["created_at"]
            if delay >= 0:
                alert_delivery_s.append(delay)
            else:
                missing_delivery += 1

    report = {
        "risk_update_latency_ms (full on-device prediction+fusion)": {
            **_stats(inference_ms),
            "missing": total_samples - len(inference_ms),
        },
        "camera_processing_latency_ms (front)": {
            **_stats(front_ms),
            "missing": total_samples - len(front_ms),
        },
        "camera_processing_latency_ms (back)": {
            **_stats(back_ms),
            "missing": total_samples - len(back_ms),
        },
        "alert_notification_send_latency_seconds": {
            **_stats(alert_delivery_s),
            "excluded_negative_or_malformed": missing_delivery,
        },
        "note": (
            "No separately-instrumented vehicle-only inference timer exists; "
            "risk_update_latency_ms is the full on-device prediction+fusion span."
        ),
    }
    for name, stats in report.items():
        if name == "note":
            continue
        print(f"{name}:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    print(f"\n{report['note']}")


if __name__ == "__main__":
    main()
