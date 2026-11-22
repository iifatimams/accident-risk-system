"""Export one row per actual alert event (never per high-risk sample), with
its timestamps, severity, delivery status, and an evaluator's verification
outcome when one has been entered.

Run:
  .venv/bin/python scripts/export_alert_verifications.py
  .venv/bin/python scripts/export_alert_verifications.py --session-id 38
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import Database  # noqa: E402

FIELDS = [
    "alert_event_id", "session_id", "driver_id", "kind", "status",
    "created_at", "created_at_utc", "risk_index", "risk_medium_threshold",
    "risk_high_threshold", "risk_critical_threshold",
    "notification_count", "first_delivery_status", "first_notification_sent_at",
    "first_notification_acknowledged_at",
    "verification_result", "verified_by", "verified_at", "verification_notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--session-id", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    db = Database(Path(args.db))
    alerts = db.list_alerts(session_id=args.session_id, limit=100000)
    out_path = Path(args.out) if args.out else Path(args.db).parent / "alert_verifications.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for alert in alerts:
            thresholds = db.effective_thresholds_at(alert["created_at"])
            notifications = db.list_family_notifications(alert["id"])
            first = notifications[0] if notifications else None
            verification = db.get_alert_verification(alert["id"])
            writer.writerow({
                "alert_event_id": alert["id"],
                "session_id": alert["session_id"],
                "driver_id": alert["user_id"],
                "kind": alert["kind"],
                "status": alert["status"],
                "created_at": alert["created_at"],
                "created_at_utc": datetime.fromtimestamp(
                    alert["created_at"], tz=UTC
                ).isoformat(),
                "risk_index": alert["risk_index"],
                "risk_medium_threshold": thresholds["risk_medium_threshold"],
                "risk_high_threshold": thresholds["risk_high_threshold"],
                "risk_critical_threshold": thresholds["risk_critical_threshold"],
                "notification_count": len(notifications),
                # delivery_status is the (mock) send attempt's own report --
                # NOT proof of real delivery; acknowledged_at is the only
                # signal that a human actually received and opened it.
                "first_delivery_status": first["delivery_status"] if first else None,
                "first_notification_sent_at": first["sent_at"] if first else None,
                "first_notification_acknowledged_at": first["acknowledged_at"] if first else None,
                "verification_result": verification["verification_result"] if verification else "unreviewed",
                "verified_by": verification["verified_by"] if verification else None,
                "verified_at": verification["verified_at"] if verification else None,
                "verification_notes": verification["notes"] if verification else "",
            })

    print(f"{len(alerts)} alert events -> {out_path}")


if __name__ == "__main__":
    main()
