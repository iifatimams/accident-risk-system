"""Export one row per controlled, staged collision-detection test, with its
expected and actual outcomes and the derived pass/fail result.

Run:
  .venv/bin/python scripts/export_collision_tests.py
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
    "test_id", "test_timestamp_utc", "scenario_type",
    "expected_collision_detection", "actual_collision_detection",
    "detection_latency_ms", "test_result", "recorded_by", "notes",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    db = Database(Path(args.db))
    tests = db.list_collision_tests()
    out_path = Path(args.out) if args.out else Path(args.db).parent / "collision_tests.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for t in tests:
            writer.writerow({
                "test_id": t["id"],
                "test_timestamp_utc": datetime.fromtimestamp(
                    t["test_timestamp"], tz=UTC
                ).isoformat(),
                "scenario_type": t["scenario_type"],
                "expected_collision_detection": t["expected_collision_detection"],
                "actual_collision_detection": t["actual_collision_detection"],
                "detection_latency_ms": t["detection_latency_ms"],
                "test_result": t["test_result"],
                "recorded_by": t["recorded_by"],
                "notes": t["notes"],
            })

    print(f"{len(tests)} collision tests -> {out_path}")


if __name__ == "__main__":
    main()
