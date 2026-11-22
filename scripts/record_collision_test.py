"""Record one controlled, staged collision-detection test (FR7 verification).
Kept entirely separate from real driving sessions/alerts -- a staged test is
never counted as, or confused with, an actual road collision.

pass/fail is derived automatically (expected == actual), never entered by hand.

Run:
  .venv/bin/python scripts/record_collision_test.py \\
      --scenario "staged_rear_impact_2.5g" --expected true --actual true \\
      --latency-ms 420 --recorder-phone "+9715..." --notes "parking lot, 15 km/h"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import Database, normalise_phone  # noqa: E402


def _bool(value: str) -> bool:
    if value.lower() in ("true", "1", "yes"):
        return True
    if value.lower() in ("false", "0", "no"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--scenario", required=True)
    parser.add_argument("--expected", type=_bool, required=True)
    parser.add_argument("--actual", type=_bool, required=True)
    parser.add_argument("--latency-ms", type=float, default=None)
    parser.add_argument("--notes", default="")
    parser.add_argument("--recorder-phone", default=None)
    args = parser.parse_args()

    db = Database(Path(args.db))
    recorder_id = None
    if args.recorder_phone:
        user = db.get_user_by_phone(normalise_phone(args.recorder_phone))
        if user is None:
            raise SystemExit(f"No account with phone {args.recorder_phone!r}")
        recorder_id = user["id"]

    row = db.record_collision_test(
        args.scenario,
        expected_collision_detection=args.expected,
        actual_collision_detection=args.actual,
        detection_latency_ms=args.latency_ms,
        notes=args.notes,
        recorded_by=recorder_id,
    )
    print(f"Test {row['id']} ({row['scenario_type']}): {row['test_result']}")


if __name__ == "__main__":
    main()
