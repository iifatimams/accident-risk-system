"""Mark one alert event as a correct detection, a false alarm, or unreviewed.
Entered by an authorized evaluator; separate from the alert's own
operational status/lifecycle (pending/confirmed/...).

Run:
  .venv/bin/python scripts/verify_alert.py --alert-id 12 --result correct \\
      --verifier-phone "+9715..." --notes "matched dashcam footage"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import ALERT_VERIFICATION_RESULTS, Database, normalise_phone  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--alert-id", type=int, required=True)
    parser.add_argument("--result", choices=ALERT_VERIFICATION_RESULTS, required=True)
    parser.add_argument("--verifier-phone", required=True)
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    db = Database(Path(args.db))
    verifier = db.get_user_by_phone(normalise_phone(args.verifier_phone))
    if verifier is None:
        raise SystemExit(f"No account with phone {args.verifier_phone!r}")
    if verifier["role"] != "administrator":
        raise SystemExit("Only an administrator account may verify an alert")

    try:
        row = db.verify_alert(args.alert_id, args.result, verified_by=verifier["id"], notes=args.notes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    print(f"Alert {args.alert_id} marked {row['verification_result']!r}")


if __name__ == "__main__":
    main()
