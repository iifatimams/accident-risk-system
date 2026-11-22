"""Add a ground-truth evaluation annotation for one 10-second window of a
recorded session. Entered by an authorized evaluator (an administrator
account) after independently reviewing the drive -- never derived from the
model's own recorded prediction.

Updating an already-annotated window inserts a NEW revision rather than
overwriting the old one; the full history survives (see --history).

Run:
  .venv/bin/python scripts/annotate_session.py --session-id 38 \\
      --window-start 1795298833.2 --window-seconds 10 \\
      --hazard safe --distracted attentive --annotator-phone "+9715..."

  .venv/bin/python scripts/annotate_session.py --session-id 38 --list
  .venv/bin/python scripts/annotate_session.py --session-id 38 \\
      --window-start 1795298833.2 --window-seconds 10 --history
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import (  # noqa: E402
    GROUND_TRUTH_ATTENTION_STATES,
    GROUND_TRUTH_HAZARD_STATES,
    Database,
    normalise_phone,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--window-start", type=float, help="window_start_ts (Unix seconds)")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--hazard", choices=GROUND_TRUTH_HAZARD_STATES)
    parser.add_argument("--distracted", choices=GROUND_TRUTH_ATTENTION_STATES)
    parser.add_argument("--notes", default="")
    parser.add_argument("--annotator-phone", help="Must belong to an administrator account.")
    parser.add_argument("--list", action="store_true", help="List current annotations for this session.")
    parser.add_argument("--history", action="store_true", help="Show every revision for one window.")
    args = parser.parse_args()

    db = Database(Path(args.db))

    if args.list:
        for row in db.latest_annotations(args.session_id):
            print(
                f"[{row['window_start_ts']:.1f}, {row['window_end_ts']:.1f}) "
                f"hazard={row['ground_truth_hazard']} distracted={row['ground_truth_distracted']} "
                f"rev={row['revision']} by={row['annotator_id']} notes={row['annotation_notes']!r}"
            )
        return

    if args.history:
        if args.window_start is None:
            parser.error("--history requires --window-start")
        for row in db.annotation_history(
            args.session_id, args.window_start, args.window_start + args.window_seconds
        ):
            print(
                f"rev={row['revision']} hazard={row['ground_truth_hazard']} "
                f"distracted={row['ground_truth_distracted']} annotated_at={row['annotated_at']} "
                f"notes={row['annotation_notes']!r}"
            )
        return

    if args.window_start is None or args.hazard is None or args.distracted is None:
        parser.error("Adding an annotation requires --window-start, --hazard and --distracted")
    if not args.annotator_phone:
        parser.error("--annotator-phone is required")

    evaluator = db.get_user_by_phone(normalise_phone(args.annotator_phone))
    if evaluator is None:
        raise SystemExit(f"No account with phone {args.annotator_phone!r}")
    if evaluator["role"] != "administrator":
        raise SystemExit("Only an administrator account may enter ground-truth annotations")

    row = db.add_annotation(
        args.session_id,
        args.window_start,
        args.window_start + args.window_seconds,
        ground_truth_hazard=args.hazard,
        ground_truth_distracted=args.distracted,
        annotator_id=evaluator["id"],
        annotation_notes=args.notes,
    )
    print(f"Recorded annotation id={row['id']} revision={row['revision']}")


if __name__ == "__main__":
    main()
