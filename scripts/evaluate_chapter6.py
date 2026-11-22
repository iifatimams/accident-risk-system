"""Generate Chapter 6 evaluation results: join recorded predictions with
independent ground-truth annotations over fixed windows, and compute
TP/TN/FP/FN-based metrics for the vehicle-only, camera-inclusive (fused),
and driver-attention pathways -- overall, per-driver, and per-session.

Nothing here is hardcoded: every number comes from session_samples (what
RoadSense actually recorded) joined against evaluation_annotations (what an
authorized evaluator independently entered). Sessions or windows with no
annotation are excluded and counted, never silently treated as "safe".

Run:
  .venv/bin/python scripts/evaluate_chapter6.py --session-id 38
  .venv/bin/python scripts/evaluate_chapter6.py --all-annotated --out results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import Database  # noqa: E402
from accident_risk.training.chapter6_evaluation import (  # noqa: E402
    evaluate_pathway,
    join_windows_with_annotations,
)


def sessions_with_annotations(db: Database) -> list[int]:
    with db.connect() as connection:
        rows = connection.execute(
            "SELECT DISTINCT session_id FROM evaluation_annotations ORDER BY session_id"
        ).fetchall()
        return [int(r["session_id"]) for r in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--session-id", type=int, action="append", default=[])
    parser.add_argument("--all-annotated", action="store_true", help="Every session with at least one annotation.")
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--vehicle-decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--camera-decision-threshold", type=float, default=None,
        help="Default: use each window's OWN recorded risk_high_threshold instead of one fixed value.",
    )
    parser.add_argument("--attention-decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--vehicle-score-field", default="vehicle_risk_score",
        help=(
            "Which session_samples payload field holds the vehicle-only score. "
            "Override for a batch where vehicle_risk_score is known not to hold "
            "the real vehicle-only value -- e.g. --vehicle-score-field "
            "five_second_ahead_vehicle_hazard_prob for the sessions imported "
            "from RoadSense_7200.csv."
        ),
    )
    parser.add_argument("--out", default=None, help="Write the full JSON result here too.")
    args = parser.parse_args()

    db = Database(Path(args.db))
    session_ids = list(args.session_id)
    if args.all_annotated:
        session_ids = sorted(set(session_ids) | set(sessions_with_annotations(db)))
    if not session_ids:
        raise SystemExit("Pass --session-id (one or more) or --all-annotated")

    joined_by_session = {
        sid: join_windows_with_annotations(
            db, sid, window_seconds=args.window_seconds,
            vehicle_score_field=args.vehicle_score_field,
        )
        for sid in session_ids
    }

    driver_by_session = {
        sid: (db.get_session(sid) or {}).get("user_id") for sid in session_ids
    }

    results = {}
    for pathway in ("vehicle_only", "camera_inclusive", "attention"):
        result = evaluate_pathway(
            joined_by_session,
            pathway,
            vehicle_decision_threshold=args.vehicle_decision_threshold,
            camera_decision_threshold=args.camera_decision_threshold,
            attention_decision_threshold=args.attention_decision_threshold,
        )
        per_driver: dict[int, dict[str, int]] = {}
        for sid, counts in result.per_session.items():
            driver = driver_by_session.get(sid)
            if driver is None:
                continue
            bucket = per_driver.setdefault(driver, {"tp": 0, "tn": 0, "fp": 0, "fn": 0})
            for key, value in counts.items():
                bucket[key] += value

        results[pathway] = {
            "evaluated_windows": result.evaluated_windows,
            "excluded_unannotated": result.excluded_unannotated,
            "excluded_unknown": result.excluded_unknown,
            "confusion_counts": result.confusion_counts,
            "metrics": result.metrics,
            "per_session": result.per_session,
            "per_driver": per_driver,
        }

    print(json.dumps(results, indent=2, default=str))
    if args.out:
        Path(args.out).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nWrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
