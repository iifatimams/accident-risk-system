"""Export one session's 10-second evaluation windows to CSV: window bounds,
driver/session identifiers, recorded prediction scores and thresholds,
independent ground-truth labels (when annotated), and the automatically
calculated TP/TN/FP/FN outcome for each of the three pathways.

No window_id column: (session_id, window_start_ts, window_end_ts) already
identifies a window uniquely.

Run:
  .venv/bin/python scripts/export_evaluation_windows.py --session-id 38
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from accident_risk.api.db import Database  # noqa: E402
from accident_risk.training.chapter6_evaluation import join_windows_with_annotations  # noqa: E402

FIELDS = [
    "session_id", "driver_id", "window_start_ts", "window_end_ts",
    "sample_count",
    "vehicle_only_risk_mean", "fused_risk_mean", "distracted_fraction",
    "risk_medium_threshold", "risk_high_threshold", "risk_critical_threshold",
    "ground_truth_hazard", "ground_truth_distracted", "annotation_notes",
    "predicted_hazard_vehicle_only", "predicted_hazard_camera_inclusive", "predicted_distracted",
    "outcome_vehicle_only", "outcome_camera_inclusive", "outcome_attention",
]


def _outcome(truth: str | None, predicted_positive: bool | None, positive: str) -> str:
    if truth is None:
        return "excluded_unannotated"
    if truth == "unknown":
        return "excluded_unknown"
    if predicted_positive is None:
        return "excluded_no_score"
    truth_positive = truth == positive
    if truth_positive and predicted_positive:
        return "TP"
    if not truth_positive and not predicted_positive:
        return "TN"
    if not truth_positive and predicted_positive:
        return "FP"
    return "FN"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(REPO_ROOT / "data" / "app.sqlite3"))
    parser.add_argument("--session-id", type=int, required=True)
    parser.add_argument("--window-seconds", type=float, default=10.0)
    parser.add_argument("--vehicle-decision-threshold", type=float, default=0.5)
    parser.add_argument("--attention-decision-threshold", type=float, default=0.5)
    parser.add_argument(
        "--vehicle-score-field", default="vehicle_risk_score",
        help="See scripts/evaluate_chapter6.py --vehicle-score-field.",
    )
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    db = Database(Path(args.db))
    session = db.get_session(args.session_id)
    if session is None:
        raise SystemExit(f"Session {args.session_id} not found")

    joined = join_windows_with_annotations(
        db, args.session_id, window_seconds=args.window_seconds,
        vehicle_score_field=args.vehicle_score_field,
    )
    out_path = Path(args.out) if args.out else Path(args.db).parent / f"session_{args.session_id}_windows.csv"

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for jw in joined:
            a = jw.aggregate
            vehicle_pred = (
                a.vehicle_only_risk_mean >= args.vehicle_decision_threshold
                if a.vehicle_only_risk_mean is not None else None
            )
            camera_pred = (
                a.fused_risk_mean >= a.risk_high_threshold if a.fused_risk_mean is not None else None
            )
            attention_pred = (
                a.distracted_fraction >= args.attention_decision_threshold
                if a.distracted_fraction is not None else None
            )
            writer.writerow({
                "session_id": a.session_id, "driver_id": session["user_id"],
                "window_start_ts": a.window_start_ts, "window_end_ts": a.window_end_ts,
                "sample_count": a.sample_count,
                "vehicle_only_risk_mean": a.vehicle_only_risk_mean,
                "fused_risk_mean": a.fused_risk_mean,
                "distracted_fraction": a.distracted_fraction,
                "risk_medium_threshold": a.risk_medium_threshold,
                "risk_high_threshold": a.risk_high_threshold,
                "risk_critical_threshold": a.risk_critical_threshold,
                "ground_truth_hazard": jw.ground_truth_hazard,
                "ground_truth_distracted": jw.ground_truth_distracted,
                "annotation_notes": jw.annotation_notes,
                "predicted_hazard_vehicle_only": vehicle_pred,
                "predicted_hazard_camera_inclusive": camera_pred,
                "predicted_distracted": attention_pred,
                "outcome_vehicle_only": _outcome(jw.ground_truth_hazard, vehicle_pred, "hazardous"),
                "outcome_camera_inclusive": _outcome(jw.ground_truth_hazard, camera_pred, "hazardous"),
                "outcome_attention": _outcome(jw.ground_truth_distracted, attention_pred, "distracted"),
            })

    print(f"Session {args.session_id}: {len(joined)} windows -> {out_path}")


if __name__ == "__main__":
    main()
