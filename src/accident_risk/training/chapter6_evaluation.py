"""Chapter 6 evaluation: join recorded predictions with independent ground-truth
annotations over fixed-width windows, and compute TP/TN/FP/FN-based metrics.

Window alignment: windows are fixed-width (default 10 seconds) and aligned to
each session's own started_at, i.e. window k covers
[started_at + k*window_seconds, started_at + (k+1)*window_seconds). An
annotation is matched to a window only when its own (window_start_ts,
window_end_ts) are exactly the same two numbers this module would compute for
that session and k -- so annotate with the same window_seconds the
evaluation itself uses (see scripts/annotate_session.py --window-seconds).

Ground truth is never derived from the recorded prediction: predictions come
from session_samples (what RoadSense itself output), labels come from
evaluation_annotations (what an authorized evaluator entered by watching the
drive/footage independently). Windows with no annotation, or an annotation of
"unknown", are excluded from every supervised metric and counted separately.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from accident_risk.api.db import Database
from accident_risk.training.evaluate import classification_metrics

HAZARD_CLASSES = ["safe", "hazardous"]
ATTENTION_CLASSES = ["attentive", "distracted"]


@dataclass
class WindowAggregate:
    session_id: int
    window_start_ts: float
    window_end_ts: float
    sample_count: int
    # Camera-inclusive/fused: mean of the recorded overall_risk_index [0,1].
    fused_risk_mean: float | None
    # Vehicle-only: mean of the recorded vehicle_risk_score [0,1] -- the
    # device's own breakdown.fiveSecondAheadVehicleHazardProb, never the
    # blended multimodal forecast.
    vehicle_only_risk_mean: float | None
    # Fraction of samples in this window with attention_state != "attentive"
    # (and not a confidence-only state like face-not-detected/uncertain).
    distracted_fraction: float | None
    risk_medium_threshold: float
    risk_high_threshold: float
    risk_critical_threshold: float


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _fixed_grid_buckets(
    samples: list[dict[str, Any]], started_at: float, window_seconds: float
) -> dict[tuple[float, float], list[dict[str, Any]]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in samples:
        index = int((row["ts"] - started_at) // window_seconds)
        buckets.setdefault(index, []).append(row)
    return {
        (started_at + index * window_seconds, started_at + (index + 1) * window_seconds): rows
        for index, rows in buckets.items()
    }


def _value_change_buckets(
    samples: list[dict[str, Any]], score_field: str
) -> dict[tuple[float, float], list[dict[str, Any]]]:
    """Group consecutive samples sharing the same recorded score value into
    one window each, bounded by [this run's first ts, next run's first ts).
    Used for pre-aggregated/stepped historical data (e.g. an evaluation CSV
    recorded at irregular ~10s intervals) where real windows are NOT evenly
    spaced and a fixed time grid would silently misalign scores against
    ground truth. Never used for ordinary live recordings, where a
    continuously-varying score would make every sample its own "run"."""
    ordered = sorted(samples, key=lambda r: r["ts"])
    runs: list[tuple[float, Any, list[dict[str, Any]]]] = []
    for row in ordered:
        value = row.get(score_field)
        if runs and runs[-1][1] == value:
            runs[-1][2].append(row)
        else:
            runs.append((row["ts"], value, [row]))
    buckets = {}
    for i, (start_ts, _value, rows) in enumerate(runs):
        end_ts = runs[i + 1][0] if i + 1 < len(runs) else rows[-1]["ts"] + 1e-6
        buckets[(start_ts, end_ts)] = rows
    return buckets


def build_window_aggregates(
    db: Database, session_id: int, *, window_seconds: float = 10.0,
    vehicle_score_field: str = "vehicle_risk_score",
    window_mode: str = "fixed_grid",
) -> list[WindowAggregate]:
    """One row per window that has at least one recorded sample.

    vehicle_score_field defaults to the correct, go-forward field
    (vehicle_risk_score). Override it only for a batch where that column is
    known NOT to hold the real vehicle-only value -- e.g. the historical
    sessions imported from RoadSense_7200.csv, where vehicle_risk_score was
    recorded as a duplicate of overall_risk_index and the genuine
    vehicle-only score instead sits under
    five_second_ahead_vehicle_hazard_prob.

    window_mode="fixed_grid" (default, correct for ordinary live recordings):
    windows are fixed-width, aligned to the session's own started_at.
    window_mode="value_change": windows are instead each run of consecutive
    samples sharing the same overall_risk_index -- for pre-aggregated
    historical data recorded at irregular intervals, where a fixed grid
    would not line up with the data's own real window boundaries."""
    session = db.get_session(session_id)
    if session is None:
        raise ValueError(f"No such session: {session_id}")
    samples = db.session_samples(session_id)
    if not samples:
        return []
    started_at = session["started_at"]

    if window_mode == "fixed_grid":
        buckets = _fixed_grid_buckets(samples, started_at, window_seconds)
    elif window_mode == "value_change":
        buckets = _value_change_buckets(samples, "overall_risk_index")
    else:
        raise ValueError(f"Unknown window_mode: {window_mode!r}")

    aggregates = []
    for (window_start_ts, window_end_ts), rows in sorted(buckets.items()):
        thresholds = db.effective_thresholds_at(window_start_ts)

        fused = [r["overall_risk_index"] for r in rows if r.get("overall_risk_index") is not None]
        vehicle_only = [
            r[vehicle_score_field] for r in rows if r.get(vehicle_score_field) is not None
        ]
        attention_observed = [
            r["attention_state"] for r in rows
            if r.get("attention_state") not in (None, "face_not_detected", "uncertain")
        ]
        distracted_fraction = (
            sum(1 for a in attention_observed if a != "attentive") / len(attention_observed)
            if attention_observed else None
        )

        aggregates.append(
            WindowAggregate(
                session_id=session_id,
                window_start_ts=window_start_ts,
                window_end_ts=window_end_ts,
                sample_count=len(rows),
                fused_risk_mean=_mean(fused),
                vehicle_only_risk_mean=_mean(vehicle_only),
                distracted_fraction=distracted_fraction,
                risk_medium_threshold=thresholds["risk_medium_threshold"],
                risk_high_threshold=thresholds["risk_high_threshold"],
                risk_critical_threshold=thresholds["risk_critical_threshold"],
            )
        )
    return aggregates


@dataclass
class JoinedWindow:
    """One window with both the recorded prediction and (if present) its
    independent ground-truth annotation."""
    aggregate: WindowAggregate
    ground_truth_hazard: str | None  # None if no annotation exists at all
    ground_truth_distracted: str | None
    annotation_notes: str = ""


def join_windows_with_annotations(
    db: Database, session_id: int, *, window_seconds: float = 10.0,
    vehicle_score_field: str = "vehicle_risk_score",
    window_mode: str = "fixed_grid",
) -> list[JoinedWindow]:
    aggregates = build_window_aggregates(
        db, session_id, window_seconds=window_seconds, vehicle_score_field=vehicle_score_field,
        window_mode=window_mode,
    )
    annotations = {
        (round(a["window_start_ts"], 3), round(a["window_end_ts"], 3)): a
        for a in db.latest_annotations(session_id)
    }
    joined = []
    for agg in aggregates:
        key = (round(agg.window_start_ts, 3), round(agg.window_end_ts, 3))
        annotation = annotations.get(key)
        joined.append(
            JoinedWindow(
                aggregate=agg,
                ground_truth_hazard=annotation["ground_truth_hazard"] if annotation else None,
                ground_truth_distracted=annotation["ground_truth_distracted"] if annotation else None,
                annotation_notes=annotation["annotation_notes"] if annotation else "",
            )
        )
    return joined


@dataclass
class EvaluationResult:
    pathway: str  # "vehicle_only" | "camera_inclusive" | "attention"
    metrics: dict[str, Any] | None
    confusion_counts: dict[str, int]  # tp/tn/fp/fn
    evaluated_windows: int
    excluded_unannotated: int
    excluded_unknown: int
    per_session: dict[int, dict[str, int]] = field(default_factory=dict)


def _confusion_counts(y_true: list[str], y_pred: list[str], positive: str) -> dict[str, int]:
    tp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == positive and p == positive)
    tn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != positive and p != positive)
    fp = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t != positive and p == positive)
    fn = sum(1 for t, p in zip(y_true, y_pred, strict=True) if t == positive and p != positive)
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn}


def evaluate_pathway(
    joined_by_session: dict[int, list[JoinedWindow]],
    pathway: str,
    *,
    vehicle_decision_threshold: float = 0.5,
    camera_decision_threshold: float | None = None,
    attention_decision_threshold: float = 0.5,
) -> EvaluationResult:
    """pathway is one of "vehicle_only", "camera_inclusive", "attention".

    camera_decision_threshold defaults to None, meaning: use each window's
    OWN recorded risk_high_threshold (the app's actual alert threshold at
    that time) rather than a single hardcoded value -- set it explicitly to
    evaluate against a different, fixed research decision threshold instead.
    """
    y_true: list[str] = []
    y_pred: list[str] = []
    scores: list[float] = []
    excluded_unannotated = 0
    excluded_unknown = 0
    evaluated = 0
    per_session: dict[int, dict[str, int]] = {}

    classes = ATTENTION_CLASSES if pathway == "attention" else HAZARD_CLASSES
    positive = "distracted" if pathway == "attention" else "hazardous"

    for session_id, windows in joined_by_session.items():
        session_true: list[str] = []
        session_pred: list[str] = []
        for jw in windows:
            if pathway == "attention":
                truth = jw.ground_truth_distracted
            else:
                truth = jw.ground_truth_hazard
            if truth is None:
                excluded_unannotated += 1
                continue
            if truth == "unknown":
                excluded_unknown += 1
                continue

            if pathway == "vehicle_only":
                score = jw.aggregate.vehicle_only_risk_mean
                if score is None:
                    excluded_unannotated += 1  # no vehicle-only score available this window
                    continue
                predicted = "hazardous" if score >= vehicle_decision_threshold else "safe"
            elif pathway == "camera_inclusive":
                score = jw.aggregate.fused_risk_mean
                if score is None:
                    excluded_unannotated += 1
                    continue
                threshold = (
                    camera_decision_threshold
                    if camera_decision_threshold is not None
                    else jw.aggregate.risk_high_threshold
                )
                predicted = "hazardous" if score >= threshold else "safe"
            else:  # attention
                score = jw.aggregate.distracted_fraction
                if score is None:
                    excluded_unannotated += 1
                    continue
                predicted = "distracted" if score >= attention_decision_threshold else "attentive"

            y_true.append(truth)
            y_pred.append(predicted)
            scores.append(score)
            session_true.append(truth)
            session_pred.append(predicted)
            evaluated += 1

        if session_true:
            per_session[session_id] = _confusion_counts(session_true, session_pred, positive)

    metrics = None
    if y_true:
        probabilities = np.zeros((len(y_true), len(classes)))
        positive_index = classes.index(positive)
        negative_index = 1 - positive_index
        for i, s in enumerate(scores):
            probabilities[i, positive_index] = s
            probabilities[i, negative_index] = 1 - s
        metrics = classification_metrics(
            y_true, np.array(y_pred), probabilities, classes, [positive]
        )
        # classification_metrics' own precision/recall/f1 are macro-averaged
        # across BOTH classes, which is the wrong number here: with ~95% of
        # windows "safe", averaging in that class's near-perfect score
        # inflates the result and hides how the model actually does on the
        # class that matters ("hazardous"/"distracted"). Dropped entirely,
        # not just relabeled, so it can never be read by mistake again --
        # only the single positive-class precision/recall/f1 below (the
        # standard convention, and the one this project's report uses) is
        # reported.
        del metrics["precision"], metrics["recall"], metrics["f1"]
        counts = _confusion_counts(y_true, y_pred, positive)
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        positive_precision = tp / (tp + fp) if (tp + fp) else None
        positive_recall = tp / (tp + fn) if (tp + fn) else None
        positive_f1 = (
            2 * positive_precision * positive_recall / (positive_precision + positive_recall)
            if positive_precision and positive_recall and (positive_precision + positive_recall) > 0
            else None
        )
        metrics["positive_class_precision"] = positive_precision
        metrics["positive_class_recall"] = positive_recall
        metrics["positive_class_f1"] = positive_f1
        metrics["positive_class"] = positive

    return EvaluationResult(
        pathway=pathway,
        metrics=metrics,
        confusion_counts=_confusion_counts(y_true, y_pred, positive),
        evaluated_windows=evaluated,
        excluded_unannotated=excluded_unannotated,
        excluded_unknown=excluded_unknown,
        per_session=per_session,
    )
