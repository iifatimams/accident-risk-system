"""Evaluation with undefined metrics reported as null, never fabricated."""

from typing import Any

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)


def classification_metrics(
    y_true: list | np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    classes: list[str],
    score_labels: list[str],
) -> dict[str, Any]:
    """Macro/per-class classification and one-vs-rest PR-AUC (average precision).

    Binary PR-AUC/ROC-AUC measure configured score_labels versus the rest.
    Undefined one-class discrimination metrics are null with an explanation.
    """
    true = np.asarray(y_true)
    if len(true) == 0:
        raise ValueError("No evaluation examples")
    if set(true) - set(classes):
        raise ValueError("Evaluation includes classes absent from training")
    precision, recall, f1, _ = precision_recall_fscore_support(
        true, y_pred, labels=classes, average="macro", zero_division=0
    )
    binary = np.isin(true, score_labels).astype(int)
    score = probabilities[:, [i for i, c in enumerate(classes) if c in score_labels]].sum(axis=1)
    per_class_ap = {}
    for i, label in enumerate(classes):
        target = (true == label).astype(int)
        per_class_ap[label] = (
            float(average_precision_score(target, probabilities[:, i]))
            if len(set(target)) == 2
            else None
        )
    defined = len(set(binary)) == 2
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": confusion_matrix(true, y_pred, labels=classes).tolist(),
        "classes": classes,
        "classification_report": classification_report(
            true, y_pred, labels=classes, output_dict=True, zero_division=0
        ),
        "roc_auc": float(roc_auc_score(binary, score)) if defined else None,
        "pr_auc": float(average_precision_score(binary, score)) if defined else None,
        "pr_auc_definition": "Average precision for configured score_labels versus all remaining labels",
        "per_class_pr_auc": per_class_ap,
        "discrimination_note": None
        if defined
        else "Undefined: partition lacks positive or negative examples",
        "n_examples": len(true),
    }


def event_metrics(
    predictions: list[tuple[float, float]],
    event_times: list[float],
    observation_intervals: list[tuple[float, float]],
    threshold: float = 0.6,
    warning_horizon_seconds: float = 5,
    cooldown_seconds: float = 10,
) -> dict[str, Any]:
    """Evaluate one stream using supplied exposure intervals and actual event onsets.

    Warning episodes are threshold crossings separated by cooldown. A warning
    matches at most one event in [warning, warning+horizon]. Right-censored
    warnings/events are excluded. Hours use supplied driving exposure, not gaps.
    """
    if not 0 <= threshold <= 1 or warning_horizon_seconds < 0 or cooldown_seconds < 0:
        raise ValueError("Invalid event metric parameters")
    intervals = sorted(observation_intervals)
    if any(a >= b for a, b in intervals) or any(
        intervals[i][0] < intervals[i - 1][1] for i in range(1, len(intervals))
    ):
        raise ValueError("Exposure intervals must be positive and nonoverlapping")
    duration = sum(b - a for a, b in intervals)
    eligible = sorted(
        t
        for t in event_times
        if any(a <= t - warning_horizon_seconds and t <= b for a, b in intervals)
    )
    detected: set[int] = set()
    leads = []
    false_alarms = 0
    last_alarm = -float("inf")
    was_high = False
    previous_interval = None
    for timestamp, score in sorted(predictions):
        interval = next(((a, b) for a, b in intervals if a <= timestamp <= b), None)
        if interval is None:
            continue
        if interval != previous_interval:
            was_high = False
            last_alarm = -float("inf")
        previous_interval = interval
        high = score >= threshold
        if (
            high
            and not was_high
            and timestamp - last_alarm >= cooldown_seconds
            and timestamp + warning_horizon_seconds <= interval[1]
        ):
            last_alarm = timestamp
            match = next(
                (
                    i
                    for i, t in enumerate(eligible)
                    if i not in detected and timestamp <= t <= timestamp + warning_horizon_seconds
                ),
                None,
            )
            if match is None:
                false_alarms += 1
            else:
                detected.add(match)
                leads.append(eligible[match] - timestamp)
        was_high = high
    return {
        "false_alarms_per_driving_hour": false_alarms / (duration / 3600) if duration else None,
        "event_detection_rate": len(detected) / len(eligible) if eligible else None,
        "warning_lead_time_seconds": float(np.mean(leads)) if leads else None,
        "eligible_events": len(eligible),
        "detected_events": len(detected),
        "false_alarm_episodes": false_alarms,
        "driving_hours": duration / 3600,
    }
