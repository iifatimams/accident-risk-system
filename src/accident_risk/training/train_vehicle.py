"""Configuration-driven grouped baseline experiments using the online feature path."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from accident_risk.datasets.base import DatasetRecord, load_prepared
from accident_risk.datasets.registry import get_adapter
from accident_risk.models.physiology_model import PhysiologyModel
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.processing.pipeline import FeaturePipeline, FeatureSnapshot
from accident_risk.processing.windows import future_event_target
from accident_risk.schemas import PHYSIOLOGY_FIELDS
from accident_risk.settings import PipelineSettings
from accident_risk.training.evaluate import classification_metrics
from accident_risk.training.splits import group_split

VEHICLE_INPUTS = (
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "vehicle_speed",
    "gps_speed",
    "rpm",
    "throttle_position",
    "engine_load",
)


def usable(snapshot: FeatureSnapshot, physiology: bool = False) -> bool:
    """Require at least one current nonstale modality reading; never score on history alone."""
    fields = PHYSIOLOGY_FIELDS[:-1] if physiology else VEHICLE_INPUTS
    return any(getattr(snapshot.frame, name) is not None for name in fields)


def build_examples(
    records: list[DatasetRecord],
    config: dict[str, Any],
    physiology: bool = False,
    report: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """Extract causal windows with labels separate; reject censored future targets."""
    settings = PipelineSettings(**config.get("pipeline", {}))
    pipeline = FeaturePipeline(settings)
    counts = report if report is not None else {}
    for key in (
        "input_frames",
        "cadence_or_warmup",
        "unavailable_modality",
        "missing_label",
        "right_censored",
        "event_in_features",
        "followup_gap",
        "emitted_windows",
    ):
        counts[key] = 0
    options = config.get("training", {})
    if settings.prediction_horizon_seconds and not options.get("future_annotations_complete"):
        raise ValueError(
            "Future prediction requires training.future_annotations_complete: true after annotation coverage review"
        )
    max_gap = float(options.get("max_followup_gap_seconds", 2))
    if not 0 < max_gap < float("inf"):
        raise ValueError("max_followup_gap_seconds must be positive and finite")
    rows, labels, groups = [], [], []
    by_stream: dict[str, list[DatasetRecord]] = {}
    for record in records:
        by_stream.setdefault(record.frame.stream_id, []).append(record)
    if settings.prediction_horizon_seconds and not any(
        r.event_timestamp is not None for r in records
    ):
        raise ValueError(
            "Future prediction requires verified event onset timestamps; clip labels are insufficient"
        )
    for stream_id, stream in sorted(by_stream.items()):
        stream.sort(key=lambda r: r.frame.timestamp)
        # A stream is a single trip/event; changing identity inside it is a schema error.
        if any(r.groups != stream[0].groups for r in stream):
            raise ValueError(
                f"Group identity changes within stream {stream_id}; provide distinct trip/event stream IDs"
            )
        event_times = sorted({r.event_timestamp for r in stream if r.event_timestamp is not None})
        observed_until = stream[-1].frame.timestamp
        stream_times = np.array([r.frame.timestamp for r in stream])
        for record in stream:
            counts["input_frames"] += 1
            snapshot = pipeline.update(record.frame)
            if snapshot is None:
                counts["cadence_or_warmup"] += 1
                continue
            if not usable(snapshot, physiology):
                counts["unavailable_modality"] += 1
                continue
            label = record.label
            if settings.prediction_horizon_seconds:
                if record.frame.timestamp + settings.prediction_horizon_seconds > observed_until:
                    counts["right_censored"] += 1
                    continue
                end = record.frame.timestamp + settings.prediction_horizon_seconds
                followup = stream_times[
                    (stream_times >= record.frame.timestamp) & (stream_times <= end)
                ]
                if np.diff(np.append(followup, end)).max(initial=0) > max_gap:
                    counts["followup_gap"] += 1
                    continue
                # Do not train post-onset windows as pre-event negatives.
                feature_start = max(
                    0,
                    record.frame.timestamp
                    - (
                        settings.physiology_window_seconds
                        if physiology
                        else settings.vehicle_window_seconds
                    ),
                )
                if any(feature_start <= t <= snapshot.window.end for t in event_times):
                    counts["event_in_features"] += 1
                    continue
                label = str(
                    future_event_target(
                        snapshot.window,
                        event_times,
                        settings.prediction_horizon_seconds,
                        observed_until,
                    )
                )
            if label is None:
                counts["missing_label"] += 1
                continue
            counts["emitted_windows"] += 1
            rows.append(snapshot.physiology if physiology else snapshot.vehicle)
            labels.append(label)
            groups.append(
                record.groups | {"stream_id": stream_id, "timestamp": record.frame.timestamp}
            )
    if not rows:
        raise ValueError(
            "No labeled usable windows; check mappings, cadence, horizon and sensor quality"
        )
    return pd.DataFrame(rows), np.array(labels), pd.DataFrame(groups)


def train(
    config: dict[str, Any],
    prepared: str | Path | None = None,
    records: list[DatasetRecord] | None = None,
    physiology: bool = False,
) -> dict[str, Any]:
    """Compare on validation, then evaluate the selected model once on held-out test data."""
    dataset = config.get("dataset", {})
    if records is None:
        records = (
            load_prepared(prepared)
            if prepared
            else get_adapter(
                dataset.get("name"), dataset.get("path"), dataset.get("mapping")
            ).records()
        )
    preprocessing_report: dict[str, int] = {}
    X, y, groups = build_examples(records, config, physiology, report=preprocessing_report)
    dataset_name = (
        "SYNTHETIC_TEST_ONLY"
        if config.get("synthetic_test_only")
        else (dataset.get("name") or "prepared")
    )
    options = config.get("training", {})
    train_idx, val_idx, test_idx = group_split(
        groups,
        options.get("validation_size", 0.2),
        options.get("test_size", 0.2),
        config.get("random_seed", 42),
    )
    output = Path(config.get("paths", {}).get("artifacts", "artifacts"))
    if "raw" in output.parts:
        raise ValueError("Training artifacts must not be inside data/raw")
    output.mkdir(parents=True, exist_ok=True)
    prefix = "physiology" if physiology else "vehicle"
    score_labels = options.get("score_labels", [])
    if config.get("pipeline", {}).get("prediction_horizon_seconds", 0):
        if score_labels != ["1"]:
            raise ValueError("Future binary target requires training.score_labels: ['1']")
    models: list[VehicleModel] = []
    comparisons: dict[str, Any] = {}
    metric = options.get("selection_metric", "pr_auc")
    if metric not in {"pr_auc", "recall", "f1"}:
        raise ValueError("selection_metric must be pr_auc, recall or f1")
    for family in options.get("models", ["logistic_regression", "random_forest", "xgboost"]):
        model_type = PhysiologyModel if physiology else VehicleModel
        target = options.get(
            "target", "driver_state_risk" if physiology else "event_detection_score"
        )
        model = model_type(family, config.get("random_seed", 42), target, score_labels)
        model.fit(
            X.iloc[train_idx],
            y[train_idx],
            dataset=dataset_name,
            configuration=config,
            class_weight=options.get("class_weight", "balanced"),
        )
        validation = classification_metrics(
            y[val_idx],
            model.predict(X.iloc[val_idx]),
            model.predict_proba(X.iloc[val_idx]),
            model.encoder.classes_.tolist(),
            score_labels,
        )
        if validation[metric] is None:
            raise ValueError(
                f"Validation {metric} undefined. Need more independent positive/negative groups; do not window-random-split or search test seeds."
            )
        comparisons[family] = validation
        model.metadata["evaluation_summary"] = {"validation": validation}
        model.save(output / f"{prefix}_{family}.joblib")
        models.append(model)
    if not models:
        raise ValueError("At least one baseline must be configured")
    best = max(models, key=lambda m: comparisons[m.family][metric])
    test = classification_metrics(
        y[test_idx],
        best.predict(X.iloc[test_idx]),
        best.predict_proba(X.iloc[test_idx]),
        best.encoder.classes_.tolist(),
        score_labels,
    )
    best.metadata["evaluation_summary"]["test"] = test
    best.save(output / f"{prefix}_model.joblib")
    heldout = X.iloc[test_idx].reset_index(drop=True)
    heldout["__label"] = y[test_idx]
    heldout.to_parquet(output / f"{prefix}_test_features.parquet", index=False)
    membership = groups.copy()
    membership["partition"] = ""
    for name, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        membership.loc[indices, "partition"] = name
    membership.to_csv(output / f"{prefix}_split_manifest.csv", index=False)
    summary = {
        "dataset": dataset_name,
        "synthetic_test_only": bool(config.get("synthetic_test_only", False)),
        "preprocessing_report": preprocessing_report,
        "selected_family": best.family,
        "selection_metric": metric,
        "validation_comparison": comparisons,
        "test": test,
        "input_frames": len(records),
        "eligible_labeled_windows": len(X),
        "partition_windows": {
            "train": len(train_idx),
            "validation": len(val_idx),
            "test": len(test_idx),
        },
        "calibrated": False,
        "temporal_metrics": None,
        "temporal_metrics_note": "Use event_metrics with verified onsets and continuous driving exposure; classification windows alone do not establish driving hours",
    }
    (output / f"{prefix}_metrics.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    return summary
