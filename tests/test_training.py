"""Full grouped training on explicitly synthetic software fixtures."""

import pandas as pd
import pytest

from accident_risk.datasets.base import DatasetRecord
from accident_risk.datasets.registry import get_adapter
from accident_risk.inference.engine import InferenceEngine
from accident_risk.inference.replay_runner import run_source
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.schemas import SensorFrame
from accident_risk.sources.replay import ReplaySource
from accident_risk.synthetic import write_synthetic_fixture
from accident_risk.training.train_physiology import train_physiology
from accident_risk.training.train_vehicle import build_examples, train


def test_three_baselines_and_saved_replay(tmp_path):
    config = write_synthetic_fixture(tmp_path)
    summary = train(config)
    assert len(summary["validation_comparison"]) == 3
    model = VehicleModel.load(tmp_path / "models/vehicle_model.joblib")
    dataset = config["dataset"]
    records = get_adapter("vzcrash", dataset["path"], dataset["mapping"]).records()
    predictions, stats = run_source(
        ReplaySource([r.frame for r in records], maximum_speed=True),
        InferenceEngine(model),
        tmp_path / "predictions.parquet",
    )
    assert stats["processed_frames"] == 160
    assert predictions
    membership = pd.read_csv(tmp_path / "models/vehicle_split_manifest.csv")
    assert membership.groupby("participant")["partition"].nunique().max() == 1
    assert membership.groupby("event")["partition"].nunique().max() == 1


def test_physiology_training_hr_only(tmp_path):
    records = []
    for person in range(10):
        for state in range(2):
            for t in range(4):
                records.append(
                    DatasetRecord(
                        SensorFrame(
                            timestamp=100 + t,
                            stream_id=f"{person}-{state}",
                            source="synthetic",
                            heart_rate=60 + 20 * state,
                        ),
                        "high" if state else "normal",
                        {"participant": str(person), "trip": f"{person}-{state}"},
                    )
                )
    config = {
        "paths": {"artifacts": str(tmp_path)},
        "training": {
            "models": ["logistic_regression"],
            "target": "driver_state_risk",
            "score_labels": ["high"],
        },
    }
    train_physiology(config, records=records)
    assert (tmp_path / "physiology_model.joblib").exists()


def test_future_targets_no_post_event_features_or_censored_negatives():
    records = [
        DatasetRecord(
            SensorFrame(timestamp=100 + t, stream_id="trip", source="synthetic", vehicle_speed=t),
            "normal",
            {"trip": "trip"},
            event_timestamp=106,
        )
        for t in range(11)
    ]
    config = {
        "pipeline": {"prediction_horizon_seconds": 2, "vehicle_window_seconds": 2},
        "training": {"future_annotations_complete": True},
    }
    X, y, groups = build_examples(records, config)
    assert groups.timestamp.max() <= 108
    assert not any(106 <= t <= 108 for t in groups.timestamp)
    assert y[list(groups.timestamp).index(104)] == "1"
    assert X.iloc[list(groups.timestamp).index(104)]["speed_max"] == 4
    for record in records:
        record.event_timestamp = None
    with pytest.raises(ValueError, match="onset"):
        build_examples(records, config)


def test_future_annotation_confirmation_gaps_and_exclusion_report():
    records = [
        DatasetRecord(
            SensorFrame(timestamp=t, stream_id="trip", source="synthetic", vehicle_speed=10),
            "normal",
            {"trip": "trip"},
            event_timestamp=114,
        )
        for t in [*range(100, 106), *range(110, 116)]
    ]
    config = {
        "pipeline": {"prediction_horizon_seconds": 2},
        "training": {"max_followup_gap_seconds": 1},
    }
    with pytest.raises(ValueError, match="coverage"):
        build_examples(records, config)
    config["training"]["future_annotations_complete"] = True
    report = {}
    _, _, groups = build_examples(records, config, report=report)
    assert 105 not in groups.timestamp.tolist()
    assert report["followup_gap"] > 0
    assert report["right_censored"] > 0
    assert report["input_frames"] == len(records)
    assert sum(v for k, v in report.items() if k != "input_frames") == len(records)
