"""All fixtures are synthetic software test data, never real performance evidence."""

import pandas as pd
import pytest

from accident_risk.models.physiology_model import PhysiologyModel
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.processing.pipeline import FeaturePipeline
from accident_risk.schemas import SensorFrame
from accident_risk.settings import PipelineSettings


@pytest.fixture
def settings():
    return PipelineSettings(
        vehicle_window_seconds=3,
        physiology_window_seconds=6,
        max_age_seconds={"heart_rate": 2, "spo2": 3, "physiology_signal_quality": 3},
    )


@pytest.fixture
def models(settings):
    rows, labels = [], []
    for group in range(8):
        pipeline = FeaturePipeline(settings)
        for t in range(6):
            snapshot = pipeline.update(
                SensorFrame(
                    timestamp=100 + t,
                    stream_id=str(group),
                    source="synthetic_test",
                    vehicle_speed=10 + group + t,
                    accel_x=group + t,
                    accel_y=0,
                    accel_z=9.8,
                    heart_rate=60 + 10 * (group % 2),
                    physiology_signal_quality=1,
                )
            )
            if snapshot:
                rows.append(snapshot)
                labels.append("high" if group % 2 else "normal")
    config = {"pipeline": settings.model_dump()}
    vehicle = VehicleModel(score_labels=["high"]).fit(
        pd.DataFrame([r.vehicle for r in rows]),
        labels,
        dataset="SYNTHETIC_TEST_ONLY",
        configuration=config,
    )
    physiology = PhysiologyModel(score_labels=["high"]).fit(
        pd.DataFrame([r.physiology for r in rows]),
        labels,
        dataset="SYNTHETIC_TEST_ONLY",
        configuration=config,
    )
    return vehicle, physiology
