"""Persistence, fusion, evaluation definitions, and optional temporal execution."""

import numpy as np
import pytest

from accident_risk.models.fusion import RiskFusion
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.training.evaluate import classification_metrics, event_metrics


def test_persistence(models, tmp_path):
    model = models[0]
    model.save(tmp_path / "model.joblib")
    restored = VehicleModel.load(tmp_path / "model.joblib")
    assert restored.metadata == model.metadata
    assert (tmp_path / "model.features.json").exists()


def test_fusion_missing_components():
    fusion = RiskFusion()
    assert fusion.combine({"vehicle_risk_score": 0.7, "driver_state_risk": None}) == (0.7, "HIGH")
    assert fusion.combine({"vehicle_risk_score": None}) == (None, None)
    with pytest.raises(ValueError):
        RiskFusion(weights={"vehicle_risk_score": -1})


def test_metrics_and_event_exposure():
    result = classification_metrics(
        ["normal", "crash"],
        np.array(["normal", "crash"]),
        np.array([[0.9, 0.1], [0.1, 0.9]]),
        ["normal", "crash"],
        ["crash"],
    )
    assert result["recall"] == 1
    undefined = classification_metrics(
        ["normal"], np.array(["normal"]), np.array([[0.9, 0.1]]), ["normal", "crash"], ["crash"]
    )
    assert undefined["pr_auc"] is None
    temporal = event_metrics([(10, 0.8), (11, 0.8), (20, 0.1), (100, 0.8)], [14], [(0, 3600)])
    assert temporal["event_detection_rate"] == 1
    assert temporal["warning_lead_time_seconds"] == 4
    assert temporal["false_alarms_per_driving_hour"] == 1


def test_temporal_cpu_and_device_priority():
    """Exercise the optional runtime in its own process (macOS OpenMP isolation)."""
    import subprocess
    import sys

    code = """
from unittest.mock import patch
import torch
from accident_risk.models.temporal_torch import TemporalCNN, select_device, create_temporal_model
with patch.object(torch.backends.mps, "is_available", return_value=True), patch.object(torch.cuda, "is_available", return_value=True):
    assert str(select_device()) == "mps"
with patch.object(torch.backends.mps, "is_available", return_value=False), patch.object(torch.cuda, "is_available", return_value=True):
    assert str(select_device()) == "cuda"
with patch.object(torch.backends.mps, "is_available", return_value=False), patch.object(torch.cuda, "is_available", return_value=False):
    assert str(select_device()) == "cpu"
model = TemporalCNN(3, 2)
output = model(torch.zeros(2, 3, 20))
assert output.shape == (2, 2)
output.sum().backward()
available_model = create_temporal_model(3, 2)
device = next(available_model.parameters()).device
assert available_model(torch.zeros(1, 3, 20, device=device)).shape == (1, 2)
"""
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert completed.returncode == 0, completed.stderr


def test_macos_native_runtime_guard(monkeypatch):
    import sys

    from accident_risk.models.baselines import make_baseline

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "torch", object())
    with pytest.raises(RuntimeError, match="separate processes"):
        make_baseline("xgboost")
