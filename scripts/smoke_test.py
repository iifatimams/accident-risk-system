"""Exercise adapters, preparation, all baselines, replay and API on SYNTHETIC TEST DATA."""

import argparse
import json
from pathlib import Path

import yaml
from fastapi.testclient import TestClient

from accident_risk.api.main import create_app
from accident_risk.datasets.base import save_prepared
from accident_risk.datasets.registry import get_adapter
from accident_risk.inference.engine import InferenceEngine
from accident_risk.inference.replay_runner import run_source
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.sources.replay import ReplaySource
from accident_risk.synthetic import write_synthetic_fixture
from accident_risk.training.train_vehicle import train


def main() -> None:
    """Write only to a clearly labeled ignored synthetic artifact directory."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="artifacts/synthetic_smoke")
    args = parser.parse_args()
    root = Path(args.output)
    config = write_synthetic_fixture(root)
    (root / "config.yaml").write_text(yaml.safe_dump(config))
    dataset = config["dataset"]
    adapter = get_adapter(dataset["name"], dataset["path"], dataset["mapping"])
    records = adapter.records()
    save_prepared(records, root / "prepared.parquet")
    train(config, prepared=root / "prepared.parquet")
    model = VehicleModel.load(root / "models/vehicle_model.joblib")
    predictions, statistics = run_source(
        ReplaySource([r.frame for r in records], maximum_speed=True),
        InferenceEngine(model),
        root / "predictions.csv",
    )
    assert predictions and predictions[-1].overall_risk_index is not None
    with TestClient(create_app(InferenceEngine(model))) as client:
        assert client.get("/health").status_code == 200
        for record in records[:3]:
            response = client.post("/ingest", json=record.frame.model_dump())
            assert response.status_code == 200
        assert response.json()["event_detection_score"] is not None
    print(
        json.dumps(
            {
                "synthetic_test_only": True,
                "status": "passed",
                "trained_baselines": config["training"]["models"],
                "replay": statistics,
                "note": "Software smoke test, not real model-performance evidence",
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
