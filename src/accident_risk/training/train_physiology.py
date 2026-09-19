"""Independent physiology experiment entry point; unrelated datasets are never joined."""

from pathlib import Path
from typing import Any

from accident_risk.datasets.base import DatasetRecord
from accident_risk.training.train_vehicle import train


def train_physiology(
    config: dict[str, Any],
    prepared: str | Path | None = None,
    records: list[DatasetRecord] | None = None,
) -> dict[str, Any]:
    """Fit only driver-state features to labels from the same physiological recording."""
    return train(config, prepared, records, physiology=True)
