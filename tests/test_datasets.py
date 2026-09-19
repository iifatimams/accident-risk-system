"""All fixture columns are invented solely for adapter contract tests."""

import pandas as pd
import pytest

from accident_risk.datasets.base import load_prepared, save_prepared, timestamp_seconds
from accident_risk.datasets.registry import get_adapter
from accident_risk.synthetic import write_synthetic_fixture


def test_adapter_inspection_mapping_roundtrip(tmp_path):
    config = write_synthetic_fixture(tmp_path)
    ds = config["dataset"]
    adapter = get_adapter("vzcrash", ds["path"], ds["mapping"])
    report = adapter.inspect()[0]
    assert "test_epoch_s" in report["columns"]
    assert report["sample_rate_hz"] == 1
    records = adapter.records()
    assert len(records) == 160
    assert records[0].groups["event"]
    save_prepared(records, tmp_path / "prepared.parquet")
    assert load_prepared(tmp_path / "prepared.parquet") == records
    ds["mapping"]["confirmed"] = False
    with pytest.raises(ValueError, match="unconfirmed"):
        adapter.records()


def test_missing_dataset_and_fabricated_column(tmp_path):
    with pytest.raises(FileNotFoundError, match="Place licensed"):
        get_adapter("vzcrash", tmp_path / "absent").records()
    config = write_synthetic_fixture(tmp_path)
    ds = config["dataset"]
    ds["mapping"]["columns"]["rpm"] = "nonexistent"
    with pytest.raises(ValueError, match="missing"):
        get_adapter("polidriving", ds["path"], ds["mapping"]).records()


def test_units_bad_values_and_timezone(tmp_path):
    config = write_synthetic_fixture(tmp_path)
    ds = config["dataset"]
    ds["mapping"]["units"]["vehicle_speed"] = "km/h"
    records = get_adapter("polidriving", ds["path"], ds["mapping"]).records()
    assert records[0].frame.vehicle_speed == pytest.approx(10 / 3.6)
    assert timestamp_seconds(1000, "ms") == 1
    with pytest.raises(ValueError, match="timezone"):
        timestamp_seconds("2026-01-01T00:00:00", "iso8601")
    path = tmp_path / "input/SYNTHETIC_ONLY.csv"
    table = pd.read_csv(path)
    table.loc[0, "test_speed_mps"] = -1
    table.to_csv(path, index=False)
    with pytest.raises(ValueError, match="row 0"):
        get_adapter("polidriving", ds["path"], ds["mapping"]).records()
