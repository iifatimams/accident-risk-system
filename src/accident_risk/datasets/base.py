"""Inspection-first tabular adapters. Mapping is supplied by the dataset custodian."""

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from accident_risk.schemas import SENSOR_FIELDS, SensorFrame

GROUP_PRIORITY = ("participant", "driver", "vehicle", "trip", "event")
UNITS = {
    **{f"accel_{a}": {"m/s^2": 1, "g": 9.80665} for a in "xyz"},
    **{f"gyro_{a}": {"rad/s": 1, "deg/s": np.pi / 180} for a in "xyz"},
    **{s: {"m/s": 1, "km/h": 1 / 3.6, "mph": 0.44704} for s in ("gps_speed", "vehicle_speed")},
    "gps_latitude": {"degrees": 1},
    "gps_longitude": {"degrees": 1},
    "gps_heading": {"rad": 1, "degrees": np.pi / 180},
    "rpm": {"rpm": 1},
    "throttle_position": {"percent": 1, "fraction": 100},
    "engine_load": {"percent": 1, "fraction": 100},
    "steering_angle": {"rad": 1, "degrees": np.pi / 180},
    "tire_pressure_fl": {"kPa": 1, "psi": 6.8947572932},
    "tire_pressure_fr": {"kPa": 1, "psi": 6.8947572932},
    "tire_pressure_rl": {"kPa": 1, "psi": 6.8947572932},
    "tire_pressure_rr": {"kPa": 1, "psi": 6.8947572932},
    "heart_rate": {"bpm": 1},
    "spo2": {"percent": 1, "fraction": 100},
    "hrv_rmssd": {"ms": 1, "s": 1000},
    "rr_interval_ms": {"ms": 1, "s": 1000},
    "physiology_signal_quality": {"fraction": 1, "percent": 0.01},
}


@dataclass
class DatasetRecord:
    """A frame and separate supervision/provenance, never features from label columns."""

    frame: SensorFrame
    label: str | None
    groups: dict[str, str]
    event_timestamp: float | None = None
    context: dict[str, Any] = field(default_factory=dict)


def read_table(path: Path) -> pd.DataFrame:
    """Read supported local formats without mutating the source."""
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(path)
        case ".parquet":
            return pd.read_parquet(path)
        case ".jsonl":
            return pd.read_json(path, lines=True)
        case _:
            raise ValueError(
                f"Unsupported table format: {path}; export CSV, Parquet or JSONL first"
            )


def timestamp_seconds(value: Any, unit: str) -> float:
    """Convert explicit numeric epoch units or timezone-aware ISO-8601 to UTC seconds."""
    if unit == "iso8601":
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is None:
            raise ValueError("ISO timestamp requires an explicit timezone")
        return stamp.timestamp()
    scales = {"s": 1, "ms": 1e-3, "us": 1e-6, "ns": 1e-9}
    if unit not in scales:
        raise ValueError("timestamp_unit must be s/ms/us/ns (Unix epoch) or iso8601")
    return float(value) * scales[unit]


class DatasetAdapter(ABC):
    """Dataset inspection, validated conversion, labels, and leakage-safe identifiers."""

    @abstractmethod
    def inspect(self) -> list[dict[str, Any]]:
        """Describe actual local schema without assuming columns."""

    @abstractmethod
    def records(self) -> list[DatasetRecord]:
        """Return verified standardized records with separate labels and identifiers."""


class MappedTabularAdapter(DatasetAdapter):
    """Reusable explicit mapping implementation for locally inspected datasets.

    Missing cells become null; impossible/nonfinite measurements fail with file/row
    context. Columns, units, labels and grouping are never inferred for training.
    """

    name = "unregistered"

    def __init__(
        self, path: str | Path | None = None, mapping: dict[str, Any] | None = None
    ) -> None:
        self.path = Path(path or f"data/raw/{self.name}")
        self.mapping = mapping or {}

    def files(self, selected: bool = False) -> list[Path]:
        """List actual files; training requires an explicit selection to avoid duplicates."""
        if not self.path.exists():
            raise FileNotFoundError(
                f"{self.name} is absent. Place licensed dataset files in {self.path}; run scripts/inspect_dataset.py before confirming mappings."
            )
        patterns = (
            self.mapping.get("files", [])
            if selected
            else ["**/*.csv", "**/*.parquet", "**/*.jsonl"]
        )
        if selected and not patterns:
            raise ValueError(
                "mapping.files must explicitly select inspected files (do not pool raw and preprocessed copies)"
            )
        files = sorted({p for pattern in patterns for p in self.path.glob(pattern) if p.is_file()})
        if any(not p.resolve().is_relative_to(self.path.resolve()) for p in files):
            raise ValueError("Dataset files must stay inside the supplied dataset path")
        if not files:
            raise FileNotFoundError(
                f"No supported/selected tables in {self.path}; inspect the archive and export supported tables without altering originals"
            )
        return files

    def inspect(self) -> list[dict[str, Any]]:
        """Report columns, dtypes, missingness, labels, identifiers, and inferred cadence."""
        reports = []
        for path in self.files():
            table = read_table(path)
            labels = [
                c
                for c in table
                if any(token in str(c).lower() for token in ("label", "class", "risk", "event"))
            ]
            if self.mapping.get("label_column") in table:
                labels.append(self.mapping["label_column"])
            report: dict[str, Any] = {
                "file": str(path),
                "shape": list(table.shape),
                "columns": list(table.columns),
                "dtypes": table.dtypes.astype(str).to_dict(),
                "missingness": table.isna().mean().to_dict(),
                "candidate_identifiers": [
                    c for c in table if any(t in str(c).lower() for t in (*GROUP_PRIORITY, "id"))
                ],
                "candidate_label_distributions": {
                    c: table[c].astype(str).value_counts().head(30).to_dict() for c in set(labels)
                },
                "sample_rate_hz": None,
            }
            time_column, unit = (
                self.mapping.get("timestamp_column"),
                self.mapping.get("timestamp_unit"),
            )
            if time_column in table and unit:
                t = (
                    table[time_column]
                    .dropna()
                    .map(lambda value, unit=unit: timestamp_seconds(value, unit))
                )
                differences = t.diff()
                positive = differences[differences > 0]
                if len(positive):
                    report["sample_rate_hz"] = float(1 / positive.median())
                    report["sample_rate_note"] = (
                        "Median positive row cadence; verify per sensor and stream"
                    )
            reports.append(report)
        return reports

    def validate_columns(self, table: pd.DataFrame) -> None:
        """Reject unconfirmed mappings, absent columns, ambiguous labels and absent groups."""
        m = self.mapping
        if not m.get("confirmed"):
            raise ValueError(
                "Mapping unconfirmed: inspect actual local columns/units/labels, edit configs, then set mapping.confirmed: true"
            )
        if not m.get("timestamp_unit") or not m.get("stream_column"):
            raise ValueError("Explicit timestamp_unit and stream_column are required")
        if not m.get("label_interpretation") or not m.get("label_map"):
            raise ValueError(
                "Document label_interpretation and actual label_map before preparation"
            )
        if not m.get("groups") or set(m["groups"]) - set(GROUP_PRIORITY):
            raise ValueError("Provide participant/driver/vehicle/trip/event group mappings")
        if not m.get("columns") or set(m["columns"]) - set(SENSOR_FIELDS):
            raise ValueError("columns must map standardized sensor fields to real columns")
        required = [
            m.get("timestamp_column"),
            m.get("stream_column"),
            m.get("label_column"),
            *m["columns"].values(),
            *m["groups"].values(),
            *m.get("context_columns", []),
        ]
        if m.get("event_timestamp_column"):
            required.append(m["event_timestamp_column"])
        missing = [c for c in required if c not in table.columns]
        if missing:
            raise ValueError(
                f"Mapped columns missing: {missing}. Actual columns: {list(table.columns)}"
            )
        for sensor in m["columns"]:
            if m.get("units", {}).get(sensor) not in UNITS[sensor]:
                raise ValueError(
                    f"Explicit supported unit required for {sensor}: {list(UNITS[sensor])}"
                )
        if m["label_column"] in m["columns"].values():
            raise ValueError("Label column must never be mapped as a sensor feature")

    def records(self) -> list[DatasetRecord]:
        """Read only selected tables and fail on any invalid row, with actionable context."""
        result = []
        m = self.mapping
        for path in self.files(selected=True):
            table = read_table(path)
            self.validate_columns(table)
            for index, row in table.iterrows():
                try:
                    groups = {key: str(row[col]) for key, col in m["groups"].items()}
                    if any(
                        pd.isna(row[col]) for col in [m["stream_column"], *m["groups"].values()]
                    ):
                        raise ValueError("Missing stream/group identifier")
                    payload = {
                        sensor: None
                        if pd.isna(row[column])
                        else float(row[column]) * UNITS[sensor][m["units"][sensor]]
                        for sensor, column in m["columns"].items()
                    }
                    frame = SensorFrame(
                        timestamp=timestamp_seconds(
                            row[m["timestamp_column"]], m["timestamp_unit"]
                        ),
                        stream_id=str(row[m["stream_column"]]),
                        source=self.name,
                        **payload,
                    )
                    raw_label = row[m["label_column"]]
                    label_map = {str(k): str(v) for k, v in m["label_map"].items()}
                    if pd.isna(raw_label):
                        label = None
                    elif str(raw_label) not in label_map:
                        raise ValueError(f"Unmapped label {raw_label!r}")
                    else:
                        label = label_map[str(raw_label)]
                    event_value = (
                        row[m["event_timestamp_column"]]
                        if m.get("event_timestamp_column")
                        else None
                    )
                    event_time = (
                        None
                        if event_value is None or pd.isna(event_value)
                        else timestamp_seconds(event_value, m["timestamp_unit"])
                    )
                    result.append(
                        DatasetRecord(
                            frame,
                            label,
                            groups,
                            event_time,
                            {
                                c: None if pd.isna(row[c]) else row[c]
                                for c in m.get("context_columns", [])
                            },
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{path}, row {index}: {exc}") from exc
        return sorted(result, key=lambda r: (r.frame.stream_id, r.frame.timestamp))


def save_prepared(records: list[DatasetRecord], path: str | Path) -> None:
    """Save standardized frames and separate annotations as portable Parquet."""
    path = Path(path)
    if "raw" in path.parts:
        raise ValueError("Prepared output must not be inside data/raw")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [
            {
                "frame_json": r.frame.model_dump_json(),
                "label": r.label,
                "groups_json": json.dumps(r.groups),
                "event_timestamp": r.event_timestamp,
                "context_json": json.dumps(r.context, default=str),
            }
            for r in records
        ]
    ).to_parquet(path, index=False)


def load_prepared(path: str | Path) -> list[DatasetRecord]:
    """Revalidate the common schema when reading prepared data."""
    result = []
    for row in pd.read_parquet(path).to_dict("records"):
        result.append(
            DatasetRecord(
                SensorFrame.model_validate_json(row["frame_json"]),
                None if pd.isna(row["label"]) else str(row["label"]),
                json.loads(row["groups_json"]),
                None if pd.isna(row["event_timestamp"]) else float(row["event_timestamp"]),
                json.loads(row["context_json"]),
            )
        )
    return result
