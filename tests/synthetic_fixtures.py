"""Loader for the RoadSense_Synthetic_Driving_Scenarios_v1 fixture package
(tests/fixtures/roadsense_synthetic_v1/). Test-only, not imported by any
runtime/production code path.

SYNTHETIC SOFTWARE TEST DATA ONLY. See that directory's README.md for full
provenance. Per its manifest: not real vehicle data, not real UAE routes, not
usable to claim model accuracy, not for training without independent
validation, not for real alert delivery. Every frame here was already
validated against the real accident_risk.schemas.SensorFrame model before
being committed (see test_synthetic_fixtures.py::test_fixture_matches_schema).

Ground truth (the *_labels.csv files) is deliberately loaded as a SEPARATE
list of plain dicts, never merged into the SensorFrame objects used as model
input -- mixing them would let ground truth leak into whatever consumes
these frames, which is exactly what the source package's own README warns
against.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

from accident_risk.schemas import SensorFrame

FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "roadsense_synthetic_v1"
_MANIFEST = json.loads((FIXTURE_ROOT / "manifest.json").read_text())

SCENARIO_NAMES: tuple[str, ...] = tuple(s["scenario"] for s in _MANIFEST["scenario_list"])


@dataclass(frozen=True)
class SyntheticScenario:
    name: str
    outcome: str  # "normal" | "near_miss" | "collision"
    stream_id: str
    onset_s: float | None  # synthetic collision-contact offset, or None
    frames: tuple[SensorFrame, ...]
    labels: tuple[dict, ...]  # ground truth ONLY -- never fed as model input


def load_scenario(name: str) -> SyntheticScenario:
    entry = next((s for s in _MANIFEST["scenario_list"] if s["scenario"] == name), None)
    if entry is None:
        raise KeyError(f"Unknown synthetic scenario: {name!r}; known: {SCENARIO_NAMES}")
    frames_path = FIXTURE_ROOT / entry["frames"]
    labels_path = FIXTURE_ROOT / entry["labels"]
    frames = tuple(
        SensorFrame.model_validate_json(line)
        for line in frames_path.read_text().splitlines()
        if line.strip()
    )
    labels = tuple(csv.DictReader(labels_path.open()))
    return SyntheticScenario(
        name=name,
        outcome=entry["outcome"],
        stream_id=entry["stream_id"],
        onset_s=entry["onset_s"],
        frames=frames,
        labels=labels,
    )


def iter_scenarios() -> list[SyntheticScenario]:
    return [load_scenario(name) for name in SCENARIO_NAMES]
