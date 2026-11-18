"""Runtime recording preserves observations, including when the API is offline."""

import csv
import json
import time
from concurrent.futures import ThreadPoolExecutor

from accident_risk.schemas import SensorFrame
from accident_risk.sources.recording import CSV_FIELDS, CSVRecorder


def test_rows_are_readable_immediately_and_missing_is_not_zero(tmp_path):
    recorder = CSVRecorder(tmp_path)
    now = time.time()
    frame = SensorFrame(
        timestamp=now,
        source="obd",
        stream_id="trip",
        vehicle_speed=0,
        sensor_timestamps={"vehicle_speed": now - 1},
    )
    path = recorder.write(frame)
    with path.open() as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert reader.fieldnames == list(CSV_FIELDS)
    assert len(rows) == 1
    assert float(rows[0]["vehicle_speed"]) == 0
    assert rows[0]["heart_rate"] == "" and rows[0]["rpm"] == ""
    assert json.loads(rows[0]["sensor_timestamps"]) == {"vehicle_speed": now - 1}
    recorder.write(frame)
    with path.open() as handle:
        assert len(list(csv.DictReader(handle))) == 2
    second = CSVRecorder(tmp_path)
    assert second.directory != recorder.directory
    assert path.exists()


def test_parallel_sources_and_single_header(tmp_path):
    recorder = CSVRecorder(tmp_path)
    frames = [
        SensorFrame(timestamp=time.time(), stream_id="trip", source="obd", rpm=i) for i in range(40)
    ]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(recorder.write, frames))
    with (recorder.directory / "obd.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 40
    assert {float(row["rpm"]) for row in rows} == set(range(40))
    recorder.write(
        SensorFrame(timestamp=time.time(), stream_id="trip", source="phone", gps_latitude=0)
    )
    assert (recorder.directory / "phone.csv").exists()


