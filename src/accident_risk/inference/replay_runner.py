"""Connect any DataSource to the same inference engine and serialize replay results."""

import json
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd

from accident_risk.datasets.base import DatasetAdapter
from accident_risk.inference.engine import InferenceEngine
from accident_risk.schemas import PredictionResult
from accident_risk.sources.base import DataSource
from accident_risk.sources.replay import ReplaySource


def run_source(
    source: DataSource, engine: InferenceEngine, output: str | Path | None = None
) -> tuple[list[PredictionResult], dict[str, Any]]:
    """Measure all-frame processing latency separately from replay wall-clock duration."""
    started = perf_counter()
    predictions, latencies, timestamps = [], [], []
    source.start()
    try:
        for frame in source:
            tick = perf_counter()
            prediction = engine.process(frame)
            latencies.append((perf_counter() - tick) * 1000)
            timestamps.append(frame.timestamp)
            if prediction is not None:
                predictions.append(prediction)
    finally:
        source.stop()
    statistics = {
        "processed_frames": len(timestamps),
        "predictions": len(predictions),
        "elapsed_seconds": perf_counter() - started,
        "simulated_duration_seconds": max(timestamps) - min(timestamps) if timestamps else 0,
        "mean_inference_latency_ms": float(np.mean(latencies)) if latencies else None,
        "p95_inference_latency_ms": float(np.percentile(latencies, 95)) if latencies else None,
    }
    if output:
        path = Path(output)
        if "raw" in path.parts:
            raise ValueError("Replay output must not be inside data/raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = []
        for prediction in predictions:
            row = prediction.model_dump()
            row["quality_flags"] = json.dumps(row["quality_flags"])
            row["sensor_age_seconds"] = json.dumps(row["sensor_age_seconds"])
            rows.append(row)
        table = pd.DataFrame(rows, columns=list(PredictionResult.model_fields))
        if path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        elif path.suffix == ".csv":
            table.to_csv(path, index=False)
        else:
            raise ValueError("Prediction output must be .csv or .parquet")
        path.with_suffix(".stats.json").write_text(json.dumps(statistics, indent=2))
    return predictions, statistics


def run_replay(
    adapter: DatasetAdapter,
    engine: InferenceEngine,
    speed: float = 1,
    maximum_speed: bool = False,
    output: str | Path | None = None,
) -> tuple[list[PredictionResult], dict[str, Any]]:
    """DatasetAdapter -> ReplaySource -> source-independent InferenceEngine."""
    return run_source(
        ReplaySource([r.frame for r in adapter.records()], speed, maximum_speed), engine, output
    )
