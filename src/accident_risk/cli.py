"""Thin shared CLI handlers; all computation lives in importable package modules."""

import argparse
import json
from pathlib import Path

import pandas as pd

from accident_risk.datasets.base import load_prepared, save_prepared
from accident_risk.datasets.registry import get_adapter
from accident_risk.inference.engine import InferenceEngine
from accident_risk.inference.replay_runner import run_source
from accident_risk.logging_utils import configure_logging
from accident_risk.models.fusion import RiskFusion
from accident_risk.models.physiology_model import PhysiologyModel
from accident_risk.models.vehicle_model import VehicleModel
from accident_risk.settings import load_config
from accident_risk.sources.replay import ReplaySource
from accident_risk.training.evaluate import classification_metrics
from accident_risk.training.train_physiology import train_physiology
from accident_risk.training.train_vehicle import train


def inspect_cli() -> None:
    """Inspect local files without confirming or inventing mappings."""
    parser = argparse.ArgumentParser(description=inspect_cli.__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--path", required=True)
    parser.add_argument("--config")
    args = parser.parse_args()
    mapping = load_config(args.config).get("dataset", {}).get("mapping", {}) if args.config else {}
    print(
        json.dumps(get_adapter(args.dataset, args.path, mapping).inspect(), indent=2, default=str)
    )


def prepare_cli(dataset: str) -> None:
    """Validate explicit local mappings and write common-format Parquet plus provenance."""
    parser = argparse.ArgumentParser(description=prepare_cli.__doc__)
    parser.add_argument("--config", default=f"configs/{dataset}.yaml")
    parser.add_argument("--output", default=f"data/processed/{dataset}.parquet")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["dataset"]
    if settings["name"] != dataset:
        parser.error("Dataset name does not match preparation script")
    adapter = get_adapter(dataset, settings.get("path"), settings.get("mapping"))
    records = adapter.records()
    save_prepared(records, args.output)
    Path(args.output).with_suffix(".provenance.json").write_text(
        json.dumps(
            {
                "configuration": config,
                "inspection": adapter.inspect(),
                "prepared_records": len(records),
                "label_counts": pd.Series([r.label for r in records])
                .value_counts(dropna=False)
                .to_dict(),
            },
            indent=2,
            default=str,
        )
    )
    print(f"Prepared {len(records)} records: {args.output}")


def train_cli(physiology: bool = False) -> None:
    """Fit and compare group-safe baselines using local/prepared data."""
    parser = argparse.ArgumentParser(description=train_cli.__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared")
    args = parser.parse_args()
    config = load_config(args.config)
    configure_logging(config.get("logging", {}).get("level", "INFO"))
    summary = (
        train_physiology(config, args.prepared) if physiology else train(config, args.prepared)
    )
    print(json.dumps(summary, indent=2))


def evaluate_cli() -> None:
    """Re-evaluate a trusted saved model on held-out feature rows, never refit."""
    parser = argparse.ArgumentParser(description=evaluate_cli.__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--features", required=True, help="Saved test_features.parquet containing __label"
    )
    parser.add_argument("--output")
    args = parser.parse_args()
    model = VehicleModel.load(args.model)
    data = pd.read_parquet(args.features)
    labels = data.pop("__label").astype(str).to_numpy()
    metrics = classification_metrics(
        labels,
        model.predict(data),
        model.predict_proba(data),
        model.encoder.classes_.tolist(),
        model.score_labels,
    )
    text = json.dumps(metrics, indent=2, allow_nan=False)
    if args.output:
        output = Path(args.output)
        if "raw" in output.parts:
            raise ValueError("Metrics must not be written inside data/raw")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text)
    print(text)


def replay_cli() -> None:
    """Run the identical inference engine over a prepared or verified dataset."""
    parser = argparse.ArgumentParser(description=replay_cli.__doc__)
    parser.add_argument("--dataset", default="vzcrash")
    parser.add_argument("--config")
    parser.add_argument("--replay-config", default="configs/replay.yaml")
    parser.add_argument("--prepared")
    parser.add_argument("--model", required=True)
    parser.add_argument("--physiology-model")
    parser.add_argument("--speed", help="Positive multiplier or max")
    parser.add_argument("--output")
    args = parser.parse_args()
    replay_config = load_config(args.replay_config)
    options = replay_config.get("replay", {})
    speed = (
        args.speed
        if args.speed is not None
        else ("max" if options.get("maximum_speed") else str(options.get("speed", 1)))
    )
    model = VehicleModel.load(args.model)
    physiology = PhysiologyModel.load(args.physiology_model) if args.physiology_model else None
    engine = InferenceEngine(
        model, physiology, fusion=RiskFusion(**replay_config.get("fusion", {}))
    )
    if args.prepared:
        records = load_prepared(args.prepared)
    else:
        config = load_config(args.config or f"configs/{args.dataset}.yaml")
        dataset = config["dataset"]
        records = get_adapter(args.dataset, dataset.get("path"), dataset.get("mapping")).records()
    _, statistics = run_source(
        ReplaySource(
            [r.frame for r in records],
            speed=1 if speed == "max" else float(speed),
            maximum_speed=speed == "max",
        ),
        engine,
        args.output or options.get("output", "artifacts/replay_predictions.csv"),
    )
    print(json.dumps(statistics, indent=2))
