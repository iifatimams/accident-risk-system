"""Versioned classical classifier with explicit score semantics and artifact metadata."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight

from accident_risk.models.baselines import make_baseline


class VehicleModel:
    """Class scores are estimator outputs, not calibrated accident probabilities."""

    def __init__(
        self,
        family: str = "logistic_regression",
        seed: int = 42,
        target: str = "event_detection_score",
        score_labels: list[str] | None = None,
    ) -> None:
        if target not in {"event_detection_score", "vehicle_risk_score", "driver_state_risk"}:
            raise ValueError("Unsupported model score target")
        self.family, self.seed, self.target = family, seed, target
        self.score_labels = score_labels or []
        self.estimator = make_baseline(family, seed)
        self.encoder = LabelEncoder()
        self.feature_names: list[str] = []
        self.metadata: dict[str, Any] = {}

    def fit(
        self,
        X: pd.DataFrame,
        y: list[str] | np.ndarray,
        *,
        dataset: str,
        configuration: dict[str, Any],
        class_weight: str | dict | None = "balanced",
    ) -> "VehicleModel":
        """Fit training data only; store feature contract, label mapping and configuration."""
        if len(X) == 0 or X.isna().all().all():
            raise ValueError("No usable training features")
        labels = self.encoder.fit_transform(y)
        if len(self.encoder.classes_) < 2:
            raise ValueError(
                "Training partition needs at least two classes; collect more independent groups"
            )
        if not self.score_labels or set(self.score_labels) - set(self.encoder.classes_):
            raise ValueError("score_labels must explicitly select observed training labels")
        self.feature_names = list(X.columns)
        weights = (
            compute_sample_weight(class_weight, np.asarray(y)) if class_weight else np.ones(len(y))
        )
        self.estimator.fit(X, labels, classifier__sample_weight=weights)
        self.metadata = {
            "training_timestamp": datetime.now(UTC).isoformat(),
            "dataset": dataset,
            "feature_names": self.feature_names,
            "model_family": self.family,
            "model_version": f"0.1.0-{uuid4().hex[:12]}",
            "target": self.target,
            "label_mapping": {str(c): int(i) for i, c in enumerate(self.encoder.classes_)},
            "score_labels": self.score_labels,
            "configuration": configuration,
            "calibrated": False,
            "evaluation_summary": None,
        }
        return self

    def _matrix(self, X: pd.DataFrame) -> pd.DataFrame:
        """Reorder known features and reject a mismatched preprocessing contract."""
        if set(X.columns) != set(self.feature_names):
            raise ValueError("Feature schema does not match saved training feature list")
        return X.loc[:, self.feature_names]

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return original class labels."""
        return self.encoder.inverse_transform(self.estimator.predict(self._matrix(X)).astype(int))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return uncalibrated class-conditional estimator outputs in label_mapping order."""
        return self.estimator.predict_proba(self._matrix(X))

    def predict_score(self, X: pd.DataFrame) -> np.ndarray:
        """Sum only the explicitly configured classes of interest."""
        indices = [i for i, label in enumerate(self.encoder.classes_) if label in self.score_labels]
        return self.predict_proba(X)[:, indices].sum(axis=1)

    def save(self, path: str | Path) -> None:
        """Write the model and inspectable metadata/feature/configuration sidecars."""
        path = Path(path)
        if "raw" in path.parts:
            raise ValueError("Models must not be written inside data/raw")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        path.with_suffix(".metadata.json").write_text(
            json.dumps(self.metadata, indent=2, allow_nan=False)
        )
        path.with_suffix(".features.json").write_text(json.dumps(self.feature_names, indent=2))
        path.with_suffix(".config.json").write_text(
            json.dumps(self.metadata["configuration"], indent=2)
        )

    @classmethod
    def load(cls, path: str | Path) -> "VehicleModel":
        """Load a trusted local artifact only; joblib can execute code."""
        model = joblib.load(path)
        if not isinstance(model, cls):
            raise TypeError("Artifact does not contain the requested model type")
        return model
