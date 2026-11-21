"""Driver-state summaries. HRV requires actual RR/IBI intervals."""

import numpy as np

from accident_risk.processing.vehicle_features import series, statistics
from accident_risk.processing.windows import Window


def physiology_features(window: Window) -> dict[str, float]:
    """Compute unique-reading summaries, preserving missingness and age in seconds."""
    features: dict[str, float] = {}
    for name in ("heart_rate",):
        times, values = series(window, name)
        features.update(statistics(name, times, values))
        features[f"{name}_age_seconds"] = (
            float(window.end - times[-1]) if len(times) else float("nan")
        )
    _, rr = series(window, "rr_interval_ms")
    features["rmssd_ms"] = (
        float(np.sqrt(np.mean(np.diff(rr) ** 2))) if len(rr) >= 2 else float("nan")
    )
    features["sdnn_ms"] = float(np.std(rr, ddof=1)) if len(rr) >= 2 else float("nan")
    return features
