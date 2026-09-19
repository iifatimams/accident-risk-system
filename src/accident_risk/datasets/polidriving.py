"""POLIDriving adapter with explicit file, unit, label and identifier confirmation."""

from accident_risk.datasets.base import MappedTabularAdapter


class POLIDrivingAdapter(MappedTabularAdapter):
    """Preserve genuine context separately, nullable sensors and provenance groups.

    Do not select the synthetic `furious` driver for real-data evaluation, or pool
    expert and semi-supervised labels without an explicit research protocol.
    Context is retained but not modeled by the baseline feature extractor.
    """

    name = "polidriving"
