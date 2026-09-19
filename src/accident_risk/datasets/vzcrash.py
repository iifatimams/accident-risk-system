"""VZCrash event adapter; no unverified sensor or label-column defaults."""

import pandas as pd

from accident_risk.datasets.base import MappedTabularAdapter


class VZCrashAdapter(MappedTabularAdapter):
    """Map verified crash/near_miss/normal event annotations, preserving event identity.

    A clip label is evidence for event classification, not a future-crash target.
    Non-tabular releases need an inspected format-specific reader before mapping.
    """

    name = "vzcrash"

    def validate_columns(self, table: pd.DataFrame) -> None:
        """Require explicit event identity and the supported conceptual event labels."""
        super().validate_columns(table)
        if "event" not in self.mapping["groups"]:
            raise ValueError("VZCrash requires an explicit event ID for safe splitting")
        if set(map(str, self.mapping["label_map"].values())) - {"crash", "near_miss", "normal"}:
            raise ValueError(
                "VZCrash conceptual labels must be crash, near_miss or normal; verify source semantics"
            )
