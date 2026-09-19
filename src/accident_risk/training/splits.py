"""Leakage-safe grouping, including cross-linked identifiers and shared events."""

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from accident_risk.datasets.base import GROUP_PRIORITY


def choose_group_column(groups: pd.DataFrame) -> str:
    """Prefer participant, driver, vehicle, trip, event, requiring complete identifiers."""
    for name in GROUP_PRIORITY:
        if name in groups and groups[name].notna().all():
            return name
    raise ValueError("No complete participant/driver/vehicle/trip/event identifiers available")


def connected_groups(groups: pd.DataFrame) -> np.ndarray:
    """Union rows sharing any known identity, conservatively protecting all events.

    Identifiers must be globally unique within each column. This can collapse
    shared vehicles/drivers into one component; fail rather than leak if too few remain.
    """
    choose_group_column(groups)
    parent = list(range(len(groups)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen: dict[tuple[str, str], int] = {}
    for i, row in enumerate(groups.to_dict("records")):
        for column in (*GROUP_PRIORITY, "stream_id"):
            value = row.get(column)
            if value is None or pd.isna(value):
                continue
            key = (column, str(value))
            if key in seen:
                parent[root(i)] = root(seen[key])
            else:
                seen[key] = i
    return np.array([root(i) for i in range(len(groups))])


def group_split(
    groups: pd.DataFrame, validation_size: float = 0.2, test_size: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split independent connected groups without any neighboring-window random split."""
    if not 0 < validation_size < 1 or not 0 < test_size < 1 or validation_size + test_size >= 1:
        raise ValueError("Positive validation/test fractions must sum to less than one")
    identities = connected_groups(groups)
    if len(set(identities)) < 3:
        raise ValueError("Need at least three independent groups after linking shared identifiers")
    idx = np.arange(len(groups))
    trainval, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(
            idx, groups=identities
        )
    )
    tr, val = next(
        GroupShuffleSplit(
            n_splits=1, test_size=validation_size / (1 - test_size), random_state=seed
        ).split(trainval, groups=identities[trainval])
    )
    train, validation = trainval[tr], trainval[val]
    for left, right in ((train, validation), (train, test), (validation, test)):
        if set(identities[left]) & set(identities[right]):
            raise AssertionError("Group leakage detected")
    return train, validation, test
