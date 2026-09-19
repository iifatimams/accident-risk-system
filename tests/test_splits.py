"""Prove no participant, trip, event, or cross-linked group overlaps."""

import pandas as pd
import pytest

from accident_risk.training.splits import choose_group_column, connected_groups, group_split


def test_group_overlap():
    groups = pd.DataFrame(
        [
            {"participant": f"p{i}", "trip": f"trip{i}-{j // 2}", "event": f"e{i}-{j}"}
            for i in range(12)
            for j in range(4)
        ]
    )
    partitions = group_split(groups)
    assert choose_group_column(groups) == "participant"
    for i in range(3):
        for j in range(i + 1, 3):
            for column in groups:
                assert not set(groups.iloc[partitions[i]][column]) & set(
                    groups.iloc[partitions[j]][column]
                )


def test_cross_linked_event_and_insufficient_groups():
    groups = pd.DataFrame(
        {"participant": ["a", "b", "c", "d"], "event": ["shared", "shared", "c", "d"]}
    )
    connected = connected_groups(groups)
    assert connected[0] == connected[1]
    with pytest.raises(ValueError, match="three"):
        group_split(pd.DataFrame({"event": ["a", "a", "b"]}))
