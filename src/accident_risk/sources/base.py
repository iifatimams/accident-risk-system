"""All acquisition backends yield the same wire schema."""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from accident_risk.schemas import SensorFrame


class DataSource(ABC):
    """Start/stop lifecycle with deterministic SensorFrame iteration."""

    @abstractmethod
    def start(self) -> None:
        """Enable acquisition."""

    @abstractmethod
    def stop(self) -> None:
        """Stop acquisition."""

    @abstractmethod
    def __iter__(self) -> Iterator[SensorFrame]:
        """Yield observations, never source-specific records."""
