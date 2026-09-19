"""Timestamp-paced replay; speed changes wall time only."""

import math
import threading
import time
from collections.abc import Callable, Iterable, Iterator

from accident_risk.schemas import SensorFrame
from accident_risk.sources.base import DataSource


class ReplaySource(DataSource):
    """Stable-sort input and replay against an absolute monotonic clock.

    Equal timestamps retain input order (complementary sensors are allowed).
    Inject clock/sleep for timing tests. Iteration automatically starts replay.
    """

    def __init__(
        self,
        frames: Iterable[SensorFrame],
        speed: float = 1,
        maximum_speed: bool = False,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        if not math.isfinite(speed) or speed <= 0:
            raise ValueError("Replay speed must be positive and finite")
        self.frames = sorted(
            (SensorFrame.model_validate(f) for f in frames), key=lambda f: f.timestamp
        )
        self.speed, self.maximum_speed, self.clock = speed, maximum_speed, clock
        self._stopped = threading.Event()
        self.sleep = sleep or self._stopped.wait

    def start(self) -> None:
        """Restart from the beginning on the next iteration."""
        self._stopped.clear()

    def stop(self) -> None:
        """Interrupt even a long realtime wait."""
        self._stopped.set()

    def __iter__(self) -> Iterator[SensorFrame]:
        """Yield stable-ordered measurements without accumulating inference-time drift."""
        self.start()
        started = self.clock()
        origin = self.frames[0].timestamp if self.frames else 0
        for frame in self.frames:
            if self._stopped.is_set():
                break
            if not self.maximum_speed:
                delay = (frame.timestamp - origin) / self.speed - (self.clock() - started)
                if delay > 0:
                    self.sleep(delay)
            if self._stopped.is_set():
                break
            yield frame
