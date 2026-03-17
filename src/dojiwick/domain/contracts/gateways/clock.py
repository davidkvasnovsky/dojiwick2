"""Clock protocol for deterministic timing."""

from datetime import datetime
from typing import Protocol


class ClockPort(Protocol):
    """Returns current UTC time and monotonic/epoch timestamps."""

    def now_utc(self) -> datetime:
        """Return current UTC timestamp."""
        ...

    def monotonic_ns(self) -> int:
        """Return monotonic clock in nanoseconds."""
        ...

    def epoch_ms(self) -> int:
        """Return Unix epoch time in milliseconds."""
        ...
