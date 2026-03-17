"""UTC wall clock adapter."""

import time
from datetime import UTC, datetime


class SystemClock:
    """Provides current UTC timestamp and monotonic/epoch clocks."""

    def now_utc(self) -> datetime:
        """Return timezone-aware UTC now."""

        return datetime.now(UTC)

    def monotonic_ns(self) -> int:
        """Return monotonic clock in nanoseconds."""
        return time.monotonic_ns()

    def epoch_ms(self) -> int:
        """Return Unix epoch time in milliseconds."""
        return int(time.time() * 1000)
