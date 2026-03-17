"""Clock test doubles."""

from datetime import UTC, datetime, timedelta


class FixedClock:
    """Clock frozen at a controllable timestamp."""

    def __init__(
        self,
        at: datetime | None = None,
        monotonic_step_ns: int = 1_000_000,
    ) -> None:
        self._now = at if at is not None else datetime(2026, 1, 1, tzinfo=UTC)
        self._monotonic_ns = 0
        self._monotonic_step_ns = monotonic_step_ns

    def now_utc(self) -> datetime:
        return self._now

    def monotonic_ns(self) -> int:
        """Return monotonic clock value, then advance by step."""
        value = self._monotonic_ns
        self._monotonic_ns += self._monotonic_step_ns
        return value

    def epoch_ms(self) -> int:
        """Return epoch milliseconds derived from the fixed timestamp."""
        return int(self._now.timestamp() * 1000)

    def set_monotonic_ns(self, value: int) -> None:
        """Set the monotonic counter to an exact value."""
        self._monotonic_ns = value

    def advance_monotonic_ns(self, delta: int) -> None:
        """Advance the monotonic counter by delta nanoseconds."""
        self._monotonic_ns += delta

    def advance(self, seconds: float) -> None:
        """Move the clock forward by the given seconds."""
        self._now += timedelta(seconds=seconds)

    def set(self, at: datetime) -> None:
        """Jump the clock to an exact timestamp."""
        self._now = at
