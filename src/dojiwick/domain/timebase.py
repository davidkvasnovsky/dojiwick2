"""Timebase contract — bar-close alignment and anti-leakage validation."""

from datetime import datetime, timezone

from dojiwick.domain.errors import DataQualityError


def last_confirmed_bar_close(observed_at: datetime, interval_sec: int) -> datetime:
    """Return the most recent confirmed bar close via floor-division epoch math.

    A bar that opens at time T closes at T + interval_sec.  The last
    *confirmed* close at or before ``observed_at`` is the largest multiple
    of ``interval_sec`` that does not exceed the observation timestamp.
    """
    epoch = int(observed_at.timestamp())
    floored = (epoch // interval_sec) * interval_sec
    return datetime.fromtimestamp(floored, tz=timezone.utc)


def assert_timebase_valid(
    observed_at: datetime,
    bar_close: datetime,
    interval_sec: int,
    *,
    max_staleness_bars: int = 2,
) -> None:
    """Raise ``DataQualityError`` if bar data is too stale.

    Staleness is measured as the number of full intervals between
    ``bar_close`` and the last confirmed bar close for ``observed_at``.
    If the gap exceeds ``max_staleness_bars`` intervals the data is
    considered unreliable.
    """
    latest_close = last_confirmed_bar_close(observed_at, interval_sec)
    gap_sec = (latest_close - bar_close).total_seconds()
    max_gap_sec = max_staleness_bars * interval_sec
    if gap_sec > max_gap_sec:
        raise DataQualityError(
            f"bar data too stale: bar_close={bar_close.isoformat()}, "
            f"latest_close={latest_close.isoformat()}, "
            f"gap={gap_sec}s > max={max_gap_sec}s"
        )
