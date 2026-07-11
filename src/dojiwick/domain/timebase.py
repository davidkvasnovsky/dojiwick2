"""Timebase contract — bar-close alignment and anti-leakage validation."""

from datetime import UTC, datetime

from dojiwick.domain.errors import DataQualityError

_INTERVAL_UNIT_SECONDS = {"m": 60, "h": 3_600, "d": 86_400, "w": 604_800}


def interval_to_seconds(interval: str) -> int:
    """Convert a candle interval string (``1m``, ``1h``, ``4h``, ``1d``, ``1w``) to seconds.

    Month intervals (``1M``) are rejected: months have no fixed duration.
    """
    if len(interval) < 2:
        raise ValueError(f"invalid candle interval: {interval!r}")
    unit = interval[-1]
    seconds = _INTERVAL_UNIT_SECONDS.get(unit)
    if seconds is None:
        raise ValueError(f"unsupported candle interval unit: {interval!r}")
    try:
        count = int(interval[:-1])
    except ValueError:
        raise ValueError(f"invalid candle interval: {interval!r}") from None
    if count <= 0:
        raise ValueError(f"invalid candle interval: {interval!r}")
    return count * seconds


def last_confirmed_bar_close(observed_at: datetime, interval_sec: int) -> datetime:
    """Return the most recent confirmed bar close via floor-division epoch math.

    A bar that opens at time T closes at T + interval_sec.  The last
    *confirmed* close at or before ``observed_at`` is the largest multiple
    of ``interval_sec`` that does not exceed the observation timestamp.
    """
    epoch = int(observed_at.timestamp())
    floored = (epoch // interval_sec) * interval_sec
    return datetime.fromtimestamp(floored, tz=UTC)


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
