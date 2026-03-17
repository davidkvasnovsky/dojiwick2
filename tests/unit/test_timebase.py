"""Unit tests for timebase contract — bar-close alignment and staleness validation."""

from datetime import datetime, timezone

import pytest

from dojiwick.domain.errors import DataQualityError
from dojiwick.domain.timebase import assert_timebase_valid, last_confirmed_bar_close


def _utc(epoch: int) -> datetime:
    return datetime.fromtimestamp(epoch, tz=timezone.utc)


class TestLastConfirmedBarClose:
    def test_aligned(self) -> None:
        """On-interval timestamp returns that exact time."""
        interval = 90
        aligned = _utc(900)  # 900 is a multiple of 90
        assert last_confirmed_bar_close(aligned, interval) == aligned

    def test_mid_interval(self) -> None:
        """Mid-interval timestamp returns the previous bar close."""
        interval = 90
        mid = _utc(900 + 45)  # halfway through the next bar
        expected = _utc(900 + 45 - (45 % 90))  # floor to 900
        result = last_confirmed_bar_close(mid, interval)
        assert result == _utc(900)
        assert result == expected


class TestAssertTimebaseValid:
    def test_fresh_data_passes(self) -> None:
        """Data within staleness window raises nothing."""
        interval = 90
        observed_at = _utc(900)
        bar_close = _utc(900)
        assert_timebase_valid(observed_at, bar_close, interval)

    def test_stale_data_raises(self) -> None:
        """Data exceeding max_staleness_bars raises DataQualityError."""
        interval = 90
        observed_at = _utc(900)
        bar_close = _utc(900 - 3 * 90)  # 3 bars old, max_staleness_bars=2
        with pytest.raises(DataQualityError, match="bar data too stale"):
            assert_timebase_valid(observed_at, bar_close, interval)

    def test_boundary_passes(self) -> None:
        """Data exactly at max_staleness_bars is accepted (not stale)."""
        interval = 90
        observed_at = _utc(900)
        bar_close = _utc(900 - 2 * 90)  # exactly 2 bars old, max_staleness_bars=2
        assert_timebase_valid(observed_at, bar_close, interval, max_staleness_bars=2)
