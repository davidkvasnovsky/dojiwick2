"""Funding event binning tests for the backtest builder."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
import pytest

from dojiwick.application.services.backtest_builder import _bin_funding  # pyright: ignore[reportPrivateUsage]
from dojiwick.domain.models.value_objects.funding_rate import FundingRate

_T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _rate(hours: float, rate: str) -> FundingRate:
    return FundingRate(symbol="BTCUSDT", funding_time=_T0 + timedelta(hours=hours), rate=Decimal(rate))


def _hourly_times(n: int) -> list[datetime]:
    return [_T0 + timedelta(hours=i) for i in range(n)]


def test_events_land_in_their_candle_bins() -> None:
    times = _hourly_times(24)
    rates = (_rate(0, "0.0001"), _rate(8, "-0.0002"), _rate(16, "0.0003"))
    binned = _bin_funding("BTCUSDT", rates, times)

    assert binned[0] == pytest.approx(0.0001)  # pyright: ignore[reportUnknownMemberType]
    assert binned[8] == pytest.approx(-0.0002)  # pyright: ignore[reportUnknownMemberType]
    assert binned[16] == pytest.approx(0.0003)  # pyright: ignore[reportUnknownMemberType]
    assert np.count_nonzero(binned) == 3


def test_daily_candle_sums_multiple_events() -> None:
    times = [_T0, _T0 + timedelta(days=1)]
    rates = (_rate(0, "0.0001"), _rate(8, "0.0001"), _rate(16, "0.0001"), _rate(24, "0.0005"))
    binned = _bin_funding("BTCUSDT", rates, times)

    assert binned[0] == pytest.approx(0.0003)  # pyright: ignore[reportUnknownMemberType]
    assert binned[1] == pytest.approx(0.0005)  # pyright: ignore[reportUnknownMemberType]


def test_empty_rates_raise() -> None:
    with pytest.raises(ValueError, match="no funding events"):
        _bin_funding("BTCUSDT", (), _hourly_times(24))


def test_missing_head_coverage_raises() -> None:
    times = _hourly_times(48)
    rates = (_rate(24, "0.0001"), _rate(32, "0.0001"), _rate(40, "0.0001"))
    with pytest.raises(ValueError, match=r"starts .* after candles"):
        _bin_funding("BTCUSDT", rates, times)


def test_missing_tail_coverage_raises() -> None:
    times = _hourly_times(48)
    rates = (_rate(0, "0.0001"), _rate(8, "0.0001"), _rate(16, "0.0001"))
    with pytest.raises(ValueError, match=r"ends .* before candles"):
        _bin_funding("BTCUSDT", rates, times)


def test_events_before_first_candle_are_dropped() -> None:
    times = _hourly_times(24)
    rates = (_rate(-4, "0.5"), _rate(0, "0.0001"), _rate(8, "0.0001"), _rate(16, "0.0001"))
    binned = _bin_funding("BTCUSDT", rates, times)
    assert binned.sum() == pytest.approx(0.0003)  # pyright: ignore[reportUnknownMemberType]
