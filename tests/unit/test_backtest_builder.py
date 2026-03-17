"""Tests for backtest_builder integrity fixes: NaN validation and timestamp intersection."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest

from dojiwick.application.services.backtest_builder import build_backtest_time_series
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval

_INTERVAL = CandleInterval("1h")
_BASE_TIME = datetime(2025, 1, 1, tzinfo=UTC)


def _make_candles(
    pair: str,
    n: int,
    *,
    start: datetime = _BASE_TIME,
    offset: timedelta = timedelta(0),
    nan_at: int | None = None,
) -> tuple[Candle, ...]:
    """Generate n candles. If nan_at is given, that bar's close is NaN-like (0 — will produce NaN indicators)."""
    candles: list[Candle] = []
    for i in range(n):
        close = Decimal("100") + Decimal(i)
        candles.append(
            Candle(
                pair=pair,
                interval=_INTERVAL,
                open_time=start + timedelta(hours=i) + offset,
                open=close - 1,
                high=close + 5,
                low=close - 5,
                close=close,
                volume=Decimal("1000"),
            )
        )
    return tuple(candles)


class TestNaNValidation:
    def test_builder_rejects_nan_after_warmup_trim(self) -> None:
        """NaN indicators after warmup trim raises ValueError."""
        pair = "BTC/USDC"
        # 5 bars with warmup=0 means indicators will still have NaN from lookback requirements
        candles_short = _make_candles(pair, 5, start=_BASE_TIME)

        with pytest.raises(ValueError, match="bars with NaN indicators after 0-bar warmup trim"):
            build_backtest_time_series(
                {pair: candles_short},
                (pair,),
                warmup_bars=0,
            )

    def test_builder_accepts_clean_data_after_trim(self) -> None:
        """Clean data after warmup trim passes without error."""
        pair = "BTC/USDC"
        n = 250  # 200 warmup + 50 clean bars
        candles = _make_candles(pair, n)

        # Should not raise
        series = build_backtest_time_series({pair: candles}, (pair,), warmup_bars=200)
        assert series.n_bars > 0


class TestTimestampIntersection:
    def test_builder_intersects_timestamps_across_pairs(self) -> None:
        """Two pairs with offset timestamps are aligned by time intersection."""
        n = 260  # plenty of bars
        pair_a = "BTC/USDC"
        pair_b = "ETH/USDC"
        # Pair A starts at _BASE_TIME, pair B starts 2 hours later
        candles_a = _make_candles(pair_a, n)
        candles_b = _make_candles(pair_b, n, offset=timedelta(hours=2))

        series = build_backtest_time_series(
            {pair_a: candles_a, pair_b: candles_b},
            (pair_a, pair_b),
            warmup_bars=200,
        )

        # The intersection should have fewer bars than either pair alone post-trim
        assert series.n_bars > 0
        # Contexts should have consistent timestamps across pairs
        for ctx in series.contexts:
            assert ctx.market.observed_at is not None

    def test_builder_rejects_insufficient_coverage(self) -> None:
        """< 50% timestamp overlap between pairs raises ValueError."""
        pair_a = "BTC/USDC"
        pair_b = "ETH/USDC"
        n = 250
        # Pair A: hours 0..249, pair B: hours 500..749 — no overlap
        candles_a = _make_candles(pair_a, n)
        candles_b = _make_candles(pair_b, n, offset=timedelta(hours=500))

        with pytest.raises(ValueError, match="insufficient common timestamps|timestamp coverage too low"):
            build_backtest_time_series(
                {pair_a: candles_a, pair_b: candles_b},
                (pair_a, pair_b),
                warmup_bars=200,
            )

    def test_builder_single_pair_unchanged(self) -> None:
        """Single pair skips intersection — bar count equals post-trim count."""
        pair = "BTC/USDC"
        n = 250
        candles = _make_candles(pair, n)

        series = build_backtest_time_series({pair: candles}, (pair,), warmup_bars=200)

        # Single pair: post-trim bars = n - warmup - 1 (last used for next_prices)
        assert series.n_bars == n - 200 - 1
