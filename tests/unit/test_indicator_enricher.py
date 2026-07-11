"""Live indicator enricher parity tests: volume, confirmed bars, exclusion semantics."""

from datetime import UTC, datetime, timedelta
from decimal import Decimal

import numpy as np
from fixtures.fakes.clock import FixedClock
from fixtures.fakes.market_data_provider import InMemoryMarketDataProvider

from dojiwick.domain.indicator_schema import INDICATOR_INDEX
from dojiwick.domain.models.value_objects.candle import Candle
from dojiwick.domain.type_aliases import CandleInterval
from dojiwick.infrastructure.exchange.indicator_enricher import IndicatorEnricher

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_INTERVAL = CandleInterval("1h")


def _candles(n: int, base: float = 100.0) -> list[Candle]:
    out: list[Candle] = []
    for i in range(n):
        price = Decimal(str(base + 0.01 * i))
        out.append(
            Candle(
                pair="BTCUSDT",
                interval=_INTERVAL,
                open_time=_T0 + timedelta(hours=i),
                open=price,
                high=price + Decimal("1"),
                low=price - Decimal("1"),
                close=price,
                volume=Decimal("1000"),
                quote_volume=Decimal("100000"),
            )
        )
    return out


def _enricher(candles: list[Candle], now: datetime) -> IndicatorEnricher:
    provider = InMemoryMarketDataProvider()
    provider.set_candles("BTCUSDT", _INTERVAL, candles)
    return IndicatorEnricher(
        market_data=provider,
        clock=FixedClock(at=now),
        candle_interval=_INTERVAL,
        candle_lookback=600,
    )


async def test_volume_ema_ratio_is_computed() -> None:
    """The live path must feed volume — a zero ratio blocks every strategy entry."""
    candles = _candles(260)
    now = candles[-1].open_time + timedelta(hours=2)
    enricher = _enricher(candles, now)

    matrix = await enricher.compute_for_pairs(("BTCUSDT",))

    ratio = matrix[0, INDICATOR_INDEX["volume_ema_ratio"]]
    assert np.isfinite(ratio) and ratio > 0.0


async def test_ema_trend_is_finite_with_enough_history() -> None:
    candles = _candles(260)
    now = candles[-1].open_time + timedelta(hours=2)
    enricher = _enricher(candles, now)

    matrix = await enricher.compute_for_pairs(("BTCUSDT",))

    assert matrix[0, INDICATOR_INDEX["ema_trend"]] > 0.0


async def test_forming_candle_is_dropped() -> None:
    """Indicators must come from closed bars only — the backtest never sees the forming bar."""
    candles = _candles(261)
    # Clock sits inside the last candle's interval: it is still forming
    now = candles[-1].open_time + timedelta(minutes=30)
    enricher = _enricher(candles, now)

    matrix_live = await enricher.compute_for_pairs(("BTCUSDT",))

    closed_only = _enricher(candles[:-1], now)
    matrix_closed = await closed_only.compute_for_pairs(("BTCUSDT",))

    np.testing.assert_array_equal(matrix_live, matrix_closed)


async def test_insufficient_history_excludes_pair() -> None:
    """Too few closed bars for EMA-200 yields an all-zero row (regime-invalid), not garbage."""
    candles = _candles(60)
    now = candles[-1].open_time + timedelta(hours=2)
    enricher = _enricher(candles, now)

    matrix = await enricher.compute_for_pairs(("BTCUSDT",))

    assert np.all(matrix[0] == 0.0)
