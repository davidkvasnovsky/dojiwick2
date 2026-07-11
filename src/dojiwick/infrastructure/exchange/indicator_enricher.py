"""Live-path indicator enricher — fetches recent candles and computes indicators."""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from dojiwick.compute.kernels.indicators.compute import compute_indicators
from dojiwick.domain.contracts.gateways.clock import ClockPort
from dojiwick.domain.contracts.gateways.market_data_provider import MarketDataProviderPort
from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.numerics import candles_to_ohlc, decimals_to_array
from dojiwick.domain.timebase import interval_to_seconds
from dojiwick.domain.type_aliases import CandleInterval, FloatMatrix

_DEFAULT_INTERVAL = CandleInterval("1h")

log = logging.getLogger(__name__)


@dataclass(slots=True)
class IndicatorEnricher:
    """Computes indicator matrix from recent candles for the live path.

    Parity with the backtest builder: the same kernel over the same inputs,
    volume included (``volume_ema_ratio`` gates every strategy's entries),
    computed on closed bars only — the still-forming candle is dropped, the
    backtest never sees it either. A symbol without enough closed history for
    the longest indicator period returns an all-zero row, which the regime
    classifier rejects (``atr > 0`` fails) so the pair sits out the tick
    instead of trading on garbage.
    """

    market_data: MarketDataProviderPort
    clock: ClockPort
    candle_interval: CandleInterval = _DEFAULT_INTERVAL
    candle_lookback: int = 600
    rsi_period: int = 14
    ema_fast_period: int = 12
    ema_slow_period: int = 26
    ema_base_period: int = 50
    ema_trend_period: int = 200
    atr_period: int = 14
    adx_period: int = 14
    bb_period: int = 20
    bb_std: float = 2.0
    volume_ema_period: int = 20

    def _min_bars(self) -> int:
        longest = max(
            self.rsi_period,
            self.ema_fast_period,
            self.ema_slow_period,
            self.ema_base_period,
            self.ema_trend_period,
            self.atr_period,
            self.adx_period,
            self.bb_period,
            self.volume_ema_period,
        )
        return longest + 1

    async def _compute_for_symbol(self, symbol: str) -> np.ndarray:
        candles = await self.market_data.fetch_candles(symbol, self.candle_interval, self.candle_lookback)

        interval_sec = interval_to_seconds(self.candle_interval)
        now = self.clock.now_utc()
        while candles and candles[-1].open_time.timestamp() + interval_sec > now.timestamp():
            candles = candles[:-1]

        if len(candles) < self._min_bars():
            log.warning(
                "%s: %d closed candles < %d required — pair excluded this tick",
                symbol,
                len(candles),
                self._min_bars(),
            )
            return np.zeros(INDICATOR_COUNT, dtype=np.float64)

        close, high, low = candles_to_ohlc(candles)
        volume = decimals_to_array([c.volume for c in candles])
        matrix = compute_indicators(
            close,
            high,
            low,
            volume=volume,
            rsi_period=self.rsi_period,
            ema_fast_period=self.ema_fast_period,
            ema_slow_period=self.ema_slow_period,
            ema_base_period=self.ema_base_period,
            ema_trend_period=self.ema_trend_period,
            atr_period=self.atr_period,
            adx_period=self.adx_period,
            bb_period=self.bb_period,
            bb_std=self.bb_std,
            volume_ema_period=self.volume_ema_period,
        )
        last_row = matrix[-1]
        if not np.all(np.isfinite(last_row)):
            # Element-wise zeroing would poison gates (a 0.0 EMA-200 opens
            # every long); an all-zero row excludes the pair cleanly instead.
            log.warning("%s: non-finite indicators on closed data — pair excluded this tick", symbol)
            return np.zeros(INDICATOR_COUNT, dtype=np.float64)
        return last_row

    async def compute_for_pairs(
        self,
        symbols: tuple[str, ...],
    ) -> FloatMatrix:
        """Return ``(N_pairs, INDICATOR_COUNT)`` indicator matrix from closed candle history."""
        rows = await asyncio.gather(*(self._compute_for_symbol(s) for s in symbols))
        return np.stack(rows)
