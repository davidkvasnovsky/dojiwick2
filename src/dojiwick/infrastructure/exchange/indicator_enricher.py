"""Live-path indicator enricher — fetches recent candles and computes indicators."""

import asyncio
from dataclasses import dataclass

import numpy as np

from dojiwick.compute.kernels.indicators.compute import compute_indicators
from dojiwick.domain.contracts.gateways.market_data_provider import MarketDataProviderPort
from dojiwick.domain.indicator_schema import INDICATOR_COUNT
from dojiwick.domain.numerics import candles_to_ohlc
from dojiwick.domain.type_aliases import CandleInterval, FloatMatrix


@dataclass(slots=True)
class IndicatorEnricher:
    """Computes indicator matrix from recent candles for the live path."""

    market_data: MarketDataProviderPort
    candle_interval: CandleInterval = CandleInterval("1h")
    candle_lookback: int = 60
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

    async def _compute_for_symbol(self, symbol: str) -> np.ndarray:
        candles = await self.market_data.fetch_candles(symbol, self.candle_interval, self.candle_lookback)
        if len(candles) < 2:
            return np.zeros(INDICATOR_COUNT, dtype=np.float64)
        close, high, low = candles_to_ohlc(candles)
        matrix = compute_indicators(
            close,
            high,
            low,
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
        # Full matrix reused by backtest path; live path only needs last row.
        last_row = matrix[-1]
        return np.where(np.isfinite(last_row), last_row, 0.0)

    async def compute_for_pairs(
        self,
        symbols: tuple[str, ...],
    ) -> FloatMatrix:
        """Return ``(N_pairs, INDICATOR_COUNT)`` indicator matrix from recent candle history."""
        rows = await asyncio.gather(*(self._compute_for_symbol(s) for s in symbols))
        return np.stack(rows)
